import os
import json
import logging
import httpx
import base64
import re
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from core.config import Settings
from core.ai import OpenAICompat
from core.database import MemoryStore, _today_key
from core.tts import AliyunTTS
from utils.common import is_allowed, parse_model_arg, limit_chars, redact_secrets
from utils.media import download_first_image

logger = logging.getLogger("tg-grok-bot")

# Helper Functions (Internal to handlers)
def load_personas_dir(dir_path: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    try:
        os.makedirs(dir_path, exist_ok=True)
        for name in sorted(os.listdir(dir_path)):
            if not name.lower().endswith(".txt"):
                continue
            key = os.path.splitext(name)[0]
            path = os.path.join(dir_path, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
            except Exception:
                continue
            if not raw:
                continue
            lines = raw.splitlines()
            display = lines[0].strip() if lines else key
            system = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw
            out[key] = {"name": display or key, "system": system}
    except Exception:
        pass
    if "default" not in out:
        out["default"] = {"name": "默认助手", "system": "你是一个乐于助人的助手。"}
    return out

def tts_enabled_for_chat(store: MemoryStore, chat_id: int, settings: Settings) -> bool:
    if not settings.tts_enabled:
        return False
    st = store.get_state(chat_id)
    if not st:
        return bool(settings.tts_default_on)
    return bool(int(st.get("tts_enabled", 0)))

def ocr_enabled_for_chat(store: MemoryStore, chat_id: int, settings: Settings) -> bool:
    if not settings.ocr_enabled:
        return False
    st = store.get_state(chat_id)
    if not st:
        return bool(settings.ocr_default_on)
    return bool(int(st.get("ocr_enabled", 0)))

def gf_enabled_for_chat(store: MemoryStore, chat_id: int, settings: Settings) -> bool:
    st = store.get_state(chat_id)
    if not st:
        return False
    return bool(int(st.get("gf_enabled", 0)))

def _sanitize_scene_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_scene_prompt(scene: str) -> str:
    scene = _sanitize_scene_text(scene)
    # Scene description is ABSOLUTE priority
    # Character details are flexible reference, not requirements
    prompt = (
        f"{scene}. "
        "The girlfriend is Yor Forger from Spy x Family: adult woman, long straight black hair with sidelocks, striking red eyes with shining gaze, gold earrings, mature elegant aura. "
        "Quality: Masterpiece, best quality, ultra-detailed, 8K resolution, high dynamic range, stunningly beautiful, intricate details, sharp focus, cinematic color grading, realistic anime illustration style. "
        "Atmosphere: Intimate romantic, sensual, NSFW allowed, mature content, sexy mood. "
        "Avoid: worst quality, low quality, extra digits, bad hands, missing fingers, child, loli, watermark, censored, deformed hands, shiny skin, text, multiple people, bystanders."
    )
    return prompt


# Command Handlers
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    await update.message.reply_text(
        "你好！我是基于 Grok 的 AI 助手。\n\n"
        "常用指令：\n"
        "/ask - 开始对话\n"
        "/img - 绘画\n"
        "/scene - 生成场景图\n"
        "/persona - 切换人格\n"
        "/models - 可用模型清单"
    )

async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    await update.message.reply_text(
        "可用模型：\n"
        "- grok-4.1-fast（默认，文本对话）\n"
        "- grok-imagine-1.0（图像生成）\n\n"
        "提示：使用 /model grok-4.1-fast 设置默认模型\n"
        "语音：使用 /voice on 开启语音回复（每日限额）"
    )

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    args = context.args or []
    if not args:
        st = store.get_state(update.effective_chat.id)
        cur = (st.get("model") if st else "") or s.default_model
        await update.message.reply_text(f"当前模型：{cur}\n用法：/model <模型名>")
        return
    new_model = args[0].strip()
    store.set_model(update.effective_chat.id, new_model)
    await update.message.reply_text(f"好的，已设置默认模型为：{new_model}")

async def cmd_persona(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    personas = load_personas_dir(s.personas_dir)
    args = context.args or []
    if not args:
        st = store.get_state(update.effective_chat.id)
        cur_key = (st.get("persona_key") if st else "default") or "default"
        cur_name = personas.get(cur_key, {}).get("name", cur_key)
        lines = ["当前人格：" + cur_name, "\n可用列表："]
        for k, p in personas.items():
            lines.append(f"- {k}: {p['name']}")
        lines.append("\n用法：/persona <Key>")
        await update.message.reply_text("\n".join(lines))
        return
    key = args[0].strip().lower()
    if key not in personas:
        await update.message.reply_text(f"无效的 Key。可用：{', '.join(personas.keys())}")
        return
    store.set_persona(update.effective_chat.id, key)
    name = personas[key]["name"]
    await update.message.reply_text(f"好的，已切换人格为：{name}（{key}）")

async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    if not s.tts_enabled:
        await update.message.reply_text("语音功能未启用（服务端 TTS_ENABLED=0）。")
        return
    tts: AliyunTTS = context.application.bot_data.get("tts")
    if not tts or not tts.available():
        await update.message.reply_text("语音功能不可用：未配置 DASHSCOPE_API_KEY 或缺少依赖。")
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    args = context.args or []
    if not args:
        st = store.get_state(update.effective_chat.id)
        enabled = bool(int(st.get("tts_enabled", 0))) if st else bool(s.tts_default_on)
        cnt = int(st.get("tts_count", 0)) if st and st.get("tts_day") == _today_key() else 0
        
        provider = (s.tts_provider or "dashscope").strip().lower()
        if provider in ("qwen", "qwen-tts", "qwen_tts"):
            cur_voice = f"{s.qwen_tts_voice}（{s.qwen_tts_model}）"
        else:
            cur_voice = f"{s.tts_default_voice}（{s.tts_default_model}）"

        await update.message.reply_text(
            f"语音回复：{'开启' if enabled else '关闭'}（默认：{'开启' if s.tts_default_on else '关闭'}）\n"
            f"今日用量：{cnt}/{s.tts_daily_limit}\n"
            f"当前音色：{cur_voice}\n"
            f"音量：{s.tts_volume}"
        )
        return
    opt = args[0].strip().lower()
    if opt in ("on", "1", "true", "enable"):
        store.set_tts_enabled(update.effective_chat.id, True)
        await update.message.reply_text("好的，已开启语音回复（会受每日限额影响）。")
        return
    if opt in ("off", "0", "false", "disable"):
        store.set_tts_enabled(update.effective_chat.id, False)
        await update.message.reply_text("好的，已关闭语音回复。")
        return
    await update.message.reply_text("用法：/voice on 或 /voice off （不带参数可查看状态）")

async def cmd_ocr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    if not s.ocr_enabled:
        await update.message.reply_text("OCR 功能未在服务端启用。")
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    args = context.args or []
    if not args:
        st = store.get_state(update.effective_chat.id)
        enabled = bool(int(st.get("ocr_enabled", 0))) if st else bool(s.ocr_default_on)
        await update.message.reply_text(f"OCR 自动识别当前状态：{'开启' if enabled else '关闭'} (用法：/ocr on/off)")
        return
    opt = args[0].strip().lower()
    if opt in ("on", "1", "true", "enable"):
        store.set_ocr_enabled(update.effective_chat.id, True)
        await update.message.reply_text("好的，已开启图片文字自动识别。")
    elif opt in ("off", "0", "false", "disable"):
        store.set_ocr_enabled(update.effective_chat.id, False)
        await update.message.reply_text("好的，已关闭图片文字自动识别。")
    else:
        await update.message.reply_text("用法：/ocr on 或 /ocr off")

async def cmd_gf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    args = context.args or []
    if not args:
        st = store.get_state(update.effective_chat.id)
        enabled = bool(int(st.get("gf_enabled", 0))) if st else False
        await update.message.reply_text(f"女友模式：{'开启' if enabled else '关闭'}")
        return
    opt = args[0].strip().lower()
    if opt in ("on", "1", "true", "enable"):
        store.set_gf_enabled(update.effective_chat.id, True)
        await update.message.reply_text("好的，已开启女友对话模式。")
    elif opt in ("off", "0", "false", "disable"):
        store.set_gf_enabled(update.effective_chat.id, False)
        await update.message.reply_text("好的，已切回普通模式。")

async def cmd_daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This command uses OpenAI to generate a short greeting
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    chat_id = update.effective_chat.id
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(chat_id, s)
    day = _today_key()
    st = store.get_state(chat_id)
    if st.get("last_daily") == day:
        # Already did it
        return
    prompt = "你是我的女朋友约尔，现在是早晨，请给我一个简短甜美的问候（限制40字）。" if bool(int(st.get("gf_enabled", 0))) else "请写一个简短的每日问候语。"
    try:
        out = (await client.chat(model=s.default_model, user_text=prompt)).strip()
        out = limit_chars(out, 40)
        store.set_last_daily(chat_id, day)
        await update.message.reply_text(out or "(无内容)")
    except Exception as e:
        logger.error(f"Daily command failed: {e}")

async def cmd_mem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    summary = store.get_summary(update.effective_chat.id)
    if not summary:
        await update.message.reply_text("我目前还没有关于你的长期记忆。")
        return
    await update.message.reply_text("我对你的长期记忆（摘要）：\n\n" + redact_secrets(summary))

async def cmd_mem_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.set_summary(update.effective_chat.id, "")
    await update.message.reply_text("好的，我已经遗忘了关于你的所有长期记忆。")

async def cmd_mem_gc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    n = store.gc(update.effective_chat.id, s.memory_max_log_messages)
    await update.message.reply_text(f"已清理旧对话记录：{n} 条。")

async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from .messages import _maybe_reply_tts # Circular dependency avoid
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    model_arg, text = parse_model_arg(update.message.text)
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    st = store.get_state(update.effective_chat.id)
    model = model_arg or (st.get("model") if st else "") or s.default_model
    if not text:
        await update.message.reply_text("用法：/ask [模型ID] <内容>")
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        out = await client.chat(model=model, user_text=text)
        if not out:
            await update.message.reply_text("(无内容)")
            return
        await _maybe_reply_tts(update, context, out)
        await update.message.reply_text(out)
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"上游接口错误：{e.response.status_code} {e.response.text}")
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")

async def cmd_img(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return
    prompt = " ".join(context.args or []).strip()
    if not prompt:
        await update.message.reply_text("用法：/img <提示词>")
        return
    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    try:
        data = await client.images_generate(model=s.image_model, prompt=prompt, n=1, response_format="url")
        items = (data or {}).get("data") or []
        urls = [it.get("url") for it in items if isinstance(it, dict) and it.get("url")]
        b64s = [it.get("b64_json") for it in items if isinstance(it, dict) and it.get("b64_json")]
        if b64s:
            try:
                img = base64.b64decode(b64s[0])
                await update.message.reply_photo(photo=img)
                return
            except Exception: pass
        if not urls:
            snippet = json.dumps(data, ensure_ascii=False)[:800]
            await update.message.reply_text("(未返回图片链接) 上游返回：" + snippet)
            return
        img = await download_first_image(urls[:4], timeout=s.http_timeout)
        if not img:
            await update.message.reply_text("图片生成成功但下载失败。\n\n" + "\n".join(urls[:4]))
            return
        await update.message.reply_photo(photo=img)
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")

async def cmd_scene(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    # Prefer direct arguments, fallback to replied message
    scene_text = " ".join(context.args or []).strip()
    if not scene_text and update.message.reply_to_message:
        rt = update.message.reply_to_message
        scene_text = (rt.text or rt.caption or "").strip()

    if not scene_text:
        await update.message.reply_text("用法：/scene <场景描述>（或回复一条消息再发送 /scene）。")
        return

    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    try:
        logger.info(f"/scene input: {scene_text[:100]}")
        
        # Step 1: summarize/extract key scene info to improve accuracy
        if s.scene_summarize:
            sum_prompt = (
                "你是一个场景提炼器。请将用户提供的生动描述，提炼成用于 AI 绘图的精炼场景描述。\n"
                "要求：\n"
                "- 提取核心视觉元素：人物、动作、环境、氛围。\n"
                "- 语言风格：精炼的中文短句。确保这是一个完整的描述，不要中途截断。\n"
                "- 禁止：不要输出 Markdown、引号或编号。不要包含对话或非视觉描述。\n\n"
                f"原文：{scene_text}"
            )
            try:
                summarized = await client.chat(model=s.scene_summarize_model, user_text=sum_prompt, max_tokens=1000)
                if summarized:
                    scene_text = summarized.strip()
                    logger.info(f"/scene summarized (full): {scene_text}")
            except Exception as e:
                logger.warning(f"/scene summarize failed: {e}")

        # Step 2: build final image prompt
        # Remove newlines to avoid confusing the image generator
        clean_scene = scene_text.replace("\n", " ").strip()
        prompt = build_scene_prompt(clean_scene)
        
        data = await client.images_generate(model=s.image_model, prompt=prompt, n=1, response_format="url")
        items = (data or {}).get("data") or []
        urls = [it.get("url") for it in items if isinstance(it, dict) and it.get("url")]
        b64s = [it.get("b64_json") for it in items if isinstance(it, dict) and it.get("b64_json")]

        if b64s:
            try:
                img = base64.b64decode(b64s[0])
                await update.message.reply_photo(photo=img)
                return
            except Exception: pass

        if not urls:
            snippet = json.dumps(data, ensure_ascii=False)[:400]
            await update.message.reply_text("(未返回图片链接) 上游返回：" + snippet)
            return

        img = await download_first_image(urls[:4], timeout=s.http_timeout)
        if not img:
            await update.message.reply_text("图片下载失败。\n\n" + "\n".join(urls[:4]))
            return

        await update.message.reply_photo(photo=img)

    except Exception as e:
        logger.error(f"/scene logical error: {e}")
        await update.message.reply_text(f"错误：{e}")
