import asyncio
import io
import logging
import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from core.config import Settings
from core.ai import OpenAICompat
from core.database import MemoryStore
from core.tts import QwenTTS
from utils.common import is_allowed, limit_chars, redact_secrets
from utils.media import run_ocr
from utils.voice import trim_wav_bytes, wav_to_ogg_opus
from handlers.commands import tts_enabled_for_chat, ocr_enabled_for_chat, gf_enabled_for_chat, cmd_scene

logger = logging.getLogger("tg-grok-bot")

async def update_memory_if_needed(store: MemoryStore, client: OpenAICompat, chat_id: int, settings: Settings) -> None:
    st = store.get_state(chat_id)
    if not st or int(st.get("msg_since_summary", 0)) < int(settings.memory_summary_update_every):
        return
    old = store.get_summary(chat_id)
    recent = store.get_recent_messages(chat_id, limit=20)
    convo = "\n".join([f"{'用户' if m['role']=='user' else '助手'}：{redact_secrets(m['content'])}" for m in recent if m.get("content")])
    prompt = f"你是一个记忆整理器。请把‘旧摘要’与‘最近对话’融合成新的长期摘要。\n旧摘要：\n{redact_secrets(old)}\n最近对话：\n{convo}"
    try:
        new = await client.chat(model="grok-4.1-fast", user_text=prompt, system_prompt="You are a memory summarizer.")
        if new:
            store.set_summary(chat_id, limit_chars(new.strip(), settings.memory_summary_max_chars))
    except Exception: pass

async def _maybe_reply_tts(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    s: Settings = context.application.bot_data["settings"]
    store: MemoryStore = context.application.bot_data["store"]
    chat_id = update.effective_chat.id
    if not text or not tts_enabled_for_chat(store, chat_id, s) or not store.tts_can_use(chat_id, s):
        return False
    tts_text = limit_chars(text.strip(), min(int(s.tts_max_chars), 120))
    tts = context.application.bot_data.get("tts")
    if not tts or not tts.available():
        return False
    try:
        await update.effective_chat.send_action(ChatAction.RECORD_VOICE)
        if isinstance(tts, QwenTTS):
            wav = await asyncio.to_thread(tts.synthesize_wav, tts_text)
            wav = await asyncio.to_thread(trim_wav_bytes, wav, 20.0)
            opus = await asyncio.to_thread(wav_to_ogg_opus, wav)
            await update.message.reply_voice(voice=io.BytesIO(opus), filename="tts.ogg")
        else:
            audio = await asyncio.to_thread(tts.synthesize_mp3, tts_text)
            await update.message.reply_voice(voice=io.BytesIO(audio), filename="tts.mp3")
        store.tts_mark_used(chat_id)
        return True
    except Exception: return False

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]
    if update.effective_chat is None or update.message is None or not update.message.photo: return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids): return
    store: MemoryStore = context.application.bot_data["store"]
    chat_id = update.effective_chat.id
    if not ocr_enabled_for_chat(store, chat_id, s): return
    try:
        photo = update.message.photo[-1]
        f = await context.bot.get_file(photo.file_id)
        img_bytes = bytes(await f.download_as_bytearray())
        await update.effective_chat.send_action(ChatAction.TYPING)
        text = run_ocr(img_bytes)
        if not text:
            await update.message.reply_text("我没识别到清晰文字。")
            return
        question = (update.message.caption or "").strip()
        prompt = f"你是一个OCR结果解读助手。\nOCR文字：\n{text}\n用户问题：{question or '（无）'}"
        out = await client.chat(model=s.ocr_model, user_text=prompt)
        if out: await update.message.reply_text(out)
    except Exception: pass

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]
    if update.effective_chat is None or update.message is None or not update.message.text: return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids): return
    raw = update.message.text.strip()
    if not raw: return
    
    # Handle mis-typed commands
    stripped = update.message.text.lstrip()
    if stripped.startswith(("/", "／")) and "scene" in stripped.split()[0].lower():
        old = getattr(context, "args", None)
        try:
            context.args = stripped.split()[1:]
            await cmd_scene(update, context)
        finally: context.args = old
        return

    force_voice = any(raw.startswith(p) for p in ("语音：", "语音:"))
    force_text = any(raw.startswith(p) for p in ("文字：", "文字:", "text:", "TEXT:"))
    text = re.sub(r"^(语音|文字|text)[:：]\s*", "", raw, flags=re.I).strip()
    if not text: return

    store: MemoryStore = context.application.bot_data["store"]
    chat_id = update.effective_chat.id
    store.append_message(chat_id, "user", text)
    st = store.get_state(chat_id)
    model = (st.get("model") if st else "") or s.default_model

    personas = context.application.bot_data["personas"]
    sys = personas.get(st.get("persona_key", "default"), personas["default"])["system"]
    summary = store.get_summary(chat_id)
    if summary: sys += "\n\n关于用户的长期记忆（摘要）：\n" + summary
    if gf_enabled_for_chat(store, chat_id, s):
        sys += "\n\n额外风格要求：你以‘温柔体贴的女友’口吻与用户交流。"

    msgs = [{"role": "system", "content": sys}]
    recent = store.get_recent_messages(chat_id, limit=12)
    msgs.extend([{"role": m["role"], "content": m["content"]} for m in recent if m.get("content")])

    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        out = await client.chat_messages(model=model, messages=msgs)
        if not out: return
        store.append_message(chat_id, "assistant", out)
        await update_memory_if_needed(store, client, chat_id, s)
        store.gc(chat_id, s.memory_max_log_messages)

        if force_voice:
            short_prompt = f"请把下面这段回复压缩成不超过30个汉字的中文短句：\n{out}"
            out = (await client.chat(model="grok-4.1-fast", user_text=short_prompt)).strip() or out
            out = limit_chars(out, 30)
            if await _maybe_reply_tts(update, context, out):
                await update.message.reply_text(out)
                return
        
        if not force_text and s.tts_enabled and tts_enabled_for_chat(store, chat_id, s) and store.tts_can_use(chat_id, s) and len(out) <= s.tts_max_chars:
            if await client.decide_tts(model="grok-4.1-fast", user_text=text, assistant_text=out):
                await _maybe_reply_tts(update, context, out)

        await update.message.reply_text(out)
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")
