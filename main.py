import logging
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from core.config import load_settings
from core.ai import OpenAICompat
from core.database import MemoryStore
from core.tts import AliyunTTS, QwenTTS
from handlers.commands import (
    cmd_start, cmd_models, cmd_model, cmd_persona, cmd_gf, cmd_daily, 
    cmd_ocr, cmd_voice, cmd_mem, cmd_mem_forget, cmd_mem_gc, 
    cmd_ask, cmd_img, cmd_scene, load_personas_dir
)
from handlers.messages import on_photo, on_text

# Logging setup
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tg-grok-bot")

async def post_init(app: Application):
    s = app.bot_data["settings"]
    app.bot_data["client"] = OpenAICompat(base_url=s.openai_base_url, api_key=s.openai_api_key, timeout=s.http_timeout)
    app.bot_data["store"] = MemoryStore(s.memory_db_path)
    app.bot_data["personas"] = load_personas_dir(s.personas_dir)
    
    if s.tts_enabled:
        provider = (s.tts_provider or "dashscope").strip().lower()
        if provider in ("qwen", "qwen-tts", "qwen_tts"):
            app.bot_data["tts"] = QwenTTS(
                api_key=s.dashscope_api_key,
                base_url=s.qwen_tts_base_url,
                model=s.qwen_tts_model,
                voice=s.qwen_tts_voice,
                language_type=s.qwen_tts_language,
                timeout=s.http_timeout,
            )
        else:
            app.bot_data["tts"] = AliyunTTS(
                api_key=s.dashscope_api_key,
                model=s.tts_default_model,
                voice=s.tts_default_voice,
                volume=s.tts_volume,
                fallback_model=s.tts_fallback_model,
                fallback_voice=s.tts_fallback_voice,
            )

async def post_shutdown(app: Application):
    client = app.bot_data.get("client")
    if client:
        await client.close()

def main():
    s = load_settings()
    app = Application.builder().token(s.telegram_token).post_init(post_init).post_shutdown(post_shutdown).build()
    app.bot_data["settings"] = s

    # Register Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("persona", cmd_persona))
    app.add_handler(CommandHandler("gf", cmd_gf))
    app.add_handler(CommandHandler("daily", cmd_daily))
    app.add_handler(CommandHandler("ocr", cmd_ocr))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(CommandHandler("mem", cmd_mem))
    app.add_handler(CommandHandler("mem_forget", cmd_mem_forget))
    app.add_handler(CommandHandler("mem_gc", cmd_mem_gc))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("img", cmd_img))
    app.add_handler(CommandHandler("scene", cmd_scene))
    
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Bot started...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
