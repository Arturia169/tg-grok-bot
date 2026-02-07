from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env relative to the project root (parent of 'core' directory)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

@dataclass
class Settings:
    telegram_token: str
    openai_base_url: str
    openai_api_key: str
    default_model: str
    image_model: str
    image_edit_model: str
    allowed_chat_ids: set[int]
    http_timeout: float

    # Aliyun Model Studio (DashScope) TTS
    dashscope_api_key: str
    tts_enabled: bool
    tts_provider: str

    # DashScope (CosyVoice) TTS
    tts_default_model: str
    tts_default_voice: str
    tts_fallback_model: str
    tts_fallback_voice: str

    # Qwen-TTS
    qwen_tts_base_url: str
    qwen_tts_model: str
    qwen_tts_voice: str
    qwen_tts_language: str

    # Shared TTS limits
    tts_daily_limit: int
    tts_max_chars: int
    tts_volume: int
    tts_default_on: bool

    # Memory persistence
    memory_db_path: str
    memory_max_log_messages: int
    memory_summary_max_chars: int
    memory_summary_update_every: int

    # Personas
    personas_dir: str

    # Scene prompt summarization
    scene_summarize: bool
    scene_summarize_model: str

    # OCR
    ocr_enabled: bool
    ocr_default_on: bool
    ocr_model: str


def _env_required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"missing env var: {name}")
    return v


def load_settings() -> Settings:
    token = _env_required("TELEGRAM_BOT_TOKEN")
    base = _env_required("OPENAI_BASE_URL").rstrip("/")
    key = _env_required("OPENAI_API_KEY")

    default_model = os.environ.get("DEFAULT_MODEL", "grok-4.1-fast")
    # Some providers require `grok-imagine-1.0` for /images/generations.
    image_model = os.environ.get("IMAGE_MODEL", "grok-imagine-1.0")
    # Keep a separate model id for /images/edits (if supported by the upstream).
    image_edit_model = os.environ.get("IMAGE_EDIT_MODEL", "grok-imagine-1.0-edit")

    allowed_raw = os.environ.get("ALLOWED_CHAT_IDS", "").strip()
    allowed: set[int] = set()
    if allowed_raw:
        for part in allowed_raw.split(","):
            part = part.strip()
            if part:
                allowed.add(int(part))

    timeout = float(os.environ.get("HTTP_TIMEOUT", "120"))

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    tts_enabled = os.environ.get("TTS_ENABLED", "0").strip() in ("1", "true", "True")
    tts_provider = os.environ.get("TTS_PROVIDER", "dashscope").strip() or "dashscope"

    # DashScope (CosyVoice) TTS
    tts_model = os.environ.get("TTS_DEFAULT_MODEL", "cosyvoice-v3-flash").strip() or "cosyvoice-v3-flash"
    tts_voice = os.environ.get("TTS_DEFAULT_VOICE", "longyang").strip() or "longyang"
    tts_fallback_model = os.environ.get("TTS_FALLBACK_MODEL", "cosyvoice-v3-flash").strip() or "cosyvoice-v3-flash"
    tts_fallback_voice = os.environ.get("TTS_FALLBACK_VOICE", "longyang").strip() or "longyang"

    # Qwen-TTS
    qwen_tts_base_url = os.environ.get("QWEN_TTS_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").strip().rstrip("/")
    qwen_tts_model = os.environ.get("QWEN_TTS_MODEL", "qwen3-tts-flash").strip() or "qwen3-tts-flash"
    qwen_tts_voice = os.environ.get("QWEN_TTS_VOICE", "Cherry").strip() or "Cherry"
    qwen_tts_language = os.environ.get("QWEN_TTS_LANGUAGE", "Chinese").strip() or "Chinese"

    # Shared limits
    tts_daily_limit = int(os.environ.get("TTS_DAILY_LIMIT", "10"))
    tts_max_chars = int(os.environ.get("TTS_MAX_CHARS", "200"))
    tts_volume = int(os.environ.get("TTS_VOLUME", "100"))
    tts_default_on = os.environ.get("TTS_DEFAULT_ON", "1").strip() in ("1", "true", "True")
    if tts_volume < 0:
        tts_volume = 0
    if tts_volume > 100:
        tts_volume = 100

    memory_db_path = os.environ.get("MEMORY_DB_PATH", "/data/memory.sqlite").strip() or "/data/memory.sqlite"
    memory_max_log_messages = int(os.environ.get("MEMORY_MAX_LOG_MESSAGES", "120"))
    memory_summary_max_chars = int(os.environ.get("MEMORY_SUMMARY_MAX_CHARS", "2000"))
    memory_summary_update_every = int(os.environ.get("MEMORY_SUMMARY_UPDATE_EVERY", "12"))

    # Personas_dir should be absolute or relative to project root
    personas_dir = os.environ.get("PERSONAS_DIR", "/data/personas").strip() or "/data/personas"

    scene_summarize = os.environ.get("SCENE_SUMMARIZE", "1").strip() in ("1", "true", "True")
    # Default to the same model used for chat; some proxies reject unknown model ids.
    scene_summarize_model = os.environ.get("SCENE_SUMMARIZE_MODEL", default_model).strip() or default_model

    ocr_enabled = os.environ.get("OCR_ENABLED", "1").strip() in ("1", "true", "True")
    ocr_default_on = os.environ.get("OCR_DEFAULT_ON", "0").strip() in ("1", "true", "True")
    ocr_model = os.environ.get("OCR_MODEL", "grok-4.1-fast").strip() or "grok-4.1-fast"

    return Settings(
        telegram_token=token,
        openai_base_url=base,
        openai_api_key=key,
        default_model=default_model,
        image_model=image_model,
        image_edit_model=image_edit_model,
        allowed_chat_ids=allowed,
        http_timeout=timeout,
        dashscope_api_key=dashscope_key,
        tts_enabled=tts_enabled,
        tts_provider=tts_provider,
        tts_default_model=tts_model,
        tts_default_voice=tts_voice,
        tts_fallback_model=tts_fallback_model,
        tts_fallback_voice=tts_fallback_voice,
        qwen_tts_base_url=qwen_tts_base_url,
        qwen_tts_model=qwen_tts_model,
        qwen_tts_voice=qwen_tts_voice,
        qwen_tts_language=qwen_tts_language,
        tts_daily_limit=tts_daily_limit,
        tts_max_chars=tts_max_chars,
        tts_volume=tts_volume,
        tts_default_on=tts_default_on,
        memory_db_path=memory_db_path,
        memory_max_log_messages=memory_max_log_messages,
        memory_summary_max_chars=memory_summary_max_chars,
        memory_summary_update_every=memory_summary_update_every,
        personas_dir=personas_dir,
        scene_summarize=scene_summarize,
        scene_summarize_model=scene_summarize_model,
        ocr_enabled=ocr_enabled,
        ocr_default_on=ocr_default_on,
        ocr_model=ocr_model,
    )
