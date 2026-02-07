from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import sqlite3
import time
import wave
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from PIL import Image
import pytesseract

import httpx
from dotenv import load_dotenv
from telegram import InputMediaPhoto, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

try:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer
except Exception:  # optional dependency
    dashscope = None
    SpeechSynthesizer = None

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tg-grok-bot")


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

    personas_dir = os.environ.get("PERSONAS_DIR", "/data/personas").strip() or "/data/personas"

    scene_summarize = os.environ.get("SCENE_SUMMARIZE", "1").strip() in ("1", "true", "True")
    # Default to the same model used for chat; some proxies reject unknown model ids.
    scene_summarize_model = os.environ.get("SCENE_SUMMARIZE_MODEL", default_model).strip() or default_model

    ocr_enabled = os.environ.get("OCR_ENABLED", "1").strip() in ("1", "true", "True")
    ocr_default_on = os.environ.get("OCR_DEFAULT_ON", "0").strip() in ("1", "true", "True")
    ocr_model = os.environ.get("OCR_MODEL", "grok-4.1-expert").strip() or "grok-4.1-expert"

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


class OpenAICompat:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=10),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def images_generate(
        self,
        model: str,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            "stream": False,
        }
        r = await self._client.post("/images/generations", json=payload)
        r.raise_for_status()
        return r.json()

    async def images_edits(
        self,
        model: str,
        prompt: str,
        image_bytes: bytes,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> dict:
        files = {
            "image": ("image.png", image_bytes, "image/png"),
        }
        data = {
            "model": model,
            "prompt": prompt,
            "n": str(n),
            "size": size,
            "response_format": response_format,
            "stream": "false",
        }
        # multipart/form-data, do not set Content-Type header manually
        headers = {k: v for k, v in self._client.headers.items() if k.lower() != "content-type"}
        r = await self._client.post("/images/edits", data=data, files=files, headers=headers)
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self._client.aclose()

    async def chat(self, model: str, user_text: str, system_prompt: str = "你是一个乐于助人的助手。") -> str:
        return await self.chat_messages(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )

    async def chat_messages(self, model: str, messages: list[dict], temperature: float = 0.7) -> str:
        payload = {
            "model": model,
            "stream": False,
            "temperature": temperature,
            "messages": messages,
        }
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""

    async def decide_tts(self, model: str, user_text: str, assistant_text: str) -> bool:
        # Minimal token usage classifier. Output must be exactly VOICE or TEXT.
        prompt = (
            "你是一个分类器。只输出一个单词：VOICE 或 TEXT。\n"
            "规则：\n"
            "- 如果内容包含代码块/命令/大量链接/表格/长清单，选 TEXT。\n"
            "- 如果是简短日常对话、情绪安慰、鼓励、结论、提醒，选 VOICE。\n"
            "- 默认偏向 TEXT（省额度）。\n\n"
            f"用户：{user_text}\n"
            f"助手拟回复：{assistant_text}\n"
        )
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "max_tokens": 1,
            "messages": [
                {"role": "system", "content": "You are a strict classifier."},
                {"role": "user", "content": prompt},
            ],
        }
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        out = ""
        try:
            out = (data["choices"][0]["message"]["content"] or "").strip().upper()
        except Exception:
            out = ""
        return out == "VOICE"


class QwenTTS:
    def __init__(self, api_key: str, base_url: str, model: str, voice: str, language_type: str, timeout: float):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").strip().rstrip("/")
        self.model = (model or "qwen3-tts-flash").strip() or "qwen3-tts-flash"
        self.voice = (voice or "Cherry").strip() or "Cherry"
        self.language_type = (language_type or "Chinese").strip() or "Chinese"
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key) and bool(self.base_url)

    def synthesize_wav(self, text: str) -> bytes:
        if not self.available():
            raise RuntimeError("QwenTTS not configured")

        url = self.base_url + "/services/aigc/multimodal-generation/generation"
        payload = {
            "model": self.model,
            "input": {
                "text": text,
                "voice": self.voice,
                "language_type": self.language_type,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10), follow_redirects=True) as c:
            r = c.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            audio_url = (((data or {}).get("output") or {}).get("audio") or {}).get("url")
            if not audio_url:
                raise RuntimeError(f"QwenTTS returned no audio url: {str(data)[:200]}")

            r2 = c.get(audio_url)
            r2.raise_for_status()
            return bytes(r2.content)


class AliyunTTS:
    def __init__(self, api_key: str, model: str, voice: str, volume: int, fallback_model: str, fallback_voice: str):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.volume = volume
        self.fallback_model = fallback_model
        self.fallback_voice = fallback_voice

    def available(self) -> bool:
        return bool(self.api_key) and dashscope is not None and SpeechSynthesizer is not None

    def synthesize_mp3(self, text: str) -> bytes:
        # DashScope SDK uses env var or dashscope.api_key
        dashscope.api_key = self.api_key
        try:
            synthesizer = SpeechSynthesizer(model=self.model, voice=self.voice, volume=self.volume)
            audio = synthesizer.call(text)
            return bytes(audio)
        except Exception:
            # Fallback to cheaper model/voice
            if self.fallback_model and self.fallback_voice:
                synthesizer = SpeechSynthesizer(model=self.fallback_model, voice=self.fallback_voice, volume=self.volume)
                audio = synthesizer.call(text)
                return bytes(audio)
            raise


IMG_MD_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
URL_RE = re.compile(r"https?://\S+")
JSON_URL_RE = re.compile(r"\"url\"\s*:\s*\"(https?://[^\"]+)\"")


def extract_image_urls(text: str) -> list[str]:
    """Extract likely image URLs from upstream output.

    Upstream image models often return:
    - Markdown image: ![image](https://...)
    - Plain URL (sometimes without file extension)
    - JSON-ish payloads containing a url field

    We keep this permissive because we validate via Content-Type on download.
    """

    if not text:
        return []

    urls = IMG_MD_RE.findall(text)
    if urls:
        return urls

    urls = JSON_URL_RE.findall(text)
    if urls:
        return urls

    # fallback: any http(s) URL, but avoid picking obvious non-image links
    candidates = URL_RE.findall(text)
    out: list[str] = []
    for u in candidates:
        ul = u.lower()
        if any(x in ul for x in (".mp4", ".mov", ".m3u8", ".mp3")):
            continue
        out.append(u.rstrip("),.\"'"))

    # de-dup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped


async def download_images(urls: Iterable[str], timeout: float) -> list[bytes]:
    images: list[bytes] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10), follow_redirects=True) as c:
        for u in urls:
            try:
                r = await c.get(u)
                r.raise_for_status()
                ct = r.headers.get("content-type", "")
                if "image" not in ct:
                    continue
                images.append(r.content)
            except Exception:
                continue
    return images


async def download_first_image(urls: list[str], timeout: float, retries: int = 2) -> Optional[bytes]:
    if not urls:
        return None
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10), follow_redirects=True) as c:
        for attempt in range(retries + 1):
            for u in urls:
                try:
                    r = await c.get(u)
                    r.raise_for_status()
                    ct = r.headers.get("content-type", "")
                    if "image" not in ct:
                        continue
                    return r.content
                except Exception:
                    continue
            # brief backoff
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


def parse_model_arg(text: str) -> tuple[Optional[str], str]:
    # supports: /ask grok-4-fast hello
    parts = (text or "").strip().split(maxsplit=2)
    if len(parts) >= 3:
        return parts[1], parts[2]
    if len(parts) == 2:
        return None, parts[1]
    return None, ""


def is_allowed(chat_id: int, allowed: set[int]) -> bool:
    if not allowed:
        return True
    return chat_id in allowed


def _today_key() -> str:
    return time.strftime("%Y-%m-%d")


class MemoryStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_state (
                    chat_id INTEGER PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    persona_key TEXT NOT NULL DEFAULT 'default',
                    tts_enabled INTEGER NOT NULL DEFAULT 0,
                    tts_day TEXT NOT NULL DEFAULT '',
                    tts_count INTEGER NOT NULL DEFAULT 0,
                    ocr_enabled INTEGER NOT NULL DEFAULT 0,
                    gf_enabled INTEGER NOT NULL DEFAULT 0,
                    last_daily TEXT NOT NULL DEFAULT '',
                    msg_since_summary INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS message_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_message_log_chat ON message_log(chat_id, id)")

            # Migrations (best-effort)
            for stmt in (
                "ALTER TABLE chat_state ADD COLUMN msg_since_summary INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE chat_state ADD COLUMN persona_key TEXT NOT NULL DEFAULT 'default'",
                "ALTER TABLE chat_state ADD COLUMN ocr_enabled INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE chat_state ADD COLUMN gf_enabled INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE chat_state ADD COLUMN last_daily TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    c.execute(stmt)
                except Exception:
                    pass

    def ensure_chat(self, chat_id: int, settings: Settings) -> None:
        with self._conn() as c:
            row = c.execute("SELECT chat_id FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO chat_state(chat_id, persona_key, tts_enabled, ocr_enabled, gf_enabled, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        chat_id,
                        "default",
                        1 if settings.tts_default_on else 0,
                        1 if settings.ocr_default_on else 0,
                        0,
                        int(time.time()),
                    ),
                )

    def get_state(self, chat_id: int) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT * FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
            return dict(row) if row else {}

    def set_model(self, chat_id: int, model: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE chat_state SET model=?, updated_at=? WHERE chat_id=?", (model, int(time.time()), chat_id))

    def set_persona(self, chat_id: int, persona_key: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chat_state SET persona_key=?, updated_at=? WHERE chat_id=?",
                (persona_key, int(time.time()), chat_id),
            )

    def set_tts_enabled(self, chat_id: int, enabled: bool) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chat_state SET tts_enabled=?, updated_at=? WHERE chat_id=?",
                (1 if enabled else 0, int(time.time()), chat_id),
            )

    def set_ocr_enabled(self, chat_id: int, enabled: bool) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chat_state SET ocr_enabled=?, updated_at=? WHERE chat_id=?",
                (1 if enabled else 0, int(time.time()), chat_id),
            )

    def set_gf_enabled(self, chat_id: int, enabled: bool) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chat_state SET gf_enabled=?, updated_at=? WHERE chat_id=?",
                (1 if enabled else 0, int(time.time()), chat_id),
            )

    def set_last_daily(self, chat_id: int, day: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chat_state SET last_daily=?, updated_at=? WHERE chat_id=?",
                (day, int(time.time()), chat_id),
            )

    def tts_can_use(self, chat_id: int, settings: Settings) -> bool:
        day = _today_key()
        with self._conn() as c:
            row = c.execute("SELECT tts_day, tts_count FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
            if not row:
                return False
            tts_day = row["tts_day"]
            tts_count = int(row["tts_count"])
            if tts_day != day:
                tts_count = 0
                c.execute("UPDATE chat_state SET tts_day=?, tts_count=? WHERE chat_id=?", (day, 0, chat_id))
            return tts_count < int(settings.tts_daily_limit)

    def tts_mark_used(self, chat_id: int) -> None:
        day = _today_key()
        with self._conn() as c:
            row = c.execute("SELECT tts_day, tts_count FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
            if not row:
                return
            tts_day = row["tts_day"]
            tts_count = int(row["tts_count"])
            if tts_day != day:
                tts_count = 0
                c.execute("UPDATE chat_state SET tts_day=?, tts_count=? WHERE chat_id=?", (day, 0, chat_id))
            c.execute("UPDATE chat_state SET tts_count=? WHERE chat_id=?", (tts_count + 1, chat_id))

    def append_message(self, chat_id: int, role: str, content: str) -> None:
        now = int(time.time())
        with self._conn() as c:
            c.execute(
                "INSERT INTO message_log(chat_id, role, content, created_at) VALUES(?, ?, ?, ?)",
                (chat_id, role, content, now),
            )
            if role == "user":
                c.execute(
                    "UPDATE chat_state SET msg_since_summary = msg_since_summary + 1, updated_at=? WHERE chat_id=?",
                    (now, chat_id),
                )

    def get_recent_messages(self, chat_id: int, limit: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content FROM message_log WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        out = [dict(r) for r in rows]
        out.reverse()
        return out

    def get_summary(self, chat_id: int) -> str:
        with self._conn() as c:
            row = c.execute("SELECT summary FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
            return (row["summary"] if row else "") or ""

    def set_summary(self, chat_id: int, summary: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE chat_state SET summary=?, msg_since_summary=0, updated_at=? WHERE chat_id=?",
                (summary, int(time.time()), chat_id),
            )

    def gc(self, chat_id: int, max_messages: int) -> int:
        # Keep only the last max_messages for this chat.
        with self._conn() as c:
            row = c.execute("SELECT COUNT(1) AS n FROM message_log WHERE chat_id=?", (chat_id,)).fetchone()
            n = int(row["n"]) if row else 0
            if n <= max_messages:
                return 0
            to_delete = n - max_messages
            # delete oldest rows
            ids = c.execute(
                "SELECT id FROM message_log WHERE chat_id=? ORDER BY id ASC LIMIT ?",
                (chat_id, to_delete),
            ).fetchall()
            if not ids:
                return 0
            c.executemany("DELETE FROM message_log WHERE id=?", [(int(r["id"]),) for r in ids])
            return to_delete


def load_personas_dir(dir_path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    try:
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

            # Format:
            # line1: persona display name (optional)
            # rest: system prompt
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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    await update.message.reply_text(
        "你好！\n\n"
        "可用命令：\n"
        "/models - 查看可用模型列表\n"
        "/model <模型ID> - 设置本聊天默认模型\n"
        "/persona [人格键] - 查看/切换人格\n"
        "/voice on|off - 开关语音回复（受每日限额影响）\n"
        "/mem - 查看长期记忆摘要\n"
        "/mem_forget - 清空长期记忆摘要\n"
        "/mem_gc - 清理旧对话记录\n"
        "/ask [模型ID] <内容> - 提问（可选指定模型）\n"
        "/img <提示词> - 使用 grok-imagine-1.0 生成图片\n"
        "/scene <场景描述> - 生成二次元情景配图（可回复一条消息再发送 /scene）\n"
        "\n"
        "小提示：你也可以直接发文字，我会用默认模型回复。"
    )


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    # Static list (matches your grok2api deployment)
    await update.message.reply_text(
        "可用模型：\n"
        "- grok-4\n"
        "- grok-4-fast\n"
        "- grok-4.1\n"
        "- grok-4.1-thinking\n"
        "- grok-imagine-1.0\n"
        "\n"
        "提示：使用 /model grok-4-fast 设置默认模型\n"
        "语音：使用 /voice on 开启语音回复（每日限额）"
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    args = context.args or []
    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)

    if not args:
        st = store.get_state(update.effective_chat.id)
        cur = (st.get("model") if st else "") or s.default_model
        await update.message.reply_text(f"当前默认模型：{cur}")
        return

    mid = args[0].strip()
    store.set_model(update.effective_chat.id, mid)
    await update.message.reply_text(f"好的，已设置本聊天默认模型为：{mid}")


async def cmd_expert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick toggle for grok-4.1-expert in the current chat."""

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
        await update.message.reply_text(
            "用法：/expert on 或 /expert off\n"
            f"当前默认模型：{cur}"
        )
        return

    opt = args[0].strip().lower()
    if opt in ("on", "1", "true", "enable"):
        store.set_model(update.effective_chat.id, "grok-4.1-expert")
        await update.message.reply_text("好的，已切到：grok-4.1-expert")
        return
    if opt in ("off", "0", "false", "disable"):
        store.set_model(update.effective_chat.id, s.default_model)
        await update.message.reply_text(f"好的，已切回默认模型：{s.default_model}")
        return

    await update.message.reply_text("用法：/expert on 或 /expert off")


async def cmd_persona(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    store: MemoryStore = context.application.bot_data["store"]
    personas: Dict[str, Dict[str, str]] = context.application.bot_data["personas"]
    store.ensure_chat(update.effective_chat.id, s)
    st = store.get_state(update.effective_chat.id)

    args = context.args or []
    if not args:
        key = (st.get("persona_key") if st else "") or "default"
        name = personas.get(key, personas.get("default", {})).get("name", key)
        keys = ", ".join(sorted(personas.keys()))
        await update.message.reply_text(
            f"当前人格：{name}（{key}）\n"
            f"可用人格键：{keys}\n"
            f"用法：/persona <人格键>"
        )
        return

    key = args[0].strip()
    if key not in personas:
        keys = ", ".join(sorted(personas.keys()))
        await update.message.reply_text(f"未知人格键：{key}\n可用：{keys}")
        return

    store.set_persona(update.effective_chat.id, key)
    name = personas.get(key, {}).get("name", key)
    await update.message.reply_text(f"好的，已切换人格为：{name}（{key}）")


async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return


async def cmd_ocr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]

    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    if not s.ocr_enabled:
        await update.message.reply_text("OCR 功能未启用（服务端 OCR_ENABLED=0）。")
        return

    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    chat_id = update.effective_chat.id

    args = context.args or []
    if args and args[0].strip().lower() in ("on", "1", "true", "enable"):
        store.set_ocr_enabled(chat_id, True)
        await update.message.reply_text("好的，已开启：收到图片时自动 OCR。")
        return
    if args and args[0].strip().lower() in ("off", "0", "false", "disable"):
        store.set_ocr_enabled(chat_id, False)
        await update.message.reply_text("好的，已关闭：图片自动 OCR。")
        return

    # One-shot OCR: must reply to a photo message
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        enabled = ocr_enabled_for_chat(store, chat_id, s)
        await update.message.reply_text(
            "用法：\n"
            "- 回复一张图片发送 /ocr （识别文字+解读）\n"
            "- /ocr on  开启自动识图\n"
            "- /ocr off  关闭自动识图\n"
            f"当前自动识图：{'开启' if enabled else '关闭'}"
        )
        return

    photo = update.message.reply_to_message.photo[-1]
    try:
        f = await context.bot.get_file(photo.file_id)
        data = await f.download_as_bytearray()
        img_bytes = bytes(data)
    except Exception as e:
        await update.message.reply_text(f"下载图片失败：{e}")
        return

    await update.effective_chat.send_action(ChatAction.TYPING)
    text = run_ocr(img_bytes)
    if not text:
        await update.message.reply_text("没识别到清晰文字（可能图片太糊/字体太小/是纯图片无文字）。")
        return

    # Optional question after /ocr
    question = " ".join(args).strip()
    prompt = (
        "你是一个OCR结果解读助手。\n"
        "以下是从图片中识别出来的文字，请先给出简短要点总结，然后回答用户的问题（如果有）。\n"
        "注意：OCR 可能有错字，允许你纠正明显错误。\n\n"
        f"OCR文字：\n{text}\n\n"
        f"用户问题：{question or '（无）'}"
    )

    try:
        out = await client.chat(model=s.ocr_model, user_text=prompt)
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"上游接口错误：{e.response.status_code} {e.response.text}")
        return
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")
        return

    await update.message.reply_text(out or "(无内容)")


async def cmd_gf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    chat_id = update.effective_chat.id

    args = context.args or []
    if not args:
        enabled = gf_enabled_for_chat(store, chat_id, s)
        await update.message.reply_text(
            "用法：/gf on 或 /gf off\n"
            f"当前女友模式：{'开启' if enabled else '关闭'}"
        )
        return

    opt = args[0].strip().lower()
    if opt in ("on", "1", "true", "enable"):
        store.set_gf_enabled(chat_id, True)
        await update.message.reply_text("好的，女友模式已开启🥰")
        return
    if opt in ("off", "0", "false", "disable"):
        store.set_gf_enabled(chat_id, False)
        await update.message.reply_text("好的，女友模式已关闭🍵")
        return

    await update.message.reply_text("用法：/gf on 或 /gf off")


async def cmd_daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]

    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    chat_id = update.effective_chat.id

    day = _today_key()
    st = store.get_state(chat_id)
    if st and (st.get("last_daily") or "") == day:
        await update.message.reply_text("今天的每日一句已经发过啦，明天再来🥰")
        return

    summary = store.get_summary(chat_id)
    prompt = (
        "你要对用户说一句甜甜的每日一句（中文），不超过40个字，最多1个emoji。\n"
        "要自然、贴近生活，不要土味尬撩。\n"
        "如果有用户摘要记忆，可以轻轻带一句（不要暴露隐私）。\n\n"
        f"用户摘要：{(summary or '')[:300]}"
    )

    try:
        out = await client.chat(model=s.default_model, user_text=prompt)
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")
        return

    out = _limit_chars(out, 40)
    store.set_last_daily(chat_id, day)
    await update.message.reply_text(out or "(无内容)")


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
        await update.message.reply_text(
            f"语音回复：{'开启' if enabled else '关闭'}（默认：{'开启' if s.tts_default_on else '关闭'}）\n"
            f"今日用量：{cnt}/{s.tts_daily_limit}\n"
            f"当前音色：{s.tts_default_voice}（{s.tts_default_model}）\n"
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
        await update.message.reply_text("当前还没有长期记忆摘要。")
        return
    await update.message.reply_text("当前长期记忆摘要：\n\n" + summary)


async def cmd_mem_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    store.set_summary(update.effective_chat.id, "")
    await update.message.reply_text("好的，已清空长期记忆摘要。")


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


def _redact_secrets(text: str) -> str:
    if not text:
        return text
    # crude redaction for common secrets
    text = re.sub(r"sk-[A-Za-z0-9]{10,}", "sk-***", text)
    text = re.sub(r"\b\d{9,10}:[A-Za-z0-9_-]{20,}\b", "<telegram-bot-token>", text)
    return text


def _limit_chars(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def run_ocr(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Prefer Chinese+English; fallback to default.
        try:
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        except Exception:
            text = pytesseract.image_to_string(img)
        return (text or "").strip()
    except Exception:
        return ""


async def update_memory_if_needed(store: MemoryStore, client: OpenAICompat, chat_id: int, settings: Settings) -> None:
    st = store.get_state(chat_id)
    if not st:
        return
    n = int(st.get("msg_since_summary", 0))
    if n < int(settings.memory_summary_update_every):
        return

    old = store.get_summary(chat_id)
    recent = store.get_recent_messages(chat_id, limit=20)
    lines = []
    for m in recent:
        role = m.get("role")
        content = _redact_secrets(m.get("content", ""))
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        else:
            lines.append(f"助手：{content}")
    convo = "\n".join(lines)

    prompt = (
        "你是一个记忆整理器。请把‘旧摘要’与‘最近对话’融合成新的长期摘要。\n"
        "要求：\n"
        "- 用中文，尽量精炼（重点是用户偏好、正在做的事、未完成事项、重要事实）。\n"
        "- 不要记录任何密钥、token、账号、网址中的敏感参数。\n"
        "- 不要写流水账，不要写无关闲聊。\n"
        "- 输出只要摘要正文，不要标题，不要列表编号也行。\n\n"
        f"旧摘要：\n{_redact_secrets(old)}\n\n"
        f"最近对话：\n{convo}\n"
    )

    try:
        new_summary = await client.chat(model="grok-4-fast", user_text=prompt, system_prompt="You are a memory summarizer.")
    except Exception:
        return

    new_summary = (new_summary or "").strip()
    if not new_summary:
        return
    if len(new_summary) > settings.memory_summary_max_chars:
        new_summary = new_summary[: settings.memory_summary_max_chars]

    store.set_summary(chat_id, new_summary)


def _trim_wav_bytes(wav_bytes: bytes, max_seconds: float) -> bytes:
    if not wav_bytes:
        return wav_bytes
    try:
        inp = io.BytesIO(wav_bytes)
        with wave.open(inp, "rb") as r:
            fr = r.getframerate() or 1
            max_frames = int(fr * max_seconds)
            frames = r.readframes(max_frames)
            params = r.getparams()

        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(frames)
        return out.getvalue()
    except Exception:
        return wav_bytes


def _wav_to_ogg_opus(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to OGG/Opus for Telegram voice messages.

    Requires ffmpeg in the container.
    """

    import subprocess

    p = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-c:a",
            "libopus",
            "-b:a",
            "24k",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if p.returncode != 0 or not p.stdout:
        err = (p.stderr or b"")[:200].decode("utf-8", "ignore")
        raise RuntimeError(f"ffmpeg opus transcode failed: {err}")

    return bytes(p.stdout)


async def _maybe_reply_tts(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    s: Settings = context.application.bot_data["settings"]
    store: MemoryStore = context.application.bot_data["store"]
    chat_id = update.effective_chat.id

    if not text:
        return False
    if not tts_enabled_for_chat(store, chat_id, s):
        return False
    if not store.tts_can_use(chat_id, s):
        return False

    # Keep audio short. We enforce a conservative text limit and (for WAV) trim by duration.
    tts_text = (text or "").strip()
    tts_text = _limit_chars(tts_text, min(int(s.tts_max_chars), 120))
    if not tts_text:
        return False

    tts = context.application.bot_data.get("tts")
    if not tts or not getattr(tts, "available", lambda: False)():
        return False

    try:
        await update.effective_chat.send_action(ChatAction.RECORD_VOICE)

        if isinstance(tts, QwenTTS):
            wav = await asyncio.to_thread(tts.synthesize_wav, tts_text)
            wav = await asyncio.to_thread(_trim_wav_bytes, wav, 20.0)

            # Convert WAV -> OGG/Opus for Telegram "voice" bubble.
            opus = await asyncio.to_thread(_wav_to_ogg_opus, wav)
            bio = io.BytesIO(opus)
            bio.name = "tts.ogg"
            await update.message.reply_voice(voice=bio)
        else:
            audio = await asyncio.to_thread(tts.synthesize_mp3, tts_text)
            bio = io.BytesIO(audio)
            bio.name = "tts.mp3"
            await update.message.reply_voice(voice=bio)

        store.tts_mark_used(chat_id)
        return True
    except Exception:
        return False


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]

    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    model_arg = None
    text = ""
    if update.message.text:
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

        # If TTS decides to speak, still send the text.
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
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"上游接口错误：{e.response.status_code} {e.response.text}")
        return
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")
        return

    items = (data or {}).get("data") or []
    urls = [it.get("url") for it in items if isinstance(it, dict) and it.get("url")]
    b64s = [it.get("b64_json") for it in items if isinstance(it, dict) and it.get("b64_json")]

    if b64s:
        try:
            img = base64.b64decode(b64s[0])
            await update.message.reply_photo(photo=img)
            return
        except Exception:
            pass

    if not urls:
        snippet = json.dumps(data, ensure_ascii=False)[:800]
        await update.message.reply_text("(未返回图片链接) 上游返回：" + snippet)
        return

    img = await download_first_image(urls[:4], timeout=s.http_timeout, retries=2)
    if not img:
        await update.message.reply_text(
            "图片生成成功但下载失败（可能图片短暂丢失/链接过期）。你可以重试一次 /img。\n\n" + "\n".join(urls[:4])
        )
        return

    await update.message.reply_photo(photo=img)
    await update.message.reply_text("\n".join(urls[:4]), disable_web_page_preview=True)


def _sanitize_scene_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_scene_prompt(scene: str) -> str:
    # Anime-style romantic scene. No explicit sexual content.
    # Character is an inspiration reference, not a direct copy.
    scene = _sanitize_scene_text(scene)

    # Make constraints explicit and repeat them in both CN + EN.
    # Image models can "hallucinate" background people or an extra male body.
    base = (
        "二次元动漫插画风，柔和光影，高质量细节，电影感构图，高分辨率。\n"
        "\n"
        "【硬性输出规范（必须严格遵守）】\n"
        "1) 人物数量：画面里只允许出现 2 个人：‘我(男)’ + ‘女友(女)’。禁止任何第三人（包括路人、服务员、剪影、倒影、海报/屏幕里的人像）。\n"
        "2) POV：男性第一人称视角。男方只能以‘前景两只手/前臂’出镜；禁止出现男方的脸/头部/上半身/完整背影；禁止出现第二个男性。\n"
        "3) 互动：女友只能与‘我’互动，不与任何陌生人互动。\n"
        "4) 手部：只允许出现两只男手（我的）+ 女友的手；禁止多余手/多余手臂。\n"
        "5) 关系：一对异性情侣（1男1女）。禁止女女/百合。\n"
        "6) 内容：浪漫暧昧但不露骨；禁止裸露、禁止性行为、禁止露点。\n"
        "\n"
        "【English constraints】\n"
        "- Exactly 2 people only: me (male POV) and my girlfriend (female). No third person, no bystanders.\n"
        "- Male POV ONLY: show only two male hands/forearms in foreground. No male face/body/back. No other man.\n"
        "- No extra hands/arms. No crowd. No strangers.\n"
        "\n"
        "【强负面词（尽量避免）】\n"
        "third person, extra person, extra people, bystander, crowd, stranger, waiter, background person, silhouette, reflection, poster people,\n"
        "other man, male face, male body, second male, man standing,\n"
        "extra hands, extra arms, multiple hands,\n"
        "text, caption, subtitle, watermark, logo, speech bubble, chat bubble, screenshot, UI\n"
        "\n"
        "女友形象：黑色长发、红色眼睛、成熟温柔的御姐气质，红黑配色，优雅又带点害羞；整体气质参考《间谍过家家》荆棘公主风格元素（仅作灵感参考，不要直接复刻原角色或原服装细节）。\n"
        "场景描述："
    )

    # Repeat the key constraint right before the scene (helps many models).
    tail = (
        "\n\n"
        "再次强调：画面中只能有 2 个人；男方只能是 POV 的两只手/前臂；不允许出现任何其他男性或第三人。\n"
    )

    return base + scene + tail


async def cmd_scene(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]

    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    # Prefer replying to the message you want to illustrate.
    scene_text = " ".join(context.args or []).strip()
    if not scene_text and update.message.reply_to_message:
        rt = update.message.reply_to_message
        if rt.text:
            scene_text = rt.text.strip()
        elif rt.caption:
            scene_text = rt.caption.strip()

    if not scene_text:
        rt = update.message.reply_to_message
        rt_kind = "none"
        if rt is not None:
            if rt.text:
                rt_kind = "text"
            elif rt.caption:
                rt_kind = "caption"
            elif rt.photo:
                rt_kind = "photo"
            else:
                rt_kind = "other"
        await update.message.reply_text(
            "用法：/scene <场景描述>（或回复一条消息再发送 /scene）。"
            f"\n\n(调试信息：reply_kind={rt_kind}, args_len={len(context.args or [])})"
        )
        return

    logger.info(
        "/scene invoked: chat=%s msg=%s reply=%s args=%s",
        update.effective_chat.id,
        update.message.message_id,
        bool(update.message.reply_to_message),
        context.args,
    )
    logger.info("/scene raw text head: %s", (scene_text or "")[:200])

    scene_text = _sanitize_scene_text(scene_text)
    logger.info("/scene sanitized head: %s", scene_text[:200])

    # Step 1: summarize/extract key scene info to improve accuracy
    if s.scene_summarize:
        try:
            sum_prompt = (
                "你是一个场景提炼器。把用户输入的文字提炼成用于‘画面生成’的精炼中文场景描述（1-3句）。\n"
                "要求：\n"
                "- 只保留人物关系、动作、地点、时间、氛围、关键物件。\n"
                "- 把抽象/对话/叙述改写成‘能被画出来’的镜头语言。\n"
                "- 绝对不要把‘原文’当成要画出来的文字：不要出现‘画出这段话/把文字写在图上/截图/字幕/海报文字/聊天气泡文字’等。\n"
                "- 输出里不要出现引号、代码块、markdown、链接。\n"
                "- 输出必须是纯文本，不要编号。\n\n"
                f"原文：{scene_text}"
            )
            scene_text = (await client.chat(model=s.scene_summarize_model, user_text=sum_prompt)).strip() or scene_text
            scene_text = _sanitize_scene_text(scene_text)
            logger.info("/scene summarized: %s", scene_text[:200])
        except Exception as e:
            logger.warning("/scene summarize failed (model=%s): %r", s.scene_summarize_model, e)
            # Fallback: try the chat default model once.
            if s.scene_summarize_model != s.default_model:
                try:
                    scene_text2 = (await client.chat(model=s.default_model, user_text=sum_prompt)).strip() or scene_text
                    scene_text2 = _sanitize_scene_text(scene_text2)
                    scene_text = scene_text2
                    logger.info("/scene summarized (fallback): %s", scene_text[:200])
                except Exception as e2:
                    logger.warning("/scene summarize fallback failed (model=%s): %r", s.default_model, e2)

    # Step 2: build final image prompt
    prompt = build_scene_prompt(scene_text)
    logger.info("/scene prompt head: %s", prompt[:200])

    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    try:
        data = await client.images_generate(model=s.image_model, prompt=prompt, n=1, response_format="url")
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"上游接口错误：{e.response.status_code} {e.response.text}")
        return
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")
        return

    items = (data or {}).get("data") or []
    urls = [it.get("url") for it in items if isinstance(it, dict) and it.get("url")]
    b64s = [it.get("b64_json") for it in items if isinstance(it, dict) and it.get("b64_json")]

    if b64s:
        try:
            img = base64.b64decode(b64s[0])
            await update.message.reply_photo(photo=img)
            return
        except Exception:
            pass

    if not urls:
        snippet = json.dumps(data, ensure_ascii=False)[:800]
        await update.message.reply_text("(未返回图片链接) 上游返回：" + snippet)
        return

    img = await download_first_image(urls[:4], timeout=s.http_timeout, retries=2)
    if not img:
        await update.message.reply_text(
            "图片生成成功但下载失败（可能图片短暂丢失/链接过期）。你可以重试一次 /scene。\n\n" + "\n".join(urls[:4])
        )
        return

    await update.message.reply_photo(photo=img)
    await update.message.reply_text("\n".join(urls[:4]), disable_web_page_preview=True)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]

    if update.effective_chat is None or update.message is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    chat_id = update.effective_chat.id

    if not update.message.photo:
        return

    if not ocr_enabled_for_chat(store, chat_id, s):
        return

    # Auto OCR: if the user provided a caption, treat it as the question.
    question = (update.message.caption or "").strip()

    photo = update.message.photo[-1]
    try:
        f = await context.bot.get_file(photo.file_id)
        data = await f.download_as_bytearray()
        img_bytes = bytes(data)
    except Exception:
        return

    await update.effective_chat.send_action(ChatAction.TYPING)
    text = run_ocr(img_bytes)
    if not text:
        await update.message.reply_text("我没识别到清晰文字（可以试试更清晰的截图/放大文字）。")
        return

    prompt = (
        "你是一个OCR结果解读助手。\n"
        "请基于OCR文字给出要点总结，并回答用户问题（如果有）。\n"
        "注意：OCR可能有错字，允许你纠正明显错误。\n\n"
        f"OCR文字：\n{text}\n\n"
        f"用户问题：{question or '（无）'}"
    )

    try:
        out = await client.chat(model=s.ocr_model, user_text=prompt)
    except Exception:
        out = ""

    if out:
        await update.message.reply_text(out)
    else:
        await update.message.reply_text("已识别到文字，但解读失败了；你可以回复图片后再发 /ocr 重试。")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 兜底：把普通文本当成提问
    s: Settings = context.application.bot_data["settings"]
    client: OpenAICompat = context.application.bot_data["client"]

    if update.effective_chat is None or update.message is None or update.message.text is None:
        return
    if not is_allowed(update.effective_chat.id, s.allowed_chat_ids):
        return

    raw0 = update.message.text
    raw = raw0.strip()
    if not raw:
        return

    # Telegram only treats messages starting with '/' (no leading whitespace) as commands.
    # Users sometimes reply with leading spaces or a fullwidth slash, causing /scene to fall into this handler.
    stripped = raw0.lstrip()
    if stripped.startswith("/scene") or stripped.startswith("/scene@") or stripped.startswith("／scene") or stripped.startswith("／scene@"):
        old_args = getattr(context, "args", None)
        try:
            parts = stripped.replace("／", "/", 1).split()
            context.args = parts[1:]
            await cmd_scene(update, context)
        finally:
            context.args = old_args
        return

    # Force mode prefixes
    force_voice = False
    force_text = False
    text = raw
    for p in ("语音：", "语音:"):
        if text.startswith(p):
            force_voice = True
            text = text[len(p) :].strip()
            break
    for p in ("文字：", "文字:", "text:", "TEXT:"):
        if text.startswith(p):
            force_text = True
            text = text[len(p) :].strip()
            break

    if not text:
        await update.message.reply_text("(无内容)")
        return

    store: MemoryStore = context.application.bot_data["store"]
    store.ensure_chat(update.effective_chat.id, s)
    chat_id = update.effective_chat.id

    # log user
    store.append_message(chat_id, "user", text)

    st = store.get_state(chat_id)
    model = (st.get("model") if st else "") or s.default_model

    # build system prompt with persona + summary
    personas: Dict[str, Dict[str, str]] = context.application.bot_data["personas"]
    persona_key = (st.get("persona_key") if st else "") or "default"
    persona_sys = personas.get(persona_key, personas.get("default", {})).get("system", "你是一个乐于助人的助手。")

    summary = store.get_summary(chat_id)
    sys = persona_sys
    if summary:
        sys += "\n\n关于用户的长期记忆（摘要）：\n" + summary

    if gf_enabled_for_chat(store, chat_id, s):
        sys += (
            "\n\n额外风格要求：你以‘温柔体贴的女友’口吻与用户交流。"
            "语气亲近自然，适度使用emoji（不要堆叠），同时保持回答有用、清晰。"
        )

    # add short history (for real continuity)
    recent = store.get_recent_messages(chat_id, limit=12)
    msgs = [{"role": "system", "content": sys}]
    for m in recent:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        msgs.append({"role": role, "content": content.strip()})

    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        out = await client.chat_messages(model=model, messages=msgs)
        if not out:
            await update.message.reply_text("(无内容)")
            return

        # log assistant
        store.append_message(chat_id, "assistant", out)

        # memory update + gc
        await update_memory_if_needed(store, client, chat_id, s)
        store.gc(chat_id, s.memory_max_log_messages)

        # TTS decision
        if force_voice:
            # When force voice is requested, constrain the next output to <=30 chars.
            short_out = out
            try:
                compress_prompt = (
                    "请把下面这段回复压缩成不超过30个汉字的中文短句（只输出短句本身）。\n"
                    "要求：\n"
                    "- 必须 <=30 个字（含标点）。\n"
                    "- 保留核心意思，语气自然。\n"
                    "- 不要表情包式长串emoji，最多1个。\n\n"
                    f"原文：{out}"
                )
                short_out = (await client.chat(model="grok-4-fast", user_text=compress_prompt)).strip() or out
            except Exception:
                short_out = out

            short_out = _limit_chars(short_out, 30)

            if await _maybe_reply_tts(update, context, short_out):
                await update.message.reply_text(short_out)
                return

            await update.message.reply_text(short_out)
            return

        if force_text:
            await update.message.reply_text(out)
            return

        # auto decision (方案2)
        use_voice = False
        if s.tts_enabled and tts_enabled_for_chat(store, chat_id, s) and store.tts_can_use(chat_id, s) and len(out) <= s.tts_max_chars:
            try:
                use_voice = await client.decide_tts(model="grok-4-fast", user_text=text, assistant_text=out)
            except Exception:
                use_voice = False

        if use_voice:
            # If TTS decides to speak, still send the text.
            await _maybe_reply_tts(update, context, out)

        await update.message.reply_text(out)
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(f"上游接口错误：{e.response.status_code} {e.response.text}")
    except Exception as e:
        await update.message.reply_text(f"错误：{e}")


async def post_init(app: Application):
    s: Settings = app.bot_data["settings"]
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
    client: OpenAICompat = app.bot_data.get("client")
    if client:
        await client.close()


def main():
    s = load_settings()

    app = (
        Application.builder()
        .token(s.telegram_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.bot_data["settings"] = s

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("expert", cmd_expert))
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

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
