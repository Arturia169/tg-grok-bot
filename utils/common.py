import re
from typing import Optional

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

def redact_secrets(text: str) -> str:
    if not text:
        return text
    # crude redaction for common secrets
    text = re.sub(r"sk-[A-Za-z0-9]{10,}", "sk-***", text)
    text = re.sub(r"\b\d{9,10}:[A-Za-z0-9_-]{20,}\b", "<telegram-bot-token>", text)
    return text

def limit_chars(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()
