import os
import sqlite3
import time
from core.config import Settings

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
