"""
SQLite storage for highlights, translations, chats, vocab (SM-2 SRS), and a Claude prompt cache.
"""

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

DB_PATH = os.environ.get("READER_DB", "reader_data.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS highlights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    chapter_index INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('translate', 'chat')),
    text TEXT NOT NULL,
    sentence_hash TEXT NOT NULL,
    start_hint TEXT NOT NULL,
    end_hint TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_highlights_chapter ON highlights (book_id, chapter_index);

CREATE TABLE IF NOT EXISTS translations (
    highlight_id INTEGER PRIMARY KEY,
    sentence_en TEXT NOT NULL,
    context_note TEXT NOT NULL DEFAULT '',
    words_json TEXT,
    FOREIGN KEY (highlight_id) REFERENCES highlights (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    highlight_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (highlight_id) REFERENCES highlights (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chats_highlight ON chats (highlight_id, created_at);

CREATE TABLE IF NOT EXISTS vocab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fr TEXT NOT NULL,
    en TEXT NOT NULL,
    context_sentence TEXT NOT NULL DEFAULT '',
    source_book_id TEXT,
    ease REAL NOT NULL DEFAULT 2.5,
    interval_days REAL NOT NULL DEFAULT 0,
    due_at REAL NOT NULL,
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE (fr, en)
);
CREATE INDEX IF NOT EXISTS idx_vocab_due ON vocab (due_at);

CREATE TABLE IF NOT EXISTS translation_cache (
    prompt_sha256 TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def init() -> None:
    with connect() as c:
        c.executescript(SCHEMA)
        # Idempotent migrations for existing dbs
        cols = {r["name"] for r in c.execute("PRAGMA table_info(translations)").fetchall()}
        if "verbs_json" not in cols:
            c.execute("ALTER TABLE translations ADD COLUMN verbs_json TEXT")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(r: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(r) if r is not None else None


# ---------- Highlights ----------

def create_highlight(
    book_id: str, chapter_index: int, kind: str, text: str,
    start_hint: str, end_hint: str,
) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO highlights (book_id, chapter_index, kind, text, sentence_hash, "
            "start_hint, end_hint, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (book_id, chapter_index, kind, text, h, start_hint, end_hint, time.time()),
        )
        return cur.lastrowid


def get_highlight(highlight_id: int) -> Optional[dict]:
    with connect() as c:
        return _row(c.execute("SELECT * FROM highlights WHERE id = ?", (highlight_id,)).fetchone())


def list_highlights(book_id: str, chapter_index: int) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT h.*, t.sentence_en, t.context_note "
            "FROM highlights h LEFT JOIN translations t ON t.highlight_id = h.id "
            "WHERE h.book_id = ? AND h.chapter_index = ? ORDER BY h.created_at",
            (book_id, chapter_index),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_highlight(highlight_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM highlights WHERE id = ?", (highlight_id,))


# ---------- Translations ----------

def save_translation(highlight_id: int, sentence_en: str, context_note: str) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO translations (highlight_id, sentence_en, context_note, words_json) "
            "VALUES (?, ?, ?, COALESCE((SELECT words_json FROM translations WHERE highlight_id = ?), NULL))",
            (highlight_id, sentence_en, context_note, highlight_id),
        )


def get_translation(highlight_id: int) -> Optional[dict]:
    with connect() as c:
        r = _row(c.execute("SELECT * FROM translations WHERE highlight_id = ?", (highlight_id,)).fetchone())
        if r and r.get("words_json"):
            r["words"] = json.loads(r["words_json"])
        if r and r.get("verbs_json"):
            r["verbs"] = json.loads(r["verbs_json"])
        return r


def save_words(highlight_id: int, words: list[dict]) -> None:
    with connect() as c:
        c.execute(
            "UPDATE translations SET words_json = ? WHERE highlight_id = ?",
            (json.dumps(words, ensure_ascii=False), highlight_id),
        )


def save_verbs(highlight_id: int, verbs: list[dict]) -> None:
    with connect() as c:
        c.execute(
            "UPDATE translations SET verbs_json = ? WHERE highlight_id = ?",
            (json.dumps(verbs, ensure_ascii=False), highlight_id),
        )


# ---------- Chats ----------

def add_chat_message(highlight_id: int, role: str, content: str) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO chats (highlight_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (highlight_id, role, content, time.time()),
        )
        return cur.lastrowid


def list_chat_messages(highlight_id: int) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, role, content, created_at FROM chats WHERE highlight_id = ? ORDER BY id",
            (highlight_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- Vocab + SM-2 ----------

def add_vocab(fr: str, en: str, context_sentence: str, source_book_id: Optional[str]) -> int:
    now = time.time()
    with connect() as c:
        try:
            cur = c.execute(
                "INSERT INTO vocab (fr, en, context_sentence, source_book_id, due_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fr, en, context_sentence, source_book_id, now, now),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            row = c.execute("SELECT id FROM vocab WHERE fr = ? AND en = ?", (fr, en)).fetchone()
            return row["id"]


def list_due_vocab(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM vocab WHERE due_at <= ? ORDER BY due_at LIMIT ?",
            (time.time(), limit),
        ).fetchall()
        return [dict(r) for r in rows]


def list_all_vocab(limit: int = 500) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM vocab ORDER BY created_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def vocab_stats() -> dict:
    with connect() as c:
        total = c.execute("SELECT COUNT(*) AS n FROM vocab").fetchone()["n"]
        due = c.execute("SELECT COUNT(*) AS n FROM vocab WHERE due_at <= ?", (time.time(),)).fetchone()["n"]
        return {"total": total, "due": due}


# SM-2 grades: 0 Again, 1 Hard, 2 Good, 3 Easy
DAY = 86400.0
AGAIN_DELAY = 10 * 60.0


def review_vocab(vocab_id: int, grade: int) -> Optional[dict]:
    if grade not in (0, 1, 2, 3):
        raise ValueError("grade must be 0..3")
    with connect() as c:
        r = c.execute("SELECT * FROM vocab WHERE id = ?", (vocab_id,)).fetchone()
        if r is None:
            return None
        ease = r["ease"]
        interval = r["interval_days"]
        reps = r["reps"]
        lapses = r["lapses"]
        now = time.time()

        if grade == 0:
            ease = max(1.3, ease - 0.2)
            interval = 0
            lapses += 1
            due_at = now + AGAIN_DELAY
        else:
            if grade == 1:
                ease = max(1.3, ease - 0.15)
                interval = max(1.0, interval * 1.2) if interval > 0 else 1.0
            elif grade == 2:
                interval = 1.0 if interval == 0 else interval * ease
            else:  # 3 Easy
                ease = ease + 0.15
                interval = (1.0 if interval == 0 else interval * ease) * 1.3
            reps += 1
            due_at = now + interval * DAY

        c.execute(
            "UPDATE vocab SET ease = ?, interval_days = ?, reps = ?, lapses = ?, due_at = ? WHERE id = ?",
            (ease, interval, reps, lapses, due_at, vocab_id),
        )
        return dict(c.execute("SELECT * FROM vocab WHERE id = ?", (vocab_id,)).fetchone())


# ---------- Translation cache ----------

def cache_get(prompt: str) -> Optional[dict]:
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    with connect() as c:
        r = c.execute("SELECT response_json FROM translation_cache WHERE prompt_sha256 = ?", (key,)).fetchone()
        return json.loads(r["response_json"]) if r else None


def cache_put(prompt: str, response: Any) -> None:
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO translation_cache (prompt_sha256, response_json, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(response, ensure_ascii=False), time.time()),
        )
