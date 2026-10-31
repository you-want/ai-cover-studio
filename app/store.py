"""SQLite 持久化：声音样本与翻唱任务。线程安全。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    duration REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    title TEXT,
    song_name TEXT,
    song_path TEXT,
    engine TEXT,
    separator TEXT,
    voice_sample_id TEXT,
    options TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT DEFAULT '',
    progress REAL DEFAULT 0,
    error TEXT,
    notes TEXT,
    created_at REAL,
    updated_at REAL
);
"""


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["options"] = json.loads(d.get("options") or "{}")
    d["notes"] = json.loads(d.get("notes") or "[]")
    return d


class Store:
    def __init__(self, path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur

    # ---------- 声音样本 ----------

    def add_sample(self, sid: str, name: str, path, duration: float):
        self._exec("INSERT INTO samples (id, name, path, duration, created_at) VALUES (?,?,?,?,?)",
                   (sid, name, str(path), float(duration), time.time()))

    def get_sample(self, sid: str) -> dict | None:
        row = self._exec("SELECT * FROM samples WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def list_samples(self) -> list[dict]:
        rows = self._exec("SELECT * FROM samples ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_sample(self, sid: str):
        self._exec("DELETE FROM samples WHERE id=?", (sid,))

    # ---------- 翻唱任务 ----------

    def create_job(self, jid: str, *, title: str, song_name: str, song_path,
                   engine: str, separator: str, voice_sample_id: str, options: dict):
        self._exec(
            "INSERT INTO jobs (id, title, song_name, song_path, engine, separator,"
            " voice_sample_id, options, status, stage, progress, notes, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?, 'queued', '', 0, '[]', ?, ?)",
            (jid, title, song_name, str(song_path), engine, separator,
             voice_sample_id, json.dumps(options, ensure_ascii=False), time.time(), time.time()))

    # jobs 表允许更新的字段白名单（防止将来把用户可控字段名直接拼进 SQL）
    _UPDATABLE_FIELDS = {"status", "stage", "progress", "error", "notes", "updated_at"}

    def update_job(self, jid: str, **fields):
        if not fields:
            return
        fields["updated_at"] = time.time()
        bad = [k for k in fields if k not in self._UPDATABLE_FIELDS]
        if bad:
            raise ValueError(f"非法的 jobs 更新字段: {bad}")
        sets = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), jid))

    def get_job(self, jid: str) -> dict | None:
        row = self._exec("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return _row_to_dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        rows = self._exec("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def delete_job(self, jid: str):
        self._exec("DELETE FROM jobs WHERE id=?", (jid,))
