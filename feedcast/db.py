"""SQLite pool: ingested items, Laya answer cache, judgment log, embeddings, run history."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    outlet TEXT NOT NULL,
    feed_url TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    lead TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    published REAL NOT NULL,
    fetched REAL NOT NULL,
    full_text INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'feed'
);
CREATE INDEX IF NOT EXISTS items_hash ON items(content_hash);
CREATE INDEX IF NOT EXISTS items_published ON items(published);

CREATE TABLE IF NOT EXISTS answers (
    key TEXT PRIMARY KEY,
    p REAL NOT NULL,
    created REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS judgments (
    ts REAL NOT NULL,
    run_id TEXT,
    item_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    p REAL NOT NULL,
    threshold REAL NOT NULL,
    verdict INTEGER NOT NULL,
    cached INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    backend TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS judgments_item ON judgments(item_id, question);

CREATE TABLE IF NOT EXISTS embeddings (
    item_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (item_id, model)
);

CREATE TABLE IF NOT EXISTS feed_state (
    url TEXT PRIMARY KEY,
    etag TEXT,
    modified TEXT,
    last_ok REAL,
    last_error TEXT,
    last_checked REAL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    started REAL NOT NULL,
    finished REAL,
    status TEXT NOT NULL,
    detail TEXT,
    episode TEXT
);
"""


@dataclass
class Item:
    id: int
    outlet: str
    feed_url: str
    url: str
    canonical_url: str
    title: str
    lead: str
    text: str
    content_hash: str
    published: float
    fetched: float
    full_text: bool
    source: str


ITEM_COLUMNS = (
    "id, outlet, feed_url, url, canonical_url, title, lead, text, content_hash, "
    "published, fetched, full_text, source"
)


def _item(row: tuple) -> Item:
    item = Item(*row)
    item.full_text = bool(item.full_text)
    return item


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # items

    def has_url(self, canonical_url: str) -> bool:
        row = self.db.execute("SELECT 1 FROM items WHERE canonical_url = ?", (canonical_url,))
        return row.fetchone() is not None

    def has_hash(self, content_hash: str) -> bool:
        row = self.db.execute("SELECT 1 FROM items WHERE content_hash = ?", (content_hash,))
        return row.fetchone() is not None

    def add_item(self, **fields) -> int | None:
        """Insert unless the canonical URL or content hash is already in the pool."""
        if self.has_url(fields["canonical_url"]) or self.has_hash(fields["content_hash"]):
            return None
        fields.setdefault("fetched", time.time())
        fields.setdefault("source", "feed")
        fields["full_text"] = int(bool(fields.get("full_text")))
        cols = ", ".join(fields)
        marks = ", ".join(f":{k}" for k in fields)
        cur = self.db.execute(f"INSERT INTO items ({cols}) VALUES ({marks})", fields)
        self.db.commit()
        return cur.lastrowid

    def items_since(self, since: float, *, source: str | None = "feed") -> list[Item]:
        sql = f"SELECT {ITEM_COLUMNS} FROM items WHERE published >= ?"
        args: list = [since]
        if source:
            sql += " AND source = ?"
            args.append(source)
        rows = self.db.execute(sql + " ORDER BY published DESC", args).fetchall()
        return [_item(r) for r in rows]

    def items_by_id(self, ids: list[int]) -> list[Item]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT {ITEM_COLUMNS} FROM items WHERE id IN ({marks})", ids)
        by_id = {r[0]: _item(r) for r in rows.fetchall()}
        return [by_id[i] for i in ids if i in by_id]

    def prune(self, older_than: float) -> int:
        old = "SELECT id FROM items WHERE published < ?"
        self.db.execute(f"DELETE FROM embeddings WHERE item_id IN ({old})", (older_than,))
        cur = self.db.execute("DELETE FROM items WHERE published < ?", (older_than,))
        self.db.commit()
        return cur.rowcount

    def count_items(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    # Laya answer cache and judgment log

    def cached_answers(self, keys: list[str]) -> dict[str, float]:
        found: dict[str, float] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(f"SELECT key, p FROM answers WHERE key IN ({marks})", chunk)
            found.update(rows.fetchall())
        return found

    def cache_answers(self, values: dict[str, float]) -> None:
        now = time.time()
        self.db.executemany(
            "INSERT OR REPLACE INTO answers (key, p, created) VALUES (?, ?, ?)",
            [(k, p, now) for k, p in values.items()],
        )
        self.db.commit()

    def log_judgments(self, rows: list[dict]) -> None:
        now = time.time()
        self.db.executemany(
            "INSERT INTO judgments (ts, run_id, item_id, question, p, threshold, verdict, cached, "
            "truncated, backend) VALUES (:ts, :run_id, :item_id, :question, :p, :threshold, "
            ":verdict, :cached, :truncated, :backend)",
            [{"ts": now, **r} for r in rows],
        )
        self.db.commit()

    # embeddings

    def embeddings(self, model: str, ids: list[int]) -> dict[int, tuple[str, np.ndarray]]:
        found: dict[int, tuple[str, np.ndarray]] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT item_id, text_hash, vec FROM embeddings WHERE model = ? AND item_id IN ({marks})",
                [model, *chunk],
            )
            for item_id, text_hash, blob in rows.fetchall():
                found[item_id] = (text_hash, np.frombuffer(blob, dtype=np.float32))
        return found

    def save_embeddings(self, model: str, rows: list[tuple[int, str, np.ndarray]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO embeddings (item_id, model, text_hash, vec) VALUES (?, ?, ?, ?)",
            [(i, model, h, np.asarray(v, dtype=np.float32).tobytes()) for i, h, v in rows],
        )
        self.db.commit()

    # feed state (conditional GET)

    def feed_state(self, url: str) -> dict:
        row = self.db.execute("SELECT etag, modified FROM feed_state WHERE url = ?", (url,)).fetchone()
        return {"etag": row[0], "modified": row[1]} if row else {}

    def save_feed_state(self, url: str, *, etag=None, modified=None, error=None) -> None:
        now = time.time()
        self.db.execute(
            "INSERT INTO feed_state (url, etag, modified, last_ok, last_error, last_checked) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(url) DO UPDATE SET "
            "etag = COALESCE(excluded.etag, etag), modified = COALESCE(excluded.modified, modified), "
            "last_ok = COALESCE(excluded.last_ok, last_ok), last_error = excluded.last_error, "
            "last_checked = excluded.last_checked",
            (url, etag, modified, None if error else now, error, now),
        )
        self.db.commit()

    # runs

    def start_run(self, run_id: str, kind: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO runs (id, kind, started, status) VALUES (?, ?, ?, 'running')",
            (run_id, kind, time.time()),
        )
        self.db.commit()

    def finish_run(self, run_id: str, status: str, detail: str = "", episode: str = "") -> None:
        self.db.execute(
            "UPDATE runs SET finished = ?, status = ?, detail = ?, episode = ? WHERE id = ?",
            (time.time(), status, detail, episode, run_id),
        )
        self.db.commit()

    def last_failure(self, kind: str, detail: str) -> float | None:
        row = self.db.execute(
            "SELECT MAX(finished) FROM runs WHERE kind = ? AND status = 'failed' AND detail = ? "
            "AND episode != ''",
            (kind, detail),
        ).fetchone()
        return row[0]
