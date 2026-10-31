"""SQLite database: one file, WAL mode, a connection per thread."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, subject TEXT, predicate TEXT, value TEXT,
  kind TEXT NOT NULL CHECK(kind IN ('verified','unverified','preference','generated')),
  source TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0.5,
  status TEXT NOT NULL DEFAULT 'active', fingerprint TEXT NOT NULL,
  conflicts_with TEXT, created REAL, updated REAL);
CREATE UNIQUE INDEX IF NOT EXISTS facts_fp ON facts(fingerprint);
CREATE INDEX IF NOT EXISTS facts_subj ON facts(subject, predicate);

CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY, title TEXT, body TEXT NOT NULL, source TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'unverified', fingerprint TEXT UNIQUE, created REAL);

CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, title TEXT, created REAL);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, model_version TEXT, meta TEXT, created REAL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY, message_id INTEGER, rating INTEGER NOT NULL, correction TEXT,
  prompt TEXT, response TEXT, model_version TEXT, created REAL, used INTEGER DEFAULT 0);

CREATE TABLE IF NOT EXISTS examples (
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL,
  origin TEXT NOT NULL CHECK(origin IN ('user_correction','user_added','local_file','generated','web')),
  fingerprint TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','accepted','rejected','eval')),
  goal_id INTEGER, dataset_id TEXT, reason TEXT, created REAL);

CREATE TABLE IF NOT EXISTS goals (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, description TEXT, kind TEXT,
  keywords TEXT, metric TEXT, baseline REAL, target REAL,
  status TEXT NOT NULL DEFAULT 'proposed', created REAL, updated REAL);

CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, params TEXT,
  progress TEXT, result TEXT, error TEXT, ckpt_path TEXT, created REAL, updated REAL);

CREATE TABLE IF NOT EXISTS model_versions (
  id TEXT PRIMARY KEY, parent TEXT, path TEXT NOT NULL, status TEXT NOT NULL,
  dataset_id TEXT, metrics TEXT, report TEXT, note TEXT, created REAL, updated REAL);

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, parent TEXT, path TEXT NOT NULL, sha256 TEXT NOT NULL,
  n_tokens INTEGER, n_examples INTEGER, sources TEXT, created REAL);

CREATE TABLE IF NOT EXISTS eval_sets (
  name TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('heldout','regression')),
  fmt TEXT NOT NULL CHECK(fmt IN ('text','qa')), path TEXT NOT NULL, sha256 TEXT NOT NULL,
  n_items INTEGER, description TEXT, created REAL);

CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, ts REAL, actor TEXT, action TEXT, detail TEXT);

CREATE TABLE IF NOT EXISTS attachments (
  id INTEGER PRIMARY KEY, conversation_id INTEGER, name TEXT NOT NULL, kind TEXT, language TEXT,
  path TEXT NOT NULL, chars INTEGER, created REAL);

CREATE TABLE IF NOT EXISTS web_pages (
  url TEXT PRIMARY KEY, title TEXT, fetched REAL, paragraphs INTEGER, accepted INTEGER, error TEXT);

CREATE TABLE IF NOT EXISTS image_versions (
  id TEXT PRIMARY KEY, path TEXT NOT NULL, status TEXT NOT NULL, metrics TEXT,
  config TEXT, created REAL);
"""

FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(title, body, content='notes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, title, body) VALUES (new.id, new.title, new.body); END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, body) VALUES('delete', old.id, old.title, old.body); END;
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, content='facts', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, text) VALUES('delete', old.id, old.text); END;
"""


class Database:
    def __init__(self, path):
        self.path = str(path)
        self._local = threading.local()
        self.write_lock = threading.RLock()
        con = self.conn()
        con.executescript(SCHEMA)
        self._migrate(con)
        try:
            con.executescript(FTS)
            self.has_fts = True
        except sqlite3.OperationalError:
            self.has_fts = False  # fall back to LIKE search
        con.commit()

    def _migrate(self, con):
        """Older databases: allow the 'web' origin for examples (SQLite cannot alter a CHECK)."""
        row = con.execute("SELECT sql FROM sqlite_master WHERE name='examples'").fetchone()
        if row and "'web'" not in row[0]:
            con.executescript("""
            BEGIN;
            ALTER TABLE examples RENAME TO examples_old;
            CREATE TABLE examples (
              id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL,
              origin TEXT NOT NULL CHECK(origin IN ('user_correction','user_added','local_file','generated','web')),
              fingerprint TEXT NOT NULL UNIQUE,
              status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','accepted','rejected','eval')),
              goal_id INTEGER, dataset_id TEXT, reason TEXT, created REAL);
            INSERT INTO examples SELECT * FROM examples_old;
            DROP TABLE examples_old;
            COMMIT;""")

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.c = c
        return c

    def execute(self, sql, params=()):
        with self.write_lock:
            cur = self.conn().execute(sql, params)
            self.conn().commit()
            return cur

    def query(self, sql, params=()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, params).fetchall()]

    def one(self, sql, params=()) -> dict | None:
        r = self.conn().execute(sql, params).fetchone()
        return dict(r) if r else None

    def audit(self, actor: str, action: str, **detail):
        self.execute("INSERT INTO audit(ts, actor, action, detail) VALUES (?,?,?,?)",
                     (time.time(), actor, action, json.dumps(detail, default=str)))

    def close(self):
        c = getattr(self._local, "c", None)
        if c is not None:
            c.close()
            self._local.c = None
