"""Knowledge base for Nexo 1.2 (lessons, examples, video transcripts).

What "learning" means here, honestly:
  Nexo 1.2 uses ready-made models (built-in engine) that cannot be retrained on a
  laptop. Instead, lessons you give it are stored here, split into chunks, and
  the most relevant chunks are placed into the model's prompt when you ask
  something. The model's weights do not change; it *reads* its notes every
  time. Good results you mark are stored as worked examples the same way.

Search: SQLite FTS5 with simple prefix stemming, so Polish word forms
("pętla", "pętli", "pętlę") still match.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from .config import PROJECT_ROOT
from .lm.data import fingerprint
from .memory.db import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_docs (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, kind TEXT NOT NULL, source TEXT,
  topic TEXT, chars INTEGER, fingerprint TEXT UNIQUE, created REAL);
CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, text TEXT NOT NULL,
  FOREIGN KEY(doc_id) REFERENCES knowledge_docs(id) ON DELETE CASCADE);
"""
FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(text, content='knowledge_chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS kc_ai AFTER INSERT ON knowledge_chunks BEGIN
  INSERT INTO knowledge_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS kc_ad AFTER DELETE ON knowledge_chunks BEGIN
  INSERT INTO knowledge_fts(knowledge_fts, rowid, text) VALUES('delete', old.id, old.text); END;
"""
KINDS = ("lesson", "example", "video", "file", "web")
PACK_DIR = PROJECT_ROOT / "data" / "knowledge"
STOP = set("""i a w z na do że to jest się nie jak co czy o od po za dla oraz lub ale the and of to in is for with
on it be as by or an this that how what please proszę mi mnie zrób zrob napisz stwórz stworz dodaj""".split())


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Split at headings / blank lines; keep code blocks whole when possible."""
    text = text.replace("\r\n", "\n").strip()
    blocks, cur, in_code = [], [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and (line.startswith("#") and cur):
            blocks.append("\n".join(cur)); cur = []
        cur.append(line)
        if not in_code and not line.strip() and sum(len(x) for x in cur) > max_chars * 0.6:
            blocks.append("\n".join(cur)); cur = []
    if cur:
        blocks.append("\n".join(cur))
    out, buf = [], ""
    for b in (x.strip() for x in blocks if x.strip()):
        if len(buf) + len(b) + 2 <= max_chars:
            buf = f"{buf}\n\n{b}" if buf else b
            continue
        if buf:
            out.append(buf)
        while len(b) > max_chars * 2:  # very long block: hard split on lines
            cut = b.rfind("\n", 0, max_chars * 2)
            cut = cut if cut > 0 else max_chars * 2
            out.append(b[:cut]); b = b[cut:].lstrip("\n")
        buf = b
    if buf:
        out.append(buf)
    return out


def _terms(query: str) -> list[str]:
    words = re.findall(r"[\wąćęłńóśźż]+", query.lower())
    terms = []
    for w in words:
        if w in STOP or len(w) < 2:
            continue
        w = w[:6] if len(w) > 6 else w  # crude stemming: prefix match covers word endings
        if w not in terms:
            terms.append(w)
    return terms[:14]


class KnowledgeBase:
    def __init__(self, db: Database):
        self.db = db
        con = db.conn()
        con.executescript(SCHEMA)
        try:
            con.executescript(FTS)
            self.fts = True
        except Exception:  # noqa: BLE001 - FTS5 missing: fall back to LIKE
            self.fts = False
        con.commit()

    def add(self, title: str, text: str, kind: str = "lesson", source: str = "user", topic: str = "") -> dict:
        text = (text or "").strip()
        if len(text) < 20:
            raise ValueError("the text is too short to be useful (at least 20 characters)")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        fp = fingerprint(text)
        dup = self.db.one("SELECT * FROM knowledge_docs WHERE fingerprint=?", (fp,))
        if dup:
            return {"doc": dup, "duplicate": True, "chunks": 0}
        title = (title or text.split("\n", 1)[0])[:120].strip("# ").strip() or "Bez tytułu"
        with self.db.write_lock:
            did = self.db.execute("INSERT INTO knowledge_docs(title,kind,source,topic,chars,fingerprint,created) "
                                  "VALUES (?,?,?,?,?,?,?)",
                                  (title, kind, source, topic, len(text), fp, time.time())).lastrowid
            chunks = chunk_text(text)
            for c in chunks:
                self.db.execute("INSERT INTO knowledge_chunks(doc_id, text) VALUES (?,?)", (did, f"[{title}]\n{c}"))
        self.db.audit("user", "knowledge_added", id=did, kind=kind, chunks=len(chunks))
        return {"doc": self.get(did), "duplicate": False, "chunks": len(chunks)}

    def get(self, did: int) -> dict | None:
        return self.db.one("SELECT * FROM knowledge_docs WHERE id=?", (did,))

    def text_of(self, did: int) -> str:
        rows = self.db.query("SELECT text FROM knowledge_chunks WHERE doc_id=? ORDER BY id", (did,))
        return "\n\n".join(r["text"].split("\n", 1)[1] if "\n" in r["text"] else r["text"] for r in rows)

    def list(self) -> list[dict]:
        return self.db.query("SELECT d.*, (SELECT COUNT(*) FROM knowledge_chunks c WHERE c.doc_id=d.id) AS chunks "
                             "FROM knowledge_docs d ORDER BY created DESC")

    def delete(self, did: int):
        self.db.execute("DELETE FROM knowledge_chunks WHERE doc_id=?", (did,))
        self.db.execute("DELETE FROM knowledge_docs WHERE id=?", (did,))
        self.db.audit("user", "knowledge_deleted", id=did)

    def search(self, query: str, limit: int = 6, max_chars: int = 3500, topic: str | None = None) -> list[dict]:
        terms = _terms(query)
        if not terms:
            return []
        if self.fts:
            q = " OR ".join(f'"{t}"*' for t in terms)
            rows = self.db.query(
                "SELECT c.id, c.text, d.title, d.kind, d.topic, bm25(knowledge_fts) AS score FROM knowledge_fts "
                "JOIN knowledge_chunks c ON c.id = knowledge_fts.rowid JOIN knowledge_docs d ON d.id = c.doc_id "
                "WHERE knowledge_fts MATCH ? ORDER BY score LIMIT ?", (q, limit * 3))
        else:
            rows = []
            for t in terms[:5]:
                rows += self.db.query("SELECT c.id, c.text, d.title, d.kind, d.topic, 0 AS score FROM knowledge_chunks c "
                                      "JOIN knowledge_docs d ON d.id=c.doc_id WHERE c.text LIKE ? LIMIT ?",
                                      (f"%{t}%", limit))
        if topic:  # prefer chunks from the matching topic (e.g. 'roblox' or 'blender')
            rows.sort(key=lambda r: (r["topic"] not in (topic, "", None), r["score"]))
        out, used, seen = [], 0, set()
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            if used + len(r["text"]) > max_chars and out:
                break
            out.append(r)
            used += len(r["text"])
            if len(out) >= limit:
                break
        return out

    # ----------------------------------------------------------- packs
    @staticmethod
    def packs() -> list[dict]:
        out = []
        for p in sorted(PACK_DIR.glob("*.md")):
            first = p.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
            out.append({"id": p.stem, "title": first, "chars": p.stat().st_size})
        return out

    def import_pack(self, pack_id: str) -> dict:
        if not re.fullmatch(r"[\w-]+", pack_id or ""):
            raise ValueError("bad pack name")
        p = PACK_DIR / f"{pack_id}.md"
        if not p.exists():
            raise KeyError(pack_id)
        text = p.read_text(encoding="utf-8")
        topic = "roblox" if "roblox" in pack_id else "blender" if "blender" in pack_id else ""
        return self.add(text.split("\n", 1)[0].lstrip("# ").strip(), text, "lesson", f"pack:{pack_id}", topic)
