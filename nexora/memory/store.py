"""Long-term memory: facts, notes, conversations, feedback.

IMPORTANT DISTINCTION (shown in the UI as well):
  * A fact stored here is *remembered* - it can be looked up and shown.
  * It is NOT learned by the neural network. Only training changes weights,
    and training only happens through a candidate model that you approve.
"""
from __future__ import annotations

import json
import re
import time

from ..lm.data import fingerprint
from .db import Database

KINDS = ("verified", "unverified", "preference", "generated")

_FACT_PATTERNS = [
    re.compile(r"^(?:my\s+)?(?P<s>.+?)\s+(?P<p>is not|isn't|is|are not|aren't|are|was|were|equals|=)\s+(?P<v>.+)$", re.I),
    re.compile(r"^i\s+(?P<p>like|love|prefer|hate|dislike|live in|work at|work as)\s+(?P<v>.+)$", re.I),
]
_NEGATIONS = {"is not": "is", "isn't": "is", "are not": "are", "aren't": "are"}
_SAME_SLOT = {"is": "is", "are": "is", "was": "is", "were": "is", "equals": "is", "=": "is",
              "live in": "live in", "work at": "work at", "work as": "work as"}


def parse_fact(text: str) -> tuple[str | None, str | None, str | None, bool]:
    """Very small rule-based parser: 'X is Y' -> (x, 'is', y, negated).
    Only used for duplicate / contradiction bookkeeping, never to answer."""
    t = text.strip().rstrip(".!")
    for i, pat in enumerate(_FACT_PATTERNS):
        m = pat.match(t)
        if m:
            s = "user" if i == 1 else m.group("s").lower().strip()
            s = re.sub(r"^(the|a|an)\s+", "", s)
            p = m.group("p").lower()
            neg = p in _NEGATIONS
            p = _NEGATIONS.get(p, p)
            return s, _SAME_SLOT.get(p, p), m.group("v").lower().strip(), neg
    return None, None, None, False


def _fts_query(q: str) -> str:
    words = re.findall(r"\w+", q.lower())
    return " OR ".join(f'"{w}"' for w in words[:12])


class MemoryStore:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ facts
    def add_fact(self, text: str, kind: str = "unverified", source: str = "user",
                 confidence: float | None = None) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("empty fact")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if confidence is None:
            confidence = {"verified": 0.95, "preference": 0.9, "unverified": 0.5,
                          "generated": 0.2}[kind]
        fp = fingerprint(text)
        dup = self.db.one("SELECT * FROM facts WHERE fingerprint=?", (fp,))
        if dup:
            return {"fact": dup, "duplicate": True, "conflicts": []}
        s, p, v, neg = parse_fact(text)
        conflicts = []
        if s and p:
            for row in self.db.query(
                    "SELECT * FROM facts WHERE subject=? AND predicate=? AND status!='deleted'", (s, p)):
                other_neg = row["text"] and parse_fact(row["text"])[3]
                same_val = row["value"] == v
                # contradiction: same slot, different value (for single-valued slots)
                # or same value with opposite polarity
                if (same_val and other_neg != neg) or (not same_val and not neg and not other_neg
                                                       and p in ("is", "live in", "work at", "work as")):
                    conflicts.append(row)
        status = "conflict" if conflicts else "active"
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO facts(text,subject,predicate,value,kind,source,confidence,status,"
            "fingerprint,conflicts_with,created,updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (text, s, p, v, kind, source, confidence, status, fp,
             json.dumps([c["id"] for c in conflicts]) if conflicts else None, now, now))
        fid = cur.lastrowid
        for c in conflicts:
            self.db.execute("UPDATE facts SET status='conflict', updated=? WHERE id=?", (now, c["id"]))
        self.db.audit("user" if source == "user" else "system", "fact_added", id=fid, kind=kind,
                      conflicts=[c["id"] for c in conflicts])
        return {"fact": self.get_fact(fid), "duplicate": False, "conflicts": conflicts}

    def get_fact(self, fid: int) -> dict | None:
        return self.db.one("SELECT * FROM facts WHERE id=?", (fid,))

    def list_facts(self, kind: str | None = None, status: str | None = None, limit=200) -> list[dict]:
        sql, args = "SELECT * FROM facts WHERE status!='deleted'", []
        if kind:
            sql += " AND kind=?"; args.append(kind)
        if status:
            sql += " AND status=?"; args.append(status)
        return self.db.query(sql + " ORDER BY updated DESC LIMIT ?", (*args, limit))

    def update_fact(self, fid: int, kind: str | None = None, status: str | None = None,
                    confidence: float | None = None) -> dict:
        f = self.get_fact(fid)
        if not f:
            raise KeyError(fid)
        if kind and kind not in KINDS:
            raise ValueError("bad kind")
        self.db.execute("UPDATE facts SET kind=?, status=?, confidence=?, updated=? WHERE id=?",
                        (kind or f["kind"], status or f["status"],
                         f["confidence"] if confidence is None else confidence, time.time(), fid))
        self.db.audit("user", "fact_updated", id=fid, kind=kind, status=status)
        return self.get_fact(fid)

    def delete_fact(self, fid: int):
        self.db.execute("DELETE FROM facts WHERE id=?", (fid,))
        self.db.audit("user", "fact_deleted", id=fid)

    # ------------------------------------------------------------------ notes
    def add_note(self, body: str, title: str = "", source: str = "user", kind: str = "unverified"):
        fp = fingerprint(title + "\n" + body)
        dup = self.db.one("SELECT * FROM notes WHERE fingerprint=?", (fp,))
        if dup:
            return {"note": dup, "duplicate": True}
        cur = self.db.execute("INSERT INTO notes(title,body,source,kind,fingerprint,created) "
                              "VALUES (?,?,?,?,?,?)", (title, body, source, kind, fp, time.time()))
        return {"note": self.db.one("SELECT * FROM notes WHERE id=?", (cur.lastrowid,)),
                "duplicate": False}

    def delete_note(self, nid: int):
        self.db.execute("DELETE FROM notes WHERE id=?", (nid,))
        self.db.audit("user", "note_deleted", id=nid)

    def list_notes(self, limit=200):
        return self.db.query("SELECT * FROM notes ORDER BY created DESC LIMIT ?", (limit,))

    # ----------------------------------------------------------------- search
    def search(self, q: str, limit: int = 10) -> dict:
        q = q.strip()
        if not q:
            return {"facts": [], "notes": []}
        if self.db.has_fts and _fts_query(q):
            fq = _fts_query(q)
            facts = self.db.query(
                "SELECT f.* FROM facts_fts JOIN facts f ON f.id=facts_fts.rowid "
                "WHERE facts_fts MATCH ? AND f.status!='deleted' ORDER BY bm25(facts_fts) LIMIT ?",
                (fq, limit))
            notes = self.db.query(
                "SELECT n.* FROM notes_fts JOIN notes n ON n.id=notes_fts.rowid "
                "WHERE notes_fts MATCH ? ORDER BY bm25(notes_fts) LIMIT ?", (fq, limit))
        else:
            like = f"%{q}%"
            facts = self.db.query("SELECT * FROM facts WHERE text LIKE ? AND status!='deleted' "
                                  "LIMIT ?", (like, limit))
            notes = self.db.query("SELECT * FROM notes WHERE body LIKE ? OR title LIKE ? LIMIT ?",
                                  (like, like, limit))
        return {"facts": facts, "notes": notes}

    # ---------------------------------------------------------- conversations
    def new_conversation(self, title: str = "New chat") -> int:
        return self.db.execute("INSERT INTO conversations(title, created) VALUES (?,?)",
                               (title, time.time())).lastrowid

    def add_message(self, cid: int, role: str, content: str, model_version: str | None = None,
                    meta: dict | None = None) -> int:
        return self.db.execute(
            "INSERT INTO messages(conversation_id,role,content,model_version,meta,created) "
            "VALUES (?,?,?,?,?,?)",
            (cid, role, content, model_version, json.dumps(meta or {}), time.time())).lastrowid

    def messages(self, cid: int, limit: int = 200) -> list[dict]:
        return self.db.query("SELECT * FROM messages WHERE conversation_id=? ORDER BY id LIMIT ?",
                             (cid, limit))

    def recent_exchanges(self, cid: int, n: int = 3) -> list[tuple[str, str]]:
        """Short-term context: last n (user, assistant) pairs."""
        msgs = self.db.query("SELECT role, content FROM messages WHERE conversation_id=? "
                             "ORDER BY id DESC LIMIT ?", (cid, n * 2 + 2))[::-1]
        pairs, pending = [], None
        for m in msgs:
            if m["role"] == "user":
                pending = m["content"]
            elif m["role"] == "assistant" and pending is not None:
                pairs.append((pending, m["content"]))
                pending = None
        return pairs[-n:]

    def conversations(self, limit=50):
        return self.db.query(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS n "
            "FROM conversations c ORDER BY id DESC LIMIT ?", (limit,))

    def delete_conversation(self, cid: int):
        self.db.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        self.db.execute("DELETE FROM conversations WHERE id=?", (cid,))
        self.db.audit("user", "conversation_deleted", id=cid)

    # --------------------------------------------------------------- feedback
    def add_feedback(self, message_id: int, rating: int, correction: str | None = None) -> int:
        msg = self.db.one("SELECT * FROM messages WHERE id=?", (message_id,))
        if not msg or msg["role"] != "assistant":
            raise ValueError("feedback must target an assistant message")
        prev = self.db.one("SELECT content FROM messages WHERE conversation_id=? AND id<? AND "
                           "role='user' ORDER BY id DESC LIMIT 1",
                           (msg["conversation_id"], message_id))
        fid = self.db.execute(
            "INSERT INTO feedback(message_id,rating,correction,prompt,response,model_version,created)"
            " VALUES (?,?,?,?,?,?,?)",
            (message_id, 1 if rating > 0 else -1, (correction or "").strip() or None,
             prev["content"] if prev else "", msg["content"], msg["model_version"],
             time.time())).lastrowid
        self.db.audit("user", "feedback", id=fid, rating=rating, has_correction=bool(correction))
        return fid

    def list_feedback(self, limit=200):
        return self.db.query("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,))

    def wipe_all(self):
        """User-requested deletion of all memory (facts, notes, chats, feedback)."""
        for t in ("facts", "notes", "messages", "conversations", "feedback"):
            self.db.execute(f"DELETE FROM {t}")
        self.db.audit("user", "memory_wiped")
