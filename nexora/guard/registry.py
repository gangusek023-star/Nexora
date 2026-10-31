"""Versioned storage for models, datasets and immutable evaluation sets."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from ..config import Paths
from ..lm.data import shingles, tokens_to_array
from ..lm.model import NexoraLM, load_model, save_model
from ..memory.db import Database


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TamperError(RuntimeError):
    pass


def rel(paths: Paths, p) -> str:
    """Store paths relative to the NEXORA home so the folder can be moved/copied."""
    try:
        return Path(p).resolve().relative_to(paths.root.resolve()).as_posix()
    except ValueError:
        return str(p)


def absol(paths: Paths, p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else paths.root / q)


def _next_id(db: Database, table: str, prefix: str) -> str:
    rows = db.query(f"SELECT id FROM {table} WHERE id LIKE ?", (prefix + "%",))
    n = max([int(r["id"][len(prefix):]) for r in rows if r["id"][len(prefix):].isdigit()], default=0)
    return f"{prefix}{n + 1:04d}"


# ===================================================================== models
class ModelRegistry:
    """Every model version is an immutable file. Exactly one is 'active'.

    statuses: active | approved (previously active, kept for rollback) |
              training | pending_approval | rejected | failed"""

    def __init__(self, db: Database, paths: Paths):
        self.db, self.paths = db, paths

    def register(self, model: NexoraLM, *, parent: str | None, status: str,
                 dataset_id: str | None, metrics: dict | None = None, note: str = "") -> str:
        with self.db.write_lock:
            vid = _next_id(self.db, "model_versions", "v")
            path = self.paths.models / f"{vid}.pt"
            save_model(model, path, extra={"version": vid, "parent": parent})
            now = time.time()
            self.db.execute(
                "INSERT INTO model_versions(id,parent,path,status,dataset_id,metrics,note,created,updated)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (vid, parent, rel(self.paths, path), status, dataset_id, json.dumps(metrics or {}), note, now, now))
        self.db.audit("system", "model_registered", id=vid, parent=parent, status=status)
        return vid

    def get(self, vid: str) -> dict | None:
        r = self.db.one("SELECT * FROM model_versions WHERE id=?", (vid,))
        if r:
            r["path"] = absol(self.paths, r["path"])
            r["metrics"] = json.loads(r["metrics"] or "{}")
            r["report"] = json.loads(r["report"]) if r["report"] else None
        return r

    def list(self) -> list[dict]:
        rows = self.db.query("SELECT * FROM model_versions ORDER BY created DESC")
        for r in rows:
            r["path"] = absol(self.paths, r["path"])
            r["metrics"] = json.loads(r["metrics"] or "{}")
            r["report"] = json.loads(r["report"]) if r["report"] else None
            r["size_mb"] = round(os.path.getsize(r["path"]) / 1e6, 2) if os.path.exists(r["path"]) else None
        return rows

    def active(self) -> dict | None:
        r = self.db.one("SELECT id FROM model_versions WHERE status='active'")
        return self.get(r["id"]) if r else None

    def load(self, vid: str) -> NexoraLM:
        r = self.get(vid)
        if not r:
            raise KeyError(f"unknown model version {vid}")
        return load_model(r["path"])

    def set_status(self, vid: str, status: str, **fields):
        sets, args = ["status=?", "updated=?"], [status, time.time()]
        for k in ("metrics", "report"):
            if k in fields:
                sets.append(f"{k}=?"); args.append(json.dumps(fields[k], default=str))
        if "note" in fields:
            sets.append("note=?"); args.append(fields["note"])
        self.db.execute(f"UPDATE model_versions SET {', '.join(sets)} WHERE id=?", (*args, vid))

    def activate(self, vid: str, actor: str = "user", reason: str = "approved"):
        """Make `vid` the active model. The old active one stays as 'approved'
        so it can be restored later (rollback)."""
        r = self.get(vid)
        if not r:
            raise KeyError(vid)
        if r["status"] in ("rejected", "failed", "training"):
            raise ValueError(f"cannot activate a model with status '{r['status']}'")
        if not os.path.exists(r["path"]):
            raise FileNotFoundError(r["path"])
        with self.db.write_lock:
            prev = self.active()
            if prev and prev["id"] != vid:
                self.set_status(prev["id"], "approved")
            self.set_status(vid, "active")
        self.db.audit(actor, "model_activated", id=vid, previous=prev["id"] if prev else None,
                      reason=reason)
        return prev["id"] if prev else None

    def backup_active(self) -> str | None:
        """Copy the active weights to the backup folder before any training."""
        a = self.active()
        if not a:
            return None
        dst = self.paths.backups / f"{a['id']}-{time.strftime('%Y%m%d-%H%M%S')}.pt"
        shutil.copy2(a["path"], dst)
        backups = sorted(self.paths.backups.glob("*.pt"))
        for old in backups[:-5]:  # keep the 5 most recent backup copies
            old.unlink(missing_ok=True)
        self.db.audit("system", "backup_created", id=a["id"], path=str(dst))
        return str(dst)

    def delete_version(self, vid: str):
        r = self.get(vid)
        if not r:
            raise KeyError(vid)
        if r["status"] == "active":
            raise ValueError("cannot delete the active model")
        Path(r["path"]).unlink(missing_ok=True)
        self.db.execute("DELETE FROM model_versions WHERE id=?", (vid,))
        self.db.audit("user", "model_deleted", id=vid)


# =================================================================== datasets
class DatasetStore:
    """Immutable, content-hashed training datasets. A new learning cycle never
    edits a dataset; it creates a child version."""

    def __init__(self, db: Database, paths: Paths):
        self.db, self.paths = db, paths

    def create(self, docs: list[dict], tokenizer, parent: str | None = None, note: str = "") -> str:
        """docs: [{"text":..., "source":..., "origin":...}]"""
        if not docs:
            raise ValueError("dataset needs at least one document")
        ids: list[int] = []
        for d in docs:
            ids.extend(tokenizer.encode(d["text"]))
            ids.append(tokenizer.eot_id)
        with self.db.write_lock:
            did = _next_id(self.db, "datasets", "ds")
            folder = self.paths.datasets / did
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "tokens.npy", tokens_to_array(ids))
            with open(folder / "documents.jsonl", "w", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            sources = sorted({d.get("source", "?") for d in docs})
            manifest = {"id": did, "parent": parent, "n_docs": len(docs), "n_tokens": len(ids),
                        "sources": sources, "note": note, "created": time.time()}
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            digest = sha256_file(folder / "tokens.npy")
            self.db.execute("INSERT INTO datasets(id,parent,path,sha256,n_tokens,n_examples,sources,created)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            (did, parent, rel(self.paths, folder), digest, len(ids), len(docs),
                             json.dumps(sources), time.time()))
        self.db.audit("system", "dataset_created", id=did, parent=parent, n_docs=len(docs),
                      n_tokens=len(ids))
        return did

    def get(self, did: str) -> dict | None:
        r = self.db.one("SELECT * FROM datasets WHERE id=?", (did,))
        if r:
            r["path"] = absol(self.paths, r["path"])
            r["sources"] = json.loads(r["sources"] or "[]")
        return r

    def list(self) -> list[dict]:
        rows = self.db.query("SELECT * FROM datasets ORDER BY created DESC")
        for r in rows:
            r["path"] = absol(self.paths, r["path"])
            r["sources"] = json.loads(r["sources"] or "[]")
        return rows

    def tokens(self, did: str) -> np.ndarray:
        r = self.get(did)
        if not r:
            raise KeyError(did)
        f = Path(r["path"]) / "tokens.npy"
        if sha256_file(f) != r["sha256"]:
            raise TamperError(f"dataset {did} was modified on disk (hash mismatch)")
        return np.load(f)

    def documents(self, did: str, limit: int | None = None) -> list[dict]:
        r = self.get(did)
        out = []
        with open(Path(r["path"]) / "documents.jsonl", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
        return out

    def lineage(self, did: str) -> list[str]:
        chain, cur = [], did
        while cur:
            chain.append(cur)
            r = self.get(cur)
            cur = r["parent"] if r else None
        return chain


# ================================================================ eval sets
class EvalSuite:
    """Immutable held-out evaluation sets and the regression suite.

    * 'heldout'    - general held-out data, never used for training.
    * 'regression' - skills a previously approved model learned; any
                     candidate must not get meaningfully worse on them.
    Files are hashed at creation; a modified file is refused (TamperError)."""

    def __init__(self, db: Database, paths: Paths):
        self.db, self.paths = db, paths

    def create(self, name: str, kind: str, fmt: str, data, description: str = "") -> dict:
        if self.db.one("SELECT name FROM eval_sets WHERE name=?", (name,)):
            raise ValueError(f"eval set '{name}' already exists and is immutable")
        if fmt == "text":
            if not isinstance(data, str) or len(data) < 200:
                raise ValueError("text eval set needs at least 200 characters")
            n = len(data)
        elif fmt == "qa":
            if not data or not all("prompt" in d and "answer" in d for d in data):
                raise ValueError("qa eval set needs [{prompt, answer}] items")
            n = len(data)
        else:
            raise ValueError("fmt must be text or qa")
        path = self.paths.eval_sets / f"{name}.json"
        path.write_text(json.dumps({"name": name, "fmt": fmt, "data": data}, ensure_ascii=False),
                        encoding="utf-8")
        try:
            os.chmod(path, 0o444)  # read-only on disk as an extra guard
        except OSError:
            pass
        self.db.execute("INSERT INTO eval_sets(name,kind,fmt,path,sha256,n_items,description,created)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (name, kind, fmt, rel(self.paths, path), sha256_file(path), n, description, time.time()))
        self.db.audit("system", "eval_set_created", name=name, kind=kind, fmt=fmt, n=n)
        return self.get(name)

    def get(self, name):
        r = self.db.one("SELECT * FROM eval_sets WHERE name=?", (name,))
        if r:
            r["path"] = absol(self.paths, r["path"])
        return r

    def list(self, kind: str | None = None):
        if kind:
            return self.db.query("SELECT * FROM eval_sets WHERE kind=? ORDER BY created", (kind,))
        return self.db.query("SELECT * FROM eval_sets ORDER BY created")

    def promote_to_regression(self, name: str):
        self.db.execute("UPDATE eval_sets SET kind='regression' WHERE name=?", (name,))
        self.db.audit("system", "eval_set_promoted", name=name)

    def load(self, name):
        r = self.get(name)
        if not r:
            raise KeyError(name)
        if sha256_file(r["path"]) != r["sha256"]:
            raise TamperError(f"eval set '{name}' was modified on disk (hash mismatch)")
        return json.loads(Path(r["path"]).read_text(encoding="utf-8"))["data"]

    def verify_all(self) -> dict:
        out = {}
        for r in self.list():
            try:
                self.load(r["name"]); out[r["name"]] = "ok"
            except TamperError as e:
                out[r["name"]] = str(e)
        return out

    def all_shingles(self) -> set[int]:
        """Fingerprints of all evaluation text - used to keep it out of training."""
        sh: set[int] = set()
        for r in self.list():
            data = self.load(r["name"])
            if r["fmt"] == "text":
                sh |= shingles(data)
            else:
                for it in data:
                    sh |= shingles(it["prompt"] + " " + it["answer"])
        return sh
