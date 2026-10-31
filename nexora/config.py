"""Configuration and filesystem layout.

All runtime state lives under one "home" directory (the project folder by
default, or the NEXORA_HOME environment variable - tests use a temp dir).
Defaults come from config/default.json; user changes are written to
<home>/settings.json and merged on top.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.json"

_lock = threading.Lock()


def home() -> Path:
    h = Path(os.environ.get("NEXORA_HOME", PROJECT_ROOT / "runtime"))
    h.mkdir(parents=True, exist_ok=True)
    return h


class Paths:
    """Every directory the app writes to. Created lazily."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else home()

    def _d(self, *parts) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db(self) -> Path: return self._d("db") / "nexora.sqlite3"
    @property
    def models(self) -> Path: return self._d("models", "lm")
    @property
    def backups(self) -> Path: return self._d("models", "lm_backups")
    @property
    def job_ckpt(self) -> Path: return self._d("jobs")
    @property
    def datasets(self) -> Path: return self._d("datasets")
    @property
    def eval_sets(self) -> Path: return self._d("eval_sets")
    @property
    def tokenizer(self) -> Path: return self._d("tokenizer") / "tokenizer.json"
    @property
    def image_models(self) -> Path: return self._d("models", "image")
    @property
    def image_data(self) -> Path: return self._d("image_data")
    @property
    def generated(self) -> Path: return self._d("generated")
    @property
    def reports(self) -> Path: return self._d("reports")
    @property
    def logs(self) -> Path: return self._d("logs")
    @property
    def uploads(self) -> Path: return self._d("uploads")
    @property
    def settings(self) -> Path: return self.root / "settings.json"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(paths: Paths | None = None) -> dict:
    paths = paths or Paths()
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    if paths.settings.exists():
        try:
            with open(paths.settings, encoding="utf-8") as f:
                cfg = _deep_merge(cfg, json.load(f))
        except (OSError, json.JSONDecodeError):
            pass  # a broken settings file must never stop the app
    return cfg


def save_settings(changes: dict, paths: Paths | None = None) -> dict:
    """Merge `changes` into settings.json (only keys that exist in defaults)."""
    paths = paths or Paths()
    with _lock:
        with open(DEFAULT_CONFIG, encoding="utf-8") as f:
            defaults = json.load(f)
        current = {}
        if paths.settings.exists():
            try:
                current = json.loads(paths.settings.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
        for section, vals in changes.items():
            if section not in defaults or not isinstance(vals, dict):
                raise ValueError(f"Unknown settings section: {section}")
            for k in vals:
                if k not in defaults[section]:
                    raise ValueError(f"Unknown setting: {section}.{k}")
            current.setdefault(section, {}).update(vals)
        tmp = paths.settings.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
        os.replace(tmp, paths.settings)
    return load_config(paths)
