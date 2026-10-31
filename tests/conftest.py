"""Shared fixtures. Every test runs in its own temporary NEXORA_HOME with a
tiny model configuration, so the suite finishes in a few minutes on a CPU."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "data" / "raw" / "tinyshakespeare.txt"

TINY = {
    "resources": {"torch_threads": 2, "max_process_ram_mb": 0, "min_free_disk_mb": 50},
    "lm": {"vocab_size": 300, "block_size": 32, "n_layer": 2, "n_head": 2, "n_embd": 32,
           "batch_size": 8, "max_steps": 40, "eval_every": 20, "checkpoint_every": 20,
           "warmup_steps": 5, "learning_rate": 0.003},
    "learning": {"finetune_steps": 30, "min_improvement_pct": 0.1, "max_regression_pct": 25.0},
    "image": {"resolution": 16, "base_channels": 16, "channel_mults": [1, 2], "text_dim": 32,
              "timesteps": 50, "batch_size": 8, "max_steps": 30, "checkpoint_every": 10,
              "sample_steps": 5},
}


def make_home(path: Path, overrides=None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(TINY))
    for k, v in (overrides or {}).items():
        cfg.setdefault(k, {}).update(v)
    (path / "settings.json").write_text(json.dumps(cfg))
    return path


def small_corpus(path: Path, chars=120_000) -> Path:
    p = path / "corpus.txt"
    p.write_text(CORPUS.read_text(encoding="utf-8")[:chars], encoding="utf-8")
    return p


def wait_job(app, jid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = app.jobs.get(jid)
        if j["status"] not in ("queued", "running", "paused"):
            return j
        time.sleep(0.2)
    raise TimeoutError(f"job {jid} did not finish")


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = make_home(tmp_path / "home")
    monkeypatch.setenv("NEXORA_HOME", str(h))
    return h


@pytest.fixture
def app(home):
    from nexora.core import Nexora
    a = Nexora()
    yield a
    a.shutdown()


@pytest.fixture(scope="session")
def trained_home(tmp_path_factory):
    """A home with prepared data and one tiny trained base model (v0001)."""
    h = make_home(tmp_path_factory.mktemp("trained") / "home")
    old = os.environ.get("NEXORA_HOME")
    os.environ["NEXORA_HOME"] = str(h)
    from nexora.core import Nexora
    from nexora.learning.base import prepare_base
    a = Nexora()
    prepare_base(a, [small_corpus(h.parent)], log=lambda *_: None)
    j = wait_job(a, a.jobs.submit("base_train", {}))
    assert j["status"] == "completed", j
    a.shutdown()
    if old is None:
        os.environ.pop("NEXORA_HOME", None)
    else:
        os.environ["NEXORA_HOME"] = old
    return h


@pytest.fixture
def trained_app(trained_home, tmp_path, monkeypatch):
    """Fresh copy of the trained home for each test (tests may mutate it)."""
    import shutil
    dst = tmp_path / "home"
    # copying also proves the runtime folder is relocatable (relative paths in the DB)
    shutil.copytree(trained_home, dst)
    monkeypatch.setenv("NEXORA_HOME", str(dst))
    from nexora.core import Nexora
    a = Nexora()
    yield a
    a.shutdown()
