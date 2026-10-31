"""Learning Guard (comparison, immutability, leakage), registry/rollback and job control."""
import os
import time

import pytest
import torch

from conftest import wait_job
from nexora.guard.guard import compare
from nexora.guard.registry import TamperError
from nexora.jobs.control import JobControl, StopTraining
from nexora.lm.model import LMConfig, NexoraLM

M = lambda loss, kind="heldout": {"loss": loss, "kind": kind}  # noqa: E731


def test_compare_detects_regression():
    cur = {"a": M(2.0), "reg": M(1.0, "regression"), "goal": M(3.0)}
    cand = {"a": M(2.0), "reg": M(1.10, "regression"), "goal": M(2.5)}
    r = compare(cur, cand, ["goal"], max_regression_pct=3, min_improvement_pct=0.5)
    assert not r["passed"]
    assert any("reg" in x and "worse" in x for x in r["reasons"])
    assert [x for x in r["rows"] if x["set"] == "reg"][0]["verdict"] == "regression"


def test_compare_requires_target_improvement_and_passes_good_candidate():
    cur = {"a": M(2.0), "goal": M(3.0)}
    same = compare(cur, {"a": M(2.0), "goal": M(3.0)}, ["goal"], max_regression_pct=3, min_improvement_pct=0.5)
    assert not same["passed"]
    good = compare(cur, {"a": M(2.02), "goal": M(2.5)}, ["goal"], max_regression_pct=3, min_improvement_pct=0.5)
    assert good["passed"], good
    nan = compare(cur, {"a": M(float("nan")), "goal": M(2.5)}, ["goal"], max_regression_pct=3, min_improvement_pct=0.5)
    assert not nan["passed"]
    notarget = compare(cur, {"a": M(1.0)}, [], max_regression_pct=3, min_improvement_pct=0.5)
    assert not notarget["passed"]
    bad_sanity = compare(cur, {"a": M(2.0), "goal": M(2.0)}, ["goal"], max_regression_pct=3,
                         min_improvement_pct=0.5, sanity={"ok": False, "distinct2": 0.1})
    assert not bad_sanity["passed"]


def test_eval_sets_are_immutable_and_tamper_evident(app):
    app.evals.create("t1", "heldout", "text", "Lorem ipsum dolor sit amet. " * 20)
    with pytest.raises(ValueError):
        app.evals.create("t1", "heldout", "text", "x" * 300)
    assert app.evals.verify_all() == {"t1": "ok"}
    p = app.evals.get("t1")["path"]
    os.chmod(p, 0o644)
    with open(p, "a") as f:
        f.write(" ")
    with pytest.raises(TamperError):
        app.evals.load("t1")


def test_datasets_versioned_and_tamper_evident(app):
    from nexora.lm.tokenizer import BPETokenizer
    tok = BPETokenizer.train("hello world " * 200, 260)
    d1 = app.datasets.create([{"text": "hello world", "source": "s"}], tok)
    d2 = app.datasets.create([{"text": "hello again", "source": "s"}], tok, parent=d1)
    assert app.datasets.lineage(d2) == [d2, d1]
    assert len(app.datasets.tokens(d1)) > 0
    f = os.path.join(app.datasets.get(d1)["path"], "tokens.npy")
    with open(f, "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(TamperError):
        app.datasets.tokens(d1)


def test_registry_activation_rollback_and_rejection(app):
    cfg = LMConfig(vocab_size=50, block_size=8, n_layer=1, n_head=1, n_embd=8)
    v1 = app.registry.register(NexoraLM(cfg), parent=None, status="active", dataset_id=None)
    v2 = app.registry.register(NexoraLM(cfg), parent=v1, status="pending_approval", dataset_id=None)
    v3 = app.registry.register(NexoraLM(cfg), parent=v1, status="rejected", dataset_id=None)
    assert app.registry.active()["id"] == v1
    app.registry.activate(v2)
    assert app.registry.active()["id"] == v2 and app.registry.get(v1)["status"] == "approved"
    app.registry.activate(v1)  # rollback
    assert app.registry.active()["id"] == v1
    with pytest.raises(ValueError):
        app.registry.activate(v3)
    with pytest.raises(ValueError):
        app.registry.delete_version(v1)
    b = app.registry.backup_active()
    assert b and os.path.exists(b)
    a = app.registry.load(v1)
    assert torch.equal(a.tok_emb.weight, app.registry.load(v1).tok_emb.weight)


def test_example_validation_dedup_and_leakage(app):
    held = ("Now is the winter of our discontent made glorious summer by this sun of York "
            "and all the clouds that lour'd upon our house in the deep bosom of the ocean buried. ") * 3
    app.evals.create("held", "heldout", "text", held)
    sh = app.evals.all_shingles()
    L = app.learning
    ok = L.add_example("The garden behind the old house was full of roses and quiet bees.", "user_added", "t", eval_shingles=sh)
    assert ok["status"] == "accepted"
    assert L.add_example("The garden behind the old house was full of roses and quiet bees.", "user_added", "t")["status"] == "duplicate"
    leak = L.add_example(held[:200], "local_file", "t", eval_shingles=sh)
    assert leak["status"] == "rejected" and "leakage" in leak["reason"]
    gen = L.add_example("A sentence the model produced by itself, maybe true maybe not.", "generated", "m")
    assert gen["status"] == "pending"  # model output never trusted automatically


# ------------------------------------------------------------------ jobs
def _loop_handler(steps=200, delay=0.01):
    def h(job, control):
        n = 0
        try:
            while n < steps:
                control.tick()
                time.sleep(delay)
                n += 1
                control.report(step=n, max_steps=steps)
        except StopTraining as e:
            return {"status": e.reason.split(":")[0], "step": n}
        return {"status": "completed", "step": n}
    return h


def test_job_pause_resume_cancel(app):
    app.jobs.handlers["loop"] = _loop_handler()
    jid = app.jobs.submit("loop")
    while app.jobs.get(jid)["status"] != "running":
        time.sleep(0.05)
    time.sleep(0.2)
    app.jobs.pause(jid)
    time.sleep(0.3)
    s1 = app.jobs.live[jid]["step"]
    time.sleep(0.4)
    assert app.jobs.live[jid]["step"] == s1  # really paused
    assert app.jobs.get(jid)["status"] == "paused"
    app.jobs.resume(jid)
    time.sleep(0.3)
    assert app.jobs.live[jid]["step"] > s1
    app.jobs.cancel(jid)
    j = wait_job(app, jid)
    assert j["status"] == "cancelled" and j["result"]["step"] < 200


def test_job_time_limit_and_failure(app):
    app.jobs.handlers["loop"] = _loop_handler(steps=100000)
    jid = app.jobs.submit("loop", {"max_minutes": 0.01})  # 0.6 s
    j = wait_job(app, jid, timeout=30)
    assert j["status"] == "stopped" and "time_limit" in j["error"]

    def boom(job, control):
        raise RuntimeError("kaboom")
    app.jobs.handlers["boom"] = boom
    j = wait_job(app, app.jobs.submit("boom"))
    assert j["status"] == "failed" and "kaboom" in j["error"]


def test_jobs_recover_after_restart(app):
    jid = app.db.execute("INSERT INTO jobs(kind,status,params,created,updated) VALUES "
                         "('base_train','running','{}',0,0)").lastrowid
    from nexora.core import Nexora
    b = Nexora(start_workers=False)  # simulates restarting the app
    assert b.jobs.get(jid)["status"] == "interrupted"
    b.shutdown()


def test_memory_limit_stops_training(tmp_path):
    c = JobControl(max_minutes=0, max_ram_mb=1, min_free_disk_mb=0)
    c._last_resource_check = 0
    with pytest.raises(StopTraining) as e:
        c.tick()
    assert e.value.reason.startswith("memory_limit")
    c2 = JobControl(max_minutes=0, max_ram_mb=0, min_free_disk_mb=10**12)
    with pytest.raises(StopTraining):
        c2.check_disk()


def test_schedule_window_and_due(app):
    from nexora.learning.scheduler import in_window
    assert in_window("0-24", 3) and in_window("22-7", 23) and in_window("22-7", 2)
    assert not in_window("22-7", 12) and in_window("9-17", 9) and not in_window("9-17", 17)
    ok, why = app.scheduler.due()
    assert not ok and why == "schedule disabled"


def test_instance_lock_prevents_two_workers(app):
    """A second NEXORA process must not hijack (or 'recover') the first one's jobs."""
    import subprocess
    import sys as _sys
    other = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (app.paths.root / "nexora.lock").write_text(str(other.pid))
        assert app.jobs.other_instance() == other.pid
        from nexora.core import Nexora
        with pytest.raises(RuntimeError, match="already running"):
            Nexora()
    finally:
        other.kill()
