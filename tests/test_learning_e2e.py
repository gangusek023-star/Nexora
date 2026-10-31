"""Full learning cycle on a tiny trained model: feedback -> dataset -> candidate ->
guard evaluation -> approval / rejection / rollback."""
import torch

from conftest import wait_job

CORRECTIONS = [
    ("Who is the king?", "The king is Richard, and he rules all of England."),
    ("Where is the queen?", "The queen is walking slowly in the rose garden."),
    ("What hour is it?", "It is now nine of the clock in the morning."),
    ("Who are you?", "I am Nexora, a small model that learns from you."),
    ("Shall we fight?", "No, let us make peace with our neighbours today."),
    ("What is your name?", "My name is Nexora and I am glad to meet you."),
]


def _feedback(app):
    cid = app.memory.new_conversation()
    for q, a in CORRECTIONS:
        app.memory.add_message(cid, "user", q)
        mid = app.memory.add_message(cid, "assistant", "thou art a villain", "v0001")
        app.memory.add_feedback(mid, -1, a)


def test_base_model_trained_from_random_weights(trained_app):
    a = trained_app
    v = a.registry.active()
    assert v["id"] == "v0001" and v["parent"] is None
    hist = v["metrics"]["_train"]["history"]
    assert hist["train"][-1][1] < hist["train"][0][1]  # loss went down
    assert set(k for k in v["metrics"] if not k.startswith("_")) == {"heldout_base_text", "skill_dialogue_continuation"}
    out = a.lm.chat("Good morrow!", [], max_new_tokens=15, seed=1)
    assert out["version"] == "v0001" and out["tokens"] > 0


def test_cycle_candidate_approve_and_rollback(trained_app):
    a = trained_app
    w_before = a.registry.load("v0001").state_dict()["tok_emb.weight"].clone()
    _feedback(a)
    log = a.learning.run_cycle("test")
    msgs = [s["msg"] for s in log["steps"]]
    assert "created candidate dataset" in msgs and "candidate training queued" in msgs
    goals = a.learning.goals("active")
    assert any(g["kind"] == "corrections" for g in goals)
    # held-out slice of the corrections became an immutable eval set
    target = [s for s in log["steps"] if s["msg"] == "created candidate dataset"][0]["target_set"]
    assert a.evals.get(target)["kind"] == "heldout"
    held = a.db.query("SELECT * FROM examples WHERE status='eval'")
    trained_on = a.datasets.documents(a.datasets.list()[0]["id"])
    assert not {h["text"] for h in held} & {d["text"] for d in trained_on}  # never train on eval
    jid = [s for s in log["steps"] if "job_id" in s][0]["job_id"]
    j = wait_job(a, jid)
    assert j["status"] == "completed", j
    cand = a.registry.get(j["result"]["candidate"])
    rep = cand["report"]
    assert rep["current"] == "v0001" and {r["set"] for r in rep["comparison"]["rows"]} >= {target, "heldout_base_text"}
    assert a.registry.active()["id"] == "v0001"  # nothing changes before approval
    assert len(list(a.paths.backups.glob("v0001-*.pt"))) == 1
    if cand["status"] == "pending_approval":
        assert "waiting for your approval" in a.learning.run_cycle("test")["steps"][-1]["msg"]
        a.learning.decide(cand["id"], approve=True)
        assert a.registry.active()["id"] == cand["id"]
        assert a.evals.get(target)["kind"] == "regression"  # new skill now protected
        assert not torch.equal(a.registry.load(cand["id"]).state_dict()["tok_emb.weight"], w_before)
        assert a.lm.chat("Who are you?", [], max_new_tokens=5)["version"] == cand["id"]
        a.learning.rollback("v0001")
        assert a.registry.active()["id"] == "v0001"
    else:
        assert cand["status"] == "rejected" and rep["auto_action"]


def test_candidate_rejection_keeps_active_model(trained_app):
    a = trained_app
    _feedback(a)
    log = a.learning.run_cycle("test")
    jid = [s for s in log["steps"] if "job_id" in s][0]["job_id"]
    j = wait_job(a, jid)
    vid = j["result"]["candidate"]
    if a.registry.get(vid)["status"] == "pending_approval":
        a.learning.decide(vid, approve=False, note="not convinced")
    v = a.registry.get(vid)
    assert v["status"] == "rejected"
    assert a.registry.active()["id"] == "v0001"
    import pytest
    with pytest.raises(ValueError):
        a.learning.decide(vid, approve=True)  # cannot approve a rejected model


def test_guard_auto_rejects_a_damaged_candidate(trained_app):
    """Force a bad candidate (huge learning rate) and check the guard rejects it."""
    a = trained_app
    _feedback(a)
    from nexora.config import save_settings
    save_settings({"learning": {"finetune_lr": 0.5, "anchor_l2": 0.0, "max_regression_pct": 3.0}}, a.paths)
    a.reload_config()
    log = a.learning.run_cycle("test")
    jid = [s for s in log["steps"] if "job_id" in s][0]["job_id"]
    j = wait_job(a, jid)
    v = a.registry.get(j["result"]["candidate"])
    assert v["status"] == "rejected" and not v["report"]["comparison"]["passed"]
    assert a.registry.active()["id"] == "v0001"


def test_not_enough_data_does_nothing(trained_app):
    log = trained_app.learning.run_cycle("test")
    assert "not enough new validated data" in log["steps"][-1]["msg"]
    assert not trained_app.jobs.busy()
