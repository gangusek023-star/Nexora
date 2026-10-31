"""Base corpus preparation and the first training run (from random weights)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..lm.data import BatchSampler, normalize_text, tokens_to_array
from ..lm.evaluate import eval_text
from ..lm.model import LMConfig, NexoraLM
from ..lm.tokenizer import BPETokenizer
from ..lm.train import train
from ..guard.guard import evaluate_all, sanity_check

_TURN = re.compile(r"^([A-Z][A-Za-z' ]{1,40}):\n((?:.+\n?)+)", re.M)


def dialogue_items(text: str, max_items=40, max_answer_chars=160) -> list[dict]:
    """Build 'continue the dialogue' evaluation items from play-formatted text:
    prompt = previous turn + next speaker tag, answer = next speaker's words."""
    turns = [(m.group(1), m.group(2).strip()) for m in _TURN.finditer(text)]
    items = []
    for (s1, t1), (s2, t2) in zip(turns, turns[1:]):
        if len(t2) < 20:
            continue
        items.append({"prompt": f"{s1}:\n{t1[-300:]}\n\n{s2}:\n", "answer": t2[:max_answer_chars]})
    if len(items) > max_items:
        step = len(items) / max_items
        items = [items[int(i * step)] for i in range(max_items)]
    return items


def _cut_at_paragraph(text: str, pos: int) -> int:
    j = text.find("\n\n", pos)
    return j + 2 if j != -1 else pos


def prepare_base(app, corpus_files: list[str | Path], log=print) -> dict:
    """Split corpus -> train / validation / held-out, train tokenizer, create
    dataset ds0001 and the immutable evaluation sets. Idempotent-ish: refuses
    to run twice (the tokenizer is frozen once created)."""
    if app.paths.tokenizer.exists() and app.datasets.list():
        raise RuntimeError("Base data already prepared. Delete the runtime folder to start over.")
    text = "\n\n".join(normalize_text(Path(f).read_text(encoding="utf-8", errors="replace"))
                       for f in corpus_files)
    if len(text) < 20_000:
        raise ValueError("Corpus too small (need at least ~20,000 characters)")
    n = len(text)
    a = _cut_at_paragraph(text, int(n * 0.90))
    b = _cut_at_paragraph(text, int(n * 0.95))
    train_txt, val_txt, held_txt = text[:a], text[a:b], text[b:]
    log(f"corpus {n:,} chars -> train {len(train_txt):,} / validation {len(val_txt):,} "
        f"/ held-out {len(held_txt):,}")

    cfg = app.cfg()["lm"]
    log(f"training BPE tokenizer (vocab {cfg['vocab_size']}) on the training split only ...")
    tok = BPETokenizer.train(train_txt, cfg["vocab_size"])
    tok.save(app.paths.tokenizer)

    # Training docs = ~paragraph-aligned chunks of the training split
    docs, pos, chunk = [], 0, 20_000
    while pos < len(train_txt):
        end = _cut_at_paragraph(train_txt, pos + chunk) if pos + chunk < len(train_txt) else len(train_txt)
        docs.append({"text": train_txt[pos:end].strip(), "source": "base corpus",
                     "origin": "local_file"})
        pos = end
    did = app.datasets.create(docs, tok, parent=None, note="base corpus (training split)")
    np.save(app.paths.datasets / "validation.npy", tokens_to_array(tok.encode(val_txt)))

    app.evals.create("heldout_base_text", "heldout", "text", held_txt,
                     "Held-out final 5% of the base corpus - never trained on")
    items = dialogue_items(held_txt)
    if len(items) >= 5:
        app.evals.create("skill_dialogue_continuation", "regression", "qa", items,
                         "Continue a dialogue turn (from held-out text) - core skill regression test")
    return {"dataset_id": did, "vocab_size": tok.vocab_size, "train_tokens": len(app.datasets.tokens(did)),
            "eval_sets": [r["name"] for r in app.evals.list()]}


def base_train_handler(app):
    """Job handler: train a fresh model from random initialisation."""

    def run(job, control):
        p = job["params"]
        cfg = {**app.cfg()["lm"], **p.get("overrides", {})}
        tok = BPETokenizer.load(app.paths.tokenizer)
        did = p.get("dataset_id") or app.datasets.list()[-1]["id"]
        arr = app.datasets.tokens(did)
        val = np.load(app.paths.datasets / "validation.npy")
        mcfg = LMConfig(vocab_size=tok.vocab_size, block_size=cfg["block_size"],
                        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                        dropout=cfg["dropout"])
        import torch
        torch.manual_seed(p.get("seed", 1337))
        model = NexoraLM(mcfg)  # random weights
        control.report(state="running", message=f"model with {model.num_params():,} parameters, "
                                                f"random init", params=model.num_params())
        sampler = BatchSampler([(arr, 1.0)], cfg["block_size"], cfg["batch_size"],
                               seed=p.get("seed", 1337))
        res = train(model, sampler, max_steps=cfg["max_steps"], lr=cfg["learning_rate"],
                    weight_decay=cfg["weight_decay"], warmup=cfg["warmup_steps"],
                    grad_clip=cfg["grad_clip"], eval_every=cfg["eval_every"],
                    checkpoint_every=cfg["checkpoint_every"],
                    ckpt_path=app.paths.job_ckpt / f"job-{job['id']}.pt", control=control,
                    eval_fn=lambda m: eval_text(m, val, max_windows=24)["loss"],
                    seed=p.get("seed", 1337))
        if res["status"] != "completed":
            return {**res, "reason": res["status"]}
        control.report(state="evaluating", message="evaluating on held-out sets")
        metrics = evaluate_all(model, tok, app.evals)
        sanity = sanity_check(model, tok)
        metrics["_sanity"] = {"distinct2": sanity["distinct2"], "ok": sanity["ok"]}
        metrics["_train"] = {"steps": res["step"], "final_train_loss": res["history"]["train"][-1][1],
                             "final_val_loss": res["history"]["val"][-1][1] if res["history"]["val"] else None,
                             "history": res["history"]}
        first = app.registry.active() is None
        vid = app.registry.register(model, parent=None, dataset_id=did, metrics=metrics,
                                    status="active" if first else "pending_approval",
                                    note="trained from random initialisation")
        if not first:
            app.registry.set_status(vid, "pending_approval", report={
                "kind": "retrain_from_scratch", "metrics": metrics, "samples": sanity["samples"]})
        app.db.audit("system", "base_model_trained", id=vid, activated=first)
        return {**res, "model_version": vid, "activated": first,
                "history": {"train": res["history"]["train"][-50:], "val": res["history"]["val"]}}

    return run
