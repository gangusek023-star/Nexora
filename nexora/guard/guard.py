"""Learning Guard: decides whether a candidate model may be proposed.

A candidate is compared with the currently active model on every immutable
evaluation set (held-out + regression suite) using ground-truth loss, plus
objective generation sanity checks. The candidate's own confidence is never
used as evidence of quality.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from ..lm.data import USER_TAG, BOT_TAG
from ..lm.evaluate import eval_qa, eval_text
from ..lm.model import NexoraLM
from .registry import EvalSuite

SANITY_PROMPTS = [
    f"{USER_TAG}Good morning to you.\n\n{BOT_TAG}",
    f"{USER_TAG}What news do you bring?\n\n{BOT_TAG}",
    f"{USER_TAG}Tell me of the king.\n\n{BOT_TAG}",
    "First Citizen:\n",
]


def evaluate_all(model: NexoraLM, tokenizer, suite: EvalSuite, names=None) -> dict:
    out = {}
    for r in suite.list():
        if names and r["name"] not in names:
            continue
        data = suite.load(r["name"])
        if r["fmt"] == "text":
            arr = np.asarray(tokenizer.encode(data), dtype=np.int64)
            m = eval_text(model, arr)
        else:
            m = eval_qa(model, tokenizer, data)
        out[r["name"]] = {**{k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()},
                          "kind": r["kind"]}
    return out


def sanity_check(model: NexoraLM, tokenizer, max_new_tokens=60, seed=0) -> dict:
    """Objective checks on sampled text: not empty, not degenerate loops."""
    gen = torch.Generator().manual_seed(seed)
    samples, distinct = [], []
    for p in SANITY_PROMPTS:
        ids = torch.tensor([tokenizer.encode(p)])
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40,
                             generator=gen)
        text = tokenizer.decode(out[0, ids.shape[1]:].tolist())
        samples.append({"prompt": p, "output": text})
        words = re.findall(r"\w+", text.lower())
        bigrams = list(zip(words, words[1:]))
        distinct.append(len(set(bigrams)) / len(bigrams) if bigrams else 0.0)
    d2 = float(np.mean(distinct))
    return {"distinct2": round(d2, 4), "samples": samples,
            "ok": d2 >= 0.5 and all(s["output"].strip() for s in samples)}


def compare(current: dict, candidate: dict, target_sets: list[str], *,
            max_regression_pct: float, min_improvement_pct: float,
            sanity: dict | None = None) -> dict:
    """Loss-based comparison. delta_pct > 0 means the candidate is better."""
    rows, reasons, passed = [], [], True
    for name, cm in candidate.items():
        if name not in current:
            continue
        cur, cand = current[name]["loss"], cm["loss"]
        delta = (cur - cand) / cur * 100 if cur else 0.0
        role = "target" if name in target_sets else cm.get("kind", "heldout")
        row = {"set": name, "role": role, "current_loss": cur, "candidate_loss": cand,
               "delta_pct": round(delta, 3), "verdict": "ok"}
        if not np.isfinite(cand):
            row["verdict"] = "invalid"; passed = False
            reasons.append(f"{name}: candidate loss is not finite")
        elif role != "target" and delta < -max_regression_pct:
            row["verdict"] = "regression"; passed = False
            reasons.append(f"{name}: {(-delta):.2f}% worse (limit {max_regression_pct}%)")
        elif delta > 0.05:
            row["verdict"] = "improved"
        elif delta < -0.05:
            row["verdict"] = "slightly worse"
        rows.append(row)
    target_rows = [r for r in rows if r["role"] == "target"]
    if not target_rows:
        passed = False
        reasons.append("no target evaluation set to measure the learning goal")
    for r in target_rows:
        if r["delta_pct"] < min_improvement_pct:
            passed = False
            reasons.append(f"{r['set']}: improved only {r['delta_pct']:.2f}% "
                           f"(need {min_improvement_pct}%)")
    if sanity is not None and not sanity.get("ok", False):
        passed = False
        reasons.append(f"generation sanity check failed (distinct-2 = {sanity.get('distinct2')})")
    if passed:
        reasons = ["all checks passed"]
        worse = [r["set"] for r in rows if r["verdict"] == "slightly worse"]
        if worse:
            reasons.append(f"{', '.join(worse)}: slightly worse, but within the "
                           f"{max_regression_pct}% limit - check the sample outputs before approving")
    return {"passed": passed, "rows": rows, "reasons": reasons}
