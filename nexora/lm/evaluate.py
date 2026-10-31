"""Deterministic evaluation against ground-truth data.

The model is never scored by its own confidence: every metric here compares
the model's predictions with held-out reference text it was not trained on.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .model import NexoraLM


@torch.no_grad()
def eval_text(model: NexoraLM, arr: np.ndarray, max_windows: int = 48, batch: int = 8) -> dict:
    """Mean next-token cross-entropy over evenly spaced fixed windows."""
    T = model.cfg.block_size
    if len(arr) < T + 1:
        T = len(arr) - 1
    if T < 2:
        raise ValueError("evaluation text too short")
    n_possible = (len(arr) - 1) // T
    starts = np.linspace(0, (n_possible - 1) * T, num=min(max_windows, n_possible)).astype(int)
    was = model.training
    model.eval()
    tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        x = torch.stack([torch.from_numpy(arr[j:j + T].astype(np.int64)) for j in s])
        y = torch.stack([torch.from_numpy(arr[j + 1:j + T + 1].astype(np.int64)) for j in s])
        logits, loss = model(x, y)
        n = y.numel()
        tot_loss += loss.item() * n
        tot_correct += (logits.argmax(-1) == y).sum().item()
        tot_tokens += n
    model.train(was)
    loss = tot_loss / tot_tokens
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "token_acc": tot_correct / tot_tokens,
            "tokens": tot_tokens}


@torch.no_grad()
def eval_qa(model: NexoraLM, tokenizer, items: list[dict]) -> dict:
    """Teacher-forced loss of the reference answer given the prompt.

    Each item: {"prompt": str, "answer": str}. Only answer tokens count."""
    if not items:
        raise ValueError("no evaluation items")
    T = model.cfg.block_size
    was = model.training
    model.eval()
    tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
    for it in items:
        p = tokenizer.encode(it["prompt"])
        a = tokenizer.encode(it["answer"])[: T // 2]
        seq = (p + a)[-(T + 1):]
        n_ans = len(a)
        x = torch.tensor(seq[:-1]).unsqueeze(0)
        y = torch.tensor(seq[1:]).unsqueeze(0)
        mask = torch.zeros_like(y, dtype=torch.float32)
        mask[0, -n_ans:] = 1.0
        logits, loss = model(x, y, loss_mask=mask)
        tot_loss += loss.item() * n_ans
        tot_correct += ((logits.argmax(-1) == y).float() * mask).sum().item()
        tot_tokens += n_ans
    model.train(was)
    loss = tot_loss / max(tot_tokens, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 20)),
            "token_acc": tot_correct / max(tot_tokens, 1), "tokens": tot_tokens,
            "items": len(items)}
