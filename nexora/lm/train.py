"""Training loop: next-token cross-entropy, backprop, AdamW.

Used for both the first training run (from random weights) and candidate
fine-tuning. Supports:
  * resumable checkpoints (model + optimiser + sampler RNG + step)
  * pause / cancel / time / memory / disk limits via JobControl.tick()
  * replay buffers (through the BatchSampler sources)
  * L2-SP anchoring to the previous approved weights (anti-forgetting)
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch

from ..jobs.control import JobControl, StopTraining
from .model import NexoraLM


def lr_at(step, base_lr, warmup, max_steps, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def _save_ckpt(path, model, opt, step, sampler, history, extra):
    tmp = str(path) + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "sampler": sampler.state(), "history": history, "extra": extra,
                "torch_rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, path)


def train(model: NexoraLM, sampler, *, max_steps: int, lr: float, weight_decay: float = 0.05,
          warmup: int = 100, grad_clip: float = 1.0, eval_every: int = 250,
          checkpoint_every: int = 250, ckpt_path: str | Path | None = None,
          control: JobControl | None = None, eval_fn=None,
          anchor_state: dict | None = None, anchor_l2: float = 0.0, seed: int = 1337) -> dict:
    """Returns {"status", "step", "history"}; status in
    completed | cancelled | time_limit | memory_limit | disk_space."""
    control = control or JobControl(max_minutes=0, max_ram_mb=0, min_free_disk_mb=0)
    torch.manual_seed(seed)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    step, history = 0, {"train": [], "val": []}

    if ckpt_path and Path(ckpt_path).exists():  # resume after pause/restart
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        sampler.set_state(ck["sampler"])
        torch.set_rng_state(ck["torch_rng"])
        control.report(state="running", message=f"resumed from step {step}")

    anchors = None
    if anchor_state is not None and anchor_l2 > 0:
        names = dict(model.named_parameters())
        anchors = [(names[k], v.detach().clone()) for k, v in anchor_state.items()
                   if k in names and names[k].requires_grad]

    def save():
        if ckpt_path:
            control.check_disk(needed_mb=50)
            _save_ckpt(ckpt_path, model, opt, step, sampler, history, {})

    model.train()
    status = "completed"
    t_last, running = time.time(), None
    try:
        while step < max_steps:
            control.tick(save_fn=save)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, lr, warmup, max_steps)
            x, y = sampler.next()
            _, loss = model(x, y)
            total = loss
            if anchors:
                reg = sum(((p - p0) ** 2).sum() for p, p0 in anchors)
                total = loss + anchor_l2 * reg
            opt.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1
            lv = loss.item()
            running = lv if running is None else 0.95 * running + 0.05 * lv
            if step % 10 == 0 or step == max_steps:
                history["train"].append([step, round(running, 4)])
                now = time.time()
                control.report(state="running", step=step, max_steps=max_steps,
                               train_loss=round(running, 4), lr=opt.param_groups[0]["lr"],
                               sec_per_step=round((now - t_last) / 10, 3))
                t_last = now
            if eval_fn and (step % eval_every == 0 or step == max_steps):
                v = eval_fn(model)
                history["val"].append([step, round(v, 4)])
                control.report(val_loss=round(v, 4), step=step)
            if checkpoint_every and step % checkpoint_every == 0:
                save()
    except StopTraining as e:
        status = e.reason.split(":")[0]
        try:
            save()
        except StopTraining:
            pass  # out of disk: keep previous checkpoint
    if status == "completed":
        save()
    return {"status": status, "step": step, "history": history}
