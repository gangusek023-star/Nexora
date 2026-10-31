"""NEXORA language model: a small decoder-only Transformer written from scratch.

Every component (attention, MLP, normalisation, embeddings) is implemented
here with plain torch tensor ops - no pretrained weights, no model zoo.
Weights are randomly initialised (normal, std 0.02, residual projections
scaled by 1/sqrt(2 * n_layer) as in GPT-2).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "LMConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj.is_residual = True
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # causal mask: position t may only attend to positions <= t
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.proj.is_residual = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm Transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NexoraLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying saves memory
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "is_residual", False):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None):
        """idx: (B, T) token ids. Returns logits (B, T, V) and mean CE loss.

        loss_mask (B, T) optionally restricts the loss to some positions
        (used to score only the answer part of an evaluation item)."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                                 reduction="none").view(B, T)
            if loss_mask is not None:
                loss = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                loss = ce.mean()
        return logits, loss

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40, top_p=0.95,
                 repetition_penalty=1.0, stop_ids=None, generator=None, stop_fn=None):
        """Autoregressive sampling from the model's own learned distribution."""
        was_training = self.training
        self.eval()
        stop_ids = set(stop_ids or [])
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if repetition_penalty and repetition_penalty != 1.0:
                prev = torch.unique(ctx[0])
                sel = logits[0, prev]
                logits[0, prev] = torch.where(sel > 0, sel / repetition_penalty,
                                              sel * repetition_penalty)
            if temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if top_p and 0 < top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    sp[(cum - sp) > top_p] = 0.0
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if int(nxt[0, 0]) in stop_ids or (stop_fn is not None and stop_fn(idx)):
                break
        if was_training:
            self.train()
        return idx


def save_model(model: NexoraLM, path, extra: dict | None = None):
    payload = {"config": model.cfg.to_dict(), "state_dict": model.state_dict(),
               "format": "nexora-lm-v1"}
    if extra:
        payload["extra"] = extra
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    import os
    os.replace(tmp, path)


def load_model(path, map_location="cpu") -> NexoraLM:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != "nexora-lm-v1":
        raise ValueError(f"{path} is not a NEXORA LM checkpoint")
    model = NexoraLM(LMConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model
