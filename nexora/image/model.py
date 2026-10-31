"""Text-conditioned diffusion model built from scratch (random init).

Components
  * CaptionTokenizer   - word-level vocabulary built from your own captions
  * TextEncoder        - small Transformer encoder, trained jointly with the UNet
  * UNet               - noise predictor with timestep + text conditioning
                         (FiLM in every res-block, cross-attention at low res)
  * Diffusion          - forward noising q(x_t|x_0), DDPM / DDIM reverse process,
                         classifier-free guidance
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== text
class CaptionTokenizer:
    PAD, UNK, NULL = 0, 1, 2

    def __init__(self, words: list[str], max_len: int = 16):
        self.words = ["<pad>", "<unk>", "<null>"] + [w for w in words if w not in ("<pad>", "<unk>", "<null>")]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.max_len = max_len

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @classmethod
    def build(cls, captions: list[str], max_vocab: int = 2000, max_len: int = 16):
        from collections import Counter
        c = Counter(w for cap in captions for w in cls.split(cap))
        return cls([w for w, _ in c.most_common(max_vocab)], max_len)

    def encode(self, text: str) -> list[int]:
        ids = [self.index.get(w, self.UNK) for w in self.split(text)][: self.max_len]
        if not ids:
            ids = [self.NULL]  # empty caption = unconditional (for guidance)
        return ids + [self.PAD] * (self.max_len - len(ids))

    def batch(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

    def to_json(self):
        return json.dumps({"words": self.words, "max_len": self.max_len})

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        return cls(d["words"][3:], d["max_len"])

    @property
    def vocab_size(self):
        return len(self.words)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, layers: int = 2, heads: int = 4, max_len: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dim)

    def forward(self, ids):
        pad = ids == CaptionTokenizer.PAD
        x = self.emb(ids) + self.pos[:, : ids.shape[1]]
        x = self.ln(self.enc(x, src_key_padding_mask=pad))
        keep = (~pad).float().unsqueeze(-1)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        return x, pooled, pad


# ====================================================================== unet
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = _gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(cond_dim, cout * 2)
        self.n2, self.c2 = _gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(F.silu(cond))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Image positions attend to caption tokens."""

    def __init__(self, ch, text_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = _gn(ch)
        self.q = nn.Linear(ch, ch)
        self.kv = nn.Linear(text_dim, ch * 2)
        self.out = nn.Linear(ch, ch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, ctx, pad):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # B, HW, C
        q = self.q(h).view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k = k.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.heads)
        att = att.masked_fill(pad[:, None, None, :], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, H * W, C)
        return x + self.out(y).transpose(1, 2).view(B, C, H, W)


@dataclass
class ImageModelConfig:
    resolution: int = 32
    base_channels: int = 32
    channel_mults: list = field(default_factory=lambda: [1, 2, 2])
    text_dim: int = 128
    vocab_size: int = 100
    max_caption_len: int = 16
    timesteps: int = 400

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class UNet(nn.Module):
    def __init__(self, cfg: ImageModelConfig):
        super().__init__()
        ch, cd = cfg.base_channels, cfg.base_channels * 4
        self.cfg = cfg
        self.text = TextEncoder(cfg.vocab_size, cfg.text_dim, max_len=cfg.max_caption_len)
        self.t_mlp = nn.Sequential(nn.Linear(ch, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.txt_proj = nn.Linear(cfg.text_dim, cd)
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        chans = [ch * m for m in cfg.channel_mults]
        self.down, self.down_attn, self.downsample = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cin, skips = ch, [ch]
        for i, c in enumerate(chans):
            self.down.append(ResBlock(cin, c, cd))
            self.down_attn.append(CrossAttention(c, cfg.text_dim) if i == len(chans) - 1 else nn.Identity())
            skips.append(c)
            cin = c
            self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity())
        self.mid1 = ResBlock(cin, cin, cd)
        self.mid_attn = CrossAttention(cin, cfg.text_dim)
        self.mid2 = ResBlock(cin, cin, cd)
        self.up, self.up_attn, self.upsample = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, c in reversed(list(enumerate(chans))):
            self.up.append(ResBlock(cin + skips.pop(), c, cd))
            self.up_attn.append(CrossAttention(c, cfg.text_dim) if i == len(chans) - 1 else nn.Identity())
            cin = c
            self.upsample.append(nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity())
        self.out = nn.Sequential(_gn(cin + skips.pop()), nn.SiLU(), nn.Conv2d(cin + ch, 3, 3, padding=1))

    def forward(self, x, t, caption_ids):
        tokens, pooled, pad = self.text(caption_ids)
        cond = self.t_mlp(timestep_embedding(t, self.cfg.base_channels)) + self.txt_proj(pooled)
        h = self.inp(x)
        hs = [h]
        for blk, attn, ds in zip(self.down, self.down_attn, self.downsample):
            h = blk(h, cond)
            h = attn(h, tokens, pad) if isinstance(attn, CrossAttention) else h
            hs.append(h)
            h = ds(h)
        h = self.mid2(self.mid_attn(self.mid1(h, cond), tokens, pad), cond)
        for blk, attn, us in zip(self.up, self.up_attn, self.upsample):
            h = blk(torch.cat([h, hs.pop()], 1), cond)
            h = attn(h, tokens, pad) if isinstance(attn, CrossAttention) else h
            h = us(h)
        return self.out(torch.cat([h, hs.pop()], 1))


# ================================================================= diffusion
class Diffusion:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""

    def __init__(self, timesteps: int = 400):
        self.T = timesteps
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1)
        ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(1e-5, 0.999)
        self.betas = betas
        self.alphas = 1 - betas
        self.ac = torch.cumprod(self.alphas, 0)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(a_t) x_0 + sqrt(1-a_t) eps."""
        a = self.ac[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model, x0, caption_ids, generator=None, t=None):
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (B,), generator=generator)
        noise = torch.randn(x0.shape, generator=generator)
        pred = model(self.q_sample(x0, t, noise), t, caption_ids)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, caption_ids, null_ids, steps=50, guidance=3.0, generator=None,
               progress=None):
        """DDIM (eta=0) reverse process with classifier-free guidance."""
        model.eval()
        B = caption_ids.shape[0]
        r = model.cfg.resolution
        x = torch.randn((B, 3, r, r), generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps).long()
        for i, t in enumerate(ts):
            tt = torch.full((B,), int(t), dtype=torch.long)
            if guidance and guidance != 1.0:
                eps = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([caption_ids, null_ids]))
                e_c, e_u = eps.chunk(2)
                eps = e_u + guidance * (e_c - e_u)
            else:
                eps = model(x, tt, caption_ids)
            a = self.ac[t]
            a_prev = self.ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            if progress:
                progress(i + 1, steps)
        return x.clamp(-1, 1)
