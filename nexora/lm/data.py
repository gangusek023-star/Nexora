"""Text data utilities: chat formatting, cleaning, splitting and batching."""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import torch

USER_TAG = "YOU:\n"
BOT_TAG = "NEXORA:\n"
TURN_SEP = "\n\n"


def format_exchange(user: str, answer: str) -> str:
    """One training example in the same dialogue layout the chat uses."""
    return f"{USER_TAG}{user.strip()}{TURN_SEP}{BOT_TAG}{answer.strip()}{TURN_SEP}"


def chat_prompt(history: list[tuple[str, str]], message: str) -> str:
    parts = [f"{USER_TAG}{u.strip()}{TURN_SEP}{BOT_TAG}{a.strip()}{TURN_SEP}" for u, a in history]
    parts.append(f"{USER_TAG}{message.strip()}{TURN_SEP}{BOT_TAG}")
    return "".join(parts)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fingerprint(text: str) -> str:
    """Whitespace/case-insensitive hash used for de-duplication."""
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def split_paragraphs(text: str, min_chars: int = 20) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= min_chars]


def shingles(text: str, n: int = 8) -> set[int]:
    """Hashed word n-grams, used to detect overlap with held-out eval data."""
    w = re.findall(r"\w+", text.lower())

    def h(s: str) -> int:  # stable across processes (unlike built-in hash)
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
    if len(w) < n:
        return {h(" ".join(w))} if w else set()
    return {h(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def validate_text(text: str, min_chars=20, max_chars=200_000) -> tuple[bool, str]:
    """Basic quality gate for a candidate training document."""
    if not text or len(text.strip()) < min_chars:
        return False, "too short"
    if len(text) > max_chars:
        return False, "too long"
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    if printable < 0.97:
        return False, "too many non-printable characters"
    letters = sum(ch.isalpha() for ch in text) / len(text)
    if letters < 0.4:
        return False, "does not look like natural-language text"
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 20 and len(set(words)) / len(words) < 0.15:
        return False, "highly repetitive"
    return True, "ok"


def tokens_to_array(ids: list[int]) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.uint16 if max(ids, default=0) < 65535 else np.int32)
    return arr


class BatchSampler:
    """Random fixed-length windows from one or more token arrays.

    `sources` is a list of (array, weight). Replay buffers are mixed in this
    way so that fine-tuning keeps seeing old data (anti-forgetting)."""

    def __init__(self, sources, block_size: int, batch_size: int, seed: int = 0):
        self.sources = [(a, w) for a, w in sources if len(a) > block_size + 1 and w > 0]
        if not self.sources:
            raise ValueError("Not enough training tokens for the configured block_size")
        tot = sum(w for _, w in self.sources)
        self.probs = np.array([w / tot for _, w in self.sources])
        self.block, self.batch = block_size, batch_size
        self.rng = np.random.default_rng(seed)

    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, s):
        self.rng.bit_generator.state = s

    def next(self):
        which = self.rng.choice(len(self.sources), size=self.batch, p=self.probs)
        xs, ys = [], []
        for s in which:
            arr = self.sources[s][0]
            i = int(self.rng.integers(0, len(arr) - self.block - 1))
            chunk = torch.from_numpy(arr[i:i + self.block + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs), torch.stack(ys)
