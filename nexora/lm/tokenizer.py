"""Byte-level BPE tokenizer, trained from scratch on our own corpus.

* Base vocabulary = the 256 byte values, so ANY text (any language, emoji,
  typos) can be encoded - nothing is ever "unknown".
* Merges are learned by counting adjacent symbol pairs in the corpus and
  repeatedly merging the most frequent pair (classic BPE).
* One special token, <|eot|>, separates documents / conversation turns.

The tokenizer is frozen once a model has been trained with it: changing the
vocabulary would invalidate the model's embedding table.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Pre-tokenisation: split text into word-ish chunks so merges never cross
# word boundaries (keeps the vocabulary meaningful and training fast).
_PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-zÀ-ɏ]+| ?[0-9]{1,3}"
    r"| ?[^\sA-Za-z0-9À-ɏ]+|\s+(?!\S)|\s+"
)

EOT = "<|eot|>"


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(self.merges)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + 1  # + <|eot|>

    # --------------------------------------------------------------- training
    @classmethod
    def train(cls, text: str, vocab_size: int, progress=None) -> "BPETokenizer":
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 (256 bytes + eot)")
        n_merges = vocab_size - 257
        word_freq = Counter(_PAT.findall(text))
        words = [list(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())

        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, sym in enumerate(words):
            f = freqs[wi]
            for p in zip(sym, sym[1:]):
                pair_counts[p] += f
                where[p].add(wi)

        merges: list[tuple[int, int]] = []
        for m in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            if pair_counts[best] < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for wi in list(where.get(best, ())):
                sym, f = words[wi], freqs[wi]
                # remove this word's old pairs
                for p in zip(sym, sym[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                # merge
                out, i = [], 0
                while i < len(sym):
                    if i + 1 < len(sym) and sym[i] == best[0] and sym[i + 1] == best[1]:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(sym[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
            where.pop(best, None)
            if progress and m % 50 == 0:
                progress(m, n_merges)
        return cls(merges)

    # --------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        sym = list(chunk.encode("utf-8"))
        while len(sym) > 1:
            best_rank, best_i = None, -1
            for i in range(len(sym) - 1):
                r = self.ranks.get((sym[i], sym[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            sym[best_i:best_i + 2] = [256 + best_rank]
        if len(self._cache) < 200_000:
            self._cache[chunk] = sym
        return sym

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        ids: list[int] = []
        parts = text.split(EOT) if allow_special else [text]
        for n, part in enumerate(parts):
            if n > 0:
                ids.append(self.eot_id)
            for chunk in _PAT.findall(part):
                ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i == self.eot_id:
                out += EOT.encode()
            elif i in self.vocab:
                out += self.vocab[i]
        return out.decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        return self.vocab.get(int(i), b"")

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "nexora-bpe", "version": 1,
                                    "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "nexora-bpe":
            raise ValueError(f"{path} is not a NEXORA tokenizer file")
        return cls([tuple(m) for m in data["merges"]])
