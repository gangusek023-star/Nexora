"""Serves the active language model for chat (thread-safe, hot-reload on approval)."""
from __future__ import annotations

import threading
import time

import torch

from .data import BOT_TAG, TURN_SEP, USER_TAG, chat_prompt
from .model import NexoraLM
from .tokenizer import BPETokenizer


class LanguageService:
    def __init__(self, registry, tokenizer_path):
        self.registry = registry
        self.tokenizer_path = tokenizer_path
        self.tokenizer: BPETokenizer | None = None
        self.model: NexoraLM | None = None
        self.version: str | None = None
        self._lock = threading.Lock()

    def ensure_loaded(self) -> bool:
        active = self.registry.active()
        if not active:
            return False
        if self.version != active["id"]:
            with self._lock:
                if self.tokenizer is None:
                    self.tokenizer = BPETokenizer.load(self.tokenizer_path)
                m = self.registry.load(active["id"])
                m.eval()
                self.model, self.version = m, active["id"]
        return True

    def status(self) -> dict:
        ok = self.ensure_loaded()
        if not ok:
            return {"ready": False, "version": None}
        return {"ready": True, "version": self.version, "params": self.model.num_params(),
                "config": self.model.cfg.to_dict()}

    def generate_raw(self, prompt: str, max_new_tokens=120, temperature=0.8, top_k=40, top_p=0.95,
                     repetition_penalty=1.1, seed=None, stop_at_turn=True) -> dict:
        if not self.ensure_loaded():
            raise RuntimeError("No trained model yet. Start the base training from the Training page.")
        tok, model = self.tokenizer, self.model
        ids = tok.encode(prompt)
        budget = model.cfg.block_size - 8
        ids = ids[-budget:]  # keep the most recent context (short-term memory window)
        x = torch.tensor([ids])
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        n0 = x.shape[1]

        def stop_fn(seq):
            if not stop_at_turn:
                return False
            tail = tok.decode(seq[0, n0:].tolist())
            return TURN_SEP in tail or USER_TAG.strip() in tail

        t0 = time.time()
        with self._lock:
            out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                                 top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                 generator=gen, stop_ids=[tok.eot_id], stop_fn=stop_fn)
        new = out[0, n0:].tolist()
        text = tok.decode(new)
        if stop_at_turn:
            for cut in (TURN_SEP, USER_TAG.strip(), "<|eot|>"):
                text = text.split(cut)[0]
        return {"text": text.strip(), "tokens": len(new), "seconds": round(time.time() - t0, 2),
                "version": self.version, "context_tokens": len(ids)}

    def chat(self, message: str, history: list[tuple[str, str]], persona: dict | None = None,
             **gen) -> dict:
        from .persona import build_prompt_prefix, postprocess
        persona = persona or {}
        prefix = build_prompt_prefix(persona)
        prompt = prefix + chat_prompt(history, message)
        if self.ensure_loaded():  # drop old turns first so the personality context survives
            budget = self.model.cfg.block_size - 8
            while history and len(self.tokenizer.encode(prompt)) > budget:
                history = history[1:]
                prompt = prefix + chat_prompt(history, message)
        out = self.generate_raw(prompt, **gen)
        out["raw_text"] = out["text"]
        out["text"], out["postprocess"] = postprocess(out["text"], persona)
        return out
