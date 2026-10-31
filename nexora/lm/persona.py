"""Personality settings: how NEXORA behaves and how it speaks.

Everything here acts on the real model, nothing is a canned reply:
  * style        -> sampling settings (how predictable or adventurous the text is)
  * context      -> text placed before the conversation, which the model continues from
  * max_sentences, blocked_words -> post-processing of what the model generated
  * name, greeting -> how the assistant is presented in the chat window
  * examples     -> "teach this personality": example exchanges added to the
                    training data, so a future approved model actually learns the style
"""
from __future__ import annotations

import re

STYLES = {
    # name: (sampling overrides, description)
    "balanced":  ({}, "Uses the sampling defaults from Settings."),
    "precise":   ({"temperature": 0.5, "top_k": 20, "top_p": 0.9, "repetition_penalty": 1.15},
                  "Safer, more repetitive word choices."),
    "creative":  ({"temperature": 1.0, "top_k": 80, "top_p": 0.98, "repetition_penalty": 1.05},
                  "More surprising, less coherent text."),
    "brief":     ({"max_new_tokens": 40}, "Short answers (one sentence)."),
    "talkative": ({"max_new_tokens": 200, "temperature": 0.85}, "Long answers."),
}
MAX_CONTEXT_CHARS = 300


def sampling_for(persona: dict, generation: dict) -> dict:
    p = {k: generation[k] for k in ("temperature", "top_k", "top_p", "max_new_tokens", "repetition_penalty")}
    p.update(STYLES.get(persona.get("style", "balanced"), STYLES["balanced"])[0])
    return p


def max_sentences(persona: dict) -> int:
    n = int(persona.get("max_sentences", 0) or 0)
    return 1 if persona.get("style") == "brief" and n == 0 else n


def build_prompt_prefix(persona: dict) -> str:
    ctx = (persona.get("context") or "").strip()[:MAX_CONTEXT_CHARS]
    return f"{ctx}\n\n" if ctx else ""


def postprocess(text: str, persona: dict) -> tuple[str, list[str]]:
    """Apply sentence limit and blocked words. Returns (text, notes)."""
    notes = []
    n = max_sentences(persona)
    if n > 0:
        parts = re.split(r"(?<=[.!?…])\s+", text.strip())
        if len(parts) > n:
            text = " ".join(parts[:n])
            notes.append(f"shortened to {n} sentence(s)")
    hits = 0
    for w in persona.get("blocked_words", []) or []:
        w = w.strip()
        if not w:
            continue
        text, k = re.subn(rf"(?i)\b{re.escape(w)}\b", "***", text)
        hits += k
    if hits:
        notes.append(f"{hits} blocked word(s) hidden")
    return text, notes
