"""Helpers shared by the Assistant and Nexo 1.2 (prompts, model routing, attachments).

The models run in NEXORA's built-in engine (localmodels.py). The functions
status() and stream_chat() below are only used when Settings → assistant.backend
is set to an external Ollama-compatible server (advanced, not needed).

Original description – an optional ready-made open model:

This is NOT NEXORA's own network. It exists because a model trained from
scratch on a laptop cannot hold a normal conversation or write working code.
It is off by default; you install Ollama and download the models yourself
(setup_assistant.ps1), and everything still runs on your computer. Answers
are labelled with the model that wrote them, and they are never used as
training data unless you explicitly accept them.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
CODE_HINTS = re.compile(
    r"```|\b(kod|kodu|code|coding|program|programu|skrypt|script|funkcj\w*|function|klas[aeęy]|class|"
    r"python|javascript|typescript|java|c\+\+|c#|csharp|golang|rust|php|ruby|sql|html|css|bash|powershell|"
    r"kotlin|swift|lua|luau|gdscript|regex|algorytm\w*|algorithm|bug|błąd|error|exception|traceback|"
    r"kompil\w*|compile|debug\w*|api|json|import|def |return)\b", re.I)


class AssistantError(Exception):
    pass


def check_local_url(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in LOCAL_HOSTS:
        raise AssistantError("The assistant address must be on this computer (127.0.0.1 / localhost).")
    return url.rstrip("/")


def _request(url: str, payload: dict | None = None, timeout: float = 10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def status(cfg: dict) -> dict:
    out = {"enabled": bool(cfg.get("enabled")), "reachable": False, "installed": [],
           "chat_model_ready": False, "code_model_ready": False}
    try:
        base = check_local_url(cfg.get("ollama_url", ""))
        with _request(base + "/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
        out["installed"] = sorted(m["name"] for m in tags.get("models", []))
        out["reachable"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = ("Ollama is not running or not installed. Install it from https://ollama.com/download "
                        f"and run setup_assistant.ps1. ({type(e).__name__})")
        return out

    def ready(name):
        if not name:
            return False
        names = set(out["installed"])
        return name in names or f"{name}:latest" in names or any(n.lower() == name.lower() for n in names)
    out["chat_model_ready"] = ready(cfg.get("chat_model"))
    out["code_model_ready"] = ready(cfg.get("code_model"))
    return out


def wants_code(message: str, attachments: list[dict]) -> bool:
    return any(a.get("kind") == "code" for a in attachments) or bool(CODE_HINTS.search(message))


def pick_model(cfg: dict, message: str, attachments: list[dict]) -> str:
    if cfg.get("auto_code_model", True) and cfg.get("code_model") and wants_code(message, attachments):
        return cfg["code_model"]
    return cfg.get("chat_model") or cfg.get("code_model")


LANG_NAMES = {"pl": "po polsku", "en": "in English", "de": "auf Deutsch", "es": "en español", "uk": "українською"}


def system_prompt(persona: dict, ui_language: str, extra: str = "") -> str:
    name = persona.get("name") or "NEXORA"
    lang = LANG_NAMES.get(ui_language, "po polsku")
    parts = [f"Nazywasz się {name}. Jesteś pomocnym asystentem działającym lokalnie na komputerze "
             f"użytkownika. Odpowiadaj {lang}, chyba że użytkownik pisze w innym języku - wtedy w jego języku.",
             "Gdy piszesz kod, umieszczaj go w blokach ``` z nazwą języka i krótko wyjaśnij, co robi. "
             "Jeśli czegoś nie wiesz, powiedz to wprost zamiast zgadywać."]
    style = persona.get("style", "balanced")
    if style == "brief":
        parts.append("Odpowiadaj bardzo krótko, najlepiej jednym-dwoma zdaniami.")
    elif style == "talkative":
        parts.append("Odpowiadaj wyczerpująco i szczegółowo.")
    elif style == "precise":
        parts.append("Bądź rzeczowy i precyzyjny.")
    elif style == "creative":
        parts.append("Możesz być kreatywny i swobodny w stylu.")
    if persona.get("context"):
        parts.append("Opis zachowania od użytkownika: " + persona["context"].strip())
    if persona.get("max_sentences"):
        parts.append(f"Nie pisz więcej niż {int(persona['max_sentences'])} zdań (bloki kodu się nie liczą).")
    if extra.strip():
        parts.append(extra.strip())
    return "\n".join(parts)


def temperature_for(persona: dict, cfg: dict) -> float:
    return {"precise": 0.2, "creative": 0.9, "talkative": 0.6}.get(persona.get("style"),
                                                                   float(cfg.get("temperature", 0.4)))


def attachment_block(att: dict, max_chars: int) -> str:
    text = att["text"]
    cut = len(text) > max_chars
    text = text[:max_chars]
    fence = att.get("language") or ""
    note = f"\n[... file truncated: showing the first {max_chars} of {len(att['text'])} characters]" if cut else ""
    return f"Plik: {att['name']}\n```{fence}\n{text}\n```{note}"


def build_messages(system: str, history: list[dict], message: str, attachments: list[dict],
                   max_file_chars: int) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    msgs += history
    content = message
    if attachments:
        content = "\n\n".join(attachment_block(a, max_file_chars) for a in attachments) + "\n\n" + message
    msgs.append({"role": "user", "content": content})
    return msgs


def stream_chat(cfg: dict, model: str, messages: list[dict], temperature: float):
    """Yield text pieces from Ollama's /api/chat stream."""
    base = check_local_url(cfg.get("ollama_url", ""))
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"temperature": temperature, "num_ctx": int(cfg.get("context_tokens", 4096))},
               "keep_alive": cfg.get("keep_alive", "5m")}
    try:
        resp = _request(base + "/api/chat", payload, timeout=float(cfg.get("timeout_seconds", 600)))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        if e.code == 404 or "not found" in body:
            raise AssistantError(f"The model '{model}' is not downloaded yet. Run setup_assistant.ps1 "
                                 f"or: ollama pull {model}") from e
        raise AssistantError(f"Ollama error {e.code}: {body[:300]}") from e
    except Exception as e:  # noqa: BLE001
        raise AssistantError("Cannot reach Ollama. Is it running? (Start the Ollama app, then try again.)") from e
    with resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("error"):
                raise AssistantError(f"Ollama: {ev['error']}")
            piece = (ev.get("message") or {}).get("content", "")
            if piece:
                yield piece
            if ev.get("done"):
                break


def blocked_filter(text: str, persona: dict) -> str:
    for w in persona.get("blocked_words", []) or []:
        if w.strip():
            text = re.sub(rf"(?i)\b{re.escape(w.strip())}\b", "***", text)
    return text

