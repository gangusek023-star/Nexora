"""Built-in engine for ready-made models (no extra program needed).

NEXORA downloads a model file (GGUF format) from Hugging Face once and runs it
inside its own process with llama.cpp (the `llama-cpp-python` package, CPU
build). Only one model is kept in memory at a time so it fits in 8 GB of RAM.

These models are NOT NEXORA's own network; they are open models made by
others (licences listed in CATALOG). They answer in Nexo 1.1's "Assistant"
mode and power Nexo 1.2.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

CATALOG = {
    "bielik-1.5b": {
        "name": "Bielik 1.5B v3 (Polish)", "role": "chat", "size_gb": 1.7, "ram_gb": 2.5,
        "repo": "speakleash/Bielik-1.5B-v3.0-Instruct-GGUF", "file": "Bielik-1.5B-v3.0-Instruct.Q8_0.gguf",
        "license": "Apache-2.0", "author": "SpeakLeash", "recommended": True,
        "about": "Polish conversation. Light and fast enough for 8 GB RAM."},
    "bielik-4.5b": {
        "name": "Bielik 4.5B v3 (Polish)", "role": "chat", "size_gb": 5.06, "ram_gb": 6.0,
        "repo": "speakleash/Bielik-4.5B-v3.0-Instruct-GGUF", "file": "Bielik-4.5B-v3.0-Instruct.Q8_0.gguf",
        "license": "Apache-2.0", "author": "SpeakLeash", "recommended": False,
        "about": "Better Polish answers, but slow and needs about 6 GB of free RAM."},
    "qwen-coder-3b": {
        "name": "Qwen2.5-Coder 3B", "role": "code", "size_gb": 2.1, "ram_gb": 3.0,
        "repo": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "file": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "license": "Qwen Research (non-commercial use only)", "author": "Alibaba Qwen", "recommended": True,
        "about": "Code in many languages (Lua/Luau, Python, JavaScript, C#…), Roblox Studio and Blender tasks, reading files."},
    "qwen-coder-1.5b": {
        "name": "Qwen2.5-Coder 1.5B", "role": "code", "size_gb": 1.12, "ram_gb": 2.0,
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "file": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "license": "Apache-2.0", "author": "Alibaba Qwen", "recommended": False,
        "about": "Lighter and faster code model; makes more mistakes."},
}
HF_URL = "https://huggingface.co/{repo}/resolve/main/{file}"


from .assistant import AssistantError


class EngineError(AssistantError):
    pass


def llama_available() -> tuple[bool, str]:
    try:
        import llama_cpp  # noqa: F401
        return True, llama_cpp.__version__
    except Exception as e:  # noqa: BLE001
        return False, str(e)


class LocalModels:
    def __init__(self, app):
        self.app = app
        self.dir = app.paths.root / "models" / "llm"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.catalog = dict(CATALOG)
        self._lock = threading.Lock()
        self._loaded: tuple[str, object] | None = None

    # ------------------------------------------------------------- files
    def path(self, mid: str) -> Path:
        if mid not in self.catalog:
            raise EngineError(f"unknown model '{mid}'")
        return self.dir / self.catalog[mid]["file"]

    def installed(self, mid: str) -> bool:
        try:
            return self.path(mid).exists()
        except EngineError:
            return False

    def list(self) -> list[dict]:
        cfg = self.app.cfg()["assistant"]
        out = []
        for mid, m in self.catalog.items():
            p = self.dir / m["file"]
            part = p.with_name(p.name + ".part")
            out.append({"id": mid, **m, "installed": p.exists(),
                        "partial_mb": round(part.stat().st_size / 1e6) if part.exists() else 0,
                        "file_mb": round(p.stat().st_size / 1e6) if p.exists() else None,
                        "selected_chat": cfg.get("chat_model") == mid, "selected_code": cfg.get("code_model") == mid})
        return out

    def delete(self, mid: str):
        with self._lock:
            if self._loaded and self._loaded[0] == mid:
                self._loaded = None
        p = self.path(mid)
        p.unlink(missing_ok=True)
        p.with_name(p.name + ".part").unlink(missing_ok=True)
        self.app.db.audit("user", "model_file_deleted", id=mid)

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        cfg = self.app.cfg()["assistant"]
        if cfg.get("backend") == "ollama":
            from . import assistant as asst
            return asst.status(cfg)
        ok, info = llama_available()
        st = {"enabled": bool(cfg.get("enabled")), "backend": "builtin", "reachable": ok,
              "installed": [mid for mid in self.catalog if self.installed(mid)],
              "chat_model_ready": self.installed(cfg.get("chat_model", "")),
              "code_model_ready": self.installed(cfg.get("code_model", ""))}
        if not ok:
            st["error"] = ("The built-in model engine (llama-cpp-python) is not installed. Run setup.ps1 again.")
        elif not st["installed"]:
            st["error"] = "No ready-made model downloaded yet. Open the AI models page and download one."
        st["loaded"] = self._loaded[0] if self._loaded else None
        return st

    def pick(self, mid: str | None) -> str | None:
        """Use the requested model if downloaded, otherwise any downloaded one of a sensible role."""
        if mid and self.installed(mid):
            return mid
        cfg = self.app.cfg()["assistant"]
        for cand in (cfg.get("code_model"), cfg.get("chat_model"), *self.catalog):
            if cand and self.installed(cand):
                return cand
        return None

    # ------------------------------------------------------------ running
    def _load(self, mid: str):
        if self._loaded and self._loaded[0] == mid:
            return self._loaded[1]
        ok, info = llama_available()
        if not ok:
            raise EngineError("The built-in model engine (llama-cpp-python) is not installed. Run setup.ps1 again.")
        if not self.installed(mid):
            raise EngineError(f"The model '{mid}' is not downloaded yet. Download it on the AI models page.")
        from llama_cpp import Llama
        self._loaded = None  # free the previous model first (RAM)
        cfg = self.app.cfg()
        threads = max(1, int(cfg["resources"].get("torch_threads", 4)))
        try:
            m = Llama(model_path=str(self.path(mid)), n_ctx=int(cfg["assistant"].get("context_tokens", 4096)),
                      n_threads=threads, n_batch=256, verbose=False)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"Could not load the model file (it may be damaged – delete and download it again): {e}") from e
        self._loaded = (mid, m)
        return m

    def stream_chat(self, model: str, messages: list[dict], temperature: float):
        """Yield text pieces. Same contract as the optional external backend."""
        cfg = self.app.cfg()["assistant"]
        if cfg.get("backend") == "ollama":
            from . import assistant as asst
            yield from asst.stream_chat(cfg, model, messages, temperature)
            return
        with self._lock:  # one generation at a time (one CPU, one model in RAM)
            try:
                llm = self._load(model)
                max_tokens = int(cfg.get("max_answer_tokens", 1024))
                stream = llm.create_chat_completion(messages=messages, temperature=float(temperature),
                                                    max_tokens=max_tokens, stream=True, repeat_penalty=1.08)
                for chunk in stream:
                    piece = chunk["choices"][0].get("delta", {}).get("content")
                    if piece:
                        yield piece
            except EngineError:
                raise
            except Exception as e:  # noqa: BLE001
                raise EngineError(f"The model stopped with an error: {e}") from e

    # ----------------------------------------------------------- download
    def download_handler(self):
        def run(job, control):
            from .jobs.control import StopTraining
            mid = job["params"]["id"]
            m = self.catalog.get(mid)
            if not m:
                raise EngineError(f"unknown model '{mid}'")
            dest = self.path(mid)
            if dest.exists():
                return {"status": "completed", "id": mid, "message": "already downloaded"}
            part = dest.with_name(dest.name + ".part")
            need = int(m["size_gb"] * 1e9)
            have = part.stat().st_size if part.exists() else 0
            free = shutil.disk_usage(self.dir).free
            if free - (need - have) < 1e9:
                raise EngineError(f"Not enough disk space: need {(need - have) / 1e9:.1f} GB plus 1 GB spare, "
                                  f"free {free / 1e9:.1f} GB.")
            url = job["params"].get("url") or HF_URL.format(**m)
            headers = {"User-Agent": "NEXORA-AI/1.0"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
            except Exception as e:  # noqa: BLE001
                raise EngineError(f"Could not start the download (check your internet connection): {e}") from e
            with resp:
                if have and resp.status != 206:  # server ignored the range: start over
                    have = 0
                total = have + int(resp.headers.get("Content-Length") or 0) or need
                t0, done_at_start = time.time(), have
                try:
                    with open(part, "ab" if have else "wb") as f:
                        while True:
                            control.tick()
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            have += len(chunk)
                            speed = (have - done_at_start) / max(time.time() - t0, 1e-3)
                            control.report(step=int(have / 1e6), max_steps=int(total / 1e6),
                                           message=f"{have / 1e9:.2f} / {total / 1e9:.2f} GB · {speed / 1e6:.1f} MB/s")
                except StopTraining as e:
                    return {"status": e.reason.split(":")[0], "reason": e.reason, "id": mid,
                            "message": "paused – resume to continue the download"}
            if total and have < total:
                raise EngineError(f"Download incomplete ({have} of {total} bytes) – resume to continue.")
            with open(part, "rb") as f:
                if f.read(4) != b"GGUF":
                    part.unlink(missing_ok=True)
                    raise EngineError("The downloaded file is not a GGUF model (the link may have changed).")
            os.replace(part, dest)
            self.app.db.audit("user", "model_file_downloaded", id=mid, bytes=have)
            return {"status": "completed", "id": mid, "mb": round(have / 1e6)}
        return run
