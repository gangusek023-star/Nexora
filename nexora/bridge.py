"""Bridge between Nexo 1.2 and desktop apps (Roblox Studio plugin, Blender add-on).

The apps connect to NEXORA (not the other way round): their plugin keeps a
long-poll request open to http://127.0.0.1:<port>/api/bridge/<app>/poll and
receives commands ("run this code", "describe the scene", "render a preview"),
then posts the result back. Every request must carry the pairing token that is
built into the plugin/add-on you download from NEXORA.
"""
from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections import deque

APPS = ("roblox", "blender")
ONLINE_SECONDS = 25


class BridgeError(Exception):
    pass


class _Pending:
    def __init__(self, cmd: dict):
        self.cmd = cmd
        self.event = threading.Event()
        self.result: dict | None = None


class BridgeHub:
    def __init__(self, token_path):
        self.token_path = token_path
        self._lock = threading.Lock()
        self._queues = {a: deque() for a in APPS}
        self._wake = {a: threading.Condition(self._lock) for a in APPS}
        self._pending: dict[str, _Pending] = {}
        self.clients: dict[str, dict] = {}
        _ = self.token  # create the pairing token on first start

    # ------------------------------------------------------------ token
    @property
    def token(self) -> str:
        p = self.token_path
        if not p.exists():
            p.write_text(secrets.token_urlsafe(24), encoding="utf-8")
        return p.read_text(encoding="utf-8").strip()

    def check_token(self, token: str | None) -> bool:
        return bool(token) and secrets.compare_digest(token, self.token)

    # ----------------------------------------------------- app side
    def poll(self, app: str, info: dict | None = None, wait: float = 10.0) -> dict | None:
        if app not in APPS:
            raise BridgeError(f"unknown app {app}")
        deadline = time.time() + max(0.0, min(wait, 25.0))
        with self._lock:
            c = self.clients.setdefault(app, {})
            c["last_seen"] = time.time()
            if info:
                c["info"] = info
            while not self._queues[app]:
                left = deadline - time.time()
                if left <= 0:
                    return None
                self._wake[app].wait(left)
                self.clients[app]["last_seen"] = time.time()
            cmd = self._queues[app].popleft()
            return cmd

    def deliver(self, app: str, result: dict) -> bool:
        with self._lock:
            self.clients.setdefault(app, {})["last_seen"] = time.time()
            p = self._pending.pop(result.get("id", ""), None)
        if not p:
            return False
        p.result = result
        p.event.set()
        return True

    # --------------------------------------------------- NEXORA side
    def online(self, app: str) -> bool:
        c = self.clients.get(app) or {}
        return time.time() - c.get("last_seen", 0) < ONLINE_SECONDS

    def status(self) -> dict:
        return {a: {"online": self.online(a), "info": (self.clients.get(a) or {}).get("info"),
                    "last_seen": (self.clients.get(a) or {}).get("last_seen")} for a in APPS}

    def call(self, app: str, kind: str, timeout: float = 90.0, **payload) -> dict:
        """Send a command to the app and wait for its result."""
        if app not in APPS:
            raise BridgeError(f"unknown app {app}")
        if not self.online(app):
            raise BridgeError(f"{app} is not connected")
        cmd = {"id": uuid.uuid4().hex, "kind": kind, **payload}
        p = _Pending(cmd)
        with self._lock:
            self._pending[cmd["id"]] = p
            self._queues[app].append(cmd)
            self._wake[app].notify_all()
        if not p.event.wait(timeout):
            with self._lock:
                self._pending.pop(cmd["id"], None)
                try:
                    self._queues[app].remove(cmd)
                except ValueError:
                    pass
            raise BridgeError(f"{app} did not answer within {int(timeout)} s")
        return p.result or {}
