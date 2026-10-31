"""Runs learning cycles on a configurable schedule (off by default)."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import psutil

log = logging.getLogger("nexora.scheduler")


def in_window(spec: str, hour: int) -> bool:
    """'22-7' -> 22:00..06:59 (wraps midnight); '0-24' -> always."""
    try:
        a, b = (int(x) for x in spec.split("-"))
    except ValueError:
        return True
    if a == b or (a == 0 and b == 24):
        return True
    return a <= hour < b if a < b else (hour >= a or hour < b)


class Scheduler:
    def __init__(self, app, poll_seconds: int = 60):
        self.app = app
        self.poll = poll_seconds
        self.last_run = 0.0
        self.last_result = None
        self._stop = threading.Event()
        self._t = None

    def due(self, now: float | None = None) -> tuple[bool, str]:
        s = self.app.cfg()["schedule"]
        now = now or time.time()
        if not s["enabled"]:
            return False, "schedule disabled"
        if not in_window(s["allowed_hours"], datetime.fromtimestamp(now).hour):
            return False, "outside allowed hours"
        if now - self.last_run < s["interval_minutes"] * 60:
            return False, "interval not reached"
        cpu = psutil.cpu_percent(interval=1)
        if cpu > s["max_cpu_percent_to_start"]:
            return False, f"computer busy (CPU {cpu:.0f}%)"
        return True, "due"

    def start(self):
        self._t = threading.Thread(target=self._loop, name="nexora-scheduler", daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.poll):
            try:
                ok, why = self.due()
                if ok:
                    self.last_run = time.time()
                    self.last_result = self.app.learning.run_cycle(actor="scheduler")
            except Exception:  # noqa: BLE001
                log.exception("scheduled learning cycle failed")
