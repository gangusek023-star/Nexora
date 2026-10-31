"""Cooperative control object shared between a running job and the job manager.

Training loops call `control.tick()` every step. That single call handles
pause (blocks until resumed), cancellation, time limits and resource limits,
so no training loop can run unbounded.
"""
from __future__ import annotations

import os
import shutil
import threading
import time

import psutil


class StopTraining(Exception):
    """Raised inside a training loop to stop it gracefully (after a checkpoint)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class JobControl:
    def __init__(self, max_minutes: float = 60, max_ram_mb: float = 3500,
                 min_free_disk_mb: float = 1500, disk_path: str = ".",
                 on_progress=None):
        self._pause = threading.Event()
        self._cancel = threading.Event()
        self.paused_flag = False
        self.max_seconds = max_minutes * 60 if max_minutes else None
        self.max_ram_mb = max_ram_mb
        self.min_free_disk_mb = min_free_disk_mb
        self.disk_path = disk_path
        self.on_progress = on_progress
        self.started = time.time()
        self.paused_seconds = 0.0
        self._proc = psutil.Process(os.getpid())
        self._last_resource_check = 0.0
        self.save_requested = threading.Event()  # set on pause so the loop checkpoints

    # ------------------------------------------------ called by the manager
    def pause(self):
        self._pause.set()
        self.save_requested.set()

    def resume(self):
        self._pause.clear()

    def cancel(self):
        self._cancel.set()
        self._pause.clear()

    @property
    def is_paused(self):
        return self._pause.is_set()

    @property
    def is_cancelled(self):
        return self._cancel.is_set()

    # ------------------------------------------------ called by the job
    def report(self, **progress):
        if self.on_progress:
            self.on_progress(progress)

    def elapsed(self) -> float:
        return time.time() - self.started - self.paused_seconds

    def ram_mb(self) -> float:
        return self._proc.memory_info().rss / 1e6

    def free_disk_mb(self) -> float:
        return shutil.disk_usage(self.disk_path).free / 1e6

    def check_disk(self, needed_mb: float = 0):
        free = self.free_disk_mb()
        if free - needed_mb < self.min_free_disk_mb:
            raise StopTraining(f"disk_space: only {free:.0f} MB free "
                               f"(minimum {self.min_free_disk_mb:.0f} MB)")

    def tick(self, save_fn=None):
        """Call once per training step. May block (pause) or raise StopTraining."""
        if self.save_requested.is_set() and save_fn:
            save_fn()
            self.save_requested.clear()
        if self._pause.is_set():
            self.paused_flag = True
            self.report(state="paused")
            t0 = time.time()
            while self._pause.is_set() and not self._cancel.is_set():
                time.sleep(0.2)
            self.paused_seconds += time.time() - t0
            self.paused_flag = False
            self.report(state="running")
        if self._cancel.is_set():
            raise StopTraining("cancelled")
        if self.max_seconds and self.elapsed() > self.max_seconds:
            raise StopTraining("time_limit")
        now = time.time()
        if now - self._last_resource_check > 5:
            self._last_resource_check = now
            if self.max_ram_mb and self.ram_mb() > self.max_ram_mb:
                raise StopTraining(f"memory_limit: process uses {self.ram_mb():.0f} MB "
                                   f"(limit {self.max_ram_mb:.0f} MB)")
            self.check_disk()


class NullControl(JobControl):
    """For scripts/tests: no limits unless given."""

    def __init__(self, **kw):
        kw.setdefault("max_minutes", 0)
        kw.setdefault("max_ram_mb", 0)
        kw.setdefault("min_free_disk_mb", 0)
        super().__init__(**kw)
