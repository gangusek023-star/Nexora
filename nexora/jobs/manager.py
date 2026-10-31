"""Background job manager: one job at a time, persisted in SQLite.

Job lifecycle:
  queued -> running <-> paused -> completed | failed | cancelled
                               -> stopped  (time / memory / disk limit; resumable)
On application start, jobs that were running or paused become 'interrupted'
and can be resumed from their last checkpoint.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback

import psutil
import torch

from ..config import Paths
from ..memory.db import Database
from .control import JobControl

log = logging.getLogger("nexora.jobs")

RESUMABLE = ("interrupted", "stopped", "paused")
ACTIVE = ("queued", "running", "paused")


class JobManager:
    def __init__(self, db: Database, paths: Paths, cfg_getter, handlers: dict):
        self.db, self.paths = db, paths
        self.cfg = cfg_getter  # callable -> current config dict
        self.handlers = handlers
        self.current_id: int | None = None
        self.control: JobControl | None = None
        self.live: dict[int, dict] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_write = 0.0
        self.recover()

    # --------------------------------------------------------------- lifecycle
    def other_instance(self) -> int | None:
        """PID of another live NEXORA process using this runtime folder, if any."""
        lock = self.paths.root / "nexora.lock"
        try:
            pid = int(lock.read_text())
        except (OSError, ValueError):
            return None
        if pid != os.getpid() and psutil.pid_exists(pid):
            try:
                if "python" in psutil.Process(pid).name().lower():
                    return pid
            except psutil.Error:
                pass
        return None

    def recover(self):
        if self.other_instance():
            return  # that process owns the running jobs; leave them alone
        for j in self.db.query("SELECT id, status FROM jobs WHERE status IN ('running','paused')"):
            self._set(j["id"], status="interrupted",
                      error="application was closed while this job was active; resume to continue "
                            "from the last checkpoint")
            self.db.audit("system", "job_interrupted", id=j["id"])

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        other = self.other_instance()
        if other:
            raise RuntimeError(f"NEXORA is already running (process {other}) on this runtime folder. "
                               "Stop it first (stop.ps1) or use the web UI.")
        (self.paths.root / "nexora.lock").write_text(str(os.getpid()))
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nexora-jobs", daemon=True)
        self._thread.start()

    def shutdown(self, timeout=30):
        """Graceful: ask the running job to checkpoint and stop."""
        self._stop.set()
        if self.control:
            self.control.cancel()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)
            lock = self.paths.root / "nexora.lock"
            try:
                if lock.read_text() == str(os.getpid()):
                    lock.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------- API
    def submit(self, kind: str, params: dict | None = None) -> int:
        if kind not in self.handlers:
            raise ValueError(f"unknown job kind {kind}")
        now = time.time()
        jid = self.db.execute("INSERT INTO jobs(kind,status,params,progress,created,updated) "
                              "VALUES (?,?,?,?,?,?)",
                              (kind, "queued", json.dumps(params or {}), "{}", now, now)).lastrowid
        self.db.audit("system", "job_submitted", id=jid, kind=kind)
        self._wake.set()
        return jid

    def get(self, jid: int) -> dict | None:
        j = self.db.one("SELECT * FROM jobs WHERE id=?", (jid,))
        if not j:
            return None
        for k in ("params", "progress", "result"):
            j[k] = json.loads(j[k]) if j[k] else {}
        if jid in self.live:
            j["progress"] = {**j["progress"], **self.live[jid]}
        return j

    def list(self, limit=50) -> list[dict]:
        ids = [r["id"] for r in self.db.query("SELECT id FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]
        return [self.get(i) for i in ids]

    def busy(self) -> bool:
        return bool(self.db.one("SELECT id FROM jobs WHERE status IN ('queued','running','paused')"))

    def pause(self, jid: int):
        if not self.get(jid):
            raise KeyError(jid)
        if jid == self.current_id and self.control:
            self.control.pause()
            self._set(jid, status="paused")
            self.db.audit("user", "job_paused", id=jid)
        else:
            raise ValueError("only the running job can be paused")

    def resume(self, jid: int):
        j = self.get(jid)
        if not j:
            raise KeyError(jid)
        if jid == self.current_id and self.control and j["status"] == "paused":
            self.control.resume()
            self._set(jid, status="running")
        elif j["status"] in RESUMABLE:
            self._set(jid, status="queued", error=None)
            self._wake.set()
        else:
            raise ValueError(f"job is {j['status']}, cannot resume")
        self.db.audit("user", "job_resumed", id=jid)

    def cancel(self, jid: int):
        j = self.get(jid)
        if not j:
            raise KeyError(jid)
        if jid == self.current_id and self.control:
            self.control.cancel()  # loop saves a checkpoint and exits
        elif j["status"] in ("queued",) + RESUMABLE:
            self._set(jid, status="cancelled")
        else:
            raise ValueError(f"job is {j['status']}")
        self.db.audit("user", "job_cancelled", id=jid)

    # -------------------------------------------------------------- internals
    def _set(self, jid, **fields):
        sets, args = ["updated=?"], [time.time()]
        for k, v in fields.items():
            sets.append(f"{k}=?")
            args.append(json.dumps(v, default=str) if k in ("progress", "result", "params") else v)
        self.db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", (*args, jid))

    def _on_progress(self, jid):
        def cb(p: dict):
            live = self.live.setdefault(jid, {})
            live.update(p)
            live["updated"] = time.time()
            if p.get("state") == "paused":
                self._set(jid, status="paused")
            if time.time() - self._last_write > 3:
                self._last_write = time.time()
                self._set(jid, progress=live)
        return cb

    def _apply_resource_limits(self):
        res = self.cfg()["resources"]
        torch.set_num_threads(max(1, int(res.get("torch_threads", 2))))
        if res.get("lower_priority"):
            try:
                p = psutil.Process(os.getpid())
                if os.name == "nt":
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                elif p.nice() < 10:
                    p.nice(10)
            except (psutil.Error, OSError):
                pass

    def _next(self):
        return self.db.one("SELECT id FROM jobs WHERE status='queued' ORDER BY id LIMIT 1")

    def _loop(self):
        while not self._stop.is_set():
            nxt = self._next()
            if not nxt:
                self._wake.wait(2)
                self._wake.clear()
                continue
            self._run(nxt["id"])

    def _run(self, jid):
        job = self.get(jid)
        cfg = self.cfg()
        kind_limits = cfg["image"] if job["kind"].startswith("image") else cfg["lm"]
        res = cfg["resources"]
        self.control = JobControl(max_minutes=job["params"].get("max_minutes", kind_limits["max_minutes"]),
                                  max_ram_mb=res["max_process_ram_mb"],
                                  min_free_disk_mb=res["min_free_disk_mb"],
                                  disk_path=str(self.paths.root), on_progress=self._on_progress(jid))
        self.current_id = jid
        self.live[jid] = dict(job["progress"] or {}, state="running")
        self._set(jid, status="running", error=None)
        self._apply_resource_limits()
        self.db.audit("system", "job_started", id=jid, kind=job["kind"])
        try:
            self.control.check_disk()
            result = self.handlers[job["kind"]](job, self.control) or {}
            st = result.get("status", "completed")
            status = {"completed": "completed", "cancelled": "cancelled"}.get(st, "stopped")
            if self._stop.is_set() and st == "cancelled":
                status = "interrupted"  # app shutdown, not a user cancel
            self._set(jid, status=status, result=result, progress=self.live.get(jid, {}),
                      error=None if status in ("completed", "cancelled") else
                      f"stopped: {result.get('reason', st)} - resume to continue")
            self.db.audit("system", "job_finished", id=jid, status=status)
        except Exception as e:  # noqa: BLE001 - any failure must be recorded, not crash the app
            tb = traceback.format_exc()
            log.error("job %s failed: %s", jid, tb)
            (self.paths.logs / f"job-{jid}-error.log").write_text(tb, encoding="utf-8")
            self._set(jid, status="failed", error=f"{type(e).__name__}: {e}",
                      progress=self.live.get(jid, {}))
            self.db.audit("system", "job_failed", id=jid, error=str(e))
        finally:
            self.current_id = None
            self.control = None
