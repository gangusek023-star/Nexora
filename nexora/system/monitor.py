"""Hardware usage snapshot (CPU, RAM, disk, process)."""
from __future__ import annotations

import os
import platform
import shutil

import psutil
import torch


def snapshot(disk_path: str) -> dict:
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(disk_path)
    proc = psutil.Process(os.getpid())
    with proc.oneshot():
        rss = proc.memory_info().rss
        threads = proc.num_threads()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "cpu_per_core": psutil.cpu_percent(interval=None, percpu=True),
        "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": round(vm.total / 1e6), "ram_used_mb": round(vm.used / 1e6),
        "ram_percent": vm.percent, "process_ram_mb": round(rss / 1e6), "process_threads": threads,
        "disk_total_gb": round(du.total / 1e9, 1), "disk_free_gb": round(du.free / 1e9, 2),
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "gpu": "none used (CPU only)", "platform": platform.platform(),
        "python": platform.python_version(),
    }


def folder_size_mb(path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 1e6, 1)
