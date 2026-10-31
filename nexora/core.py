"""Wires every component together. One instance per running application."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import torch

from .config import Paths, load_config
from .guard.registry import DatasetStore, EvalSuite, ModelRegistry
from .image.service import ImageService
from .jobs.manager import JobManager
from .learning.base import base_train_handler
from .learning.manager import LearningManager
from .learning.scheduler import Scheduler
from .lm.service import LanguageService
from .memory.db import Database
from .memory.store import MemoryStore
from .video import video_job_handler


class Nexora:
    def __init__(self, root=None, start_workers: bool = True):
        self.paths = Paths(root)
        self._cfg = load_config(self.paths)
        self._setup_logging()
        torch.set_num_threads(max(1, int(self._cfg["resources"]["torch_threads"])))
        self.db = Database(self.paths.db)
        self.memory = MemoryStore(self.db)
        self.registry = ModelRegistry(self.db, self.paths)
        self.datasets = DatasetStore(self.db, self.paths)
        self.evals = EvalSuite(self.db, self.paths)
        self.lm = LanguageService(self.registry, self.paths.tokenizer)
        from .agent import Agent
        from .bridge import BridgeHub
        from .knowledge import KnowledgeBase
        self.knowledge = KnowledgeBase(self.db)
        self.bridge = BridgeHub(self.paths.root / "bridge_token.txt")
        self.agent = Agent(self)
        from .localmodels import LocalModels
        self.llm = LocalModels(self)
        self.learning = LearningManager(self)
        self.images = ImageService(self)
        self.jobs = JobManager(self.db, self.paths, self.cfg, {
            "base_train": base_train_handler(self),
            "candidate_train": self.learning.candidate_handler(),
            "image_train": self.images.train_handler(),
            "video_learn": video_job_handler(self),
            "model_download": self.llm.download_handler(),
        })
        self.scheduler = Scheduler(self)
        if start_workers:
            self.jobs.start()
            self.scheduler.start()

    def cfg(self) -> dict:
        return self._cfg

    def reload_config(self):
        self._cfg = load_config(self.paths)

    def _setup_logging(self):
        root = logging.getLogger("nexora")
        if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
            h = RotatingFileHandler(self.paths.logs / "nexora.log", maxBytes=2_000_000, backupCount=3,
                                    encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(h)
            root.setLevel(logging.INFO)

    def shutdown(self):
        self.scheduler.stop()
        self.jobs.shutdown()
        self.db.close()
