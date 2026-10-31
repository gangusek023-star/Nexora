"""Image model: dataset preparation, background training job, generation, versions."""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import torch

from ..guard.registry import absol, rel
from ..jobs.control import StopTraining
from .data import (load_dataset, load_folder, save_dataset, synthetic_shapes, to_png, to_tensor)
from .model import CaptionTokenizer, Diffusion, ImageModelConfig, UNet


class ImageService:
    def __init__(self, app):
        self.app = app
        self._cache = None  # (version, model, tokenizer, diffusion)

    # ---------------------------------------------------------- datasets
    def datasets(self) -> list[dict]:
        out = []
        for p in sorted(self.app.paths.image_data.glob("*.npz")):
            try:
                imgs, caps, meta = load_dataset(p)
                out.append({"name": p.stem, "images": int(len(imgs)), "resolution": int(imgs.shape[1]),
                            "size_mb": round(p.stat().st_size / 1e6, 2), "examples": caps[:5], **meta})
            except Exception as e:  # noqa: BLE001
                out.append({"name": p.stem, "error": str(e)})
        return out

    def prepare(self, source: str, name: str, n: int = 4000, folder: str | None = None,
                resolution: int | None = None) -> dict:
        res = resolution or self.app.cfg()["image"]["resolution"]
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("dataset name: letters, digits, - and _ only")
        if source == "synthetic_shapes":
            imgs, caps = synthetic_shapes(n, res, seed=0)
            meta = {"source": "synthetic_shapes (generated locally, no copyright)", "license": "n/a"}
        elif source == "folder":
            imgs, caps, skipped = load_folder(folder, res)
            meta = {"source": f"folder {folder}", "skipped": skipped,
                    "license": "user-provided: you are responsible for the rights to these images"}
        else:
            raise ValueError("source must be synthetic_shapes or folder")
        path = self.app.paths.image_data / f"{name}.npz"
        save_dataset(path, imgs, caps, {**meta, "created": time.time()})
        self.app.db.audit("user", "image_dataset_prepared", name=name, n=len(imgs))
        return {"name": name, "images": len(imgs), "resolution": res}

    # ---------------------------------------------------------- versions
    def versions(self):
        rows = self.app.db.query("SELECT * FROM image_versions ORDER BY created DESC")
        for r in rows:
            r["metrics"] = json.loads(r["metrics"] or "{}")
            r["config"] = json.loads(r["config"] or "{}")
        return rows

    def active(self):
        r = self.app.db.one("SELECT id FROM image_versions WHERE status='active'")
        return next((v for v in self.versions() if r and v["id"] == r["id"]), None)

    def activate(self, vid):
        if not self.app.db.one("SELECT id FROM image_versions WHERE id=?", (vid,)):
            raise KeyError(vid)
        self.app.db.execute("UPDATE image_versions SET status='available' WHERE status='active'")
        self.app.db.execute("UPDATE image_versions SET status='active' WHERE id=?", (vid,))
        self.app.db.audit("user", "image_model_activated", id=vid)

    def _save_version(self, model, ema, tok, metrics, cfg) -> str:
        rows = self.app.db.query("SELECT id FROM image_versions")
        n = max([int(r["id"][3:]) for r in rows], default=0) + 1
        vid = f"img{n:04d}"
        path = self.app.paths.image_models / f"{vid}.pt"
        tmp = str(path) + ".tmp"
        torch.save({"format": "nexora-img-v1", "config": cfg.to_dict(), "model": model.state_dict(),
                    "ema": ema.state_dict(), "tokenizer": tok.to_json()}, tmp)
        os.replace(tmp, path)
        first = self.active() is None
        self.app.db.execute("INSERT INTO image_versions(id,path,status,metrics,config,created) "
                            "VALUES (?,?,?,?,?,?)", (vid, rel(self.app.paths, path), "active" if first else "available",
                                                     json.dumps(metrics), json.dumps(cfg.to_dict()),
                                                     time.time()))
        self.app.db.audit("system", "image_model_saved", id=vid, steps=metrics.get("total_steps"))
        return vid

    def _load(self, vid):
        r = self.app.db.one("SELECT * FROM image_versions WHERE id=?", (vid,))
        if not r:
            raise KeyError(vid)
        ck = torch.load(absol(self.app.paths, r["path"]), map_location="cpu", weights_only=True)
        cfg = ImageModelConfig.from_dict(ck["config"])
        model = UNet(cfg)
        model.load_state_dict(ck["ema"])
        model.eval()
        return model, CaptionTokenizer.from_json(ck["tokenizer"]), cfg, ck

    # ---------------------------------------------------------- training
    def train_handler(self):
        def run(job, control):
            app = self.app
            p, icfg = job["params"], app.cfg()["image"]
            imgs, caps, meta = load_dataset(app.paths.image_data / f"{p['dataset']}.npz")
            n_val = max(8, len(imgs) // 20)
            x_all = to_tensor(imgs)
            x_tr, x_val, c_tr, c_val = x_all[n_val:], x_all[:n_val], caps[n_val:], caps[:n_val]
            prior_steps = 0
            if p.get("resume_from"):  # continue an existing model (e.g. on better hardware)
                model, tok, cfg, ck = self._load(p["resume_from"])
                model.load_state_dict(ck["model"]); model.train()
                prior_steps = json.loads(app.db.one("SELECT metrics FROM image_versions WHERE id=?",
                                                    (p["resume_from"],))["metrics"]).get("total_steps", 0)
            else:
                tok = CaptionTokenizer.build(c_tr)
                cfg = ImageModelConfig(resolution=int(imgs.shape[1]), base_channels=icfg["base_channels"],
                                       channel_mults=list(icfg["channel_mults"]), text_dim=icfg["text_dim"],
                                       vocab_size=tok.vocab_size, timesteps=icfg["timesteps"])
                torch.manual_seed(0)
                model = UNet(cfg)  # random initialisation
            ema = copy.deepcopy(model).eval()
            for q in ema.parameters():
                q.requires_grad_(False)
            diff = Diffusion(cfg.timesteps)
            opt = torch.optim.AdamW(model.parameters(), lr=icfg["learning_rate"], weight_decay=0.0)
            ids_tr = tok.batch(c_tr)
            ids_val = tok.batch(c_val)
            null = torch.full((1, tok.max_len), CaptionTokenizer.PAD); null[:, 0] = CaptionTokenizer.NULL
            g = torch.Generator().manual_seed(job["id"])
            step, max_steps = 0, int(p.get("steps", icfg["max_steps"]))
            hist = {"train": [], "val": []}
            ck_path = app.paths.job_ckpt / f"job-{job['id']}.pt"
            if ck_path.exists():
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
                opt.load_state_dict(ck["opt"]); step, hist = ck["step"], ck["hist"]
                g.set_state(ck["gen"])

            def save():
                control.check_disk(50)
                tmp = str(ck_path) + ".tmp"
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                            "step": step, "hist": hist, "gen": g.get_state()}, tmp)
                os.replace(tmp, ck_path)

            def val_loss():
                vg = torch.Generator().manual_seed(1234)  # fixed noise -> reproducible
                with torch.no_grad():
                    t = torch.linspace(0, cfg.timesteps - 1, len(x_val)).long()
                    return diff.loss(ema, x_val, ids_val, generator=vg, t=t).item()

            control.report(state="running", params=sum(q.numel() for q in model.parameters()),
                           message=f"{len(x_tr)} training images at {cfg.resolution}px")
            status, running, t_last = "completed", None, time.time()
            bs = icfg["batch_size"]
            try:
                while step < max_steps:
                    control.tick(save_fn=save)
                    idx = torch.randint(0, len(x_tr), (bs,), generator=g)
                    x0 = x_tr[idx]  # no flips: captions mention left/right
                    ids = ids_tr[idx].clone()
                    drop = torch.rand(bs, generator=g) < icfg["caption_dropout"]
                    ids[drop] = null  # learn the unconditional model for guidance
                    loss = diff.loss(model, x0, ids, generator=g)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    with torch.no_grad():
                        d = min(0.999, (1 + step) / (10 + step))
                        for pe, pm in zip(ema.parameters(), model.parameters()):
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    step += 1
                    running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
                    if step % 20 == 0:
                        hist["train"].append([step, round(running, 5)])
                        control.report(step=step, max_steps=max_steps, train_loss=round(running, 5),
                                       sec_per_step=round((time.time() - t_last) / 20, 3))
                        t_last = time.time()
                    if step % 250 == 0 or step == max_steps:
                        v = val_loss()
                        hist["val"].append([step, round(v, 5)])
                        control.report(val_loss=round(v, 5))
                    if step % icfg["checkpoint_every"] == 0:
                        save()
            except StopTraining as e:
                status = e.reason.split(":")[0]
                try:
                    save()
                except StopTraining:
                    pass
            if status != "completed":
                return {"status": status, "reason": status, "step": step}
            metrics = {"total_steps": prior_steps + step, "steps_this_run": step,
                       "final_train_loss": running, "val_loss": val_loss(), "dataset": p["dataset"],
                       "dataset_images": len(imgs), "history": hist, "parent": p.get("resume_from")}
            vid = self._save_version(model, ema, tok, metrics, cfg)
            return {"status": "completed", "step": step, "image_version": vid,
                    "val_loss": metrics["val_loss"]}
        return run

    # -------------------------------------------------------- generation
    def generate(self, prompt: str, n: int = 4, steps: int | None = None, guidance: float | None = None,
                 seed: int | None = None, version: str | None = None) -> dict:
        icfg = self.app.cfg()["image"]
        v = self.app.db.one("SELECT * FROM image_versions WHERE id=?", (version,)) if version else None
        v = v or self.active()
        if not v:
            raise RuntimeError("No image model trained yet. Prepare a dataset and start image training.")
        if not self._cache or self._cache[0] != v["id"]:
            model, tok, cfg, _ = self._load(v["id"])
            self._cache = (v["id"], model, tok, Diffusion(cfg.timesteps))
        _, model, tok, diff = self._cache
        n = max(1, min(int(n), 8))
        ids = tok.batch([prompt] * n)
        null = torch.full((n, tok.max_len), CaptionTokenizer.PAD); null[:, 0] = CaptionTokenizer.NULL
        unknown = [w for w in CaptionTokenizer.split(prompt) if w not in tok.index]
        g = torch.Generator().manual_seed(seed if seed is not None else int(time.time()))
        t0 = time.time()
        x = diff.sample(model, ids, null, steps=steps or icfg["sample_steps"],
                        guidance=icfg["guidance_scale"] if guidance is None else guidance, generator=g)
        files = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i in range(n):
            f = self.app.paths.generated / f"{stamp}-{v['id']}-{i}.png"
            to_png(x[i], f)
            files.append(f.name)
        metrics = json.loads(v["metrics"]) if isinstance(v["metrics"], str) else v["metrics"]
        steps_trained = metrics.get("total_steps", 0)
        warn = None
        if steps_trained < 3000:
            warn = (f"This image model has only {steps_trained} training steps. Expect blurry, "
                    "rough results; it only knows the kinds of images in its training set.")
        return {"files": files, "version": v["id"], "seconds": round(time.time() - t0, 1),
                "unknown_words": unknown, "warning": warn, "steps_trained": steps_trained,
                "vocabulary_sample": tok.words[3:40]}
