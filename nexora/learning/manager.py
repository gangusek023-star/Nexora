"""Autonomous learning manager.

One learning cycle:
  1. analyse   - evaluate the active model, read user feedback
  2. gaps      - weak evaluation sets, corrected answers, negative feedback
  3. goals     - propose learning goals (stored, user may dismiss)
  4. gather    - user corrections, user-added examples, approved local folders
  5. validate  - quality checks, de-duplication, eval-leakage check
  6. dataset   - new immutable dataset version (+ a held-out slice for the goal)
  7. train     - candidate fine-tune in the background (replay + L2 anchoring)
  8-10.        - evaluate vs. active model, compare, write report
  11. approval - candidate waits for the user (or is auto-rejected if it fails)
  12. resume   - next cycle can start once the decision is made

Three different things are kept strictly apart:
  * memory fact  -> stored in SQLite, looked up; weights unchanged
  * dataset item -> queued for a future candidate; weights unchanged
  * training     -> new weights, only active after your approval
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..guard.guard import compare, evaluate_all, sanity_check
from ..lm.data import (BatchSampler, fingerprint, format_exchange, normalize_text, shingles,
                       split_paragraphs, validate_text)
from ..lm.evaluate import eval_text
from ..lm.tokenizer import BPETokenizer
from ..lm.train import train

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those i you he she we they me my your our their his her them what
which who whom how why when where do does did not no yes so than too very can could would should
will shall may might must have has had thy thou thee shall art hath doth o tell""".split())


def keywords(texts: list[str], k=6) -> list[str]:
    c = Counter(w for t in texts for w in re.findall(r"[a-zA-Z]{3,}", t.lower()) if w not in STOPWORDS)
    return [w for w, _ in c.most_common(k)]


class LearningManager:
    def __init__(self, app):
        self.app = app
        self.db = app.db
        self.last_cycle: dict | None = None

    # ------------------------------------------------------------- examples
    def add_example(self, text: str, origin: str, source: str, goal_id=None,
                    eval_shingles: set | None = None, kind: str = "text") -> dict:
        """Validate + de-duplicate a training example. Returns {status, reason}.
        kind='code' uses the source-code quality checks instead of the prose ones."""
        from ..files import validate_code
        text = normalize_text(text) if kind != "code" else text.replace("\r\n", "\n").strip("\n")
        ok, reason = validate_code(text) if kind == "code" else validate_text(text)
        fp = fingerprint(text)
        if self.db.one("SELECT id FROM examples WHERE fingerprint=?", (fp,)):
            return {"status": "duplicate", "reason": "already in the example pool"}
        if ok and eval_shingles is not None:
            sh = shingles(text)
            if sh and len(sh & eval_shingles) / len(sh) > 0.2:
                ok, reason = False, "overlaps held-out evaluation data (leakage)"
        # Model-generated text is never trusted automatically: it waits for review.
        status = "pending" if origin == "generated" else ("accepted" if ok else "rejected")
        self.db.execute("INSERT INTO examples(text,source,origin,fingerprint,status,goal_id,reason,created)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (text, source, origin, fp, status, goal_id, reason, time.time()))
        return {"status": status, "reason": reason}

    def learn_from_user_text(self, text: str, source: str, explicit: bool) -> dict:
        """Text written by the user becomes training data (never model output).

        explicit=True  -> '/learn <text>' command: the text itself is the example.
        explicit=False -> an ordinary chat prompt, collected only when the
                          'learn_from_prompts' setting is on; stored in dialogue
                          format so the model learns how you phrase requests."""
        lc = self.app.cfg()["learning"]
        if not explicit and not lc.get("learn_from_prompts", False):
            return {"status": "off", "reason": "learning from prompts is turned off in Settings"}
        doc = text.strip() if explicit else f"YOU:\n{text.strip()}\n\n"
        ok, reason = validate_text(normalize_text(doc))
        if not ok and not explicit:
            return {"status": "skipped", "reason": reason}  # e.g. "hi" - too short to be useful
        r = self.add_example(doc, "user_added", source, eval_shingles=self.app.evals.all_shingles())
        if r["status"] == "accepted" and lc.get("prompts_need_review", False):
            self.db.execute("UPDATE examples SET status='pending', reason='waiting for your review' "
                            "WHERE fingerprint=?", (fingerprint(normalize_text(doc)),))
            r = {"status": "pending", "reason": "waiting for your review in Datasets"}
        return r

    def learn_from_url(self, url: str, eval_shingles: set | None = None) -> dict:
        """Download one allowed web page and put its paragraphs into the example pool."""
        from ..webaccess import WebError, fetch_page
        ic = self.app.cfg()["internet"]
        if not ic.get("learn_from_web"):
            raise PermissionError("Learning from web pages is turned off (Settings → Internet).")
        try:
            page = fetch_page(url, ic.get("allowed_sites", []))
        except WebError as e:
            self.db.execute("INSERT OR REPLACE INTO web_pages(url,title,fetched,paragraphs,accepted,error) "
                            "VALUES (?,?,?,?,?,?)", (url, None, time.time(), 0, 0, str(e)))
            raise
        sh = eval_shingles if eval_shingles is not None else self.app.evals.all_shingles()
        stats = Counter()
        for para in page["paragraphs"][: int(ic.get("max_paragraphs_per_page", 200))]:
            stats[self.add_example(para[:4000], "web", page["url"], eval_shingles=sh)["status"]] += 1
        self.db.execute("INSERT OR REPLACE INTO web_pages(url,title,fetched,paragraphs,accepted,error) "
                        "VALUES (?,?,?,?,?,NULL)", (url, page["title"], time.time(),
                                                    len(page["paragraphs"]), stats["accepted"]))
        self.db.audit("user", "web_page_learned", url=url, **stats)
        return {"url": page["url"], "title": page["title"], "paragraphs": len(page["paragraphs"]), **stats}

    def review_example(self, eid: int, accept: bool):
        ex = self.db.one("SELECT * FROM examples WHERE id=?", (eid,))
        if not ex:
            raise KeyError(eid)
        if ex["dataset_id"]:
            raise ValueError("example is already part of an immutable dataset")
        self.db.execute("UPDATE examples SET status=?, reason=? WHERE id=?",
                        ("accepted" if accept else "rejected",
                         "approved by user" if accept else "rejected by user", eid))
        self.db.audit("user", "example_reviewed", id=eid, accept=accept)

    def delete_example(self, eid: int):
        ex = self.db.one("SELECT * FROM examples WHERE id=?", (eid,))
        if ex and ex["dataset_id"]:
            raise ValueError("example belongs to dataset %s (immutable); it can only be "
                             "excluded from future datasets" % ex["dataset_id"])
        self.db.execute("DELETE FROM examples WHERE id=?", (eid,))

    def list_examples(self, status=None, limit=300):
        if status:
            return self.db.query("SELECT * FROM examples WHERE status=? ORDER BY id DESC LIMIT ?",
                                 (status, limit))
        return self.db.query("SELECT * FROM examples ORDER BY id DESC LIMIT ?", (limit,))

    # ---------------------------------------------------------------- goals
    def goals(self, status=None):
        rows = self.db.query("SELECT * FROM goals" + (" WHERE status=?" if status else "") +
                             " ORDER BY id DESC", (status,) if status else ())
        for r in rows:
            r["keywords"] = json.loads(r["keywords"] or "[]")
        return rows

    def _propose(self, title, description, kind, kw, metric=None, baseline=None, target=None):
        open_same = self.db.one("SELECT id FROM goals WHERE kind=? AND title=? AND status IN "
                                "('proposed','active')", (kind, title))
        if open_same:
            return open_same["id"], False
        gid = self.db.execute("INSERT INTO goals(title,description,kind,keywords,metric,baseline,target,"
                              "status,created,updated) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (title, description, kind, json.dumps(kw), metric, baseline, target,
                               "active", time.time(), time.time())).lastrowid
        self.db.audit("system", "goal_proposed", id=gid, title=title)
        return gid, True

    def set_goal_status(self, gid, status):
        if status not in ("active", "dismissed", "done", "proposed"):
            raise ValueError("bad status")
        self.db.execute("UPDATE goals SET status=?, updated=? WHERE id=?", (status, time.time(), gid))
        self.db.audit("user", "goal_status", id=gid, status=status)

    def add_goal(self, title: str, description: str, kw: list[str]):
        return self._propose(title, description, "user", kw)[0]

    def data_goals(self) -> list[int]:
        """Goals for data that is waiting to be trained on (your texts, web pages)."""
        new_ids = []
        from_user = self.db.query("SELECT text FROM examples WHERE origin='user_added' AND "
                                  "status='accepted' AND dataset_id IS NULL")
        if from_user:
            kw = keywords([e["text"] for e in from_user])
            gid, new = self._propose("Learn from your prompts and /learn requests",
                                     f"{len(from_user)} texts you wrote are waiting to be trained on "
                                     f"(topics: {', '.join(kw) or 'mixed'}).", "prompts", kw)
            if new:
                new_ids.append(gid)
        from_web = self.db.query("SELECT text FROM examples WHERE origin='web' AND "
                                 "status='accepted' AND dataset_id IS NULL")
        if from_web:
            kw = keywords([e["text"] for e in from_web])
            gid, new = self._propose("Learn from the web pages you added",
                                     f"{len(from_web)} paragraphs from web pages are waiting to be trained on "
                                     f"(topics: {', '.join(kw) or 'mixed'}).", "web", kw)
            if new:
                new_ids.append(gid)
        return new_ids

    # -------------------------------------------------------------- analyse
    def analyse(self) -> dict:
        app, lc = self.app, self.app.cfg()["learning"]
        active = app.registry.active()
        out = {"weak_sets": [], "feedback": {}, "new_goals": []}
        if not active:
            return out
        # weak performance: eval sets where the active model is worse than the
        # best loss any approved version ever reached (relative threshold)
        best: dict[str, float] = {}
        for v in app.registry.list():
            if v["status"] in ("active", "approved"):
                for k, m in v["metrics"].items():
                    if not k.startswith("_") and isinstance(m, dict):
                        best[k] = min(best.get(k, 1e9), m["loss"])
        cur = {k: m for k, m in active["metrics"].items() if not k.startswith("_")}
        if cur:
            losses = {k: m["loss"] for k, m in cur.items()}
            for k, l in losses.items():
                if l > best.get(k, l) * (1 + lc["weak_loss_threshold"]):
                    out["weak_sets"].append({"set": k, "loss": l, "best": best.get(k)})
        # feedback
        fb = self.db.query("SELECT * FROM feedback WHERE used=0")
        neg = [f for f in fb if f["rating"] < 0]
        corr = [f for f in fb if f["correction"]]
        out["feedback"] = {"unused": len(fb), "negative": len(neg), "with_correction": len(corr)}
        if corr:
            kw = keywords([f["prompt"] for f in corr])
            gid, new = self._propose("Learn from your corrected answers",
                                     f"{len(corr)} answers were corrected by you. Train on the "
                                     f"corrected versions (topics: {', '.join(kw) or 'mixed'}).",
                                     "corrections", kw)
            if new:
                out["new_goals"].append(gid)
        out["new_goals"] += self.data_goals()
        if len(neg) >= lc["negative_feedback_trigger"]:
            kw = keywords([f["prompt"] for f in neg])
            gid, new = self._propose(f"Improve answers about: {', '.join(kw[:3]) or 'general'}",
                                     f"{len(neg)} answers were rated negatively. Looking for relevant "
                                     f"text in approved folders.", "topic", kw)
            if new:
                out["new_goals"].append(gid)
        for w in out["weak_sets"]:
            gid, new = self._propose(f"Recover performance on {w['set']}",
                                     f"Loss {w['loss']:.3f} is above the best seen ({w['best']}). "
                                     f"Replay base data.", "weak_set", [], metric=w["set"],
                                     baseline=w["loss"], target=w["best"])
            if new:
                out["new_goals"].append(gid)
        return out

    # --------------------------------------------------------------- gather
    def gather(self) -> dict:
        """Collect candidate examples from permitted sources only."""
        app, lc = self.app, self.app.cfg()["learning"]
        eval_sh = app.evals.all_shingles()
        stats = Counter()
        # 1. corrections written by the user (human ground truth)
        for f in self.db.query("SELECT * FROM feedback WHERE used=0 AND correction IS NOT NULL"):
            r = self.add_example(format_exchange(f["prompt"], f["correction"]), "user_correction",
                                 f"feedback #{f['id']}", eval_shingles=eval_sh)
            self.db.execute("UPDATE feedback SET used=1 WHERE id=?", (f["id"],))
            stats[f"correction_{r['status']}"] += 1
        # 2. web pages you listed (only when internet learning is on)
        ic = app.cfg().get("internet", {})
        if ic.get("learn_from_web"):
            for url in ic.get("web_sources", []):
                seen = self.db.one("SELECT fetched, error FROM web_pages WHERE url=?", (url,))
                if seen and not seen["error"] and time.time() - seen["fetched"] < ic.get("refetch_days", 7) * 86400:
                    continue
                try:
                    r = self.learn_from_url(url, eval_shingles=eval_sh)
                    stats["web_accepted"] += r.get("accepted", 0)
                except Exception:  # noqa: BLE001 - one bad page must not stop the cycle
                    stats["web_failed"] += 1
        # 3. approved local folders, filtered by active goal keywords
        kw = set()
        for g in self.goals("active"):
            kw |= set(g["keywords"])
        max_bytes = lc["max_file_mb"] * 1e6
        for folder in lc.get("approved_folders", []):
            root = Path(folder)
            if not root.is_dir():
                stats["missing_folder"] += 1
                continue
            from ..files import CODE_EXT, FileError, code_chunks, extract_text
            for fp in sorted(root.rglob("*")):
                ext = fp.suffix.lower()
                if ext not in CODE_EXT and ext not in (".txt", ".md", ".pdf", ".docx", ".odt"):
                    continue
                if any(part in (".git", "node_modules", "__pycache__", ".venv", "venv") for part in fp.parts):
                    continue
                try:
                    if fp.stat().st_size > max_bytes:
                        stats["file_too_large"] += 1
                        continue
                    ex = extract_text(fp.name, fp.read_bytes())
                except (OSError, FileError):
                    stats["unreadable"] += 1
                    continue
                if ex["kind"] == "code":  # source code: whole blocks, no keyword filter
                    for chunk in code_chunks(ex["text"]):
                        r = self.add_example(chunk, "local_file", str(fp), eval_shingles=eval_sh, kind="code")
                        stats[f"code_{r['status']}"] += 1
                    continue
                text = ex["text"]
                for para in split_paragraphs(normalize_text(text), min_chars=40):
                    if kw and not (set(re.findall(r"[a-z]{3,}", para.lower())) & kw):
                        continue
                    r = self.add_example(para[:4000], "local_file", str(fp), eval_shingles=eval_sh)
                    stats[f"file_{r['status']}"] += 1
        return dict(stats)

    # -------------------------------------------------------------- dataset
    def build_candidate_dataset(self, min_examples=3) -> dict | None:
        """New accepted examples -> held-out slice (eval set) + new dataset version."""
        app = self.app
        ex = self.db.query("SELECT * FROM examples WHERE status='accepted' AND dataset_id IS NULL "
                           "ORDER BY id")
        if len(ex) < min_examples:
            return None
        # deterministic held-out slice: every 5th example (at least 1) is never trained on
        held = ex[::5] if len(ex) >= 5 else ex[:1]
        held_ids = {e["id"] for e in held}
        train_ex = [e for e in ex if e["id"] not in held_ids]
        tok = BPETokenizer.load(app.paths.tokenizer)
        last = app.datasets.list()
        parent = last[0]["id"] if last else None
        did = app.datasets.create([{"text": e["text"], "source": e["source"], "origin": e["origin"]}
                                   for e in train_ex], tok, parent=parent,
                                  note=f"{len(train_ex)} new examples")
        name = f"goal_{did}_heldout"
        items = []
        for e in held:
            if "\nNEXORA:\n" in e["text"]:
                p, a = e["text"].split("\nNEXORA:\n", 1)
                items.append({"prompt": p + "\nNEXORA:\n", "answer": a.strip()})
            else:
                words = e["text"].split()
                cut = max(3, len(words) // 3)
                items.append({"prompt": " ".join(words[:cut]) + " ", "answer": " ".join(words[cut:])})
        app.evals.create(name, "heldout", "qa", items,
                         f"Held-out slice of the new examples in {did} (target of the learning goal)")
        for e in train_ex:
            self.db.execute("UPDATE examples SET dataset_id=? WHERE id=?", (did, e["id"]))
        for e in held:
            self.db.execute("UPDATE examples SET status='eval', dataset_id=? WHERE id=?", (name, e["id"]))
        return {"dataset_id": did, "target_set": name, "train_examples": len(train_ex),
                "heldout_examples": len(held)}

    # ---------------------------------------------------------------- cycle
    def waiting_for_decision(self) -> dict | None:
        return self.db.one("SELECT id FROM model_versions WHERE status='pending_approval'")

    def run_cycle(self, actor="scheduler") -> dict:
        app = self.app
        log = {"started": time.time(), "actor": actor, "steps": []}

        def step(msg, **kw):
            log["steps"].append({"msg": msg, **kw})

        if not app.registry.active():
            step("no active model yet - run base training first")
        elif self.waiting_for_decision():
            step("a candidate is waiting for your approval - cycle paused until you decide")
        elif app.jobs.busy():
            step("another training job is queued or running - try again later")
        else:
            a = self.analyse()
            step("analysed active model and feedback", weak_sets=len(a["weak_sets"]),
                 feedback=a["feedback"], new_goals=a["new_goals"])
            g = self.gather()
            self.data_goals()  # data that arrived during gathering (e.g. web pages)
            step("gathered and validated examples", **g)
            ds = self.build_candidate_dataset()
            weak_goal = next((g for g in self.goals("active") if g["kind"] == "weak_set"), None)
            if ds:
                step("created candidate dataset", **ds)
                jid = app.jobs.submit("candidate_train", {"dataset_id": ds["dataset_id"],
                                                          "target_sets": [ds["target_set"]]})
                step("candidate training queued", job_id=jid)
            elif weak_goal:
                jid = app.jobs.submit("candidate_train", {"dataset_id": None,
                                                          "target_sets": [weak_goal["metric"]],
                                                          "goal_id": weak_goal["id"]})
                step("replay-only candidate queued to recover a weak set", job_id=jid)
            else:
                step("not enough new validated data for a candidate yet (need 3 accepted examples)")
        log["finished"] = time.time()
        self.last_cycle = log
        self.db.audit(actor, "learning_cycle", steps=log["steps"])
        return log

    # ------------------------------------------------------- candidate job
    def candidate_handler(self):
        def run(job, control):
            app, lc = self.app, self.app.cfg()["learning"]
            p = job["params"]
            active = app.registry.active()
            if not active:
                raise RuntimeError("no active model to improve")
            tok = BPETokenizer.load(app.paths.tokenizer)
            backup = app.registry.backup_active()
            control.report(state="running", message=f"backup of {active['id']} saved", backup=backup)
            model = app.registry.load(active["id"])
            anchor = {k: v.clone() for k, v in model.state_dict().items()}
            base_ds = app.datasets.list()[-1]["id"]  # oldest = base corpus
            # replay buffer: base corpus + every earlier dataset in the lineage
            sources = []
            new_did = p.get("dataset_id")
            if new_did:
                sources.append((app.datasets.tokens(new_did), 1.0 - lc["replay_fraction"]))
                history = [d for d in app.datasets.lineage(new_did)[1:]]
            else:
                history = [d["id"] for d in app.datasets.list()]
            hist_arrs = [app.datasets.tokens(d) for d in history] or [app.datasets.tokens(base_ds)]
            w = (lc["replay_fraction"] if new_did else 1.0) / len(hist_arrs)
            sources += [(a, w) for a in hist_arrs]
            cfg = app.cfg()["lm"]
            sampler = BatchSampler(sources, model.cfg.block_size, cfg["batch_size"], seed=job["id"])
            val = np.load(app.paths.datasets / "validation.npy")
            steps = p.get("steps", lc["finetune_steps"])
            res = train(model, sampler, max_steps=steps, lr=lc["finetune_lr"],
                        weight_decay=cfg["weight_decay"], warmup=min(50, steps // 5),
                        grad_clip=cfg["grad_clip"], eval_every=max(50, steps // 4),
                        checkpoint_every=100, ckpt_path=app.paths.job_ckpt / f"job-{job['id']}.pt",
                        control=control, eval_fn=lambda m: eval_text(m, val, max_windows=16)["loss"],
                        anchor_state=anchor, anchor_l2=lc["anchor_l2"], seed=job["id"])
            if res["status"] != "completed":
                return {**res, "reason": res["status"]}
            return self.evaluate_candidate(model, tok, active, new_did, p.get("target_sets", []),
                                           res, control, job["id"])
        return run

    def evaluate_candidate(self, model, tok, active, dataset_id, target_sets, res, control, job_id):
        app, lc = self.app, self.app.cfg()["learning"]
        control.report(state="evaluating", message="comparing candidate with the active model")
        current_model = app.registry.load(active["id"])
        cur_metrics = evaluate_all(current_model, tok, app.evals)
        cand_metrics = evaluate_all(model, tok, app.evals)
        sanity = sanity_check(model, tok)
        cur_sanity = sanity_check(current_model, tok)
        cmp = compare(cur_metrics, cand_metrics, target_sets,
                      max_regression_pct=lc["max_regression_pct"],
                      min_improvement_pct=lc["min_improvement_pct"], sanity=sanity)
        metrics = {**cand_metrics, "_sanity": {"distinct2": sanity["distinct2"], "ok": sanity["ok"]},
                   "_train": {"steps": res["step"], "history": res["history"]}}
        status = "pending_approval" if cmp["passed"] else "rejected"
        vid = app.registry.register(model, parent=active["id"], status="training",
                                    dataset_id=dataset_id, metrics=metrics,
                                    note=f"candidate from job {job_id}")
        report = {"candidate": vid, "current": active["id"], "dataset": dataset_id,
                  "target_sets": target_sets, "comparison": cmp,
                  "current_metrics": cur_metrics, "candidate_metrics": cand_metrics,
                  "samples": {"current": cur_sanity["samples"], "candidate": sanity["samples"]},
                  "training": {"steps": res["step"], "final_train_loss":
                               res["history"]["train"][-1][1] if res["history"]["train"] else None},
                  "created": time.time()}
        if not cmp["passed"]:
            report["auto_action"] = ("Rejected automatically by the Learning Guard. The active "
                                     "model was never changed, so nothing needed rolling back.")
        app.registry.set_status(vid, status, report=report)
        (app.paths.reports / f"{vid}.json").write_text(json.dumps(report, indent=2, default=str),
                                                       encoding="utf-8")
        app.db.audit("guard", "candidate_evaluated", id=vid, passed=cmp["passed"],
                     reasons=cmp["reasons"])
        return {**res, "status": "completed", "candidate": vid, "passed": cmp["passed"],
                "reasons": cmp["reasons"],
                "history": {"train": res["history"]["train"][-40:], "val": res["history"]["val"]}}

    # ------------------------------------------------------------- decision
    def decide(self, vid: str, approve: bool, note: str = "") -> dict:
        app = self.app
        v = app.registry.get(vid)
        if not v or v["status"] != "pending_approval":
            raise ValueError("this model is not waiting for approval")
        if approve:
            prev = app.registry.activate(vid, actor="user", reason="candidate approved")
            for name in (v["report"] or {}).get("target_sets", []):
                ev = app.evals.get(name)
                if ev and ev["kind"] == "heldout" and name.startswith("goal_"):
                    app.evals.promote_to_regression(name)  # protect this new skill from now on
            self.db.execute("UPDATE goals SET status='done', updated=? WHERE status='active'",
                            (time.time(),))
            # keep metrics of the newly active model current
            app.registry.set_status(vid, "active", note=note or v["note"])
            result = {"activated": vid, "previous": prev}
        else:
            app.registry.set_status(vid, "rejected", note=note or "rejected by user")
            result = {"rejected": vid}
        self.db.audit("user", "candidate_decision", id=vid, approve=approve, note=note)
        return result

    def rollback(self, vid: str) -> dict:
        prev = self.app.registry.activate(vid, actor="user", reason="manual rollback")
        self.db.audit("user", "rollback", to=vid, previous=prev)
        return {"activated": vid, "previous": prev}
