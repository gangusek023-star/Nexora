"""Train (or continue training) the diffusion image model from the command line.

    python scripts/train_image.py shapes                       # new model, random weights
    python scripts/train_image.py shapes --steps 20000 --minutes 240
    python scripts/train_image.py shapes --resume-from img0001  # continue, e.g. on a faster PC

Ctrl+C stops gracefully (checkpoint saved); run the same command to continue.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nexora.core import Nexora  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--resume-from")
    a = ap.parse_args()
    app = Nexora()
    old = app.db.one("SELECT id FROM jobs WHERE kind='image_train' AND status IN ('interrupted','stopped') "
                     "ORDER BY id DESC")
    if old:
        jid = old["id"]; app.jobs.resume(jid); print(f"Resuming job {jid}")
    else:
        p = {"dataset": a.dataset}
        if a.steps: p["steps"] = a.steps
        if a.minutes: p["max_minutes"] = a.minutes
        if a.resume_from: p["resume_from"] = a.resume_from
        jid = app.jobs.submit("image_train", p); print(f"Started job {jid}")
    try:
        last = None
        while True:
            j = app.jobs.get(jid); p = j["progress"]
            line = (f"[{j['status']}] step {p.get('step', 0)}/{p.get('max_steps', '?')} "
                    f"loss {p.get('train_loss', '-')} val {p.get('val_loss', '-')} ({p.get('sec_per_step', '-')} s/step)")
            if line != last:
                print(line, flush=True); last = line
            if j["status"] not in ("queued", "running", "paused"):
                print("Finished:", j["status"], j["error"] or "", j["result"]); break
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping gracefully...")
        app.jobs.cancel(jid)
        while app.jobs.get(jid)["status"] in ("running", "paused"):
            time.sleep(0.5)
        app.db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (jid,))
    app.shutdown()


if __name__ == "__main__":
    main()
