"""Train the base language model from random weights, from the command line.

    python scripts/train_lm.py            # start, or resume an interrupted run
    python scripts/train_lm.py --steps 500

Press Ctrl+C to stop gracefully: a checkpoint is saved and running the
command again continues from it. (You can also do all of this from the web UI.)
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
    ap.add_argument("--steps", type=int, help="override lm.max_steps")
    ap.add_argument("--minutes", type=float, help="override the time limit")
    args = ap.parse_args()
    app = Nexora()
    if not app.paths.tokenizer.exists():
        sys.exit("Run scripts/prepare_lm_data.py first.")
    old = app.db.one("SELECT id FROM jobs WHERE kind='base_train' AND status IN "
                     "('interrupted','stopped') ORDER BY id DESC")
    if old:
        jid = old["id"]
        app.jobs.resume(jid)
        print(f"Resuming job {jid} from its checkpoint")
    else:
        params = {}
        if args.steps:
            params["overrides"] = {"max_steps": args.steps}
        if args.minutes:
            params["max_minutes"] = args.minutes
        jid = app.jobs.submit("base_train", params)
        print(f"Started job {jid}")
    try:
        last = None
        while True:
            j = app.jobs.get(jid)
            p = j["progress"]
            line = (f"[{j['status']}] step {p.get('step', 0)}/{p.get('max_steps', '?')} "
                    f"train_loss {p.get('train_loss', '-')} val_loss {p.get('val_loss', '-')} "
                    f"({p.get('sec_per_step', '-')} s/step)")
            if line != last:
                print(line, flush=True)
                last = line
            if j["status"] not in ("queued", "running", "paused"):
                print("Finished:", j["status"], j["error"] or "",
                      {k: v for k, v in j["result"].items() if k != "history"})
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping gracefully (saving checkpoint)...")
        app.jobs.cancel(jid)
        while app.jobs.get(jid)["status"] in ("running", "paused"):
            time.sleep(0.5)
        app.db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (jid,))
        print("Saved. Run this command again to continue.")
    app.shutdown()


if __name__ == "__main__":
    main()
