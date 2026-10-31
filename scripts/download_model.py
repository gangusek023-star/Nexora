"""Download a ready-made model for NEXORA's built-in engine (resumable).

    python scripts/download_model.py bielik-1.5b qwen-coder-3b
    python scripts/download_model.py --list
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nexora.core import Nexora  # noqa: E402
from nexora.jobs.control import NullControl  # noqa: E402
from nexora.localmodels import CATALOG  # noqa: E402


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--list":
        for mid, m in CATALOG.items():
            print(f"{mid:16} {m['size_gb']:>5} GB  {m['name']}  [{m['license']}]")
        return
    app = Nexora(start_workers=False)
    last = [""]

    def show(p):
        msg = p.get("message")
        if msg and msg != last[0]:
            last[0] = msg
            print("\r  " + msg + " " * 10, end="", flush=True)
    for mid in args:
        if mid not in CATALOG:
            sys.exit(f"Unknown model {mid}. Use --list.")
        print(f"{CATALOG[mid]['name']} ({CATALOG[mid]['size_gb']} GB, licence: {CATALOG[mid]['license']})")
        try:
            res = app.llm.download_handler()({"id": 0, "params": {"id": mid}}, NullControl(on_progress=show))
        except KeyboardInterrupt:
            sys.exit("\nStopped. Run the same command again to continue the download.")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"\nERROR: {e}")
        print(f"\n  done: {res.get('message') or 'downloaded'}")
    app.shutdown()


if __name__ == "__main__":
    main()
