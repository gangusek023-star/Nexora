"""Start the NEXORA server:  python -m nexora  [--port 8765] [--host 127.0.0.1]"""
import argparse
import os
import sys

from .config import Paths, load_config


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(prog="nexora")
    ap.add_argument("--host", default=cfg["server"]["host"])
    ap.add_argument("--port", type=int, default=cfg["server"]["port"])
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding to {args.host} exposes NEXORA to your network. "
              "There is no login; only do this on a trusted network.", file=sys.stderr)
    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn is not installed. Run setup.ps1 first.")
    from .api.app import create_app
    pid_file = Paths().root / "nexora.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"NEXORA AI running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    finally:
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
