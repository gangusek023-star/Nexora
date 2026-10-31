"""Prepare the base language corpus: train/validation/held-out split,
BPE tokenizer training and immutable evaluation sets.

Usage:
    python scripts/prepare_lm_data.py                      # bundled Tiny Shakespeare
    python scripts/prepare_lm_data.py my_texts/*.txt       # your own UTF-8 text files

Data licensing: Tiny Shakespeare (data/raw/tinyshakespeare.txt) is a ~1 MB
extract of Shakespeare's plays, which are in the public domain. If you use
your own files, make sure you have the right to use them.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nexora.config import Paths  # noqa: E402
from nexora.core import Nexora  # noqa: E402
from nexora.learning.base import prepare_base  # noqa: E402


def main():
    files = sys.argv[1:] or [str(ROOT / "data" / "raw" / "tinyshakespeare.txt")]
    for f in files:
        if not Path(f).exists():
            sys.exit(f"ERROR: file not found: {f}")
    free_mb = shutil.disk_usage(Paths().root).free / 1e6
    if free_mb < 500:
        sys.exit(f"ERROR: only {free_mb:.0f} MB free disk space; need at least 500 MB.")
    app = Nexora(start_workers=False)
    try:
        info = prepare_base(app, files)
    except RuntimeError as e:
        sys.exit(f"Nothing to do: {e}")
    print("Prepared:", info)
    app.shutdown()


if __name__ == "__main__":
    main()
