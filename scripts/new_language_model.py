"""Build a NEXORA language model that speaks another language (e.g. Polish).

    python scripts/new_language_model.py --lang pl              # Polish only
    python scripts/new_language_model.py --lang pl --also en    # Polish + English
    python scripts/new_language_model.py --lang de --max-mb 4
    python scripts/new_language_model.py --lang code             # a model that writes code-like text

What it does:
  1. downloads public-domain books for the language(s) into data/raw/<lang>/
  2. moves the current runtime folder to runtime_backup_<date> (nothing deleted)
     and starts a new one that KEEPS your memory, chats, feedback, examples and
     image model
  3. trains a new tokenizer and a new model from random weights on that text
     (about 20-60 minutes on a laptop CPU; Ctrl+C stops safely, and
     `python scripts/train_lm.py` continues)

Stop NEXORA (stop.ps1) before running this.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nexora.config import Paths, save_settings  # noqa: E402
from nexora.learning.corpus import LANG_NAMES, download, fresh_language_runtime, local_files, running_pid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(LANG_NAMES))
    ap.add_argument("--also", nargs="*", default=[], choices=sorted(LANG_NAMES))
    ap.add_argument("--max-mb", type=float, default=6.0, help="max MB of text to download per language")
    ap.add_argument("--vocab", type=int, help="tokenizer size (default: 1024, or 2048 for several languages)")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    a = ap.parse_args()
    langs = [a.lang] + [x for x in a.also if x != a.lang]
    paths = Paths()
    if running_pid(paths.root):
        sys.exit("NEXORA seems to be running. Stop it first:  powershell -ExecutionPolicy Bypass -File .\\stop.ps1")
    if shutil.disk_usage(paths.root).free < 1.5e9:
        sys.exit("ERROR: need at least 1.5 GB of free disk space.")

    files = []
    for lang in langs:
        have = local_files(lang)
        if have:
            print(f"{LANG_NAMES[lang]}: using {len(have)} file(s) already in data/raw/{lang}")
        else:
            print(f"{LANG_NAMES[lang]}: downloading public-domain texts ...")
            try:
                have = download(lang, a.max_mb)
            except Exception as e:  # noqa: BLE001
                sys.exit(f"Download failed ({e}). Check your internet connection, or put UTF-8 .txt files "
                         f"into data/raw/{lang}/ yourself and run this again.")
        if not have:
            sys.exit(f"No texts found for {lang}. Put UTF-8 .txt files into data/raw/{lang}/ and run again.")
        files += have
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"Corpus: {len(files)} files, {size:.1f} MB")

    if not a.yes:
        ans = input("This replaces the active LANGUAGE model (your current one is kept in a backup "
                    "folder; memory, chats and images are kept). Continue? [y/N] ")
        if ans.strip().lower() not in ("y", "yes", "t", "tak"):
            sys.exit("Cancelled.")
    backup = fresh_language_runtime(paths.root)
    print(f"Previous runtime saved in: {backup}")
    vocab = a.vocab or (2048 if len(langs) > 1 else 1024)
    changes = {"lm": {"vocab_size": vocab}}
    if a.lang in ("pl", "en", "de", "es", "uk"):
        changes["ui"] = {"language": a.lang}
    save_settings(changes, Paths())

    from nexora.core import Nexora
    from nexora.learning.base import prepare_base
    app = Nexora(start_workers=False)
    # Your own texts (/learn, corrections, prompts, web pages, files) join the base training right away
    own = app.db.query("SELECT id, text FROM examples WHERE status='accepted' AND origin != 'generated'")
    if own:
        mine = paths.root / "your_examples.txt"
        mine.write_text("\n\n".join(e["text"] for e in own), encoding="utf-8")
        files = [mine] + files  # first: the last 10% of the corpus becomes validation/held-out
        print(f"Including {len(own)} of your own examples in the base training")
    info = prepare_base(app, files)
    if own:
        app.db.execute(f"UPDATE examples SET dataset_id='{info['dataset_id']}' WHERE status='accepted' "
                       "AND origin != 'generated'")
    print("Prepared:", info)
    app.shutdown()
    if a.no_train:
        print("Now train it:  python scripts/train_lm.py   (or 'Train from random weights' in the web app)")
        return
    sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts" / "train_lm.py")]))


if __name__ == "__main__":
    main()
