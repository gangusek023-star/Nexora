"""Public-domain text corpora in several languages, and rebuilding the language
model for a new language while keeping memory, chats and the image model.

Sources (downloaded on YOUR computer, only when you run the script):
  pl  - Wolne Lektury (wolnelektury.pl): Polish classics in the public domain
  en  - bundled Tiny Shakespeare (+ optional Project Gutenberg books)
  de, es, uk - Project Gutenberg books marked "not copyrighted", found via the
        Gutendex catalogue API (gutendex.com)
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
import urllib.request
from pathlib import Path

from ..config import PROJECT_ROOT

UA = {"User-Agent": "NEXORA-AI corpus downloader (personal, non-commercial)"}
LANG_NAMES = {"pl": "Polski", "en": "English", "de": "Deutsch", "es": "Español", "uk": "Українська",
              "code": "Kod / Code (Python + your own code folders)"}

# Polish authors whose own works are in the public domain (died > 70 years ago).
# Fredro wrote comedies in dialogue form - useful for a chat model.
PL_AUTHORS = ["aleksander-fredro", "adam-mickiewicz", "boleslaw-prus", "henryk-sienkiewicz",
              "eliza-orzeszkowa", "stefan-zeromski", "juliusz-slowacki",
              "wladyslaw-stanislaw-reymont", "maria-konopnicka", "ignacy-krasicki"]


def corpus_dir(lang: str) -> Path:
    d = PROJECT_ROOT / "data" / "raw" / lang
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, timeout=30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def clean_wolnelektury(text: str) -> str:
    """Remove the licence footer and editorial header lines Wolne Lektury adds."""
    text = text.replace("\r\n", "\n")
    cut = text.find("\n-----\n")
    if cut != -1:
        text = text[:cut]
    lines = text.split("\n")
    # header: author, title, blank lines, ISBN line - drop the first few short lines
    while lines and (len(lines[0].strip()) < 60 and ("ISBN" in lines[0] or not lines[0].strip())):
        lines.pop(0)
    return "\n".join(lines).strip()


def clean_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n")
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if s:
        text = text[s.end():e.start() if e else None]
    return text.strip()


def download_polish(max_mb: float = 6.0, log=print) -> list[Path]:
    out, total = [], 0
    d = corpus_dir("pl")
    for author in PL_AUTHORS:
        if total > max_mb * 1e6:
            break
        try:
            books = json.loads(_get(f"https://wolnelektury.pl/api/authors/{author}/books/"))
        except Exception as e:  # noqa: BLE001
            log(f"  skip {author}: {e}")
            continue
        for b in books:
            if total > max_mb * 1e6:
                break
            slug = b.get("slug") or b.get("href", "").rstrip("/").split("/")[-1]
            if not slug or b.get("kind") not in (None, "Liryka", "Epika", "Dramat"):
                continue
            f = d / f"{slug}.txt"
            if f.exists():
                total += f.stat().st_size; out.append(f); continue
            try:
                raw = _get(f"https://wolnelektury.pl/media/book/txt/{slug}.txt").decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 - not every entry has a txt file
                continue
            txt = clean_wolnelektury(raw)
            if len(txt) < 2000:
                continue
            f.write_text(txt, encoding="utf-8")
            total += f.stat().st_size; out.append(f)
            log(f"  {slug}: {len(txt) / 1e3:.0f} kB")
            time.sleep(0.5)  # be polite to the free library
    return out


def download_gutenberg(lang: str, max_mb: float = 6.0, max_books: int = 40, log=print) -> list[Path]:
    out, total, page = [], 0, f"https://gutendex.com/books/?languages={lang}&copyright=false"
    d = corpus_dir(lang)
    while page and total < max_mb * 1e6 and len(out) < max_books:
        data = json.loads(_get(page))
        for b in data.get("results", []):
            if total > max_mb * 1e6 or len(out) >= max_books:
                break
            if b.get("copyright") or lang not in b.get("languages", []):
                continue
            url = next((u for k, u in b.get("formats", {}).items()
                        if k.startswith("text/plain") and not u.endswith(".zip")), None)
            if not url:
                continue
            f = d / f"gutenberg-{b['id']}.txt"
            if not f.exists():
                try:
                    txt = clean_gutenberg(_get(url, 60).decode("utf-8", "replace"))
                except Exception:  # noqa: BLE001
                    continue
                if len(txt) < 2000:
                    continue
                f.write_text(txt, encoding="utf-8")
                log(f"  {b.get('title', '')[:60]}: {len(txt) / 1e3:.0f} kB")
                time.sleep(1.0)
            total += f.stat().st_size; out.append(f)
        page = data.get("next")
    return out


def download(lang: str, max_mb: float = 6.0, log=print) -> list[Path]:
    if lang == "en":
        return [PROJECT_ROOT / "data" / "raw" / "tinyshakespeare.txt"]
    if lang == "pl":
        return download_polish(max_mb, log)
    if lang == "code":
        files = python_stdlib_files(max_mb)
        log(f"  using {len(files)} Python standard-library files from this computer (no download)")
        return files
    if lang in ("de", "es", "uk"):
        return download_gutenberg(lang, max_mb, log=log)
    raise ValueError(f"unsupported language {lang}")


def python_stdlib_files(max_mb: float = 6.0) -> list[Path]:
    """Source code of Python's own standard library, already on this computer
    (Python Software Foundation licence) - no download needed."""
    import sysconfig
    root = Path(sysconfig.get_paths()["stdlib"])
    skip = ("test", "tests", "idlelib", "lib2to3", "site-packages", "ensurepip", "turtledemo", "__pycache__")
    out, total = [], 0
    for f in sorted(root.rglob("*.py")):
        if any(part in skip for part in f.relative_to(root).parts):
            continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size < 2000 or size > 400_000:
            continue
        out.append(f)
        total += size
        if total > max_mb * 1e6:
            break
    return out


def local_files(lang: str) -> list[Path]:
    if lang == "code":
        own = sorted(corpus_dir("code").rglob("*"))
        return [f for f in own if f.is_file()] or python_stdlib_files()
    if lang == "en":
        return [PROJECT_ROOT / "data" / "raw" / "tinyshakespeare.txt"]
    return sorted(corpus_dir(lang).glob("*.txt"))


# ------------------------------------------------------------------ rebuild
def running_pid(runtime: Path) -> int | None:
    """PID of a live NEXORA server using this runtime folder (stale lock files are ignored)."""
    import psutil
    try:
        pid = int((Path(runtime) / "nexora.lock").read_text())
    except (OSError, ValueError):
        return None
    try:
        return pid if psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower() else None
    except psutil.Error:
        return None


LM_TABLES_RESET = ("model_versions", "datasets", "eval_sets", "jobs", "goals")


def fresh_language_runtime(runtime: Path) -> Path:
    """Move the current runtime folder to a dated backup (nothing is deleted) and
    create a new one that keeps memory, chats, feedback, examples and the image
    model, but has no language model, tokenizer, datasets or eval sets.
    Returns the backup path."""
    runtime = Path(runtime)
    backup = runtime.with_name(f"{runtime.name}_backup_{time.strftime('%Y%m%d-%H%M%S')}")
    if not runtime.exists():
        runtime.mkdir(parents=True)
        return backup
    if running_pid(runtime):
        raise RuntimeError("Stop NEXORA (stop.ps1) before rebuilding the language model.")
    shutil.move(str(runtime), str(backup))
    runtime.mkdir(parents=True)
    for keep in ("db", "models/image", "image_data", "settings.json", "secret.key"):
        src = backup / keep
        if src.is_dir():
            shutil.copytree(src, runtime / keep)
        elif src.exists():
            (runtime / keep).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, runtime / keep)
    db = runtime / "db" / "nexora.sqlite3"
    if db.exists():
        con = sqlite3.connect(db)
        for t in LM_TABLES_RESET:
            con.execute(f"DELETE FROM {t}")
        # examples stay as your data; they can be trained into the new model
        con.execute("UPDATE examples SET dataset_id=NULL, status='accepted' WHERE status IN ('accepted','eval')")
        con.execute("UPDATE feedback SET used=0")
        con.execute("INSERT INTO audit(ts, actor, action, detail) VALUES (?,?,?,?)",
                    (time.time(), "user", "language_model_reset", json.dumps({"backup": str(backup)})))
        con.commit()
        con.close()
    return backup
