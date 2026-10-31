"""Reading files people attach in chat or put in approved folders.

Supported: plain text and source code in any common language, Markdown, CSV,
JSON, HTML, PDF (text layer, via pypdf), Word .docx and OpenDocument .odt.
Everything is local; nothing is uploaded anywhere.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

CODE_EXT = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".java": "java", ".kt": "kotlin", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp",
    ".hpp": "cpp", ".cs": "csharp", ".go": "go", ".rs": "rust", ".php": "php", ".rb": "ruby",
    ".swift": "swift", ".scala": "scala", ".lua": "lua", ".r": "r", ".m": "objective-c",
    ".sql": "sql", ".sh": "bash", ".ps1": "powershell", ".bat": "batch", ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".vue": "vue", ".dart": "dart", ".pl": "perl", ".hs": "haskell",
    ".xml": "xml", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini", ".json": "json",
    ".gd": "gdscript", ".luau": "luau", ".asm": "assembly", ".v": "verilog", ".f90": "fortran",
}
TEXT_EXT = {".txt", ".md", ".csv", ".log", ".rst", ".tex", ".srt"}
DOC_EXT = {".pdf", ".docx", ".odt"}
SUPPORTED = set(CODE_EXT) | TEXT_EXT | DOC_EXT


class FileError(Exception):
    pass


def language_of(name: str) -> str | None:
    return CODE_EXT.get(Path(name).suffix.lower())


def decode_text(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise FileError("this looks like a binary file, not text")
    for enc in ("utf-8-sig", "cp1250", "latin-1"):  # cp1250 = Polish Windows files
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _docx(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
        paras.append(_unxml(t))
    return "\n".join(paras)


def _odt(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("content.xml").decode("utf-8", "replace")
    paras = re.findall(r"<text:(?:p|h)[^>]*>(.*?)</text:(?:p|h)>", xml, re.S)
    return "\n".join(_unxml(re.sub(r"<[^>]+>", "", p)) for p in paras)


def _unxml(s: str) -> str:
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
             .replace("&apos;", "'").replace("&amp;", "&"))


def _pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise FileError("PDF support needs the 'pypdf' package (run setup.ps1 again)") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages[:300])
    except Exception as e:  # noqa: BLE001
        raise FileError(f"could not read this PDF: {e}") from e
    if not text.strip():
        raise FileError("this PDF has no text layer (it is probably a scan)")
    return text


def extract_text(name: str, data: bytes) -> dict:
    """Return {"name", "kind": code|text|document, "language", "text"}."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise FileError(f"unsupported file type {ext or '(none)'}; supported: code, text, PDF, DOCX, ODT")
    if ext == ".pdf":
        text, kind = _pdf(data), "document"
    elif ext == ".docx":
        try:
            text = _docx(data)
        except (zipfile.BadZipFile, KeyError) as e:
            raise FileError("this .docx file is damaged") from e
        kind = "document"
    elif ext == ".odt":
        try:
            text = _odt(data)
        except (zipfile.BadZipFile, KeyError) as e:
            raise FileError("this .odt file is damaged") from e
        kind = "document"
    else:
        text = decode_text(data)
        kind = "code" if ext in CODE_EXT else "text"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return {"name": name, "kind": kind, "language": CODE_EXT.get(ext), "text": text}


def validate_code(text: str, min_chars=40, max_chars=200_000) -> tuple[bool, str]:
    """Quality gate for source code used as training data."""
    if not text or len(text.strip()) < min_chars:
        return False, "too short"
    if len(text) > max_chars:
        return False, "too long"
    lines = text.split("\n")
    if max(len(x) for x in lines) > 1000 or sum(len(x) for x in lines) / max(len(lines), 1) > 200:
        return False, "looks minified or generated (very long lines)"
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    if printable < 0.97:
        return False, "too many non-printable characters"
    if re.search(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}", text):
        return False, "contains what looks like a password or key - not used for training"
    return True, "ok"


def code_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Split a source file at blank lines into chunks of whole blocks (functions, classes)."""
    blocks = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for b in blocks:
        if len(cur) + len(b) + 2 > max_chars and cur:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{b}" if cur else b
        while len(cur) > max_chars:  # a single huge block
            chunks.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur.strip():
        chunks.append(cur)
    return [c.strip("\n") for c in chunks if c.strip()]
