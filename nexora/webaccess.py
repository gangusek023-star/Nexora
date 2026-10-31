"""Optional internet access (off by default).

Two features, each with its own switch in Settings → Internet:
  * learn from web pages - download the text of pages you add (only from
    allowed sites, respecting robots.txt) into the training data pool.
  * search in chat       - look the message up on Wikipedia (official API,
    no key) and show the results next to the model's reply. The model does
    not read or understand them; they are shown, clearly labelled.

Nothing here runs unless you turn it on. All requests are plain HTTPS GETs
with a size limit and timeout.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser

UA = "NEXORA-AI/1.0 (personal local assistant; learning from user-approved pages)"
MAX_BYTES = 3_000_000
_robots_cache: dict[str, tuple[urllib.robotparser.RobotFileParser | None, float]] = {}


class WebError(Exception):
    pass


def http_get(url: str, timeout: int = 20, max_bytes: int = MAX_BYTES) -> tuple[bytes, str]:
    """GET a URL. Returns (body, content_type). Tests replace this function."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "pl,en;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(max_bytes + 1)
            ctype = r.headers.get("Content-Type", "")
    except Exception as e:  # noqa: BLE001
        raise WebError(f"could not download {url}: {e}") from e
    if len(body) > max_bytes:
        raise WebError(f"page larger than {max_bytes // 1_000_000} MB")
    return body, ctype


def domain_of(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").lower()


def domain_allowed(url: str, allowed: list[str]) -> bool:
    """'wikipedia.org' allows pl.wikipedia.org, en.wikipedia.org, ..."""
    host = domain_of(url)
    for d in allowed or []:
        d = d.strip().lower().lstrip(".")
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def robots_allows(url: str) -> bool:
    p = urllib.parse.urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    cached = _robots_cache.get(base)
    if not cached or time.time() - cached[1] > 3600:
        rp = urllib.robotparser.RobotFileParser()
        try:
            body, _ = http_get(base + "/robots.txt", timeout=10, max_bytes=500_000)
            rp.parse(body.decode("utf-8", "replace").splitlines())
        except WebError:
            rp = None  # no robots.txt reachable -> treat as allowed
        cached = (rp, time.time())
        _robots_cache[base] = cached
    rp = cached[0]
    return True if rp is None else rp.can_fetch(UA, url)


class _TextExtractor(HTMLParser):
    """Keeps readable text from <p>, headings, list items and quotes; drops menus,
    scripts, styles, navigation, headers and footers."""
    SKIP = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "button"}
    BLOCK = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "dd", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.inblock = 0
        self.cur: list[str] = []
        self.paras: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.inblock += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "br" and self.inblock:
            self.cur.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK and self.inblock:
            self.inblock -= 1
            if not self.inblock:
                t = re.sub(r"[ \t]+", " ", "".join(self.cur)).strip()
                if t:
                    self.paras.append(t)
                self.cur = []
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self.inblock and not self.skip:
            self.cur.append(data)


def html_to_text(html: str) -> tuple[str, list[str]]:
    p = _TextExtractor()
    p.feed(html)
    paras = [re.sub(r"\[\d+\]", "", x) for x in p.paras]  # drop [1]-style footnote marks
    return p.title.strip(), [x for x in paras if len(x) >= 40]


def fetch_page(url: str, allowed: list[str]) -> dict:
    """Download one page and return {"title", "paragraphs", "url"}."""
    if not re.match(r"^https?://", url or ""):
        raise WebError("the address must start with http:// or https://")
    if not domain_allowed(url, allowed):
        raise WebError(f"{domain_of(url)} is not on the list of allowed sites (Settings → Internet)")
    if not robots_allows(url):
        raise WebError(f"{domain_of(url)} does not allow automatic downloading of this page (robots.txt)")
    m = re.match(r"^https?://([a-z\-]+)\.wikipedia\.org/wiki/([^#?]+)", url)
    if m:  # Wikipedia: use the official API for clean plain text
        return wikipedia_article(m.group(1), urllib.parse.unquote(m.group(2)))
    body, ctype = http_get(url)
    charset = re.search(r"charset=([\w-]+)", ctype or "")
    text = body.decode(charset.group(1) if charset else "utf-8", "replace")
    if "html" in (ctype or "") or "<html" in text[:2000].lower():
        title, paras = html_to_text(text)
    elif "text/plain" in (ctype or ""):
        title, paras = url, [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 40]
    else:
        raise WebError(f"unsupported content type: {ctype or 'unknown'}")
    if not paras:
        raise WebError("no readable text found on this page")
    return {"url": url, "title": title or url, "paragraphs": paras}


def _api(lang: str, params: dict) -> dict:
    if not re.fullmatch(r"[a-z\-]{2,12}", lang):
        raise WebError("bad Wikipedia language code")
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    body, _ = http_get(f"https://{lang}.wikipedia.org/w/api.php?{q}", timeout=15)
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise WebError("unexpected answer from Wikipedia") from e


def wikipedia_search(query: str, lang: str = "pl", limit: int = 3) -> list[dict]:
    data = _api(lang, {"action": "query", "list": "search", "srsearch": query[:200],
                       "srlimit": limit, "srprop": "snippet"})
    out = []
    for r in data.get("query", {}).get("search", []):
        snippet = re.sub(r"<[^>]+>", "", r.get("snippet", ""))
        snippet = snippet.replace("&quot;", '"').replace("&amp;", "&")
        title = r["title"]
        out.append({"title": title, "snippet": snippet, "source": "Wikipedia",
                    "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"})
    return out


def wikipedia_article(lang: str, title: str) -> dict:
    data = _api(lang, {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                       "titles": title.replace("_", " ")})
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or not pages[0].get("extract"):
        raise WebError(f"Wikipedia article not found: {title}")
    text = pages[0]["extract"]
    text = re.sub(r"\n=+ .*? =+\n", "\n\n", text)  # drop section headings
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if len(p.strip()) >= 40]
    return {"url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "title": pages[0]["title"], "paragraphs": paras}
