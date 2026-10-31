"""Internet features with the network replaced by fake responses (no real downloads)."""
import json

import pytest
from fastapi.testclient import TestClient

from nexora import webaccess
from nexora.api.app import create_app
from nexora.config import save_settings

PAGE = """<html><head><title>Ogród botaniczny</title><script>var x = 'ignore me';</script></head><body>
<nav><p>Menu menu menu menu menu menu menu menu menu menu</p></nav>
<h1>Ogród botaniczny</h1>
<p>Ogród botaniczny to miejsce, w którym uprawia się rośliny z całego świata w celach naukowych.[1]</p>
<p>Wiosną kwitną tu magnolie, a latem odwiedzający mogą podziwiać wielkie liście egzotycznych lilii wodnych.</p>
<footer><p>Copyright footer text that should not be learned by the model at all.</p></footer>
</body></html>"""


@pytest.fixture
def fake_net(monkeypatch):
    calls = []

    def fake_get(url, timeout=20, max_bytes=webaccess.MAX_BYTES):
        calls.append(url)
        if url.endswith("/robots.txt"):
            if "blocked.example" in url:
                return b"User-agent: *\nDisallow: /", "text/plain"
            return b"User-agent: *\nAllow: /", "text/plain"
        if "list=search" in url:
            return json.dumps({"query": {"search": [{"title": "Mikołaj Kopernik",
                                                     "snippet": "polski <span>astronom</span> &amp; matematyk"}]}}).encode(), "application/json"
        if "prop=extracts" in url:
            return json.dumps({"query": {"pages": [{"title": "Mikołaj Kopernik", "extract":
                "Mikołaj Kopernik był polskim astronomem, matematykiem i ekonomistą epoki renesansu.\n\n== Życie ==\n"
                "Urodził się w Toruniu w roku 1473 i studiował w Krakowie, Bolonii oraz Padwie."}]}}).encode(), "application/json"
        if "garden.example" in url or "blocked.example" in url:
            return PAGE.encode(), "text/html; charset=utf-8"
        raise webaccess.WebError("could not download " + url + ": offline")
    monkeypatch.setattr(webaccess, "http_get", fake_get)
    webaccess._robots_cache.clear()
    return calls


@pytest.fixture
def client(trained_app):
    with TestClient(create_app(trained_app)) as c:
        yield c


def test_html_extraction_keeps_content_drops_chrome():
    title, paras = webaccess.html_to_text(PAGE)
    assert title == "Ogród botaniczny"
    assert len(paras) == 2 and "[1]" not in paras[0]
    assert not any("Menu" in p or "Copyright" in p or "ignore" in p for p in paras)


def test_domain_rules():
    assert webaccess.domain_allowed("https://pl.wikipedia.org/wiki/X", ["wikipedia.org"])
    assert not webaccess.domain_allowed("https://wikipedia.org.evil.com/x", ["wikipedia.org"])
    assert not webaccess.domain_allowed("https://example.com/", [])


def test_web_learning_is_off_by_default(client, fake_net):
    r = client.post("/api/web/learn", json={"url": "https://garden.example/ogrod"})
    assert r.status_code == 403 and not fake_net  # nothing was downloaded
    assert client.post("/api/chat", json={"message": "Kopernik", "max_new_tokens": 3}).json()["web_results"] == []
    assert not fake_net


def test_learn_from_allowed_page(client, trained_app, fake_net):
    save_settings({"internet": {"learn_from_web": True, "allowed_sites": ["garden.example", "wikipedia.org"]}},
                  trained_app.paths); trained_app.reload_config()
    r = client.post("/api/web/learn", json={"url": "https://garden.example/ogrod"}).json()
    assert r["accepted"] == 2 and r["title"] == "Ogród botaniczny"
    ex = client.get("/api/examples", params={"status": "accepted"}).json()
    assert {e["origin"] for e in ex} == {"web"} and ex[0]["source"] == "https://garden.example/ogrod"
    assert client.post("/api/web/learn", json={"url": "https://other.example/x"}).status_code == 400
    blocked = client.post("/api/web/learn", json={"url": "https://blocked.example/x"})
    save_settings({"internet": {"allowed_sites": ["garden.example", "blocked.example"]}}, trained_app.paths)
    trained_app.reload_config()
    blocked = client.post("/api/web/learn", json={"url": "https://blocked.example/x"})
    assert blocked.status_code == 400 and "robots.txt" in blocked.json()["detail"]
    pages = client.get("/api/web/pages").json()
    assert any(p["error"] for p in pages) and any(p["accepted"] == 2 for p in pages)
    # Wikipedia pages go through the official API (clean text, headings removed)
    save_settings({"internet": {"allowed_sites": ["wikipedia.org"]}}, trained_app.paths); trained_app.reload_config()
    w = client.post("/api/web/learn", json={"url": "https://pl.wikipedia.org/wiki/Miko%C5%82aj_Kopernik"}).json()
    assert w["accepted"] == 2 and w["title"] == "Mikołaj Kopernik"


def test_web_sources_feed_the_learning_cycle(trained_app, fake_net):
    save_settings({"internet": {"learn_from_web": True, "allowed_sites": ["garden.example"],
                                "web_sources": ["https://garden.example/a"]}}, trained_app.paths)
    trained_app.reload_config()
    trained_app.learning.add_example("A third example about gardens and the plants growing in them in May.",
                                     "user_added", "t")
    log = trained_app.learning.run_cycle("test")
    gathered = [s for s in log["steps"] if s["msg"] == "gathered and validated examples"][0]
    assert gathered["web_accepted"] == 2
    assert any(g["kind"] == "web" for g in trained_app.learning.goals())
    assert "candidate training queued" in [s["msg"] for s in log["steps"]]
    n = len(fake_net)
    trained_app.learning.gather()
    assert len(fake_net) == n  # not downloaded again within refetch_days


def test_search_in_chat(client, trained_app, fake_net):
    save_settings({"internet": {"search_in_chat": True}}, trained_app.paths); trained_app.reload_config()
    r = client.post("/api/chat", json={"message": "Kim był Kopernik?", "max_new_tokens": 5}).json()
    assert r["web_results"][0]["title"] == "Mikołaj Kopernik"
    assert r["web_results"][0]["snippet"] == "polski astronom & matematyk"
    assert r["web_results"][0]["url"].startswith("https://pl.wikipedia.org/wiki/")
    assert "pl.wikipedia.org" in fake_net[-1]
    msgs = client.get(f"/api/conversations/{r['conversation_id']}").json()
    assert json.loads(msgs[-1]["meta"])["web_results"]  # stored with the answer


def test_settings_validation_for_internet(client):
    assert client.put("/api/settings", json={"internet": {"web_sources": ["ftp://x"]}}).status_code == 400
    assert client.put("/api/settings", json={"internet": {"web_sources": ["https://not-allowed.example/"]}}).status_code == 400
    assert client.put("/api/settings", json={"internet": {"wikipedia_language": "pl; drop"}}).status_code == 400


def test_old_database_is_migrated(tmp_path):
    import sqlite3
    from nexora.memory.db import Database
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE examples (id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL, "
                "origin TEXT NOT NULL CHECK(origin IN ('user_correction','user_added','local_file','generated')), "
                "fingerprint TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'pending', goal_id INTEGER, "
                "dataset_id TEXT, reason TEXT, created REAL)")
    con.execute("INSERT INTO examples(text,source,origin,fingerprint,status) VALUES ('t','s','user_added','f','accepted')")
    con.commit(); con.close()
    d = Database(db)
    d.execute("INSERT INTO examples(text,source,origin,fingerprint,status) VALUES ('w','u','web','g','accepted')")
    assert d.one("SELECT COUNT(*) n FROM examples")["n"] == 2
