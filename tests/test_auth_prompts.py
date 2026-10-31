"""Learning from user prompts / '/learn' requests, and optional Google sign-in."""
import base64
import json
import time
import urllib.parse

import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.config import save_settings


@pytest.fixture
def client(trained_app):
    with TestClient(create_app(trained_app)) as c:
        yield c


def test_learn_command_adds_training_data_not_weights(client, trained_app):
    w = trained_app.registry.active()["id"]
    r = client.post("/api/chat", json={"message": "/learn The river flows past the old mill every spring morning."}).json()
    assert r["kind"] == "learn" and r["example"]["status"] == "accepted"
    assert "only after" in r["text"]
    ex = client.get("/api/examples", params={"status": "accepted"}).json()
    assert ex[0]["text"].startswith("The river flows") and ex[0]["origin"] == "user_added"
    assert trained_app.registry.active()["id"] == w  # weights untouched
    bad = client.post("/api/chat", json={"message": "/learn hi"}).json()
    assert bad["kind"] == "learn" and "Not added" in bad["text"]
    dup = client.post("/api/chat", json={"message": "/naucz The river flows past the old mill every spring morning."}).json()
    assert "already" in dup["text"]


def test_prompts_are_collected_when_enabled(client, trained_app):
    r = client.post("/api/chat", json={"message": "Please tell me a long story about the brave knight and his horse.",
                                       "max_new_tokens": 5}).json()
    assert r["prompt_learning"] == "accepted"
    assert client.post("/api/chat", json={"message": "hi", "max_new_tokens": 3}).json()["prompt_learning"] == "skipped"
    save_settings({"learning": {"prompts_need_review": True}}, trained_app.paths); trained_app.reload_config()
    r = client.post("/api/chat", json={"message": "Could you describe the castle gardens in the summer evening light?",
                                       "max_new_tokens": 5}).json()
    assert r["prompt_learning"] == "pending"
    save_settings({"learning": {"learn_from_prompts": False}}, trained_app.paths); trained_app.reload_config()
    r = client.post("/api/chat", json={"message": "Another fairly long request about kings and their many quarrels.",
                                       "max_new_tokens": 5}).json()
    assert r["prompt_learning"] == "off"


def test_prompts_flow_into_a_learning_cycle(trained_app):
    L = trained_app.learning
    for t in ["Tell me about the stars over the sea tonight, please.",
              "Describe a quiet village in the mountains at dawn.",
              "Write a short greeting for my grandmother's birthday party.",
              "Explain why the knight left the castle before the storm came."]:
        assert L.learn_from_user_text(t, "test", explicit=False)["status"] == "accepted"
    log = L.run_cycle("test")
    assert any(g["kind"] == "prompts" for g in L.goals())
    assert "candidate training queued" in [s["msg"] for s in log["steps"]]


# ------------------------------------------------------------------ Google login
CID = "test-client.apps.googleusercontent.com"


def _id_token(**claims):
    base = {"iss": "https://accounts.google.com", "aud": CID, "exp": time.time() + 600,
            "email": "owner@gmail.com", "email_verified": True}
    base.update(claims)
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{b({'alg': 'RS256'})}.{b(base)}.sig"


def _enable(client):
    r = client.put("/api/settings", json={"auth": {"enabled": True, "google_client_id": CID,
                                                    "google_client_secret": "s3cret", "allowed_emails": ["Owner@gmail.com"]}})
    assert r.status_code == 200
    assert r.json()["auth"]["google_client_secret"] == "********"  # never sent back


def test_enabling_requires_client_and_emails(client):
    assert client.put("/api/settings", json={"auth": {"enabled": True}}).status_code == 400
    assert client.get("/api/status").status_code == 200  # still open


def _login(client, monkeypatch, **claims):
    from nexora.api import auth as auth_mod
    monkeypatch.setattr(auth_mod.GoogleAuth, "exchange_code", lambda self, code, v: {"id_token": _id_token(**claims)})
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"].startswith("https://accounts.google.com/")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(r.headers["location"]).query)
    assert q["code_challenge_method"] == ["S256"] and q["client_id"] == [CID]
    return client.get(f"/auth/callback?code=abc&state={q['state'][0]}", follow_redirects=False)


def test_google_login_flow(client, monkeypatch):
    _enable(client)
    client.cookies.clear()
    assert client.get("/api/status").status_code == 401
    assert client.get("/", follow_redirects=False).headers["location"] == "/login"
    assert client.get("/login").status_code == 200
    r = _login(client, monkeypatch)
    assert r.headers["location"] == "/" and "nexora_session" in r.cookies
    client.cookies.set("nexora_session", r.cookies["nexora_session"])
    assert client.get("/api/status").status_code == 200
    assert client.get("/auth/me").json()["email"] == "owner@gmail.com"
    client.get("/auth/logout")
    client.cookies.clear()
    assert client.get("/api/status").status_code == 401


@pytest.mark.parametrize("claims,msg", [
    ({"email": "intruder@gmail.com"}, "not on the list"),
    ({"aud": "someone-else"}, "different application"),
    ({"email_verified": False}, "not verified"),
    ({"exp": time.time() - 10}, "expired"),
    ({"iss": "https://evil.example"}, "not issued by Google"),
])
def test_google_login_rejections(client, monkeypatch, claims, msg):
    _enable(client)
    client.cookies.clear()
    r = _login(client, monkeypatch, **claims)
    assert r.headers["location"].startswith("/login?error=") and msg.replace(" ", "%20") in r.headers["location"]
    assert "nexora_session" not in r.cookies


def test_forged_or_stale_state_and_cookie(client, monkeypatch):
    _enable(client)
    client.cookies.clear()
    r = client.get("/auth/callback?code=abc&state=forged", follow_redirects=False)
    assert "expired%20or%20invalid" in r.headers["location"]
    client.cookies.set("nexora_session", "eyJlbWFpbCI6Im93bmVyQGdtYWlsLmNvbSIsImV4cCI6OTk5OTk5OTk5OX0.forged")
    assert client.get("/api/status").status_code == 401


def test_emergency_switch(client, monkeypatch):
    _enable(client)
    client.cookies.clear()
    monkeypatch.setenv("NEXORA_DISABLE_AUTH", "1")
    assert client.get("/api/status").status_code == 200
