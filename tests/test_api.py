"""HTTP API endpoints (FastAPI TestClient)."""
import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app


@pytest.fixture
def client(trained_app):
    with TestClient(create_app(trained_app)) as c:
        yield c


def test_index_and_status(client):
    assert "NEXORA" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    s = client.get("/api/status").json()
    assert s["lm"]["ready"] and s["lm"]["version"] == "v0001" and s["base_prepared"]


def test_chat_feedback_and_memory_command(client):
    r = client.post("/api/chat", json={"message": "Good morrow, friend", "max_new_tokens": 10}).json()
    assert r["kind"] == "model" and r["version"] == "v0001"
    cid = r["conversation_id"]
    msgs = client.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    fb = client.post("/api/feedback", json={"message_id": r["message_id"], "rating": -1, "correction": "Good morrow to thee as well."})
    assert fb.status_code == 200
    m = client.post("/api/chat", json={"message": "/remember The castle is in Elsinore", "conversation_id": cid}).json()
    assert m["kind"] == "memory" and "does not change" in m["text"]
    facts = client.get("/api/memory/facts").json()
    assert facts[0]["text"] == "The castle is in Elsinore" and facts[0]["kind"] == "unverified"
    r2 = client.post("/api/chat", json={"message": "Where is the castle?", "conversation_id": cid, "max_new_tokens": 5}).json()
    assert r2["memory_recall"] and "Elsinore" in r2["memory_recall"][0]["text"]
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_memory_endpoints(client):
    f = client.post("/api/memory/facts", json={"text": "Paris is in France", "kind": "verified"}).json()
    assert not f["duplicate"]
    assert client.post("/api/memory/facts", json={"text": "paris is in france"}).json()["duplicate"]
    fid = f["fact"]["id"]
    assert client.patch(f"/api/memory/facts/{fid}", json={"kind": "preference"}).json()["kind"] == "preference"
    client.post("/api/memory/notes", json={"title": "n", "body": "buy bread tomorrow"})
    assert client.get("/api/memory/search", params={"q": "bread"}).json()["notes"]
    assert client.delete(f"/api/memory/facts/{fid}").status_code == 200
    assert client.post("/api/memory/wipe", params={"confirm": "no"}).status_code == 400
    assert client.post("/api/memory/wipe", params={"confirm": "DELETE"}).status_code == 200
    assert client.get("/api/memory/notes").json() == []


def test_training_learning_and_eval_endpoints(client):
    assert client.post("/api/lm/prepare").status_code == 409  # already prepared
    assert len(client.get("/api/models").json()) == 1
    ev = client.get("/api/evals").json()
    assert all(v == "ok" for v in ev["integrity"].values())
    r = client.post("/api/models/v0001/evaluate").json()
    assert r["heldout_base_text"]["loss"] > 0
    assert client.post("/api/models/v9999/activate").status_code == 404
    assert client.post("/api/learning/cycle").status_code == 200
    ex = client.post("/api/examples", json={"prompt": "How fares the realm?", "answer": "The realm is calm and the harvest is good."}).json()
    assert ex["status"] == "accepted"
    assert client.post("/api/examples", json={}).status_code == 400
    g = client.post("/api/goals", json={"title": "Learn about harvests", "keywords": ["harvest"]}).json()
    assert client.post(f"/api/goals/{g['id']}/status/dismissed").status_code == 200
    assert client.get("/api/jobs").status_code == 200
    assert client.post("/api/jobs/999/pause").status_code == 404
    assert client.get("/api/datasets").json()[0]["id"] == "ds0001"
    assert client.get("/api/datasets/ds0001/documents", params={"limit": 2}).status_code == 200


def test_settings_validation(client):
    assert client.put("/api/settings", json={"lm": {"no_such_key": 1}}).status_code == 400
    assert client.put("/api/settings", json={"learning": {"approved_folders": ["/definitely/missing"]}}).status_code == 400
    r = client.put("/api/settings", json={"generation": {"temperature": 0.5}}).json()
    assert r["generation"]["temperature"] == 0.5
    assert r["server"]["host"] == "127.0.0.1"


def test_hardware_audit_and_safety(client):
    h = client.get("/api/hardware").json()
    assert h["ram_total_mb"] > 0 and "models" in h["storage_mb"]
    assert client.get("/api/audit").json()
    assert client.get("/generated/..%2F..%2Fsettings.json").status_code in (400, 404)
    assert client.post("/api/images/generate", json={"prompt": "a red circle"}).status_code == 409  # no image model
    assert client.post("/api/images/train", json={"dataset": "nope"}).status_code == 404


def test_shutdown_endpoint_is_safe_in_tests(client):
    r = client.post("/api/shutdown")
    assert r.status_code == 200 and r.json()["ok"]
    assert client.get("/api/status").status_code == 200  # injected app is never killed
