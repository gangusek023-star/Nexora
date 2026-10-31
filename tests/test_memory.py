"""SQLite memory: persistence, duplicates, contradictions, search, deletion."""
import pytest

from nexora.memory.db import Database
from nexora.memory.store import MemoryStore, parse_fact


def store(tmp_path):
    return MemoryStore(Database(tmp_path / "m.sqlite3"))


def test_facts_persist_across_restarts(tmp_path):
    m = store(tmp_path)
    f = m.add_fact("The capital of Poland is Warsaw", kind="verified", source="atlas")["fact"]
    cid = m.new_conversation("hi")
    m.add_message(cid, "user", "hello")
    m.db.close()
    m2 = store(tmp_path)  # new connection = app restart
    got = m2.get_fact(f["id"])
    assert got["text"] == "The capital of Poland is Warsaw"
    assert got["kind"] == "verified" and got["source"] == "atlas" and got["confidence"] == 0.95
    assert m2.messages(cid)[0]["content"] == "hello"


def test_duplicate_detection(tmp_path):
    m = store(tmp_path)
    a = m.add_fact("My cat is called Mruczek")
    b = m.add_fact("my  cat is called   mruczek")
    assert not a["duplicate"] and b["duplicate"] and b["fact"]["id"] == a["fact"]["id"]
    assert len(m.list_facts()) == 1


def test_contradiction_detection(tmp_path):
    m = store(tmp_path)
    m.add_fact("My favourite colour is blue", kind="preference")
    r = m.add_fact("My favourite colour is green", kind="preference")
    assert r["conflicts"] and r["fact"]["status"] == "conflict"
    assert all(f["status"] == "conflict" for f in m.list_facts())
    r2 = m.add_fact("The sky is not green")
    m.add_fact("The sky is green")  # opposite polarity -> conflict
    assert m.get_fact(r2["fact"]["id"])["status"] == "conflict"
    ok = m.add_fact("Water boils at 100 degrees")
    assert not ok["conflicts"]


def test_parse_fact():
    assert parse_fact("The king is Richard.")[:3] == ("king", "is", "richard")
    assert parse_fact("I live in Białystok")[:3] == ("user", "live in", "białystok")
    assert parse_fact("hello there")[0] is None


def test_kinds_and_validation(tmp_path):
    m = store(tmp_path)
    with pytest.raises(ValueError):
        m.add_fact("x is y", kind="certainly-true")
    g = m.add_fact("The moon is cheese", kind="generated")["fact"]
    assert g["confidence"] == 0.2
    m.update_fact(g["id"], kind="unverified")
    assert m.get_fact(g["id"])["kind"] == "unverified"


def test_search_and_delete(tmp_path):
    m = store(tmp_path)
    f = m.add_fact("Copernicus was born in Toruń")["fact"]
    m.add_note("Remember to water the tomatoes every evening", title="Garden")
    r = m.search("tomatoes garden")
    assert r["notes"] and r["notes"][0]["title"] == "Garden"
    assert m.search("Copernicus")["facts"][0]["id"] == f["id"]
    m.delete_fact(f["id"])
    assert m.get_fact(f["id"]) is None and not m.search("Copernicus")["facts"]


def test_conversation_context_feedback_and_wipe(tmp_path):
    m = store(tmp_path)
    cid = m.new_conversation()
    for i in range(4):
        m.add_message(cid, "user", f"q{i}")
        aid = m.add_message(cid, "assistant", f"a{i}", "v0001")
    assert m.recent_exchanges(cid, 2) == [("q2", "a2"), ("q3", "a3")]
    fid = m.add_feedback(aid, -1, "better answer")
    fb = m.list_feedback()[0]
    assert fb["id"] == fid and fb["prompt"] == "q3" and fb["correction"] == "better answer"
    with pytest.raises(ValueError):
        m.add_feedback(aid - 1, 1)  # a user message
    m.wipe_all()
    assert not m.conversations() and not m.list_feedback() and not m.list_facts()
