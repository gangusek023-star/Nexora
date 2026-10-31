"""Personality settings, multilingual corpus helpers and rebuilding the LM for a new language."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from conftest import small_corpus, wait_job
from nexora.api.app import create_app
from nexora.learning.corpus import clean_gutenberg, clean_wolnelektury, fresh_language_runtime
from nexora.lm.persona import STYLES, build_prompt_prefix, postprocess, sampling_for


@pytest.fixture
def client(trained_app):
    with TestClient(create_app(trained_app)) as c:
        yield c


GEN = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "max_new_tokens": 120, "repetition_penalty": 1.1}


def test_styles_change_sampling():
    assert sampling_for({"style": "balanced"}, GEN) == GEN
    assert sampling_for({"style": "precise"}, GEN)["temperature"] < GEN["temperature"]
    assert sampling_for({"style": "creative"}, GEN)["temperature"] > GEN["temperature"]
    assert sampling_for({"style": "brief"}, GEN)["max_new_tokens"] < GEN["max_new_tokens"]
    assert set(STYLES) == {"balanced", "precise", "creative", "brief", "talkative"}


def test_postprocess_limits_and_blocked_words():
    t, notes = postprocess("One. Two! Three? Four.", {"max_sentences": 2})
    assert t == "One. Two!" and notes
    t, notes = postprocess("Thou villain, thou VILLAIN!", {"blocked_words": ["villain"]})
    assert t == "Thou ***, thou ***!" and "2 blocked" in notes[0]
    t, _ = postprocess("A. B. C.", {"style": "brief"})
    assert t == "A."
    assert build_prompt_prefix({"context": "x" * 500}) == "x" * 300 + "\n\n"


def test_personality_applies_to_chat_and_preview(client, trained_app):
    r = client.put("/api/settings", json={"persona": {"name": "Hamlet", "style": "brief", "greeting": "Witaj!",
                                                       "context": "A quiet castle hall.", "blocked_words": ["the"]}})
    assert r.status_code == 200 and r.json()["persona"]["name"] == "Hamlet"
    c = client.post("/api/chat", json={"message": "Speak, good friend, what news?"}).json()
    assert c["params"]["max_new_tokens"] == 40  # style 'brief'
    assert " the " not in f" {c['text'].lower()} "
    c2 = client.post("/api/chat", json={"message": "Speak again", "temperature": 0.2}).json()
    assert c2["params"]["temperature"] == 0.2  # explicit value beats the style
    p = client.post("/api/persona/preview", json={"message": "Hello there", "persona": {"style": "creative"}}).json()
    assert p["params"]["temperature"] == 1.0
    assert client.get("/api/settings").json()["persona"]["style"] == "brief"  # preview did not save
    assert client.get("/api/persona").json()["persona"]["greeting"] == "Witaj!"


def test_personality_validation_and_teaching(client):
    assert client.put("/api/settings", json={"persona": {"style": "rude"}}).status_code == 400
    assert client.put("/api/settings", json={"persona": {"context": "x" * 301}}).status_code == 400
    assert client.put("/api/settings", json={"persona": {"name": ""}}).status_code == 400
    assert client.put("/api/settings", json={"ui": {"language": "xx"}}).status_code == 400
    assert client.put("/api/settings", json={"ui": {"language": "pl"}}).json()["ui"]["language"] == "pl"
    r = client.post("/api/persona/teach", json={"examples": [
        {"prompt": "Hello!", "answer": "Good day to you, friend. How can I help you today?"},
        {"prompt": "Bye", "answer": "Farewell, and come back whenever you wish to talk."}]}).json()
    assert r["accepted"] == 2


def test_corpus_cleaners():
    wl = "Adam Mickiewicz\n\nPan Tadeusz\n\nISBN 978-83\n\nLitwo! Ojczyzno moja! ty jesteś jak zdrowie.\n\n-----\nTa lektura, podobnie jak tysiące innych..."
    out = clean_wolnelektury(wl)
    assert "Ta lektura" not in out and "Litwo! Ojczyzno moja!" in out
    g = "header\n*** START OF THE PROJECT GUTENBERG EBOOK FAUST ***\nHabe nun, ach!\n*** END OF THE PROJECT GUTENBERG EBOOK ***\nlicence"
    assert clean_gutenberg(g) == "Habe nun, ach!"


def test_rebuild_for_new_language_keeps_memory(trained_app, tmp_path, monkeypatch):
    a = trained_app
    a.memory.add_fact("Mój kot ma na imię Mruczek", kind="preference")
    a.learning.learn_from_user_text("Opowiedz mi bajkę o smoku wawelskim i dzielnym szewczyku.", "t", explicit=True)
    a.images.prepare("synthetic_shapes", "shapes", n=60)
    root = a.paths.root
    a.shutdown()
    backup = fresh_language_runtime(root)
    assert (backup / "models" / "lm" / "v0001.pt").exists()  # old model kept in the backup
    con = sqlite3.connect(root / "db" / "nexora.sqlite3")
    assert con.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM examples WHERE status='accepted' AND dataset_id IS NULL").fetchone()[0] >= 1
    con.close()
    assert (root / "image_data" / "shapes.npz").exists() and not (root / "tokenizer").exists()
    # train a new "Polish" model on Polish text (a repeated public-domain fragment for the test)
    pl = tmp_path / "pl.txt"
    pl.write_text(("Litwo! Ojczyzno moja! ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko się dowie, "
                   "kto cię stracił. Dziś piękność twą w całej ozdobie widzę i opisuję, bo tęsknię po tobie.\n\n") * 300,
                  encoding="utf-8")
    from nexora.core import Nexora
    from nexora.learning.base import prepare_base
    b = Nexora()
    prepare_base(b, [pl], log=lambda *_: None)
    j = wait_job(b, b.jobs.submit("base_train", {}))
    assert j["status"] == "completed"
    out = b.lm.chat("Litwo!", [], max_new_tokens=30, seed=0)
    assert out["version"] == "v0001" and out["tokens"] > 0
    word = "Ojczyzno moja"
    assert len(b.lm.tokenizer.encode(word)) < len(word.encode("utf-8"))  # merges learned from Polish text
    assert b.memory.list_facts()[0]["text"] == "Mój kot ma na imię Mruczek"
    b.shutdown()
