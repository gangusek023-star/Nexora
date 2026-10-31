"""Assistant (Ollama) with a fake local Ollama HTTP server, file reading and code learning."""
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from nexora import assistant as asst
from nexora.api.app import create_app
from nexora.config import save_settings
from nexora.files import FileError, code_chunks, extract_text, validate_code

SEEN = []


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            body = json.dumps({"models": [{"name": "bielik-test:latest"}, {"name": "coder-test:latest"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(req)
        if req["model"] == "missing-model":
            self.send_response(404); self.end_headers(); self.wfile.write(b'{"error":"model not found"}'); return
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
        reply = "```python\nprint('hej')\n```" if req["model"] == "coder-test" else "Dzień dobry! Jak mogę pomóc?"
        for i in range(0, len(reply), 6):
            self.wfile.write((json.dumps({"message": {"role": "assistant", "content": reply[i:i + 6]},
                                          "done": False}) + "\n").encode())
        self.wfile.write((json.dumps({"done": True}) + "\n").encode())


@pytest.fixture(scope="module")
def ollama():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def client(trained_app):
    with TestClient(create_app(trained_app)) as c:
        yield c


def _enable(app, url, **extra):
    save_settings({"assistant": {"enabled": True, "backend": "ollama", "ollama_url": url, "chat_model": "bielik-test",
                                 "code_model": "coder-test", **extra}}, app.paths)
    app.reload_config()


def _stream(client, body):
    lines = [json.loads(x) for x in client.post("/api/assistant/chat", json=body).text.strip().split("\n")]
    return lines[0], "".join(l.get("delta", "") for l in lines), lines[-1]


# --------------------------------------------------------------- files
def test_extract_text_formats():
    assert extract_text("a.py", b"def f():\n    return 1\n")["kind"] == "code"
    assert extract_text("a.py", b"x=1")["language"] == "python"
    assert extract_text("pl.txt", "Zażółć gęślą jaźń".encode("cp1250"))["text"] == "Zażółć gęślą jaźń"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", '<w:document><w:body><w:p><w:r><w:t>Ala ma </w:t></w:r><w:r>'
                   '<w:t>kota &amp; psa</w:t></w:r></w:p><w:p><w:r><w:t>Drugi akapit</w:t></w:r></w:p></w:body></w:document>')
    d = extract_text("doc.docx", buf.getvalue())
    assert d["text"] == "Ala ma kota & psa\nDrugi akapit" and d["kind"] == "document"
    with pytest.raises(FileError):
        extract_text("x.exe", b"MZ")
    with pytest.raises(FileError):
        extract_text("x.txt", b"\x00\x01binary")


def test_pdf_extraction():
    from pypdf import PdfWriter
    w = PdfWriter(); w.add_blank_page(100, 100)
    b = io.BytesIO(); w.write(b)
    with pytest.raises(FileError, match="no text layer"):
        extract_text("scan.pdf", b.getvalue())
    rl = pytest.importorskip("reportlab.pdfgen.canvas")
    b2 = io.BytesIO(); c = rl.Canvas(b2); c.drawString(72, 720, "Hello from a PDF file"); c.save()
    assert "Hello from a PDF file" in extract_text("doc.pdf", b2.getvalue())["text"]


def test_code_validation_and_chunks():
    good = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    assert validate_code(good)[0]
    assert not validate_code("x" * 5000)[0]  # one giant line = minified
    assert "password" in validate_code("password = 'hunter2hunter2'\n" * 3)[1]
    chunks = code_chunks("\n\n".join([f"def f{i}():\n    return {i}" for i in range(400)]), max_chars=500)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 5


def test_upload_and_learn_from_file(client):
    code = b"def greet(name):\n    return f'Hello {name}'\n\n\nclass Counter:\n    def __init__(self):\n        self.n = 0\n"
    r = client.post("/api/files", files={"file": ("tools.py", code, "text/x-python")}).json()
    assert r["kind"] == "code" and r["language"] == "python" and r["chars"] == len(code)
    learned = client.post(f"/api/files/{r['id']}/learn").json()
    assert learned.get("accepted", 0) >= 1
    ex = client.get("/api/examples", params={"status": "accepted"}).json()
    assert "def greet" in ex[0]["text"] and ex[0]["text"].startswith("def greet")  # indentation kept
    assert client.post("/api/files", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.delete(f"/api/files/{r['id']}").status_code == 200


def test_code_files_in_approved_folders(trained_app, tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "app.js").write_text("function add(a, b) {\n  return a + b;\n}\n\nexport default add;\n")
    (tmp_path / "proj" / "node_modules").mkdir()
    (tmp_path / "proj" / "node_modules" / "lib.js").write_text("function skipMe() {\n  return 'should be ignored';\n}\n")
    save_settings({"learning": {"approved_folders": [str(tmp_path / "proj")]}}, trained_app.paths)
    trained_app.reload_config()
    stats = trained_app.learning.gather()
    assert stats.get("code_accepted") == 1
    texts = [e["text"] for e in trained_app.learning.list_examples()]
    assert any("function add" in t for t in texts) and not any("skipMe" in t for t in texts)


# ----------------------------------------------------------- assistant
def test_assistant_needs_a_downloaded_model(client, trained_app):
    st = client.get("/api/assistant/status").json()
    assert st["backend"] == "builtin" and st["installed"] == [] and "AI models" in st["error"]
    assert client.post("/api/assistant/chat", json={"message": "cześć"}).status_code == 409
    save_settings({"assistant": {"enabled": False}}, trained_app.paths); trained_app.reload_config()
    assert client.post("/api/assistant/chat", json={"message": "cześć"}).status_code == 403


def test_assistant_only_local_ollama(client):
    assert client.put("/api/settings", json={"assistant": {"ollama_url": "http://evil.example:11434"}}).status_code == 400
    with pytest.raises(asst.AssistantError):
        asst.check_local_url("https://example.com")


def test_assistant_status_and_polish_chat(client, trained_app, ollama):
    _enable(trained_app, ollama)
    st = client.get("/api/assistant/status").json()
    assert st["reachable"] and st["chat_model_ready"] and st["code_model_ready"]
    SEEN.clear()
    start, text, end = _stream(client, {"message": "Cześć, jak się masz?"})
    assert start["model"] == "bielik-test" and text == "Dzień dobry! Jak mogę pomóc?"
    assert end["done"] and end["text"] == text and end["error"] is None
    sys_msg = SEEN[-1]["messages"][0]
    assert sys_msg["role"] == "system" and "po polsku" in sys_msg["content"]
    msgs = client.get(f"/api/conversations/{start['conversation_id']}").json()
    assert msgs[-1]["model_version"] == "assistant:bielik-test"
    # second turn carries the history
    _stream(client, {"message": "A co u ciebie?", "conversation_id": start["conversation_id"]})
    roles = [m["role"] for m in SEEN[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_code_questions_and_files_go_to_code_model(client, trained_app, ollama):
    _enable(trained_app, ollama)
    start, text, _ = _stream(client, {"message": "Napisz funkcję w Pythonie, która sortuje listę"})
    assert start["model"] == "coder-test" and "print('hej')" in text
    up = client.post("/api/files", files={"file": ("main.rs", b"fn main() {\n    println!(\"hi\");\n}\n", "text/plain")}).json()
    start, _, _ = _stream(client, {"message": "Co robi ten plik?", "attachment_ids": [up["id"]]})
    assert start["model"] == "coder-test"
    user_msg = SEEN[-1]["messages"][-1]["content"]
    assert "Plik: main.rs" in user_msg and "```rust" in user_msg and "println!" in user_msg


def test_persona_and_blocked_words_apply_to_assistant(client, trained_app, ollama):
    _enable(trained_app, ollama)
    client.put("/api/settings", json={"persona": {"name": "Zosia", "style": "brief", "blocked_words": ["pomóc"],
                                                   "context": "Mów jak pirat."}})
    _, _, end = _stream(client, {"message": "Hej!"})
    sysm = SEEN[-1]["messages"][0]["content"]
    assert "Zosia" in sysm and "krótko" in sysm and "pirat" in sysm
    assert "pomóc" not in end["text"] and "***" in end["text"]


def test_missing_model_and_ollama_down(client, trained_app, ollama):
    _enable(trained_app, ollama, chat_model="missing-model", auto_code_model=False)
    _, text, end = _stream(client, {"message": "Hej"})
    assert text == "" and "ollama pull missing-model" in end["error"] and end["message_id"] is None
    _enable(trained_app, "http://127.0.0.1:1")
    _, _, end = _stream(client, {"message": "Hej"})
    assert "Cannot reach Ollama" in end["error"]
    assert client.get("/api/assistant/status").json()["reachable"] is False
