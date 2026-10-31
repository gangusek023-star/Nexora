"""Built-in model engine (no external program): real llama.cpp with a tiny random GGUF model,
resumable downloads, model management API and Nexo 1.2 running on it."""
import json
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import wait_job
from nexora.api.app import create_app
from nexora.config import save_settings

pytest.importorskip("llama_cpp")
pytest.importorskip("gguf")
HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    p = tmp_path_factory.mktemp("gguf") / "tiny.gguf"
    subprocess.run([sys.executable, str(HERE / "tiny_gguf.py"), str(p)], check=True, capture_output=True)
    return p


def install_tiny(app, tiny, mid="tiny"):
    app.llm.catalog[mid] = {"name": "Tiny test", "role": "code", "size_gb": 0.001, "ram_gb": 0.1, "repo": "x/y",
                            "file": "tiny-test.gguf", "license": "test", "author": "test", "recommended": False,
                            "about": "test"}
    import shutil
    shutil.copy(tiny, app.llm.path(mid))
    save_settings({"assistant": {"chat_model": mid, "code_model": mid, "context_tokens": 512,
                                 "max_answer_tokens": 24}}, app.paths)
    app.reload_config()


def test_engine_generates_text(trained_app, tiny):
    install_tiny(trained_app, tiny)
    st = trained_app.llm.status()
    assert st["reachable"] and st["chat_model_ready"] and st["installed"] == ["tiny"]
    pieces = list(trained_app.llm.stream_chat("tiny", [{"role": "system", "content": "s"},
                                                         {"role": "user", "content": "Cześć"}], 0.7))
    assert pieces and sum(len(p) for p in pieces) > 0
    assert trained_app.llm.status()["loaded"] == "tiny"


def test_assistant_and_nexo12_run_on_builtin_engine(trained_app, tiny):
    install_tiny(trained_app, tiny)
    with TestClient(create_app(trained_app)) as c:
        lines = [json.loads(x) for x in c.post("/api/assistant/chat", json={"message": "Hej"}).text.strip().split("\n")]
        assert lines[0]["model"] == "tiny" and lines[-1]["done"] and lines[-1]["error"] is None
        ev = [json.loads(x) for x in c.post("/api/agent/chat", json={"message": "Napisz coś", "target": "chat"}).text.strip().split("\n")]
        assert ev[0]["model"] == "tiny" and any(e["type"] == "delta" for e in ev) and ev[-1]["error"] is None
        msgs = c.get(f"/api/conversations/{ev[0]['conversation_id']}").json()
        assert msgs[-1]["model_version"] == "nexo-1.2:tiny"


class RangeHandler(SimpleHTTPRequestHandler):
    """Static file server that supports 'Range: bytes=N-' (like Hugging Face) and can cut a transfer short."""
    cut_after = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = Path(self.directory) / self.path.lstrip("/")
        data = path.read_bytes()
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = data[start:]
        if RangeHandler.cut_after and not rng:
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:RangeHandler.cut_after])  # connection drops mid-way
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_download_resume_and_api(trained_app, tiny, tmp_path):
    srv_dir = tmp_path / "srv"; srv_dir.mkdir()
    (srv_dir / "m.gguf").write_bytes(tiny.read_bytes())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(srv_dir)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/m.gguf"
    a = trained_app
    a.llm.catalog["dl"] = {"name": "DL", "role": "chat", "size_gb": 0.001, "ram_gb": 0.1, "repo": "x", "file": "dl.gguf",
                           "license": "test", "author": "t", "recommended": False, "about": "t"}
    try:
        RangeHandler.cut_after = 50_000
        j = wait_job(a, a.jobs.submit("model_download", {"id": "dl", "url": url}))
        assert j["status"] == "failed" and "incomplete" in j["error"].lower()
        part = a.llm.path("dl").with_name("dl.gguf.part")
        assert part.exists() and part.stat().st_size == 50_000
        RangeHandler.cut_after = None
        j = wait_job(a, a.jobs.submit("model_download", {"id": "dl", "url": url}))
        assert j["status"] == "completed", j
        assert a.llm.path("dl").read_bytes() == tiny.read_bytes()  # resumed file is byte-identical
        (srv_dir / "bad.gguf").write_bytes(b"<html>not a model</html>")
        a.llm.catalog["bad"] = {**a.llm.catalog["dl"], "file": "bad.gguf"}
        j = wait_job(a, a.jobs.submit("model_download", {"id": "bad", "url": url.replace("m.gguf", "bad.gguf")}))
        assert j["status"] == "failed" and "not a GGUF" in j["error"]
        with TestClient(create_app(a)) as c:
            lst = c.get("/api/llm/models").json()
            assert lst["engine"]["installed"] and any(m["id"] == "dl" and m["installed"] for m in lst["models"])
            assert {m["id"] for m in lst["models"]} >= {"bielik-1.5b", "bielik-4.5b", "qwen-coder-3b", "qwen-coder-1.5b"}
            assert c.post("/api/llm/models/dl/use", params={"role": "chat"}).status_code == 200
            assert a.cfg()["assistant"]["chat_model"] == "dl"
            assert c.post("/api/llm/models/dl/download").json()["already"]
            assert c.post("/api/llm/models/nope/download").status_code == 404
            assert c.delete("/api/llm/models/dl").status_code == 200 and not a.llm.installed("dl")
    finally:
        srv.shutdown()


def test_catalog_points_to_real_files():
    from nexora.localmodels import CATALOG, HF_URL
    for mid, m in CATALOG.items():
        assert m["file"].endswith(".gguf") and "/" in m["repo"]
        assert HF_URL.format(**m).startswith("https://huggingface.co/") and m["license"]
