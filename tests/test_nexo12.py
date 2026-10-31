"""Nexo 1.2: knowledge base, app bridge, agent loop, Roblox plugin, Blender add-on, renders, video learning."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT, wait_job
from nexora.api.app import create_app
from nexora.config import save_settings
from nexora.knowledge import chunk_text
from nexora.render3d import render_scene


# ------------------------------------------------------------ fixtures
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class LiveServer:
    """The real NEXORA app served by uvicorn on a free port (plugins need real HTTP)."""

    def __init__(self, app_obj):
        import uvicorn
        self.port = free_port()
        cfg = uvicorn.Config(create_app(app_obj), host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(5)


SCRIPT: list = []      # replies the fake Ollama will give, in order
SEEN: list = []


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({"models": [{"name": "coder:latest"}, {"name": "chat:latest"},
                                                {"name": "eyes:latest"}]}).encode())

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(req)
        if not req.get("stream", True):  # vision call
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"message": {"content": "Widać czerwony domek z dachem na zielonej trawie."}}).encode())
            return
        reply = SCRIPT.pop(0) if SCRIPT else "Dzień dobry!"
        self.send_response(200); self.end_headers()
        for i in range(0, len(reply), 7):
            self.wfile.write((json.dumps({"message": {"content": reply[i:i + 7]}, "done": False}) + "\n").encode())
        self.wfile.write(b'{"done": true}\n')


@pytest.fixture(scope="module")
def ollama():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def enable(app, ollama_url, **agent):
    save_settings({"assistant": {"enabled": True, "backend": "ollama", "ollama_url": ollama_url, "chat_model": "chat",
                                 "code_model": "coder"},
                   "agent": {"auto_run": True, "vision_model": "", **agent}}, app.paths)
    app.reload_config()


def events(client, body):
    r = client.post("/api/agent/chat", json=body)
    assert r.status_code == 200, r.text
    return [json.loads(x) for x in r.text.strip().split("\n")]


class FakeRobloxPlugin(threading.Thread):
    """Behaves like NexoraBridge.lua: polls NEXORA, 'runs' code, returns snapshots."""

    def __init__(self, url, token, fail_first=False):
        super().__init__(daemon=True)
        self.url, self.token, self.fail_first = url, token, fail_first
        self.ran, self.running = [], True
        self.parts = []

    def req(self, method, path, body=None):
        r = urllib.request.Request(self.url + path, method=method, data=json.dumps(body).encode() if body else None,
                                   headers={"Content-Type": "application/json", "X-Nexora-Token": self.token})
        with urllib.request.urlopen(r, timeout=30) as resp:
            return json.loads(resp.read())

    def run(self):
        while self.running:
            try:
                cmd = self.req("GET", "/api/bridge/roblox/poll?wait=1&place=Test")["command"]
            except Exception:  # noqa: BLE001
                time.sleep(0.1); continue
            if not cmd:
                continue
            if cmd["kind"] == "run":
                self.ran.append(cmd["code"])
                if self.fail_first and len(self.ran) == 1:
                    res = {"ok": False, "error": "Nexo:3: attempt to index nil with 'Size'", "output": ""}
                else:
                    self.parts.append({"p": [0, 4, 0], "s": [10, 8, 10], "r": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                       "col": [0.8, 0.2, 0.2], "t": 0, "sh": "Block"})
                    res = {"ok": True, "output": "Zbudowano domek"}
            else:
                res = {"ok": True, "snapshot": {"parts": self.parts, "total_parts": len(self.parts),
                                                "tree": ["Model Domek (5 inside)"] if self.parts else [], "scripts": []}}
            self.req("POST", "/api/bridge/roblox/result", {"id": cmd["id"], **res})


@pytest.fixture
def live(trained_app):
    s = LiveServer(trained_app)
    yield s
    s.stop()


# ------------------------------------------------------------ knowledge
def test_chunking_keeps_code_blocks():
    text = "# A\n" + "zdanie. " * 50 + "\n\n```lua\nlocal x = 1\n\nprint(x)\n```\n\n# B\nkoniec tekstu tutaj"
    chunks = chunk_text(text, max_chars=300)
    assert any("local x = 1\n\nprint(x)" in c for c in chunks)
    assert all(c.count("```") % 2 == 0 for c in chunks)


def test_knowledge_search_polish_forms_and_packs(trained_app):
    kb = trained_app.knowledge
    kb.add("Pętle", "W języku Lua pętla for liczy od 1. Pętlę while kończ warunkiem.", "lesson")
    kb.add("Kolory", "Kolor części ustawiasz przez Color3.fromRGB.", "lesson")
    hits = kb.search("jak zrobić pętlę w lua?")
    assert hits and hits[0]["title"] == "Pętle"
    assert kb.add("x", "W języku Lua pętla for liczy od 1. Pętlę while kończ warunkiem.")["duplicate"]
    with pytest.raises(ValueError):
        kb.add("x", "za krótko")
    packs = {p["id"] for p in kb.packs()}
    assert {"lua_roblox", "blender_modeling"} <= packs
    r = kb.import_pack("lua_roblox")
    assert r["chunks"] > 10 and r["doc"]["topic"] == "roblox"
    kb.import_pack("blender_modeling")
    top = kb.search("dodaj modyfikator bevel do kostki", topic="blender")
    assert top and "blender" in top[0]["topic"]
    lb = kb.search("tabela wyników leaderstats", topic="roblox")
    assert any("leaderstats" in h["text"] for h in lb)


def test_knowledge_api_and_learn_command(trained_app, ollama):
    enable(trained_app, ollama)
    with TestClient(create_app(trained_app)) as c:
        assert c.post("/api/knowledge/packs/lua_roblox").status_code == 200
        assert c.post("/api/knowledge/packs/../etc").status_code in (400, 404, 405)
        assert c.post("/api/knowledge/packs/a.b").status_code == 400
        ev = events(c, {"message": "/naucz W mojej grze waluta nazywa się Kryształy i zbiera się je na wyspie."})
        assert ev[-1]["type"] == "done" and "Zapisałem lekcję" in ev[-1]["text"]
        docs = c.get("/api/knowledge").json()["docs"]
        assert any("Kryształy" in c.get(f"/api/knowledge/{d['id']}").json()["text"] for d in docs)
        f = c.post("/api/knowledge/file", files={"file": ("notatki.md", "# Moje notatki\nKrótki opis gry o smokach i rycerzach.".encode(), "text/markdown")})
        assert f.status_code == 200
        d = docs[0]["id"]
        assert c.delete(f"/api/knowledge/{d}").status_code == 200


# ------------------------------------------------------------ agent: chat
def test_nexo12_chat_uses_knowledge_and_streams(trained_app, ollama):
    enable(trained_app, ollama)
    trained_app.knowledge.add("Gra", "W mojej grze waluta nazywa się Kryształy.", "lesson")
    SCRIPT[:] = ["Waluta w Twojej grze to Kryształy."]
    SEEN.clear()
    with TestClient(create_app(trained_app)) as c:
        ev = events(c, {"message": "Jak nazywa się waluta w mojej grze?", "target": "chat"})
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "start" and "step" in kinds and "delta" in kinds and kinds[-1] == "done"
    assert ev[-1]["text"] == "Waluta w Twojej grze to Kryształy."
    assert "Kryształy" in SEEN[-1]["messages"][0]["content"]  # notes were put into the prompt
    assert any("Przeszukuję" in e.get("text", "") for e in ev if e["type"] == "step")


def test_nexo12_without_models(trained_app):
    with TestClient(create_app(trained_app)) as c:
        ev = events(c, {"message": "hej", "target": "chat"})       # built-in engine, nothing downloaded
        assert "Modele AI" in ev[-1]["error"]
        save_settings({"assistant": {"enabled": False}}, trained_app.paths); trained_app.reload_config()
        assert c.post("/api/agent/chat", json={"message": "hej"}).status_code == 403


# ------------------------------------------------------------ agent: roblox
CODE_BAD = "PLAN: Buduję domek.\n```lua\nlocal p = nil\nprint(p.Size)\n```"
CODE_OK = "PLAN: Poprawiam i buduję domek z czerwonych części.\n```lua\nlocal m = Instance.new('Model')\nm.Name = 'Domek'\nm.Parent = workspace\nprint('Zbudowano domek')\n```"


def test_roblox_agent_fixes_error_and_renders(trained_app, ollama, live):
    enable(trained_app, ollama)
    plugin = FakeRobloxPlugin(live.url, trained_app.bridge.token, fail_first=True)
    plugin.start()
    try:
        for _ in range(50):
            if trained_app.bridge.online("roblox"):
                break
            time.sleep(0.1)
        SCRIPT[:] = [CODE_BAD, CODE_OK]
        body = json.dumps({"message": "Zbuduj w Roblox domek", "target": "auto"}).encode()
        r = urllib.request.urlopen(urllib.request.Request(live.url + "/api/agent/chat", data=body,
                                                          headers={"Content-Type": "application/json"}), timeout=60)
        ev = [json.loads(x) for x in r.read().decode().strip().split("\n")]
    finally:
        plugin.running = False
    assert ev[0]["target"] == "roblox"
    texts = " | ".join(e.get("text", "") for e in ev if e["type"] == "step")
    assert "Błąd" in texts and "poprawiam" in texts.lower() and "Kod zadziałał" in texts
    assert len(plugin.ran) == 2 and "Instance.new('Model')" in plugin.ran[1]
    img = [e for e in ev if e["type"] == "image"][0]
    png = trained_app.paths.generated / img["url"].rsplit("/", 1)[1]
    assert png.exists() and png.stat().st_size > 500
    done = ev[-1]
    assert done["error"] is None and "```lua" in done["text"] and "Zbudowano domek" in done["text"]
    msg = trained_app.db.one("SELECT * FROM messages WHERE id=?", (done["message_id"],))
    meta = json.loads(msg["meta"])
    assert meta["code"].startswith("local m") and any(t["type"] == "code" for t in meta["thoughts"])
    # remember the working solution as an example
    ex = trained_app.agent.save_example(done["message_id"])
    assert ex["doc"]["kind"] == "example" and ex["doc"]["topic"] == "roblox"
    # the error message the model saw on its second try
    assert any("attempt to index nil" in m["content"] for r in SEEN for m in r.get("messages", []))
    # after success it read the scene back and checked the result against the request
    assert any("Scena po zmianach" in m["content"] and "Model Domek" in m["content"]
               for r in SEEN for m in r.get("messages", []))


def test_approval_flow_and_risky_code(trained_app, ollama, live):
    enable(trained_app, ollama, auto_run=False)
    plugin = FakeRobloxPlugin(live.url, trained_app.bridge.token)
    plugin.start()
    try:
        for _ in range(50):
            if trained_app.bridge.online("roblox"):
                break
            time.sleep(0.1)
        SCRIPT[:] = [CODE_OK]
        body = json.dumps({"message": "Zbuduj domek", "target": "roblox"}).encode()
        resp = urllib.request.urlopen(urllib.request.Request(live.url + "/api/agent/chat", data=body,
                                                             headers={"Content-Type": "application/json"}), timeout=60)
        got = []
        for line in resp:
            e = json.loads(line)
            got.append(e)
            if e["type"] == "approve":
                assert plugin.ran == []  # nothing runs before approval
                urllib.request.urlopen(urllib.request.Request(
                    f"{live.url}/api/agent/{e['run_id']}/decision", data=b'{"approve": false}',
                    headers={"Content-Type": "application/json"}), timeout=10)
        assert got[-1]["error"].startswith("Nie uruchomiono") and plugin.ran == []
    finally:
        plugin.running = False
    from nexora.agent import RISKY
    assert RISKY.search("import os\nos.remove('x')") and RISKY.search("game:GetService('InsertService')")
    assert not RISKY.search("bpy.ops.mesh.primitive_cube_add()")


def test_not_connected_and_target_detection(trained_app, ollama):
    from nexora.agent import pick_target, split_reply
    enable(trained_app, ollama)
    with TestClient(create_app(trained_app)) as c:
        ev = events(c, {"message": "Zrób kostkę", "target": "blender"})
    assert "nie jest połączony" in ev[-1]["error"]

    class B:
        def __init__(self, on): self.on = on
        def online(self, a): return a in self.on
    assert pick_target("zbuduj w roblox wieżę", "auto", B({"roblox"})) == "roblox"
    assert pick_target("zbuduj w roblox wieżę", "auto", B(set())) == "chat"
    assert pick_target("wymodeluj krzesło", "auto", B({"blender", "roblox"})) == "blender"
    assert pick_target("zbuduj zamek", "auto", B({"blender"})) == "blender"
    assert pick_target("jak się masz?", "auto", B({"blender"})) == "chat"
    plan, code = split_reply("PLAN: Robię.\n```lua\nprint(1)\n```", "lua")
    assert plan == "Robię." and code == "print(1)"
    assert split_reply("Po prostu odpowiedź.", "lua") == ("Po prostu odpowiedź.", None)


def test_vision_check_loop(trained_app, ollama, live):
    enable(trained_app, ollama, vision_model="eyes")
    plugin = FakeRobloxPlugin(live.url, trained_app.bridge.token)
    plugin.start()
    try:
        for _ in range(50):
            if trained_app.bridge.online("roblox"):
                break
            time.sleep(0.1)
        SCRIPT[:] = [CODE_OK, "GOTOWE: Domek stoi na trawie."]
        body = json.dumps({"message": "Zbuduj w roblox czerwony domek", "target": "roblox"}).encode()
        r = urllib.request.urlopen(urllib.request.Request(live.url + "/api/agent/chat", data=body,
                                                          headers={"Content-Type": "application/json"}), timeout=60)
        ev = [json.loads(x) for x in r.read().decode().strip().split("\n")]
    finally:
        plugin.running = False
    steps = [e["text"] for e in ev if e["type"] == "step"]
    assert any(s.startswith("Widzę: Widać czerwony domek") for s in steps)
    vision_req = [s for s in SEEN if not s.get("stream", True)][-1]
    assert vision_req["model"] == "eyes" and vision_req["messages"][0]["images"]
    assert "Domek stoi na trawie" in ev[-1]["text"]


# ------------------------------------------------------------ bridge security
def test_bridge_requires_token(trained_app, live):
    def get(token):
        r = urllib.request.Request(live.url + "/api/bridge/roblox/poll?wait=0", headers={"X-Nexora-Token": token})
        try:
            return urllib.request.urlopen(r, timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code
    assert get("wrong") == 403 and get("") == 403
    assert get(trained_app.bridge.token) == 200
    with TestClient(create_app(trained_app)) as c:
        lua = c.get("/api/integrations/roblox/download").text
        assert trained_app.bridge.token in lua and "{{TOKEN}}" not in lua
        py = c.get("/api/integrations/blender/download").text
        assert f'TOKEN = "{trained_app.bridge.token}"' in py


# ------------------------------------------------------------ roblox plugin source
def test_roblox_plugin_is_valid_lua():
    lupa = pytest.importorskip("lupa")
    src = (ROOT / "integrations" / "roblox" / "NexoraBridge.lua").read_text(encoding="utf-8")
    rt = lupa.LuaRuntime()
    compiled = rt.eval("function(src) local f, e = load(src) return f ~= nil, e end")(src)
    assert compiled[0], compiled[1]
    for needle in ("TryBeginRecording", "FinishRecording", "X-Nexora-Token", "/api/bridge/roblox/poll",
                   "RunService:IsRunning()", "LogService.MessageOut"):
        assert needle in src


def test_iso_render_draws_parts():
    from PIL import Image
    import io
    parts = [{"p": [0, 0.5, 0], "s": [40, 1, 40], "r": [1, 0, 0, 0, 1, 0, 0, 0, 1], "col": [0.3, 0.6, 0.3], "t": 0},
             {"p": [0, 4, 0], "s": [8, 7, 8], "r": [1, 0, 0, 0, 1, 0, 0, 0, 1], "col": [0.9, 0.1, 0.1], "t": 0},
             {"p": [6, 3, 0], "s": [3, 3, 3], "col": [0.1, 0.1, 0.9], "t": 0, "sh": "Ball"}]
    im = Image.open(io.BytesIO(render_scene(parts))).convert("RGB")
    colors = {im.getpixel((x, y)) for x in range(0, im.width, 8) for y in range(0, im.height, 8)}
    assert any(r > 150 and g < 60 for r, g, b in colors)   # red house visible
    assert any(b > 150 and r < 60 for r, g, b in colors)   # blue ball visible
    assert len(colors) > 6


# ------------------------------------------------------------ blender add-on (real Blender)
BLENDER_CLIENT = r'''
import importlib.util, json, sys, time
spec = importlib.util.spec_from_file_location("nexora_bridge", sys.argv[1])
nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
done = 0
deadline = time.time() + 120
while done < int(sys.argv[2]) and time.time() < deadline:
    cmd = nb.poll_once(wait=2)
    if cmd:
        nb.send_result(nb.handle(cmd))
        done += 1
print("CLIENT DONE", done)
'''


def test_blender_addon_with_real_blender(trained_app, ollama, live, tmp_path):
    pytest.importorskip("bpy")
    enable(trained_app, ollama)
    with TestClient(create_app(trained_app)) as c:
        addon = c.get("/api/integrations/blender/download").text.replace(
            f"127.0.0.1:{trained_app.cfg()['server']['port']}", f"127.0.0.1:{live.port}")
    (tmp_path / "addon.py").write_text(addon, encoding="utf-8")
    (tmp_path / "client.py").write_text(BLENDER_CLIENT, encoding="utf-8")
    # expected commands: snapshot, run (fails), run (ok), render
    proc = subprocess.Popen([sys.executable, str(tmp_path / "client.py"), str(tmp_path / "addon.py"), "4"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(300):
            if trained_app.bridge.online("blender"):
                break
            time.sleep(0.1)
        assert trained_app.bridge.online("blender"), "add-on did not connect"
        SCRIPT[:] = ["PLAN: Robię stół.\n```python\nbpy.ops.mesh.primitive_cube_add(size=1)\nundefined_name.x = 1\n```",
                     "PLAN: Poprawiony stół.\n```python\nbpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.75))\n"
                     "o = bpy.context.active_object\no.name = 'Blat'\no.scale = (1.6, 0.9, 0.06)\n"
                     "m = bpy.data.materials.new('Drewno')\nm.diffuse_color = (0.45, 0.28, 0.12, 1)\no.data.materials.append(m)\n"
                     "print('Stół gotowy')\n```"]
        body = json.dumps({"message": "Wymodeluj w Blenderze stół", "target": "blender"}).encode()
        r = urllib.request.urlopen(urllib.request.Request(live.url + "/api/agent/chat", data=body,
                                                          headers={"Content-Type": "application/json"}), timeout=240)
        ev = [json.loads(x) for x in r.read().decode().strip().split("\n")]
        out, _ = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    steps = " | ".join(e.get("text", "") for e in ev if e["type"] == "step")
    assert "NameError" in steps, steps            # real Python error from Blender came back
    assert "Kod zadziałał" in steps and "Stół gotowy" in steps
    img = [e for e in ev if e["type"] == "image"]
    assert img, steps
    png = trained_app.paths.generated / img[0]["url"].rsplit("/", 1)[1]
    from PIL import Image
    im = Image.open(png).convert("RGB")
    assert im.size == (480, 360)
    # the brown table top must be visible in the render
    px = [im.getpixel((x, y)) for x in range(0, 480, 4) for y in range(0, 360, 4)]
    brown = [p for p in px if p[0] > p[1] + 12 > p[2] + 24]
    assert len(brown) > 200, "table top not visible"
    grey = [p for p in px if abs(p[0] - p[1]) < 4 and abs(p[1] - p[2]) < 4 and 60 < p[0] < 220]
    assert len(grey) < 200, "the cube from the failed attempt was not removed"
    assert "Removed" in steps
    assert ev[-1]["error"] is None, ev[-1]
    assert "CLIENT DONE 4" in out, out[-2000:]


# ------------------------------------------------------------ video learning
def make_video(path, seconds=6, fps=5):
    av = pytest.importorskip("av")
    import numpy as np
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 160, 120, "yuv420p"
        for i in range(seconds * fps):
            arr = np.zeros((120, 160, 3), dtype=np.uint8)
            arr[:, :, 0] = 40 * (i // fps)
            for pkt in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_subtitles_and_frames(tmp_path):
    from nexora.video import extract_frames, parse_subtitles
    srt = "1\n00:00:01,000 --> 00:00:03,000\nWitaj w lekcji Lua.\n\n2\n00:01:05,500 --> 00:01:08,000\n<i>Tworzymy część</i>\n"
    assert parse_subtitles(srt) == [(1.0, "Witaj w lekcji Lua."), (65.0, "Tworzymy część")]
    vtt = "WEBVTT\n\n00:02.000 --> 00:04.000\nPierwsza linia\n\n00:10.000 --> 00:12.000\nDruga linia\n"
    assert parse_subtitles(vtt) == [(2.0, "Pierwsza linia"), (10.0, "Druga linia")]
    pytest.importorskip("av")
    make_video(tmp_path / "v.mp4")
    frames = extract_frames(str(tmp_path / "v.mp4"), tmp_path, every=2, max_frames=5)
    assert 2 <= len(frames) <= 5 and all(p.exists() for _, p in frames)


def test_video_job_builds_lesson(trained_app, ollama, monkeypatch, tmp_path):
    import nexora.video as vid
    enable(trained_app, ollama, vision_model="eyes")
    save_settings({"video": {"describe_frames": True, "frame_every_seconds": 2}}, trained_app.paths)
    trained_app.reload_config()
    monkeypatch.setattr(vid, "transcribe", lambda path, size, lang, progress=None: iter(
        [(0.0, "Dziś uczymy się, jak zrobić w Roblox znikającą platformę."),
         (4.0, "Ustawiamy Transparency na 1 i CanCollide na false po dotknięciu.")]))
    with TestClient(create_app(trained_app)) as c:
        make_video(tmp_path / "lekcja.mp4")
        r = c.post("/api/video", files={"file": ("lekcja.mp4", (tmp_path / "lekcja.mp4").read_bytes(), "video/mp4")},
                   data={"also_training": "true", "topic": "roblox"}).json()
        j = wait_job(trained_app, r["job_id"], timeout=120)
        assert j["status"] == "completed", j
        res = j["result"]
        assert res["lines"] == 2 and res["frames"] and res["captions"] >= 1 and res["training_examples"] >= 1
        txt = c.get(f"/api/knowledge/{res['knowledge_doc']}").json()["text"]
        assert "[00:04] Ustawiamy Transparency" in txt and "Co widać na filmie" in txt
        assert trained_app.knowledge.search("znikająca platforma roblox")
        assert c.get(f"/generated/{res['frames'][0]}").status_code == 200
        assert c.get("/generated/video_1/..%2F..%2Fsettings.json").status_code in (400, 404)
        assert c.post("/api/video", files={"file": ("x.exe", b"MZ", "application/x-msdownload")}).status_code == 400
        sub = c.post("/api/video", files={"file": ("napisy.srt", b"1\n00:00:01,000 --> 00:00:02,000\nLekcja modelowania krzesla w Blenderze.\n", "text/plain")}).json()
        assert wait_job(trained_app, sub["job_id"])["status"] == "completed"
