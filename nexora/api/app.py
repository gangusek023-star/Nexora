"""FastAPI backend. Serves the JSON API under /api and the web UI at /."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import save_settings
from ..lm.persona import MAX_CONTEXT_CHARS, STYLES, sampling_for
from .auth import COOKIE, MASK, PUBLIC_PREFIXES, GoogleAuth
from ..guard.guard import evaluate_all
from ..guard.registry import TamperError
from ..learning.base import prepare_base
from ..lm.data import format_exchange
from ..system.monitor import folder_size_mb, snapshot

WEB = Path(__file__).resolve().parent.parent / "web"
ROOT = Path(__file__).resolve().parent.parent.parent
LANGUAGES = ("pl", "en", "de", "es", "uk")


# ------------------------------------------------------------------ schemas
class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    top_k: int | None = Field(None, ge=0, le=1000)
    top_p: float | None = Field(None, gt=0, le=1)
    max_new_tokens: int | None = Field(None, ge=1, le=400)
    repetition_penalty: float | None = Field(None, ge=1, le=2)
    history_turns: int = Field(2, ge=0, le=8)


class PreviewIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    persona: dict = {}
    seed: int | None = 0


class AssistantIn(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    conversation_id: int | None = None
    attachment_ids: list[int] = []


class AgentIn(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    conversation_id: int | None = None
    attachment_ids: list[int] = []
    target: str = "auto"


class DecisionIn(BaseModel):
    approve: bool


class KnowledgeIn(BaseModel):
    title: str = Field("", max_length=200)
    text: str = Field(min_length=20, max_length=500_000)
    topic: str = ""


class UrlIn(BaseModel):
    url: str = Field(min_length=8, max_length=2000)


class TeachIn(BaseModel):
    examples: list[dict] = Field(min_length=1, max_length=200)


class FeedbackIn(BaseModel):
    message_id: int
    rating: int
    correction: str | None = Field(None, max_length=2000)


class FactIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    kind: str = "unverified"
    source: str = "user"
    confidence: float | None = Field(None, ge=0, le=1)


class FactPatch(BaseModel):
    kind: str | None = None
    status: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)


class NoteIn(BaseModel):
    title: str = ""
    body: str = Field(min_length=1, max_length=20000)
    source: str = "user"


class ExampleIn(BaseModel):
    text: str | None = None
    prompt: str | None = None
    answer: str | None = None


class GoalIn(BaseModel):
    title: str
    description: str = ""
    keywords: list[str] = []


class DecideIn(BaseModel):
    approve: bool
    note: str = ""


class TrainIn(BaseModel):
    steps: int | None = Field(None, ge=10, le=200000)
    max_minutes: float | None = Field(None, gt=0, le=24 * 60)


class ImageDataIn(BaseModel):
    source: str = "synthetic_shapes"
    name: str = "shapes"
    n: int = Field(4000, ge=50, le=50000)
    folder: str | None = None


class ImageTrainIn(BaseModel):
    dataset: str
    steps: int | None = Field(None, ge=10, le=500000)
    resume_from: str | None = None
    max_minutes: float | None = Field(None, gt=0, le=24 * 60)


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    n: int = Field(4, ge=1, le=8)
    steps: int | None = Field(None, ge=5, le=400)
    guidance: float | None = Field(None, ge=0, le=15)
    seed: int | None = None


def _404(e):
    raise HTTPException(404, str(e))


def create_app(nexora=None) -> FastAPI:
    state = {"app": nexora}

    @asynccontextmanager
    async def lifespan(_):
        if state["app"] is None:
            from ..core import Nexora
            state["app"] = Nexora()
        yield
        if nexora is None:
            state["app"].shutdown()

    api = FastAPI(title="NEXORA AI", version="1.0", lifespan=lifespan)

    def A():
        return state["app"]

    def AUTH() -> GoogleAuth:
        if state.get("auth") is None or state["auth"].app is not A():
            state["auth"] = GoogleAuth(A())
        return state["auth"]

    # ------------------------------------------------------------------ auth
    @api.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if state["app"] is None or path.startswith(PUBLIC_PREFIXES) or path == "/api/shutdown" \
                or path.startswith("/api/bridge/"):
            return await call_next(request)
        auth = AUTH()
        if auth.enabled():
            email = auth.read_session(request.cookies.get(COOKIE))
            if not email:
                if path.startswith("/api/") or path.startswith("/generated/"):
                    return JSONResponse({"detail": "Sign in with Google first."}, status_code=401)
                return RedirectResponse("/login")
            request.state.user = email
        return await call_next(request)

    @api.get("/login")
    def login_page():
        return FileResponse(WEB / "login.html")

    @api.get("/auth/login")
    def auth_login():
        auth = AUTH()
        if not auth.enabled():
            return RedirectResponse("/")
        return RedirectResponse(auth.login_url())

    @api.get("/auth/callback")
    def auth_callback(request: Request, code: str = "", error: str = ""):
        from urllib.parse import quote

        def fail(msg):
            return RedirectResponse("/login?error=" + quote(msg[:200]))
        if error or not code:
            return fail(error or "Google did not return a login code")
        try:
            email = AUTH().finish(code, request.query_params.get("state", ""))
        except ValueError as e:
            return fail(str(e))
        except OSError:
            return fail("could not reach Google - check your internet connection")
        resp = RedirectResponse("/")
        resp.set_cookie(COOKIE, AUTH().make_session(email), httponly=True, samesite="lax",
                        max_age=int(A().cfg()["auth"].get("session_hours", 168) * 3600))
        return resp

    @api.get("/auth/logout")
    def auth_logout():
        resp = RedirectResponse("/login" if AUTH().enabled() else "/")
        resp.delete_cookie(COOKIE)
        return resp

    @api.get("/auth/me")
    def auth_me(request: Request):
        auth = AUTH()
        return {"enabled": auth.enabled(), "email": auth.read_session(request.cookies.get(COOKIE))}

    # ---------------------------------------------------------------- status
    @api.get("/api/status")
    def status():
        a = A()
        lm = a.lm.status()
        running = a.db.one("SELECT id, kind, status FROM jobs WHERE status IN ('running','paused','queued') "
                           "ORDER BY id LIMIT 1")
        pending = a.db.one("SELECT COUNT(*) n FROM model_versions WHERE status='pending_approval'")["n"]
        return {"lm": lm, "base_prepared": a.paths.tokenizer.exists() and bool(a.datasets.list()),
                "job": a.jobs.get(running["id"]) if running else None, "pending_candidates": pending,
                "image_model": (a.images.active() or {}).get("id"),
                "schedule": {**a.cfg()["schedule"], "last_run": a.scheduler.last_run},
                "time": time.time()}

    # ------------------------------------------------------------------ chat
    @api.post("/api/chat")
    def chat(body: ChatIn):
        a = A()
        cid = body.conversation_id or a.memory.new_conversation(body.message[:40])
        msg = body.message.strip()
        uid = a.memory.add_message(cid, "user", msg)
        recall = a.memory.search(msg, limit=3)
        m = re.match(r"^/remember\s+(.+)$", msg, re.I | re.S)
        if m:  # explicit memory command - handled by the memory system, not the network
            r = a.memory.add_fact(m.group(1), kind="unverified", source=f"chat #{cid}")
            note = ("Already in memory (duplicate)." if r["duplicate"] else
                    "Stored in memory as an unverified fact." +
                    (f" It conflicts with {len(r['conflicts'])} existing fact(s) - review it in Memory."
                     if r["conflicts"] else ""))
            note += " This does not change the neural network's weights."
            aid = a.memory.add_message(cid, "system", note)
            return {"conversation_id": cid, "user_message_id": uid, "message_id": aid, "kind": "memory",
                    "text": note, "fact": r["fact"]}
        m = re.match(r"^/(learn|naucz)\s+(.+)$", msg, re.I | re.S)
        if m:  # explicit request: add this text to the training data
            r = a.learning.learn_from_user_text(m.group(2), f"/learn in chat #{cid}", explicit=True)
            note = {"accepted": "Added to the training data. The network learns it only after the next "
                                "learning cycle trains a candidate and you approve it.",
                    "pending": "Added to the training data, waiting for your review in Datasets.",
                    "duplicate": "This text is already in the training data."}.get(
                r["status"], f"Not added: {r['reason']}.")
            aid = a.memory.add_message(cid, "system", note)
            return {"conversation_id": cid, "user_message_id": uid, "message_id": aid,
                    "kind": "learn", "text": note, "example": r}
        learned = None
        if not msg.startswith("/"):
            learned = a.learning.learn_from_user_text(msg, f"chat prompt #{uid}", explicit=False)
        persona = a.cfg()["persona"]
        # personality style gives the defaults; values set in the chat panel win
        params = sampling_for(persona, a.cfg()["generation"])
        params.update({k: getattr(body, k) for k in params if getattr(body, k, None) is not None})
        # short-term context: earlier completed exchanges of this conversation
        history = a.memory.recent_exchanges(cid, body.history_turns) if body.history_turns else []
        try:
            out = a.lm.chat(msg, history, persona=persona, **params)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        web_results, web_error = [], None
        ic = a.cfg().get("internet", {})
        if ic.get("search_in_chat"):
            from ..webaccess import WebError, wikipedia_search
            try:
                web_results = wikipedia_search(msg, ic.get("wikipedia_language", "pl"))
            except WebError as e:
                web_error = str(e)
        aid = a.memory.add_message(cid, "assistant", out["text"] or "…", out["version"],
                                   {"params": params, "tokens": out["tokens"],
                                    "postprocess": out["postprocess"], "web_results": web_results})
        return {"conversation_id": cid, "user_message_id": uid, "message_id": aid, "kind": "model",
                **out, "params": params,
                "prompt_learning": learned["status"] if learned else None,
                "web_results": web_results, "web_error": web_error,
                "memory_recall": [{"id": f["id"], "text": f["text"], "kind": f["kind"],
                                   "source": f["source"]} for f in recall["facts"]]}

    # ----------------------------------------------------------- personality
    @api.get("/api/persona")
    def persona_get():
        return {"persona": A().cfg()["persona"], "max_context_chars": MAX_CONTEXT_CHARS,
                "styles": {k: {"settings": v[0], "description": v[1]} for k, v in STYLES.items()}}

    @api.post("/api/persona/preview")
    def persona_preview(body: PreviewIn):
        """Try personality settings without saving them or storing the chat."""
        a = A()
        persona = {**a.cfg()["persona"], **body.persona}
        if persona.get("style") not in STYLES:
            raise HTTPException(400, "unknown style")
        params = sampling_for(persona, a.cfg()["generation"])
        try:
            out = a.lm.chat(body.message, [], persona=persona, seed=body.seed, **params)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {**out, "params": params}

    @api.post("/api/persona/teach")
    def persona_teach(body: TeachIn):
        """Add example exchanges in the wanted style to the training data."""
        a = A()
        sh = a.evals.all_shingles()
        results = []
        for ex in body.examples:
            if ex.get("prompt", "").strip() and ex.get("answer", "").strip():
                results.append(a.learning.add_example(format_exchange(ex["prompt"], ex["answer"]),
                                                      "user_added", "Personality examples", eval_shingles=sh))
        return {"results": results, "accepted": sum(r["status"] == "accepted" for r in results)}

    # ---------------------------------------------------------------- files
    def _attachment(a, fid: int) -> dict:
        row = a.db.one("SELECT * FROM attachments WHERE id=?", (fid,))
        if not row:
            raise HTTPException(404, f"no such file #{fid}")
        p = a.paths.uploads / row["path"]
        row["text"] = p.read_text(encoding="utf-8") if p.exists() else ""
        return row

    @api.post("/api/files")
    async def upload_file(file: UploadFile = File(...), conversation_id: int | None = Form(None)):
        from ..files import FileError, extract_text
        a = A()
        limit = int(a.cfg()["assistant"].get("max_upload_mb", 10)) * 1_000_000
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"file larger than {limit // 1_000_000} MB")
        name = os.path.basename(file.filename or "file.txt")[:200]
        try:
            ex = extract_text(name, data)
        except FileError as e:
            raise HTTPException(400, str(e))
        fid = a.db.execute("INSERT INTO attachments(conversation_id,name,kind,language,path,chars,created) "
                           "VALUES (?,?,?,?,?,?,?)", (conversation_id, name, ex["kind"], ex["language"], "",
                                                     len(ex["text"]), time.time())).lastrowid
        fname = f"{fid}.txt"
        (a.paths.uploads / fname).write_text(ex["text"], encoding="utf-8")
        a.db.execute("UPDATE attachments SET path=? WHERE id=?", (fname, fid))
        a.db.audit("user", "file_attached", id=fid, name=name, chars=len(ex["text"]))
        return {"id": fid, "name": name, "kind": ex["kind"], "language": ex["language"],
                "chars": len(ex["text"]), "preview": ex["text"][:300]}

    @api.post("/api/files/{fid}/learn")
    def learn_file(fid: int):
        from ..files import code_chunks
        from ..lm.data import split_paragraphs
        a = A()
        f = _attachment(a, fid)
        sh = a.evals.all_shingles()
        stats: dict[str, int] = {}
        pieces = code_chunks(f["text"]) if f["kind"] == "code" else split_paragraphs(f["text"], min_chars=40)
        for piece in pieces:
            r = a.learning.add_example(piece[:4000], "user_added", f"file: {f['name']}", eval_shingles=sh,
                                       kind="code" if f["kind"] == "code" else "text")
            stats[r["status"]] = stats.get(r["status"], 0) + 1
        return {"name": f["name"], "pieces": len(pieces), **stats}

    @api.delete("/api/files/{fid}")
    def delete_file(fid: int):
        a = A()
        f = _attachment(a, fid)
        (a.paths.uploads / f["path"]).unlink(missing_ok=True)
        a.db.execute("DELETE FROM attachments WHERE id=?", (fid,))
        return {"ok": True}

    # ----------------------------------------------------- built-in AI models
    @api.get("/api/llm/models")
    def llm_models():
        a = A()
        from ..localmodels import llama_available
        ok, info = llama_available()
        active = {j["params"].get("id"): j for j in a.jobs.list(30)
                  if j["kind"] == "model_download" and j["status"] in ("queued", "running", "paused", "stopped", "interrupted")}
        return {"engine": {"installed": ok, "info": info}, "models": a.llm.list(),
                "downloads": {k: {"job_id": v["id"], "status": v["status"], "progress": v["progress"], "error": v["error"]}
                              for k, v in active.items()}}

    @api.post("/api/llm/models/{mid}/download")
    def llm_download(mid: str):
        a = A()
        if mid not in a.llm.catalog:
            raise HTTPException(404, "unknown model")
        if a.llm.installed(mid):
            return {"ok": True, "already": True}
        for j in a.jobs.list(30):
            if j["kind"] == "model_download" and j["params"].get("id") == mid:
                if j["status"] in ("queued", "running", "paused"):
                    return {"job_id": j["id"]}
                if j["status"] in ("stopped", "interrupted"):
                    a.jobs.resume(j["id"])
                    return {"job_id": j["id"], "resumed": True}
        return {"job_id": a.jobs.submit("model_download", {"id": mid, "max_minutes": 600})}

    @api.delete("/api/llm/models/{mid}")
    def llm_delete(mid: str):
        a = A()
        if mid not in a.llm.catalog:
            raise HTTPException(404, "unknown model")
        a.llm.delete(mid)
        return {"ok": True}

    @api.post("/api/llm/models/{mid}/use")
    def llm_use(mid: str, role: str):
        a = A()
        if mid not in a.llm.catalog or role not in ("chat", "code"):
            raise HTTPException(400, "bad model or role")
        save_settings({"assistant": {f"{role}_model": mid}}, a.paths)
        a.reload_config()
        return {"ok": True}

    # ------------------------------------------------------------ assistant
    @api.get("/api/assistant/status")
    def assistant_status():
        a = A()
        st = a.llm.status()
        st["chat_model"] = a.cfg()["assistant"]["chat_model"]
        st["code_model"] = a.cfg()["assistant"]["code_model"]
        st["default_engine"] = a.cfg()["assistant"].get("default_engine", "assistant")
        return st

    @api.post("/api/assistant/chat")
    def assistant_chat(body: AssistantIn):
        """Streams NDJSON lines: {"delta": "..."} ... then {"done": true, ...}."""
        from .. import assistant as asst
        a = A()
        cfg = a.cfg()["assistant"]
        if not cfg.get("enabled"):
            raise HTTPException(403, "The assistant is turned off (Settings → Assistant → enabled).")
        atts = [_attachment(a, i) for i in body.attachment_ids[:5]]
        cid = body.conversation_id or a.memory.new_conversation(body.message[:40])
        for f in atts:
            a.db.execute("UPDATE attachments SET conversation_id=? WHERE id=?", (cid, f["id"]))
        names = "".join(f"\n[file: {f['name']}]" for f in atts)
        uid = a.memory.add_message(cid, "user", body.message.strip() + names)
        if not body.message.startswith("/"):
            a.learning.learn_from_user_text(body.message, f"chat prompt #{uid}", explicit=False)
        n = int(cfg.get("history_turns", 6))
        prev = a.db.query("SELECT role, content FROM messages WHERE conversation_id=? AND id<? AND "
                          "role IN ('user','assistant') ORDER BY id DESC LIMIT ?", (cid, uid, n * 2))[::-1]
        history = [{"role": m["role"], "content": m["content"]} for m in prev]
        persona = a.cfg()["persona"]
        model = asst.pick_model(cfg, body.message, atts)
        if cfg.get("backend", "builtin") == "builtin":
            model = a.llm.pick(model)
        if not model:
            raise HTTPException(409, "No ready-made model downloaded yet. Open the AI models page and download one.")
        system = asst.system_prompt(persona, a.cfg()["ui"]["language"], cfg.get("extra_instructions", ""))
        messages = asst.build_messages(system, history, body.message, atts, int(cfg.get("max_file_chars", 12000)))
        temp = asst.temperature_for(persona, cfg)

        def gen():
            parts, err = [], None
            t0 = time.time()
            yield json.dumps({"start": True, "conversation_id": cid, "user_message_id": uid,
                              "model": model}) + "\n"
            try:
                for piece in a.llm.stream_chat(model, messages, temp):
                    parts.append(piece)
                    yield json.dumps({"delta": piece}) + "\n"
            except asst.AssistantError as e:
                err = str(e)
            text = asst.blocked_filter("".join(parts).strip(), persona)
            aid = None
            if text:
                aid = a.memory.add_message(cid, "assistant", text, f"assistant:{model}",
                                           {"engine": "assistant", "model": model,
                                            "files": [f["name"] for f in atts],
                                            "seconds": round(time.time() - t0, 1)})
            yield json.dumps({"done": True, "message_id": aid, "text": text, "error": err,
                              "model": model, "seconds": round(time.time() - t0, 1)}) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    # ------------------------------------------------------------- Nexo 1.2
    @api.post("/api/agent/chat")
    def agent_chat(body: AgentIn):
        """Nexo 1.2. Streams NDJSON events: start, step, think, code, approve, image, delta, done."""
        a = A()
        if not a.cfg()["assistant"].get("enabled"):
            raise HTTPException(403, "Nexo 1.2 is turned off (Settings → Assistant → enabled).")
        if body.target not in ("auto", "chat", "roblox", "blender"):
            raise HTTPException(400, "target must be auto, chat, roblox or blender")
        atts = [_attachment(a, i) for i in body.attachment_ids[:5]]
        m = re.match(r"^/(learn|naucz)\s+(.+)$", body.message.strip(), re.I | re.S)
        if m:
            cid = body.conversation_id or a.memory.new_conversation(body.message[:40])
            a.memory.add_message(cid, "user", body.message.strip())
            try:
                r = a.knowledge.add("", m.group(2), "lesson", f"/learn in chat #{cid}")
                note = ("Ta wiedza już jest w moich notatkach." if r["duplicate"] else
                        f"Zapisałem lekcję w wiedzy Nexo 1.2 ({r['chunks']} fragmentów). Będę z niej korzystać przy "
                        "odpowiedziach; same wagi modelu się nie zmieniają.")
            except ValueError as e:
                note = f"Nie zapisano: {e}"
            mid = a.memory.add_message(cid, "system", note)

            def once():
                yield json.dumps({"type": "start", "conversation_id": cid}) + "\n"
                yield json.dumps({"type": "done", "message_id": mid, "text": note, "note": True}) + "\n"
            return StreamingResponse(once(), media_type="application/x-ndjson")
        if not body.message.startswith("/"):
            a.learning.learn_from_user_text(body.message, "Nexo 1.2 chat", explicit=False)

        def gen():
            for e in a.agent.run(body.message, body.conversation_id, atts, body.target):
                yield json.dumps(e, ensure_ascii=False) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @api.post("/api/agent/{run_id}/decision")
    def agent_decision(run_id: str, body: DecisionIn):
        if not A().agent.runs.decide(run_id, body.approve):
            raise HTTPException(404, "this run is not waiting any more")
        return {"ok": True}

    @api.post("/api/agent/{run_id}/stop")
    def agent_stop(run_id: str):
        return {"ok": A().agent.runs.stop(run_id)}

    @api.post("/api/agent/example/{message_id}")
    def agent_example(message_id: int):
        try:
            r = A().agent.save_example(message_id)
        except KeyError:
            raise HTTPException(404, "no such answer")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "duplicate": r["duplicate"], "doc": r["doc"]}

    # ------------------------------------------------------------ knowledge
    @api.get("/api/knowledge")
    def knowledge_list():
        a = A()
        return {"docs": a.knowledge.list(), "packs": a.knowledge.packs()}

    @api.post("/api/knowledge")
    def knowledge_add(body: KnowledgeIn):
        try:
            return A().knowledge.add(body.title, body.text, "lesson", "user", body.topic)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.post("/api/knowledge/packs/{pack_id}")
    def knowledge_pack(pack_id: str):
        try:
            return A().knowledge.import_pack(pack_id)
        except KeyError:
            raise HTTPException(404, "no such lesson pack")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.post("/api/knowledge/file")
    async def knowledge_file(file: UploadFile = File(...), topic: str = Form("")):
        from ..files import FileError, extract_text
        a = A()
        data = await file.read(20_000_001)
        if len(data) > 20_000_000:
            raise HTTPException(413, "file larger than 20 MB")
        name = os.path.basename(file.filename or "file.txt")
        try:
            ex = extract_text(name, data)
            return a.knowledge.add(name, ex["text"], "file", f"file:{name}", topic)
        except (FileError, ValueError) as e:
            raise HTTPException(400, str(e))

    @api.get("/api/knowledge/{doc_id}")
    def knowledge_get(doc_id: int):
        a = A()
        d = a.knowledge.get(doc_id)
        if not d:
            raise HTTPException(404, "no such document")
        return {**d, "text": a.knowledge.text_of(doc_id)}

    @api.delete("/api/knowledge/{doc_id}")
    def knowledge_delete(doc_id: int):
        A().knowledge.delete(doc_id)
        return {"ok": True}

    @api.get("/api/knowledge-search")
    def knowledge_search(q: str):
        return A().knowledge.search(q)

    # ---------------------------------------------------------------- video
    @api.post("/api/video")
    async def video_upload(file: UploadFile = File(...), also_training: bool = Form(False), topic: str = Form("")):
        from ..video import SUB_EXT, VIDEO_EXT
        a = A()
        name = os.path.basename(file.filename or "video.mp4")
        ext = os.path.splitext(name)[1].lower()
        if ext not in VIDEO_EXT | SUB_EXT:
            raise HTTPException(400, f"unsupported file type {ext}; use a video/audio file or .srt/.vtt subtitles")
        limit = int(a.cfg()["video"].get("max_upload_mb", 2000)) * 1_000_000
        import shutil as _sh
        if _sh.disk_usage(a.paths.root).free < limit + 1_000_000_000:
            raise HTTPException(507, "not enough free disk space for this video")
        dest = a.paths.uploads / f"video_{int(time.time())}{ext}"
        size = 0
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    f.close(); dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"file larger than {limit // 1_000_000} MB")
                f.write(chunk)
        jid = a.jobs.submit("video_learn", {"path": str(dest), "name": name, "also_training": also_training,
                                            "topic": topic, "max_minutes": 240})
        return {"job_id": jid, "name": name, "mb": round(size / 1e6, 1)}

    # --------------------------------------------------------- integrations
    def _plugin_text(kind: str) -> str:
        a = A()
        base = Path(__file__).resolve().parent.parent.parent / "integrations"
        src = base / ("roblox/NexoraBridge.lua" if kind == "roblox" else "blender/nexora_bridge.py")
        port = str(a.cfg()["server"]["port"])
        return src.read_text(encoding="utf-8").replace("{{PORT}}", port).replace("{{TOKEN}}", a.bridge.token)

    @api.get("/api/integrations")
    def integrations():
        a = A()
        return {"status": a.bridge.status(), "port": a.cfg()["server"]["port"]}

    @api.get("/api/integrations/{kind}/download")
    def integration_download(kind: str):
        from fastapi.responses import Response
        if kind not in ("roblox", "blender"):
            raise HTTPException(404, "unknown integration")
        fname = "NexoraBridge.lua" if kind == "roblox" else "nexora_bridge.py"
        return Response(_plugin_text(kind), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    def _bridge_auth(request: Request):
        if not A().bridge.check_token(request.headers.get("X-Nexora-Token")):
            raise HTTPException(403, "bad or missing NEXORA token - download the plugin/add-on again from NEXORA")

    @api.get("/api/bridge/{app_name}/poll")
    def bridge_poll(app_name: str, request: Request, wait: float = 10):
        from ..bridge import BridgeError
        _bridge_auth(request)
        info = {k: v for k, v in request.query_params.items() if k not in ("wait",)}
        try:
            cmd = A().bridge.poll(app_name, info, wait)
        except BridgeError as e:
            raise HTTPException(404, str(e))
        return {"command": cmd}

    @api.post("/api/bridge/{app_name}/result")
    async def bridge_result(app_name: str, request: Request):
        _bridge_auth(request)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "bad JSON")
        return {"accepted": A().bridge.deliver(app_name, body)}

    # -------------------------------------------------------------- internet
    @api.post("/api/web/learn")
    def web_learn(body: UrlIn):
        from ..webaccess import WebError
        try:
            return A().learning.learn_from_url(body.url.strip())
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except WebError as e:
            raise HTTPException(400, str(e))

    @api.get("/api/web/pages")
    def web_pages():
        return A().db.query("SELECT * FROM web_pages ORDER BY fetched DESC LIMIT 200")

    @api.get("/api/web/search")
    def web_search(q: str):
        from ..webaccess import WebError, wikipedia_search
        a = A()
        if not a.cfg()["internet"].get("search_in_chat"):
            raise HTTPException(403, "Internet search is turned off (Settings → Internet).")
        try:
            return wikipedia_search(q, a.cfg()["internet"].get("wikipedia_language", "pl"))
        except WebError as e:
            raise HTTPException(502, str(e))

    @api.get("/api/conversations")
    def conversations():
        return A().memory.conversations()

    @api.get("/api/conversations/{cid}")
    def conversation(cid: int):
        return A().memory.messages(cid)

    @api.delete("/api/conversations/{cid}")
    def del_conversation(cid: int):
        A().memory.delete_conversation(cid)
        return {"ok": True}

    @api.post("/api/feedback")
    def feedback(body: FeedbackIn):
        try:
            fid = A().memory.add_feedback(body.message_id, body.rating, body.correction)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"id": fid, "note": "Feedback stored. Corrections become training examples at the next "
                                   "learning cycle; the active model only changes after you approve a candidate."}

    @api.get("/api/feedback")
    def list_feedback():
        return A().memory.list_feedback()

    # -------------------------------------------------------------- training
    @api.post("/api/lm/prepare")
    def lm_prepare():
        a = A()
        try:
            return prepare_base(a, [ROOT / "data" / "raw" / "tinyshakespeare.txt"])
        except (RuntimeError, ValueError) as e:
            raise HTTPException(409, str(e))

    @api.post("/api/lm/train")
    def lm_train(body: TrainIn):
        a = A()
        if not a.paths.tokenizer.exists():
            raise HTTPException(409, "Prepare the base data first.")
        p = {}
        if body.steps:
            p["overrides"] = {"max_steps": body.steps}
        if body.max_minutes:
            p["max_minutes"] = body.max_minutes
        return {"job_id": a.jobs.submit("base_train", p)}

    @api.get("/api/jobs")
    def jobs():
        return A().jobs.list()

    @api.get("/api/jobs/{jid}")
    def job(jid: int):
        j = A().jobs.get(jid)
        if not j:
            raise HTTPException(404, "no such job")
        return j

    @api.post("/api/jobs/{jid}/{action}")
    def job_action(jid: int, action: str):
        a = A()
        fn = {"pause": a.jobs.pause, "resume": a.jobs.resume, "cancel": a.jobs.cancel}.get(action)
        if not fn:
            raise HTTPException(404, "unknown action")
        try:
            fn(jid)
        except KeyError:
            raise HTTPException(404, "no such job")
        except ValueError as e:
            raise HTTPException(409, str(e))
        return a.jobs.get(jid)

    # -------------------------------------------------------------- learning
    @api.get("/api/goals")
    def goals():
        return A().learning.goals()

    @api.post("/api/goals")
    def add_goal(body: GoalIn):
        return {"id": A().learning.add_goal(body.title, body.description, body.keywords)}

    @api.post("/api/goals/{gid}/status/{st}")
    def goal_status(gid: int, st: str):
        try:
            A().learning.set_goal_status(gid, st)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @api.post("/api/learning/cycle")
    def cycle():
        return A().learning.run_cycle(actor="user")

    @api.get("/api/learning/last")
    def last_cycle():
        a = A()
        return {"last": a.learning.last_cycle,
                "history": a.db.query("SELECT * FROM audit WHERE action='learning_cycle' ORDER BY id DESC LIMIT 10")}

    # ---------------------------------------------------------------- memory
    @api.get("/api/memory/facts")
    def facts(kind: str | None = None, status: str | None = None):
        return A().memory.list_facts(kind, status)

    @api.post("/api/memory/facts")
    def add_fact(body: FactIn):
        try:
            return A().memory.add_fact(body.text, body.kind, body.source, body.confidence)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.patch("/api/memory/facts/{fid}")
    def patch_fact(fid: int, body: FactPatch):
        try:
            return A().memory.update_fact(fid, body.kind, body.status, body.confidence)
        except KeyError as e:
            _404(e)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.delete("/api/memory/facts/{fid}")
    def del_fact(fid: int):
        A().memory.delete_fact(fid)
        return {"ok": True}

    @api.get("/api/memory/notes")
    def notes():
        return A().memory.list_notes()

    @api.post("/api/memory/notes")
    def add_note(body: NoteIn):
        return A().memory.add_note(body.body, body.title, body.source)

    @api.delete("/api/memory/notes/{nid}")
    def del_note(nid: int):
        A().memory.delete_note(nid)
        return {"ok": True}

    @api.get("/api/memory/search")
    def search(q: str):
        return A().memory.search(q)

    @api.post("/api/memory/wipe")
    def wipe(confirm: str):
        if confirm != "DELETE":
            raise HTTPException(400, "pass confirm=DELETE")
        A().memory.wipe_all()
        return {"ok": True}

    # -------------------------------------------------------------- datasets
    @api.get("/api/datasets")
    def datasets():
        return A().datasets.list()

    @api.get("/api/datasets/{did}/documents")
    def dataset_docs(did: str, limit: int = 50):
        a = A()
        if not a.datasets.get(did):
            raise HTTPException(404, "no such dataset")
        return [{**d, "text": d["text"][:1500]} for d in a.datasets.documents(did, limit)]

    @api.get("/api/examples")
    def examples(status: str | None = None):
        return A().learning.list_examples(status)

    @api.post("/api/examples")
    def add_example(body: ExampleIn):
        if body.prompt and body.answer:
            text = format_exchange(body.prompt, body.answer)
        elif body.text:
            text = body.text
        else:
            raise HTTPException(400, "give text, or prompt + answer")
        a = A()
        return a.learning.add_example(text, "user_added", "added in Dataset manager",
                                      eval_shingles=a.evals.all_shingles())

    @api.post("/api/examples/from_message/{mid}")
    def example_from_message(mid: int):
        a = A()
        m = a.db.one("SELECT * FROM messages WHERE id=? AND role='assistant'", (mid,))
        if not m:
            raise HTTPException(404, "no such assistant message")
        prev = a.db.one("SELECT content FROM messages WHERE conversation_id=? AND id<? AND role='user' "
                        "ORDER BY id DESC LIMIT 1", (m["conversation_id"], mid))
        return a.learning.add_example(format_exchange(prev["content"] if prev else "", m["content"]),
                                      "generated", f"model output, message #{mid}")

    @api.post("/api/examples/{eid}/review")
    def review(eid: int, accept: bool):
        try:
            A().learning.review_example(eid, accept)
        except KeyError as e:
            _404(e)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @api.delete("/api/examples/{eid}")
    def del_example(eid: int):
        try:
            A().learning.delete_example(eid)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    # ------------------------------------------------------------ evaluation
    @api.get("/api/evals")
    def evals():
        a = A()
        return {"sets": a.evals.list(), "integrity": a.evals.verify_all()}

    @api.get("/api/models")
    def models():
        return A().registry.list()

    @api.post("/api/models/{vid}/evaluate")
    def evaluate(vid: str):
        a = A()
        try:
            m = a.registry.load(vid)
        except KeyError as e:
            _404(e)
        from ..lm.tokenizer import BPETokenizer
        try:
            res = evaluate_all(m, BPETokenizer.load(a.paths.tokenizer), a.evals)
        except TamperError as e:
            raise HTTPException(409, str(e))
        a.db.audit("user", "manual_evaluation", id=vid)
        return res

    @api.get("/api/candidates")
    def candidates():
        return [v for v in A().registry.list() if v["report"] and
                v["status"] in ("pending_approval", "rejected", "active", "approved")]

    @api.post("/api/candidates/{vid}/decide")
    def decide(vid: str, body: DecideIn):
        try:
            return A().learning.decide(vid, body.approve, body.note)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @api.post("/api/models/{vid}/activate")
    def rollback(vid: str):
        try:
            return A().learning.rollback(vid)
        except KeyError as e:
            _404(e)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(409, str(e))

    @api.delete("/api/models/{vid}")
    def delete_model(vid: str):
        try:
            A().registry.delete_version(vid)
        except KeyError as e:
            _404(e)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @api.get("/api/backups")
    def backups():
        a = A()
        return [{"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2), "created": p.stat().st_mtime}
                for p in sorted(a.paths.backups.glob("*.pt"), reverse=True)]

    # ---------------------------------------------------------------- images
    @api.get("/api/images/datasets")
    def img_datasets():
        return A().images.datasets()

    @api.post("/api/images/datasets")
    def img_prepare(body: ImageDataIn):
        try:
            return A().images.prepare(body.source, body.name, body.n, body.folder)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))

    @api.post("/api/images/train")
    def img_train(body: ImageTrainIn):
        a = A()
        if not (a.paths.image_data / f"{body.dataset}.npz").exists():
            raise HTTPException(404, "no such image dataset")
        p = {"dataset": body.dataset}
        if body.steps:
            p["steps"] = body.steps
        if body.resume_from:
            p["resume_from"] = body.resume_from
        if body.max_minutes:
            p["max_minutes"] = body.max_minutes
        return {"job_id": a.jobs.submit("image_train", p)}

    @api.get("/api/images/versions")
    def img_versions():
        return [{**v, "metrics": {k: x for k, x in v["metrics"].items() if k != "history"},
                 "history": v["metrics"].get("history")} for v in A().images.versions()]

    @api.post("/api/images/versions/{vid}/activate")
    def img_activate(vid: str):
        try:
            A().images.activate(vid)
        except KeyError as e:
            _404(e)
        return {"ok": True}

    @api.post("/api/images/generate")
    def img_generate(body: GenerateIn):
        try:
            return A().images.generate(body.prompt, body.n, body.steps, body.guidance, body.seed)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @api.get("/generated/{folder}/{name}")
    def generated_sub(folder: str, name: str):
        if not re.fullmatch(r"video_\d+", folder) or not re.fullmatch(r"[\w.-]+\.png", name):
            raise HTTPException(400, "bad file name")
        p = A().paths.generated / folder / name
        if not p.exists():
            raise HTTPException(404, "not found")
        return FileResponse(p, media_type="image/png")

    @api.get("/generated/{name}")
    def generated(name: str):
        if not re.fullmatch(r"[\w.-]+\.png", name):
            raise HTTPException(400, "bad file name")
        p = A().paths.generated / name
        if not p.exists():
            raise HTTPException(404, "not found")
        return FileResponse(p, media_type="image/png")

    # -------------------------------------------------------- system & audit
    @api.get("/api/hardware")
    def hardware():
        a = A()
        snap = snapshot(str(a.paths.root))
        snap["storage_mb"] = {k: folder_size_mb(getattr(a.paths, k).parent if k in ("db", "tokenizer")
                                                else getattr(a.paths, k))
                              for k in ("models", "backups", "datasets", "image_data", "image_models",
                                        "generated", "job_ckpt", "db")}
        snap["limits"] = a.cfg()["resources"]
        return snap

    def masked(cfg):
        out = json.loads(json.dumps(cfg))
        if out.get("auth", {}).get("google_client_secret"):
            out["auth"]["google_client_secret"] = MASK
        return out

    @api.get("/api/settings")
    def get_settings():
        return masked(A().cfg())

    @api.put("/api/settings")
    def put_settings(body: dict):
        a = A()
        for f in body.get("learning", {}).get("approved_folders", []) or []:
            if not os.path.isdir(f):
                raise HTTPException(400, f"folder does not exist: {f}")
        pe = body.get("persona") or {}
        if "style" in pe and pe["style"] not in STYLES:
            raise HTTPException(400, f"style must be one of: {', '.join(STYLES)}")
        if len(pe.get("context") or "") > MAX_CONTEXT_CHARS:
            raise HTTPException(400, f"personality description: at most {MAX_CONTEXT_CHARS} characters")
        if "name" in pe and not (1 <= len(pe["name"].strip()) <= 30):
            raise HTTPException(400, "name: 1-30 characters")
        if (body.get("ui") or {}).get("language", "pl") not in LANGUAGES:
            raise HTTPException(400, f"language must be one of: {', '.join(LANGUAGES)}")
        asg = body.get("assistant") or {}
        if asg.get("backend") not in (None, "builtin", "ollama"):
            raise HTTPException(400, "backend must be builtin or ollama")
        if "ollama_url" in asg:
            from ..assistant import AssistantError, check_local_url
            try:
                check_local_url(asg["ollama_url"])
            except AssistantError as e:
                raise HTTPException(400, str(e))
        it = body.get("internet") or {}
        for u in it.get("web_sources", []) or []:
            if not re.match(r"^https?://", u):
                raise HTTPException(400, f"web address must start with http:// or https://: {u}")
        if it.get("web_sources"):
            from ..webaccess import domain_allowed
            sites = it.get("allowed_sites", a.cfg()["internet"]["allowed_sites"])
            bad = [u for u in it["web_sources"] if not domain_allowed(u, sites)]
            if bad:
                raise HTTPException(400, f"add these sites to 'allowed sites' first: {', '.join(bad)}")
        if "wikipedia_language" in it and not re.fullmatch(r"[a-z\-]{2,12}", it["wikipedia_language"]):
            raise HTTPException(400, "Wikipedia language: a code like pl, en, de")
        au = body.get("auth")
        if au is not None:
            if au.get("google_client_secret") == MASK:
                au.pop("google_client_secret")
            merged = {**a.cfg()["auth"], **au}
            if merged.get("enabled"):
                emails = [e for e in merged.get("allowed_emails", []) if "@" in e]
                if not merged.get("google_client_id") or not emails:
                    raise HTTPException(400, "To turn on Google login, enter the OAuth client ID and at "
                                             "least one allowed e-mail address (otherwise you would be locked out).")
        try:
            save_settings(body, a.paths)
        except ValueError as e:
            raise HTTPException(400, str(e))
        a.reload_config()
        a.db.audit("user", "settings_changed", sections=list(body))
        return masked(a.cfg())

    @api.post("/api/shutdown")
    def shutdown(request: Request):
        """Used by stop.ps1: checkpoint the running job, then exit the process."""
        if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(403, "shutdown is only allowed from this computer")
        a = A()

        def stop():
            time.sleep(0.5)
            a.shutdown()
            (a.paths.root / "nexora.pid").unlink(missing_ok=True)
            os._exit(0)
        if nexora is None:  # never kill the test process
            threading.Thread(target=stop, daemon=True).start()
        return {"ok": True, "message": "NEXORA is saving checkpoints and shutting down."}

    @api.get("/api/audit")
    def audit(limit: int = 100):
        return A().db.query("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (min(limit, 1000),))

    # ------------------------------------------------------------ frontend
    @api.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    api.mount("/static", StaticFiles(directory=WEB), name="static")
    return api
