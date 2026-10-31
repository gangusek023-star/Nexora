"""Nexo 1.2: an agent that talks, writes code and works inside Roblox Studio and Blender.

It reports what it is doing at every moment ("thoughts"): searching its notes,
looking at the scene, writing code (streamed live), waiting for your approval,
running the code, reading errors and fixing them, rendering the result and -
if a local vision model is installed - looking at the render to check its work.

The language and code models are ready-made open models that NEXORA runs
itself (built-in llama.cpp engine, see localmodels.py) on this computer. What Nexo 1.2 "learns" (lessons, videos, good examples) is
stored in the knowledge base and read into the prompt; the models' weights do
not change.
"""
from __future__ import annotations

import re
import threading
import time
import uuid

from . import assistant as asst
from .bridge import BridgeError
from .render3d import render_scene
from .video import describe_image, save_png

TARGET_WORDS = {
    "roblox": re.compile(r"\b(roblox|studio|luau|lua|obby|leaderstats|workspace|part[ys]?|gr[aęy] w roblox)\b", re.I),
    "blender": re.compile(r"\b(blender\w*|bpy|model\w* 3d|3d model\w*|render\w*|siatk\w+|mesh|modeluj\w*|modelowani\w*|wymodeluj\w*)\b", re.I),
}
RISKY = re.compile(r"\b(import\s+(os|subprocess|shutil|socket|requests|ctypes)|__import__|subprocess|shutil\.|"
                   r"os\.(remove|unlink|system|rmdir|popen)|rmtree|eval\(|exec\(|urllib|open\([^)]*['\"][wa]|"
                   r"InsertService|HttpService|LoadAsset|require\(\d+\))", re.I)
CODE_BLOCK = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.S)

APP_NAMES = {"roblox": "Roblox Studio", "blender": "Blender"}
LANG = {"roblox": "lua", "blender": "python"}

ROBLOX_RULES = """Jesteś {name} (Nexo 1.2) – agentem, który buduje w Roblox Studio. Twój kod Luau wykonuje wtyczka w Roblox Studio w trybie edycji (gra nie jest uruchomiona).
Zawsze odpowiadaj w tym formacie:
PLAN: 1-3 krótkie zdania po polsku, co zrobisz.
```lua
-- kompletny kod do wykonania
```
Zasady:
- Twórz obiekty przez Instance.new, ustaw właściwości, a Parent ustaw na końcu. Części ustawiaj Anchored = true.
- Części jednego obiektu grupuj w Model z czytelną nazwą. Y to wysokość; podłoże jest na Y = 0.
- Nie usuwaj ani nie zmieniaj istniejących obiektów, chyba że użytkownik o to prosi.
- Kod budujący nie może mieć nieskończonych pętli ani task.wait. Zachowanie w trakcie gry wstawiaj jako Script lub LocalScript ustawiając Source = [[ ... ]].
- Na końcu dodaj print() z krótkim podsumowaniem, co powstało.
- Jeśli prośba nie wymaga zmian w Studio (np. pytanie o Lua), odpowiedz zwykłym tekstem, bez bloku kodu."""

BLENDER_RULES = """Jesteś {name} (Nexo 1.2) – agentem, który modeluje w Blenderze. Twój kod Python (bpy) wykonuje dodatek w otwartym Blenderze.
Zawsze odpowiadaj w tym formacie:
PLAN: 1-3 krótkie zdania po polsku, co zrobisz.
```python
# kompletny kod do wykonania
```
Zasady:
- Używaj bpy (moduły bpy, bmesh, math i mathutils są już zaimportowane). Jednostka to metr, oś Z jest pionowa, obiekty stawiaj na Z = 0.
- Po bpy.ops.mesh.primitive_*_add pobieraj obiekt przez bpy.context.active_object i nadawaj mu czytelną nazwę.
- Kolor ustawiaj w materiale: Principled BSDF "Base Color" oraz mat.diffuse_color (podgląd).
- Używaj modyfikatorów (Bevel, Subdivision, Array, Mirror, Boolean, Solidify), żeby model wyglądał dobrze.
- Nie usuwaj istniejących obiektów, chyba że użytkownik o to prosi. Nie zapisuj plików, chyba że użytkownik poda ścieżkę.
- Na końcu dodaj print() z krótkim podsumowaniem, co powstało.
- Jeśli prośba nie wymaga zmian w Blenderze (np. pytanie), odpowiedz zwykłym tekstem, bez bloku kodu."""


class AgentRuns:
    """Pending approvals and stop requests for running agent sessions."""

    def __init__(self):
        self._lock = threading.Lock()
        self.runs: dict[str, dict] = {}

    def new(self) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock:
            self.runs[rid] = {"event": threading.Event(), "decision": None, "stop": False, "created": time.time()}
            for k in [k for k, v in self.runs.items() if time.time() - v["created"] > 3600]:
                self.runs.pop(k, None)
        return rid

    def decide(self, rid: str, approve: bool) -> bool:
        r = self.runs.get(rid)
        if not r:
            return False
        r["decision"] = approve
        r["event"].set()
        return True

    def stop(self, rid: str) -> bool:
        r = self.runs.get(rid)
        if not r:
            return False
        r["stop"] = True
        r["event"].set()
        return True

    def wait_decision(self, rid: str, timeout: float) -> bool | None:
        r = self.runs[rid]
        r["event"].clear()
        r["decision"] = None
        r["event"].wait(timeout)
        return None if r["stop"] else r["decision"]

    def stopped(self, rid: str) -> bool:
        return self.runs.get(rid, {}).get("stop", False)


def pick_target(message: str, requested: str, bridge) -> str:
    if requested in ("chat", "roblox", "blender"):
        return requested
    for app in ("roblox", "blender"):
        if TARGET_WORDS[app].search(message) and bridge.online(app):
            return app
    online = [a for a in ("roblox", "blender") if bridge.online(a)]
    if len(online) == 1 and re.search(r"\b(zbuduj|stwórz|stworz|dodaj|zrób|zrob|postaw|zmień|zmien|usuń|usun|"
                                       r"przesuń|przesun|pokoloruj|build|make|create|add)\b", message, re.I):
        return online[0]
    return "chat"


def split_reply(text: str, lang: str) -> tuple[str, str | None]:
    """-> (plan/explanation text, code or None)."""
    blocks = CODE_BLOCK.findall(text)
    code = None
    for tag, body in blocks:
        if tag.lower() in ("", lang, "luau", "lua", "py", "python3", "python") and body.strip():
            code = body.strip("\n")
            break
    plan = CODE_BLOCK.split(text)[0] if code else text
    plan = re.sub(r"^\s*(PLAN|Plan)\s*:\s*", "", plan.strip()).strip()
    return plan, code


def scene_text(app_name: str, snap: dict) -> str:
    if app_name == "roblox":
        lines = [f"STAN SCENY W ROBLOX STUDIO: {snap.get('total_parts', 0)} widocznych części."]
        if snap.get("tree"):
            lines.append("Obiekty w Workspace: " + "; ".join(snap["tree"][:25]))
        if snap.get("scripts"):
            lines.append("Skrypty: " + "; ".join(snap["scripts"][:15]))
        parts = [p for p in snap.get("parts", []) if max(p.get("s", [0])) < 400][-25:]  # skip baseplates
        if parts:
            def rgb(c):
                return "(" + ",".join(str(int(round(v * 255))) for v in c) + ")"
            lines.append("Części (nazwa: rozmiar, pozycja, kolor RGB): " + "; ".join(
                f"{p.get('n', '?')}: {[round(v, 1) for v in p['s']]}, {[round(v, 1) for v in p['p']]}, {rgb(p.get('col', [0, 0, 0]))}"
                for p in parts))
        return "\n".join(lines)[:2500]
    objs = snap.get("objects", [])
    desc = "; ".join(f"{o['name']} ({o['type']}, wymiary {o['dimensions']}, poz. {o['location']}"
                     + (f", materiały {o['materials']}" if o.get("materials") else "")
                     + (f", modyfikatory {o['modifiers']}" if o.get("modifiers") else "") + ")" for o in objs[:30])
    return f"STAN SCENY W BLENDERZE ({snap.get('blender', '')}): {snap.get('total_objects', 0)} obiektów. {desc}"[:2500]


def knowledge_text(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    body = "\n---\n".join(c["text"] for c in chunks)
    return ("TWOJE NOTATKI (lekcje i przykłady, których nauczył Cię użytkownik – korzystaj z nich, jeśli pasują):\n"
            f"---\n{body}\n---")


class Agent:
    def __init__(self, app):
        self.app = app
        self.runs = AgentRuns()

    # ----------------------------------------------------------------- helpers
    def _history(self, cid: int, before_id: int, n: int) -> list[dict]:
        rows = self.app.db.query("SELECT role, content FROM messages WHERE conversation_id=? AND id<? AND "
                                 "role IN ('user','assistant') ORDER BY id DESC LIMIT ?", (cid, before_id, n * 2))
        return [{"role": r["role"], "content": r["content"][:3000]} for r in rows[::-1]]

    def _save_image(self, png: bytes, run_id: str, n: int) -> str:
        name = f"agent_{run_id}_{n}.png"
        (self.app.paths.generated / name).write_bytes(png)
        return f"/generated/{name}"

    def _render(self, target: str, run_id: str, n: int, size: int) -> tuple[str | None, str]:
        b = self.app.bridge
        if target == "roblox":
            snap = b.call("roblox", "snapshot", timeout=60).get("snapshot") or {}
            png = render_scene(snap.get("parts", []))
            return self._save_image(png, run_id, n), f"{snap.get('total_parts', 0)} części (podgląd izometryczny)"
        res = b.call("blender", "render", timeout=180, size=size)
        if not res.get("ok"):
            raise BridgeError(res.get("error") or "render failed")
        name = f"agent_{run_id}_{n}.png"
        save_png(res["image_b64"], self.app.paths.generated / name)
        return f"/generated/{name}", f"render Blendera ({res.get('engine', '')}, {res.get('objects', 0)} obiektów)"

    # -------------------------------------------------------------------- run
    def run(self, message: str, conversation_id: int | None, attachments: list[dict], target_req: str):
        """Generator of event dicts (see module docstring)."""
        app = self.app
        acfg, gcfg = app.cfg()["assistant"], app.cfg()["agent"]
        persona = app.cfg()["persona"]
        t0 = time.time()
        run_id = self.runs.new()
        thoughts: list[dict] = []

        def ev(kind, **kw):
            e = {"type": kind, **kw}
            if kind in ("step", "code", "image", "think_block"):
                thoughts.append({**e, "t": round(time.time() - t0, 1)})
            return e

        cid = conversation_id or app.memory.new_conversation(message[:40])
        names = "".join(f"\n[file: {f['name']}]" for f in attachments)
        uid = app.memory.add_message(cid, "user", message.strip() + names)
        target = pick_target(message, target_req, app.bridge)
        lang = LANG.get(target, "")
        model = (acfg.get("code_model") or acfg.get("chat_model")) if target != "chat" else \
            asst.pick_model(acfg, message, attachments)
        if acfg.get("backend", "builtin") == "builtin":
            model = app.llm.pick(model)
        yield ev("start", conversation_id=cid, user_message_id=uid, run_id=run_id, model=model, target=target)
        if not model:
            yield ev("done", error="Nie pobrano jeszcze żadnego gotowego modelu. Otwórz stronę Modele AI i pobierz "
                                   "Bielika (rozmowa) albo Qwen Codera (kod).", text="", message_id=None)
            return

        # 1. notes
        topic = target if target != "chat" else None
        chunks = app.knowledge.search(message, limit=6, max_chars=int(gcfg.get("knowledge_chars", 3500)), topic=topic)
        if chunks:
            titles = sorted({c["title"] for c in chunks})
            yield ev("step", icon="search", text=f"Przeszukuję swoją wiedzę: {len(chunks)} fragmentów ({', '.join(titles)[:160]})")
        else:
            yield ev("step", icon="search", text="Przeszukuję swoją wiedzę: nic pasującego, odpowiadam z tego, co wiem")

        # ---------------------------------------------------------- chat only
        if target == "chat":
            yield ev("step", icon="think", text=f"Piszę odpowiedź (model {model})")
            system = asst.system_prompt(persona, app.cfg()["ui"]["language"], acfg.get("extra_instructions", ""))
            kt = knowledge_text(chunks)
            if kt:
                system += "\n\n" + kt
            msgs = asst.build_messages(system, self._history(cid, uid, int(acfg.get("history_turns", 6))),
                                       message, attachments, int(acfg.get("max_file_chars", 12000)))
            parts, err = [], None
            try:
                for piece in app.llm.stream_chat(model, msgs, asst.temperature_for(persona, acfg)):
                    if self.runs.stopped(run_id):
                        err = "Zatrzymano."
                        break
                    parts.append(piece)
                    yield ev("delta", text=piece)
            except asst.AssistantError as e:
                err = str(e)
            text = asst.blocked_filter("".join(parts).strip(), persona)
            mid = self._save(cid, text, model, thoughts, t0, target, message) if text else None
            yield ev("done", message_id=mid, text=text, error=err, seconds=round(time.time() - t0, 1))
            return

        # ------------------------------------------------------- app agent
        appname = APP_NAMES[target]
        b = app.bridge
        if not b.online(target):
            msg = (f"{appname} nie jest połączony. Zainstaluj {'wtyczkę' if target == 'roblox' else 'dodatek'} "
                   f"NEXORA (strona Połączenia) i uruchom {appname}.")
            yield ev("step", icon="error", text=msg)
            yield ev("done", message_id=None, text="", error=msg, seconds=round(time.time() - t0, 1))
            return
        yield ev("step", icon="look", text=f"Sprawdzam, co jest teraz w {appname}…")
        try:
            snap = b.call(target, "snapshot", timeout=60).get("snapshot") or {}
            st = scene_text(target, snap)
            yield ev("step", icon="look", text=st[:300])
        except BridgeError as e:
            st = ""
            yield ev("step", icon="error", text=f"Nie udało się odczytać sceny: {e}")

        rules = (ROBLOX_RULES if target == "roblox" else BLENDER_RULES).format(name=persona.get("name") or "NEXORA")
        if acfg.get("extra_instructions"):
            rules += "\n" + acfg["extra_instructions"]
        if persona.get("context"):
            rules += "\nOpis zachowania od użytkownika: " + persona["context"]
        system = "\n\n".join(x for x in (rules, knowledge_text(chunks), st) if x)
        msgs = asst.build_messages(system, self._history(cid, uid, 3), message, attachments,
                                   int(acfg.get("max_file_chars", 12000)))
        max_attempts = max(1, int(gcfg.get("max_attempts", 3)))
        vision = gcfg.get("vision_model") if acfg.get("backend") == "ollama" else ""
        look_rounds = 1 if gcfg.get("look_at_result", True) else 0
        attempt, final_plan, final_code, final_out, err, verdict = 0, "", None, "", None, ""
        pending_reply = None
        image_n = 0
        while attempt < max_attempts:
            if self.runs.stopped(run_id):
                err = "Zatrzymano."
                break
            attempt += 1
            # --- think & write code (streamed live into the thoughts panel)
            if pending_reply is None:
                yield ev("step", icon="think", text="Myślę i piszę kod" + (f" (próba {attempt}/{max_attempts})" if attempt > 1 else ""))
                parts = []
                try:
                    for piece in app.llm.stream_chat(model, msgs, 0.2):
                        if self.runs.stopped(run_id):
                            break
                        parts.append(piece)
                        yield ev("think", delta=piece)
                except asst.AssistantError as e:
                    err = str(e)
                    yield ev("step", icon="error", text=err)
                    break
                reply = "".join(parts)
            else:
                reply, pending_reply = pending_reply, None
            yield ev("think_block", text=reply)
            plan, code = split_reply(reply, lang)
            msgs.append({"role": "assistant", "content": reply})
            if not code:  # an answer without code (question, explanation, or "done")
                final_plan = plan
                break
            final_plan = plan or final_plan
            yield ev("code", lang=lang, code=code, step=attempt)
            # --- approval
            risky = bool(RISKY.search(code))
            if risky:
                yield ev("step", icon="warn", text="Kod używa plików, sieci lub innych ryzykownych funkcji – potrzebna Twoja zgoda.")
            if not gcfg.get("auto_run") or risky:
                yield ev("approve", run_id=run_id, step=attempt, lang=lang, code=code, risky=risky)
                yield ev("step", icon="wait", text=f"Czekam na Twoją zgodę na uruchomienie kodu w {appname}…")
                decision = self.runs.wait_decision(run_id, float(gcfg.get("approval_timeout_s", 900)))
                if not decision:
                    err = "Nie uruchomiono kodu (brak zgody)." if not self.runs.stopped(run_id) else "Zatrzymano."
                    yield ev("step", icon="error", text=err)
                    final_code = code
                    break
            # --- run
            yield ev("step", icon="run", text=f"Uruchamiam kod w {appname}…")
            try:
                res = b.call(target, "run", timeout=float(gcfg.get("run_timeout_s", 120)), code=code)
            except BridgeError as e:
                err = str(e)
                yield ev("step", icon="error", text=err)
                break
            output = (res.get("output") or "").strip()
            if res.get("ok"):
                final_code, final_out = code, output
                yield ev("step", icon="ok", text="Kod zadziałał." + (f" Wyjście: {output[:300]}" if output else ""))
            else:
                error_text = (res.get("error") or "unknown error").strip()
                yield ev("step", icon="error", text=f"Błąd: {error_text[:400]}")
                if attempt >= max_attempts:
                    err = f"Nie udało się po {max_attempts} próbach. Ostatni błąd: {error_text[:300]}"
                    final_code = code
                    break
                yield ev("step", icon="retry", text="Czytam błąd i poprawiam kod…")
                msgs.append({"role": "user", "content": f"BŁĄD WYKONANIA:\n{error_text[:1500]}\n"
                                                        f"{('Wyjście: ' + output[:500]) if output else ''}\n"
                                                        + ("Zmiany z tej próby zostały cofnięte. " if target == "roblox" else
                                                           "Nowe obiekty z tej próby zostały usunięte. ")
                                                        + "Popraw kod. Ten sam format: PLAN i jeden blok kodu."})
                continue
            # --- look at the result
            if gcfg.get("render_after_run", True):
                yield ev("step", icon="render", text="Renderuję to, co mam przed sobą…")
                try:
                    image_n += 1
                    url, cap = self._render(target, run_id, image_n, int(gcfg.get("render_size", 480)))
                    yield ev("image", url=url, caption=cap)
                except BridgeError as e:
                    url = None
                    yield ev("step", icon="error", text=f"Render się nie udał: {e}")
                if look_rounds > 0 and attempt < max_attempts:
                    look_rounds -= 1
                    desc = ""
                    if vision and url:
                        yield ev("step", icon="look", text=f"Oglądam render ({vision})…")
                        try:
                            desc = describe_image(acfg, vision, app.paths.generated / url.rsplit("/", 1)[1],
                                                  "To render sceny 3D zbudowanej na prośbę użytkownika. Opisz po polsku "
                                                  "w 2-4 zdaniach, co dokładnie widać: kształty, kolory, ułożenie. "
                                                  "Nie zgaduj tego, czego nie widać.")
                        except Exception as e:  # noqa: BLE001
                            yield ev("step", icon="error", text=f"Model wizyjny nie odpowiedział: {e}")
                        if desc:
                            yield ev("step", icon="look", text=f"Widzę: {desc[:500]}")
                    else:  # no vision model: check the measured scene (names, sizes, positions, colours)
                        yield ev("step", icon="look", text="Sprawdzam, co powstało w scenie…")
                        try:
                            snap2 = b.call(target, "snapshot", timeout=60).get("snapshot") or {}
                            desc = scene_text(target, snap2)
                            yield ev("step", icon="look", text=desc[:300])
                        except BridgeError as e:
                            yield ev("step", icon="error", text=f"Nie udało się odczytać sceny: {e}")
                    if desc:
                        if attempt < max_attempts:
                            yield ev("step", icon="think", text="Porównuję wynik z prośbą…")
                            msgs.append({"role": "user", "content":
                                         f"Kod zadziałał. {'Opis renderu' if vision else 'Scena po zmianach'}: {desc}\nCzy wynik spełnia prośbę: \"{message}\"? "
                                         "Jeśli tak, odpowiedz tylko: GOTOWE: <jedno zdanie podsumowania>. Jeśli nie, "
                                         "napisz PLAN i blok kodu, który POPRAWIA scenę (bez budowania wszystkiego od nowa)."})
                            parts = []
                            try:
                                for piece in app.llm.stream_chat(model, msgs, 0.2):
                                    parts.append(piece)
                                    yield ev("think", delta=piece)
                            except asst.AssistantError as e:
                                yield ev("step", icon="error", text=str(e))
                                break
                            reply = "".join(parts)
                            if not CODE_BLOCK.search(reply):  # "GOTOWE…" or any answer without new code: accept
                                verdict = re.sub(r"(?i)^\s*gotowe\s*:?\s*", "", reply.strip())
                                yield ev("think_block", text=reply)
                                yield ev("step", icon="ok", text="Wynik zgadza się z prośbą.")
                                break
                            pending_reply = reply
                            yield ev("step", icon="retry", text="Poprawiam wynik…")
                            continue
            break

        # final message
        parts = []
        if final_plan:
            parts.append(final_plan)
        if verdict:
            parts.append(verdict)
        if final_code:
            parts.append(f"```{lang}\n{final_code}\n```")
        if final_out:
            parts.append(f"Wynik w {appname}: {final_out[:600]}")
        text = asst.blocked_filter("\n\n".join(parts).strip(), persona)
        ok_run = bool(final_code and final_out is not None and not err)
        mid = self._save(cid, text or (err or ""), model, thoughts, t0, target, message,
                         code=final_code if ok_run else None, lang=lang) if (text or err) else None
        yield ev("done", message_id=mid, text=text, error=err, seconds=round(time.time() - t0, 1))

    def _save(self, cid, text, model, thoughts, t0, target, task, code=None, lang=""):
        meta = {"engine": "nexo-1.2", "model": model, "target": target, "thoughts": thoughts,
                "seconds": round(time.time() - t0, 1), "task": task[:2000]}
        if code:
            meta.update(code=code, lang=lang)
        return self.app.memory.add_message(cid, "assistant", text, f"nexo-1.2:{model}", meta)

    # ------------------------------------------------------ learn from success
    def save_example(self, message_id: int) -> dict:
        import json
        m = self.app.db.one("SELECT * FROM messages WHERE id=? AND role='assistant'", (message_id,))
        if not m:
            raise KeyError(message_id)
        meta = json.loads(m["meta"] or "{}")
        if not meta.get("code"):
            raise ValueError("this answer has no working code to remember")
        target = meta.get("target", "")
        text = (f"# Przykład ({APP_NAMES.get(target, target)}): {meta.get('task', '')[:200]}\n\n"
                f"Zadanie: {meta.get('task', '')}\n\nKod, który zadziałał:\n```{meta.get('lang', '')}\n{meta['code']}\n```")
        return self.app.knowledge.add(f"Przykład: {meta.get('task', '')[:80]}", text, "example",
                                      f"message #{message_id}", target)
