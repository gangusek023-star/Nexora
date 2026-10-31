"""NEXORA Bridge - Blender add-on.

Connects Blender to NEXORA (Nexo 1.2) running on this computer. Nexo 1.2 sends
Python (bpy) code; the add-on runs it on Blender's main thread with an undo
step, reports the printed output or the error, describes the scene and renders
a small preview image so the agent can see what it built.

Install: Edit > Preferences > Add-ons > (arrow menu) Install from Disk... > choose
this file > tick "NEXORA Bridge". Panel: 3D Viewport > sidebar (N) > NEXORA.
"""

bl_info = {
    "name": "NEXORA Bridge",
    "author": "NEXORA",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > NEXORA",
    "description": "Lets Nexo 1.2 (local AI) build and render models in Blender",
    "category": "Development",
}

import base64
import contextlib
import io
import json
import math
import os
import queue
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request

import bpy

SERVER = "http://127.0.0.1:{{PORT}}"
TOKEN = "{{TOKEN}}"
VERSION = "1.0"

_inbox: "queue.Queue[dict]" = queue.Queue()
_state = {"running": False, "connected": False, "last_error": "", "thread": None, "done": 0}


# ------------------------------------------------------------------ actions
_TRACKED = ("objects", "meshes", "materials", "lights", "cameras", "curves", "collections", "node_groups", "images")


def _datablock_names() -> dict:
    return {k: set(getattr(bpy.data, k).keys()) for k in _TRACKED}


def _remove_new(before: dict) -> int:
    """Delete data-blocks created by failed code, so a retry starts from a clean scene."""
    removed = 0
    for k in _TRACKED:  # objects first, then the data they used
        coll = getattr(bpy.data, k)
        for name in set(coll.keys()) - before[k]:
            block = coll.get(name)
            if block is None:
                continue
            try:
                if k == "objects":
                    coll.remove(block, do_unlink=True)
                else:
                    coll.remove(block)
                removed += 1
            except Exception:
                pass
    return removed


def execute_code(code: str) -> dict:
    """Run bpy code. Output is captured; errors include the traceback.
    If the code fails, everything it created is removed again."""
    try:
        bpy.ops.ed.undo_push(message="Nexo 1.2 (before)")
    except Exception:
        pass
    before = _datablock_names()
    out = io.StringIO()
    ns = {"bpy": bpy, "math": math, "__name__": "__nexo__"}
    try:
        import bmesh
        import mathutils
        ns.update(bmesh=bmesh, mathutils=mathutils, Vector=mathutils.Vector)
    except ImportError:
        pass
    ok, err = True, None
    with contextlib.redirect_stdout(out):
        try:
            exec(compile(code, "<nexo>", "exec"), ns)
        except Exception:
            ok, err = False, traceback.format_exc(limit=4)
    if not ok:
        n = _remove_new(before)
        if n:
            err += f"\n(Removed {n} new data-blocks created by this failed attempt.)"
    try:
        bpy.ops.ed.undo_push(message="Nexo 1.2")
    except Exception:
        pass
    return {"ok": ok, "output": out.getvalue()[-6000:], "error": err}


def scene_snapshot() -> dict:
    sc = bpy.context.scene
    objs = []
    for o in sc.objects:
        if len(objs) >= 150:
            break
        item = {"name": o.name, "type": o.type,
                "location": [round(v, 3) for v in o.location],
                "dimensions": [round(v, 3) for v in o.dimensions]}
        if o.type == "MESH":
            item["materials"] = [m.name for m in o.data.materials if m]
            item["modifiers"] = [f"{m.type}:{m.name}" for m in o.modifiers]
            item["vertices"] = len(o.data.vertices)
        if o.parent:
            item["parent"] = o.parent.name
        objs.append(item)
    return {"objects": objs, "total_objects": len(sc.objects), "blender": bpy.app.version_string,
            "file": bpy.data.filepath or "(unsaved)"}


def _visible_meshes(sc):
    kinds = {"MESH", "CURVE", "SURFACE", "META", "FONT"}
    return [o for o in sc.objects if o.type in kinds and o.visible_get() and not o.name.startswith("_nexo")]


def _sync_viewport_colors():
    """Workbench shows material.diffuse_color; copy Principled Base Color there when it is still default."""
    for m in bpy.data.materials:
        try:
            if tuple(round(c, 3) for c in m.diffuse_color) != (0.8, 0.8, 0.8, 1.0) or not m.node_tree:
                continue
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    m.diffuse_color = tuple(n.inputs["Base Color"].default_value)
                    break
        except Exception:
            continue


def render_preview(size: int = 480) -> dict:
    """Render all visible objects from an automatic 3/4 view to a PNG (base64)."""
    import mathutils
    sc = bpy.context.scene
    r = sc.render
    objs = _visible_meshes(sc)
    if not objs:
        return {"ok": False, "error": "the scene has no visible objects to render"}
    pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
    lo = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center, radius = (lo + hi) / 2, max((hi - lo).length / 2, 0.1)
    saved = {"engine": r.engine, "x": r.resolution_x, "y": r.resolution_y, "pct": r.resolution_percentage,
             "path": r.filepath, "fmt": r.image_settings.file_format, "camera": sc.camera}
    shading = sc.display.shading
    saved_shading = (shading.light, shading.color_type)
    cam_data = bpy.data.cameras.new("_nexo_cam")
    cam = bpy.data.objects.new("_nexo_cam", cam_data)
    sc.collection.objects.link(cam)
    temp = [cam]
    path = os.path.join(tempfile.gettempdir(), f"nexo_render_{os.getpid()}.png")
    try:
        cam_data.angle = math.radians(45)
        direction = mathutils.Vector((1.0, -1.25, 0.85)).normalized()
        cam.location = center + direction * (radius / math.sin(cam_data.angle / 2) * 1.08)
        cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam_data.clip_end = max(1000.0, radius * 20)
        sc.camera = cam
        r.resolution_x, r.resolution_y, r.resolution_percentage = size, int(size * 0.75), 100
        r.image_settings.file_format = "PNG"
        r.filepath = path
        _sync_viewport_colors()
        try:
            r.engine = "BLENDER_WORKBENCH"
            shading.light, shading.color_type = "STUDIO", "MATERIAL"
            bpy.ops.render.render(write_still=True)
            engine = "workbench"
        except Exception:
            r.engine = "CYCLES"  # fallback without GPU drawing: CPU path tracing at low quality
            sc.cycles.samples, sc.cycles.device = 12, "CPU"
            if not any(o.type == "LIGHT" for o in sc.objects):
                ld = bpy.data.lights.new("_nexo_sun", type="SUN")
                ld.energy = 3
                lo_ = bpy.data.objects.new("_nexo_sun", ld)
                sc.collection.objects.link(lo_)
                lo_.rotation_euler = (math.radians(50), 0, math.radians(30))
                temp.append(lo_)
            bpy.ops.render.render(write_still=True)
            engine = "cycles"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return {"ok": True, "image_b64": data, "engine": engine, "objects": len(objs)}
    except Exception:
        return {"ok": False, "error": traceback.format_exc(limit=3)}
    finally:
        sc.camera = saved["camera"]
        r.engine, r.resolution_x, r.resolution_y = saved["engine"], saved["x"], saved["y"]
        r.resolution_percentage, r.filepath, r.image_settings.file_format = saved["pct"], saved["path"], saved["fmt"]
        try:
            shading.light, shading.color_type = saved_shading
        except Exception:
            pass
        for o in temp:
            data_block = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if isinstance(data_block, bpy.types.Camera):
                bpy.data.cameras.remove(data_block)
            elif isinstance(data_block, bpy.types.Light):
                bpy.data.lights.remove(data_block)
        try:
            os.remove(path)
        except OSError:
            pass


def handle(cmd: dict) -> dict:
    kind = cmd.get("kind")
    if kind == "run":
        res = execute_code(cmd.get("code", ""))
    elif kind == "snapshot":
        res = {"ok": True, "snapshot": scene_snapshot()}
    elif kind == "render":
        res = render_preview(int(cmd.get("size", 480)))
    else:
        res = {"ok": False, "error": f"unknown command {kind}"}
    res["id"] = cmd.get("id")
    return res


# ------------------------------------------------------------------ network
def _request(method: str, path: str, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(SERVER + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-Nexora-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def send_result(result: dict):
    try:
        _request("POST", "/api/bridge/blender/result", result, timeout=30)
    except Exception as e:  # noqa: BLE001
        _state["last_error"] = f"could not send result: {e}"


def poll_once(wait: float = 10.0) -> dict | None:
    """One long-poll request. Returns a command or None."""
    ver = urllib.request.quote(bpy.app.version_string)
    data = _request("GET", f"/api/bridge/blender/poll?wait={wait}&v={VERSION}&blender={ver}", timeout=wait + 15)
    return data.get("command")


def _network_loop():
    while _state["running"]:
        try:
            cmd = poll_once()
            _state["connected"], _state["last_error"] = True, ""
            if cmd:
                _inbox.put(cmd)
        except Exception as e:  # noqa: BLE001
            _state["connected"], _state["last_error"] = False, str(e)
            time.sleep(5)


def _main_thread_tick():
    """Runs on Blender's main thread (bpy is not thread-safe)."""
    if not _state["running"]:
        return None
    try:
        while True:
            cmd = _inbox.get_nowait()
            result = handle(cmd)
            _state["done"] += 1
            threading.Thread(target=send_result, args=(result,), daemon=True).start()
    except queue.Empty:
        pass
    return 0.25


def start():
    if _state["running"]:
        return
    _state["running"] = True
    t = threading.Thread(target=_network_loop, name="nexora-bridge", daemon=True)
    _state["thread"] = t
    t.start()
    if not bpy.app.timers.is_registered(_main_thread_tick):
        bpy.app.timers.register(_main_thread_tick, first_interval=0.5, persistent=True)


def stop():
    _state["running"] = False
    _state["connected"] = False
    if bpy.app.timers.is_registered(_main_thread_tick):
        bpy.app.timers.unregister(_main_thread_tick)


# ------------------------------------------------------------------ UI
class NEXORA_OT_toggle(bpy.types.Operator):
    bl_idname = "nexora.toggle_bridge"
    bl_label = "Connect to NEXORA"
    bl_description = "Connect or disconnect Blender and Nexo 1.2"

    def execute(self, context):
        stop() if _state["running"] else start()
        return {"FINISHED"}


class NEXORA_PT_panel(bpy.types.Panel):
    bl_label = "NEXORA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NEXORA"

    def draw(self, context):
        col = self.layout.column()
        if _state["running"]:
            col.label(text="Connected" if _state["connected"] else "Connecting...",
                      icon="CHECKMARK" if _state["connected"] else "TIME")
            col.label(text=f"Commands done: {_state['done']}")
            if _state["last_error"]:
                col.label(text=_state["last_error"][:60], icon="ERROR")
            col.operator("nexora.toggle_bridge", text="Disconnect")
        else:
            col.label(text="Not connected")
            col.operator("nexora.toggle_bridge", text="Connect to NEXORA")
        col.label(text=SERVER)


_classes = (NEXORA_OT_toggle, NEXORA_PT_panel)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    start()


def unregister():
    stop()
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
