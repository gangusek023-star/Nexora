"""Isometric preview of a Roblox scene from the part list the Studio plugin sends.

Roblox Studio plugins cannot take screenshots, so NEXORA draws what is in
front of the agent itself: every visible part becomes a shaded box (or a disc
for balls) seen from an isometric camera, painted back to front.
"""
from __future__ import annotations

import io
import math

from PIL import Image, ImageDraw

C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))
# camera looks from (+x, +y, +z) towards the origin; light from upper left
LIGHT = (-0.4, 0.8, 0.45)


def _proj(x, y, z):
    return (x - z) * C30, (x + z) * S30 - y


def _corners(p):
    (px, py, pz), (sx, sy, sz) = p["p"], p["s"]
    r = p.get("r") or [1, 0, 0, 0, 1, 0, 0, 0, 1]
    out = []
    for dx in (-0.5, 0.5):
        for dy in (-0.5, 0.5):
            for dz in (-0.5, 0.5):
                lx, ly, lz = dx * sx, dy * sy, dz * sz
                out.append((px + r[0] * lx + r[1] * ly + r[2] * lz,
                            py + r[3] * lx + r[4] * ly + r[5] * lz,
                            pz + r[6] * lx + r[7] * ly + r[8] * lz))
    return out


# faces as indices into the 8 corners (dx,dy,dz bits: index = dx*4 + dy*2 + dz)
FACES = [((4, 5, 7, 6), (1, 0, 0)), ((0, 2, 3, 1), (-1, 0, 0)), ((2, 6, 7, 3), (0, 1, 0)),
         ((0, 1, 5, 4), (0, -1, 0)), ((1, 3, 7, 5), (0, 0, 1)), ((0, 4, 6, 2), (0, 0, -1))]
VIEW = (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))


def _rot(r, v):
    return (r[0] * v[0] + r[1] * v[1] + r[2] * v[2], r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
            r[6] * v[0] + r[7] * v[1] + r[8] * v[2])


def render_scene(parts: list[dict], size=(640, 480), max_parts: int = 1500) -> bytes:
    """parts: [{"p":[x,y,z], "s":[sx,sy,sz], "r":[9 rotation numbers], "col":[r,g,b 0-1], "t":transp, "sh":shape}]"""
    W, H = size
    img = Image.new("RGB", (W, H), (236, 240, 245))
    parts = [p for p in parts[:max_parts] if p.get("t", 0) < 0.98]
    if not parts:
        d = ImageDraw.Draw(img)
        d.text((W // 2 - 60, H // 2), "(empty scene)", fill=(120, 130, 140))
        return _png(img)
    # ignore huge baseplates when fitting the view, but still draw them
    vol = sorted(p["s"][0] * p["s"][2] for p in parts)
    big = vol[int(len(vol) * 0.95)] * 20 if len(vol) > 3 else float("inf")
    fit = [p for p in parts if p["s"][0] * p["s"][2] <= big] or parts
    pts = [_proj(*c) for p in fit for c in _corners(p)]
    xs, ys = [a for a, _ in pts], [b for _, b in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    scale = min((W - 40) / max(maxx - minx, 1e-3), (H - 40) / max(maxy - miny, 1e-3))
    ox, oy = W / 2 - (minx + maxx) / 2 * scale, H / 2 - (miny + maxy) / 2 * scale

    def S(x, y, z):
        a, b = _proj(x, y, z)
        return a * scale + ox, b * scale + oy

    polys = []
    for p in parts:
        col = tuple(int(max(0, min(1, c)) * 255) for c in p.get("col", (0.6, 0.6, 0.6)))
        r = p.get("r") or [1, 0, 0, 0, 1, 0, 0, 0, 1]
        if p.get("sh") == "Ball":
            cx, cy = S(*p["p"])
            rad = max(p["s"]) / 2 * scale
            depth = sum(p["p"])
            polys.append((depth, "ball", (cx, cy, rad), col))
            continue
        cs = _corners(p)
        for idx, n in FACES:
            nw = _rot(r, n)
            facing = nw[0] * VIEW[0] + nw[1] * VIEW[1] + nw[2] * VIEW[2]
            if facing <= 0:
                continue
            light = 0.55 + 0.45 * max(0.0, nw[0] * LIGHT[0] + nw[1] * LIGHT[1] + nw[2] * LIGHT[2])
            shade = tuple(int(c * light) for c in col)
            depth = sum(sum(cs[i]) for i in idx) / 4
            polys.append((depth, "face", [S(*cs[i]) for i in idx], shade))
    polys.sort(key=lambda t: t[0])
    d = ImageDraw.Draw(img)
    for _, kind, geo, colr in polys:
        if kind == "ball":
            cx, cy, rad = geo
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=colr, outline=tuple(int(c * 0.7) for c in colr))
        else:
            d.polygon(geo, fill=colr, outline=tuple(int(c * 0.75) for c in colr))
    return _png(img)


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()
