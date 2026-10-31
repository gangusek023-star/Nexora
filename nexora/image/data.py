"""Image-caption datasets.

Two legal sources are supported out of the box:
  1. `synthetic_shapes` - procedurally drawn shapes with exact captions.
     Generated locally, no copyright issues, small enough to learn on a CPU.
  2. A folder of your own (or openly licensed) images with captions:
       photo1.jpg + photo1.txt   (caption in a side-car text file), or
       captions.csv with lines   filename,caption
Prepared datasets are stored as compressed .npz (uint8 pixels + captions).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220),
          "yellow": (240, 210, 40), "white": (245, 245, 245), "black": (20, 20, 20),
          "purple": (140, 60, 180), "orange": (245, 140, 30)}
SHAPES = ["circle", "square", "triangle"]
POSITIONS = {"left": (0.3, 0.5), "right": (0.7, 0.5), "top": (0.5, 0.3), "bottom": (0.5, 0.7),
             "center": (0.5, 0.5)}
SIZES = {"small": 0.18, "large": 0.32}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def draw_shape(shape, color, bg, pos, size, res, jitter=0.0, rng=None):
    rng = rng or random.Random()
    s = 4  # draw at 4x then downsample for anti-aliasing
    R = res * s
    img = Image.new("RGB", (R, R), COLORS[bg])
    d = ImageDraw.Draw(img)
    cx, cy = POSITIONS[pos]
    cx += rng.uniform(-jitter, jitter); cy += rng.uniform(-jitter, jitter)
    rad = SIZES[size] * R * rng.uniform(0.9, 1.1)
    cx, cy = cx * R, cy * R
    if shape == "circle":
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=COLORS[color])
    elif shape == "square":
        r = rad * 0.85
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=COLORS[color])
    else:
        d.polygon([(cx, cy - rad), (cx - rad, cy + rad * 0.8), (cx + rad, cy + rad * 0.8)],
                  fill=COLORS[color])
    return img.resize((res, res), Image.LANCZOS)


def synthetic_shapes(n: int, res: int = 32, seed: int = 0):
    rng = random.Random(seed)
    images, caps = [], []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(list(COLORS))
        bg = rng.choice([c for c in COLORS if c != color])
        pos, size = rng.choice(list(POSITIONS)), rng.choice(list(SIZES))
        img = draw_shape(shape, color, bg, pos, size, res, jitter=0.04, rng=rng)
        images.append(np.asarray(img, dtype=np.uint8))
        caps.append(f"a {size} {color} {shape} on a {bg} background at the {pos}")
    return np.stack(images), caps


def _center_crop_resize(img: Image.Image, res: int) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    return np.asarray(img.resize((res, res), Image.LANCZOS), dtype=np.uint8)


def load_folder(folder: str | Path, res: int, max_images: int = 20000):
    """Read images + captions from a folder (side-car .txt or captions.csv)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    table = {}
    csv_path = folder / "captions.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] != "filename":
                    table[row[0].strip()] = row[1].strip()
    images, caps, skipped = [], [], 0
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        cap = table.get(p.name)
        side = p.with_suffix(".txt")
        if cap is None and side.exists():
            cap = side.read_text(encoding="utf-8", errors="replace").strip()
        if not cap:
            skipped += 1
            continue
        try:
            with Image.open(p) as im:
                images.append(_center_crop_resize(im, res))
            caps.append(cap[:300])
        except OSError:
            skipped += 1
        if len(images) >= max_images:
            break
    if not images:
        raise ValueError(f"no captioned images found in {folder}")
    return np.stack(images), caps, skipped


def save_dataset(path: Path, images: np.ndarray, captions: list[str], meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images, captions=np.array(json.dumps(captions)),
                        meta=np.array(json.dumps(meta)))


def load_dataset(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["images"], json.loads(str(z["captions"])), json.loads(str(z["meta"]))


def to_tensor(images: np.ndarray):
    import torch
    x = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    return x


def to_png(x, path: Path):
    """x: (3,H,W) in [-1,1] -> PNG file. Upscales small images for viewing."""
    arr = ((x.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.width < 256:
        img = img.resize((256, 256), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path
