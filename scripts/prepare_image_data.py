"""Prepare an image-caption dataset for the diffusion model.

    python scripts/prepare_image_data.py shapes                      # built-in synthetic shapes (4000)
    python scripts/prepare_image_data.py shapes --n 8000 --res 32
    python scripts/prepare_image_data.py myphotos --folder D:\\captioned_images --res 64

Folder format: image files (png/jpg/webp/bmp) each with a same-name .txt caption,
or a captions.csv with "filename,caption" lines. Only use images you have the
right to use - your own photos, public-domain or openly licensed collections
(e.g. CC0 images from Wikimedia Commons / Openverse, with their captions).
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nexora.core import Nexora  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--folder")
    ap.add_argument("--n", type=int, default=4000, help="number of synthetic images")
    ap.add_argument("--res", type=int, help="resolution (default from config)")
    a = ap.parse_args()
    app = Nexora(start_workers=False)
    if shutil.disk_usage(app.paths.root).free < 500e6:
        sys.exit("ERROR: less than 500 MB free disk space.")
    info = app.images.prepare("folder" if a.folder else "synthetic_shapes", a.name, a.n, a.folder, a.res)
    print("Prepared:", info)
    app.shutdown()


if __name__ == "__main__":
    main()
