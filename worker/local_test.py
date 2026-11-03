"""Tune locally without Content Hub:  python local_test.py shot.jpg out/

Writes, per object:
  NN_label_cutout.png          – the deliverable (transparent PNG) that would be uploaded
  _debug/NN_label_check.jpg    – same cutout on a checkerboard, to judge the edges
  _debug/NN_label_alpha.png    – the mask itself (white = kept); look here if edges look wrong
  _debug/_preview.jpg          – detection boxes on the source
"""
import io
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import numpy as np
from PIL import Image, ImageDraw

for k in ("CH_BASE_URL", "CH_CLIENT_ID", "CH_CLIENT_SECRET", "SB_CONNECTION"):
    os.environ.setdefault(k, "unused")

import imaging                          # noqa: E402
from config import Settings             # noqa: E402
from extractor import ObjectExtractor   # noqa: E402


def save_bytes(path: str, data: bytes) -> str:
    """Write via temp file + atomic replace. If the target is locked (open in Photos/Explorer preview on
    Windows), write a numbered copy instead of crashing."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    root, ext = os.path.splitext(path)
    for n in range(0, 20):
        target = path if n == 0 else f"{root}({n}){ext}"
        try:
            os.replace(tmp, target)
            if n:
                print(f"WARNING {os.path.basename(path)} is open in another program; saved as {os.path.basename(target)}")
            return target
        except OSError:
            continue
    os.remove(tmp)
    raise OSError(f"Could not write {path} - close any program showing these images and retry")


def save_image(img: Image.Image, path: str, **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if path.lower().endswith(".jpg") else "PNG", **kw)
    return save_bytes(path, buf.getvalue())


src, out = sys.argv[1], sys.argv[2]
dbg = os.path.join(out, "_debug")
os.makedirs(dbg, exist_ok=True)
data = open(src, "rb").read()
settings = Settings()
cutouts = ObjectExtractor(settings).extract(data)

rgb, _ = imaging.load_srgb(data)
preview = Image.fromarray(rgb)
d = ImageDraw.Draw(preview)
for i, c in enumerate(cutouts, 1):
    name = f"{i:02d}_{c.label.replace(' ', '-')}"
    base = os.path.join(dbg, name)
    save_bytes(os.path.join(out, f"{name}_{c.kind}.{c.ext}"), c.data)
    img = Image.open(io.BytesIO(c.data))
    if img.mode == "RGBA":
        arr = np.array(img)
        alpha = arr[..., 3].astype(np.float32) / 255.0
        save_image(imaging.checkerboard_preview(arr[..., :3], alpha), f"{base}_check.jpg", quality=92)
        save_image(img.getchannel("A"), f"{base}_alpha.png")
    print(f"{i:02d} {c.label:<14} score={c.score:.2f} size={img.width}x{img.height} kind={c.kind}")
    d.rectangle(c.bbox, outline="red", width=max(2, preview.width // 400))
    d.text((c.bbox[0] + 6, c.bbox[1] + 6), f"{i} {c.label} {c.score:.2f}", fill="red")
save_image(preview, os.path.join(dbg, "_preview.jpg"), quality=90)
print(f"{len(cutouts)} outputs -> {out}")
if not cutouts:
    print(f"\nNothing extracted. Needs >= {settings.min_objects_to_split} objects (MIN_OBJECTS_TO_SPLIT) that are "
          f">= {settings.min_output_side}px (MIN_OUTPUT_SIDE), score >= {settings.box_threshold} (BOX_THRESHOLD), "
          f"matching DETECT_LABELS. See the INFO lines above for what was detected/dropped.")
