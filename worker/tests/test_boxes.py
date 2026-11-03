import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
import numpy as np
from PIL import Image
import boxes, imaging
from boxes import Det

KW = dict(thresholds=[0.35, 0.25], generic_threshold=0.40, nms_iou=0.5, min_area_ratio=0.005,
          max_area_ratio=0.9, min_side_px=48, min_side_ratio=0.03, max_objects=10, min_objects=2)

# Umrela-like 640x535: carton + bottle, plus duplicates, a part, a group box and a whole-image box
dets = [
    Det([125, 20, 330, 470], 0.62, "box"),
    Det([128, 22, 328, 468], 0.48, "carton"),        # duplicate label on same carton
    Det([380, 100, 527, 518], 0.58, "bottle"),
    Det([381, 99, 526, 519], 0.45, "object"),          # generic duplicate
    Det([382, 340, 525, 470], 0.41, "sign"),           # label = part of bottle
    Det([120, 15, 530, 520], 0.38, "object"),          # group box around both
    Det([0, 0, 640, 535], 0.30, "box"),                # whole image
    Det([10, 10, 30, 25], 0.50, "can"),                # speck
]
sel, notes = boxes.select(dets, 640, 535, **KW)
labels = sorted(d.label for d in sel)
assert labels == ["bottle", "box"], (labels, notes); print("umrela-like ok:", labels, notes)

# generic box inherits a named label
sel = boxes.nms([Det([0, 0, 100, 100], 0.6, "object"), Det([2, 2, 100, 100], 0.4, "jar")], 0.5)
assert len(sel) == 1 and sel[0].label == "jar"; print("label inheritance ok")

# weak image: only relaxed threshold finds 2 objects
weak = [Det([50, 50, 300, 400], 0.30, "bottle"), Det([400, 60, 700, 420], 0.28, "box")]
sel, notes = boxes.select(weak, 800, 600, **KW)
assert len(sel) == 2 and any("relaxed" in n for n in notes); print("relaxed ok:", notes)

# size limits scale: in a 6000px image a 120px object is dropped, in a 640px image it's kept
small = [Det([100, 100, 220, 300], 0.6, "tube"), Det([400, 100, 520, 300], 0.6, "tube")]
assert len(boxes.select(small, 640, 640, **KW)[0]) == 2
assert len(boxes.select(small, 6000, 6000, **KW)[0]) == 0; print("relative size ok")

# a real product sitting partly in front of another is NOT removed as a 'part'
overlap = [Det([100, 100, 500, 600], 0.6, "carton"), Det([380, 300, 640, 640], 0.55, "bottle")]
assert len(boxes.select(overlap, 800, 700, **KW)[0]) == 2; print("overlap kept ok")

# prompts: 4 vocabulary chunks + generic, no people by default
p = boxes.build_prompts(None, False, True)
assert len(p) == 5 and p[-1] == "object." and "person" not in " ".join(p)
assert all(len(x.split()) < 60 for x in p)            # stay far below the 256-token text limit
assert boxes.build_prompts(["strip", "Carton"], False, False) == ["strip. carton."]
assert "person." in boxes.build_prompts(None, True, False)[0]; print("prompts ok")

# loaders
def enc(img, fmt, **kw):
    b = io.BytesIO(); img.save(b, fmt, **kw); return b.getvalue()
frames = [Image.new("RGB", (64, 64), c) for c in [(255, 0, 0), (0, 255, 0)]]
rgb, _ = imaging.load_srgb(enc(frames[0], "GIF", save_all=True, append_images=frames[1:]))
assert rgb[0, 0, 0] > 200 and rgb[0, 0, 1] < 50; print("animated gif ok")
rgb, _ = imaging.load_srgb(enc(Image.new("RGB", (50, 40), (10, 200, 30)), "WEBP", lossless=True))
assert tuple(rgb[5, 5]) == (10, 200, 30); print("webp ok")
t16 = np.zeros((20, 30, 3), np.uint16); t16[..., 2] = 65535          # BGR: red channel full
ok, buf = cv2.imencode(".tiff", t16); rgb, a = imaging.load_srgb(buf.tobytes())
assert rgb.shape == (20, 30, 3) and rgb[0, 0, 0] == 255 and rgb[0, 0, 2] == 0; print("16-bit RGB tiff ok")
