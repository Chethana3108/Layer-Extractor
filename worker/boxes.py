"""Detection vocabulary + model-free box selection (numpy only, unit-testable without torch).

Works on any kind of image without per-client tuning:
  * broad built-in vocabulary, sent to Grounding DINO in small prompt chunks (long prompts hurt accuracy)
  * a class-agnostic "object" pass catches things no label names
  * size limits scale with the image instead of fixed pixels
  * group boxes ("box around all products") and part boxes ("label on the bottle") are removed
  * threshold relaxes automatically when strict settings find too little
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GENERIC = "object"

# No person / face / body parts by default: cutting real people out of source shots creates likeness and
# consent problems downstream. INCLUDE_PEOPLE=true adds them back.
VOCABULARY: list[list[str]] = [
    # packaging / pharma / FMCG
    ["bottle", "box", "carton", "tube", "jar", "can", "pouch", "sachet", "blister pack", "tablet strip",
     "packet", "bag", "container", "tin", "spray bottle", "dropper bottle", "capsule", "syringe"],
    # personal care / food & drink
    ["cosmetic product", "lipstick", "perfume", "soap", "toothbrush", "toothpaste", "cup", "mug",
     "glass", "plate", "bowl", "fruit", "vegetable", "food", "chocolate", "biscuit", "drink"],
    # electronics / appliances / accessories
    ["phone", "laptop", "tablet computer", "television", "camera", "headphones", "speaker", "watch",
     "remote control", "appliance", "refrigerator", "washing machine", "shoe", "handbag", "clothing",
     "cap", "sunglasses", "jewelry"],
    # home / vehicles / misc
    ["chair", "sofa", "table", "lamp", "plant", "vase", "pillow", "umbrella", "car", "bicycle",
     "motorcycle", "toy", "book", "dog", "cat", "animal", "sign"],
]
PEOPLE = ["person"]


def build_prompts(custom: list[str] | None, include_people: bool, generic_pass: bool) -> list[str]:
    """Grounding DINO prompt strings: lowercase phrases each ending in '.'"""
    chunks = [custom] if custom else [list(c) for c in VOCABULARY]
    if include_people:
        chunks[0] = chunks[0] + PEOPLE
    prompts = [" ".join(f"{l.lower().strip()}." for l in c if l.strip()) for c in chunks]
    if generic_pass:
        prompts.append(f"{GENERIC}.")
    return prompts


@dataclass
class Det:
    box: list[float]   # x0, y0, x1, y1
    score: float
    label: str


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection areas, (N,4) x (M,4) -> (N,M)."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def nms(dets: list[Det], iou_thr: float) -> list[Det]:
    """Class-agnostic NMS. A named label beats the generic 'object' label on the same thing."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    if not dets:
        return []
    b = np.array([d.box for d in dets], np.float64)
    inter = _inter(b, b)
    a = _area(b)
    iou = inter / np.maximum(a[:, None] + a[None, :] - inter, 1e-9)
    alive = np.ones(len(dets), bool)
    kept: list[Det] = []
    for i in range(len(dets)):
        if not alive[i]:
            continue
        keeper = Det(list(dets[i].box), dets[i].score, dets[i].label)
        dup = alive & (iou[i] > iou_thr)
        dup[i] = False
        if keeper.label == GENERIC:
            named = [dets[j] for j in np.nonzero(dup)[0] if dets[j].label != GENERIC]
            if named:
                keeper.label = named[0].label
        alive &= ~dup
        alive[i] = False
        kept.append(keeper)
    return kept


def drop_groups_and_parts(dets: list[Det], contain: float = 0.85, part_ratio: float = 0.5) -> list[Det]:
    """Remove (1) boxes that wrap 2+ other objects (a 'group' / 'box of everything'),
    then (2) small boxes lying inside a bigger kept box (a label, cap, logo = part of that object)."""
    if len(dets) < 2:
        return dets
    b = np.array([d.box for d in dets], np.float64)
    a = _area(b)
    frac_inside = _inter(b, b) / np.maximum(a[:, None], 1e-9)   # [i, j] = share of i that lies inside j
    np.fill_diagonal(frac_inside, 0)
    inside = frac_inside >= contain
    is_group = inside.sum(axis=0) >= 2                          # column j contains >= 2 others
    keep = ~is_group
    for i in range(len(dets)):
        if not keep[i]:
            continue
        hosts = np.nonzero(inside[i] & keep)[0]
        if any(a[i] < part_ratio * a[j] for j in hosts):
            keep[i] = False
    return [d for d, k in zip(dets, keep) if k]


def select(dets: list[Det], w: int, h: int, *, thresholds: list[float], generic_threshold: float,
           nms_iou: float, min_area_ratio: float, max_area_ratio: float, min_side_px: int,
           min_side_ratio: float, max_objects: int, min_objects: int) -> tuple[list[Det], list[str]]:
    """Returns (selected detections, human-readable notes about what was dropped and why)."""
    notes: list[str] = []
    min_side = max(min_side_px, int(min_side_ratio * min(w, h)))
    img_area = float(w * h)

    def size_ok(d: Det) -> bool:
        bw, bh = d.box[2] - d.box[0], d.box[3] - d.box[1]
        ar = bw * bh / img_area
        return min_area_ratio <= ar <= max_area_ratio and min(bw, bh) >= min_side

    sized = [d for d in dets if size_ok(d)]
    if len(sized) < len(dets):
        notes.append(f"{len(dets) - len(sized)} boxes outside size limits (min side {min_side}px, "
                     f"area {min_area_ratio:.1%}-{max_area_ratio:.0%} of image)")

    chosen: list[Det] = []
    for thr in thresholds:
        cand = [d for d in sized if d.score >= (generic_threshold if d.label == GENERIC else thr)]
        chosen = drop_groups_and_parts(nms(cand, nms_iou))
        if len(chosen) >= min_objects:
            if thr != thresholds[0]:
                notes.append(f"relaxed score threshold to {thr:.2f} to find {len(chosen)} objects")
            break
    return chosen[:max_objects], notes
