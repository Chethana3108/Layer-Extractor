"""Object extraction pipeline (v2 – quality fixes).

  full-res sRGB image
    -> Grounding DINO on a downscaled copy (boxes only; it resizes to ~800px internally anyway)
    -> NMS + size filters
    -> per object: crop with context from the FULL-RES image
         -> SAM on the crop, box prompt, 3 candidate masks, pick best, SOFT logits (not binarised)
         -> optional BiRefNet matting on the crop, constrained to the SAM region (crisp product edges)
         -> alpha cleanup (speckles, pin-holes) + edge colour decontamination
    -> tight crop around the alpha, padded, saved at native resolution (never upscaled)

v1 problems this fixes:
  * SAM ran once on the whole image; its 256x256 mask was binarised and nearest-neighbour upscaled
    -> jagged, blocky, "melted" edges, worst on small objects.
  * single-mask output sometimes returned only part of a product (label/cap) -> holes, chopped shapes.
  * no colour management -> CMYK / Adobe RGB / P3 sources came out with wrong colours.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (AutoModelForImageSegmentation, AutoModelForZeroShotObjectDetection,
                          AutoProcessor, SamModel, SamProcessor)

import boxes
import imaging
from config import Settings

log = logging.getLogger(__name__)

_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Cutout:
    label: str
    score: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in ORIGINAL image pixels (of the saved output)
    data: bytes
    kind: str                         # 'cutout' (RGBA PNG) | 'flat' (JPEG on white) | 'crop' (RGB PNG)
    ext: str
    mime: str


class ObjectExtractor:
    def __init__(self, s: Settings):
        self.s = s
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading models on %s (refiner=%s)", self.device, s.refiner)
        self.det_proc = AutoProcessor.from_pretrained(s.detector_model)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(s.detector_model).to(self.device).eval()
        # fp32 on purpose: fp16 SAM visibly degrades mask edges and saves little on a T4
        self.sam_proc = SamProcessor.from_pretrained(s.sam_model)
        self.sam = SamModel.from_pretrained(s.sam_model).to(self.device).eval()
        self.refiner = None
        if s.refiner == "birefnet":
            self.refiner = AutoModelForImageSegmentation.from_pretrained(
                s.refiner_model, trust_remote_code=True).to(self.device).eval()
        self.prompts = boxes.build_prompts(s.labels or None, s.include_people, s.generic_pass)
        log.info("Detection prompts: %d chunk(s)%s", len(self.prompts), " (custom labels)" if s.labels else "")

    # ------------------------------------------------------------------ detection
    @torch.inference_mode()
    def _detect(self, rgb: np.ndarray) -> list[boxes.Det]:
        """All prompt chunks in ONE batched forward pass; returns raw detections at the lowest threshold."""
        h, w = rgb.shape[:2]
        scale = min(1.0, self.s.detect_max_side / max(h, w))
        img = Image.fromarray(rgb)
        if scale < 1.0:
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        floor = min(self.s.box_threshold, self.s.relaxed_threshold, self.s.generic_threshold)
        # GPU: all prompt chunks in one batched pass. CPU: one at a time to keep RAM bounded.
        step = len(self.prompts) if self.device == "cuda" else 1
        dets: list[boxes.Det] = []
        for i in range(0, len(self.prompts), step):
            chunk = self.prompts[i:i + step]
            n = len(chunk)
            inputs = self.det_proc(images=[img] * n, text=chunk, padding=True,
                                   return_tensors="pt").to(self.device)
            out = self.det(**inputs)
            kw = dict(text_threshold=self.s.text_threshold, target_sizes=[img.size[::-1]] * n)
            try:  # transformers >= 4.51 renamed box_threshold -> threshold
                results = self.det_proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, threshold=floor, **kw)
            except TypeError:
                results = self.det_proc.post_process_grounded_object_detection(
                    out, inputs.input_ids, box_threshold=floor, **kw)
            for res in results:
                labels = res.get("text_labels") or res.get("labels")
                for b, sc, lb in zip(res["boxes"].float().cpu().tolist(),
                                     res["scores"].float().cpu().tolist(), labels):
                    dets.append(boxes.Det([v / scale for v in b], float(sc), str(lb).strip() or boxes.GENERIC))
        return dets

    def _filter(self, dets: list[boxes.Det], w: int, h: int):
        for d in sorted(dets, key=lambda d: -d.score):
            b = d.box
            log.info("  raw: %-16s score=%.2f box=%dx%d", d.label, d.score, b[2] - b[0], b[3] - b[1])
        s = self.s
        chosen, notes = boxes.select(
            dets, w, h, thresholds=[s.box_threshold, s.relaxed_threshold], generic_threshold=s.generic_threshold,
            nms_iou=s.nms_iou, min_area_ratio=s.min_area_ratio, max_area_ratio=s.max_area_ratio,
            min_side_px=s.min_output_side, min_side_ratio=s.min_side_ratio,
            max_objects=s.max_objects, min_objects=s.min_objects_to_split)
        for note in notes:
            log.info("  note: %s", note)
        for d in chosen:
            log.info("  kept: %-16s score=%.2f", d.label, d.score)
        return [(d.box, d.score, d.label) for d in chosen]

    # ------------------------------------------------------------------ segmentation
    @torch.inference_mode()
    def _sam_alpha(self, crop: np.ndarray, box: list[float]) -> np.ndarray:
        """Soft alpha (0..1, crop resolution) from SAM with a box prompt."""
        inputs = self.sam_proc(Image.fromarray(crop), input_boxes=[[box]], return_tensors="pt").to(self.device)
        out = self.sam(**inputs, multimask_output=True)
        logits = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(),
            binarize=False)[0][0]                       # (3, H, W) logits, bilinear-upsampled
        ious = out.iou_scores.cpu()[0, 0]               # (3,)
        probs = torch.sigmoid(logits).numpy()

        # Prefer the highest-IoU candidate that actually spans the prompt box (rejects "just the label")
        bx0, by0, bx1, by1 = box
        bw, bh = bx1 - bx0, by1 - by0
        best, best_score = int(ious.argmax()), -1.0
        for i in range(probs.shape[0]):
            ys, xs = np.nonzero(probs[i] > 0.5)
            if len(xs) == 0:
                continue
            span = min((xs.max() - xs.min()) / max(bw, 1), (ys.max() - ys.min()) / max(bh, 1))
            if span >= self.s.mask_min_span and float(ious[i]) > best_score:
                best, best_score = i, float(ious[i])
        return probs[best].astype(np.float32)

    @torch.inference_mode()
    def _refine_alpha(self, crop: np.ndarray) -> np.ndarray:
        """BiRefNet high-res matting of the crop (1024x1024 model input), returned at crop resolution."""
        h, w = crop.shape[:2]
        x = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
        x = ((x - _IMNET_MEAN) / _IMNET_STD).to(self.device)
        pred = self.refiner(x)[-1].sigmoid().float().cpu()
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0].numpy()

    def _object_alpha(self, crop: np.ndarray, box_in_crop: list[float]) -> np.ndarray:
        a_sam = self._sam_alpha(crop, box_in_crop)
        if self.refiner is None:
            return a_sam
        # BiRefNet gives the clean edge; SAM decides WHICH object (neighbours in the crop are cut away)
        h, w = a_sam.shape
        k = max(3, int(round(min(h, w) * 0.03)) | 1)
        region = cv2.dilate((a_sam > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
        region = cv2.GaussianBlur(region, (k, k), 0)
        a_ref = self._refine_alpha(crop) * region
        sam_area = float((a_sam > 0.5).sum())
        ref_area = float((a_ref > 0.5).sum())
        # Object not salient to BiRefNet (e.g. small item beside a dominant one) -> trust SAM
        if sam_area == 0 or ref_area < 0.6 * sam_area or ref_area > 1.6 * sam_area:
            log.debug("Refiner disagreed with SAM (sam=%d ref=%d); using SAM alpha", sam_area, ref_area)
            return a_sam
        return a_ref

    # ------------------------------------------------------------------ pipeline
    def extract(self, data: bytes) -> list[Cutout]:
        rgb, src_alpha = imaging.load_srgb(data)
        H, W = rgb.shape[:2]
        if min(H, W) < 96:
            log.info("Image too small (%dx%d) to extract anything useful", W, H)
            return []
        if src_alpha is not None:
            # already-transparent source (e.g. a PNG cutout of several products): models must not see the
            # arbitrary colours hidden under alpha=0, so composite onto white first
            rgb = imaging.flatten(rgb, src_alpha.astype(np.float32) / 255.0)
        dets = self._filter(self._detect(rgb), W, H)
        log.info("Detected %d objects after filtering (%dx%d)", len(dets), W, H)
        if len(dets) < self.s.min_objects_to_split:
            return []

        results: list[Cutout] = []
        for box, score, label in dets:
            x0, y0, x1, y1 = box
            cx, cy = (x1 - x0) * self.s.context_ratio, (y1 - y0) * self.s.context_ratio
            c0, r0 = max(0, int(x0 - cx)), max(0, int(y0 - cy))
            c1, r1 = min(W, int(np.ceil(x1 + cx))), min(H, int(np.ceil(y1 + cy)))
            crop = rgb[r0:r1, c0:c1]
            box_in_crop = [x0 - c0, y0 - r0, x1 - c0, y1 - r0]

            if self.s.output_mode == "crop":
                px, py = (x1 - x0) * self.s.padding_ratio, (y1 - y0) * self.s.padding_ratio
                bx = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(W, int(x1 + px)), min(H, int(y1 + py)))
                results.append(Cutout(label, score, bx, imaging.to_png_rgb(rgb[bx[1]:bx[3], bx[0]:bx[2]]),
                                      "crop", "png", "image/png"))
                continue

            alpha = imaging.clean_alpha(self._object_alpha(crop, box_in_crop))
            if src_alpha is not None:
                alpha = np.minimum(alpha, src_alpha[r0:r1, c0:c1].astype(np.float32) / 255.0)
            ys, xs = np.nonzero(alpha > 0.02)
            if len(xs) == 0:
                continue

            # tight box around the actual object + padding, still at native resolution
            tx0, ty0, tx1, ty1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            pad = int(round(max(tx1 - tx0, ty1 - ty0) * self.s.padding_ratio))
            tx0, ty0 = max(0, tx0 - pad), max(0, ty0 - pad)
            tx1, ty1 = min(crop.shape[1], tx1 + pad), min(crop.shape[0], ty1 + pad)
            a = alpha[ty0:ty1, tx0:tx1]
            c = imaging.decontaminate(crop[ty0:ty1, tx0:tx1], a)
            bbox = (c0 + int(tx0), r0 + int(ty0), c0 + int(tx1), r0 + int(ty1))

            if self.s.output_background in ("transparent", "both"):
                results.append(Cutout(label, score, bbox, imaging.to_png_rgba(c, a), "cutout", "png", "image/png"))
            if self.s.output_background in ("white", "both"):
                results.append(Cutout(label, score, bbox, imaging.to_jpeg(imaging.flatten(c, a)),
                                      "flat", "jpg", "image/jpeg"))
        return results
