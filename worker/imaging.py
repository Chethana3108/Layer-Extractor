"""Model-free image handling: colour-managed loading, alpha cleanup, edge colour decontamination, encoding.

Kept separate from the models so it can be unit-tested without torch / GPU.
"""
from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

log = logging.getLogger(__name__)

try:  # HEIC/HEIF (iPhone photos); optional dependency
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

Image.MAX_IMAGE_PIXELS = 400_000_000  # allow large print/packaging files; default 89 MP rejects them

_SRGB = ImageCms.createProfile("sRGB")
_SRGB_BYTES = ImageCms.ImageCmsProfile(_SRGB).tobytes()


# ---------------------------------------------------------------- loading
def load_srgb(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode any common DAM image into 8-bit sRGB.

    Returns (rgb uint8 HxWx3, alpha uint8 HxW or None).
    Handles EXIF rotation, CMYK (print/packaging files), embedded ICC profiles (Adobe RGB, ProPhoto,
    Display P3), 16-bit greyscale and palette images. A plain .convert('RGB') on these gives the
    washed-out / oversaturated / inverted colours that look like "distortion".
    """
    try:
        img = Image.open(io.BytesIO(data))
        if getattr(img, "n_frames", 1) > 1:   # animated GIF/WebP/multi-page TIFF: first frame
            img.seek(0)
        img.load()
    except Exception as e:                     # e.g. 48-bit RGB TIFF that Pillow can't decode
        log.info("Pillow could not decode (%s); trying OpenCV", e)
        return _load_cv2(data)
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")

    alpha = None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA") if img.mode != "LA" else img
        alpha = np.array(img.getchannel("A"))
        img = img.convert("RGB") if img.mode == "RGBA" else img.convert("L")

    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.array(img, dtype=np.float32)
        hi = 65535.0 if arr.max() > 255 else 255.0
        img = Image.fromarray(np.clip(arr / hi * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")

    if img.mode not in ("RGB", "CMYK", "L"):
        img = img.convert("RGB")

    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            img = ImageCms.profileToProfile(img, src, _SRGB, outputMode="RGB",
                                            renderingIntent=ImageCms.Intent.PERCEPTUAL)
        except Exception as e:  # broken / mismatched profile: fall back, but say so
            log.warning("ICC conversion failed (%s); using naive conversion", e)
            img = img.convert("RGB")
    else:
        if img.mode == "CMYK":
            log.warning("CMYK image without ICC profile; colours are approximate")
        img = img.convert("RGB")

    return np.array(img, dtype=np.uint8), alpha


def _load_cv2(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Unsupported or corrupt image file")
    if arr.dtype == np.uint16:
        arr = (arr / 257.0 + 0.5).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB), None
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB), arr[..., 3].copy()
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), None


# ---------------------------------------------------------------- alpha
def clean_alpha(alpha: np.ndarray, keep_ratio: float = 0.05, max_hole_ratio: float = 0.01) -> np.ndarray:
    """Remove speckles and pin-holes from a soft alpha (float 0..1) while keeping its anti-aliased edge.

    - keeps connected components >= keep_ratio of the largest (so a detached cap/lid survives,
      stray blobs from neighbouring objects don't)
    - fills interior holes smaller than max_hole_ratio of the object (real openings, e.g. a handle
      loop, are larger and stay transparent)
    """
    hard = (alpha > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if n <= 1:
        return np.zeros_like(alpha)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    keep = np.isin(lab, 1 + np.nonzero(areas >= keep_ratio * big)[0]).astype(np.uint8)

    inv = (1 - keep).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = keep.shape
    obj_area = keep.sum()
    holes = np.zeros_like(keep, dtype=bool)
    for i in range(1, n2):
        x, y, bw, bh, a = st2[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and a <= max_hole_ratio * obj_area:
            holes |= lab2 == i

    # Soft edge survives: multiply by a slightly dilated version of the cleaned hard mask
    region = cv2.dilate(keep | holes.astype(np.uint8), np.ones((5, 5), np.uint8))
    out = alpha.astype(np.float32) * region
    out[holes] = 1.0
    return np.clip(out, 0.0, 1.0)


def _nearest_colour(rgb: np.ndarray, source: np.ndarray) -> np.ndarray | None:
    """For every pixel, the colour of the nearest pixel where `source` is True (distance-transform labels)."""
    if not source.any():
        return None
    _, labels = cv2.distanceTransformWithLabels((~source).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(source)
    lut = np.zeros((labels.max() + 1, 3), np.float32)
    lut[labels[ys, xs]] = rgb[ys, xs]
    near = lut[labels]
    return cv2.GaussianBlur(near, (5, 5), 0)  # de-noise the Voronoi pattern


def decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove the old background colour bleeding into semi-transparent edge pixels.

    Without this a product shot on a grey sweep keeps a grey halo when composited into a new scene.
    Solves C = a*F + (1-a)*B for F, with B estimated from the nearest pure-background pixel; where alpha
    is too low for the division to be stable, falls back to the nearest solid-foreground colour.
    """
    a = alpha.astype(np.float32)
    edge = (a > 0.02) & (a < 0.98)
    if not edge.any():
        return rgb
    C = rgb.astype(np.float32)
    F_near = _nearest_colour(C, a >= 0.98)
    if F_near is None:
        return rgb
    B_near = _nearest_colour(C, a <= 0.02)
    a3 = a[..., None]
    if B_near is not None:
        F_unmix = np.clip((C - (1 - a3) * B_near) / np.maximum(a3, 1e-3), 0, 255)
        w = np.clip((a3 - 0.15) / 0.35, 0, 1)        # alpha<0.15: nearest fg; >0.5: unmixed; ramp between
        F = w * F_unmix + (1 - w) * F_near
    else:  # object fills the crop – no background sample
        F = F_near
    out = C.copy()
    out[edge] = F[edge]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- encoding
def to_png_rgba(rgb: np.ndarray, alpha: np.ndarray) -> bytes:
    a8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgb = rgb.copy()
    # Fully transparent pixels: store white, not the old background. Viewers that ignore alpha
    # (Windows Photos/Paint, Explorer thumbnails, some DAM previews) then show the product on white
    # instead of looking "uncut". Invisible in alpha-aware tools, and compresses smaller.
    rgb[a8 == 0] = 255
    img = Image.fromarray(np.dstack([rgb, a8]), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", icc_profile=_SRGB_BYTES, compress_level=6)  # lossless, sRGB-tagged
    return buf.getvalue()


def to_png_rgb(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG", icc_profile=_SRGB_BYTES, compress_level=6)
    return buf.getvalue()


def flatten(rgb: np.ndarray, alpha: np.ndarray, bg=(255, 255, 255)) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    return np.clip(rgb * a + np.array(bg, np.float32) * (1 - a) + 0.5, 0, 255).astype(np.uint8)


def to_jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "JPEG", quality=quality, subsampling=0,  # 4:4:4, no chroma smear
                                      icc_profile=_SRGB_BYTES, optimize=True)
    return buf.getvalue()


def checkerboard_preview(rgb: np.ndarray, alpha: np.ndarray, cell: int = 16) -> Image.Image:
    """For eyeballing cutouts: transparent areas shown as a checkerboard, not black."""
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // cell + xx // cell) % 2)[..., None] == 0, 230, 190).astype(np.float32)
    board = np.repeat(board, 3, axis=2)
    a = alpha[..., None]
    return Image.fromarray(np.clip(rgb * a + board * (1 - a), 0, 255).astype(np.uint8))
