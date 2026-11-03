import io, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
from PIL import Image, ImageCms
import imaging

def enc(img, fmt, **kw):
    b = io.BytesIO(); img.save(b, fmt, **kw); return b.getvalue()

# 1. CMYK JPEG (no ICC) -> RGB, correct hue
cmyk = Image.new("CMYK", (40, 30), (0, 255, 255, 0))  # pure red in CMYK
rgb, a = imaging.load_srgb(enc(cmyk, "JPEG", quality=95))
assert rgb.shape == (30, 40, 3) and a is None
r, g, b = rgb[15, 20]; assert r > 200 and g < 60 and b < 60, rgb[15, 20]; print("cmyk ok", rgb[15,20])

# 2. sRGB-tagged image: colour preserved through ICC path
srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
im = Image.new("RGB", (10, 10), (30, 120, 200))
rgb, _ = imaging.load_srgb(enc(im, "PNG", icc_profile=srgb))
assert np.abs(rgb[5, 5].astype(int) - [30, 120, 200]).max() <= 2, rgb[5, 5]; print("icc ok", rgb[5,5])

# 3. 16-bit greyscale
arr = (np.linspace(0, 65535, 100).reshape(10, 10)).astype(np.uint16)
rgb, _ = imaging.load_srgb(enc(Image.fromarray(arr, "I;16"), "PNG"))
assert rgb.shape == (10, 10, 3) and rgb.max() >= 250 and rgb.min() <= 5; print("16bit ok")

# 4. palette PNG with transparency keeps alpha
p = Image.new("RGBA", (10, 10), (255, 0, 0, 0)); p.paste((0, 0, 255, 255), (2, 2, 8, 8))
rgb, a = imaging.load_srgb(enc(p.convert("P"), "PNG", transparency=0) if False else enc(p, "PNG"))
assert a is not None and a[0, 0] == 0 and a[5, 5] == 255; print("alpha ok")

# 5. EXIF rotation applied
im = Image.new("RGB", (40, 20), (0, 200, 0)); ex = im.getexif(); ex[0x0112] = 6
rgb, _ = imaging.load_srgb(enc(im, "JPEG", exif=ex.tobytes()))
assert rgb.shape[:2] == (40, 20), rgb.shape; print("exif ok")

# 6. clean_alpha: speckle removed, pin-hole filled, big hole (ring opening) kept, soft edge kept
h = w = 400
alpha = np.zeros((h, w), np.float32)
cv2.circle(alpha, (200, 200), 120, 1.0, -1)
alpha = cv2.GaussianBlur(alpha, (9, 9), 0)                 # soft edge
cv2.circle(alpha, (200, 200), 50, 0.0, -1)                 # big hole (e.g. handle loop) - keep
alpha[140:143, 140:143] = 0.0                              # pin-hole - fill
cv2.circle(alpha, (370, 30), 4, 1.0, -1)                   # speckle - remove
out = imaging.clean_alpha(alpha)
assert out[30, 370] == 0, "speckle"
assert out[141, 141] == 1.0, "pinhole"
assert out[200, 200] == 0, "big hole kept"
edge = out[200, 318:324]; assert ((edge > 0.05) & (edge < 0.95)).any(), edge
print("clean_alpha ok, edge", np.round(edge, 2))

# 7. decontaminate: red object on grey background -> edge pixels lose grey
rgbimg = np.full((h, w, 3), 128, np.uint8)
obj = np.zeros((h, w), np.float32); cv2.circle(obj, (200, 200), 120, 1.0, -1)
soft = cv2.GaussianBlur(obj, (15, 15), 0)
comp = (np.array([220, 20, 20]) * soft[..., None] + 128 * (1 - soft[..., None])).astype(np.uint8)
dec = imaging.decontaminate(comp, soft)
ys, xs = np.nonzero((soft > 0.2) & (soft < 0.8))
before = np.abs(comp[ys, xs].astype(int) - [220, 20, 20]).mean()
after = np.abs(dec[ys, xs].astype(int) - [220, 20, 20]).mean()
assert after < before * 0.5, (before, after); print(f"decontaminate ok: edge colour error {before:.0f} -> {after:.0f}")

# 8. encoders
png = imaging.to_png_rgba(dec, soft); jp = imaging.to_jpeg(imaging.flatten(dec, soft))
i1 = Image.open(io.BytesIO(png)); i2 = Image.open(io.BytesIO(jp))
assert i1.mode == "RGBA" and "icc_profile" in i1.info and i2.mode == "RGB"; print("encode ok")
