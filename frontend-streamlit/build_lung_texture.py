"""
build_lung_texture.py
=====================
Run once (or on image change) to pre-bake UV texture samples from the
real lung photograph into cached numpy arrays.

Usage:
    python frontend-streamlit/build_lung_texture.py
"""

from pathlib import Path
import numpy as np
from PIL import Image

ASSET_DIR = Path(__file__).parent / "assets"
IMG_PATH  = ASSET_DIR / "lung_texture.png"
OUT_PATH  = ASSET_DIR / "lung_uv_cache.npz"

RES = 90   # parametric resolution (matches _make_lobe_surface res)

img = Image.open(IMG_PATH).convert("RGB")
W, H = img.size
arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)  range 0-1


def sample_region(arr, x_frac_start, x_frac_end, y_frac_start, y_frac_end, res):
    """
    Sample a rectangular region of the image onto a (res x res) UV grid.
    Returns a (res, res) float array representing perceived brightness
    in the tissue-appropriate channel (bias toward red channel for lung tissue).
    """
    H, W, _ = arr.shape
    # pixel bounds
    x0 = int(x_frac_start * W)
    x1 = int(x_frac_end   * W)
    y0 = int(y_frac_start * H)
    y1 = int(y_frac_end   * H)

    crop = arr[y0:y1, x0:x1]           # (crop_h, crop_w, 3)

    # Resize via simple numpy interpolation to (res, res)
    from PIL import Image as PILImage
    pil_crop = PILImage.fromarray((crop * 255).astype(np.uint8))
    pil_resized = pil_crop.resize((res, res), PILImage.LANCZOS)
    resized = np.array(pil_resized, dtype=np.float32) / 255.0   # (res, res, 3)

    # Perceptual luminance weighted toward red (blood-rich tissue)
    # standard: 0.299R + 0.587G + 0.114B, but we bias red for lung tissue
    lum = 0.60 * resized[:,:,0] + 0.28 * resized[:,:,1] + 0.12 * resized[:,:,2]
    return lum   # (res, res)


# Image layout (approximate from the reference photo):
#   Left half  (x: 0%→48%): right lung  (anatomical right = image left)
#   Right half (x: 52%→100%): left lung (anatomical left  = image right)
#   Top (y: 0%→15%): trachea (skip)
#   Bottom (y: 85%→100%): diaphragm (skip)

print(f"Image size: {W}x{H}")

# Right lung region (image left half)
right_lum = sample_region(arr,
    x_frac_start=0.02, x_frac_end=0.48,
    y_frac_start=0.08, y_frac_end=0.95,
    res=RES)

# Left lung region (image right half)
left_lum = sample_region(arr,
    x_frac_start=0.52, x_frac_end=0.97,
    y_frac_start=0.08, y_frac_end=0.95,
    res=RES)

# Also extract dominant color statistics per lung for colorscale building
# These tell us the min/max brightness range in each lung
print(f"Right lung luminance range: {right_lum.min():.3f} – {right_lum.max():.3f}")
print(f"Left  lung luminance range: {left_lum.min():.3f}  – {left_lum.max():.3f}")

np.savez_compressed(str(OUT_PATH),
    right_lum=right_lum,
    left_lum=left_lum,
    res=np.array([RES]))

print(f"Saved UV texture cache → {OUT_PATH}")
