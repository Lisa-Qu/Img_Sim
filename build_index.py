"""
build_index.py — Preprocess all images into a Zernike moment index.

Scans data/images/ for PNG/JPG files, extracts a 169-dim Zernike moment
descriptor (degree=24) for each, L2-normalizes, and saves to index/zernike_index.pkl.

Pipeline: rembg background removal → alpha mask → dilate (3x3 ellipse)
→ bounding-box crop → pad to square → resize to 256x256 → Zernike moments.
"""

import glob
import os
import pickle
import sys
import io

import cv2
from PIL import Image
from rembg import remove as rembg_remove
from zernike import zernike_moments
import numpy as np


IMG_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "zernike_index.pkl")

RADIUS = 128
DEGREE = 24
IMG_SIZE = 256


def remove_background(path: str) -> np.ndarray:
    """Remove background using rembg; return binary foreground mask at full resolution.

    Uses the U2Net model via ONNX Runtime (CPU). Alpha > 0 = foreground.
    """
    with open(path, "rb") as f:
        input_bytes = f.read()
    output_bytes = rembg_remove(input_bytes)
    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    return np.where(alpha > 0, np.uint8(255), np.uint8(0))


def binarize_crop_resize(binary: np.ndarray) -> np.ndarray:
    """Dilate, bounding-box crop, pad to square, resize.

    1. Dilate with 3x3 elliptical kernel to normalize line thickness across
       rotation angles (axis-aligned 1px lines → ~3px, matching diagonal staircase lines)
    2. Find largest contour and compute its bounding box (tight scale/translation normalization)
    3. Crop to bounding box
    4. Pad to exact square (preserves aspect ratio → rotation invariance maintained)
    5. Resize to IMG_SIZE x IMG_SIZE with INTER_NEAREST (preserves binary values)
    """
    # Dilate to normalize line thickness across rotation angles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Tight bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = binary[y:y + h, x:x + w]

    # Pad to exact square (preserves aspect ratio -> rotation invariance maintained)
    ch, cw = cropped.shape
    side = max(ch, cw)
    padded = np.zeros((side, side), dtype=np.uint8)
    py, px = (side - ch) // 2, (side - cw) // 2
    padded[py:py + ch, px:px + cw] = cropped

    return cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)


def extract_zernike(path: str) -> np.ndarray:
    """Extract a 169-dim Zernike moment descriptor from an image.

    Pipeline: remove_background → binarize_crop_resize → Zernike moments
    """
    binary = remove_background(path)
    binary = binarize_crop_resize(binary)

    # Compute centroid of foreground for Zernike center
    ys, xs = np.where(binary > 0)
    if len(ys) > 0:
        cy, cx = ys.mean(), xs.mean()
    else:
        cy, cx = IMG_SIZE / 2, IMG_SIZE / 2

    moments = zernike_moments(binary, radius=RADIUS, degree=DEGREE, cm=(cy, cx))
    return moments


def parse_filename(filename: str) -> dict:
    """Parse metadata from filename like '0000_BACK_0.png'."""
    name = os.path.splitext(filename)[0]
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        return {"label": parts[0], "view": parts[1], "rotation": parts[2]}
    return {"label": name, "view": "unknown", "rotation": "0"}


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Glob all image files
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(IMG_DIR, pat), recursive=True))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {IMG_DIR}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {IMG_DIR}")
    print(f"Pipeline: rembg background removal → crop → pad → resize({IMG_SIZE}x{IMG_SIZE}) → Zernike(degree={DEGREE})")

    paths = []
    descriptors = []
    metadata = []
    errors = 0

    for i, img_path in enumerate(image_paths):
        rel_path = os.path.relpath(img_path, os.path.join(os.path.dirname(__file__), "data", "images"))
        filename = os.path.basename(img_path)
        try:
            desc = extract_zernike(img_path)
            paths.append(rel_path)
            descriptors.append(desc)
            metadata.append(parse_filename(filename))
            if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                print(f"  [{i+1}/{len(image_paths)}] {filename}")
        except Exception as e:
            print(f"  ERROR: {filename}: {e}")
            errors += 1

    descriptors = np.array(descriptors, dtype=np.float64)

    # L2-normalize each descriptor
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descriptors = descriptors / norms

    index_data = {
        "paths": paths,
        "descriptors": descriptors,
        "metadata": metadata,
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print(f"\nIndexed {len(paths)} images → {INDEX_PATH}")
    if errors:
        print(f"  ({errors} errors skipped)")
    print(f"  Descriptor shape: {descriptors.shape}")


if __name__ == "__main__":
    main()
