"""
build_index.py — Preprocess all images into a Zernike moment index.

Scans data/images/ for PNG/JPG files, extracts a 169-dim Zernike moment
descriptor (degree=24) for each, L2-normalizes, and saves to index/zernike_index.pkl.

Pipeline: load at full resolution → binarize (fixed threshold) → dilate (3x3 ellipse)
→ circumscribed-circle crop → pad to square → resize to 256x256 → Zernike moments.
"""

import glob
import os
import pickle
import sys

import cv2
from zernike import zernike_moments
import numpy as np


IMG_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "zernike_index.pkl")

RADIUS = 128
DEGREE = 24
IMG_SIZE = 256
BINARY_THRESHOLD = 220  # User-adjustable after seeing pixel_sampling.png


def load_fullres_gray(path: str) -> np.ndarray:
    """Load an image at full resolution and convert to grayscale."""
    # Use imdecode to handle non-ASCII paths on Windows
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binarize_crop_resize(gray: np.ndarray) -> np.ndarray:
    """Binarize, dilate, circumscribed-circle crop, pad to square, resize.

    1. Binarize: gray < BINARY_THRESHOLD → foreground (255), else 0
    2. Dilate with 3x3 elliptical kernel to normalize line thickness across
       rotation angles (axis-aligned 1px lines → ~3px, matching diagonal staircase lines)
    3. Compute centroid and max distance (circumscribed radius) of foreground
    4. Crop a square of side 2*r_max centered on centroid (rotation-invariant scale)
    5. Pad to exact square (handles edge clipping)
    6. Resize to IMG_SIZE x IMG_SIZE with INTER_NEAREST (preserves binary values)
    """
    binary = np.where(gray < BINARY_THRESHOLD, np.uint8(255), np.uint8(0))

    # Dilate to normalize line thickness across rotation angles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)

    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Centroid of foreground
    cy, cx = ys.mean(), xs.mean()

    # Max distance from centroid to any foreground pixel (circumscribed radius)
    dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = dists.max()

    # Crop a square of side 2*r_max centered on centroid
    half = int(np.ceil(r_max)) + 1
    h, w = binary.shape
    y0 = max(0, int(round(cy)) - half)
    y1 = min(h, int(round(cy)) + half)
    x0 = max(0, int(round(cx)) - half)
    x1 = min(w, int(round(cx)) + half)
    cropped = binary[y0:y1, x0:x1]

    # Pad to exact square (handles edge clipping)
    ch, cw = cropped.shape
    side = max(ch, cw)
    padded = np.zeros((side, side), dtype=np.uint8)
    py, px = (side - ch) // 2, (side - cw) // 2
    padded[py:py + ch, px:px + cw] = cropped

    return cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)


def extract_zernike(path: str) -> np.ndarray:
    """Extract a 169-dim Zernike moment descriptor from an image.

    Pipeline: load_fullres_gray → binarize_crop_resize → Zernike moments
    """
    gray = load_fullres_gray(path)
    binary = binarize_crop_resize(gray)

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
    print(f"Pipeline: binarize (threshold={BINARY_THRESHOLD}) → crop → pad → resize({IMG_SIZE}x{IMG_SIZE}) → Zernike(degree={DEGREE})")

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
