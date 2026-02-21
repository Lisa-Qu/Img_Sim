"""
main.py — FastAPI server for Zernike-based image similarity search.

Loads a pre-built pickle index on startup and serves a /search endpoint
that accepts an uploaded image and returns all images with >= 80% similarity.

Pipeline: rembg background removal → alpha mask → dilate (3x3 ellipse)
→ circumscribed-circle crop → pad to square → resize to 256x256 → Zernike moments.
"""

import math
import os
import pickle
import io
import tempfile
from typing import List

import cv2
from PIL import Image
from rembg import remove as rembg_remove
from zernike import zernike_moments
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "index", "zernike_index.pkl")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
PRODUCTS_DIR = os.path.join(IMAGES_DIR, "products")

RADIUS = 128
DEGREE = 24
IMG_SIZE = 256

app = FastAPI(title="Zernike Image Similarity")

# CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global index data
index_data = {}


@app.on_event("startup")
def load_index():
    global index_data
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(
            f"Index not found at {INDEX_PATH}. Run build_index.py first."
        )
    with open(INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    n = len(index_data["paths"])
    print(f"Loaded index: {n} images, descriptor shape {index_data['descriptors'].shape}")


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
    """Dilate, circumscribed-circle crop, pad to square, resize.

    1. Dilate with 3x3 elliptical kernel to normalize line thickness across
       rotation angles (axis-aligned 1px lines → ~3px, matching diagonal staircase lines)
    2. Compute centroid and max distance (circumscribed radius) of foreground
    3. Crop a square of side 2*r_max centered on centroid (rotation-invariant scale)
    4. Pad to exact square (handles edge clipping)
    5. Resize to IMG_SIZE x IMG_SIZE with INTER_NEAREST (preserves binary values)
    """
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

    Pipeline: remove_background → binarize_crop_resize → Zernike moments
    """
    binary = remove_background(path)
    binary = binarize_crop_resize(binary)

    ys, xs = np.where(binary > 0)
    if len(ys) > 0:
        cy, cx = ys.mean(), xs.mean()
    else:
        cy, cx = IMG_SIZE / 2, IMG_SIZE / 2

    moments = zernike_moments(binary, radius=RADIUS, degree=DEGREE, cm=(cy, cx))
    return moments


@app.post("/search")
async def search(file: UploadFile = File(...)):
    # Save uploaded file to temp location (cv2 needs a file path)
    suffix = os.path.splitext(file.filename or "upload.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Extract query descriptor
        query_desc = extract_zernike(tmp_path)

        # L2-normalize
        norm = np.linalg.norm(query_desc)
        if norm > 0:
            query_desc = query_desc / norm

        # Cosine similarity (index descriptors are already L2-normalized)
        similarities = index_data["descriptors"] @ query_desc

        # All results with similarity >= 80%
        THRESHOLD = 0.80
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = [i for i in sorted_indices if similarities[i] >= THRESHOLD]
        # Always return at least 1 result (the best match)
        if not top_indices:
            top_indices = [sorted_indices[0]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            rel_path = index_data["paths"][idx]
            meta = index_data["metadata"][idx]
            filename = os.path.basename(rel_path)
            results.append({
                "rank": rank,
                "filename": filename,
                "label": meta.get("label", ""),
                "score": round(float(similarities[idx]), 4),
                "image_url": f"/images/{rel_path}",
            })

        return JSONResponse({
            "query_filename": file.filename or "upload.png",
            "results": results,
        })
    finally:
        os.unlink(tmp_path)


def parse_filename(filename: str) -> dict:
    """Parse metadata from filename like '0000_BACK_0.png'."""
    name = os.path.splitext(filename)[0]
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        return {"label": parts[0], "view": parts[1], "rotation": parts[2]}
    return {"label": name, "view": "unknown", "rotation": "0"}


def save_index():
    """Persist the current in-memory index to pickle."""
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)


# --------------- Database management endpoints ---------------

@app.get("/database/stats")
def database_stats():
    labels = {m.get("label", "") for m in index_data.get("metadata", [])}
    return {
        "total_images": len(index_data.get("paths", [])),
        "total_products": len(labels),
    }


@app.get("/database/images")
def database_images(page: int = 1, page_size: int = 20):
    paths = index_data.get("paths", [])
    metadata = index_data.get("metadata", [])
    total = len(paths)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size

    images = []
    for i in range(start, min(end, total)):
        rel_path = paths[i]
        meta = metadata[i]
        filename = os.path.basename(rel_path)
        images.append({
            "filename": filename,
            "label": meta.get("label", ""),
            "view": meta.get("view", ""),
            "rotation": meta.get("rotation", ""),
            "image_url": f"/images/{rel_path}",
        })

    return {
        "images": images,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.post("/database/upload")
async def database_upload(files: List[UploadFile] = File(...)):
    uploaded = 0
    errors = []

    for file in files:
        filename = file.filename or "upload.png"
        dest = os.path.join(PRODUCTS_DIR, filename)
        try:
            contents = await file.read()
            with open(dest, "wb") as f:
                f.write(contents)

            desc = extract_zernike(dest)
            norm = np.linalg.norm(desc)
            if norm > 0:
                desc = desc / norm

            rel_path = os.path.relpath(dest, IMAGES_DIR)
            meta = parse_filename(filename)

            index_data["paths"].append(rel_path)
            index_data["metadata"].append(meta)
            index_data["descriptors"] = np.vstack(
                [index_data["descriptors"], desc.reshape(1, -1)]
            )
            uploaded += 1
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
            if os.path.exists(dest):
                os.unlink(dest)

    if uploaded > 0:
        save_index()

    return {"uploaded": uploaded, "errors": errors}


@app.delete("/database/images/{filename}")
def database_delete(filename: str):
    paths = index_data.get("paths", [])
    # Find the image in the index
    target_idx = None
    for i, p in enumerate(paths):
        if os.path.basename(p) == filename:
            target_idx = i
            break

    if target_idx is None:
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in index")

    # Remove file from disk
    full_path = os.path.join(IMAGES_DIR, paths[target_idx])
    if os.path.exists(full_path):
        os.unlink(full_path)

    # Remove from index
    index_data["paths"].pop(target_idx)
    index_data["metadata"].pop(target_idx)
    index_data["descriptors"] = np.delete(index_data["descriptors"], target_idx, axis=0)
    save_index()

    return {"deleted": filename}


@app.get("/health")
def health():
    return {"status": "healthy"}


# Serve images as static files
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# --- Serve pre-built frontend as SPA (must be AFTER all API routes) ---
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend_dist")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="frontend-assets")
    _fonts_dir = os.path.join(FRONTEND_DIR, "fonts")
    if os.path.isdir(_fonts_dir):
        app.mount("/fonts", StaticFiles(directory=_fonts_dir), name="fonts")

    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
