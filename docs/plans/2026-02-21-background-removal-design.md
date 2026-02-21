# Background Removal Design

**Date:** 2026-02-21
**Branch:** feature/Background_Removal
**Status:** Approved

## Problem

The pipeline binarizes images using a hardcoded grayscale threshold (`BINARY_THRESHOLD = 220`). This assumes a bright-white studio background. Any real-world photo (product on a desk, held in a hand, cluttered shelf) will include background pixels as foreground, corrupting the Zernike moment descriptor and breaking similarity search.

## Solution

Replace the threshold-based binarization with `rembg` background removal. `rembg` uses the U2Net neural network via ONNX Runtime on CPU — consistent with the project's offline-first, no-GPU philosophy.

## Approach: Replace binarization with rembg alpha mask

`rembg.remove()` returns RGBA bytes. The alpha channel is already a perfect binary foreground mask (255 = foreground, 0 = background). We use this directly in place of the threshold step.

Rejected alternatives:
- **rembg with threshold fallback** — fragile heuristics, inconsistent descriptors between index and query images
- **Adaptive thresholding** — no new dependencies, but fails on complex real-world backgrounds (does not solve the stated problem)

## Architecture Change

### New data flow

```
Image file (any background)
    │
    ▼
rembg.remove() → RGBA bytes
    │
    ▼
alpha channel → (alpha > 0) → binary mask (foreground=255, bg=0)
    │
    ▼
[UNCHANGED] dilate (3×3 ellipse) → circumscribed-circle crop → pad → resize 256×256
    │
    ▼
[UNCHANGED] Zernike moments (degree=24, radius=128) → 169-dim → L2 normalize
```

### Code changes (minimal)

Both `main.py` and `build_index.py` share the same pipeline logic (currently duplicated). Both get identical edits:

1. **Add** `remove_background(path: str) -> np.ndarray` — loads image bytes, calls `rembg.remove()`, extracts alpha channel, returns binary mask at full resolution.
2. **Update** `binarize_crop_resize` — remove threshold logic, accept pre-computed binary mask directly.
3. **Update** `extract_zernike` — chain `remove_background` → `binarize_crop_resize` instead of `load_fullres_gray` → `binarize_crop_resize`.
4. **Remove** `BINARY_THRESHOLD = 220` constant and `load_fullres_gray` function (no longer needed in the main pipeline).

### Dependencies

Add to `requirements.txt`:
```
rembg>=2.0.50
onnxruntime>=1.16.0
```

U2Net model (~170MB) is downloaded to `~/.u2net/` on first use, cached permanently. No internet needed after initial download.

## Verification

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # installs rembg + onnxruntime
```

Note: First run of any pipeline call triggers U2Net model download (~170MB).

### Tests

1. **Index rebuild** — `python build_index.py` completes without error; descriptor shape is still `(N, 169)`.
2. **Server smoke test** — `python -m uvicorn main:app --host 127.0.0.1 --port 8000` starts without error.
3. **Studio image query** — upload an existing studio (white-background) image; confirm top result has score ≥ 0.80.
4. **Real-world image query** — upload a product photo on a non-white background; confirm correct product is returned at rank 1 with score ≥ 0.80.
5. **Edge case** — upload a blank/all-white image; confirm server returns a result (fallback in `extract_zernike` handles empty mask via existing `if len(ys) == 0` guard).

## Breaking Change

The existing `index/zernike_index.pkl` was built with threshold=220. After this change, **the index must be rebuilt** before querying. Descriptors from the old pipeline are incompatible with the new pipeline.

## README Updates

- Update the pipeline diagram to replace `Binarize (threshold=220)` with `rembg background removal → alpha mask`
- Add `rembg, onnxruntime` to the dependency table
- Add note about U2Net model download on first use
- Update quickstart to show venv setup
