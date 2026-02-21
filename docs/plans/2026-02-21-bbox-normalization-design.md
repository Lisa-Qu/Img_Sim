# Scale & Translation Normalization via Bounding Box — Design

**Date:** 2026-02-21
**Branch:** feature/BBox_Normalization (based on feature/Background_Removal)
**Status:** Approved

## Problem

Zernike moments are sensitive to scale and translation. The current pipeline uses a "circumscribed-circle crop": it computes the centroid of foreground pixels, finds the max distance from the centroid to any foreground pixel (`r_max`), and crops a square of side `2*r_max` centered on the centroid.

For non-circular shapes (bottles, elongated products), `r_max` equals the diagonal half-length, leaving substantial dead space around the short axis. When resized to 256×256, the product occupies an inconsistently-sized region of the frame, degrading Zernike accuracy across different product shapes and distances.

## Solution

Replace the centroid+circumscribed-radius crop with a tight bounding box crop using `cv2.findContours` + `cv2.boundingRect`. Pad the result to a square before resizing to preserve aspect ratio (required for rotation invariance).

## Approach: Bounding box crop, padded to square

Rejected alternatives:
- **Strict tight crop (no padding)** — resizing a non-square bbox directly to 256×256 squashes the aspect ratio differently at each rotation angle, breaking rotation invariance.
- **Shrink circumscribed-circle factor** — cosmetic tweak, doesn't solve the fundamental scale inconsistency.

## Architecture Change

### New pipeline (combined with rembg from parent branch)

```
Image file (any background)
    │
    ▼
rembg.remove() → alpha mask (foreground = alpha > 0)
    │
    ▼
[NEW] cv2.findContours → largest contour → cv2.boundingRect [x, y, w, h]
    │
    ▼
[NEW] Tight crop binary[y:y+h, x:x+w]  (replaces centroid+r_max crop)
    │
    ▼
[UNCHANGED] Pad to square → resize 256×256 with INTER_NEAREST
    │
    ▼
[UNCHANGED] Zernike moments (degree=24, radius=128) → 169-dim → L2 normalize
```

### Code change (minimal)

Only `binarize_crop_resize` changes, identically in `main.py` and `build_index.py`.

**Remove** (centroid + circumscribed-radius block):
```python
cy, cx = ys.mean(), xs.mean()
dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
r_max = dists.max()
half = int(np.ceil(r_max)) + 1
h, w = binary.shape
y0 = max(0, int(round(cy)) - half)
y1 = min(h, int(round(cy)) + half)
x0 = max(0, int(round(cx)) - half)
x1 = min(w, int(round(cx)) + half)
cropped = binary[y0:y1, x0:x1]
```

**Replace with** (bounding box of largest contour):
```python
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
cropped = binary[y:y+h, x:x+w]
```

The pad-to-square and resize logic below is unchanged.

Also remove the now-unused `ys, xs = np.where(binary > 0)` + empty-mask guard (that guard moves up before `findContours`, since contours will be empty if the mask is empty).

## Breaking Change

The existing `index/zernike_index.pkl` must be rebuilt. Descriptors computed with the old circumscribed-circle crop are incompatible with the new bounding-box crop.

## Verification

### Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tests (additions to tests/test_pipeline.py)

1. **Off-center product** — synthetic image with product in one corner; bbox crop should extract just the product, producing a non-trivial descriptor.
2. **Elongated product** — tall rectangle; descriptor shape still (169,).
3. **Descriptor shape** — `extract_zernike` returns shape (169,) — existing test, still passes.
4. **Empty mask guard** — blank image does not raise — existing test, still passes.

## README Updates

- Update pipeline diagram to show bbox crop replacing circumscribed-circle
- Update "How It Works" prose
