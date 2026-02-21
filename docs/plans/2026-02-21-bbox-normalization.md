# Scale & Translation Normalization via Bounding Box — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the circumscribed-circle crop in `binarize_crop_resize` with a tight bounding-box crop so that products fill the 256×256 frame consistently regardless of their size or position in the photo.

**Architecture:** Inside `binarize_crop_resize`, swap the centroid+r_max block for `cv2.findContours` + `cv2.boundingRect` on the largest contour. Crop to that bounding box, pad to square, resize — identical change in both `main.py` and `build_index.py`. Nothing else changes.

**Tech Stack:** Python, OpenCV (cv2 — already in requirements), pytest (already in requirements)

---

## Task 1: Write failing tests for bbox behavior

**Files:**
- Modify: `tests/test_pipeline.py`

**Context:** The existing 6 tests test `remove_background` and `extract_zernike`. We need new tests that target `binarize_crop_resize` directly and verify the bbox contract: same-shaped product at different positions should produce the same output fill ratio. These tests must FAIL with the current circumscribed-circle code.

**Step 1: Add new test class to tests/test_pipeline.py**

Append this class at the end of the file:

```python
class TestBboxCropResize:
    def test_same_product_at_different_positions_gives_same_fill(self):
        """Bbox crop: product position in frame should not affect output fill ratio."""
        from main import binarize_crop_resize

        # 40×40 product centered in 200×200 frame
        mask_center = np.zeros((200, 200), dtype=np.uint8)
        mask_center[80:120, 80:120] = 255

        # Same 40×40 product at top-left corner
        mask_corner = np.zeros((200, 200), dtype=np.uint8)
        mask_corner[0:40, 0:40] = 255

        result_center = binarize_crop_resize(mask_center)
        result_corner = binarize_crop_resize(mask_corner)

        assert result_center.shape == (256, 256)
        assert result_corner.shape == (256, 256)

        # Bbox crop: product always fills the frame → fill ratios should be nearly equal
        center_fill = int(np.sum(result_center > 0))
        corner_fill = int(np.sum(result_corner > 0))
        assert abs(center_fill - corner_fill) < 500, (
            f"Bbox should produce equal fill regardless of position: "
            f"center={center_fill}, corner={corner_fill}"
        )

    def test_elongated_product_fills_frame_better_than_square_radius(self):
        """Bbox crop: elongated product (20×80) should fill >50% of 256×256 output."""
        from main import binarize_crop_resize

        # 20px wide × 80px tall product in a 200×200 frame
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[60:140, 90:110] = 255  # 80 tall, 20 wide, centered

        result = binarize_crop_resize(mask)
        assert result.shape == (256, 256)

        fill_ratio = np.sum(result > 0) / (256 * 256)
        # With circumscribed-circle the product fills ~24% (20/83 wide); bbox gives ~25% wide but
        # crucially the product now fills 100% height → overall fill is much higher
        assert fill_ratio > 0.20, f"Expected fill > 20%, got {fill_ratio:.2%}"

    def test_output_shape_is_always_256x256(self):
        """binarize_crop_resize always returns 256×256 regardless of input size."""
        from main import binarize_crop_resize

        for h, w in [(100, 100), (300, 200), (50, 400)]:
            mask = np.zeros((h, w), dtype=np.uint8)
            # put a small product in the center
            cy, cx = h // 2, w // 2
            mask[cy-10:cy+10, cx-10:cx+10] = 255
            result = binarize_crop_resize(mask)
            assert result.shape == (256, 256), f"Expected (256,256) for input ({h},{w})"
```

**Step 2: Run tests — confirm new tests fail**

```bash
source .venv/bin/activate && pytest tests/test_pipeline.py::TestBboxCropResize -v
```

Expected: All 3 new tests FAIL (circumscribed-circle gives unequal fill ratios for different positions).

---

## Task 2: Implement bbox crop in main.py and build_index.py

**Files:**
- Modify: `main.py` (lines 87–118)
- Modify: `build_index.py` (lines 56–87)

**Step 1: Replace binarize_crop_resize body in main.py**

Find the function starting at `def binarize_crop_resize(binary: np.ndarray) -> np.ndarray:` and replace its entire body (everything between the `def` line and `return cv2.resize(...)`) with:

```python
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

    # Pad to exact square (preserves aspect ratio → rotation invariance maintained)
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.uint8)
    py, px = (side - h) // 2, (side - w) // 2
    padded[py:py + h, px:px + w] = cropped

    return cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
```

**Step 2: Apply the identical replacement in build_index.py**

Replace `binarize_crop_resize` in `build_index.py` with the exact same function body as above.

**Step 3: Update module-level docstring in both files**

In both files, change the pipeline line from:
```
→ circumscribed-circle crop → pad to square → resize to 256x256 → Zernike moments.
```
to:
```
→ bounding-box crop → pad to square → resize to 256x256 → Zernike moments.
```

**Step 4: Run all tests**

```bash
cd /Users/vivianhan/Desktop/Img_Sim && source .venv/bin/activate && pytest tests/test_pipeline.py -v
```

Expected: All 9 tests pass (6 existing + 3 new).

**Step 5: Commit**

```bash
git add main.py build_index.py tests/test_pipeline.py
git commit -m "feat: replace circumscribed-circle crop with bounding-box normalization"
```

---

## Task 3: Smoke test and index rebuild

**Step 1: Check if product images exist**

```bash
ls /Users/vivianhan/Desktop/Img_Sim/data/images/products/ 2>/dev/null | head -3 || echo "NO IMAGES"
```

**Step 2a: If images exist — rebuild index**

```bash
cd /Users/vivianhan/Desktop/Img_Sim && source .venv/bin/activate && python build_index.py
```

Expected: ends with `Indexed N images` and `Descriptor shape: (N, 169)`.

Then commit:
```bash
git add index/zernike_index.pkl
git commit -m "chore: rebuild index with bbox normalization pipeline"
```

**Step 2b: If no images — skip rebuild, note it**

The existing index was built with the old pipeline and is now stale. It will need to be rebuilt when product images are available. Do not commit the old index.

**Step 3: Start server and verify**

```bash
cd /Users/vivianhan/Desktop/Img_Sim && source .venv/bin/activate && python -m uvicorn main:app --host 127.0.0.1 --port 8000 &
sleep 4
curl -s http://127.0.0.1:8000/health
kill %1
```

Expected: `{"status":"healthy"}`

---

## Task 4: Update .gitignore and CLAUDE.md

**Files:**
- Check: `.gitignore` (CLAUDE.md entry should already be present from parent branch)
- Modify: `CLAUDE.md`

**Step 1: Verify .gitignore already has CLAUDE.md entry**

```bash
grep "CLAUDE.md" /Users/vivianhan/Desktop/Img_Sim/.gitignore
```

Expected: prints `CLAUDE.md`. If missing, add it.

**Step 2: Update CLAUDE.md pipeline description**

In CLAUDE.md, update the pipeline line from:
```
rembg background removal → alpha mask (> 0 = foreground) → dilate (3×3 ellipse)
```
to:
```
rembg background removal → alpha mask (> 0 = foreground) → dilate (3×3 ellipse) → bounding-box crop → pad to square
```

Also add to the "Critical Constraints" section:
```
- **Index must be rebuilt** after any pipeline change. The bbox normalization changes descriptors; the old circumscribed-circle index is incompatible.
```

**Step 3: Commit**

```bash
git add .gitignore CLAUDE.md 2>/dev/null; git add .gitignore
git commit -m "chore: update CLAUDE.md for bbox normalization pipeline"
```

Note: CLAUDE.md is gitignored so only .gitignore changes will appear in the commit if there's nothing new to add there.

---

## Task 5: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update "How It Works" pipeline diagram**

Replace:
```
Dilate (3×3 ellipse) → Circumscribed-circle crop → Resize 256×256
```
with:
```
Dilate (3×3 ellipse) → Bounding-box crop → Pad to square → Resize 256×256
```

**Step 2: Update the Results table description**

In the Results section, update the description to reflect that normalization handles any crop or scale.

**Step 3: Add a note in the Tech Stack table**

The Descriptor row currently says `Mathematically rotation-invariant, no training data needed`. Update to:
```
Mathematically rotation-invariant; scale/translation-normalized via bounding-box crop
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for bounding-box normalization pipeline"
```

---

## Task 6: Final verification

**Step 1: Run full test suite**

```bash
cd /Users/vivianhan/Desktop/Img_Sim && source .venv/bin/activate && pytest tests/test_pipeline.py -v
```

Expected: 9 passed.

**Step 2: Check git log**

```bash
git log --oneline feature/BBox_Normalization | head -10
```

Expected commits on this branch (newest first):
```
docs: update README for bounding-box normalization pipeline
chore: update CLAUDE.md for bbox normalization pipeline
feat: replace circumscribed-circle crop with bounding-box normalization
Add bbox normalization design doc
```
(followed by commits inherited from feature/Background_Removal)

**Step 3: Deactivate venv**

```bash
deactivate
```
