# Robust Background Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the hardcoded BINARY_THRESHOLD=220 binarization step with rembg background removal so the pipeline works on real-world photos with any background.

**Architecture:** Add a `remove_background(path)` function that calls `rembg.remove()` and extracts the alpha channel as a binary foreground mask. Update `binarize_crop_resize` to accept this pre-computed mask instead of thresholding grayscale. Changes are identical in `main.py` and `build_index.py`. Everything downstream (dilate → crop → resize → Zernike) is untouched.

**Tech Stack:** Python, rembg>=2.0.50, onnxruntime>=1.16.0, Pillow (already in requirements), pytest (new, dev only)

---

## Task 1: Set up virtual environment and install dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Create and activate venv**

```bash
cd /Users/vivianhan/Desktop/Img_Sim
python3 -m venv .venv
source .venv/bin/activate
```

**Step 2: Add rembg to requirements.txt**

Add these two lines after the existing `Pillow` line:

```
rembg>=2.0.50
onnxruntime>=1.16.0
```

**Step 3: Install all dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without error. On first import of rembg, the U2Net model (~170MB) downloads to ~/.u2net/. This only happens once.

**Step 4: Verify rembg is importable**

```bash
python -c "from rembg import remove; print('rembg ok')"
```

Expected: `rembg ok`

**Step 5: Commit**

```bash
git add requirements.txt
git commit -m "feat: add rembg and onnxruntime dependencies"
```

---

## Task 2: Write failing tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_pipeline.py`

**Step 1: Create tests directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 2: Create tests/test_pipeline.py with this content:**

```python
"""
Pipeline tests for background removal integration.

Run with:
    pytest tests/test_pipeline.py -v
"""
import numpy as np
from PIL import Image


def _make_test_image(path: str, width: int = 200, height: int = 200) -> None:
    """Create a synthetic test image: light gray background, dark gray rectangle."""
    img = Image.new("RGB", (width, height), (240, 240, 240))
    pixels = img.load()
    for y in range(60, 140):
        for x in range(60, 140):
            pixels[x, y] = (40, 40, 40)
    img.save(path)


def _make_blank_image(path: str, width: int = 200, height: int = 200) -> None:
    """Create an all-white image (no foreground after removal)."""
    Image.new("RGB", (width, height), (255, 255, 255)).save(path)


class TestRemoveBackground:
    def test_returns_binary_mask_correct_shape(self, tmp_path):
        img_path = str(tmp_path / "product.png")
        _make_test_image(img_path)
        from main import remove_background
        mask = remove_background(img_path)
        assert mask.shape == (200, 200), f"Expected (200, 200), got {mask.shape}"

    def test_returns_uint8_dtype(self, tmp_path):
        img_path = str(tmp_path / "product.png")
        _make_test_image(img_path)
        from main import remove_background
        mask = remove_background(img_path)
        assert mask.dtype == np.uint8

    def test_values_are_binary(self, tmp_path):
        img_path = str(tmp_path / "product.png")
        _make_test_image(img_path)
        from main import remove_background
        mask = remove_background(img_path)
        unique_vals = set(np.unique(mask).tolist())
        assert unique_vals.issubset({0, 255}), f"Non-binary values: {unique_vals}"


class TestExtractZernike:
    def test_descriptor_has_169_dimensions(self, tmp_path):
        img_path = str(tmp_path / "product.png")
        _make_test_image(img_path)
        from main import extract_zernike
        desc = extract_zernike(img_path)
        assert desc.shape == (169,), f"Expected (169,), got {desc.shape}"

    def test_descriptor_is_float_array(self, tmp_path):
        img_path = str(tmp_path / "product.png")
        _make_test_image(img_path)
        from main import extract_zernike
        desc = extract_zernike(img_path)
        assert desc.dtype in (np.float32, np.float64)

    def test_blank_image_does_not_raise(self, tmp_path):
        """Empty mask is handled by existing guard - returns zero descriptor, no crash."""
        img_path = str(tmp_path / "blank.png")
        _make_blank_image(img_path)
        from main import extract_zernike
        desc = extract_zernike(img_path)
        assert desc.shape == (169,)
```

**Step 3: Run tests - confirm they all fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: All 6 tests FAIL with `ImportError: cannot import name 'remove_background' from 'main'`

---

## Task 3: Implement changes in main.py

**Files:**
- Modify: `main.py`

**Step 1: Add rembg imports at the top**

After `import cv2`, add:

```python
import io
from PIL import Image
from rembg import remove as rembg_remove
```

**Step 2: Remove BINARY_THRESHOLD constant**

Delete this line:
```python
BINARY_THRESHOLD = 220  # Must match build_index.py
```

**Step 3: Delete load_fullres_gray function**

Delete the entire `load_fullres_gray` function (loads image and converts to grayscale). It is no longer called anywhere.

**Step 4: Add remove_background function before binarize_crop_resize**

```python
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
```

**Step 5: Update binarize_crop_resize**

Change signature: `def binarize_crop_resize(gray: np.ndarray)` → `def binarize_crop_resize(binary: np.ndarray)`

Remove these lines (the threshold step):
```python
    binary = np.where(gray < BINARY_THRESHOLD, np.uint8(255), np.uint8(0))

    # Dilate to normalize line thickness across rotation angles
```

Replace with just:
```python
    # Dilate to normalize line thickness across rotation angles
```

Update the docstring to remove the binarize step and rename param from `gray` to `binary`.

**Step 6: Update extract_zernike**

Replace:
```python
    gray = load_fullres_gray(path)
    binary = binarize_crop_resize(gray)
```
with:
```python
    binary = remove_background(path)
    binary = binarize_crop_resize(binary)
```

---

## Task 4: Implement identical changes in build_index.py

**Files:**
- Modify: `build_index.py`

Apply exactly the same changes as Task 3 to `build_index.py`:
- Add `import io`, `from PIL import Image`, `from rembg import remove as rembg_remove` after `import cv2`
- Delete `BINARY_THRESHOLD = 220`
- Delete `load_fullres_gray` function
- Add `remove_background` function (identical code)
- Update `binarize_crop_resize` (identical change)
- Update `extract_zernike` (identical change)

**Commit after both files are updated:**

```bash
git add main.py build_index.py tests/
git commit -m "feat: replace threshold binarization with rembg background removal"
```

---

## Task 5: Run tests - confirm they all pass

**Step 1: Run full test suite**

```bash
pytest tests/test_pipeline.py -v
```

Expected:
```
tests/test_pipeline.py::TestRemoveBackground::test_returns_binary_mask_correct_shape PASSED
tests/test_pipeline.py::TestRemoveBackground::test_returns_uint8_dtype PASSED
tests/test_pipeline.py::TestRemoveBackground::test_values_are_binary PASSED
tests/test_pipeline.py::TestExtractZernike::test_descriptor_has_169_dimensions PASSED
tests/test_pipeline.py::TestExtractZernike::test_descriptor_is_float_array PASSED
tests/test_pipeline.py::TestExtractZernike::test_blank_image_does_not_raise PASSED

6 passed
```

If any test fails, debug before proceeding.

---

## Task 6: Rebuild the index

**Step 1: Run build_index.py**

```bash
python build_index.py
```

Expected: ends with `Indexed N images` and `Descriptor shape: (N, 169)`.

Note: First run triggers U2Net model download if not already cached.

**Step 2: Verify index**

```bash
python -c "
import pickle, numpy as np
with open('index/zernike_index.pkl', 'rb') as f:
    d = pickle.load(f)
print('paths:', len(d['paths']))
print('descriptors:', d['descriptors'].shape)
print('ok' if d['descriptors'].shape[1] == 169 else 'WRONG SHAPE')
"
```

Expected: `descriptors: (N, 169)` and `ok`.

**Step 3: Commit rebuilt index**

```bash
git add index/zernike_index.pkl
git commit -m "chore: rebuild index with rembg pipeline"
```

---

## Task 7: Manual smoke test

**Step 1: Start server**

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Expected: `Loaded index: N images, descriptor shape (N, 169)` then `Application startup complete.`

**Step 2: Health check**

```bash
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"healthy"}`

**Step 3: Search with an existing product image**

```bash
curl -X POST http://127.0.0.1:8000/search \
  -F "file=@data/images/products/<any_existing_image>.png"
```

Expected: JSON with `results` array, top score >= 0.80.

**Step 4: Stop server (Ctrl+C)**

---

## Task 8: Update .gitignore and CLAUDE.md

**Files:**
- Modify: `.gitignore`
- Modify: `CLAUDE.md`

**Step 1: Add to .gitignore** (check .venv/ isn't already there first)

```
# Claude guidance (local only)
CLAUDE.md

# Virtual environment
.venv/
```

**Step 2: Update CLAUDE.md commands section**

Replace the install command with venv-based workflow:

```bash
# Set up virtual environment (first time)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
# Note: first run downloads U2Net model (~170MB) to ~/.u2net/

# Run tests
pytest tests/ -v
```

Update pipeline description: replace `binarize (< 220 = foreground)` with `rembg background removal → alpha mask (> 0 = foreground)`.

Remove BINARY_THRESHOLD from the "Constants must stay in sync" section.

**Step 3: Commit**

```bash
git add .gitignore CLAUDE.md
git commit -m "chore: update gitignore and CLAUDE.md for rembg pipeline"
```

---

## Task 9: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update "How It Works" pipeline diagram**

Replace `Grayscale → Binarize (threshold=220)` with:
```
rembg background removal → alpha mask (foreground = alpha > 0)
```

**Step 2: Update Quickstart (macOS/Linux)**

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Note: first run downloads U2Net model (~170MB) to ~/.u2net/

# 2. Build frontend (first time only)
cd frontend && npm install && npm run build && cd ..
cp -r frontend/dist frontend_dist

# 3. Rebuild index
python build_index.py

# 4. Start server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Step 3: Update Dependencies section**

Add to the dependency list:
```
rembg, onnxruntime              # Background removal (U2Net, CPU-only)
```

Add note: `rembg downloads the U2Net model (~170MB) to ~/.u2net/ on first use. No internet required after that.`

**Step 4: Update Tech Stack table**

Add row:
```
| Background removal | rembg + U2Net (ONNX Runtime) | CPU-only, handles any real-world background |
```

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README for rembg background removal pipeline"
```

---

## Task 10: Final verification

**Step 1: Run tests one last time**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 6 passed.

**Step 2: Check git log**

```bash
git log --oneline
```

Expected commits on this branch (newest first):
```
docs: update README for rembg background removal pipeline
chore: update gitignore and CLAUDE.md for rembg pipeline
chore: rebuild index with rembg pipeline
feat: replace threshold binarization with rembg background removal
feat: add rembg and onnxruntime dependencies
Add background removal design doc
```

**Step 3: Deactivate venv**

```bash
deactivate
```
