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
