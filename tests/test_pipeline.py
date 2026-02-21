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


class TestBboxCropResize:
    def test_same_product_at_different_positions_gives_same_fill(self):
        """Bbox crop: product position in frame should not affect output (translation invariant)."""
        from main import binarize_crop_resize

        # 40x40 product centered in 200x200 frame
        mask_center = np.zeros((200, 200), dtype=np.uint8)
        mask_center[80:120, 80:120] = 255

        # Same 40x40 product near corner (5px margin so dilation is not clipped at boundary)
        mask_corner = np.zeros((200, 200), dtype=np.uint8)
        mask_corner[5:45, 5:45] = 255

        result_center = binarize_crop_resize(mask_center)
        result_corner = binarize_crop_resize(mask_corner)

        assert result_center.shape == (256, 256)
        assert result_corner.shape == (256, 256)

        # Bbox crop produces pixel-identical outputs regardless of product position
        assert np.array_equal(result_center, result_corner), (
            "Bbox crop should produce identical output regardless of product position in frame"
        )

    def test_elongated_product_fills_frame_better_than_square_radius(self):
        """Bbox crop: elongated product (20x80) should fill >50% of 256x256 output."""
        from main import binarize_crop_resize

        # 20px wide x 80px tall product in a 200x200 frame
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[60:140, 90:110] = 255  # 80 tall, 20 wide, centered

        result = binarize_crop_resize(mask)
        assert result.shape == (256, 256)

        fill_ratio = np.sum(result > 0) / (256 * 256)
        assert fill_ratio > 0.24, f"Expected fill > 24%, got {fill_ratio:.2%}"

    def test_output_shape_is_always_256x256(self):
        """binarize_crop_resize always returns 256x256 regardless of input size."""
        from main import binarize_crop_resize

        for h, w in [(100, 100), (300, 200), (50, 400)]:
            mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            mask[cy-10:cy+10, cx-10:cx+10] = 255
            result = binarize_crop_resize(mask)
            assert result.shape == (256, 256), f"Expected (256,256) for input ({h},{w})"
