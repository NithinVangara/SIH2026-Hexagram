import cv2
import numpy as np
import pytest

from font_measurement import estimate_character_height


def make_text_image():
    image = np.zeros((100, 200), dtype=np.uint8)
    image[30:50, 20:35] = 255
    image[30:50, 45:60] = 255
    image[30:50, 70:85] = 255
    return image


def test_character_height_is_measured():
    result = estimate_character_height(make_text_image(), [10, 20, 100, 60])
    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)
    assert result["components_used"] >= 3


def test_estimate_uses_robust_central_height():
    image = np.zeros((100, 200), dtype=np.uint8)
    image[30:50, 20:35] = 255
    image[30:49, 45:60] = 255
    image[30:51, 70:85] = 255
    image[30:90, 100:115] = 255
    result = estimate_character_height(image, [10, 20, 130, 95])
    assert result["status"] == "MEASURED"
    assert result["character_height_px"] < 30


def test_small_noise_components_are_ignored():
    image = make_text_image()
    image[10, 10] = 255
    image[12, 50] = 255
    image[80, 150] = 255
    result = estimate_character_height(image, [10, 20, 100, 60])
    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)


def test_empty_region_returns_no_foreground():
    image = np.zeros((100, 100), dtype=np.uint8)
    result = estimate_character_height(image, [10, 20, 90, 60])
    assert result["status"] == "NO_FOREGROUND"
    assert result["character_height_px"] is None
    assert result["character_height_confidence"] == 0.0
    assert result["components_used"] == 0


def test_invalid_bbox_is_rejected():
    with pytest.raises(ValueError):
        estimate_character_height(make_text_image(), [50, 20, 40, 60])


def test_confidence_is_between_zero_and_one():
    result = estimate_character_height(make_text_image(), [10, 20, 100, 60])
    assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_dark_text_on_light_background():
    image = np.full((100, 200), 255, dtype=np.uint8)
    image[30:50, 20:35] = 0
    image[30:50, 45:60] = 0
    image[30:50, 70:85] = 0
    result = estimate_character_height(image, [10, 20, 100, 60])
    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)


def test_light_text_on_dark_background():
    result = estimate_character_height(make_text_image(), [10, 20, 100, 60])
    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)


def test_texture_without_text_is_rejected():
    rng = np.random.default_rng(42)
    image = rng.integers(80, 180, size=(120, 160), dtype=np.uint8)
    result = estimate_character_height(image, [10, 10, 150, 110])
    assert result["status"] == "NO_FOREGROUND"
    assert result["character_height_px"] is None
    assert result["character_height_confidence"] == 0.0


def test_background_like_blobs_without_row_structure_are_rejected():
    image = np.full((100, 100), 140, dtype=np.uint8)
    image[10:20, 10:18] = 220
    image[45:52, 55:83] = 40
    image[70:100, 15:55] = 210
    result = estimate_character_height(image, [0, 0, 100, 100])
    assert result["status"] == "NO_FOREGROUND"


def test_text_row_structure_is_preserved():
    image = np.full((100, 200), 255, dtype=np.uint8)
    image[30:50, 20:35] = 0
    image[31:50, 45:60] = 0
    image[29:50, 70:85] = 0
    result = estimate_character_height(image, [10, 20, 100, 60])
    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)
