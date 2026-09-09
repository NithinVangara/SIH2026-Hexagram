import cv2
import numpy as np
import pytest

from backend.cv.font_measurement import estimate_character_height


def make_text_image():
    image = np.zeros((100, 200), dtype=np.uint8)

    # Three synthetic character-like blocks.
    image[30:50, 20:35] = 255
    image[30:50, 45:60] = 255
    image[30:50, 70:85] = 255

    return image


def test_character_height_is_measured():
    image = make_text_image()

    result = estimate_character_height(
        image=image,
        bbox=[10, 20, 100, 60],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)
    assert result["components_used"] >= 3


def test_estimate_uses_robust_central_height():
    image = np.zeros((100, 200), dtype=np.uint8)

    image[30:50, 20:35] = 255       # 20 px
    image[30:49, 45:60] = 255       # 19 px
    image[30:51, 70:85] = 255       # 21 px
    image[30:90, 100:115] = 255     # Large outlier

    result = estimate_character_height(
        image=image,
        bbox=[10, 20, 130, 95],
    )

    assert result["status"] == "MEASURED"

    # Median should resist the large 60 px outlier.
    assert result["character_height_px"] < 30


def test_small_noise_components_are_ignored():
    image = make_text_image()

    # Add tiny isolated noise pixels.
    image[10, 10] = 255
    image[12, 50] = 255
    image[80, 150] = 255

    result = estimate_character_height(
        image=image,
        bbox=[10, 20, 100, 60],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)


def test_empty_region_returns_no_foreground():
    image = np.zeros((100, 100), dtype=np.uint8)

    result = estimate_character_height(
        image=image,
        bbox=[10, 20, 90, 60],
    )

    assert result["status"] == "NO_FOREGROUND"
    assert result["character_height_px"] is None
    assert result["character_height_confidence"] == 0.0
    assert result["components_used"] == 0


def test_invalid_bbox_is_rejected():
    image = make_text_image()

    with pytest.raises(ValueError):
        estimate_character_height(
            image=image,
            bbox=[50, 20, 40, 60],
        )


def test_confidence_is_between_zero_and_one():
    image = make_text_image()

    result = estimate_character_height(
        image=image,
        bbox=[10, 20, 100, 60],
    )

    assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_dark_text_on_light_background():
    image = np.full((100, 200), 255, dtype=np.uint8)

    image[30:50, 20:35] = 0
    image[30:50, 45:60] = 0
    image[30:50, 70:85] = 0

    result = estimate_character_height(
        image=image,
        bbox=[10, 20, 100, 60],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)


def test_light_text_on_dark_background():
    image = np.zeros((100, 200), dtype=np.uint8)

    image[30:50, 20:35] = 255
    image[30:50, 45:60] = 255
    image[30:50, 70:85] = 255

    result = estimate_character_height(
        image=image,
        bbox=[10, 20, 100, 60],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(20.0)