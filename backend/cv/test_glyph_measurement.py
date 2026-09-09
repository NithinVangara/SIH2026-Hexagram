import cv2
import numpy as np
import pytest

from backend.cv.glyph_measurement import estimate_glyph_height


def test_glyph_height_is_smaller_than_ocr_region():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    cv2.putText(
        image,
        "TEST",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    result = estimate_glyph_height(
        image,
        [10, 20, 150, 80],
    )

    assert result["status"] == "MEASURED"
    assert result["glyph_height"] > 0
    assert result["glyph_height"] < 60


def test_glyph_measurement_returns_foreground_geometry():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cv2.rectangle(
        image,
        (20, 30),
        (70, 60),
        (255, 255, 255),
        -1,
    )

    result = estimate_glyph_height(
        image,
        [10, 20, 80, 70],
    )

    assert result["status"] == "MEASURED"
    assert result["glyph_height"] > 0
    assert result["foreground_width"] > 0
    assert result["foreground_area"] > 0
    assert 0.0 <= result["measurement_confidence"] <= 1.0


def test_empty_foreground_is_reported():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = estimate_glyph_height(
        image,
        [10, 10, 80, 80],
    )

    assert result["status"] == "NO_FOREGROUND"
    assert result["glyph_height"] == 0
    assert result["measurement_confidence"] == 0.0


def test_bbox_is_clipped_to_image():
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    cv2.rectangle(
        image,
        (5, 5),
        (30, 30),
        (255, 255, 255),
        -1,
    )

    result = estimate_glyph_height(
        image,
        [-20, -20, 60, 60],
    )

    assert result["status"] == "MEASURED"
    assert result["glyph_height"] > 0


def test_invalid_bbox_is_rejected():
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        estimate_glyph_height(
            image,
            [20, 20, 10, 10],
        )


def test_empty_image_is_rejected():
    image = np.empty((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        estimate_glyph_height(
            image,
            [0, 0, 10, 10],
        )