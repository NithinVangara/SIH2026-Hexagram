import cv2
import numpy as np
import pytest

from backend.cv.quality import (
    assess_image_quality,
    calculate_blur_score,
    calculate_brightness,
    calculate_contrast,
)


def make_clear_text_image():
    """
    Create a synthetic image with strong edges and readable text-like
    structures.

    The exact numerical values of the quality metrics are not important.
    The test only needs a clearly usable image.
    """
    image = np.full((300, 400, 3), 255, dtype=np.uint8)

    cv2.rectangle(
        image,
        (30, 30),
        (370, 270),
        (0, 0, 0),
        3,
    )

    cv2.putText(
        image,
        "MRP 120",
        (60, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "500 g",
        (80, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    return image


def test_clear_image_is_accepted():
    image = make_clear_text_image()

    result = assess_image_quality(image)

    assert result["status"] == "ACCEPTED"
    assert 0.0 <= result["score"] <= 1.0
    assert result["warnings"] == []


def test_quality_score_is_bounded():
    image = make_clear_text_image()

    result = assess_image_quality(image)

    assert 0.0 <= result["score"] <= 1.0


def test_blurry_image_produces_warning():
    image = make_clear_text_image()

    blurred = cv2.GaussianBlur(
        image,
        (15, 15),
        0,
    )

    result = assess_image_quality(blurred)

    assert result["status"] in {"REVIEW", "REJECTED"}
    assert "BLURRY" in result["warnings"]


def test_very_dark_image_produces_warning():
    image = make_clear_text_image()

    dark = cv2.convertScaleAbs(
        image,
        alpha=0.05,
        beta=0,
    )

    result = assess_image_quality(dark)

    assert result["status"] in {"REVIEW", "REJECTED"}
    assert "TOO_DARK" in result["warnings"]


def test_very_bright_image_produces_warning():
    image = make_clear_text_image()

    bright = cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=100,
    )

    result = assess_image_quality(bright)

    assert result["status"] in {"REVIEW", "REJECTED"}
    assert "TOO_BRIGHT" in result["warnings"]


def test_low_contrast_image_produces_warning():
    image = np.full(
        (300, 400, 3),
        128,
        dtype=np.uint8,
    )

    # Introduce only a very small intensity difference.
    cv2.rectangle(
        image,
        (80, 80),
        (320, 220),
        (132, 132, 132),
        -1,
    )

    result = assess_image_quality(image)

    assert result["status"] in {"REVIEW", "REJECTED"}
    assert "LOW_CONTRAST" in result["warnings"]


def test_empty_image_is_rejected():
    image = np.empty(
        (0, 0, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        assess_image_quality(image)


def test_none_image_is_rejected():
    with pytest.raises(ValueError):
        assess_image_quality(None)


def test_tiny_image_is_rejected():
    image = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    result = assess_image_quality(image)

    assert result["status"] == "REJECTED"
    assert result["score"] == 0.0
    assert "IMAGE_TOO_SMALL" in result["warnings"]


def test_quality_metrics_are_numeric():
    image = make_clear_text_image()

    blur = calculate_blur_score(image)
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)

    assert isinstance(blur, float)
    assert isinstance(brightness, float)
    assert isinstance(contrast, float)

    assert blur >= 0.0
    assert 0.0 <= brightness <= 255.0
    assert contrast >= 0.0


def test_quality_result_contains_expected_structure():
    image = make_clear_text_image()

    result = assess_image_quality(image)

    assert "status" in result
    assert "score" in result
    assert "warnings" in result
    assert "metrics" in result

    assert isinstance(result["warnings"], list)
    assert isinstance(result["metrics"], dict)

    assert "blur_score" in result["metrics"]
    assert "brightness" in result["metrics"]
    assert "contrast" in result["metrics"]


def test_normal_quality_result_does_not_generate_invalid_confidence():
    image = make_clear_text_image()

    result = assess_image_quality(image)

    assert 0.0 <= result["score"] <= 1.0

def test_blurry_image_produces_warning():
    image = make_clear_text_image()

    blurred = cv2.GaussianBlur(
        image,
        (15, 15),
        0,
    )

    result = assess_image_quality(blurred)

    print("\n================ BLUR TEST ================")
    print("Blur score:", result["metrics"]["blur_score"])
    print("Brightness:", result["metrics"]["brightness"])
    print("Contrast:", result["metrics"]["contrast"])
    print("Quality score:", result["score"])
    print("Status:", result["status"])
    print("Warnings:", result["warnings"])
    print("============================================")

    assert result["status"] in {"REVIEW", "REJECTED"}
    assert "BLURRY" in result["warnings"]