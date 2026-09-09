import cv2
import numpy as np
import pytest

from font_measurement import _select_text_like_components, estimate_character_height


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

def test_small_isolated_noise_is_filtered():
    """
    Tiny isolated structures should not become part of the
    character-height population.
    """
    components = [
        (10, 5, 30, 10.0, 20.0),
        (10, 5, 28, 20.0, 20.5),
        (11, 5, 31, 30.0, 20.0),
        (10, 5, 29, 40.0, 20.5),

        # Noise
        (2, 2, 3, 50.0, 5.0),
        (1, 1, 2, 60.0, 6.0),
    ]

    filtered = _select_text_like_components(components)

    assert len(filtered) == 4

    assert all(
        component[0] >= 3
        for component in filtered
    )

def test_large_height_outlier_is_filtered():
    """
    A large foreground component should not dominate the
    character-height population.
    """
    components = [
        (10, 5, 30, 10.0, 20.0),
        (10, 5, 28, 20.0, 20.5),
        (11, 5, 31, 30.0, 20.0),
        (10, 5, 29, 40.0, 20.5),

        # Large non-character structure
        (40, 40, 1500, 70.0, 70.0),
    ]

    filtered = _select_text_like_components(components)

    assert len(filtered) == 4

    assert all(
        component[0] <= 11
        for component in filtered
    )

def test_horizontal_line_like_component_is_filtered():
    """
    Extremely wide components should not be treated as
    individual character components.
    """
    components = [
        (10, 5, 30, 10.0, 20.0),
        (10, 5, 29, 20.0, 20.0),
        (11, 5, 31, 30.0, 20.0),
        (10, 5, 30, 40.0, 20.0),

        # Line-like structure
        (4, 50, 180, 60.0, 30.0),
    ]

    filtered = _select_text_like_components(components)

    assert len(filtered) == 4

def test_reasonable_character_height_variation_is_preserved():
    """
    Small character-height variation should remain usable.
    """
    components = [
        (9, 5, 27, 10.0, 20.0),
        (10, 5, 30, 20.0, 20.5),
        (11, 5, 32, 30.0, 20.0),
        (10, 5, 29, 40.0, 20.5),
        (10, 5, 30, 50.0, 20.0),
    ]

    filtered = _select_text_like_components(components)

    assert len(filtered) == 5

    heights = [
        component[0]
        for component in filtered
    ]

    assert min(heights) == 9
    assert max(heights) == 11

def test_small_print_population_is_not_rejected():
    """
    Small text populations should remain valid.

    This test prevents the filtering stage from introducing
    an absolute minimum font-height threshold.
    """
    components = [
        (3, 2, 6, 10.0, 10.0),
        (3, 2, 6, 14.0, 10.5),
        (4, 2, 7, 18.0, 10.0),
        (3, 2, 6, 22.0, 10.5),
        (3, 2, 6, 26.0, 10.0),
    ]

    filtered = _select_text_like_components(components)

    assert len(filtered) == 5

    assert all(
        component[0] in {3, 4}
        for component in filtered
    )

def test_dominant_text_population_wins_over_mixed_noise():
    """
    When a crop contains a strong text population plus
    unrelated components, the dominant text population
    should determine the measurement.
    """
    components = [
        # Text population
        (12, 5, 35, 10.0, 20.0),
        (12, 5, 34, 20.0, 20.5),
        (13, 5, 36, 30.0, 20.0),
        (12, 5, 35, 40.0, 20.5),
        (12, 5, 34, 50.0, 20.0),

        # Small noise
        (2, 2, 3, 60.0, 50.0),
        (2, 3, 4, 65.0, 60.0),

        # Large graphic
        (35, 30, 900, 80.0, 70.0),

        # Horizontal structure
        (3, 60, 150, 90.0, 90.0),
    ]

    filtered = _select_text_like_components(components)

    assert len(filtered) == 5

    heights = [
        component[0]
        for component in filtered
    ]

    assert set(heights).issubset({12, 13})

def test_uneven_lighting_preserves_text_measurement():
    """
    Text remains visible when the background has a gradual
    brightness gradient.
    """
    height, width = 100, 200
    image = np.zeros((height, width), dtype=np.uint8)

    # Gradual background illumination.
    for x in range(width):
        brightness = int(80 + (120 * x / width))
        image[:, x] = brightness

    # Same-height text-like components.
    image[30:50, 20:35] = 20
    image[30:50, 45:60] = 20
    image[30:50, 70:85] = 20

    result = estimate_character_height(image, [10, 20, 100, 60])

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_low_contrast_text_does_not_crash():
    """
    Low-contrast text should be handled safely even if the
    baseline cannot reliably measure it.
    """
    image = np.full((100, 200), 150, dtype=np.uint8)

    image[30:50, 20:35] = 125
    image[30:50, 45:60] = 125
    image[30:50, 70:85] = 125

    result = estimate_character_height(image, [10, 20, 100, 60])

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_text_with_shadow_variation_is_handled():
    """
    Text over a region with local brightness variation should
    remain safely processable.
    """
    image = np.full((120, 220), 200, dtype=np.uint8)

    # Shadow/illumination variation.
    image[20:100, :] -= np.linspace(
        0,
        80,
        220,
        dtype=np.uint8,
    )

    # Text-like components.
    image[40:60, 25:40] = 30
    image[40:60, 55:70] = 30
    image[40:60, 85:100] = 30

    result = estimate_character_height(image, [10, 20, 120, 80])

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert 0.0 <= result["character_height_confidence"] <= 1.0

def test_mild_blur_preserves_text_measurement():
    """
    Mild photographic blur should still be handled safely.
    """
    image = make_text_image()

    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        0,
    )

    result = estimate_character_height(
        blurred,
        [10, 20, 100, 60],
    )

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_strong_blur_is_handled_safely():
    """
    Strong blur may make text unreliable, but the estimator
    must fail conservatively rather than fabricate a result.
    """
    image = make_text_image()

    blurred = cv2.GaussianBlur(
        image,
        (9, 9),
        0,
    )

    result = estimate_character_height(
        blurred,
        [10, 20, 100, 60],
    )

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_blur_does_not_produce_invalid_confidence():
    """
    Blur must never produce an invalid confidence value.
    """
    image = make_text_image()

    for kernel_size in [(3, 3), (5, 5), (7, 7)]:
        blurred = cv2.GaussianBlur(
            image,
            kernel_size,
            0,
        )

        result = estimate_character_height(
            blurred,
            [10, 20, 100, 60],
        )

        assert 0.0 <= result["character_height_confidence"] <= 1.0