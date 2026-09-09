import cv2
import numpy as np
import pytest

from backend.cv.font_measurement import estimate_character_height


def make_text_image(
    character_height: int = 20,
    character_width: int = 10,
    gap: int = 10,
    count: int = 5,
) -> np.ndarray:
    """
    Create a synthetic image containing several rectangular
    character-like foreground components.
    """
    width = count * character_width + (count - 1) * gap + 20
    height = character_height + 40

    image = np.zeros((height, width), dtype=np.uint8)

    x = 10
    y = 20

    for _ in range(count):
        image[y:y + character_height, x:x + character_width] = 255
        x += character_width + gap

    return image


def test_measurement_is_stable_for_multiple_known_heights():
    """
    Character-height estimates should remain close to the
    known synthetic character heights.
    """
    for expected_height in [10, 15, 20, 25, 30]:
        image = make_text_image(
            character_height=expected_height,
            count=5,
        )

        result = estimate_character_height(
            image=image,
            bbox=[0, 0, image.shape[1], image.shape[0]],
        )

        assert result["status"] == "MEASURED"
        assert result["character_height_px"] == pytest.approx(
            expected_height,
            abs=1.0,
        )


def test_measurement_is_stable_with_small_noise():
    """
    Small isolated foreground noise should not significantly
    change the estimated character height.
    """
    image = make_text_image(character_height=20, count=5)

    rng = np.random.default_rng(42)

    for _ in range(20):
        x = int(rng.integers(0, image.shape[1]))
        y = int(rng.integers(0, image.shape[0]))

        image[y, x] = 255

    result = estimate_character_height(
        image=image,
        bbox=[0, 0, image.shape[1], image.shape[0]],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(
        20.0,
        abs=1.0,
    )


def test_dark_text_on_light_background():
    """
    The measurement should work when the foreground is dark
    and the background is light.
    """
    image = np.full((80, 180), 255, dtype=np.uint8)

    image[25:45, 20:30] = 0
    image[25:45, 50:60] = 0
    image[25:45, 80:90] = 0
    image[25:45, 110:120] = 0
    image[25:45, 140:150] = 0

    result = estimate_character_height(
        image=image,
        bbox=[0, 0, image.shape[1], image.shape[0]],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(
        20.0,
        abs=1.0,
    )


def test_light_text_on_dark_background():
    """
    The measurement should work when the foreground is light
    and the background is dark.
    """
    image = make_text_image(
        character_height=20,
        count=5,
    )

    result = estimate_character_height(
        image=image,
        bbox=[0, 0, image.shape[1], image.shape[0]],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(
        20.0,
        abs=1.0,
    )


def test_multiple_characters_produce_consistent_measurement():
    """
    Several characters with the same height should produce
    a consistent central estimate.
    """
    image = np.zeros((100, 250), dtype=np.uint8)

    heights = [20, 20, 20, 20, 20]
    x_positions = [10, 50, 90, 130, 170]

    for x, height in zip(x_positions, heights):
        image[30:30 + height, x:x + 12] = 255

    result = estimate_character_height(
        image=image,
        bbox=[0, 0, image.shape[1], image.shape[0]],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(
        20.0,
        abs=1.0,
    )

    assert result["components_used"] >= 5


def test_inconsistent_character_heights_reduce_confidence():
    """
    Large variation between component heights should reduce
    confidence compared with a consistent character set.
    """
    consistent_image = make_text_image(
        character_height=20,
        count=5,
    )

    consistent_result = estimate_character_height(
        image=consistent_image,
        bbox=[
            0,
            0,
            consistent_image.shape[1],
            consistent_image.shape[0],
        ],
    )

    inconsistent_image = np.zeros((120, 300), dtype=np.uint8)

    components = [
        (10, 20),
        (50, 10),
        (90, 30),
        (130, 15),
        (170, 40),
    ]

    for x, height in components:
        inconsistent_image[50:50 + height, x:x + 12] = 255

    inconsistent_result = estimate_character_height(
        image=inconsistent_image,
        bbox=[
            0,
            0,
            inconsistent_image.shape[1],
            inconsistent_image.shape[0],
        ],
    )

    assert consistent_result["status"] == "MEASURED"
    assert inconsistent_result["status"] == "MEASURED"

    assert (
        inconsistent_result["character_height_confidence"]
        < consistent_result["character_height_confidence"]
    )


def test_background_inside_bbox_does_not_dominate_measurement():
    """
    A larger OCR bounding box containing substantial empty
    background should still measure the foreground characters.
    """
    image = np.zeros((200, 300), dtype=np.uint8)

    image[90:110, 120:132] = 255
    image[90:110, 150:162] = 255
    image[90:110, 180:192] = 255
    image[90:110, 210:222] = 255

    result = estimate_character_height(
        image=image,
        bbox=[50, 50, 280, 150],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] == pytest.approx(
        20.0,
        abs=1.0,
    )


def test_empty_measurement_returns_reviewable_result():
    """
    An empty region must not fabricate a character height.
    """
    image = np.zeros((100, 200), dtype=np.uint8)

    result = estimate_character_height(
        image=image,
        bbox=[20, 20, 180, 80],
    )

    assert result["status"] == "NO_FOREGROUND"
    assert result["character_height_px"] is None
    assert result["character_height_confidence"] == 0.0
    assert result["components_used"] == 0


def test_confidence_remains_in_valid_range():
    """
    Measurement confidence must always remain within [0, 1].
    """
    for height in [8, 12, 20, 30, 40]:
        image = make_text_image(
            character_height=height,
            count=5,
        )

        result = estimate_character_height(
            image=image,
            bbox=[0, 0, image.shape[1], image.shape[0]],
        )

        assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_large_outlier_does_not_replace_central_estimate():
    """
    One unusually large connected component should not cause
    the estimated character height to become the outlier height.
    """
    image = np.zeros((150, 300), dtype=np.uint8)

    # Normal character components: approximately 20 px.
    image[30:50, 20:32] = 255
    image[30:50, 60:72] = 255
    image[30:50, 100:112] = 255
    image[30:50, 140:152] = 255

    # Large unrelated foreground object.
    image[20:120, 220:240] = 255

    result = estimate_character_height(
        image=image,
        bbox=[0, 0, image.shape[1], image.shape[0]],
    )

    assert result["status"] == "MEASURED"

    # The robust estimate should remain close to the normal
    # character height rather than the 100 px outlier.
    assert result["character_height_px"] < 40.0


def test_measurement_preserves_pixel_units_only():
    """
    Phase 3.7 must remain an image-space validation phase.
    No physical-unit conversion should appear in the result.
    """
    image = make_text_image(
        character_height=20,
        count=5,
    )

    result = estimate_character_height(
        image=image,
        bbox=[0, 0, image.shape[1], image.shape[0]],
    )

    assert result["status"] == "MEASURED"
    assert result["character_height_px"] is not None

    assert "character_height_mm" not in result
    assert "height_mm" not in result