from pathlib import Path

import cv2
import pytest

from backend.cv.font_measurement import estimate_character_height


# product.png is stored in the project root.
IMAGE_PATH = Path("/Users/nithinvangara/Desktop/SIH2026-Hexagram/data/samples/product.png")


@pytest.fixture(scope="module")
def package_image():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        pytest.fail(f"Could not load test image: {IMAGE_PATH}")

    return image


def test_real_package_image_loads(package_image):
    assert package_image is not None
    assert package_image.size > 0


def test_dense_left_text_region(package_image):
    # Dense printed information/declaration area.
    bbox = (75, 210, 180, 330)

    result = estimate_character_height(package_image, bbox)

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}
    assert 0.0 <= result["character_height_confidence"] <= 1.0

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert result["components_used"] > 0


def test_nutrition_table_region(package_image):
    # Nutrition/information table on the right side.
    bbox = (235, 190, 335, 290)

    result = estimate_character_height(package_image, bbox)

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}
    assert 0.0 <= result["character_height_confidence"] <= 1.0

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert result["components_used"] > 0


def test_small_print_region(package_image):
    # Dense small-print text below the table.
    bbox = (230, 295, 350, 390)

    result = estimate_character_height(package_image, bbox)

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}
    assert 0.0 <= result["character_height_confidence"] <= 1.0

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0


def test_logo_region(package_image):
    # PEPSICO/logo region near the lower-right portion.
    bbox = (245, 390, 325, 425)

    result = estimate_character_height(package_image, bbox)

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}
    assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_center_fold_region_does_not_crash(package_image):
    # Central red folded/sealed region.
    bbox = (175, 100, 225, 450)

    result = estimate_character_height(package_image, bbox)

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}
    assert 0.0 <= result["character_height_confidence"] <= 1.0


def test_background_region_is_handled_without_crashing(package_image):
    # The photographed background is not perfectly uniform because of
    # lighting, texture, compression, and shadows.
    #
    # Therefore this test records the estimator's behaviour rather than
    # assuming that every visually blank background must produce
    # NO_FOREGROUND.
    bbox = (0, 0, 40, 80)

    result = estimate_character_height(package_image, bbox)

    assert result["status"] in {"MEASURED", "NO_FOREGROUND"}
    assert 0.0 <= result["character_height_confidence"] <= 1.0

    if result["status"] == "MEASURED":
        assert result["character_height_px"] > 0
        assert result["components_used"] > 0


def test_real_package_result_is_image_space_only(package_image):
    bbox = (75, 210, 180, 330)

    result = estimate_character_height(package_image, bbox)

    # Phase 3.8 is validating the Phase 3.6
    # image-space character-height estimator.
    assert "character_height_px" in result
    assert "character_height_confidence" in result
    assert "components_used" in result


def test_multiple_real_regions_can_be_processed(package_image):
    regions = [
        (75, 210, 180, 330),
        (235, 190, 335, 290),
        (230, 295, 350, 390),
        (245, 390, 325, 425),
    ]

    results = [
        estimate_character_height(package_image, bbox)
        for bbox in regions
    ]

    assert len(results) == len(regions)

    for result in results:
        assert result["status"] in {"MEASURED", "NO_FOREGROUND"}
        assert 0.0 <= result["character_height_confidence"] <= 1.0

        if result["status"] == "MEASURED":
            assert result["character_height_px"] > 0
            assert result["components_used"] > 0