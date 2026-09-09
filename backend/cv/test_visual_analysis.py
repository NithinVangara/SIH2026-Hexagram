import numpy as np

from quality import assess_image_quality
from perspective import correct_perspective
from measurement import measure_text_regions


def test_quality_contract():
    image = np.full((200, 300, 3), 128, dtype=np.uint8)
    result = assess_image_quality(image)
    assert result["status"] in {"ACCEPTED", "REVIEW", "REJECTED"}
    assert 0.0 <= result["score"] <= 1.0
    assert set(result["metrics"]) == {"brightness", "contrast", "sharpness"}


def test_perspective_without_quad_is_safe():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    corrected, meta = correct_perspective(image)
    assert corrected.shape == image.shape
    assert meta["corrected"] is False


def test_measurement_without_scale_requires_review():
    result = measure_text_regions([
        {"id": "R01", "bbox": [10, 20, 110, 50], "confidence": 0.97}
    ])
    item = result["measurements"][0]
    assert item["height_px"] == 30.0
    assert item["status"] == "REVIEW"
    assert result["scale"]["calibrated"] is False
