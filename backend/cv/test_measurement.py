# backend/cv/test_measurement.py

import pytest

from backend.cv.measurement import measure_text_block, measure_text_blocks


def test_measure_text_block_calculates_geometry():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.97,
        "bbox": [100, 200, 300, 240],
    }

    result = measure_text_block(block)

    assert result["region_id"] == "R01"
    assert result["pixel_width"] == 200
    assert result["pixel_height"] == 40
    assert result["pixel_area"] == 8000
    assert result["aspect_ratio"] == pytest.approx(5.0)
    assert result["measurement_confidence"] == pytest.approx(0.97)


def test_measure_text_blocks_preserves_ocr_order():
    blocks = [
        {
            "id": "R01",
            "text": "MRP ₹120",
            "confidence": 0.97,
            "bbox": [10, 20, 110, 40],
        },
        {
            "id": "R02",
            "text": "500 g",
            "confidence": 0.90,
            "bbox": [20, 50, 80, 65],
        },
    ]

    result = measure_text_blocks(blocks)

    assert [item["region_id"] for item in result] == ["R01", "R02"]


def test_missing_confidence_becomes_zero():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "bbox": [10, 20, 110, 40],
    }

    result = measure_text_block(block)

    assert result["measurement_confidence"] == 0.0


def test_confidence_is_clamped():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 1.5,
        "bbox": [10, 20, 110, 40],
    }

    result = measure_text_block(block)

    assert result["measurement_confidence"] == 1.0


def test_invalid_bbox_is_rejected():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.9,
        "bbox": [100, 200, 100, 240],
    }

    with pytest.raises(ValueError):
        measure_text_block(block)


def test_invalid_bbox_shape_is_rejected():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.9,
        "bbox": [100, 200, 300],
    }

    with pytest.raises(ValueError):
        measure_text_block(block)

def test_calibrated_measurement_converts_pixel_height_to_mm():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.97,
        "bbox": [100, 200, 300, 240],
    }

    scale = {
        "status": "CALIBRATED",
        "mm_per_pixel": 0.1,
        "pixels_per_mm": 10.0,
    }

    result = measure_text_block(block, scale=scale)

    assert result["region_id"] == "R01"
    assert result["pixel_height"] == 40
    assert result["height_mm"] == pytest.approx(4.0)
    assert result["calibration_status"] == "CALIBRATED"


def test_uncalibrated_measurement_keeps_physical_height_none():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.97,
        "bbox": [100, 200, 300, 240],
    }

    scale = {
        "status": "UNCALIBRATED",
    }

    result = measure_text_block(block, scale=scale)

    assert result["pixel_height"] == 40
    assert result["height_mm"] is None
    assert result["calibration_status"] == "UNCALIBRATED"


def test_measurement_without_scale_is_uncalibrated():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.97,
        "bbox": [100, 200, 300, 240],
    }

    result = measure_text_block(block)

    assert result["pixel_height"] == 40
    assert result["height_mm"] is None
    assert result["calibration_status"] == "UNCALIBRATED"


def test_calibration_does_not_change_pixel_measurement():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.97,
        "bbox": [100, 200, 300, 240],
    }

    scale = {
        "status": "CALIBRATED",
        "mm_per_pixel": 0.25,
        "pixels_per_mm": 4.0,
    }

    result = measure_text_block(block, scale=scale)

    assert result["pixel_width"] == 200
    assert result["pixel_height"] == 40
    assert result["pixel_area"] == 8000
    assert result["aspect_ratio"] == pytest.approx(5.0)


def test_measure_text_blocks_applies_calibration_to_all_regions():
    blocks = [
        {
            "id": "R01",
            "text": "MRP ₹120",
            "confidence": 0.97,
            "bbox": [10, 20, 110, 40],
        },
        {
            "id": "R02",
            "text": "500 g",
            "confidence": 0.90,
            "bbox": [20, 50, 80, 65],
        },
    ]

    scale = {
        "status": "CALIBRATED",
        "mm_per_pixel": 0.1,
        "pixels_per_mm": 10.0,
    }

    result = measure_text_blocks(blocks, scale=scale)

    assert [item["region_id"] for item in result] == ["R01", "R02"]

    assert result[0]["pixel_height"] == 20
    assert result[0]["height_mm"] == pytest.approx(2.0)
    assert result[0]["calibration_status"] == "CALIBRATED"

    assert result[1]["pixel_height"] == 15
    assert result[1]["height_mm"] == pytest.approx(1.5)
    assert result[1]["calibration_status"] == "CALIBRATED"


def test_measurement_confidence_remains_independent_of_calibration():
    block = {
        "id": "R01",
        "text": "MRP ₹120",
        "confidence": 0.72,
        "bbox": [100, 200, 300, 240],
    }

    scale = {
        "status": "CALIBRATED",
        "mm_per_pixel": 0.1,
        "pixels_per_mm": 10.0,
    }

    result = measure_text_block(block, scale=scale)

    assert result["measurement_confidence"] == pytest.approx(0.72)
    assert result["height_mm"] == pytest.approx(4.0)