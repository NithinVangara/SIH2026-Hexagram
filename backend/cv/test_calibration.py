import pytest

from backend.cv.calibration import calculate_scale, pixels_to_mm


def test_calculate_scale():
    result = calculate_scale(
        reference_length_px=100,
        reference_length_mm=10,
    )

    assert result["status"] == "CALIBRATED"
    assert result["mm_per_pixel"] == pytest.approx(0.1)
    assert result["pixels_per_mm"] == pytest.approx(10.0)


def test_pixels_to_mm():
    scale = calculate_scale(
        reference_length_px=100,
        reference_length_mm=10,
    )

    result = pixels_to_mm(
        measurement_px=25,
        scale=scale,
    )

    assert result["status"] == "CONVERTED"
    assert result["measurement_mm"] == pytest.approx(2.5)
    assert result["measurement_confidence"] == 1.0


def test_uncalibrated_measurement_is_not_converted():
    result = pixels_to_mm(
        measurement_px=25,
        scale={"status": "UNCALIBRATED"},
    )

    assert result["status"] == "UNCALIBRATED"
    assert result["measurement_mm"] is None
    assert result["measurement_confidence"] == 0.0


def test_zero_reference_pixels_are_rejected():
    with pytest.raises(ValueError):
        calculate_scale(
            reference_length_px=0,
            reference_length_mm=10,
        )


def test_zero_reference_mm_are_rejected():
    with pytest.raises(ValueError):
        calculate_scale(
            reference_length_px=100,
            reference_length_mm=0,
        )


def test_negative_measurement_is_rejected():
    scale = calculate_scale(
        reference_length_px=100,
        reference_length_mm=10,
    )

    with pytest.raises(ValueError):
        pixels_to_mm(
            measurement_px=-1,
            scale=scale,
        )
        