from __future__ import annotations


def calculate_scale(
    reference_length_px: float,
    reference_length_mm: float,
) -> dict:
    """
    Calculate physical scale from a known reference.

    Returns:
        mm_per_pixel and pixels_per_mm.

    No legal interpretation is performed.
    """
    if reference_length_px <= 0:
        raise ValueError("reference_length_px must be greater than zero")

    if reference_length_mm <= 0:
        raise ValueError("reference_length_mm must be greater than zero")

    mm_per_pixel = reference_length_mm / reference_length_px
    pixels_per_mm = reference_length_px / reference_length_mm

    return {
        "status": "CALIBRATED",
        "mm_per_pixel": mm_per_pixel,
        "pixels_per_mm": pixels_per_mm,
    }


def pixels_to_mm(
    measurement_px: float,
    scale: dict,
) -> dict:
    """
    Convert an image-space measurement to millimetres
    using an explicitly supplied calibration scale.
    """
    if measurement_px < 0:
        raise ValueError("measurement_px must not be negative")

    if not isinstance(scale, dict):
        raise TypeError("scale must be a dictionary")

    if scale.get("status") != "CALIBRATED":
        return {
            "status": "UNCALIBRATED",
            "measurement_mm": None,
            "measurement_confidence": 0.0,
        }

    mm_per_pixel = scale.get("mm_per_pixel")

    if not isinstance(mm_per_pixel, (int, float)) or mm_per_pixel <= 0:
        raise ValueError("scale must contain a valid mm_per_pixel")

    measurement_mm = measurement_px * mm_per_pixel

    return {
        "status": "CONVERTED",
        "measurement_mm": measurement_mm,
        "measurement_confidence": 1.0,
    }