# backend/cv/measurement.py

from __future__ import annotations

from backend.cv.calibration import pixels_to_mm


def _validate_bbox(bbox: list | tuple) -> tuple[float, float, float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")

    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numeric") from exc

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")

    return x1, y1, x2, y2


def _get_measurement_confidence(block: dict) -> float:
    confidence = block.get("confidence")

    if confidence is None:
        return 0.0

    try:
        return max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc


def measure_text_block(
    block: dict,
    scale: dict | None = None,
) -> dict:
    """
    Measure the image-space geometry of one OCR text region.

    If an explicit calibration scale is supplied, the pixel height
    is also converted to millimetres.

    No legal interpretation is performed.
    """
    if not isinstance(block, dict):
        raise TypeError("text block must be a dictionary")

    region_id = block.get("id")
    if not region_id:
        raise ValueError("text block must contain an id")

    x1, y1, x2, y2 = _validate_bbox(block["bbox"])

    pixel_width = x2 - x1
    pixel_height = y2 - y1
    pixel_area = pixel_width * pixel_height
    aspect_ratio = pixel_width / pixel_height

    measurement_confidence = _get_measurement_confidence(block)

    result = {
        "region_id": region_id,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area": pixel_area,
        "aspect_ratio": aspect_ratio,
        "measurement_confidence": measurement_confidence,
        "height_mm": None,
        "calibration_status": "UNCALIBRATED",
    }

    if scale is not None:
        physical_result = pixels_to_mm(
            measurement_px=pixel_height,
            scale=scale,
        )

        if physical_result["status"] == "CONVERTED":
            result["height_mm"] = physical_result["measurement_mm"]
            result["calibration_status"] = "CALIBRATED"

        else:
            result["calibration_status"] = "UNCALIBRATED"

    return result


def measure_text_blocks(
    text_blocks: list[dict],
    scale: dict | None = None,
) -> list[dict]:
    """
    Measure OCR regions while preserving their original order.

    An optional calibration scale is applied to every region.
    """
    if not isinstance(text_blocks, list):
        raise TypeError("text_blocks must be a list")

    return [
        measure_text_block(block, scale=scale)
        for block in text_blocks
    ]