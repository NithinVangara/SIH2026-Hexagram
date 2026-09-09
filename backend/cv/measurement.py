from __future__ import annotations

from typing import Any


def measure_text_regions(text_blocks: list[dict[str, Any]], pixels_per_mm: float | None = None) -> dict[str, Any]:
    """Prototype glyph/text-region measurement from OCR geometry.

    Without a physical scale, measurements are reported in pixels only and the
    result remains REVIEW-worthy. A scale calibration converts the height to mm.
    """
    measurements = []
    for block in text_blocks:
        bbox = block.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width_px = abs(x2 - x1)
        height_px = abs(y2 - y1)
        item = {
            "source_region": block.get("id"),
            "width_px": round(width_px, 3),
            "height_px": round(height_px, 3),
            "confidence": float(block.get("confidence", 0.0)),
        }
        if pixels_per_mm and pixels_per_mm > 0:
            item["width_mm"] = round(width_px / pixels_per_mm, 3)
            item["height_mm"] = round(height_px / pixels_per_mm, 3)
            item["status"] = "REVIEW" if item["confidence"] < 0.80 else "ESTIMATE"
        else:
            item["status"] = "REVIEW"
            item["reason"] = "physical_scale_unavailable"
        measurements.append(item)

    return {
        "measurements": measurements,
        "scale": {
            "pixels_per_mm": pixels_per_mm,
            "calibrated": bool(pixels_per_mm and pixels_per_mm > 0),
        },
    }
