from __future__ import annotations

import cv2
import numpy as np


def _validate_bbox(bbox: list | tuple) -> tuple[int, int, int, int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")

    try:
        x1, y1, x2, y2 = (int(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numeric") from exc

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")

    return x1, y1, x2, y2


def _prepare_foreground_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Dark-text / light-background mask
    _, dark_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Light-text / dark-background mask
    _, light_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size
    light_ratio = np.count_nonzero(light_mask) / light_mask.size

    # Prefer the mask with the smaller foreground coverage.
    # This avoids treating the entire background as foreground.
    if dark_ratio <= light_ratio:
        mask = dark_mask
        coverage = dark_ratio
    else:
        mask = light_mask
        coverage = light_ratio

    # Reject masks that essentially cover the whole region.
    if coverage >= 0.95:
        return np.zeros_like(mask)

    return mask


def _component_heights(
    mask: np.ndarray,
    min_area: int = 2,
) -> list[int]:
    if mask.size == 0:
        return []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    heights = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        width = int(stats[label, cv2.CC_STAT_WIDTH])

        if area < min_area:
            continue

        if height <= 1 or width <= 0:
            continue

        heights.append(height)

    return heights


def estimate_character_height(
    image: np.ndarray,
    bbox: list | tuple,
) -> dict:
    """
    Estimate character/glyph height in image pixels.

    This is an image-space feasibility baseline.
    It does not claim typographic font-size accuracy
    and does not perform physical-unit conversion.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    x1, y1, x2, y2 = _validate_bbox(bbox)

    image_height, image_width = image.shape[:2]

    # Clip bbox to image boundaries.
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox does not overlap the image")

    crop = image[y1:y2, x1:x2]

    mask = _prepare_foreground_mask(crop)

    heights = _component_heights(mask)

    if not heights:
        return {
            "status": "NO_FOREGROUND",
            "character_height_px": None,
            "character_height_confidence": 0.0,
            "components_used": 0,
        }

    heights_array = np.asarray(heights, dtype=np.float32)

    # Robust central estimate.
    character_height = float(np.median(heights_array))

    component_count = len(heights)

    # Confidence is intentionally heuristic.
    # More usable components and lower spread increase confidence.
    median_height = character_height

    if median_height <= 0:
        confidence = 0.0
    else:
        deviation = float(
            np.median(np.abs(heights_array - median_height))
        )

        consistency = max(
            0.0,
            1.0 - (deviation / median_height),
        )

        sample_factor = min(
            1.0,
            component_count / 5.0,
        )

        confidence = consistency * sample_factor

    confidence = max(0.0, min(1.0, confidence))

    return {
        "status": "MEASURED",
        "character_height_px": character_height,
        "character_height_confidence": confidence,
        "components_used": component_count,
    }