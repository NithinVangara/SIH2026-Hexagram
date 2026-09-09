from __future__ import annotations

import cv2
import numpy as np


def estimate_glyph_height(
    image: np.ndarray,
    bbox: list | tuple,
) -> dict:
    """
    Estimate visible text/glyph height inside an OCR bounding box.

    Returns image-space measurements only.
    No physical-unit or legal-compliance interpretation is performed.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D numpy array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")

    x1, y1, x2, y2 = map(int, bbox)

    height, width = image.shape[:2]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must define a non-empty region")

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("bbox crop must not be empty")

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Reject essentially uniform regions before thresholding.
    if int(gray.max()) == int(gray.min()):
        return {
            "status": "NO_FOREGROUND",
            "glyph_height": 0,
            "foreground_width": 0,
            "foreground_area": 0,
            "measurement_confidence": 0.0,
        }

    _, binary_normal = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    _, binary_inverted = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Choose the mask with the smaller foreground coverage.
    normal_area = int(np.count_nonzero(binary_normal))
    inverted_area = int(np.count_nonzero(binary_inverted))

    if normal_area <= inverted_area:
        binary = binary_normal
    else:
        binary = binary_inverted

    kernel = np.ones((2, 2), dtype=np.uint8)
    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    ys, xs = np.where(cleaned > 0)

    if len(ys) == 0:
        return {
            "status": "NO_FOREGROUND",
            "glyph_height": 0,
            "foreground_width": 0,
            "foreground_area": 0,
            "measurement_confidence": 0.0,
        }

    foreground_y_min = int(ys.min())
    foreground_y_max = int(ys.max())

    foreground_x_min = int(xs.min())
    foreground_x_max = int(xs.max())

    glyph_height = foreground_y_max - foreground_y_min + 1
    foreground_width = foreground_x_max - foreground_x_min + 1
    foreground_area = int(len(ys))

    crop_height = crop.shape[0]
    crop_width = crop.shape[1]

    height_ratio = glyph_height / crop_height
    area_ratio = foreground_area / (crop_height * crop_width)

    confidence = 0.0

    if foreground_area >= 4:
        confidence = 0.5

    if 0.05 <= height_ratio <= 0.95:
        confidence += 0.25

    if 0.01 <= area_ratio <= 0.80:
        confidence += 0.25

    confidence = max(0.0, min(1.0, confidence))

    return {
        "status": "MEASURED",
        "glyph_height": glyph_height,
        "foreground_width": foreground_width,
        "foreground_area": foreground_area,
        "measurement_confidence": confidence,
    }