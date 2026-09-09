from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(image_path: str):
    """Load an image from disk in OpenCV BGR format."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. Supported formats: {SUPPORTED_EXTENSIONS}"
        )
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def get_image_info(image):
    """Return basic information about a loaded image."""
    height, width = image.shape[:2]
    return {
        "width": width,
        "height": height,
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
    }


def assess_image_quality(image: np.ndarray) -> dict:
    """Return a conservative, explainable visual-quality assessment.

    The score is a screening signal, not a guarantee that OCR will succeed.
    Thresholds are intentionally exposed so they can be tuned against the
    representative package-image set during Phase 3 QA.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings: list[str] = []
    if brightness < 45:
        warnings.append("low_brightness")
    elif brightness > 220:
        warnings.append("high_brightness")
    if contrast < 25:
        warnings.append("low_contrast")
    if sharpness < 60:
        warnings.append("possible_blur")

    # Normalize metrics into bounded screening signals. These are not model confidence.
    brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    contrast_score = min(1.0, contrast / 80.0)
    sharpness_score = min(1.0, sharpness / 500.0)
    score = round(
        0.30 * brightness_score + 0.30 * contrast_score + 0.40 * sharpness_score,
        4,
    )

    if score >= 0.65 and not warnings:
        status = "ACCEPTED"
    elif score >= 0.35:
        status = "REVIEW"
    else:
        status = "REJECTED"

    return {
        "status": status,
        "score": score,
        "metrics": {
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "sharpness": round(sharpness, 3),
        },
        "warnings": warnings,
    }
