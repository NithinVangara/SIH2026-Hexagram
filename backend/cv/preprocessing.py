from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply conservative preprocessing for OCR/visual analysis.

    The function deliberately avoids aggressive transformations that could alter
    declaration text. It returns both the processed image and an audit record.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced, {
        "input_shape": list(image.shape),
        "output_shape": list(enhanced.shape),
        "operations": ["grayscale", "gaussian_blur", "clahe"],
    }
