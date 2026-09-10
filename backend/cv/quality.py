from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(image_path: str):
    """
    Load an image from disk.

    Raises:
        ValueError: if the path is unsupported or the image cannot be read.
    """
    path = Path(image_path)

    if not path.exists():
        raise ValueError(f"Image does not exist: {image_path}")

    if not path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image


def get_image_info(image):
    """
    Return basic image metadata.

    Args:
        image: OpenCV image as a NumPy array.

    Returns:
        dict containing width, height and channel count.
    """
    if image is None:
        raise ValueError("Image cannot be None")

    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array")

    if image.size == 0:
        raise ValueError("Image cannot be empty")

    height, width = image.shape[:2]

    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
    }


def _to_grayscale(image):
    """
    Convert an image to grayscale.
    """
    if image is None:
        raise ValueError("Image cannot be None")

    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array")

    if image.size == 0:
        raise ValueError("Image cannot be empty")

    if len(image.shape) == 2:
        return image

    if len(image.shape) == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError("Unsupported image shape")


def calculate_blur_score(image):
    """
    Estimate image sharpness using the variance of the Laplacian.

    Higher values generally indicate a sharper image.
    Lower values generally indicate blur/defocus.

    This is an image-quality signal, not a legal/compliance measurement.
    """
    gray = _to_grayscale(image)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return float(laplacian.var())


def calculate_brightness(image):
    """
    Calculate mean grayscale intensity.

    Range:
        0   -> completely black
        255 -> completely white
    """
    gray = _to_grayscale(image)

    return float(np.mean(gray))


def calculate_contrast(image):
    """
    Calculate grayscale standard deviation.

    Low standard deviation indicates a low-contrast image.
    """
    gray = _to_grayscale(image)

    return float(np.std(gray))


def _blur_quality(blur_score):
    """
    Convert the Laplacian variance into a normalized sharpness
    quality signal.

    This is only an image-quality heuristic. It is not a claim
    about OCR accuracy or legal compliance.
    """
    if blur_score >= 300:
        return 1.0
    if blur_score >= 150:
        return 0.8
    if blur_score >= 80:
        return 0.6
    if blur_score >= 40:
        return 0.4
    if blur_score >= 30:
        return 0.2
    return 0.0

def _brightness_quality(brightness):
    """
    Estimate quality from average brightness.

    Bright packaging/backgrounds can legitimately have high mean
    brightness, so brightness alone should only become a warning
    when the image is strongly overexposed.

    Range:
        0   -> black
        255 -> white
    """
    if 45.0 <= brightness <= 220.0:
        return 1.0

    if 30.0 <= brightness < 45.0:
        return 0.65

    if 220.0 < brightness <= 240.0:
        return 0.65

    return 0.0


def _contrast_quality(contrast):
    """
    Estimate quality from image contrast.
    """
    if contrast >= 60.0:
        return 1.0

    if contrast >= 40.0:
        return 0.8

    if contrast >= 25.0:
        return 0.6

    if contrast >= 15.0:
        return 0.35

    return 0.0


def assess_image_quality(image):
    """
    Assess whether an image is suitable for downstream OCR/CV processing.

    Returns a stable quality result containing:

        status:
            ACCEPTED
            REVIEW
            REJECTED

        score:
            normalized score in [0, 1]

        warnings:
            human-readable quality warnings

        metrics:
            raw image-quality measurements

    Important:
        This function does NOT decide legal compliance.
        It only reports image-quality concerns.
    """
    if image is None:
        raise ValueError("Image cannot be None")

    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array")

    if image.size == 0:
        raise ValueError("Image cannot be empty")

    info = get_image_info(image)

    width = info["width"]
    height = info["height"]

    min_dimension = min(width, height)

    warnings = []

    if min_dimension < 200:
        warnings.append("LOW_RESOLUTION")
    # A zero-sized image is already rejected above.
    # Very tiny images are unlikely to provide useful OCR evidence.
    if width < 32 or height < 32:
        return {
            "status": "REJECTED",
            "score": 0.0,
            "warnings": ["IMAGE_TOO_SMALL"],
            "metrics": {
                "width": width,
                "height": height,
                "blur_score": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
            },
        }

    blur_score = calculate_blur_score(image)
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)

    blur_quality = _blur_quality(blur_score)
    brightness_quality = _brightness_quality(brightness)
    contrast_quality = _contrast_quality(contrast)

    # Equal weighting keeps the quality gate simple and explainable.
    score = (
        blur_quality
        + brightness_quality
        + contrast_quality
    ) / 3.0

    score = float(np.clip(score, 0.0, 1.0))
    

    if blur_score < 300:
        warnings.append("BLURRY")

    if brightness < 30.0:
        warnings.append("TOO_DARK")

    if brightness > 240.0:
        warnings.append("TOO_BRIGHT")

    if contrast < 25.0:
        warnings.append("LOW_CONTRAST")

    # Strong failures should prevent downstream processing.
    severe_failure = (
        blur_score < 30
        or brightness < 20.0
        or brightness > 245.0
        or contrast < 15.0
        or min_dimension < 100
    )

    if severe_failure:
        status = "REJECTED"
    elif warnings:
        status = "REVIEW"
    else:
        status = "ACCEPTED"

    return {
        "status": status,
        "score": score,
        "warnings": warnings,
        "metrics": {
            "width": width,
            "height": height,
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
        },
    }