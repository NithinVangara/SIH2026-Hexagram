from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(image_path: str):
    """
    Load an image from disk.

    Returns:
        OpenCV image (BGR format).

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file format is unsupported or image cannot be read.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image


def get_image_info(image):
    """
    Return basic information about a loaded image.
    """
    height, width = image.shape[:2]

    return {
        "width": width,
        "height": height,
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
    }