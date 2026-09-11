from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.cv.quality import assess_image_quality, load_image
from backend.extraction.declarations import extract_declarations
from backend.ocr import EasyOCREngine


def inspect_image(
    image: str | Path,
    image_id: str,
    ocr_engine: EasyOCREngine | None = None,
) -> dict[str, Any]:
    """Run image-quality assessment, OCR, and declaration extraction.

    M1 owns observable evidence extraction. This function does not perform
    legal or compliance evaluation.
    """
    image_path = Path(image)
    quality = assess_image_quality(load_image(image_path))

    if quality["status"] == "REJECTED":
        return {
            "image_id": image_id,
            "quality": quality,
            "text_blocks": [],
            "declarations": {},
        }

    engine = ocr_engine or EasyOCREngine(languages=["en"], gpu=False)
    text_blocks = engine.read(image_path)

    return {
        "image_id": image_id,
        "quality": quality,
        "text_blocks": text_blocks,
        "declarations": extract_declarations(text_blocks),
    }
