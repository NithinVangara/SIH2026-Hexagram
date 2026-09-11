from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.extraction.declarations import extract_declarations
from backend.ocr import EasyOCREngine


def inspect_image(
    image: str | Path,
    image_id: str,
    ocr_engine: EasyOCREngine | None = None,
) -> dict[str, Any]:
    """Run real OCR and deterministic declaration extraction for one image.

    M1 owns observable evidence extraction. This function does not perform
    legal or compliance evaluation.
    """
    engine = ocr_engine or EasyOCREngine(languages=["en"], gpu=False)
    text_blocks = engine.read(image)

    return {
        "image_id": image_id,
        "text_blocks": text_blocks,
        "declarations": extract_declarations(text_blocks),
    }
