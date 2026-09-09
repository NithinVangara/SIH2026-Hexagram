from pathlib import Path
from typing import Any

from paddleocr import PaddleOCR


class OCREngine:
    """
    Thin wrapper around PaddleOCR.

    Responsibilities:
    - Run OCR on one image.
    - Preserve raw OCR text, confidence, polygons, and bounding boxes.
    - Convert PaddleOCR output into our stable internal text_blocks format.

    Not responsible for:
    - Legal compliance.
    - MRP/net-quantity extraction.
    - Rule evaluation.
    """

    def __init__(
        self,
        lang: str = "en",
        ocr_version: str = "PP-OCRv4",
    ):
        self.ocr = PaddleOCR(
            lang=lang,
            ocr_version=ocr_version,
        )

    def process(self, image_path: str) -> dict[str, Any]:
        """
        Run OCR on a single image.

        Args:
            image_path: Path to the image.

        Returns:
            Stable OCR result containing text blocks and raw evidence.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = self.ocr.predict(str(path))[0]

        text_blocks = []

        texts = result["rec_texts"]
        scores = result["rec_scores"]
        boxes = result["rec_boxes"]
        polygons = result["rec_polys"]

        for index, (text, score, box, polygon) in enumerate(
            zip(texts, scores, boxes, polygons),
            start=1,
        ):
            text_blocks.append(
                {
                    "id": f"R{index:02d}",
                    "text": text,
                    "confidence": float(score),
                    "bbox": box.tolist(),
                    "polygon": polygon.tolist(),
                }
            )

        return {
            "text_blocks": text_blocks,
            "orientation": {
                "textline_angles": list(
                    result["textline_orientation_angles"]
                )
            },
        }