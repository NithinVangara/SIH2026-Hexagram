from __future__ import annotations

from pathlib import Path

import numpy as np


class EasyOCREngine:
    """M1 OCR adapter that exposes raw text, confidence, and source boxes."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("EasyOCR is not installed") from exc

        self.languages = languages or ["en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    @staticmethod
    def _flatten_bbox(bbox: list[list[float]]) -> list[int]:
        points = np.asarray(bbox, dtype=float).reshape(-1, 2)
        x_min = int(round(points[:, 0].min()))
        y_min = int(round(points[:, 1].min()))
        x_max = int(round(points[:, 0].max()))
        y_max = int(round(points[:, 1].max()))
        return [x_min, y_min, x_max, y_max]

    def read(self, image: str | Path | np.ndarray) -> list[dict]:
        """Run OCR and return M1 text blocks with stable IDs."""
        result = self.reader.readtext(image)
        text_blocks: list[dict] = []

        for index, item in enumerate(result, start=1):
            bbox, text, confidence = item
            text_blocks.append(
                {
                    "id": f"R{index:02d}",
                    "text": str(text),
                    "confidence": float(confidence),
                    "bbox": self._flatten_bbox(bbox),
                }
            )

        return text_blocks

    def inspect(self, image: str | Path | np.ndarray, image_id: str) -> dict:
        """Return the OCR portion of the frozen M1 inspection contract."""
        return {"image_id": image_id, "text_blocks": self.read(image)}
