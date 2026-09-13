from __future__ import annotations

from pathlib import Path

import numpy as np


class EasyOCREngine:
    """M1 OCR adapter that exposes raw text, confidence, and source boxes."""

    DECLARATION_ANCHORS = (
        "mrp",
        "maximum retail",
        "net quantity",
        "net weight",
        "net content",
        "mfd",
        "mfg",
        "pkd",
        "packed on",
        "use by",
        "use before",
        "expiry",
        "best before",
    )

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

    @classmethod
    def _anchor_score(cls, result: list) -> int:
        score = 0
        for item in result:
            if len(item) < 2:
                continue
            text = str(item[1]).lower()
            score += sum(1 for anchor in cls.DECLARATION_ANCHORS if anchor in text)
        return score

    def _read_raw(self, image: str | np.ndarray):
        result = self.reader.readtext(image)

        # EasyOCR can detect many characters on a rotated package while
        # missing the declaration labels that matter to M1. In that case,
        # retry quarter-turn recognition and prefer the orientation exposing
        # more declaration anchors. This keeps rotation handling inside OCR
        # rather than changing the downstream observation contract.
        try:
            rotated_result = self.reader.readtext(
                image,
                rotation_info=[90, 180, 270],
            )
        except TypeError:
            rotated_result = []

        if rotated_result and self._anchor_score(rotated_result) > self._anchor_score(result):
            result = rotated_result
        elif not result and rotated_result:
            result = rotated_result

        return result

    def read(self, image: str | Path | np.ndarray) -> list[dict]:
        """Run OCR and return M1 text blocks with stable IDs."""
        if isinstance(image, Path):
            image = str(image)

        result = self._read_raw(image)
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
