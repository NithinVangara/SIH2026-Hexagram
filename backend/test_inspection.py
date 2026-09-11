from backend.inspection import inspect_image


class StubOCREngine:
    def __init__(self):
        self.calls = 0

    def read(self, image):
        self.calls += 1
        assert str(image) == "product.png"
        return [
            {
                "id": "R01",
                "text": "MRP Rs 120",
                "confidence": 0.93,
                "bbox": [10, 20, 110, 40],
            },
            {
                "id": "R02",
                "text": "Net Weight 500 g",
                "confidence": 0.91,
                "bbox": [10, 50, 150, 70],
            },
        ]


class CountingOCREngine:
    def __init__(self):
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return []


def _patch_quality(monkeypatch, status):
    quality = {
        "status": status,
        "score": 0.9,
        "warnings": [] if status == "ACCEPTED" else ["BLURRY"],
        "metrics": {
            "width": 500,
            "height": 700,
            "blur_score": 500.0,
            "brightness": 120.0,
            "contrast": 55.0,
        },
    }

    monkeypatch.setattr(
        "backend.inspection.load_image",
        lambda image: object(),
    )
    monkeypatch.setattr(
        "backend.inspection.assess_image_quality",
        lambda image: quality,
    )


def test_inspect_image_connects_ocr_to_declaration_extraction(monkeypatch):
    _patch_quality(monkeypatch, "ACCEPTED")
    result = inspect_image("product.png", "IMG001", ocr_engine=StubOCREngine())

    assert result["image_id"] == "IMG001"
    assert result["quality"]["status"] == "ACCEPTED"
    assert result["text_blocks"][0]["id"] == "R01"
    assert result["declarations"]["mrp"]["value"] == 120
    assert result["declarations"]["mrp"]["source_regions"] == ["R01"]
    assert result["declarations"]["net_quantity"]["value"] == 500
    assert result["declarations"]["net_quantity"]["source_regions"] == ["R02"]


def test_review_quality_still_runs_ocr(monkeypatch):
    _patch_quality(monkeypatch, "REVIEW")
    ocr = CountingOCREngine()

    result = inspect_image("product.png", "IMG001", ocr_engine=ocr)

    assert result["quality"]["status"] == "REVIEW"
    assert ocr.calls == 1
    assert result["text_blocks"] == []
    assert result["declarations"] == {
        "mrp": {
            "value": None,
            "currency": "INR",
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        },
        "net_quantity": {
            "value": None,
            "unit": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        },
        "manufacturing_date": {
            "value": None,
            "precision": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        },
        "packing_date": {
            "value": None,
            "precision": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        },
        "expiry_date": {
            "value": None,
            "precision": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        },
        "best_before": {
            "value": None,
            "unit": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        },
    }


def test_rejected_quality_skips_ocr(monkeypatch):
    _patch_quality(monkeypatch, "REJECTED")
    ocr = CountingOCREngine()

    result = inspect_image("product.png", "IMG001", ocr_engine=ocr)

    assert result["image_id"] == "IMG001"
    assert result["quality"]["status"] == "REJECTED"
    assert ocr.calls == 0
    assert result["text_blocks"] == []
    assert result["declarations"] == {}
