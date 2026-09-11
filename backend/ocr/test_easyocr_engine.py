from unittest.mock import patch

from backend.ocr.easyocr_engine import EasyOCREngine


def test_bbox_is_flattened_to_xyxy():
    bbox = [[10, 20], [40, 20], [40, 60], [10, 60]]
    assert EasyOCREngine._flatten_bbox(bbox) == [10, 20, 40, 60]


def test_bbox_handles_rotated_quad():
    bbox = [[20.4, 30.2], [50.8, 25.7], [55.1, 55.6], [24.9, 60.3]]
    assert EasyOCREngine._flatten_bbox(bbox) == [20, 26, 55, 60]


def test_read_maps_easyocr_output_to_m1_text_blocks():
    engine = object.__new__(EasyOCREngine)
    engine.reader = type("Reader", (), {
        "readtext": lambda self, image: [
            ([[10, 20], [40, 20], [40, 60], [10, 60]], "MRP Rs 120", 0.93),
            ([[50, 70], [90, 70], [90, 90], [50, 90]], "500 g", 0.88),
        ]
    })()

    result = engine.read("product.png")

    assert result == [
        {"id": "R01", "text": "MRP Rs 120", "confidence": 0.93, "bbox": [10, 20, 40, 60]},
        {"id": "R02", "text": "500 g", "confidence": 0.88, "bbox": [50, 70, 90, 90]},
    ]


def test_inspect_preserves_image_id_and_text_blocks():
    engine = object.__new__(EasyOCREngine)
    with patch.object(engine, "read", return_value=[
        {"id": "R01", "text": "Magic Masala", "confidence": 0.99, "bbox": [10, 20, 40, 60]}
    ]):
        result = engine.inspect("product.png", "IMG001")

    assert result == {
        "image_id": "IMG001",
        "text_blocks": [
            {"id": "R01", "text": "Magic Masala", "confidence": 0.99, "bbox": [10, 20, 40, 60]}
        ],
    }
