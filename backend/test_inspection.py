from backend.inspection import inspect_image


class StubOCREngine:
    def read(self, image):
        assert image == "product.png"
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


def test_inspect_image_connects_ocr_to_declaration_extraction():
    result = inspect_image("product.png", "IMG001", ocr_engine=StubOCREngine())

    assert result["image_id"] == "IMG001"
    assert result["text_blocks"][0]["id"] == "R01"
    assert result["declarations"]["mrp"]["value"] == 120
    assert result["declarations"]["mrp"]["source_regions"] == ["R01"]
    assert result["declarations"]["net_quantity"]["value"] == 500
    assert result["declarations"]["net_quantity"]["source_regions"] == ["R02"]
