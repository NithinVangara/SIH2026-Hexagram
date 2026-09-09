from backend.extraction.declarations import extract_declarations


def test_extract_declarations():
    text_blocks = [
        {
            "id": "R01",
            "text": "MRP ₹120",
            "confidence": 0.97,
        },
        {
            "id": "R02",
            "text": "Net Weight 500 g",
            "confidence": 0.95,
        },
        {
            "id": "R03",
            "text": "MFG 08/2026",
            "confidence": 0.94,
        },
        {
            "id": "R04",
            "text": "PKD 15/08/2026",
            "confidence": 0.93,
        },
        {
            "id": "R05",
            "text": "EXP 08/2028",
            "confidence": 0.92,
        },
        {
            "id": "R06",
            "text": "BEST BEFORE 24 MONTHS",
            "confidence": 0.91,
        },
    ]

    result = extract_declarations(text_blocks)

    assert result["mrp"]["value"] == 120

    assert result["net_quantity"]["value"] == 500
    assert result["net_quantity"]["unit"] == "g"

    assert result["manufacturing_date"]["value"] == "2026-08"
    assert result["packing_date"]["value"] == "2026-08-15"
    assert result["expiry_date"]["value"] == "2028-08"

    assert result["best_before"]["value"] == 24
    assert result["best_before"]["unit"] == "months"


def test_declarations_preserve_source_regions():
    text_blocks = [
        {
            "id": "R01",
            "text": "MRP ₹120",
            "confidence": 0.97,
        },
        {
            "id": "R02",
            "text": "Net Weight 500 g",
            "confidence": 0.95,
        },
        {
            "id": "R03",
            "text": "MFG 08/2026",
            "confidence": 0.94,
        },
    ]

    result = extract_declarations(text_blocks)

    assert result["mrp"]["source_regions"] == ["R01"]
    assert result["net_quantity"]["source_regions"] == ["R02"]
    assert result["manufacturing_date"]["source_regions"] == ["R03"]


def test_declarations_keep_missing_fields_explicit():
    text_blocks = [
        {
            "id": "R01",
            "text": "PEPSICO",
            "confidence": 0.97,
        }
    ]

    result = extract_declarations(text_blocks)

    assert result["mrp"]["extraction_status"] == "MISSING"
    assert result["net_quantity"]["extraction_status"] == "MISSING"
    assert result["manufacturing_date"]["extraction_status"] == "MISSING"
    assert result["packing_date"]["extraction_status"] == "MISSING"
    assert result["expiry_date"]["extraction_status"] == "MISSING"
    assert result["best_before"]["extraction_status"] == "MISSING"


def test_declarations_preserve_conflicting_status():
    text_blocks = [
        {
            "id": "R01",
            "text": "MRP ₹120",
            "confidence": 0.97,
        },
        {
            "id": "R02",
            "text": "MRP ₹130",
            "confidence": 0.96,
        },
        {
            "id": "R03",
            "text": "EXP 08/2028",
            "confidence": 0.95,
        },
        {
            "id": "R04",
            "text": "EXP 09/2028",
            "confidence": 0.94,
        },
    ]

    result = extract_declarations(text_blocks)

    assert result["mrp"]["extraction_status"] == "CONFLICTING"
    assert result["mrp"]["value"] is None

    assert result["expiry_date"]["extraction_status"] == "CONFLICTING"
    assert result["expiry_date"]["value"] is None