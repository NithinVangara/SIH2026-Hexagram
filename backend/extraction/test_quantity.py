from backend.extraction.quantity import extract_quantity


def test_explicit_net_quantity():
    blocks = [
        {
            "id": "R01",
            "text": "Net Quantity 500 g",
            "confidence": 0.94,
        }
    ]

    result = extract_quantity(blocks)

    assert result["value"] == 500
    assert result["unit"] == "g"
    assert result["extraction_status"] == "FOUND"
    assert result["source_regions"] == ["R01"]


def test_net_weight_kg():
    blocks = [
        {
            "id": "R01",
            "text": "Net Weight 1 kg",
            "confidence": 0.96,
        }
    ]

    result = extract_quantity(blocks)

    assert result["value"] == 1
    assert result["unit"] == "kg"
    assert result["extraction_status"] == "FOUND"


def test_net_volume_ml():
    blocks = [
        {
            "id": "R01",
            "text": "Net Volume 500 ml",
            "confidence": 0.95,
        }
    ]

    result = extract_quantity(blocks)

    assert result["value"] == 500
    assert result["unit"] == "mL"
    assert result["extraction_status"] == "FOUND"


def test_decimal_quantity():
    blocks = [
        {
            "id": "R01",
            "text": "Net Weight 1.5 kg",
            "confidence": 0.93,
        }
    ]

    result = extract_quantity(blocks)

    assert result["value"] == 1.5
    assert result["unit"] == "kg"
    assert result["extraction_status"] == "FOUND"


def test_unit_only_is_uncertain():
    blocks = [
        {
            "id": "R01",
            "text": "500 g",
            "confidence": 0.95,
        }
    ]

    result = extract_quantity(blocks)

    assert result["value"] == 500
    assert result["unit"] == "g"
    assert result["extraction_status"] == "UNCERTAIN"


def test_bare_number_is_missing():
    blocks = [
        {
            "id": "R01",
            "text": "500",
            "confidence": 0.95,
        }
    ]

    result = extract_quantity(blocks)

    assert result["value"] is None
    assert result["unit"] is None
    assert result["extraction_status"] == "MISSING"


def test_conflicting_quantities():
    blocks = [
        {
            "id": "R01",
            "text": "Net Weight 500 g",
            "confidence": 0.95,
        },
        {
            "id": "R02",
            "text": "Net Weight 1 kg",
            "confidence": 0.94,
        },
    ]

    result = extract_quantity(blocks)

    assert result["value"] is None
    assert result["unit"] is None
    assert result["extraction_status"] == "CONFLICTING"
    assert result["source_regions"] == ["R01", "R02"]


def test_same_quantity_from_multiple_regions():
    blocks = [
        {
            "id": "R01",
            "text": "Net Qty 500 g",
            "confidence": 0.90,
        },
        {
            "id": "R02",
            "text": "500 g",
            "confidence": 0.95,
        },
    ]

    result = extract_quantity(blocks)

    assert result["value"] == 500
    assert result["unit"] == "g"
    assert result["extraction_status"] == "FOUND"
    assert result["source_regions"] == ["R01", "R02"]