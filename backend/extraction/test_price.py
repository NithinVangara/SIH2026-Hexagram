from backend.extraction.price import extract_mrp


def test_explicit_mrp():
    blocks = [
        {
            "id": "R01",
            "text": "MRP ₹120",
            "confidence": 0.97,
        }
    ]

    result = extract_mrp(blocks)

    assert result["value"] == 120
    assert result["currency"] == "INR"
    assert result["extraction_status"] == "FOUND"
    assert result["source_regions"] == ["R01"]


def test_mrp_rs_format():
    blocks = [
        {
            "id": "R01",
            "text": "MRP Rs. 120",
            "confidence": 0.95,
        }
    ]

    result = extract_mrp(blocks)

    assert result["value"] == 120
    assert result["extraction_status"] == "FOUND"


def test_mrp_without_decimal():
    blocks = [
        {
            "id": "R01",
            "text": "M.R.P. Rs 99.50",
            "confidence": 0.96,
        }
    ]

    result = extract_mrp(blocks)

    assert result["value"] == 99.50
    assert result["extraction_status"] == "FOUND"


def test_split_mrp_regions():
    blocks = [
        {
            "id": "R01",
            "text": "MRP",
            "confidence": 0.95,
        },
        {
            "id": "R02",
            "text": "₹120",
            "confidence": 0.94,
        },
    ]

    result = extract_mrp(blocks)

    assert result["value"] == 120
    assert result["extraction_status"] == "FOUND"
    assert result["source_regions"] == ["R01", "R02"]


def test_currency_only_is_uncertain():
    blocks = [
        {
            "id": "R01",
            "text": "₹120",
            "confidence": 0.97,
        }
    ]

    result = extract_mrp(blocks)

    assert result["value"] == 120
    assert result["extraction_status"] == "UNCERTAIN"


def test_no_mrp_is_missing():
    blocks = [
        {
            "id": "R01",
            "text": "PEPSICO",
            "confidence": 0.97,
        }
    ]

    result = extract_mrp(blocks)

    assert result["value"] is None
    assert result["extraction_status"] == "MISSING"
    assert result["source_regions"] == []


def test_conflicting_mrp_values():
    blocks = [
        {
            "id": "R01",
            "text": "MRP ₹120",
            "confidence": 0.97,
        },
        {
            "id": "R02",
            "text": "MRP ₹125",
            "confidence": 0.96,
        },
    ]

    result = extract_mrp(blocks)

    assert result["value"] is None
    assert result["extraction_status"] == "CONFLICTING"
    assert result["source_regions"] == ["R01", "R02"]