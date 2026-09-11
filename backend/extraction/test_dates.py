from backend.extraction.dates import extract_dates


def test_manufacturing_date():
    blocks = [{"id": "R01", "text": "MFG DATE: 08/2026", "confidence": 0.95}]
    result = extract_dates(blocks)
    assert result["manufacturing_date"]["value"] == "2026-08"
    assert result["manufacturing_date"]["precision"] == "MONTH_YEAR"
    assert result["manufacturing_date"]["extraction_status"] == "FOUND"
    assert result["manufacturing_date"]["source_regions"] == ["R01"]


def test_packing_date():
    blocks = [{"id": "R01", "text": "PKD 15/08/2026", "confidence": 0.94}]
    result = extract_dates(blocks)
    assert result["packing_date"]["value"] == "2026-08-15"
    assert result["packing_date"]["precision"] == "DAY_MONTH_YEAR"
    assert result["packing_date"]["extraction_status"] == "FOUND"


def test_expiry_date():
    blocks = [{"id": "R01", "text": "EXPIRY DATE: 08/2028", "confidence": 0.96}]
    result = extract_dates(blocks)
    assert result["expiry_date"]["value"] == "2028-08"
    assert result["expiry_date"]["precision"] == "MONTH_YEAR"
    assert result["expiry_date"]["extraction_status"] == "FOUND"


def test_use_by_date():
    blocks = [{"id": "R01", "text": "USE BY 12/02/2027", "confidence": 0.95}]
    result = extract_dates(blocks)
    assert result["expiry_date"]["value"] == "2027-02-12"
    assert result["expiry_date"]["extraction_status"] == "FOUND"


def test_split_packing_date_uses_spatial_value_region():
    blocks = [
        {"id": "R01", "text": "PKD", "confidence": 0.94, "bbox": [10, 10, 45, 30]},
        {"id": "R02", "text": "13/08/2026", "confidence": 0.92, "bbox": [50, 10, 130, 30]},
        {"id": "R03", "text": "2027", "confidence": 0.99, "bbox": [300, 300, 340, 320]},
    ]
    result = extract_dates(blocks)
    assert result["packing_date"]["value"] == "2026-08-13"
    assert result["packing_date"]["source_regions"] == ["R01", "R02"]


def test_split_use_by_date_uses_spatial_value_region():
    blocks = [
        {"id": "R01", "text": "USE BY", "confidence": 0.94, "bbox": [10, 10, 60, 30]},
        {"id": "R02", "text": "12/02/2027", "confidence": 0.92, "bbox": [65, 10, 145, 30]},
    ]
    result = extract_dates(blocks)
    assert result["expiry_date"]["value"] == "2027-02-12"
    assert result["expiry_date"]["source_regions"] == ["R01", "R02"]


def test_best_before():
    blocks = [{"id": "R01", "text": "BEST BEFORE 24 MONTHS", "confidence": 0.93}]
    result = extract_dates(blocks)
    assert result["best_before"]["value"] == 24
    assert result["best_before"]["unit"] == "months"
    assert result["best_before"]["extraction_status"] == "FOUND"


def test_day_month_year():
    blocks = [{"id": "R01", "text": "MFD 15/08/2026", "confidence": 0.91}]
    result = extract_dates(blocks)
    assert result["manufacturing_date"]["value"] == "2026-08-15"
    assert result["manufacturing_date"]["precision"] == "DAY_MONTH_YEAR"


def test_month_year():
    blocks = [{"id": "R01", "text": "MFG 08-2026", "confidence": 0.90}]
    result = extract_dates(blocks)
    assert result["manufacturing_date"]["value"] == "2026-08"
    assert result["manufacturing_date"]["precision"] == "MONTH_YEAR"


def test_ocr_punctuation_variation():
    blocks = [{"id": "R01", "text": "Mfg. Date - 08.2026", "confidence": 0.88}]
    result = extract_dates(blocks)
    assert result["manufacturing_date"]["value"] == "2026-08"


def test_conflicting_expiry_dates():
    blocks = [
        {"id": "R01", "text": "EXP 08/2028", "confidence": 0.95},
        {"id": "R02", "text": "EXP 09/2028", "confidence": 0.92},
    ]
    result = extract_dates(blocks)
    assert result["expiry_date"]["value"] is None
    assert result["expiry_date"]["extraction_status"] == "CONFLICTING"
    assert result["expiry_date"]["source_regions"] == ["R01", "R02"]


def test_duplicate_identical_dates():
    blocks = [
        {"id": "R01", "text": "MFG 08/2026", "confidence": 0.90},
        {"id": "R02", "text": "MFG 08/2026", "confidence": 0.95},
    ]
    result = extract_dates(blocks)
    assert result["manufacturing_date"]["value"] == "2026-08"
    assert result["manufacturing_date"]["extraction_status"] == "FOUND"


def test_no_date():
    blocks = [{"id": "R01", "text": "PEPSICO", "confidence": 0.97}]
    result = extract_dates(blocks)
    assert result["manufacturing_date"]["extraction_status"] == "MISSING"
    assert result["packing_date"]["extraction_status"] == "MISSING"
    assert result["expiry_date"]["extraction_status"] == "MISSING"
    assert result["best_before"]["extraction_status"] == "MISSING"


def test_invalid_date():
    blocks = [{"id": "R01", "text": "MFG 32/15/2026", "confidence": 0.80}]
    result = extract_dates(blocks)
    assert result["manufacturing_date"]["extraction_status"] == "MISSING"
