from backend.extraction.dates import extract_dates
from backend.extraction.quantity import extract_quantity


def test_split_net_quantity_uses_spatial_value_region():
    blocks = [
        {
            "id": "R31",
            "text": "1N X Joyful Lavender 10g",
            "confidence": 0.91,
            "bbox": [423, 767, 637, 803],
        },
        {
            "id": "R35",
            "text": "1N X Soulful Jasmine 10g",
            "confidence": 0.916,
            "bbox": [422, 827, 633, 864],
        },
        {
            "id": "R37",
            "text": "Net Quantity:",
            "confidence": 1.0,
            "bbox": [424, 876, 542, 908],
        },
        {
            "id": "R41",
            "text": "3 Nx -",
            "confidence": 0.245,
            "bbox": [422, 916, 478, 948],
        },
        {
            "id": "R42",
            "text": "30 g",
            "confidence": 0.470,
            "bbox": [578, 908, 622, 940],
        },
    ]

    result = extract_quantity(blocks)

    assert result["value"] == 30
    assert result["unit"] == "g"
    assert result["extraction_status"] == "FOUND"
    assert result["source_regions"] == ["R37", "R42"]


def test_combined_mfd_use_by_label_associates_two_spatial_dates():
    blocks = [
        {
            "id": "R18",
            "text": "Mfd. & use by date:",
            "confidence": 0.624,
            "bbox": [503, 525, 661, 561],
        },
        {
            "id": "R25",
            "text": "05/26,04/29",
            "confidence": 0.585,
            "bbox": [468, 658, 614, 686],
        },
    ]

    result = extract_dates(blocks)

    assert result["manufacturing_date"]["value"] == "2026-05"
    assert result["manufacturing_date"]["precision"] == "MONTH_YEAR"
    assert result["manufacturing_date"]["extraction_status"] == "FOUND"
    assert result["manufacturing_date"]["source_regions"] == ["R18", "R25"]

    assert result["expiry_date"]["value"] == "2029-04"
    assert result["expiry_date"]["precision"] == "MONTH_YEAR"
    assert result["expiry_date"]["extraction_status"] == "FOUND"
    assert result["expiry_date"]["source_regions"] == ["R18", "R25"]
