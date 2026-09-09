import re
from typing import Any


QUANTITY_PATTERN = re.compile(
    r"""
    (?:
        \b(?:NET\s+QUANTITY|NET\s+QTY|NET\s+WEIGHT|NET\s+WT|NET\s+VOLUME)
        \s*[:\-]?\s*
    )?
    (?P<value>\d+(?:[.,]\d+)?)
    \s*
    (?P<unit>
        kg|kgs|kilograms?
        |g|gm|gms|grams?
        |mg|milligrams?
        |l|ltr|litre|litres|liter|liters
        |ml|millilitres?|milliliters?
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


UNIT_MAP = {
    "kg": "kg",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "g": "g",
    "gm": "g",
    "gms": "g",
    "gram": "g",
    "grams": "g",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "l": "L",
    "ltr": "L",
    "litre": "L",
    "litres": "L",
    "liter": "L",
    "liters": "L",
    "ml": "mL",
    "millilitre": "mL",
    "millilitres": "mL",
    "milliliter": "mL",
    "milliliters": "mL",
}


def _ocr_confidence(block: dict[str, Any]) -> float:
    try:
        confidence = float(block.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return max(0.0, min(1.0, confidence))


def _parse_value(value: str) -> int | float:
    value = value.replace(",", "")

    number = float(value)

    if number.is_integer():
        return int(number)

    return number


def _normalize_unit(unit: str) -> str:
    return UNIT_MAP[unit.lower()]


def extract_quantity(text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract net quantity/weight/volume from OCR text blocks.

    Returns:
        {
            "value": 500,
            "unit": "g",
            "confidence": 0.94,
            "source_regions": ["R03"],
            "extraction_status": "FOUND"
        }

    Possible extraction_status values:
        FOUND
        MISSING
        UNCERTAIN
        CONFLICTING
    """

    if not isinstance(text_blocks, list):
        raise TypeError("text_blocks must be a list")

    candidates = []

    for block in text_blocks:
        if not isinstance(block, dict):
            continue

        text = str(block.get("text", ""))
        region_id = block.get("id")

        if not text or not region_id:
            continue

        for match in QUANTITY_PATTERN.finditer(text):
            value = _parse_value(match.group("value"))
            unit = _normalize_unit(match.group("unit"))

            explicit_label = bool(
                re.search(
                    r"\b(?:NET\s+QUANTITY|NET\s+QTY|NET\s+WEIGHT|NET\s+WT|NET\s+VOLUME)\b",
                    text,
                    re.IGNORECASE,
                )
            )

            ocr_confidence = _ocr_confidence(block)

            # Explicit "Net Quantity/Weight/Volume" gives stronger evidence
            # than a standalone value + unit.
            if explicit_label:
                confidence = ocr_confidence * 0.98
                strength = "EXPLICIT_QUANTITY"
            else:
                confidence = ocr_confidence * 0.65
                strength = "UNIT_ONLY"

            candidates.append(
                {
                    "value": value,
                    "unit": unit,
                    "confidence": round(confidence, 4),
                    "source_regions": [region_id],
                    "strength": strength,
                }
            )

    if not candidates:
        return {
            "value": None,
            "unit": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        }

    # Different value/unit combinations mean conflicting evidence.
    combinations = {
        (candidate["value"], candidate["unit"])
        for candidate in candidates
    }

    if len(combinations) > 1:
        source_regions = []

        for candidate in candidates:
            source_regions.extend(candidate["source_regions"])

        # Preserve original OCR region order.
        region_order = {
            block["id"]: index
            for index, block in enumerate(text_blocks)
            if isinstance(block, dict) and block.get("id")
        }

        source_regions = list(dict.fromkeys(source_regions))
        source_regions.sort(
            key=lambda region_id: region_order.get(
                region_id,
                len(region_order),
            )
        )

        return {
            "value": None,
            "unit": None,
            "confidence": round(
                max(candidate["confidence"] for candidate in candidates),
                4,
            ),
            "source_regions": source_regions,
            "extraction_status": "CONFLICTING",
        }

    # All candidates agree.
    best_candidate = max(
        candidates,
        key=lambda candidate: candidate["confidence"],
    )

    source_regions = []

    for candidate in candidates:
        source_regions.extend(candidate["source_regions"])

    region_order = {
        block["id"]: index
        for index, block in enumerate(text_blocks)
        if isinstance(block, dict) and block.get("id")
    }

    source_regions = list(dict.fromkeys(source_regions))
    source_regions.sort(
        key=lambda region_id: region_order.get(
            region_id,
            len(region_order),
        )
    )

    if best_candidate["strength"] == "UNIT_ONLY":
        status = "UNCERTAIN"
    else:
        status = "FOUND"

    return {
        "value": best_candidate["value"],
        "unit": best_candidate["unit"],
        "confidence": best_candidate["confidence"],
        "source_regions": source_regions,
        "extraction_status": status,
    }