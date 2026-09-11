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


QUANTITY_LABEL_PATTERN = re.compile(
    r"\b(?:NET\s+QUANTITY|NET\s+QTY|NET\s+WEIGHT|NET\s+WT|NET\s+VOLUME)\b",
    re.IGNORECASE,
)


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


def _bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = block.get("bbox")

    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, raw)
    except (TypeError, ValueError):
        return None

    if x2 < x1 or y2 < y1:
        return None

    return x1, y1, x2, y2


def _spatial_distance(
    label_block: dict[str, Any],
    value_block: dict[str, Any],
) -> float | None:
    label_box = _bbox(label_block)
    value_box = _bbox(value_block)

    if label_box is None or value_box is None:
        return None

    lx1, ly1, lx2, ly2 = label_box
    vx1, vy1, vx2, vy2 = value_box

    # Distance between rectangles. Zero means they overlap/touch.
    dx = max(lx1 - vx2, vx1 - lx2, 0.0)
    dy = max(ly1 - vy2, vy1 - ly2, 0.0)

    return (dx * dx + dy * dy) ** 0.5


def _unit_only_candidate(
    block: dict[str, Any],
) -> dict[str, Any] | None:
    text = str(block.get("text", ""))
    region_id = block.get("id")

    if not text or not region_id:
        return None

    match = QUANTITY_PATTERN.fullmatch(text.strip())
    if not match:
        return None

    return {
        "value": _parse_value(match.group("value")),
        "unit": _normalize_unit(match.group("unit")),
        "confidence": round(_ocr_confidence(block) * 0.65, 4),
        "source_regions": [region_id],
        "strength": "UNIT_ONLY",
    }


def _find_spatial_quantity_candidate(
    blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Associate a labelled quantity region with a nearby value region."""
    for label_index, label_block in enumerate(blocks):
        text = str(label_block.get("text", ""))
        label_id = label_block.get("id")

        if not label_id or not QUANTITY_LABEL_PATTERN.search(text):
            continue

        # Prefer a later spatially-near OCR region containing a standalone
        # numeric + unit. This handles labels and values split by OCR.
        nearby = []

        for value_index, value_block in enumerate(blocks):
            if value_index == label_index:
                continue

            candidate = _unit_only_candidate(value_block)
            if candidate is None:
                continue

            distance = _spatial_distance(label_block, value_block)
            if distance is None or distance > 140:
                continue

            nearby.append((distance, value_index, candidate, value_block))

        if not nearby:
            continue

        nearby.sort(key=lambda item: (item[0], item[1]))
        _, _, candidate, value_block = nearby[0]

        label_conf = _ocr_confidence(label_block)
        value_conf = _ocr_confidence(value_block)

        return {
            "value": candidate["value"],
            "unit": candidate["unit"],
            "confidence": round(min(label_conf, value_conf) * 0.90, 4),
            "source_regions": [label_id, value_block.get("id")],
            "strength": "SPATIAL_LABEL",
        }

    return None


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
    explicit_candidates = []

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

            explicit_label = bool(QUANTITY_LABEL_PATTERN.search(text))
            ocr_confidence = _ocr_confidence(block)

            if explicit_label:
                confidence = ocr_confidence * 0.98
                strength = "EXPLICIT_QUANTITY"
            else:
                confidence = ocr_confidence * 0.65
                strength = "UNIT_ONLY"

            candidate = {
                "value": value,
                "unit": unit,
                "confidence": round(confidence, 4),
                "source_regions": [region_id],
                "strength": strength,
            }
            candidates.append(candidate)

            if explicit_label:
                explicit_candidates.append(candidate)

    # If the label and value are separate OCR regions, prefer the spatially
    # associated candidate over unrelated unit-only quantities elsewhere.
    spatial_candidate = _find_spatial_quantity_candidate(text_blocks)
    if spatial_candidate:
        explicit_candidates.append(spatial_candidate)

    if explicit_candidates:
        candidates = explicit_candidates

    if not candidates:
        return {
            "value": None,
            "unit": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        }

    combinations = {
        (candidate["value"], candidate["unit"])
        for candidate in candidates
    }

    if len(combinations) > 1:
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
