import re
from typing import Any


UNIT_PATTERN = r"kg|kgs|kilograms?|g|gm|gms|grams?|mg|milligrams?|l|ltr|litre|litres|liter|liters|ml|millilitres?|milliliters?"

EXPLICIT_QUANTITY_PATTERN = re.compile(
    rf"\b(?:NET\s+QUANTITY|NET\s+QTY|NET\s+WEIGHT|NET\s+WT|NET\s+VOLUME)\s*[:\-]?\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>{UNIT_PATTERN})\b",
    re.IGNORECASE,
)

UNIT_ONLY_PATTERN = re.compile(
    rf"\b(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>{UNIT_PATTERN})\b",
    re.IGNORECASE,
)

QUANTITY_LABEL_PATTERN = re.compile(
    r"\b(?:NET\s+QUANTITY|NET\s+QTY|NET\s+WEIGHT|NET\s+WT|NET\s+VOLUME)\b",
    re.IGNORECASE,
)

UNIT_MAP = {
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "l": "L", "ltr": "L", "litre": "L", "litres": "L", "liter": "L", "liters": "L",
    "ml": "mL", "millilitre": "mL", "millilitres": "mL", "milliliter": "mL", "milliliters": "mL",
}


def _ocr_confidence(block: dict[str, Any]) -> float:
    try:
        confidence = float(block.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def _parse_value(value: str) -> int | float:
    number = float(value.replace(",", ""))
    return int(number) if number.is_integer() else number


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


def _spatial_distance(label_block: dict[str, Any], value_block: dict[str, Any]) -> float | None:
    label_box = _bbox(label_block)
    value_box = _bbox(value_block)
    if label_box is None or value_box is None:
        return None
    lx1, ly1, lx2, ly2 = label_box
    vx1, vy1, vx2, vy2 = value_box
    dx = max(lx1 - vx2, vx1 - lx2, 0.0)
    dy = max(ly1 - vy2, vy1 - ly2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _make_candidate(block: dict[str, Any], match: re.Match, confidence_factor: float, strength: str) -> dict[str, Any]:
    return {
        "value": _parse_value(match.group("value")),
        "unit": _normalize_unit(match.group("unit")),
        "confidence": round(_ocr_confidence(block) * confidence_factor, 4),
        "source_regions": [block.get("id")],
        "strength": strength,
    }


def _unit_only_candidate(block: dict[str, Any]) -> dict[str, Any] | None:
    text = str(block.get("text", "")).strip()
    region_id = block.get("id")
    if not text or not region_id:
        return None
    match = UNIT_ONLY_PATTERN.fullmatch(text)
    if not match:
        return None
    return _make_candidate(block, match, 0.65, "UNIT_ONLY")


def _find_spatial_quantity_candidate(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Associate a labelled quantity region with a nearby value-only region."""
    for label_index, label_block in enumerate(blocks):
        text = str(label_block.get("text", ""))
        label_id = label_block.get("id")
        if not label_id or not QUANTITY_LABEL_PATTERN.search(text):
            continue

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

        if nearby:
            nearby.sort(key=lambda item: (item[0], item[1]))
            _, _, candidate, value_block = nearby[0]
            return {
                "value": candidate["value"],
                "unit": candidate["unit"],
                "confidence": round(min(_ocr_confidence(label_block), _ocr_confidence(value_block)) * 0.90, 4),
                "source_regions": [label_id, value_block.get("id")],
                "strength": "SPATIAL_LABEL",
            }

    return None


def _region_order(text_blocks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        block["id"]: index
        for index, block in enumerate(text_blocks)
        if isinstance(block, dict) and block.get("id")
    }


def _collect_source_regions(candidates: list[dict[str, Any]], text_blocks: list[dict[str, Any]]) -> list[str]:
    source_regions = []
    for candidate in candidates:
        source_regions.extend(candidate["source_regions"])
    region_order = _region_order(text_blocks)
    source_regions = list(dict.fromkeys(source_regions))
    source_regions.sort(key=lambda region_id: region_order.get(region_id, len(region_order)))
    return source_regions


def extract_quantity(text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract net quantity/weight/volume from OCR text blocks."""
    if not isinstance(text_blocks, list):
        raise TypeError("text_blocks must be a list")

    explicit_candidates = []
    unit_only_candidates = []

    for block in text_blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text", ""))
        region_id = block.get("id")
        if not text or not region_id:
            continue

        # Explicit evidence is matched only when the quantity value belongs
        # directly to the net-quantity/weight/volume label. This prevents a
        # second unrelated quantity later in the same OCR region from being
        # promoted to explicit evidence.
        explicit_matches = list(EXPLICIT_QUANTITY_PATTERN.finditer(text))
        for match in explicit_matches:
            explicit_candidates.append(
                _make_candidate(block, match, 0.98, "EXPLICIT_QUANTITY")
            )

        # If the OCR region is not an explicit declaration, retain unit-only
        # observations as lower-confidence corroborating evidence.
        if not explicit_matches:
            for match in UNIT_ONLY_PATTERN.finditer(text):
                unit_only_candidates.append(
                    _make_candidate(block, match, 0.65, "UNIT_ONLY")
                )

    spatial_candidate = _find_spatial_quantity_candidate(text_blocks)
    if spatial_candidate:
        explicit_candidates.append(spatial_candidate)

    if explicit_candidates:
        preferred_values = {
            (candidate["value"], candidate["unit"])
            for candidate in explicit_candidates
        }
        corroborating = [
            candidate
            for candidate in unit_only_candidates
            if (candidate["value"], candidate["unit"]) in preferred_values
        ]
        candidates = explicit_candidates + corroborating
    else:
        candidates = unit_only_candidates

    if not candidates:
        return {
            "value": None,
            "unit": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        }

    combinations = {(candidate["value"], candidate["unit"]) for candidate in candidates}
    if len(combinations) > 1:
        return {
            "value": None,
            "unit": None,
            "confidence": round(max(candidate["confidence"] for candidate in candidates), 4),
            "source_regions": _collect_source_regions(candidates, text_blocks),
            "extraction_status": "CONFLICTING",
        }

    best_candidate = max(
        candidates,
        key=lambda candidate: (
            1 if candidate["strength"] in {"EXPLICIT_QUANTITY", "SPATIAL_LABEL"} else 0,
            candidate["confidence"],
        ),
    )
    status = "UNCERTAIN" if best_candidate["strength"] == "UNIT_ONLY" else "FOUND"

    return {
        "value": best_candidate["value"],
        "unit": best_candidate["unit"],
        "confidence": best_candidate["confidence"],
        "source_regions": _collect_source_regions(candidates, text_blocks),
        "extraction_status": status,
    }
