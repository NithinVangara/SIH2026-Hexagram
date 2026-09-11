import re
from decimal import Decimal, InvalidOperation
from typing import Any


# Explicit MRP labels.
MRP_LABEL_PATTERN = re.compile(
    r"""
    (?:
        \bM\s*\.?\s*R\s*\.?\s*P\s*\.?
        |
        \bMAXIMUM\s+RETAIL\s+PRICE\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Currency + amount.
CURRENCY_AMOUNT_PATTERN = re.compile(
    r"""
    (?:
        ₹\s*
        |
        \bRS\.?\s*
        |
        \bINR\s*
    )
    (?P<amount>\d+(?:[.,]\d{1,2})?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Amount immediately following an explicit MRP label.
MRP_AMOUNT_PATTERN = re.compile(
    r"""
    (?:
        \bM\s*\.?\s*R\s*\.?\s*P\s*\.?
        |
        \bMAXIMUM\s+RETAIL\s+PRICE\b
    )
    [:\-\s₹Rs.INR]*
    (?P<amount>\d+(?:[.,]\d{1,2})?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

BARE_AMOUNT_PATTERN = re.compile(r"\d+(?:[.,]\d{1,2})?")

# Conservative OCR sanity limit. This prevents identifiers such as long
# registration/license numbers from being interpreted as a plausible MRP.
MAX_MRP_AMOUNT = Decimal("1000000")


def _parse_amount(value: str) -> int | float:
    """Convert OCR numeric text into a JSON-friendly numeric value."""
    value = value.replace(",", "").strip()

    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid price value: {value}") from exc

    if amount < 0:
        raise ValueError("Price cannot be negative")

    if amount > MAX_MRP_AMOUNT:
        raise ValueError("Price candidate is implausibly large")

    if amount == amount.to_integral_value():
        return int(amount)

    return float(amount)


def _ocr_confidence(block: dict[str, Any]) -> float:
    """Return a bounded OCR confidence value."""
    confidence = block.get("confidence", 0.0)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, confidence))


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


def _make_candidate(
    value: int | float,
    confidence: float,
    source_regions: list[str],
    strength: str,
    currency: str = "INR",
) -> dict[str, Any]:
    return {
        "value": value,
        "currency": currency,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "source_regions": source_regions,
        "strength": strength,
    }


def _find_same_block_candidate(block: dict[str, Any]) -> dict[str, Any] | None:
    """Find an MRP candidate inside a single OCR block."""
    text = str(block.get("text", ""))
    region_id = block.get("id")

    if not text or not region_id:
        return None

    ocr_conf = _ocr_confidence(block)

    # Strongest case: explicit MRP label + amount.
    match = MRP_AMOUNT_PATTERN.search(text)
    if match:
        try:
            value = _parse_amount(match.group("amount"))
        except ValueError:
            return None

        return _make_candidate(
            value=value,
            confidence=ocr_conf * 0.98,
            source_regions=[region_id],
            strength="EXPLICIT_MRP",
        )

    # Currency without an MRP label is only supporting evidence. It remains
    # UNCERTAIN so it cannot silently become a definitive MRP declaration.
    match = CURRENCY_AMOUNT_PATTERN.search(text)
    if match:
        try:
            value = _parse_amount(match.group("amount"))
        except ValueError:
            return None

        return _make_candidate(
            value=value,
            confidence=ocr_conf * 0.60,
            source_regions=[region_id],
            strength="CURRENCY_ONLY",
        )

    return None


def _find_split_mrp_candidate(
    blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Associate an MRP label with the nearest plausible value region."""
    for label_index, label_block in enumerate(blocks):
        text = str(label_block.get("text", ""))
        region_id = label_block.get("id")

        if not text or not region_id or not MRP_LABEL_PATTERN.search(text):
            continue

        nearby: list[tuple[float, int, dict[str, Any]]] = []

        for value_index, value_block in enumerate(blocks):
            if value_index == label_index:
                continue

            value_text = str(value_block.get("text", "")).strip()
            value_region_id = value_block.get("id")
            if not value_text or not value_region_id:
                continue

            distance = _spatial_distance(label_block, value_block)
            if distance is None or distance > 180:
                continue

            # A value region must be only a currency amount or a bare number.
            # This avoids pulling identifiers, quantities, or dates from a
            # neighboring OCR region merely because they are numeric.
            currency_match = CURRENCY_AMOUNT_PATTERN.fullmatch(value_text)
            bare_match = BARE_AMOUNT_PATTERN.fullmatch(value_text)

            if not currency_match and not bare_match:
                continue

            amount_text = (
                currency_match.group("amount")
                if currency_match
                else bare_match.group(0)
            )

            try:
                value = _parse_amount(amount_text)
            except ValueError:
                continue

            nearby.append(
                (
                    distance,
                    value_index,
                    _make_candidate(
                        value=value,
                        confidence=min(
                            _ocr_confidence(label_block),
                            _ocr_confidence(value_block),
                        ) * 0.90,
                        source_regions=[region_id, value_region_id],
                        strength="SPLIT_MRP",
                    ),
                )
            )

        if nearby:
            nearby.sort(key=lambda item: (item[0], item[1]))
            return nearby[0][2]

    return None


def extract_mrp(text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract MRP from OCR text blocks deterministically."""
    if not isinstance(text_blocks, list):
        raise TypeError("text_blocks must be a list")

    candidates: list[dict[str, Any]] = []

    for block in text_blocks:
        if not isinstance(block, dict):
            continue

        candidate = _find_same_block_candidate(block)
        if candidate:
            candidates.append(candidate)

    split_candidate = _find_split_mrp_candidate(text_blocks)
    if split_candidate:
        candidates.append(split_candidate)

    if not candidates:
        return {
            "value": None,
            "currency": "INR",
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        }

    values: dict[int | float, list[dict[str, Any]]] = {}
    for candidate in candidates:
        values.setdefault(candidate["value"], []).append(candidate)

    if len(values) > 1:
        source_regions: list[str] = []
        for candidate in candidates:
            source_regions.extend(candidate["source_regions"])
        return {
            "value": None,
            "currency": "INR",
            "confidence": round(
                max(candidate["confidence"] for candidate in candidates),
                4,
            ),
            "source_regions": list(dict.fromkeys(source_regions)),
            "extraction_status": "CONFLICTING",
        }

    same_value_candidates = next(iter(values.values()))
    best_candidate = max(
        same_value_candidates,
        key=lambda candidate: (
            1 if candidate["strength"] == "EXPLICIT_MRP" else 0,
            1 if candidate["strength"] == "SPLIT_MRP" else 0,
            candidate["confidence"],
        ),
    )

    region_order = {
        block["id"]: index
        for index, block in enumerate(text_blocks)
        if isinstance(block, dict) and block.get("id")
    }
    source_regions: list[str] = []
    for candidate in same_value_candidates:
        source_regions.extend(candidate["source_regions"])
    source_regions = list(dict.fromkeys(source_regions))
    source_regions.sort(
        key=lambda region_id: region_order.get(region_id, len(region_order))
    )

    if best_candidate["strength"] == "CURRENCY_ONLY":
        return {
            "value": best_candidate["value"],
            "currency": "INR",
            "confidence": best_candidate["confidence"],
            "source_regions": source_regions,
            "extraction_status": "UNCERTAIN",
        }

    return {
        "value": best_candidate["value"],
        "currency": "INR",
        "confidence": best_candidate["confidence"],
        "source_regions": source_regions,
        "extraction_status": "FOUND",
    }
