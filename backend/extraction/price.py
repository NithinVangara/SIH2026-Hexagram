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


def _parse_amount(value: str) -> int | float:
    """Convert OCR numeric text into a JSON-friendly numeric value."""
    value = value.replace(",", "").strip()

    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid price value: {value}") from exc

    if amount < 0:
        raise ValueError("Price cannot be negative")

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

    # Currency amount without an explicit MRP label.
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
    """
    Handle OCR where the MRP label and amount are returned as separate
    text regions.

    The amount region must be either:
    - a currency amount such as "₹120" / "Rs. 120"
    - or a bare numeric amount such as "120"

    We deliberately reject numbers embedded in unrelated text such as
    "Net Weight 500 g".
    """
    for index, block in enumerate(blocks):
        text = str(block.get("text", ""))
        region_id = block.get("id")

        if not text or not region_id:
            continue

        if not MRP_LABEL_PATTERN.search(text):
            continue

        # Search only nearby OCR regions.
        for next_block in blocks[index + 1 : index + 4]:
            next_text = str(next_block.get("text", "")).strip()
            next_region_id = next_block.get("id")

            if not next_text or not next_region_id:
                continue

            # Preferred split form: currency + amount.
            currency_match = CURRENCY_AMOUNT_PATTERN.fullmatch(next_text)

            if currency_match:
                try:
                    value = _parse_amount(currency_match.group("amount"))
                except ValueError:
                    continue

                label_conf = _ocr_confidence(block)
                amount_conf = _ocr_confidence(next_block)

                return _make_candidate(
                    value=value,
                    confidence=min(label_conf, amount_conf) * 0.90,
                    source_regions=[region_id, next_region_id],
                    strength="SPLIT_MRP",
                )

            # Allow a completely bare numeric OCR region.
            bare_number_match = re.fullmatch(
                r"\d+(?:[.,]\d{1,2})?",
                next_text,
            )

            if bare_number_match:
                try:
                    value = _parse_amount(bare_number_match.group(0))
                except ValueError:
                    continue

                label_conf = _ocr_confidence(block)
                amount_conf = _ocr_confidence(next_block)

                return _make_candidate(
                    value=value,
                    confidence=min(label_conf, amount_conf) * 0.90,
                    source_regions=[region_id, next_region_id],
                    strength="SPLIT_MRP",
                )

    return None


def extract_mrp(text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract MRP from OCR text blocks deterministically.

    Returns:
        {
            "value": 120,
            "currency": "INR",
            "confidence": 0.97,
            "source_regions": ["R01"],
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

    candidates: list[dict[str, Any]] = []

    # First pass: candidates contained in individual OCR regions.
    for block in text_blocks:
        if not isinstance(block, dict):
            continue

        candidate = _find_same_block_candidate(block)

        if candidate:
            candidates.append(candidate)

    # Second pass: label and amount split across OCR regions.
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

    # Group candidates by numeric value.
    values = {}

    for candidate in candidates:
        values.setdefault(candidate["value"], []).append(candidate)

    # Different MRP candidates → conflict.
    if len(values) > 1:
        source_regions = []

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

    # All candidates agree on the same value.
    same_value_candidates = next(iter(values.values()))

    best_candidate = max(
        same_value_candidates,
        key=lambda candidate: candidate["confidence"],
    )

    source_regions = []

    for candidate in same_value_candidates:
        source_regions.extend(candidate["source_regions"])

# Preserve OCR/source order while removing duplicates.
    region_order = {
        block["id"]: index
        for index, block in enumerate(text_blocks)
        if isinstance(block, dict) and block.get("id")
    }

    source_regions = list(dict.fromkeys(source_regions))
    source_regions.sort(
        key=lambda region_id: region_order.get(region_id, len(region_order))
)

    # Currency-only evidence is intentionally not treated as a definite MRP.
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