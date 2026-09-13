import re
from decimal import Decimal, InvalidOperation
from typing import Any

MRP_LABEL_PATTERN = re.compile(r"(?:\bM\s*\.?\s*R\s*\.?\s*P\s*\.?|\bMAXIMUM\s+RETAIL\s+PRICE\b)", re.IGNORECASE)
CURRENCY_AMOUNT_PATTERN = re.compile(r"(?:₹\s*|\bRS\.?\s*|\bINR\s*)(?P<amount>\d+(?:[.,]\d{1,2})?)", re.IGNORECASE)
MRP_AMOUNT_PATTERN = re.compile(r"(?:\bM\s*\.?\s*R\s*\.?\s*P\s*\.?|\bMAXIMUM\s+RETAIL\s+PRICE\b)[:\-\s₹Rs.INR]*(?P<amount>\d+(?:[.,]\d{1,2})?)", re.IGNORECASE)
BARE_AMOUNT_PATTERN = re.compile(r"\d+(?:[.,]\d{1,2})?")
MAX_MRP_AMOUNT = Decimal("1000000")
MAX_MRP_INTEGER_DIGITS = 5
MIN_MRP_AMOUNT = Decimal("1")


def _parse_amount(value: str) -> int | float:
    value = value.replace(",", "").strip()
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid price value: {value}") from exc
    if amount < MIN_MRP_AMOUNT or amount > MAX_MRP_AMOUNT:
        raise ValueError("Price candidate is implausible")
    integer_part = value.split(".", 1)[0]
    if len(integer_part.lstrip("0")) > MAX_MRP_INTEGER_DIGITS:
        raise ValueError("Price candidate has too many integer digits")
    return int(amount) if amount == amount.to_integral_value() else float(amount)


def _ocr_confidence(block: dict[str, Any]) -> float:
    try:
        confidence = float(block.get("confidence", 0.0))
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


def _same_line(label_box: tuple[float, float, float, float], value_box: tuple[float, float, float, float]) -> bool:
    overlap = max(0.0, min(label_box[3], value_box[3]) - max(label_box[1], value_box[1]))
    return overlap / max(1.0, min(label_box[3] - label_box[1], value_box[3] - value_box[1])) >= 0.35


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


def _make_candidate(value: int | float, confidence: float, source_regions: list[str], strength: str, currency: str = "INR") -> dict[str, Any]:
    return {"value": value, "currency": currency, "confidence": round(max(0.0, min(1.0, confidence)), 4), "source_regions": source_regions, "strength": strength}


def _find_same_block_candidate(block: dict[str, Any]) -> dict[str, Any] | None:
    text = str(block.get("text", ""))
    region_id = block.get("id")
    if not text or not region_id:
        return None
    ocr_conf = _ocr_confidence(block)
    match = MRP_AMOUNT_PATTERN.search(text)
    if match:
        try:
            value = _parse_amount(match.group("amount"))
        except ValueError:
            return None
        return _make_candidate(value, ocr_conf * 0.98, [region_id], "EXPLICIT_MRP")
    match = CURRENCY_AMOUNT_PATTERN.search(text)
    if match:
        try:
            value = _parse_amount(match.group("amount"))
        except ValueError:
            return None
        return _make_candidate(value, ocr_conf * 0.60, [region_id], "CURRENCY_ONLY")
    return None


def _find_split_mrp_candidate(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for label_index, label_block in enumerate(blocks):
        label_text = str(label_block.get("text", ""))
        label_id = label_block.get("id")
        label_box = _bbox(label_block)
        if not label_id or label_box is None or not MRP_LABEL_PATTERN.search(label_text):
            continue
        nearby: list[tuple[int, float, int, dict[str, Any]]] = []
        for value_index, value_block in enumerate(blocks):
            if value_index == label_index:
                continue
            value_text = str(value_block.get("text", "")).strip()
            value_id = value_block.get("id")
            value_box = _bbox(value_block)
            if not value_text or not value_id or value_box is None:
                continue
            distance = _spatial_distance(label_block, value_block)
            if distance is None or distance > 140:
                continue
            currency_match = CURRENCY_AMOUNT_PATTERN.fullmatch(value_text)
            bare_match = BARE_AMOUNT_PATTERN.fullmatch(value_text)
            if not currency_match and not bare_match:
                continue
            lx1, ly1, lx2, ly2 = label_box
            vx1, vy1, vx2, vy2 = value_box
            # A split MRP value must be on the same text line and immediately
            # to the right of the label. This is deliberately stricter than
            # generic nearest-neighbour matching because package images contain
            # many unrelated numeric identifiers.
            if not _same_line(label_box, value_box):
                continue
            if vx1 < lx2 - 8 or (vx1 - lx2) > 100:
                continue
            amount_text = currency_match.group("amount") if currency_match else bare_match.group(0)
            try:
                value = _parse_amount(amount_text)
            except ValueError:
                continue
            confidence = min(_ocr_confidence(label_block), _ocr_confidence(value_block)) * 0.90
            nearby.append((0 if currency_match else 1, distance, value_index, _make_candidate(value, confidence, [label_id, value_id], "SPLIT_MRP")))
        if nearby:
            nearby.sort(key=lambda item: (item[0], item[1], item[2]))
            return nearby[0][3]
    return None


def extract_mrp(text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(text_blocks, list):
        raise TypeError("text_blocks must be a list")
    candidates: list[dict[str, Any]] = []
    for block in text_blocks:
        if isinstance(block, dict):
            candidate = _find_same_block_candidate(block)
            if candidate:
                candidates.append(candidate)
    split_candidate = _find_split_mrp_candidate(text_blocks)
    if split_candidate:
        candidates.append(split_candidate)
    if not candidates:
        return {"value": None, "currency": "INR", "confidence": 0.0, "source_regions": [], "extraction_status": "MISSING"}
    values: dict[int | float, list[dict[str, Any]]] = {}
    for candidate in candidates:
        values.setdefault(candidate["value"], []).append(candidate)
    if len(values) > 1:
        source_regions: list[str] = []
        for candidate in candidates:
            source_regions.extend(candidate["source_regions"])
        return {"value": None, "currency": "INR", "confidence": round(max(candidate["confidence"] for candidate in candidates), 4), "source_regions": list(dict.fromkeys(source_regions)), "extraction_status": "CONFLICTING"}
    same_value_candidates = next(iter(values.values()))
    best_candidate = max(same_value_candidates, key=lambda candidate: (1 if candidate["strength"] == "EXPLICIT_MRP" else 0, 1 if candidate["strength"] == "SPLIT_MRP" else 0, candidate["confidence"]))
    region_order = {block["id"]: index for index, block in enumerate(text_blocks) if isinstance(block, dict) and block.get("id")}
    source_regions: list[str] = []
    for candidate in same_value_candidates:
        source_regions.extend(candidate["source_regions"])
    source_regions = list(dict.fromkeys(source_regions))
    source_regions.sort(key=lambda region_id: region_order.get(region_id, len(region_order)))
    status = "UNCERTAIN" if best_candidate["strength"] == "CURRENCY_ONLY" else "FOUND"
    return {"value": best_candidate["value"], "currency": "INR", "confidence": best_candidate["confidence"], "source_regions": source_regions, "extraction_status": status}
