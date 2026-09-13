import re
from datetime import datetime

DATE_PATTERNS = [
    re.compile(r"\b(0?[1-9]|[12]\d|3[01])\s*[/\-.]\s*(0?[1-9]|1[0-2])\s*[/\-.]\s*(20\d{2}|\d{2})\b"),
    re.compile(r"\b(0?[1-9]|1[0-2])\s*[/\-.]\s*(20\d{2}|\d{2})\b"),
]
DATE_VALUE_PATTERN = re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])\s*[/\-.]\s*(?:0?[1-9]|1[0-2])\s*[/\-.]\s*(?:20\d{2}|\d{2})\b|\b(?:0?[1-9]|1[0-2])\s*[/\-.]\s*(?:20\d{2}|\d{2})\b")
LABEL_PATTERNS = {
    "manufacturing_date": re.compile(r"\b(?:MFG|MFD|Mfg\.?|Mfd\.?)\s*(?:DATE)?\s*[:\-]?\s*([0-9]{1,2}\s*[/\-.]\s*[0-9]{1,2}\s*[/\-.]\s*(?:20\d{2}|\d{2})|[0-9]{1,2}\s*[/\-.]\s*(?:20\d{2}|\d{2}))", re.IGNORECASE),
    "packing_date": re.compile(r"\b(?:PKD|PKT|PACKED\s*ON|PACKING\s*DATE)\s*[:\-]?\s*([0-9]{1,2}\s*[/\-.]\s*[0-9]{1,2}\s*[/\-.]\s*(?:20\d{2}|\d{2})|[0-9]{1,2}\s*[/\-.]\s*(?:20\d{2}|\d{2}))", re.IGNORECASE),
    "expiry_date": re.compile(r"\b(?:EXP|EXPIRY|EXPIRY\s*DATE|USE\s*BY|USE\s*BEFORE)\s*[:\-]?\s*([0-9]{1,2}\s*[/\-.]\s*[0-9]{1,2}\s*[/\-.]\s*(?:20\d{2}|\d{2})|[0-9]{1,2}\s*[/\-.]\s*(?:20\d{2}|\d{2}))", re.IGNORECASE),
}
COMBINED_DATE_LABEL_PATTERN = re.compile(r"\bMFD\.?\s*(?:&|AND)\s*USE\s*BY\s*DATE\b", re.IGNORECASE)
BEST_BEFORE_PATTERN = re.compile(r"\bBEST\s*BEFORE\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(MONTHS?|YRS?|YEARS?|DAYS?)\b", re.IGNORECASE)
GENERIC_DATE_LABELS = {
    "manufacturing_date": re.compile(r"\b(?:MFG|MFD|MANUFACTURED|MANUFACTURING)\b", re.IGNORECASE),
    "packing_date": re.compile(r"\b(?:PKD|PKT|PACKED\s*ON|PACKING\s*DATE)\b", re.IGNORECASE),
    "expiry_date": re.compile(r"\b(?:EXP|EXPIRY|USE\s*BY|USE\s*BEFORE)\b", re.IGNORECASE),
}


def _normalize_date(raw_date: str) -> str | None:
    value = re.sub(r"\s+", "", raw_date).replace(".", "/").replace("-", "/")
    parts = value.split("/")
    try:
        if len(parts) == 3:
            day, month, year = map(int, parts)
            if year < 100:
                year += 2000
            return datetime(year, month, day).strftime("%Y-%m-%d")
        if len(parts) == 2:
            month, year = map(int, parts)
            if not 1 <= month <= 12:
                return None
            if year < 100:
                year += 2000
            return f"{year:04d}-{month:02d}"
    except ValueError:
        return None
    return None


def _date_precision(raw_date: str) -> str:
    value = re.sub(r"\s+", "", raw_date)
    if len(value.split("/")) == 3:
        return "DAY_MONTH_YEAR"
    for separator in ["-", "."]:
        if separator in value and len(value.split(separator)) == 3:
            return "DAY_MONTH_YEAR"
    return "MONTH_YEAR"


def _extract_candidate(field: str, raw_date: str, confidence: float, source_regions: list[str], explicit: bool) -> dict | None:
    normalized = _normalize_date(raw_date)
    if normalized is None:
        return None
    return {"value": normalized, "precision": _date_precision(raw_date), "confidence": round(min(confidence * (1.0 if explicit else 0.75), 1.0), 3), "source_regions": source_regions, "extraction_status": "FOUND" if explicit else "UNCERTAIN"}


def _bbox(block: dict) -> tuple[float, float, float, float] | None:
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


def _spatial_distance(label_block: dict, value_block: dict) -> float | None:
    label_box = _bbox(label_block)
    value_box = _bbox(value_block)
    if label_box is None or value_box is None:
        return None
    lx1, ly1, lx2, ly2 = label_box
    vx1, vy1, vx2, vy2 = value_box
    dx = max(lx1 - vx2, vx1 - lx2, 0.0)
    dy = max(ly1 - vy2, vy1 - ly2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _date_candidates_from_text(text: str, confidence: float, region_id: str, explicit: bool) -> list[dict]:
    candidates = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = _extract_candidate("date", match.group(0), confidence, [region_id], explicit)
            if candidate:
                candidates.append(candidate)
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate["value"] not in seen:
            seen.add(candidate["value"])
            unique.append(candidate)
    return unique


def _find_spatial_date_candidates(blocks: list[dict], label_block: dict, max_distance: float = 180) -> list[dict]:
    nearby = []
    for block in blocks:
        if block is label_block:
            continue
        region_id = block.get("id")
        text = str(block.get("text", "")).strip()
        if not region_id or not text:
            continue
        distance = _spatial_distance(label_block, block)
        if distance is None or distance > max_distance:
            continue
        for candidate in _date_candidates_from_text(text, float(block.get("confidence", 0.0)), region_id, True):
            nearby.append((distance, candidate))
    nearby.sort(key=lambda item: item[0])
    return [candidate for _, candidate in nearby]


def _generic_spatial_date_candidate(blocks: list[dict], label_block: dict, field: str, excluded_regions: set[str] | None = None) -> dict | None:
    excluded_regions = excluded_regions or set()
    spatial_dates = [candidate for candidate in _find_spatial_date_candidates(blocks, label_block) if not set(candidate["source_regions"]) & excluded_regions]
    if not spatial_dates:
        return None
    candidate = spatial_dates[0]
    label_confidence = float(label_block.get("confidence", 0.0))
    value_confidence = float(candidate.get("confidence", 0.0))
    return {**candidate, "confidence": round(min(label_confidence, value_confidence) * 0.90, 3), "source_regions": [label_block.get("id")] + candidate["source_regions"], "extraction_status": "FOUND"}


def extract_dates(text_blocks: list[dict]) -> dict:
    results = {"manufacturing_date": None, "packing_date": None, "expiry_date": None, "best_before": None}
    candidates = {"manufacturing_date": [], "packing_date": [], "expiry_date": [], "best_before": []}

    for block in text_blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text", ""))
        confidence = float(block.get("confidence", 0.0))
        region_id = block.get("id")
        source_regions = [region_id] if region_id else []

        for field, pattern in LABEL_PATTERNS.items():
            match = pattern.search(text)
            if match:
                candidate = _extract_candidate(field, match.group(1), confidence, source_regions, True)
                if candidate:
                    candidates[field].append(candidate)

        if region_id and COMBINED_DATE_LABEL_PATTERN.search(text):
            spatial_dates = _find_spatial_date_candidates(text_blocks, block)
            if len(spatial_dates) >= 2:
                first, second = spatial_dates[0], spatial_dates[1]
                candidates["manufacturing_date"].append({**first, "confidence": round(min(confidence, first["confidence"]) * 0.90, 3), "source_regions": [region_id] + first["source_regions"], "extraction_status": "FOUND"})
                candidates["expiry_date"].append({**second, "confidence": round(min(confidence, second["confidence"]) * 0.90, 3), "source_regions": [region_id] + second["source_regions"], "extraction_status": "FOUND"})

        for field, label_pattern in GENERIC_DATE_LABELS.items():
            if not label_pattern.search(text):
                continue
            if field == "expiry_date" and COMBINED_DATE_LABEL_PATTERN.search(text):
                continue
            used_regions = set()
            for existing in candidates["manufacturing_date"] + candidates["packing_date"] + candidates["expiry_date"]:
                used_regions.update(existing.get("source_regions", []))
            spatial_candidate = _generic_spatial_date_candidate(text_blocks, block, field, used_regions)
            if spatial_candidate:
                candidates[field].append(spatial_candidate)

        best_before_match = BEST_BEFORE_PATTERN.search(text)
        if best_before_match:
            value = float(best_before_match.group(1))
            unit = best_before_match.group(2).lower()
            normalized_unit = "months" if unit.startswith("month") else "years" if unit.startswith(("yr", "year")) else "days"
            candidates["best_before"].append({"value": value, "unit": normalized_unit, "confidence": round(confidence, 3), "source_regions": source_regions, "extraction_status": "FOUND"})

    for field in ("manufacturing_date", "packing_date", "expiry_date"):
        field_candidates = candidates[field]
        if not field_candidates:
            results[field] = {"value": None, "precision": None, "confidence": 0.0, "source_regions": [], "extraction_status": "MISSING"}
            continue
        unique_values = {candidate["value"] for candidate in field_candidates}
        if len(unique_values) > 1:
            source_regions = []
            for candidate in field_candidates:
                source_regions.extend(candidate["source_regions"])
            results[field] = {"value": None, "precision": None, "confidence": max(candidate["confidence"] for candidate in field_candidates), "source_regions": list(dict.fromkeys(source_regions)), "extraction_status": "CONFLICTING"}
        else:
            results[field] = max(field_candidates, key=lambda candidate: candidate["confidence"])

    # A single OCR date region should not silently become both packing and
    # expiry evidence. Prefer the explicit expiry interpretation when the
    # same source region/value was associated with both labels.
    packing = results["packing_date"]
    expiry = results["expiry_date"]
    if packing.get("extraction_status") == "FOUND" and expiry.get("extraction_status") == "FOUND":
        if packing.get("value") == expiry.get("value") and set(packing.get("source_regions", [])) & set(expiry.get("source_regions", [])):
            results["packing_date"] = {"value": None, "precision": None, "confidence": 0.0, "source_regions": [], "extraction_status": "MISSING"}

    best_before_candidates = candidates["best_before"]
    if not best_before_candidates:
        results["best_before"] = {"value": None, "unit": None, "confidence": 0.0, "source_regions": [], "extraction_status": "MISSING"}
    else:
        unique_values = {(candidate["value"], candidate["unit"]) for candidate in best_before_candidates}
        if len(unique_values) > 1:
            source_regions = []
            for candidate in best_before_candidates:
                source_regions.extend(candidate["source_regions"])
            results["best_before"] = {"value": None, "unit": None, "confidence": max(candidate["confidence"] for candidate in best_before_candidates), "source_regions": list(dict.fromkeys(source_regions)), "extraction_status": "CONFLICTING"}
        else:
            results["best_before"] = max(best_before_candidates, key=lambda candidate: candidate["confidence"])
    return results
