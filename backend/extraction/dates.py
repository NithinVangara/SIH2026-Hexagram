import re
from datetime import datetime


DATE_PATTERNS = [
    re.compile(
        r"\b(0?[1-9]|[12]\d|3[01])\s*[/\-.]\s*(0?[1-9]|1[0-2])\s*[/\-.]\s*(20\d{2})\b"
    ),
    re.compile(
        r"\b(0?[1-9]|1[0-2])\s*[/\-.]\s*(20\d{2})\b"
    ),
]


LABEL_PATTERNS = {
    "manufacturing_date": re.compile(
        r"\b(?:MFG|MFD|Mfg\.?|Mfd\.?)\s*(?:DATE)?\s*[:\-]?\s*"
        r"([0-9]{1,2}\s*[/\-.]\s*[0-9]{1,2}\s*[/\-.]\s*20\d{2}|"
        r"[0-9]{1,2}\s*[/\-.]\s*20\d{2})",
        re.IGNORECASE,
    ),
    "packing_date": re.compile(
        r"\b(?:PKD|PKT|PACKED\s*ON|PACKING\s*DATE)\s*[:\-]?\s*"
        r"([0-9]{1,2}\s*[/\-.]\s*[0-9]{1,2}\s*[/\-.]\s*20\d{2}|"
        r"[0-9]{1,2}\s*[/\-.]\s*20\d{2})",
        re.IGNORECASE,
    ),
    "expiry_date": re.compile(
        r"\b(?:EXP|EXPIRY|EXPIRY\s*DATE)\s*[:\-]?\s*"
        r"([0-9]{1,2}\s*[/\-.]\s*[0-9]{1,2}\s*[/\-.]\s*20\d{2}|"
        r"[0-9]{1,2}\s*[/\-.]\s*20\d{2})",
        re.IGNORECASE,
    ),
}

COMBINED_DATE_LABEL_PATTERN = re.compile(
    r"\bMFD\.?\s*(?:&|AND)\s*USE\s*BY\s*DATE\b",
    re.IGNORECASE,
)

BEST_BEFORE_PATTERN = re.compile(
    r"\bBEST\s*BEFORE\s*[:\-]?\s*"
    r"(\d+(?:\.\d+)?)\s*(MONTHS?|YRS?|YEARS?|DAYS?)\b",
    re.IGNORECASE,
)


def _normalize_date(raw_date: str) -> str | None:
    value = re.sub(r"\s+", "", raw_date)
    value = value.replace(".", "/").replace("-", "/")
    parts = value.split("/")

    try:
        if len(parts) == 3:
            day, month, year = map(int, parts)
            parsed = datetime(year, month, day)
            return parsed.strftime("%Y-%m-%d")

        if len(parts) == 2:
            month, year = map(int, parts)

            if not 1 <= month <= 12:
                return None

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


def _extract_candidate(
    field: str,
    raw_date: str,
    confidence: float,
    source_regions: list[str],
    explicit: bool,
) -> dict | None:
    normalized = _normalize_date(raw_date)

    if normalized is None:
        return None

    pattern_strength = 1.0 if explicit else 0.75

    extraction_confidence = round(
        min(confidence * pattern_strength, 1.0),
        3,
    )

    return {
        "value": normalized,
        "precision": _date_precision(raw_date),
        "confidence": extraction_confidence,
        "source_regions": source_regions,
        "extraction_status": "FOUND" if explicit else "UNCERTAIN",
    }


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


def _date_candidates_from_text(
    text: str,
    confidence: float,
    region_id: str,
    explicit: bool,
) -> list[dict]:
    candidates = []

    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = _extract_candidate(
                field="date",
                raw_date=match.group(0),
                confidence=confidence,
                source_regions=[region_id],
                explicit=explicit,
            )

            if candidate:
                candidates.append(candidate)

    return candidates


def _find_spatial_date_candidates(
    blocks: list[dict],
    label_block: dict,
) -> list[dict]:
    """Find date values in OCR regions spatially associated with a label."""
    nearby = []

    for block in blocks:
        if block is label_block:
            continue

        region_id = block.get("id")
        text = str(block.get("text", "")).strip()

        if not region_id or not text:
            continue

        distance = _spatial_distance(label_block, block)
        if distance is None or distance > 180:
            continue

        date_candidates = _date_candidates_from_text(
            text=text,
            confidence=float(block.get("confidence", 0.0)),
            region_id=region_id,
            explicit=True,
        )

        if date_candidates:
            for candidate in date_candidates:
                nearby.append((distance, candidate))

    nearby.sort(key=lambda item: item[0])
    return [candidate for _, candidate in nearby]


def extract_dates(text_blocks: list[dict]) -> dict:
    """
    Extract manufacturing, packing, expiry and best-before information
    from OCR text blocks.

    The function does not make any legal/compliance decision.
    """

    results = {
        "manufacturing_date": None,
        "packing_date": None,
        "expiry_date": None,
        "best_before": None,
    }

    candidates = {
        "manufacturing_date": [],
        "packing_date": [],
        "expiry_date": [],
        "best_before": [],
    }

    for block in text_blocks:
        text = block.get("text", "")
        confidence = float(block.get("confidence", 0.0))
        region_id = block.get("id")

        source_regions = [region_id] if region_id else []

        # Explicit labelled dates within one OCR region.
        for field, pattern in LABEL_PATTERNS.items():
            match = pattern.search(text)

            if match:
                candidate = _extract_candidate(
                    field=field,
                    raw_date=match.group(1),
                    confidence=confidence,
                    source_regions=source_regions,
                    explicit=True,
                )

                if candidate:
                    candidates[field].append(candidate)

        # Labels and values may be separated into different OCR regions.
        if region_id and COMBINED_DATE_LABEL_PATTERN.search(text):
            spatial_dates = _find_spatial_date_candidates(text_blocks, block)

            # For the common "Mfd. & use by date" layout, the first date is
            # manufacturing and the second is the use-by/expiry date.
            if len(spatial_dates) == 2:
                first, second = spatial_dates

                candidates["manufacturing_date"].append(
                    {
                        **first,
                        "confidence": round(
                            min(confidence, first["confidence"]) * 0.90,
                            3,
                        ),
                        "source_regions": [region_id] + first["source_regions"],
                        "extraction_status": "FOUND",
                    }
                )
                candidates["expiry_date"].append(
                    {
                        **second,
                        "confidence": round(
                            min(confidence, second["confidence"]) * 0.90,
                            3,
                        ),
                        "source_regions": [region_id] + second["source_regions"],
                        "extraction_status": "FOUND",
                    }
                )

        # Best before duration.
        best_before_match = BEST_BEFORE_PATTERN.search(text)

        if best_before_match:
            value = float(best_before_match.group(1))
            unit = best_before_match.group(2).lower()

            if unit.startswith("month"):
                normalized_unit = "months"
            elif unit.startswith(("yr", "year")):
                normalized_unit = "years"
            else:
                normalized_unit = "days"

            candidates["best_before"].append(
                {
                    "value": value,
                    "unit": normalized_unit,
                    "confidence": round(confidence, 3),
                    "source_regions": source_regions,
                    "extraction_status": "FOUND",
                }
            )

    for field in (
        "manufacturing_date",
        "packing_date",
        "expiry_date",
    ):
        field_candidates = candidates[field]

        if not field_candidates:
            results[field] = {
                "value": None,
                "precision": None,
                "confidence": 0.0,
                "source_regions": [],
                "extraction_status": "MISSING",
            }
            continue

        unique_values = {candidate["value"] for candidate in field_candidates}

        if len(unique_values) > 1:
            source_regions = []

            for candidate in field_candidates:
                source_regions.extend(candidate["source_regions"])

            results[field] = {
                "value": None,
                "precision": None,
                "confidence": max(
                    candidate["confidence"]
                    for candidate in field_candidates
                ),
                "source_regions": list(dict.fromkeys(source_regions)),
                "extraction_status": "CONFLICTING",
            }
        else:
            best_candidate = max(
                field_candidates,
                key=lambda candidate: candidate["confidence"],
            )
            results[field] = best_candidate

    best_before_candidates = candidates["best_before"]

    if not best_before_candidates:
        results["best_before"] = {
            "value": None,
            "unit": None,
            "confidence": 0.0,
            "source_regions": [],
            "extraction_status": "MISSING",
        }
    else:
        unique_values = {
            (candidate["value"], candidate["unit"])
            for candidate in best_before_candidates
        }

        if len(unique_values) > 1:
            source_regions = []

            for candidate in best_before_candidates:
                source_regions.extend(candidate["source_regions"])

            results["best_before"] = {
                "value": None,
                "unit": None,
                "confidence": max(
                    candidate["confidence"]
                    for candidate in best_before_candidates
                ),
                "source_regions": list(dict.fromkeys(source_regions)),
                "extraction_status": "CONFLICTING",
            }
        else:
            results["best_before"] = max(
                best_before_candidates,
                key=lambda candidate: candidate["confidence"],
            )

    return results
