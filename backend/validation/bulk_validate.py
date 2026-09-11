from __future__ import annotations

import csv
from pathlib import Path

from backend.inspection import inspect_image


IMAGE_DIR = Path("backend/validation/images")
OUTPUT_FILE = Path("backend/validation/bulk_validation_results.csv")


FIELDNAMES = [
    "image",
    "image_id",
    "quality_status",
    "quality_score",
    "warnings",
    "ocr_blocks",
    "mrp_value",
    "mrp_status",
    "mrp_confidence",
    "net_quantity_value",
    "net_quantity_unit",
    "net_quantity_status",
    "net_quantity_confidence",
    "manufacturing_date",
    "manufacturing_date_status",
    "manufacturing_date_confidence",
    "packing_date",
    "packing_date_status",
    "packing_date_confidence",
    "expiry_date",
    "expiry_date_status",
    "expiry_date_confidence",
    "best_before",
    "best_before_status",
    "best_before_confidence",
]


def _value(declarations: dict, key: str, field: str):
    return declarations.get(key, {}).get(field)


def main() -> None:
    if not IMAGE_DIR.exists():
        raise SystemExit(f"Validation image directory not found: {IMAGE_DIR}")

    images = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and not p.name.startswith(".")
    )

    if not images:
        raise SystemExit(f"No validation images found in {IMAGE_DIR}")

    rows = []
    print("=" * 80)
    print("M1 BULK VALIDATION")
    print("=" * 80)
    print(f"Images found: {len(images)}")
    print()

    for index, image_path in enumerate(images, start=1):
        image_id = image_path.stem.upper()
        print(f"[{index:02d}/{len(images):02d}] {image_path.name}")

        try:
            result = inspect_image(image_path, image_id)
            quality = result.get("quality", {})
            declarations = result.get("declarations", {})
            text_blocks = result.get("text_blocks", [])

            row = {
                "image": image_path.name,
                "image_id": result.get("image_id"),
                "quality_status": quality.get("status"),
                "quality_score": quality.get("score"),
                "warnings": "|".join(quality.get("warnings", [])),
                "ocr_blocks": len(text_blocks),
                "mrp_value": _value(declarations, "mrp", "value"),
                "mrp_status": _value(declarations, "mrp", "extraction_status"),
                "mrp_confidence": _value(declarations, "mrp", "confidence"),
                "net_quantity_value": _value(declarations, "net_quantity", "value"),
                "net_quantity_unit": _value(declarations, "net_quantity", "unit"),
                "net_quantity_status": _value(declarations, "net_quantity", "extraction_status"),
                "net_quantity_confidence": _value(declarations, "net_quantity", "confidence"),
                "manufacturing_date": _value(declarations, "manufacturing_date", "value"),
                "manufacturing_date_status": _value(declarations, "manufacturing_date", "extraction_status"),
                "manufacturing_date_confidence": _value(declarations, "manufacturing_date", "confidence"),
                "packing_date": _value(declarations, "packing_date", "value"),
                "packing_date_status": _value(declarations, "packing_date", "extraction_status"),
                "packing_date_confidence": _value(declarations, "packing_date", "confidence"),
                "expiry_date": _value(declarations, "expiry_date", "value"),
                "expiry_date_status": _value(declarations, "expiry_date", "extraction_status"),
                "expiry_date_confidence": _value(declarations, "expiry_date", "confidence"),
                "best_before": _value(declarations, "best_before", "value"),
                "best_before_status": _value(declarations, "best_before", "extraction_status"),
                "best_before_confidence": _value(declarations, "best_before", "confidence"),
            }
            rows.append(row)

            print(f"       Quality : {row['quality_status']} ({row['quality_score']:.3f})")
            if row["warnings"]:
                print(f"       Warnings: {row['warnings']}")
            print(f"       OCR     : {row['ocr_blocks']} blocks")
            print(f"       MRP     : {row['mrp_value']} [{row['mrp_status']}]")
            print(f"       Quantity: {row['net_quantity_value']} {row['net_quantity_unit']} [{row['net_quantity_status']}]")
            print(f"       MFD     : {row['manufacturing_date']} [{row['manufacturing_date_status']}]")
            print(f"       Packing : {row['packing_date']} [{row['packing_date_status']}]")
            print(f"       Expiry  : {row['expiry_date']} [{row['expiry_date_status']}]")
            print(f"       Best    : {row['best_before']} [{row['best_before_status']}]")

        except Exception as exc:
            print(f"       ERROR: {type(exc).__name__}: {exc}")
            rows.append({
                "image": image_path.name,
                "image_id": image_id,
                "quality_status": "ERROR",
                "quality_score": None,
                "warnings": f"{type(exc).__name__}: {exc}",
                "ocr_blocks": 0,
            })
        print()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("quality_status") or "ERROR"
        counts[status] = counts.get(status, 0) + 1

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for status, count in sorted(counts.items()):
        print(f"{status:10s}: {count}")
    print()
    print(f"Results written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
