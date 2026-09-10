from pathlib import Path
import csv
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.cv.quality import assess_image_quality


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
GROUND_TRUTH = BASE_DIR / "ground_truth.csv"


def normalize_status(status):
    mapping = {
        "ACCEPTED": "GOOD",
        "REVIEW": "REVIEW",
        "REJECTED": "REJECT",
    }
    return mapping.get(status, status)


def load_ground_truth():
    data = {}

    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data[row["image"]] = row["ocr_usability"]

    return data


def main():
    ground_truth = load_ground_truth()

    print("=" * 110)
    print("M1 QUALITY GATE — GROUND TRUTH COMPARISON")
    print("=" * 110)
    print()

    results = []

    for image_name, expected in ground_truth.items():

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print(f"[MISSING] {image_name}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[ERROR] Could not load {image_name}")
            continue

        quality = assess_image_quality(image)

        machine = normalize_status(quality["status"])

        results.append({
            "image": image_name,
            "expected": expected,
            "machine": machine,
            "score": quality["score"],
            "blur": quality["metrics"]["blur_score"],
            "brightness": quality["metrics"]["brightness"],
            "contrast": quality["metrics"]["contrast"],
            "warnings": quality["warnings"],
        })

    # ------------------------------------------------------------
    # Detailed comparison
    # ------------------------------------------------------------

    print(
        f"{'IMAGE':25}"
        f"{'GROUND TRUTH':15}"
        f"{'MACHINE':12}"
        f"{'RESULT':12}"
        f"{'SCORE':9}"
        f"{'BLUR':12}"
        f"{'WARNINGS'}"
    )

    print("-" * 110)

    for r in results:

        if r["expected"] == r["machine"]:
            result = "CORRECT"
        else:
            result = "MISMATCH"

        warnings = ",".join(r["warnings"]) if r["warnings"] else "-"

        print(
            f"{r['image'][:24]:25}"
            f"{r['expected']:15}"
            f"{r['machine']:12}"
            f"{result:12}"
            f"{r['score']:.3f}    "
            f"{r['blur']:.1f}       "
            f"{warnings}"
        )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    labels = ["GOOD", "REVIEW", "REJECT"]

    matrix = {
        expected: {machine: 0 for machine in labels}
        for expected in labels
    }

    for r in results:
        if r["expected"] in labels and r["machine"] in labels:
            matrix[r["expected"]][r["machine"]] += 1

    print("\n" + "=" * 110)
    print("CONFUSION MATRIX")
    print("=" * 110)

    print(
        f"{'GROUND TRUTH':15}"
        f"{'MACHINE GOOD':15}"
        f"{'MACHINE REVIEW':17}"
        f"{'MACHINE REJECT':17}"
    )

    print("-" * 65)

    for expected in labels:
        print(
            f"{expected:15}"
            f"{matrix[expected]['GOOD']:<15}"
            f"{matrix[expected]['REVIEW']:<17}"
            f"{matrix[expected]['REJECT']:<17}"
        )

    # ------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------

    correct = sum(
        1
        for r in results
        if r["expected"] == r["machine"]
    )

    total = len(results)

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    print(f"Total images: {total}")
    print(f"Correct classifications: {correct}")
    print(f"Mismatches: {total - correct}")

    if total:
        print(f"Exact classification accuracy: {correct / total:.2%}")

    # ------------------------------------------------------------
    # Safety-focused analysis
    # ------------------------------------------------------------

    false_accepts = [
        r for r in results
        if r["expected"] in ("REVIEW", "REJECT")
        and r["machine"] == "GOOD"
    ]

    false_rejects = [
        r for r in results
        if r["expected"] == "GOOD"
        and r["machine"] in ("REVIEW", "REJECT")
    ]

    print("\n" + "=" * 110)
    print("SAFETY ANALYSIS")
    print("=" * 110)

    print(f"False ACCEPTED: {len(false_accepts)}")
    for r in false_accepts:
        print(
            f"  - {r['image']}: "
            f"ground truth={r['expected']}, "
            f"machine={r['machine']}, "
            f"score={r['score']:.3f}, "
            f"blur={r['blur']:.1f}"
        )

    print(f"\nFalse REJECT/REVIEW of GOOD images: {len(false_rejects)}")
    for r in false_rejects:
        print(
            f"  - {r['image']}: "
            f"ground truth={r['expected']}, "
            f"machine={r['machine']}, "
            f"score={r['score']:.3f}"
        )


if __name__ == "__main__":
    main()