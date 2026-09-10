from pathlib import Path
import sys

import cv2

# Allow imports such as backend.cv...
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.cv.quality import assess_image_quality
from backend.cv.font_measurement import estimate_character_height


IMAGE_DIR = Path(__file__).resolve().parent / "images"


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    if not image_paths:
        print(f"No PNG images found in: {IMAGE_DIR}")
        return

    print("=" * 100)
    print("M1 IMAGE QUALITY VALIDATION")
    print("=" * 100)
    print(f"Images found: {len(image_paths)}")
    print(f"Directory: {IMAGE_DIR}")
    print()

    results = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[ERROR] Could not load: {image_path.name}")
            continue

        height, width = image.shape[:2]

        try:
            quality = assess_image_quality(image)

            # Use the full image as a safe generic measurement fixture.
            # This is NOT treated as a text-region measurement.
            font = estimate_character_height(
                image,
                [0, 0, width, height]
            )

            result = {
                "image": image_path.name,
                "width": width,
                "height": height,
                "quality_status": quality.get("status"),
                "quality_score": quality.get("score"),
                "blur_score": quality.get("metrics", {}).get("blur_score"),
                "brightness": quality.get("metrics", {}).get("brightness"),
                "contrast": quality.get("metrics", {}).get("contrast"),
                "warnings": quality.get("warnings", []),
                "font_status": font.get("status"),
                "character_height_px": font.get("character_height_px"),
                "character_height_confidence": font.get(
                    "character_height_confidence"
                ),
                "components_used": font.get("components_used"),
            }

            results.append(result)

        except Exception as exc:
            print(f"[ERROR] {image_path.name}: {exc}")

    # ------------------------------------------------------------
    # Compact table
    # ------------------------------------------------------------

    print("\n" + "=" * 100)
    print("QUALITY RESULTS")
    print("=" * 100)

    header = (
        f"{'IMAGE':24}"
        f"{'SIZE':12}"
        f"{'STATUS':12}"
        f"{'SCORE':9}"
        f"{'BLUR':12}"
        f"{'BRIGHT':10}"
        f"{'CONTRAST':10}"
        f"{'WARNINGS'}"
    )

    print(header)
    print("-" * 100)

    for r in results:
        warnings = ",".join(r["warnings"]) if r["warnings"] else "-"

        print(
            f"{r['image'][:23]:24}"
            f"{r['width']}x{r['height']:<7}"
            f"{str(r['quality_status']):12}"
            f"{fmt(r['quality_score']):9}"
            f"{fmt(r['blur_score']):12}"
            f"{fmt(r['brightness']):10}"
            f"{fmt(r['contrast']):10}"
            f"{warnings}"
        )

    # ------------------------------------------------------------
    # Font measurement summary
    # ------------------------------------------------------------

    print("\n" + "=" * 100)
    print("CHARACTER-HEIGHT RESULTS")
    print("=" * 100)

    print(
        f"{'IMAGE':24}"
        f"{'STATUS':16}"
        f"{'HEIGHT PX':12}"
        f"{'CONF':10}"
        f"{'COMPONENTS'}"
    )

    print("-" * 75)

    for r in results:
        print(
            f"{r['image'][:23]:24}"
            f"{str(r['font_status']):16}"
            f"{fmt(r['character_height_px']):12}"
            f"{fmt(r['character_height_confidence']):10}"
            f"{fmt(r['components_used'])}"
        )

    # ------------------------------------------------------------
    # Important failure summary
    # ------------------------------------------------------------

    print("\n" + "=" * 100)
    print("POTENTIAL QUALITY-GATE FAILURES")
    print("=" * 100)

    accepted_with_warnings = [
        r for r in results
        if r["quality_status"] == "ACCEPTED" and r["warnings"]
    ]

    accepted_blurry = [
        r for r in results
        if r["quality_status"] == "ACCEPTED"
        and any("BLUR" in w.upper() for w in r["warnings"])
    ]

    if accepted_with_warnings:
        for r in accepted_with_warnings:
            print(
                f"- {r['image']}: ACCEPTED despite warnings "
                f"{r['warnings']}"
            )
    else:
        print("- No ACCEPTED images with warnings.")

    if accepted_blurry:
        for r in accepted_blurry:
            print(
                f"- {r['image']}: ACCEPTED but marked blurry "
                f"{r['warnings']}"
            )
    else:
        print("- No ACCEPTED+BLURRY cases.")

    print("\nDone.")


if __name__ == "__main__":
    main()