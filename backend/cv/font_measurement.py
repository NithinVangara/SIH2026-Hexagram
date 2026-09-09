from __future__ import annotations

import cv2
import numpy as np


def _validate_bbox(bbox: list | tuple) -> tuple[int, int, int, int]:
    """
    Validate and normalize a bounding box.

    Returns:
        (x1, y1, x2, y2)

    Raises:
        ValueError: if bbox is malformed or degenerate.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")

    try:
        x1, y1, x2, y2 = (int(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numeric") from exc

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")

    return x1, y1, x2, y2


def _prepare_foreground_mask(crop: np.ndarray) -> np.ndarray:
    """
    Build a foreground mask using Otsu thresholding.

    Both dark-on-light and light-on-dark possibilities are evaluated.
    The mask with lower foreground coverage is preferred.

    This remains an image-space feasibility baseline.
    It is not a semantic text detector.
    """
    if crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    _, dark_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    _, light_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size
    light_ratio = np.count_nonzero(light_mask) / light_mask.size

    if dark_ratio <= light_ratio:
        mask = dark_mask
        coverage = dark_ratio
    else:
        mask = light_mask
        coverage = light_ratio

    # If essentially the whole crop is foreground, thresholding
    # has not isolated useful text-like structure.
    if coverage >= 0.95:
        return np.zeros_like(mask)

    return mask


def _component_heights(
    mask: np.ndarray,
    min_area: int = 2,
) -> list[tuple[int, int, int, float, float]]:
    """
    Extract connected-component geometry.

    Each component is:

        (height, width, area, center_x, center_y)

    The geometry is retained so later stages can distinguish
    text-like components from obvious noise.
    """
    if mask.size == 0:
        return []

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        width = int(stats[label, cv2.CC_STAT_WIDTH])

        if area < min_area:
            continue

        if height <= 1 or width <= 0:
            continue

        components.append(
            (
                height,
                width,
                area,
                float(centroids[label][0]),
                float(centroids[label][1]),
            )
        )

    return components


def _has_text_like_row_support(
    components: list[tuple[int, int, int, float, float]],
) -> bool:
    """
    Check for repeated text-like component geometry.

    Text normally creates several components with compatible heights
    arranged approximately along one or more horizontal rows.

    This is only a conservative feasibility gate.
    It does not replace OCR.
    """
    if len(components) < 3:
        return False

    # Tiny structures are more likely to be compression artifacts,
    # texture, or isolated noise than useful character components.
    usable = [
        component
        for component in components
        if component[0] >= 3
    ]

    if len(usable) < 3:
        return False

    best_cluster_count = 0
    best_row_count = 0

    for component in usable:
        height = component[0]

        # Components whose heights are within ±1 px are considered
        # part of the same approximate character-height population.
        cluster = [
            other
            for other in usable
            if abs(other[0] - height) <= 1
        ]

        # Text characters in the same row should have nearby
        # vertical centroids.
        row_tolerance = max(2.0, 1.5 * height)

        row_count = max(
            sum(
                abs(other[4] - member[4]) <= row_tolerance
                for other in cluster
            )
            for member in cluster
        )

        if (len(cluster), row_count) > (
            best_cluster_count,
            best_row_count,
        ):
            best_cluster_count = len(cluster)
            best_row_count = row_count

    # A large repeated-height population is strong evidence.
    if best_cluster_count >= 10:
        return True

    # A smaller but clearly horizontal population is also useful.
    return best_row_count >= 3


def _select_text_like_components(
    components: list[tuple[int, int, int, float, float]],
) -> list[tuple[int, int, int, float, float]]:
    """
    Select the dominant text-like connected components.

    The goal is NOT to identify letters perfectly.

    Instead, this function removes components that are clearly
    inconsistent with the dominant character population.

    Strategy:
        1. Remove extremely tiny components.
        2. Find the strongest height population.
        3. Keep components close to that population.
        4. Reject obvious oversized components.
        5. Preserve small-print text by avoiding absolute
           minimum-height requirements.

    Returns:
        Filtered list of component geometries.
    """
    if not components:
        return []

    # ---------------------------------------------------------
    # Step 1: Remove extremely tiny components.
    # ---------------------------------------------------------
    #
    # A component with height 1–2 px is often caused by:
    # - JPEG/compression noise
    # - small texture fragments
    # - isolated background variations
    #
    # We intentionally keep this threshold relative to the
    # existing feasibility baseline. We are NOT saying that
    # 2 px text is legally invalid.
    usable = [
        component
        for component in components
        if component[0] >= 3
    ]

    if len(usable) < 3:
        return []

    # ---------------------------------------------------------
    # Step 2: Build height clusters.
    # ---------------------------------------------------------
    #
    # Example:
    #
    #   heights = [10, 10, 11, 10, 10, 30]
    #
    # The dominant population is around 10–11 px.
    #
    # The 30 px component is likely a large logo/background
    # structure rather than a normal character.
    #
    candidate_heights = sorted(
        component[0]
        for component in usable
    )

    best_cluster = []

    for height in candidate_heights:
        cluster = [
            component
            for component in usable
            if abs(component[0] - height) <= 1
        ]

        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < 3:
        return []

    # ---------------------------------------------------------
    # Step 3: Determine the dominant character height.
    # ---------------------------------------------------------
    dominant_height = float(
        np.median(
            [
                component[0]
                for component in best_cluster
            ]
        )
    )

    if dominant_height <= 0:
        return []

    # ---------------------------------------------------------
    # Step 4: Keep components near the dominant population.
    # ---------------------------------------------------------
    #
    # The tolerance is relative rather than an absolute pixel
    # threshold.
    #
    # This matters because:
    #
    #   5 px text
    #   10 px text
    #   20 px text
    #
    # should all remain possible.
    #
    # A ±40% tolerance gives room for:
    # - perspective
    # - anti-aliasing
    # - touching characters
    # - different glyph shapes
    #
    lower_height = dominant_height * 0.60
    upper_height = dominant_height * 1.40

    filtered = [
        component
        for component in usable
        if lower_height <= component[0] <= upper_height
    ]

    if len(filtered) < 3:
        return []

    # ---------------------------------------------------------
    # Step 5: Reject obvious oversized structures.
    # ---------------------------------------------------------
    #
    # A connected component that is many times wider than its
    # height can represent:
    #
    # - a line
    # - border
    # - table separator
    # - background structure
    # - packaging graphics
    #
    # We only reject extreme cases here so legitimate glyphs
    # and joined characters remain possible.
    #
    filtered_geometry = []

    for component in filtered:
        height, width, area, cx, cy = component

        aspect_ratio = width / max(height, 1)

        if aspect_ratio > 8.0:
            continue

        # Extremely large filled components are unlikely to be
        # individual character components.
        fill_ratio = area / max(width * height, 1)

        if (
            height > dominant_height * 2.5
            and fill_ratio > 0.70
        ):
            continue

        filtered_geometry.append(component)

    if len(filtered_geometry) < 3:
        return []

    return filtered_geometry


def estimate_character_height(
    image: np.ndarray,
    bbox: list | tuple,
) -> dict:
    """
    Estimate character/glyph height in image pixels.

    This is an image-space feasibility baseline.

    It does not claim:
        - typographic font-size accuracy
        - physical-unit accuracy
        - legal compliance

    Returns:
        {
            "status": "MEASURED" | "NO_FOREGROUND",
            "character_height_px": float | None,
            "character_height_confidence": float,
            "components_used": int,
        }
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    x1, y1, x2, y2 = _validate_bbox(bbox)

    image_height, image_width = image.shape[:2]

    # Clip bbox to image boundaries.
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox does not overlap the image")

    crop = image[y1:y2, x1:x2]

    mask = _prepare_foreground_mask(crop)

    components = _component_heights(mask)

    # ---------------------------------------------------------
    # Phase 3.9.1 gate
    # ---------------------------------------------------------
    if not _has_text_like_row_support(components):
        return {
            "status": "NO_FOREGROUND",
            "character_height_px": None,
            "character_height_confidence": 0.0,
            "components_used": 0,
        }

    # ---------------------------------------------------------
    # Phase 3.9.2 filtering
    # ---------------------------------------------------------
    text_components = _select_text_like_components(
        components
    )

    if len(text_components) < 3:
        return {
            "status": "NO_FOREGROUND",
            "character_height_px": None,
            "character_height_confidence": 0.0,
            "components_used": 0,
        }

    # ---------------------------------------------------------
    # Calculate character height from the filtered population.
    # ---------------------------------------------------------
    heights_array = np.asarray(
        [
            component[0]
            for component in text_components
        ],
        dtype=np.float32,
    )

    character_height = float(
        np.median(heights_array)
    )

    component_count = len(text_components)

    if character_height <= 0:
        confidence = 0.0
    else:
        # Median absolute deviation gives us a robust estimate
        # of how consistent the selected character population is.
        deviation = float(
            np.median(
                np.abs(
                    heights_array - character_height
                )
            )
        )

        consistency = max(
            0.0,
            1.0 - (
                deviation / character_height
            ),
        )

        sample_factor = min(
            1.0,
            component_count / 5.0,
        )

        confidence = (
            consistency
            * sample_factor
        )

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    return {
        "status": "MEASURED",
        "character_height_px": character_height,
        "character_height_confidence": confidence,
        "components_used": component_count,
    }