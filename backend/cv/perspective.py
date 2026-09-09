from __future__ import annotations

import cv2
import numpy as np


def order_points(points: list | tuple) -> np.ndarray:
    """Return four points ordered as top-left, top-right, bottom-right, bottom-left."""
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError("Exactly four points are required")

    pts = np.asarray(points, dtype=np.float32)

    if pts.shape != (4, 2):
        raise ValueError("Points must have shape (4, 2)")

    # Reject duplicate points.
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("Points must be unique")

    sums = pts.sum(axis=1)
    differences = np.diff(pts, axis=1).ravel()

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(differences)]
    bottom_left = pts[np.argmax(differences)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def four_point_transform(
    image: np.ndarray,
    points: list | tuple,
) -> tuple[np.ndarray, dict]:
    """
    Rectify a quadrilateral region using a perspective transform.

    Returns:
        rectified_image, metadata
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D numpy array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    rect = order_points(points)
    tl, tr, br, bl = rect

    width_top = _distance(tr, tl)
    width_bottom = _distance(br, bl)
    height_right = _distance(br, tr)
    height_left = _distance(bl, tl)

    max_width = int(round(max(width_top, width_bottom)))
    max_height = int(round(max(height_left, height_right)))

    if max_width < 2 or max_height < 2:
        raise ValueError("Quadrilateral is too small")

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)

    rectified = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
    )

    metadata = {
        "geometry_status": "RECTIFIED",
        "source_points": rect.tolist(),
        "output_size": {
            "width": max_width,
            "height": max_height,
        },
        "transform_matrix": matrix.tolist(),
    }

    return rectified, metadata