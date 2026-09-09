from __future__ import annotations

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def detect_document_quad(image: np.ndarray) -> np.ndarray | None:
    """Find a strong four-corner package/document contour, if present."""
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.20:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx.reshape(4, 2)))

    if not candidates:
        return None

    _, quad = max(candidates, key=lambda item: item[0])
    return _order_points(quad)


def correct_perspective(image: np.ndarray, quad: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Perspective-correct a detected quadrilateral when feasible."""
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    quad = detect_document_quad(image) if quad is None else _order_points(quad)
    if quad is None:
        return image.copy(), {"corrected": False, "reason": "no_reliable_quad"}

    tl, tr, br, bl = quad
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(1, int(round(max(width_a, width_b))))
    height = max(1, int(round(max(height_a, height_b))))

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    corrected = cv2.warpPerspective(image, matrix, (width, height))

    return corrected, {"corrected": True, "source_quad": quad.tolist(), "output_size": [width, height]}
