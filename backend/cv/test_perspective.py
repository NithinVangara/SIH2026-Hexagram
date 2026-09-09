import cv2
import numpy as np
import pytest

from backend.cv.perspective import four_point_transform, order_points


def test_order_points_returns_consistent_order():
    points = [
        [300, 300],
        [50, 50],
        [300, 50],
        [50, 300],
    ]

    result = order_points(points)

    expected = np.array(
        [
            [50, 50],
            [300, 50],
            [300, 300],
            [50, 300],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected)


def test_perspective_transform_rectifies_rectangle():
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # Draw a simple white rectangle in the source image.
    cv2.rectangle(
        image,
        (50, 80),
        (250, 280),
        (255, 255, 255),
        -1,
    )

    points = [
        [50, 80],
        [250, 80],
        [250, 280],
        [50, 280],
    ]

    rectified, metadata = four_point_transform(image, points)

    assert rectified.shape[1] == 200
    assert rectified.shape[0] == 200
    assert metadata["geometry_status"] == "RECTIFIED"
    assert metadata["output_size"]["width"] == 200
    assert metadata["output_size"]["height"] == 200


def test_perspective_transform_preserves_image_content():
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    points = [
        [40, 50],
        [240, 30],
        [260, 250],
        [30, 270],
    ]

    rectified, metadata = four_point_transform(image, points)

    assert rectified.size > 0
    assert metadata["geometry_status"] == "RECTIFIED"
    assert len(metadata["transform_matrix"]) == 3
    assert all(len(row) == 3 for row in metadata["transform_matrix"])


def test_invalid_number_of_points_is_rejected():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        four_point_transform(
            image,
            [[0, 0], [99, 0], [99, 99]],
        )


def test_duplicate_points_are_rejected():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    points = [
        [0, 0],
        [99, 0],
        [99, 99],
        [99, 99],
    ]

    with pytest.raises(ValueError):
        four_point_transform(image, points)


def test_empty_image_is_rejected():
    image = np.empty((0, 0, 3), dtype=np.uint8)

    points = [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
    ]

    with pytest.raises(ValueError):
        four_point_transform(image, points)