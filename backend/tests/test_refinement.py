"""Binary brush strokes. These tests do not load SAM or touch project files."""

import math

import numpy as np
import pytest

from app.segmentation.refinement import MaskStrokeError, Stroke, apply_stroke


def _stroke(mode: str, radius: float, points: list[tuple[float, float]]) -> Stroke:
    return Stroke(mode=mode, brush_radius_px=radius, points=tuple(points))


def test_add_brush_paints_255_and_erase_paints_0() -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    added = apply_stroke(mask, _stroke("add", 1.2, [(5.5, 5.5)]))
    assert added[5, 5] == 255
    assert mask[5, 5] == 0
    erased = apply_stroke(added, _stroke("erase", 1.2, [(5.5, 5.5)]))
    assert erased[5, 5] == 0
    assert added[5, 5] == 255


def test_continuous_line_has_no_gaps() -> None:
    mask = np.zeros((12, 40), dtype=np.uint8)
    painted = apply_stroke(mask, _stroke("add", 1.25, [(0.5, 5.5), (30.5, 5.5)]))
    assert np.all(painted[5, 0:32] == 255)
    assert painted[5, 32] == 0


def test_single_point_stroke_creates_a_disk() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    painted = apply_stroke(mask, _stroke("add", 2.0, [(5.5, 5.5)]))
    assert painted[5, 5] == 255
    assert painted[5, 7] == 255
    assert painted[5, 8] == 0
    assert painted[7, 7] == 0
    outside = math.hypot((8 + 0.5) - 5.5, (5 + 0.5) - 5.5)
    assert outside > 2.0


def test_stroke_clips_to_image_bounds() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    original = mask.copy()
    painted = apply_stroke(mask, _stroke("add", 2.0, [(-10.0, 3.5), (20.0, 3.5)]))
    assert painted.shape == mask.shape
    assert painted[3, 0] == 255
    assert painted[0, 0] == 0
    assert np.array_equal(mask, original)


def test_output_is_binary_uint8_and_shape_is_unchanged() -> None:
    mask = np.zeros((9, 7), dtype=np.uint8)
    mask[1, 1] = 255
    before = mask.copy()
    painted = apply_stroke(mask, _stroke("erase", 3.0, [(1.5, 1.5), (4.5, 4.5)]))
    assert painted.dtype == np.uint8
    assert painted.shape == (9, 7)
    assert set(np.unique(painted)).issubset({0, 255})
    assert np.array_equal(mask, before)
    assert painted is not mask


def test_result_is_deterministic() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    stroke = _stroke("add", 2.5, [(1.2, 14.0), (8.0, 8.0), (15.4, 2.2)])
    assert np.array_equal(apply_stroke(mask, stroke), apply_stroke(mask, stroke))


@pytest.mark.parametrize(
    "points",
    [
        [()],
        [(math.nan, 1.0)],
        [(1.0, math.inf)],
        [(1.0, -math.inf)],
    ],
)
def test_invalid_coordinates_are_rejected(points: list[tuple[float, float]]) -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    original = mask.copy()
    with pytest.raises(MaskStrokeError):
        apply_stroke(mask, Stroke("add", 1.0, tuple(points)))  # type: ignore[arg-type]
    assert np.array_equal(mask, original)


@pytest.mark.parametrize("radius", [0.0, -2.0, math.nan, math.inf])
def test_invalid_radius_is_rejected(radius: float) -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    with pytest.raises(MaskStrokeError):
        apply_stroke(mask, _stroke("add", radius, [(1.0, 1.0)]))


def test_radius_larger_than_the_image_is_rejected() -> None:
    mask = np.zeros((4, 6), dtype=np.uint8)
    with pytest.raises(MaskStrokeError):
        apply_stroke(mask, _stroke("add", 6.1, [(1.0, 1.0)]))


def test_empty_stroke_and_non_binary_mask_are_rejected() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(MaskStrokeError):
        apply_stroke(mask, Stroke("add", 1.0, ()))
    dirty = np.zeros((4, 4), dtype=np.uint8)
    dirty[0, 0] = 10
    with pytest.raises(MaskStrokeError):
        apply_stroke(dirty, _stroke("add", 1.0, [(1.0, 1.0)]))
    assert dirty[0, 0] == 10
