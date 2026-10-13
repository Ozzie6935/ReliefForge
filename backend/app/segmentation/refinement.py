"""Binary mask strokes.

This module paints 0 and 255 only. It does not know about HTTP, project
files, or the segmentation model.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

StrokeMode = str


class MaskStrokeError(ValueError):
    """The stroke cannot be painted onto a binary mask."""


@dataclass(frozen=True)
class Stroke:
    mode: StrokeMode
    brush_radius_px: float
    points: tuple[tuple[float, float], ...]


def apply_stroke(mask: np.ndarray, stroke: Stroke) -> np.ndarray:
    """Return a new mask with one continuous brush stroke applied.

    ``mask`` is not modified. Covered pixels become 255 for ``add`` and 0
    for ``erase``. The brush is a disk of ``brush_radius_px`` around the
    polyline, so fast pointer movement does not leave gaps.
    """

    _require_binary_mask(mask)
    height, width = mask.shape
    points = _validated_points(stroke, width, height)
    result = mask.copy()
    covered = np.zeros(mask.shape, dtype=bool)
    radius = float(stroke.brush_radius_px)
    for start, end in _segments(points):
        _cover_segment(covered, start, end, radius)
    if stroke.mode == "add":
        result[covered] = np.uint8(255)
    else:
        result[covered] = np.uint8(0)
    return result


def _require_binary_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise MaskStrokeError("Mask must be a uint8 image.")
    if mask.shape[0] < 1 or mask.shape[1] < 1:
        raise MaskStrokeError("Mask must be a uint8 image.")
    if not np.isin(mask, (0, 255)).all():
        raise MaskStrokeError("Mask must contain only 0 and 255.")


def _validated_points(stroke: Stroke, width: int, height: int) -> tuple[tuple[float, float], ...]:
    if stroke.mode not in ("add", "erase"):
        raise MaskStrokeError("Stroke mode must be add or erase.")
    radius = stroke.brush_radius_px
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not isfinite(float(radius)):
        raise MaskStrokeError("Brush radius must be a finite number.")
    if float(radius) <= 0:
        raise MaskStrokeError("Brush radius must be greater than zero.")
    if float(radius) > max(width, height):
        raise MaskStrokeError("Brush radius is larger than the image.")
    if len(stroke.points) < 1:
        raise MaskStrokeError("Stroke needs at least one point.")
    clamped: list[tuple[float, float]] = []
    max_x = width - 1e-6
    max_y = height - 1e-6
    for point in stroke.points:
        if len(point) != 2:
            raise MaskStrokeError("Stroke points must be coordinates.")
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise MaskStrokeError("Stroke coordinates must be finite.")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise MaskStrokeError("Stroke coordinates must be finite.")
        if not isfinite(float(x)) or not isfinite(float(y)):
            raise MaskStrokeError("Stroke coordinates must be finite.")
        clamped.append((min(max(float(x), 0.0), max_x), min(max(float(y), 0.0), max_y)))
    return tuple(clamped)


def _segments(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    if len(points) == 1:
        return ((points[0], points[0]),)
    return tuple(zip(points, points[1:], strict=False))


def _cover_segment(
    covered: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
) -> None:
    height, width = covered.shape
    pad = radius + 1.0
    x0 = max(0, int(np.floor(min(start[0], end[0]) - pad)))
    x1 = min(width, int(np.ceil(max(start[0], end[0]) + pad)))
    y0 = max(0, int(np.floor(min(start[1], end[1]) - pad)))
    y1 = min(height, int(np.ceil(max(start[1], end[1]) + pad)))
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance = _distance_to_segment(grid_x, grid_y, start[0], start[1], end[0], end[1])
    covered[y0:y1, x0:x1] |= distance <= radius + 1e-6


def _distance_to_segment(
    xs: np.ndarray,
    ys: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq == 0.0:
        return np.hypot(xs - ax, ys - ay)
    t = ((xs - ax) * abx + (ys - ay) * aby) / length_sq
    np.clip(t, 0.0, 1.0, out=t)
    return np.hypot(xs - (ax + t * abx), ys - (ay + t * aby))
