"""Read-only cross sections of a saved physical-height array.

Pixel centers map onto the artwork rectangle. Stock placement is not applied.
Nothing here writes a project file or replaces non-finite heights with zero.
"""

import math
from typing import Literal
from uuid import UUID

import numpy as np

from app.projects.errors import InvalidProfileRequest
from app.relief.diagnostics import DTYPE_WARNING
from app.schemas.relief import (
    PhysicalHeightProfile,
    ProfileLine,
    ProfileSample,
    ProfileSummary,
)

DEFAULT_SAMPLE_COUNT = 256
MAX_SAMPLE_COUNT = 4096
MIN_SAMPLE_COUNT = 2
ProfileMode = Literal["horizontal", "vertical", "line"]

COORDINATE_CONVENTION = (
    "Pixel centers map onto the artwork rectangle. "
    "x_px increases left to right and y_px increases top to bottom. "
    "x=0,y=0 is the upper-left pixel center and "
    "x=width_px-1,y=height_px-1 is the lower-right pixel center. "
    "For more than one pixel, x_mm = x_px * artwork_width_mm / (width_px - 1) "
    "and y_mm = y_px * artwork_height_mm / (height_px - 1). "
    "A one-pixel axis is placed at 0 mm so the scale does not divide by zero. "
    "Stock size and artwork center offsets are not applied. "
    "Future mesh and export code must use this convention unless it is deliberately versioned."
)
SLOPE_NOTE = (
    "Maximum slope is the largest absolute rise between adjacent valid samples "
    "divided by their physical separation. It is not the maximum slope of a "
    "continuous surface and does not establish tool feasibility or machining safety."
)
NON_FINITE_PROFILE_WARNING = (
    "Some samples are null because interpolation touched a non-finite height."
)
NO_FINITE_PROFILE = "No finite profile samples are available."

HORIZONTAL_REQUIRES_Y = "Horizontal profiles require a finite y_px."
VERTICAL_REQUIRES_X = "Vertical profiles require a finite x_px."
LINE_REQUIRES_ENDS = (
    "Line profiles require finite start_x_px, start_y_px, end_x_px, and end_y_px."
)
OUT_OF_BOUNDS = "Profile coordinates are outside the image."
NON_FINITE_COORDINATES = "Profile coordinates must be finite."
ZERO_LENGTH = "The profile line has zero length."
SAMPLE_COUNT_DETAIL = "sample_count must be an integer from 2 to 4096."

_SNAP = 1e-9


def build_physical_profile(
    *,
    project_id: UUID,
    status: Literal["missing", "unknown", "current", "stale"],
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    mode: ProfileMode,
    y_px: float | None = None,
    x_px: float | None = None,
    start_x_px: float | None = None,
    start_y_px: float | None = None,
    end_x_px: float | None = None,
    end_y_px: float | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> PhysicalHeightProfile:
    count = _sample_count(sample_count)
    height_px, width_px = int(array.shape[0]), int(array.shape[1])
    start_x, start_y, end_x, end_y = _endpoints(
        mode,
        width_px,
        height_px,
        y_px,
        x_px,
        start_x_px,
        start_y_px,
        end_x_px,
        end_y_px,
    )
    length = _length_mm(
        start_x,
        start_y,
        end_x,
        end_y,
        width_px,
        height_px,
        artwork_width_mm,
        artwork_height_mm,
    )
    if length <= _SNAP:
        raise InvalidProfileRequest(ZERO_LENGTH)

    samples = [
        _sample(
            array,
            x,
            y,
            distance,
            width_px,
            height_px,
        )
        for x, y, distance in _stations(
            start_x,
            start_y,
            end_x,
            end_y,
            length,
            count,
        )
    ]
    warnings: list[str] = []
    if array.dtype != np.float32:
        warnings.append(DTYPE_WARNING)
    summary = _summary(length, samples)
    if summary.invalid_count:
        warnings.append(NON_FINITE_PROFILE_WARNING)
    if summary.valid_count == 0:
        warnings.append(NO_FINITE_PROFILE)
    return PhysicalHeightProfile(
        project_id=project_id,
        status=status,
        mode=mode,
        coordinate_convention=COORDINATE_CONVENTION,
        slope_note=SLOPE_NOTE,
        units="mm",
        artwork_width_mm=float(artwork_width_mm),
        artwork_height_mm=float(artwork_height_mm),
        width_px=width_px,
        height_px=height_px,
        line=ProfileLine(
            start_x_px=start_x,
            start_y_px=start_y,
            end_x_px=end_x,
            end_y_px=end_y,
        ),
        summary=summary,
        samples=samples,
        warnings=warnings,
    )


def pixel_to_artwork_mm(pixel: float, pixel_count: int, extent_mm: float) -> float:
    """Map a pixel-center coordinate onto the artwork edge span."""

    if pixel_count <= 1:
        return 0.0
    return float(pixel) * float(extent_mm) / float(pixel_count - 1)


def _sample_count(sample_count: int) -> int:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise InvalidProfileRequest(SAMPLE_COUNT_DETAIL)
    if sample_count < MIN_SAMPLE_COUNT or sample_count > MAX_SAMPLE_COUNT:
        raise InvalidProfileRequest(SAMPLE_COUNT_DETAIL)
    return sample_count


def _endpoints(
    mode: ProfileMode,
    width_px: int,
    height_px: int,
    y_px: float | None,
    x_px: float | None,
    start_x_px: float | None,
    start_y_px: float | None,
    end_x_px: float | None,
    end_y_px: float | None,
) -> tuple[float, float, float, float]:
    if mode == "horizontal":
        if y_px is None:
            raise InvalidProfileRequest(HORIZONTAL_REQUIRES_Y)
        _require_finite(y_px)
        _require_inside(y_px, height_px)
        return 0.0, float(y_px), float(width_px - 1), float(y_px)
    if mode == "vertical":
        if x_px is None:
            raise InvalidProfileRequest(VERTICAL_REQUIRES_X)
        _require_finite(x_px)
        _require_inside(x_px, width_px)
        return float(x_px), 0.0, float(x_px), float(height_px - 1)
    if start_x_px is None or start_y_px is None or end_x_px is None or end_y_px is None:
        raise InvalidProfileRequest(LINE_REQUIRES_ENDS)
    for value in (start_x_px, start_y_px, end_x_px, end_y_px):
        _require_finite(value)
    _require_inside(start_x_px, width_px)
    _require_inside(end_x_px, width_px)
    _require_inside(start_y_px, height_px)
    _require_inside(end_y_px, height_px)
    return float(start_x_px), float(start_y_px), float(end_x_px), float(end_y_px)


def _require_finite(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidProfileRequest(NON_FINITE_COORDINATES)


def _require_inside(value: float, pixel_count: int) -> None:
    if value < 0.0 or value > float(pixel_count - 1):
        raise InvalidProfileRequest(OUT_OF_BOUNDS)


def _length_mm(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width_px: int,
    height_px: int,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> float:
    dx = pixel_to_artwork_mm(end_x, width_px, artwork_width_mm) - pixel_to_artwork_mm(
        start_x, width_px, artwork_width_mm
    )
    dy = pixel_to_artwork_mm(end_y, height_px, artwork_height_mm) - pixel_to_artwork_mm(
        start_y, height_px, artwork_height_mm
    )
    return float(math.hypot(dx, dy))


def _stations(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    length: float,
    count: int,
) -> list[tuple[float, float, float]]:
    stations: list[tuple[float, float, float]] = []
    for index in range(count):
        if index == 0:
            stations.append((start_x, start_y, 0.0))
            continue
        if index == count - 1:
            stations.append((end_x, end_y, length))
            continue
        fraction = index / (count - 1)
        stations.append(
            (
                start_x + fraction * (end_x - start_x),
                start_y + fraction * (end_y - start_y),
                fraction * length,
            )
        )
    return stations


def _sample(
    array: np.ndarray,
    x_px: float,
    y_px: float,
    distance_mm: float,
    width_px: int,
    height_px: int,
) -> ProfileSample:
    x_px = _clamp_axis(x_px, width_px)
    y_px = _clamp_axis(y_px, height_px)
    return ProfileSample(
        x_px=x_px,
        y_px=y_px,
        distance_mm=distance_mm,
        height_mm=_bilinear(array, x_px, y_px),
    )


def _clamp_axis(value: float, pixel_count: int) -> float:
    limit = float(pixel_count - 1)
    if value < 0.0 and value > -_SNAP:
        return 0.0
    if value > limit and value < limit + _SNAP:
        return limit
    return value


def _bilinear(array: np.ndarray, x_px: float, y_px: float) -> float | None:
    """Read one cell at a pixel center. Otherwise blend only finite neighbors."""

    x_px = _snap(x_px)
    y_px = _snap(y_px)
    height_px, width_px = array.shape
    x0 = int(math.floor(x_px))
    y0 = int(math.floor(y_px))
    if x0 >= width_px - 1:
        x0 = width_px - 1
        x1 = x0
        tx = 0.0
    else:
        x1 = x0 + 1
        tx = x_px - x0
    if y0 >= height_px - 1:
        y0 = height_px - 1
        y1 = y0
        ty = 0.0
    else:
        y1 = y0 + 1
        ty = y_px - y0
    if tx <= _SNAP and ty <= _SNAP:
        value = array[y0, x0]
        return None if not np.isfinite(value) else float(value)
    corners = (
        (array[y0, x0], (1.0 - tx) * (1.0 - ty)),
        (array[y0, x1], tx * (1.0 - ty)),
        (array[y1, x0], (1.0 - tx) * ty),
        (array[y1, x1], tx * ty),
    )
    height = 0.0
    for value, weight in corners:
        if weight <= _SNAP:
            continue
        if not np.isfinite(value):
            return None
        height += float(value) * weight
    return height


def _snap(value: float) -> float:
    nearest = round(value)
    if abs(value - nearest) <= _SNAP:
        return float(nearest)
    return value


def _summary(length: float, samples: list[ProfileSample]) -> ProfileSummary:
    finite = [sample.height_mm for sample in samples if sample.height_mm is not None]
    valid_count = len(finite)
    maximum_delta: float | None = None
    maximum_slope: float | None = None
    for left, right in zip(samples, samples[1:]):
        if left.height_mm is None or right.height_mm is None:
            continue
        separation = right.distance_mm - left.distance_mm
        delta = abs(right.height_mm - left.height_mm)
        maximum_delta = delta if maximum_delta is None else max(maximum_delta, delta)
        if separation > _SNAP:
            slope = delta / separation
            maximum_slope = slope if maximum_slope is None else max(maximum_slope, slope)
    if valid_count == 0:
        return ProfileSummary(
            length_mm=length,
            sample_count=len(samples),
            valid_count=0,
            invalid_count=len(samples),
        )
    lowest = min(finite)
    highest = max(finite)
    return ProfileSummary(
        length_mm=length,
        sample_count=len(samples),
        valid_count=valid_count,
        invalid_count=len(samples) - valid_count,
        minimum_mm=lowest,
        maximum_mm=highest,
        range_mm=highest - lowest,
        mean_mm=sum(finite) / valid_count,
        maximum_adjacent_delta_mm=maximum_delta,
        maximum_abs_slope_mm_per_mm=maximum_slope,
    )
