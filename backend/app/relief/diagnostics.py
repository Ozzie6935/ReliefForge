"""Read-only statistics for a saved physical-height array.

The array is loaded from bytes already read by the caller. Nothing here
writes a project file or replaces non-finite samples.
"""

import io
from typing import Literal
from uuid import UUID

import numpy as np

from app.projects.errors import InvalidPhysicalHeightArtifact
from app.relief.physical import maximum_cut_depth_mm
from app.schemas.relief import (
    HeightPercentiles,
    PhysicalHeightDiagnostics,
    PhysicalHeightHistogram,
    PhysicalHeightMetrics,
    PhysicalHeightRecord,
    PhysicalHeightValidation,
    RecordedPhysicalMapping,
)

HISTOGRAM_BINS = 64
RECORD_MISSING_DETAIL = "Physical height has not been generated."
FILE_MISSING_DETAIL = "Physical height array was not found."
ArtifactStatus = Literal["missing", "unknown", "current", "stale"]
UNREADABLE_DETAIL = "Physical height array could not be read."
STRUCTURE_DETAIL = "Physical height array must be a non-empty two-dimensional numeric array."
SHAPE_DETAIL = "Physical height array dimensions do not match the saved record."
NO_FINITE_SAMPLES = "No finite height samples are available."
NON_FINITE_WARNING = "The array contains non-finite samples. Finite-sample metrics exclude them."
BELOW_ZERO_WARNING = "Some samples are below 0 mm."
ABOVE_STOCK_WARNING = "Some samples are above the recorded stock thickness."
DTYPE_WARNING = "The stored array dtype is not float32."

_PERCENTILE_LEVELS = (1, 5, 25, 50, 75, 95, 99)


def load_physical_array(data: bytes) -> np.ndarray:
    """Load one NumPy array. Pickled objects are rejected."""

    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise InvalidPhysicalHeightArtifact(UNREADABLE_DETAIL) from exc
    if not isinstance(array, np.ndarray):
        raise InvalidPhysicalHeightArtifact(STRUCTURE_DETAIL)
    return array


def diagnose_physical_height(
    project_id: UUID,
    status: ArtifactStatus,
    record: PhysicalHeightRecord,
    data: bytes,
) -> PhysicalHeightDiagnostics:
    array = load_physical_array(data)
    require_physical_array(array, record)
    finite_mask = np.isfinite(array)
    finite = np.asarray(array[finite_mask], dtype=np.float64)
    non_finite_count = int(array.size - finite.size)
    below_zero_count = int(np.count_nonzero(finite < 0.0))
    above_stock_count = int(np.count_nonzero(finite > float(record.stock_thickness_mm)))
    warnings: list[str] = []
    errors: list[str] = []
    if array.dtype != np.float32:
        warnings.append(DTYPE_WARNING)
    if non_finite_count:
        warnings.append(NON_FINITE_WARNING)
    if below_zero_count:
        warnings.append(BELOW_ZERO_WARNING)
    if above_stock_count:
        warnings.append(ABOVE_STOCK_WARNING)

    metrics: PhysicalHeightMetrics | None = None
    histogram: PhysicalHeightHistogram | None = None
    if finite.size == 0:
        errors.append(NO_FINITE_SAMPLES)
    else:
        metrics = _metrics(
            array,
            finite,
            non_finite_count,
            below_zero_count,
            above_stock_count,
        )
        histogram = _histogram(finite)

    return PhysicalHeightDiagnostics(
        project_id=project_id,
        status=status,
        width_px=int(array.shape[1]),
        height_px=int(array.shape[0]),
        metrics=metrics,
        histogram=histogram,
        validation=PhysicalHeightValidation(ok=not errors, errors=errors, warnings=warnings),
        mapping=RecordedPhysicalMapping(
            stock_thickness_mm=float(record.stock_thickness_mm),
            base_thickness_mm=float(record.base_thickness_mm),
            maximum_surface_height_mm=float(record.maximum_surface_height_mm),
            maximum_theoretical_cut_depth_mm=maximum_cut_depth_mm(
                record.stock_thickness_mm,
                record.base_thickness_mm,
            ),
        ),
    )


def require_physical_array(array: np.ndarray, record: PhysicalHeightRecord) -> None:
    numeric = np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    if array.ndim != 2 or array.size == 0 or not numeric:
        raise InvalidPhysicalHeightArtifact(STRUCTURE_DETAIL)
    if array.shape != (record.height_px, record.width_px):
        raise InvalidPhysicalHeightArtifact(SHAPE_DETAIL)


def _metrics(
    array: np.ndarray,
    finite: np.ndarray,
    non_finite_count: int,
    below_zero_count: int,
    above_stock_count: int,
) -> PhysicalHeightMetrics:
    lowest = float(finite.min())
    highest = float(finite.max())
    percentiles = np.percentile(finite, _PERCENTILE_LEVELS, method="linear")
    return PhysicalHeightMetrics(
        width_px=int(array.shape[1]),
        height_px=int(array.shape[0]),
        sample_count=int(array.size),
        minimum_mm=lowest,
        maximum_mm=highest,
        mean_mm=float(finite.mean()),
        median_mm=float(np.median(finite)),
        standard_deviation_mm=float(np.std(finite, ddof=0)),
        percentiles_mm=HeightPercentiles(
            p1=float(percentiles[0]),
            p5=float(percentiles[1]),
            p25=float(percentiles[2]),
            p50=float(percentiles[3]),
            p75=float(percentiles[4]),
            p95=float(percentiles[5]),
            p99=float(percentiles[6]),
        ),
        range_mm=highest - lowest,
        non_finite_count=non_finite_count,
        below_zero_count=below_zero_count,
        above_stock_count=above_stock_count,
    )


def _histogram(finite: np.ndarray) -> PhysicalHeightHistogram:
    """64 equal bins. A constant array keeps every edge on that height."""

    low = float(finite.min())
    high = float(finite.max())
    sample_count = int(finite.size)
    if low == high:
        counts = [0] * HISTOGRAM_BINS
        counts[0] = sample_count
        edges = [low] * (HISTOGRAM_BINS + 1)
    else:
        counted, edges_array = np.histogram(finite, bins=HISTOGRAM_BINS, range=(low, high))
        counts = [int(value) for value in counted]
        edges = [float(value) for value in edges_array]
    return PhysicalHeightHistogram(
        bins=HISTOGRAM_BINS,
        edges_mm=edges,
        counts=counts,
        sample_count=sample_count,
    )
