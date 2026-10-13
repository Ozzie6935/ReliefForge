"""Bin counts for a depth array. The array itself is not returned."""

from dataclasses import dataclass

import numpy as np

HISTOGRAM_BINS = 256


@dataclass(frozen=True)
class DepthHistogramSeries:
    bins: int
    counts: list[int]
    minimum: float
    maximum: float


def depth_histogram(
    values: np.ndarray,
    bins: int = HISTOGRAM_BINS,
    value_range: tuple[float, float] | None = None,
) -> DepthHistogramSeries:
    data = np.asarray(values)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("Histogram values must be finite.")
    if value_range is None:
        low = float(data.min())
        high = float(data.max())
        if high <= low:
            high = low + 1e-6
    else:
        low, high = value_range
        if high <= low:
            raise ValueError("Histogram range is empty.")
    counts, _edges = np.histogram(data, bins=bins, range=(low, high))
    listed = [int(count) for count in counts]
    if len(listed) != bins or sum(listed) != int(data.size):
        raise ValueError("Histogram does not account for every value.")
    return DepthHistogramSeries(bins=bins, counts=listed, minimum=low, maximum=high)
