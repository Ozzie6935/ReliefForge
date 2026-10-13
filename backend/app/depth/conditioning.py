"""Reversible conditioning of one raw depth array.

Every call starts from the array it is given. It does not read files and it
does not depend on a previous conditioned result.

Depth Anything V2 Small relative depth is disparity: a larger value is nearer.
The official app describes that output as disparity, and Angie's raw map is
low on the rear wall and high in the foreground. Orientation therefore keeps
that order. After this function, 0 is far and 1 is near unless ``invert`` is set.
"""

import numpy as np

from app.schemas.conditioning import DepthConditioningParameters

# Larger raw values are nearer. Verified from the official disparity output
# and from Angie Dance's raw_depth.npy.
LARGER_RAW_VALUE_IS_NEAR = True

_RANGE_EPSILON = np.float32(1e-8)


class ConditioningError(Exception):
    """The raw array cannot be conditioned."""


def condition_depth(
    raw_depth: np.ndarray,
    parameters: DepthConditioningParameters,
) -> np.ndarray:
    """Return a new float32 array in ``[0, 1]``. ``raw_depth`` is not modified."""

    if not isinstance(raw_depth, np.ndarray) or raw_depth.ndim != 2:
        raise ConditioningError
    if raw_depth.dtype != np.float32:
        raise ConditioningError
    working = np.array(raw_depth, dtype=np.float32, copy=True)
    if not np.isfinite(working).all():
        raise ConditioningError

    if not LARGER_RAW_VALUE_IS_NEAR:
        np.negative(working, out=working)
    if parameters.invert:
        np.negative(working, out=working)

    low = np.float32(np.percentile(working, parameters.far_percentile))
    high = np.float32(np.percentile(working, parameters.near_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < _RANGE_EPSILON:
        normalized = np.zeros(working.shape, dtype=np.float32)
    else:
        np.clip(working, low, high, out=working)
        normalized = (working - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    if parameters.gamma != 1.0:
        normalized = np.power(normalized, np.float32(parameters.gamma))
        normalized = np.clip(normalized, 0.0, 1.0)
    if parameters.contrast != 1.0:
        normalized = (normalized - np.float32(0.5)) * np.float32(parameters.contrast)
        normalized += np.float32(0.5)
        normalized = np.clip(normalized, 0.0, 1.0)
    if parameters.smoothing > 0.0:
        normalized = _edge_aware_smooth(
            normalized.astype(np.float32, copy=False),
            parameters.smoothing,
            parameters.edge_preservation,
        )
    result = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if result.shape != raw_depth.shape or not np.isfinite(result).all():
        raise ConditioningError
    return result


def bilateral_settings(
    smoothing: float,
    edge_preservation: float,
) -> tuple[int, float, float] | None:
    """Map the user sliders to an OpenCV bilateral filter.

    Smoothing 0 bypasses the filter.

    Otherwise the diameter is an odd value from 3 to 7:

        steps = clamp(round(smoothing / 100 * 3), 1, 3)
        diameter = 2 * steps + 1

    ``sigma_space`` equals that diameter, in pixels.

    ``sigma_color`` is in conditioned units, where 0 is far and 1 is near:

        sigma_color = 0.02 + (1 - edge_preservation / 100) * 0.18

    Edge preservation 100 keeps the color window tight (0.02). Edge
    preservation 0 opens it to 0.20. The default 75 gives 0.065.
    """

    if smoothing <= 0.0:
        return None
    steps = int(np.clip(np.rint((smoothing / 100.0) * 3.0), 1, 3))
    diameter = 2 * steps + 1
    sigma_space = float(diameter)
    sigma_color = 0.02 + (1.0 - edge_preservation / 100.0) * 0.18
    return diameter, sigma_color, sigma_space


def _edge_aware_smooth(
    values: np.ndarray,
    smoothing: float,
    edge_preservation: float,
) -> np.ndarray:
    settings = bilateral_settings(smoothing, edge_preservation)
    if settings is None:
        return values
    import cv2

    diameter, sigma_color, sigma_space = settings
    smoothed = cv2.bilateralFilter(values, diameter, sigma_color, sigma_space)
    return np.asarray(smoothed, dtype=np.float32)
