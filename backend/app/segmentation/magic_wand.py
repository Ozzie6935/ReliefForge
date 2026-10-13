"""Contiguous color selection for the mask editor.

This module knows nothing about HTTP, project files, SAM, or relief
composition. It compares source colors in OpenCV Lab space.

A float RGB image in 0..1 converts to Lab with L in 0..100 and a/b signed.
The tolerance is the Euclidean CIE76 distance from the seed pixel. The
supported range is 0 through 100, and the editor default is 18. Contiguous
selection uses 8-connected flood fill so a click cannot jump to a separated
region of a similar color.
"""

from math import isfinite

import cv2
import numpy as np

MIN_TOLERANCE = 0.0
MAX_TOLERANCE = 100.0
DEFAULT_TOLERANCE = 18.0


class MagicWandError(ValueError):
    """The wand selection cannot be computed."""


def select_similar(
    rgb: np.ndarray,
    x: float,
    y: float,
    tolerance: float,
    *,
    contiguous: bool = True,
) -> np.ndarray:
    """Return a boolean mask of pixels similar to the seed.

    Coordinates are source pixels and are clamped into the image, matching
    brush strokes. The input array is not modified.
    """

    height, width = _require_rgb(rgb)
    limit = _require_tolerance(tolerance)
    seed_x = _pixel_index(x, width)
    seed_y = _pixel_index(y, height)
    lab = cv2.cvtColor(rgb.astype(np.float32) / np.float32(255), cv2.COLOR_RGB2LAB)
    seed = lab[seed_y, seed_x]
    distance = np.sqrt(np.sum((lab - seed) ** 2, axis=2))
    similar = distance <= limit
    if not contiguous:
        return similar
    return _contiguous(similar, seed_x, seed_y)


def paint_selection(mask: np.ndarray, selected: np.ndarray, mode: str) -> np.ndarray:
    """Return a new uint8 mask with the selection added or subtracted.

    ``mask`` is not modified. Selected pixels become 255 for ``add`` and 0
    for ``subtract``.
    """

    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise MagicWandError("Mask must be a uint8 image.")
    if not np.isin(mask, (0, 255)).all():
        raise MagicWandError("Mask must contain only 0 and 255.")
    if not isinstance(selected, np.ndarray) or selected.dtype != np.bool_ or selected.shape != mask.shape:
        raise MagicWandError("Selection must match the mask.")
    if mode not in ("add", "subtract"):
        raise MagicWandError("Magic Wand mode must be add or subtract.")
    result = mask.copy()
    if mode == "add":
        result[selected] = np.uint8(255)
    else:
        result[selected] = np.uint8(0)
    return result


def _require_rgb(rgb: np.ndarray) -> tuple[int, int]:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise MagicWandError("Source image must be a uint8 RGB array.")
    height, width = rgb.shape[:2]
    if height < 1 or width < 1:
        raise MagicWandError("Source image must be a uint8 RGB array.")
    return height, width


def _require_tolerance(tolerance: float) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not isfinite(float(tolerance)):
        raise MagicWandError("Tolerance must be a finite Lab distance.")
    value = float(tolerance)
    if value < MIN_TOLERANCE or value > MAX_TOLERANCE:
        raise MagicWandError("Tolerance must be from 0 through 100.")
    return value


def _pixel_index(value: float, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise MagicWandError("Coordinates must be finite.")
    clamped = min(max(float(value), 0.0), limit - 1e-6)
    return min(int(np.floor(clamped)), limit - 1)


def _contiguous(similar: np.ndarray, seed_x: int, seed_y: int) -> np.ndarray:
    height, width = similar.shape
    barriers = np.where(similar, np.uint8(255), np.uint8(0))
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(barriers, flood_mask, (seed_x, seed_y), 0, flags=flags)
    return flood_mask[1:-1, 1:-1] == np.uint8(255)
