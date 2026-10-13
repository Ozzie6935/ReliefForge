"""Grayscale preview of a raw depth array.

The preview is a display copy. It does not change the authoritative array.
"""

import io

import numpy as np
from PIL import Image

from app.projects.errors import DepthInferenceFailed


def grayscale_preview(depth: np.ndarray) -> bytes:
    if depth.dtype != np.float32 or depth.ndim != 2:
        raise DepthInferenceFailed
    values = np.array(depth, dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise DepthInferenceFailed
    low = float(values.min())
    high = float(values.max())
    if high > low:
        scaled = (values - low) / (high - low)
    else:
        scaled = np.zeros_like(values)
    pixels = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def unit_interval_preview(depth: np.ndarray) -> bytes:
    """Map a [0, 1] array directly to grayscale. Values are not stretched again.

    A second min/max stretch would hide contrast and gamma. The input array
    is copied and left unchanged.
    """

    if depth.dtype != np.float32 or depth.ndim != 2:
        raise DepthInferenceFailed
    values = np.array(depth, dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise DepthInferenceFailed
    pixels = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()
