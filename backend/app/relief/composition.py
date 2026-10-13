"""Compose a dimensionless relief from conditioned depth and region masks.

The result is float32 in ``[0, 1]``, where 0 is back and 1 is front. It is not
a physical height. Each call starts from the conditioned array it is given.

For one region, the local representative level is the mean of the conditioned
depth inside the hard mask. Depth range scales variation around that level, and
elevation shifts the level:

    adjusted = level + (conditioned - level) * depth_range + elevation

Optional fine detail is applied after that elevation adjustment and before the
per-region clip. It is a signed high-frequency signal from the source
photograph, multiplied by ``detail_amount`` and by the same feather weight.
``detail_amount`` of 0 adds nothing and keeps the previous formula, including
the identity-region shortcut. A nonzero amount processes an otherwise identity
region. The composed relief is not rescaled to make room for the detail.

``detail_scale_mm`` of None keeps the fixed 2-pixel Gaussian. A finite scale
selects a separate physical-width Gaussian. The pixel-mode function is not
converted through millimeters.

Feathering blurs the mask into a blending weight. It does not blur the depth
or the detail signal. Later regions, in ascending ``order``, blend on top of
earlier ones:

    result = (1 - weight) * result + weight * clip(adjusted, 0, 1)

``apply_background`` runs after that result. Identity background settings
return the same array and do not rebuild region-feather halos. A non-identity
setting blends the complement of the hard-mask union toward a plane taken from
the original conditioned depth. A large background feather can move subject
edge pixels. It does not repair a wrong mask.
"""

import math
from dataclasses import dataclass

import numpy as np

_FEATHER_SIGMA_PER_UNIT = 0.24
DETAIL_SIGMA_PX = 2.0
DETAIL_AMOUNT_MAX = 0.05
DETAIL_SCALE_MIN_MM = 0.25
DETAIL_SCALE_MAX_MM = 1.0
_DETAIL_SIGNAL_LIMIT = 1.0
_LUMA_RED = np.float32(0.2126)
_LUMA_GREEN = np.float32(0.7152)
_LUMA_BLUE = np.float32(0.0722)


class CompositionError(Exception):
    """The relief cannot be composed from these arrays."""


@dataclass(frozen=True)
class RegionContribution:
    region_id: str
    mask: np.ndarray
    elevation: float
    depth_range: float
    feather: float
    order: int
    detail_amount: float = 0.0
    detail_scale_mm: float | None = None


def feather_weights(mask: np.ndarray, feather: float) -> np.ndarray:
    """Return blending weights in ``[0, 1]``. Feather 0 keeps a hard mask."""

    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise CompositionError
    hard = mask.astype(np.float32)
    if feather <= 0.0:
        return hard
    import cv2

    sigma = _FEATHER_SIGMA_PER_UNIT * float(feather)
    blurred = cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(np.asarray(blurred, dtype=np.float32), 0.0, 1.0)


def extract_source_detail(rgb: np.ndarray) -> np.ndarray:
    """Signed fine detail from stored RGB pixels. A constant image returns zero.

    Luminance is Rec. 709, computed in float32 from the stored channel order:

        (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255

    No EXIF transpose is applied. Row 0 is the top of the stored image and
    column 0 is the left, matching the depth grid. A constant luminance
    returns an exact zero array and does not divide. Otherwise broad
    illumination is the Gaussian of that luminance at a fixed sigma of 2
    pixels. Detail is the luminance minus that blur, clipped to [-1, 1]. It
    is not divided by a regional maximum.
    """

    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or rgb.shape[0] < 1
        or rgb.shape[1] < 1
    ):
        raise CompositionError
    channels = rgb.astype(np.float32, copy=False)
    luminance = (
        _LUMA_RED * channels[..., 0]
        + _LUMA_GREEN * channels[..., 1]
        + _LUMA_BLUE * channels[..., 2]
    ) / np.float32(255.0)
    if float(luminance.max()) == float(luminance.min()):
        return np.zeros(luminance.shape, dtype=np.float32)
    import cv2

    low = cv2.GaussianBlur(
        np.ascontiguousarray(luminance),
        (0, 0),
        sigmaX=DETAIL_SIGMA_PX,
        sigmaY=DETAIL_SIGMA_PX,
    )
    high = luminance - np.asarray(low, dtype=np.float32)
    np.clip(high, -_DETAIL_SIGNAL_LIMIT, _DETAIL_SIGNAL_LIMIT, out=high)
    detail = np.ascontiguousarray(high, dtype=np.float32)
    if detail.shape != rgb.shape[:2] or not np.isfinite(detail).all():
        raise CompositionError
    return detail


def _source_luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance in the stored channel order. No transpose."""

    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or rgb.shape[0] < 1
        or rgb.shape[1] < 1
    ):
        raise CompositionError
    channels = rgb.astype(np.float32, copy=False)
    return (
        _LUMA_RED * channels[..., 0]
        + _LUMA_GREEN * channels[..., 1]
        + _LUMA_BLUE * channels[..., 2]
    ) / np.float32(255.0)


def _positive_finite(value: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def detail_scale_is_valid(scale: float | None) -> bool:
    """None is pixel mode. A physical scale must be finite and inside 0.25–1 mm."""

    if scale is None:
        return True
    try:
        value = float(scale)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and DETAIL_SCALE_MIN_MM <= value <= DETAIL_SCALE_MAX_MM


def extract_physical_source_detail(
    rgb: np.ndarray,
    *,
    artwork_width_mm: float,
    artwork_height_mm: float,
    detail_scale_mm: float,
) -> np.ndarray:
    """Signed detail at a physical Gaussian width. Pixel mode does not call this.

    Pitch uses the artwork-local endpoint spacing. Column 0 and row 0 stay at
    0 mm, and the last column and last row stay on the artwork edges:

        pitch_x_mm = artwork_width_mm / (width - 1)
        pitch_y_mm = artwork_height_mm / (height - 1)

    The blur uses those pitches independently. The image is not resampled or
    transposed. Luminance, the signed high-pass, constant-image zeros, float32,
    and the [-1, 1] clip match ``extract_source_detail``.
    """

    if not detail_scale_is_valid(detail_scale_mm) or detail_scale_mm is None:
        raise CompositionError
    if not _positive_finite(artwork_width_mm) or not _positive_finite(artwork_height_mm):
        raise CompositionError
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or rgb.shape[0] < 2
        or rgb.shape[1] < 2
    ):
        raise CompositionError
    luminance = _source_luminance(rgb)
    if float(luminance.max()) == float(luminance.min()):
        return np.zeros(luminance.shape, dtype=np.float32)
    height, width = rgb.shape[:2]
    pitch_x_mm = float(artwork_width_mm) / (width - 1)
    pitch_y_mm = float(artwork_height_mm) / (height - 1)
    import cv2

    low = cv2.GaussianBlur(
        np.ascontiguousarray(luminance),
        (0, 0),
        sigmaX=float(detail_scale_mm) / pitch_x_mm,
        sigmaY=float(detail_scale_mm) / pitch_y_mm,
    )
    high = luminance - np.asarray(low, dtype=np.float32)
    np.clip(high, -_DETAIL_SIGNAL_LIMIT, _DETAIL_SIGNAL_LIMIT, out=high)
    detail = np.ascontiguousarray(high, dtype=np.float32)
    if detail.shape != rgb.shape[:2] or not np.isfinite(detail).all():
        raise CompositionError
    return detail


def prepare_detail_signals(
    rgb: np.ndarray,
    regions: list[RegionContribution],
    *,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> tuple[np.ndarray | None, dict[float, np.ndarray] | None]:
    """Extract each requested mode once. Amount 0 does not extract.

    Pixel mode calls ``extract_source_detail``. Physical scales call
    ``extract_physical_source_detail``. Identical scales share one array.
    The maps live only for this call.
    """

    pixel: np.ndarray | None = None
    physical: dict[float, np.ndarray] | None = None
    for region in regions:
        if region.detail_amount <= 0.0:
            continue
        if not detail_scale_is_valid(region.detail_scale_mm):
            raise CompositionError
        if region.detail_scale_mm is None:
            if pixel is None:
                pixel = extract_source_detail(rgb)
            continue
        if physical is not None and region.detail_scale_mm in physical:
            continue
        signal = extract_physical_source_detail(
            rgb,
            artwork_width_mm=artwork_width_mm,
            artwork_height_mm=artwork_height_mm,
            detail_scale_mm=region.detail_scale_mm,
        )
        if physical is None:
            physical = {}
        physical[region.detail_scale_mm] = signal
    return pixel, physical


def _detail_amount_is_valid(amount: float) -> bool:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and 0.0 <= value <= DETAIL_AMOUNT_MAX


def _detail_plane(signal: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if (
        not isinstance(signal, np.ndarray)
        or signal.shape != shape
        or signal.dtype != np.float32
        or not np.isfinite(signal).all()
    ):
        raise CompositionError
    return signal


def compose_relief(
    conditioned_depth: np.ndarray,
    regions: list[RegionContribution],
    detail: np.ndarray | None = None,
    physical_details: dict[float, np.ndarray] | None = None,
) -> np.ndarray:
    """Return a new float32 relief. ``conditioned_depth`` is not modified.

    ``detail`` is the sigma-2 pixel signal. ``physical_details`` maps a
    millimeter scale to its signal. A plane is read only when a region has
    ``detail_amount`` above zero and selects that mode. Amount 0 does not
    read either argument.
    """

    if not isinstance(conditioned_depth, np.ndarray) or conditioned_depth.ndim != 2:
        raise CompositionError
    if conditioned_depth.dtype != np.float32:
        raise CompositionError
    base = np.array(conditioned_depth, dtype=np.float32, copy=True)
    if not np.isfinite(base).all():
        raise CompositionError
    needs_pixel = any(
        region.detail_amount > 0.0 and region.detail_scale_mm is None for region in regions
    )
    if needs_pixel:
        _detail_plane(detail, base.shape)
    for region in regions:
        scale = region.detail_scale_mm
        if region.detail_amount > 0.0 and scale is not None:
            if physical_details is None or scale not in physical_details:
                raise CompositionError
            _detail_plane(physical_details[scale], base.shape)
    result = base
    ordered = sorted(regions, key=lambda region: (region.order, region.region_id))
    for region in ordered:
        if not _detail_amount_is_valid(region.detail_amount):
            raise CompositionError
        if not detail_scale_is_valid(region.detail_scale_mm):
            raise CompositionError
        if region.mask.shape != base.shape or region.mask.dtype != np.bool_:
            raise CompositionError
        if not bool(region.mask.any()):
            continue
        if region.elevation == 0.0 and region.depth_range == 1.0 and region.detail_amount == 0.0:
            continue
        level = np.float32(base[region.mask].mean())
        adjusted = level + (base - level) * np.float32(region.depth_range)
        adjusted = adjusted + np.float32(region.elevation)
        if region.detail_amount > 0.0:
            if region.detail_scale_mm is None:
                signal = detail
            else:
                assert physical_details is not None
                signal = physical_details[region.detail_scale_mm]
            assert signal is not None
            adjusted = adjusted + np.float32(region.detail_amount) * signal
        adjusted = np.clip(adjusted, 0.0, 1.0).astype(np.float32, copy=False)
        weight = feather_weights(region.mask, region.feather)
        result = (np.float32(1.0) - weight) * result + weight * adjusted
    result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if result.shape != conditioned_depth.shape or not np.isfinite(result).all():
        raise CompositionError
    if float(result.min()) < 0.0 or float(result.max()) > 1.0:
        raise CompositionError
    return result


def _finite_in_range(value: float, low: float, high: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and low <= number <= high


def background_settings_are_identity(depth_range: float, elevation: float, feather: float) -> bool:
    """True only for depth range 1, elevation 0, and feather 0."""

    return (
        _finite_in_range(depth_range, 0.0, 2.0)
        and _finite_in_range(elevation, -1.0, 1.0)
        and _finite_in_range(feather, 0.0, 100.0)
        and float(depth_range) == 1.0
        and float(elevation) == 0.0
        and float(feather) == 0.0
    )


def apply_background(
    conditioned: np.ndarray,
    composed: np.ndarray,
    masks: list[np.ndarray],
    *,
    depth_range: float = 1.0,
    elevation: float = 0.0,
    feather: float = 0.0,
) -> np.ndarray:
    """Blend the uncovered complement after region composition.

    Identity settings return ``composed`` itself. The region-feather halo is
    not rebuilt. Non-identity settings use the original conditioned depth for
    the background plane, not the elevated relief. Feather 0 keeps every hard
    union pixel equal to ``composed``. A wider feather blends across the
    boundary in both directions and can change subject edges. An empty union
    or a union that covers the frame returns ``composed`` unchanged.
    """

    if background_settings_are_identity(depth_range, elevation, feather):
        return composed
    if (
        not _finite_in_range(depth_range, 0.0, 2.0)
        or not _finite_in_range(elevation, -1.0, 1.0)
        or not _finite_in_range(feather, 0.0, 100.0)
    ):
        raise CompositionError
    if (
        not isinstance(conditioned, np.ndarray)
        or not isinstance(composed, np.ndarray)
        or conditioned.shape != composed.shape
        or conditioned.ndim != 2
        or conditioned.dtype != np.float32
        or composed.dtype != np.float32
        or not np.isfinite(conditioned).all()
        or not np.isfinite(composed).all()
    ):
        raise CompositionError
    if not masks:
        return composed
    union = np.zeros(conditioned.shape, dtype=bool)
    for mask in masks:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != conditioned.shape:
            raise CompositionError
        union |= mask
    if not bool(union.any()) or bool(union.all()):
        return composed
    background = ~union
    level = np.float32(np.mean(conditioned[background], dtype=np.float64))
    adjusted = level + (conditioned - level) * np.float32(depth_range)
    adjusted = adjusted + np.float32(elevation)
    adjusted = np.clip(adjusted, 0.0, 1.0).astype(np.float32, copy=False)
    weight = np.float32(1.0) - feather_weights(union, float(feather))
    result = (np.float32(1.0) - weight) * composed + weight * adjusted
    result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if result.shape != composed.shape or not np.isfinite(result).all():
        raise CompositionError
    if float(result.min()) < 0.0 or float(result.max()) > 1.0:
        raise CompositionError
    return result
