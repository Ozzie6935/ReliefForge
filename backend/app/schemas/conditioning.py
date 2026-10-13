"""Conditioning parameters and histogram responses.

Conditioned depth is dimensionless. It is not a CNC height.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DepthConditioningParameters(BaseModel):
    """Percentile clips and display adjustments applied to raw depth.

    Far and near clips are percentiles of the oriented raw values, not
    millimeters. ``far_percentile`` must stay below ``near_percentile``.
    """

    model_config = ConfigDict(extra="forbid")

    far_percentile: float = Field(default=1.0, ge=0.0, le=20.0)
    near_percentile: float = Field(default=99.0, ge=80.0, le=100.0)
    gamma: float = Field(default=1.0, ge=0.25, le=4.0)
    contrast: float = Field(default=1.0, ge=0.5, le=2.0)
    smoothing: float = Field(default=0.0, ge=0.0, le=100.0)
    edge_preservation: float = Field(default=75.0, ge=0.0, le=100.0)
    invert: bool = False

    @model_validator(mode="after")
    def far_clip_is_below_near_clip(self) -> "DepthConditioningParameters":
        if self.far_percentile >= self.near_percentile:
            raise ValueError("Far clip must be below the near clip.")
        return self


class ConditioningRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: DepthConditioningParameters
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dtype: Literal["float32"] = "float32"
    range_min: float
    range_max: float
    conditioned_filename: Literal["conditioned_depth.npy"]
    preview_filename: Literal["preview.png"]
    generated_at: datetime
    content_sha256: str | None = None
    raw_sha256: str | None = None
    settings_sha256: str | None = None
    status: Literal["missing", "unknown", "current", "stale"] | None = None


class HistogramSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins: int = Field(gt=0)
    counts: list[int]
    minimum: float
    maximum: float


class DepthHistogram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: HistogramSeries
    conditioned: HistogramSeries | None = None
