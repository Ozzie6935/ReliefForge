"""Project document and create-request models.

Physical sizes are millimeters. Pixel sizes belong only to the source image.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.conditioning import ConditioningRecord
from app.schemas.relief import ReliefRecord


class Stock(BaseModel):
    """XY size is required. Thickness is optional so older manifests still load.

    ``thickness_mm`` is the project override. ``None`` means the project has
    no saved thickness. Readers use the application default only as a
    suggestion and do not write it back.
    """

    model_config = ConfigDict(extra="forbid")

    width_mm: float = Field(gt=0, allow_inf_nan=False)
    height_mm: float = Field(gt=0, allow_inf_nan=False)
    thickness_mm: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class Artwork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width_mm: float = Field(gt=0, allow_inf_nan=False)
    height_mm: float = Field(gt=0, allow_inf_nan=False)
    center_x_mm: float = Field(allow_inf_nan=False)
    center_y_mm: float = Field(allow_inf_nan=False)


class SourceImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    original_filename: str
    media_type: str
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    content_sha256: str | None = None
    status: Literal["missing", "unknown", "current", "stale"] | None = None


class DepthArtifact(BaseModel):
    """Metadata for a raw relative depth result. The array itself is not stored here."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_version: str
    device: Literal["mps", "cpu"]
    input_size: int = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    raw_filename: Literal["raw_depth.npy"]
    preview_filename: Literal["preview.png"]
    generated_at: datetime
    conditioning: ConditioningRecord | None = None
    content_sha256: str | None = None
    source_sha256: str | None = None
    settings_sha256: str | None = None
    status: Literal["missing", "unknown", "current", "stale"] | None = None


class ProjectDuplicateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)


class ProjectDuplicateResult(BaseModel):
    """Identity of a new project. Artifact status is resolved by a later read."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    slug: str
    name: str
    source_project_id: UUID
    duplicated: Literal[True] = True


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    stock: Stock
    artwork: Artwork


class Project(BaseModel):
    """Schema version 1.

    ``created_at`` and ``updated_at`` are optional so projects saved before
    those fields existed, including Angie Dance, still load. New writes
    include both timestamps. ``depth`` is optional for the same reason:
    projects created before raw depth generation have no depth record.
    The directory name is the slug assigned at creation and does not follow
    later display-name changes. Stock thickness is optional for the same
    reason: projects saved before that field existed omit it. Schema version
    stays 1.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    id: UUID
    name: str
    slug: str
    units: Literal["mm"] = "mm"
    stock: Stock
    artwork: Artwork
    source: SourceImage | None = None
    depth: DepthArtifact | None = None
    relief: ReliefRecord | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
