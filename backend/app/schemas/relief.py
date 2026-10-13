"""Relief regions and segmentation prompts.

Masks stay on disk. These models store parameters only.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

RegionRole = Literal["primary", "secondary", "supporting", "background", "custom"]


class ImagePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class ImageBox(BaseModel):
    """Inclusive pixel box in original image coordinates, ``x0 < x1`` and ``y0 < y1``."""

    model_config = ConfigDict(extra="forbid")

    x0: float = Field(allow_inf_nan=False)
    y0: float = Field(allow_inf_nan=False)
    x1: float = Field(allow_inf_nan=False)
    y1: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def corners_are_ordered(self) -> "ImageBox":
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("Box corners must increase from the first corner to the second.")
        return self


class SegmentationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_points: list[ImagePoint] = Field(default_factory=list)
    negative_points: list[ImagePoint] = Field(default_factory=list)
    box: ImageBox | None = None
    name: str = Field(default="", max_length=80)
    role: RegionRole | None = None

    @model_validator(mode="after")
    def has_a_foreground_prompt(self) -> "SegmentationRequest":
        if not self.positive_points and self.box is None:
            raise ValueError("Add a positive point or a box before generating a mask.")
        return self


class MagicWandRequest(BaseModel):
    """One Magic Wand edit in original image pixels.

    Tolerance is a CIE76 Lab distance. 0 selects only the seed color and
    100 is the widest supported match. The wand does not run on slider changes.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["add", "subtract"]
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    tolerance: float = Field(ge=0, le=100, allow_inf_nan=False)
    contiguous: bool = True


class MaskStrokeRequest(BaseModel):
    """One completed brush stroke in original image pixels."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["add", "erase"]
    brush_radius_px: float = Field(gt=0, allow_inf_nan=False)
    points: list[ImagePoint] = Field(min_length=1, max_length=10000)


class ProposalDraftUpdate(BaseModel):
    """Name and role for an in-progress segmentation proposal."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", max_length=80)
    role: RegionRole | None = None


class SegmentationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: UUID
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    model_id: str
    name: str = ""
    role: RegionRole | None = None
    origin_type: Literal["new_region", "accepted_region"] = "new_region"
    region_id: UUID | None = None


class RegionMaskEditSave(BaseModel):
    """Save an accepted-region edit back onto that same region."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: UUID
    name: str = Field(min_length=1, max_length=80)
    role: RegionRole


class RegionMaskEditDiscard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: UUID


class ReliefRegionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    role: RegionRole
    proposal_id: UUID


class ReliefRegionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str = Field(min_length=1, max_length=80)
    role: RegionRole
    elevation: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    depth_range: float = Field(ge=0.0, le=2.0, allow_inf_nan=False)
    feather: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    order: int = Field(ge=0)
    detail_amount: float = Field(default=0.0, ge=0.0, le=0.05, allow_inf_nan=False)
    detail_scale_mm: float | None = Field(default=None, ge=0.25, le=1.0, allow_inf_nan=False)


class ReliefCompositionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regions: list[ReliefRegionUpdate]
    background_depth_range: float = Field(default=1.0, ge=0.0, le=2.0, allow_inf_nan=False)
    background_elevation: float = Field(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False)
    background_feather: float = Field(default=0.0, ge=0.0, le=100.0, allow_inf_nan=False)


class PhysicalHeightRequest(BaseModel):
    """Finished-surface thicknesses in millimeters. XY stock size is not used."""

    model_config = ConfigDict(extra="forbid")

    stock_thickness_mm: float = Field(allow_inf_nan=False)
    base_thickness_mm: float = Field(allow_inf_nan=False)
    maximum_surface_height_mm: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def thicknesses_are_ordered(self) -> "PhysicalHeightRequest":
        if not (
            0 < self.base_thickness_mm < self.maximum_surface_height_mm <= self.stock_thickness_mm
        ):
            raise ValueError(
                "Base thickness must be above 0 and below the maximum surface height, "
                "which cannot exceed the stock thickness."
            )
        return self


class HeightPercentiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float


class PhysicalHeightMetrics(BaseModel):
    """Observed array statistics. These are separate from the saved mapping settings."""

    model_config = ConfigDict(extra="forbid")

    width_px: int
    height_px: int
    sample_count: int
    minimum_mm: float
    maximum_mm: float
    mean_mm: float
    median_mm: float
    standard_deviation_mm: float
    percentiles_mm: HeightPercentiles
    range_mm: float
    non_finite_count: int
    below_zero_count: int
    above_stock_count: int


class PhysicalHeightHistogram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins: int
    edges_mm: list[float]
    counts: list[int]
    sample_count: int


class RecordedPhysicalMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_thickness_mm: float
    base_thickness_mm: float
    maximum_surface_height_mm: float
    maximum_theoretical_cut_depth_mm: float


class PhysicalHeightValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str]
    warnings: list[str]


class PhysicalHeightDiagnostics(BaseModel):
    """Read-only diagnostics. This response is not stored on the project."""

    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    status: Literal["missing", "unknown", "current", "stale"]
    width_px: int
    height_px: int
    metrics: PhysicalHeightMetrics | None
    histogram: PhysicalHeightHistogram | None
    validation: PhysicalHeightValidation
    mapping: RecordedPhysicalMapping


class ProfileSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_px: float
    y_px: float
    distance_mm: float
    height_mm: float | None = None


class ProfileLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_x_px: float
    start_y_px: float
    end_x_px: float
    end_y_px: float


class ProfileSummary(BaseModel):
    """Finite-sample profile statistics. Slope is between adjacent samples only."""

    model_config = ConfigDict(extra="forbid")

    length_mm: float
    sample_count: int
    valid_count: int
    invalid_count: int
    minimum_mm: float | None = None
    maximum_mm: float | None = None
    range_mm: float | None = None
    mean_mm: float | None = None
    maximum_adjacent_delta_mm: float | None = None
    maximum_abs_slope_mm_per_mm: float | None = None


class PhysicalHeightProfile(BaseModel):
    """Read-only cross section. This response is not stored on the project."""

    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    status: Literal["missing", "unknown", "current", "stale"]
    mode: Literal["horizontal", "vertical", "line"]
    coordinate_convention: str
    slope_note: str
    units: Literal["mm"] = "mm"
    artwork_width_mm: float
    artwork_height_mm: float
    width_px: int
    height_px: int
    line: ProfileLine
    summary: ProfileSummary
    samples: list[ProfileSample]
    warnings: list[str]


class PhysicalHeightRecord(BaseModel):
    """Saved physical-height settings. The height array stays on disk."""

    model_config = ConfigDict(extra="forbid")

    stock_thickness_mm: float
    base_thickness_mm: float
    maximum_surface_height_mm: float
    generated_at: datetime
    height_path: Literal["depth/composition/physical/physical_height_mm.npy"]
    preview_path: Literal["depth/composition/physical/preview.png"]
    stats_path: Literal["depth/composition/physical/stats.json"]
    source_relief_sha256: str
    source_relief_generated_at: datetime
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    dtype: Literal["float32"]
    height_min_mm: float
    height_max_mm: float
    maximum_cut_depth_mm: float
    relief_amplitude_mm: float
    content_sha256: str | None = None
    settings_sha256: str | None = None
    status: Literal["missing", "unknown", "current", "stale"] | None = None


class ReliefRegionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    role: RegionRole
    mask_filename: str
    elevation: float = 0.0
    depth_range: float = 1.0
    feather: float = 0.0
    order: int = 0
    detail_amount: float = Field(default=0.0, ge=0.0, le=0.05, allow_inf_nan=False)
    detail_scale_mm: float | None = Field(default=None, ge=0.25, le=1.0, allow_inf_nan=False)
    created_at: datetime
    updated_at: datetime


class ReliefRecord(BaseModel):
    """Saved regions and the last composed relief. Arrays are not stored here."""

    model_config = ConfigDict(extra="forbid")

    regions: list[ReliefRegionRecord] = Field(default_factory=list)
    background_depth_range: float = Field(default=1.0, ge=0.0, le=2.0, allow_inf_nan=False)
    background_elevation: float = Field(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False)
    background_feather: float = Field(default=0.0, ge=0.0, le=100.0, allow_inf_nan=False)
    conditioned_generated_at: datetime | None = None
    width_px: int | None = Field(default=None, gt=0)
    height_px: int | None = Field(default=None, gt=0)
    dtype: Literal["float32"] | None = None
    range_min: float | None = None
    range_max: float | None = None
    relief_filename: Literal["relief_depth.npy"] | None = None
    preview_filename: Literal["preview.png"] | None = None
    generated_at: datetime | None = None
    physical: PhysicalHeightRecord | None = None
    content_sha256: str | None = None
    conditioned_sha256: str | None = None
    settings_sha256: str | None = None
    settings_version: Literal[2, 3, 4] | None = None
    status: Literal["missing", "unknown", "current", "stale"] | None = None


class MeshPreviewInfo(BaseModel):
    """Read-only plan for a top-surface preview. No vertices are returned or stored."""

    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    status: Literal["missing", "unknown", "current", "stale"]
    surface: Literal["top"] = "top"
    source_width_px: int
    source_height_px: int
    preview_width_px: int
    preview_height_px: int
    vertex_count: int
    triangle_count: int
    artwork_width_mm: float
    artwork_height_mm: float
    z_minimum_mm: float
    z_maximum_mm: float
    estimated_bytes: int
    downsampling_required: bool
    max_vertices: int
    coordinate_convention: str
    winding: str
    sampling_method: str
    validation: PhysicalHeightValidation


class ExportGridSize(BaseModel):
    """Pixel dimensions. Height is the row count and width is the column count."""

    model_config = ConfigDict(extra="forbid")

    height_px: int
    width_px: int


class ExportSpacing(BaseModel):
    """End-to-end artwork span divided by the number of export intervals."""

    model_config = ConfigDict(extra="forbid")

    x_mm: float
    y_mm: float
    uniform: bool


class ExportPlacement(BaseModel):
    """Saved artwork center. It is not applied to the artwork-local plan."""

    model_config = ConfigDict(extra="forbid")

    recorded: bool
    applied: Literal[False] = False
    center_x_mm: float | None = None
    center_y_mm: float | None = None
    origin: Literal["front-left-stock"] | None = None


class ExportIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class ExportLimitsReport(BaseModel):
    """Ceilings used for this assessment. They are not a certification."""

    model_config = ConfigDict(extra="forbid")

    max_triangles: int
    max_stl_bytes: int
    max_peak_memory_bytes: int


class ExportReadiness(BaseModel):
    """Read-only export plan. No STL is generated and no mesh is certified."""

    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    provenance_status: Literal["missing", "unknown", "current", "stale"]
    export_target: Literal["artwork-local"] = "artwork-local"
    export_certificate: Literal["none"] = "none"
    coordinate_convention: str
    resolution_policy: str
    topology_note: str
    mode: Literal["native", "reduced"]
    source_grid: ExportGridSize | None
    planned_grid: ExportGridSize | None
    spacing_mm: ExportSpacing | None
    artwork_width_mm: float
    artwork_height_mm: float
    stock_width_mm: float
    stock_height_mm: float
    stock_thickness_mm: float | None
    base_thickness_mm: float | None
    maximum_surface_height_mm: float | None
    placement: ExportPlacement
    z_minimum_mm: float | None
    z_maximum_mm: float | None
    estimated_vertices: int | None
    estimated_triangles: int | None
    estimated_stl_bytes: int | None
    estimated_peak_memory_bytes: int | None
    topology_expected: Literal["closed-solid"] = "closed-solid"
    topology_validated: Literal[False] = False
    self_intersection_checked: Literal[False] = False
    zero_area_measured: Literal[False] = False
    blocking_errors: list[ExportIssue]
    warnings: list[ExportIssue]
    corrective_actions: list[str]
    ready_for_export: bool
    limits: ExportLimitsReport


class StlExportRequest(BaseModel):
    """Explicit STL export. Native mode rejects a named grid."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["native", "reduced"] = "native"
    height_px: int | None = Field(default=None, ge=2)
    width_px: int | None = Field(default=None, ge=2)


class MeshSolidInfo(BaseModel):
    """Read-only solid plan. Counts are estimates and no mesh is stored."""

    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    status: Literal["missing", "unknown", "current", "stale"]
    solid: Literal["watertight-relief"] = "watertight-relief"
    topology_validated: Literal[False] = False
    geometry_construction_available: bool
    coordinate_convention: str
    winding: str
    source_width_px: int
    source_height_px: int
    mesh_width_px: int
    mesh_height_px: int
    top_vertex_count: int
    bottom_vertex_count: int
    vertex_count: int
    top_triangle_count: int
    bottom_triangle_count: int
    sidewall_triangle_count: int
    triangle_count: int
    artwork_width_mm: float
    artwork_height_mm: float
    bounding_box_minimum_mm: list[float] | None
    bounding_box_maximum_mm: list[float] | None
    source_z_minimum_mm: float | None
    source_z_maximum_mm: float | None
    selected_z_minimum_mm: float | None
    selected_z_maximum_mm: float | None
    estimated_mesh_bytes: int
    memory_note: str
    downsampling_required: bool
    max_vertices: int
    sampling_method: str
    topology_note: str
    validation: PhysicalHeightValidation
