"""Read-only STL export planning for a saved physical-height array.

Coordinates are millimeters and match the relief solid. X increases left to
right, Y increases top to bottom, and Z increases upward. Column 0 and row 0
are 0 mm. The last column and last row are the artwork edges. Spacing along an
axis is the source-index position, index * extent_mm / (source_length - 1),
with those two endpoints forced. Stock width, stock height, and the saved
artwork center are not applied.

This plans an artwork-local relief only. It does not build a mesh, write an
STL, or choose the preview vertex budget. Native mode uses every saved sample.
A smaller grid is used only when the caller names that grid.
"""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import numpy as np

from app.projects.errors import InvalidExportRequest, InvalidMeshRequest
from app.relief.diagnostics import DTYPE_WARNING
from app.relief.mesh import endpoint_indices, mesh_memory_bytes
from app.relief.solid import HEIGHT_SEPARATION_TOLERANCE_MM
from app.schemas.relief import (
    ExportGridSize,
    ExportIssue,
    ExportLimitsReport,
    ExportPlacement,
    ExportReadiness,
    ExportSpacing,
    PhysicalHeightRecord,
)

ExportMode = Literal["native", "reduced"]
ProvenanceStatus = Literal["missing", "unknown", "current", "stale"]

MAX_EXPORT_TRIANGLES = 2_000_000
MAX_EXPORT_STL_BYTES = 100 * 1024 * 1024
MAX_EXPORT_PEAK_MEMORY_BYTES = 512 * 1024 * 1024
EXPORT_MEMORY_HEADROOM_BYTES = 32 * 1024 * 1024
BINARY_STL_HEADER_BYTES = 84
BINARY_STL_TRIANGLE_BYTES = 50

NATIVE_REJECTS_GRID = (
    "Native export does not accept height_px or width_px. "
    "Request mode=reduced to choose a smaller grid."
)
REDUCED_REQUIRES_GRID = "Reduced export requires both height_px and width_px."
REDUCED_NOT_SMALLER = (
    "Reduced export must be smaller than the saved physical-height array on at least one axis, "
    "and neither axis may be larger."
)
REDUCED_NOT_DISTINCT = "The requested export grid does not select distinct source samples."
ARTWORK_DETAIL = "Artwork width and height must be finite and greater than 0."
AXIS_DETAIL = "Export planning requires width_px >= 2 and height_px >= 2."
NON_FINITE_DETAIL = (
    "Physical height contains non-finite samples. Export planning does not replace them."
)
NON_POSITIVE_DETAIL = (
    "Physical height must be greater than 1e-9 mm at every sample, including the perimeter."
)
ABOVE_STOCK_DETAIL = (
    "Physical height exceeds the recorded stock thickness. Above-stock geometry blocks export."
)
MAPPING_DETAIL = "Saved physical mapping is not ordered as 0 < base < peak <= stock."
FILE_MISSING_DETAIL = "Physical height array was not found."
RECORD_MISSING_DETAIL = "Physical height has not been generated."
STALE_DETAIL = "Physical height is stale. Stale geometry is not export-ready."
UNKNOWN_DETAIL = "Physical-height provenance is unknown. Unknown geometry is not export-ready."

COORDINATE_CONVENTION = (
    "Artwork-local millimeters. X increases left to right and Y increases top to bottom. "
    "Z increases upward from 0 at the artwork bottom. "
    "Column 0 and row 0 are 0 mm. The last column and last row are the artwork edges. "
    "A sample at source index i is placed at i * extent_mm / (source_length - 1). "
    "The saved artwork center is measured from the front-left stock origin and is not applied. "
    "Stock width and stock height are not the export rectangle. "
    "This plan is not the preview mesh and does not use the preview vertex budget."
)
RESOLUTION_POLICY = (
    "Native export keeps every saved physical-height sample. "
    "Reduced export uses only the requested smaller grid and can omit narrow features. "
    "Exceeding a planning limit blocks the request. It does not silently choose a coarser grid."
)
TOPOLOGY_NOTE = (
    "A closed solid would use the saved heightfield, a flat bottom at Z = 0, and sidewalls "
    "that share those vertices. Topology, winding, volume, zero-area triangles, and "
    "self-intersection were not measured for this plan."
)


@dataclass(frozen=True)
class ExportResourceLimits:
    """Planning ceilings. These do not allocate a mesh or certify one."""

    max_triangles: int = MAX_EXPORT_TRIANGLES
    max_stl_bytes: int = MAX_EXPORT_STL_BYTES
    max_peak_memory_bytes: int = MAX_EXPORT_PEAK_MEMORY_BYTES


DEFAULT_EXPORT_LIMITS = ExportResourceLimits()


def validate_export_request(
    mode: ExportMode,
    height_px: int | None,
    width_px: int | None,
) -> None:
    """Reject a native request that names a grid and a reduced request that does not."""

    if mode == "native" and (height_px is not None or width_px is not None):
        raise InvalidExportRequest(NATIVE_REJECTS_GRID)
    if mode == "reduced" and (height_px is None or width_px is None):
        raise InvalidExportRequest(REDUCED_REQUIRES_GRID)
    if mode == "reduced" and (height_px < 2 or width_px < 2):
        raise InvalidExportRequest(AXIS_DETAIL)


def validate_reduced_grid(
    source_height_px: int,
    source_width_px: int,
    export_height_px: int,
    export_width_px: int,
) -> None:
    """Require an explicitly smaller grid whose endpoint samples stay distinct."""

    if (
        export_height_px > source_height_px
        or export_width_px > source_width_px
        or export_height_px * export_width_px >= source_height_px * source_width_px
    ):
        raise InvalidExportRequest(REDUCED_NOT_SMALLER)
    try:
        endpoint_indices(source_height_px, export_height_px)
        endpoint_indices(source_width_px, export_width_px)
    except InvalidMeshRequest as exc:
        raise InvalidExportRequest(REDUCED_NOT_DISTINCT) from exc


def solid_vertex_count(height_px: int, width_px: int) -> int:
    return 2 * height_px * width_px


def solid_triangle_count(height_px: int, width_px: int) -> int:
    """Closed-solid count: top, bottom, and four side walls. No arrays are built."""

    return (
        4 * (height_px - 1) * (width_px - 1)
        + 4 * (width_px - 1)
        + 4 * (height_px - 1)
    )


def binary_stl_bytes(triangle_count: int) -> int:
    """80-byte header, uint32 count, and 50 bytes per triangle."""

    return BINARY_STL_HEADER_BYTES + BINARY_STL_TRIANGLE_BYTES * triangle_count


def estimated_peak_memory_bytes(
    source_samples: int,
    vertex_count: int,
    triangle_count: int,
    source_bytes: int | None = None,
) -> int:
    """Bytes for the height array, the solid arrays, an STL buffer, and headroom.

    The solid term is float64 vertices, int32 indices, and float64 face normals.
    Nothing is allocated here.
    """

    stored = source_samples * 4 if source_bytes is None else source_bytes
    return (
        stored
        + mesh_memory_bytes(vertex_count, triangle_count)
        + binary_stl_bytes(triangle_count)
        + EXPORT_MEMORY_HEADROOM_BYTES
    )


def axis_positions(indices: np.ndarray, source_length: int, extent_mm: float) -> np.ndarray:
    """Place selected source indices on the artwork edge. Endpoints stay exact."""

    coordinates = indices.astype(np.float64) * (float(extent_mm) / (source_length - 1))
    coordinates[0] = 0.0
    coordinates[-1] = float(extent_mm)
    return coordinates


def _nominal_spacing(
    source_length: int,
    export_count: int,
    extent_mm: float,
) -> tuple[float, bool]:
    nominal = float(extent_mm) / (export_count - 1)
    if export_count == source_length:
        return nominal, True
    indices = endpoint_indices(source_length, export_count)
    steps = np.diff(indices.astype(np.float64)) * (float(extent_mm) / (source_length - 1))
    uniform = bool(np.all(np.isclose(steps, steps[0])))
    return nominal, uniform


def _issue(code: str, message: str) -> ExportIssue:
    return ExportIssue(code=code, message=message)


def _limit_message(kind: str, estimate: int, limit: int) -> str:
    return (
        f"Estimated {kind} {estimate} exceeds the export planning limit {limit}. "
        "Native resolution is not reduced automatically."
    )


def build_export_readiness(
    *,
    project_id: UUID,
    provenance_status: ProvenanceStatus,
    mode: ExportMode,
    artwork_width_mm: float,
    artwork_height_mm: float,
    stock_width_mm: float,
    stock_height_mm: float,
    center_x_mm: float | None,
    center_y_mm: float | None,
    record: PhysicalHeightRecord | None,
    file_present: bool,
    array: np.ndarray | None = None,
    limits: ExportResourceLimits = DEFAULT_EXPORT_LIMITS,
    export_height_px: int | None = None,
    export_width_px: int | None = None,
) -> ExportReadiness:
    """Assess one saved artifact. This does not allocate solid vertices or triangles."""

    errors: list[ExportIssue] = []
    warnings: list[ExportIssue] = []
    actions: list[str] = []
    source_grid = None
    planned_grid = None
    spacing = None
    z_min = None
    z_max = None
    vertices = None
    triangles = None
    stl_bytes = None
    peak_bytes = None

    placement_recorded = (
        center_x_mm is not None
        and center_y_mm is not None
        and bool(np.isfinite(center_x_mm))
        and bool(np.isfinite(center_y_mm))
    )
    placement = ExportPlacement(
        recorded=placement_recorded,
        applied=False,
        center_x_mm=float(center_x_mm) if placement_recorded else None,
        center_y_mm=float(center_y_mm) if placement_recorded else None,
        origin="front-left-stock" if placement_recorded else None,
    )
    if placement_recorded:
        warnings.append(
            _issue(
                "placement_not_applied",
                "Artwork center is recorded from the front-left stock origin and is not applied. "
                "The planned STL stays artwork-local.",
            )
        )
    else:
        warnings.append(
            _issue(
                "placement_not_recorded",
                "Stock-relative artwork placement is not recorded. No offset was invented.",
            )
        )

    if record is None:
        errors.append(_issue("physical_missing", RECORD_MISSING_DETAIL))
        actions.append(
            "Generate physical height from the current composed relief before planning an STL."
        )
    elif not file_present:
        errors.append(_issue("physical_file_missing", FILE_MISSING_DETAIL))
        actions.append(
            "Restore the saved physical-height array or generate it again. This check does not recreate it."
        )
    elif provenance_status == "stale":
        errors.append(_issue("provenance_stale", STALE_DETAIL))
        actions.append(
            "Regenerate physical height from the current composed relief. Provenance is not backfilled by this check."
        )
    elif provenance_status == "unknown":
        errors.append(_issue("provenance_unknown", UNKNOWN_DETAIL))
        actions.append(
            "Regenerate physical height so its lineage can be verified. This check does not repair provenance."
        )
    elif provenance_status != "current":
        errors.append(_issue("physical_missing", RECORD_MISSING_DETAIL))
        actions.append(
            "Generate physical height from the current composed relief before planning an STL."
        )

    if mode == "reduced" and record is not None:
        if export_height_px is None or export_width_px is None:
            raise InvalidExportRequest(REDUCED_REQUIRES_GRID)
        validate_reduced_grid(
            record.height_px,
            record.width_px,
            export_height_px,
            export_width_px,
        )

    artwork_ok = (
        bool(np.isfinite(artwork_width_mm))
        and bool(np.isfinite(artwork_height_mm))
        and float(artwork_width_mm) > 0
        and float(artwork_height_mm) > 0
    )
    if not artwork_ok:
        errors.append(_issue("artwork_dimensions", ARTWORK_DETAIL))
        actions.append("Save a finite artwork width and height greater than 0.")

    mapping_ok = False
    stock_thickness = None
    base_thickness = None
    peak_height = None
    if record is not None:
        stock_thickness = float(record.stock_thickness_mm)
        base_thickness = float(record.base_thickness_mm)
        peak_height = float(record.maximum_surface_height_mm)
        mapping_ok = bool(
            np.isfinite(stock_thickness)
            and np.isfinite(base_thickness)
            and np.isfinite(peak_height)
            and 0 < base_thickness < peak_height <= stock_thickness
        )
        if not mapping_ok:
            errors.append(_issue("mapping_parameters", MAPPING_DETAIL))
            actions.append(
                "Correct the stock, base, and maximum surface thicknesses, then regenerate physical height."
            )
        source_grid = ExportGridSize(height_px=record.height_px, width_px=record.width_px)

    export_height = record.height_px if record is not None and mode == "native" else export_height_px
    export_width = record.width_px if record is not None and mode == "native" else export_width_px
    grid_ok = (
        source_grid is not None
        and export_height is not None
        and export_width is not None
        and export_height >= 2
        and export_width >= 2
        and artwork_ok
    )
    if source_grid is not None and (source_grid.height_px < 2 or source_grid.width_px < 2):
        errors.append(_issue("grid_axis", AXIS_DETAIL))
        actions.append("Physical height needs at least 2 samples on each axis.")
        grid_ok = False
    elif mode == "reduced" and source_grid is not None and not grid_ok and artwork_ok:
        errors.append(_issue("grid_axis", AXIS_DETAIL))
        actions.append("Physical height needs at least 2 samples on each axis.")

    if array is not None and record is not None:
        if array.dtype != np.float32:
            warnings.append(_issue("dtype", DTYPE_WARNING))
        finite_mask = np.isfinite(array)
        if not bool(finite_mask.all()):
            errors.append(_issue("non_finite_height", NON_FINITE_DETAIL))
            actions.append("Replace the physical-height array with finite millimeter samples.")
        if bool(finite_mask.any()):
            finite = np.asarray(array[finite_mask], dtype=np.float64)
            z_min = float(finite.min())
            z_max = float(finite.max())
            if bool(np.any(finite <= HEIGHT_SEPARATION_TOLERANCE_MM)):
                errors.append(_issue("non_positive_height", NON_POSITIVE_DETAIL))
                actions.append(
                    "Regenerate physical height so every sample, including the perimeter, is above 1e-9 mm."
                )
            if mapping_ok and stock_thickness is not None and bool(np.any(finite > stock_thickness)):
                errors.append(_issue("above_stock", ABOVE_STOCK_DETAIL))
                actions.append(
                    "Reduce the maximum surface height or increase stock thickness, then regenerate physical height."
                )

    if grid_ok and export_height is not None and export_width is not None and source_grid is not None:
        planned_grid = ExportGridSize(height_px=export_height, width_px=export_width)
        spacing_x, uniform_x = _nominal_spacing(
            source_grid.width_px, export_width, float(artwork_width_mm)
        )
        spacing_y, uniform_y = _nominal_spacing(
            source_grid.height_px, export_height, float(artwork_height_mm)
        )
        spacing = ExportSpacing(
            x_mm=spacing_x,
            y_mm=spacing_y,
            uniform=uniform_x and uniform_y,
        )
        vertices = solid_vertex_count(export_height, export_width)
        triangles = solid_triangle_count(export_height, export_width)
        stl_bytes = binary_stl_bytes(triangles)
        peak_bytes = estimated_peak_memory_bytes(
            source_grid.height_px * source_grid.width_px,
            vertices,
            triangles,
            None if array is None else int(array.nbytes),
        )
        if triangles > limits.max_triangles:
            errors.append(
                _issue(
                    "triangle_limit",
                    _limit_message("triangle count", triangles, limits.max_triangles),
                )
            )
            actions.append(
                "Request an explicit smaller grid with mode=reduced. Native resolution is not reduced automatically."
            )
        if stl_bytes > limits.max_stl_bytes:
            errors.append(
                _issue(
                    "stl_byte_limit",
                    _limit_message("binary STL size", stl_bytes, limits.max_stl_bytes),
                )
            )
            actions.append(
                "Request an explicit smaller grid with mode=reduced. The full grid is still reported."
            )
        if peak_bytes > limits.max_peak_memory_bytes:
            errors.append(
                _issue(
                    "memory_limit",
                    _limit_message("peak memory", peak_bytes, limits.max_peak_memory_bytes),
                )
            )
            actions.append(
                "Request an explicit smaller grid with mode=reduced. This check does not allocate the mesh."
            )
        warnings.append(
            _issue(
                "topology_not_validated",
                "Closed-solid topology is expected and has not been validated.",
            )
        )
        warnings.append(
            _issue(
                "self_intersection_not_checked",
                "Self-intersection was not checked. This plan is not a manifold certificate.",
            )
        )
        warnings.append(
            _issue(
                "zero_area_not_measured",
                "Zero-area triangles were not measured. Positive artwork spacing only says the "
                "rectangle is non-degenerate.",
            )
        )
        if mode == "reduced":
            warnings.append(
                _issue(
                    "reduced_omits_features",
                    "Reduced-resolution export omits saved samples between the chosen indices. "
                    "Narrow features can be absent. This is not a filtered resample.",
                )
            )

    ready = provenance_status == "current" and not errors
    return ExportReadiness(
        project_id=project_id,
        provenance_status=provenance_status,
        export_target="artwork-local",
        export_certificate="none",
        coordinate_convention=COORDINATE_CONVENTION,
        resolution_policy=RESOLUTION_POLICY,
        topology_note=TOPOLOGY_NOTE,
        mode=mode,
        source_grid=source_grid,
        planned_grid=planned_grid,
        spacing_mm=spacing,
        artwork_width_mm=float(artwork_width_mm),
        artwork_height_mm=float(artwork_height_mm),
        stock_width_mm=float(stock_width_mm),
        stock_height_mm=float(stock_height_mm),
        stock_thickness_mm=stock_thickness,
        base_thickness_mm=base_thickness,
        maximum_surface_height_mm=peak_height,
        placement=placement,
        z_minimum_mm=z_min,
        z_maximum_mm=z_max,
        estimated_vertices=vertices,
        estimated_triangles=triangles,
        estimated_stl_bytes=stl_bytes,
        estimated_peak_memory_bytes=peak_bytes,
        topology_expected="closed-solid",
        topology_validated=False,
        self_intersection_checked=False,
        zero_area_measured=False,
        blocking_errors=errors,
        warnings=warnings,
        corrective_actions=actions,
        ready_for_export=ready,
        limits=ExportLimitsReport(
            max_triangles=limits.max_triangles,
            max_stl_bytes=limits.max_stl_bytes,
            max_peak_memory_bytes=limits.max_peak_memory_bytes,
        ),
    )
