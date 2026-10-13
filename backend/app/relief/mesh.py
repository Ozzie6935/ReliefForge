"""Read-only top surface for a saved physical-height array.

Coordinates are millimeters in the Milestone 7.2 pixel-center frame.
Y increases from the top of the artwork toward the bottom.
Z is the stored physical height and increases upward.

This builds an open heightfield surface. It is not a watertight solid,
and it does not write a project file or replace the canonical array.
"""

import math
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import numpy as np

from app.projects.errors import InvalidMeshRequest
from app.relief.diagnostics import DTYPE_WARNING, STRUCTURE_DETAIL
from app.schemas.relief import MeshPreviewInfo, PhysicalHeightValidation

DEFAULT_MAX_VERTICES = 250_000
MIN_GRID = 2
VERTEX_DTYPE = np.float64
INDEX_DTYPE = np.int32
ArtifactStatus = Literal["missing", "unknown", "current", "stale"]

COORDINATE_CONVENTION = (
    "Pixel centers map onto the artwork rectangle. "
    "X increases left to right and Y increases top to bottom. "
    "Z is physical_height_mm at that pixel and increases upward from the stock bottom. "
    "For more than one pixel, X = column * artwork_width_mm / (width_px - 1) "
    "and Y = row * artwork_height_mm / (height_px - 1). "
    "Column 0 and row 0 are 0 mm. The last column and last row are the artwork edges. "
    "Stock size and artwork center offsets are not applied. "
    "The mesh is local to the artwork rectangle."
)
WINDING = (
    "Each grid cell emits two triangles in this order: "
    "(row, column), (row, column + 1), (row + 1, column) and "
    "(row, column + 1), (row + 1, column + 1), (row + 1, column). "
    "With Y increasing downward, that winding gives face normals whose Z component is positive."
)
SAMPLING_METHOD = (
    "Preview samples use integer endpoint spacing: "
    "index[i] = i * (length - 1) // (sample_count - 1). "
    "The first and last source pixels are kept on both axes. "
    "Z is the stored height at the chosen pixel, not an average or interpolation. "
    "Pixels between chosen samples are omitted, so a narrow peak can be absent from the preview. "
    "Kept vertices still span the full artwork width and height."
)
AXIS_DETAIL = "Mesh preview requires width_px >= 2 and height_px >= 2."
NON_FINITE_DETAIL = (
    "Physical height contains non-finite samples. Mesh construction does not replace them."
)
ARTWORK_DETAIL = "Artwork width and height must be finite and greater than 0."
BUDGET_DETAIL = "max_vertices must be an integer greater than or equal to 4."
DEGENERATE_DETAIL = "The height grid does not form a non-degenerate top surface."
DOWNSAMPLE_WARNING = (
    "Preview geometry omits source pixels between the chosen samples. "
    "It is not a filtered resample of the canonical height array."
)


@dataclass(frozen=True)
class PreviewGridPlan:
    source_height_px: int
    source_width_px: int
    mesh_height_px: int
    mesh_width_px: int
    vertex_count: int
    triangle_count: int
    downsampling_required: bool
    max_vertices: int

    @property
    def estimated_bytes(self) -> int:
        return mesh_memory_bytes(self.vertex_count, self.triangle_count)


@dataclass(frozen=True)
class TopSurfaceMesh:
    """Open triangulated top surface. Vertices are float64 XYZ millimeters."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    source_height_px: int
    source_width_px: int
    mesh_height_px: int
    mesh_width_px: int
    row_indices: np.ndarray
    column_indices: np.ndarray
    downsampling_required: bool
    max_vertices: int

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def bounding_box_mm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lowest = self.vertices.min(axis=0)
        highest = self.vertices.max(axis=0)
        return (
            (float(lowest[0]), float(lowest[1]), float(lowest[2])),
            (float(highest[0]), float(highest[1]), float(highest[2])),
        )


def mesh_memory_bytes(vertex_count: int, triangle_count: int) -> int:
    """Bytes for float64 vertices, int32 triangle indices, and float64 face normals.

    This is those three arrays only. It does not include a solid, an STL file,
    or temporary allocator overhead.
    """

    return vertex_count * 3 * 8 + triangle_count * 3 * 4 + triangle_count * 3 * 8


def endpoint_indices(length: int, count: int) -> np.ndarray:
    """Choose `count` source indices, always including 0 and length - 1."""

    if length < MIN_GRID or count < MIN_GRID or count > length:
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    if count == length:
        return np.arange(length, dtype=np.int64)
    indices = (np.arange(count, dtype=np.int64) * (length - 1)) // (count - 1)
    if int(indices[0]) != 0 or int(indices[-1]) != length - 1:
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    if int(np.unique(indices).size) != count:
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    return indices


def plan_preview_grid(height_px: int, width_px: int, max_vertices: int = DEFAULT_MAX_VERTICES) -> PreviewGridPlan:
    """Choose a preview grid without allocating vertices or triangles."""

    _require_budget(max_vertices)
    if height_px < MIN_GRID or width_px < MIN_GRID:
        raise InvalidMeshRequest(AXIS_DETAIL)
    source_vertices = height_px * width_px
    if source_vertices <= max_vertices:
        mesh_height, mesh_width = height_px, width_px
        downsampling = False
    else:
        mesh_height, mesh_width = _fit_grid(height_px, width_px, max_vertices)
        downsampling = True
    vertex_count = mesh_height * mesh_width
    triangle_count = 2 * (mesh_height - 1) * (mesh_width - 1)
    if vertex_count > max_vertices or vertex_count < 4 or triangle_count < 2:
        raise InvalidMeshRequest(BUDGET_DETAIL)
    return PreviewGridPlan(
        source_height_px=height_px,
        source_width_px=width_px,
        mesh_height_px=mesh_height,
        mesh_width_px=mesh_width,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        downsampling_required=downsampling,
        max_vertices=max_vertices,
    )


def build_top_surface(
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    max_vertices: int = DEFAULT_MAX_VERTICES,
) -> TopSurfaceMesh:
    """Triangulate the top surface. The input array is not modified."""

    _require_height_array(array)
    _require_artwork(artwork_width_mm, artwork_height_mm)
    height_px, width_px = int(array.shape[0]), int(array.shape[1])
    plan = plan_preview_grid(height_px, width_px, max_vertices)
    rows = endpoint_indices(height_px, plan.mesh_height_px)
    columns = endpoint_indices(width_px, plan.mesh_width_px)
    sampled = np.asarray(array[np.ix_(rows, columns)], dtype=VERTEX_DTYPE)
    ys = _axis_coordinates(rows, height_px, float(artwork_height_mm))
    xs = _axis_coordinates(columns, width_px, float(artwork_width_mm))

    grid = np.empty((plan.mesh_height_px, plan.mesh_width_px, 3), dtype=VERTEX_DTYPE)
    grid[:, :, 0] = xs
    grid[:, :, 1] = ys[:, None]
    grid[:, :, 2] = sampled
    vertices = grid.reshape(-1, 3)
    triangles = _triangles(plan.mesh_height_px, plan.mesh_width_px)
    normals = _face_normals(vertices, triangles)
    return TopSurfaceMesh(
        vertices=vertices,
        triangles=triangles,
        normals=normals,
        source_height_px=height_px,
        source_width_px=width_px,
        mesh_height_px=plan.mesh_height_px,
        mesh_width_px=plan.mesh_width_px,
        row_indices=rows,
        column_indices=columns,
        downsampling_required=plan.downsampling_required,
        max_vertices=max_vertices,
    )


def describe_mesh_preview(
    *,
    project_id: UUID,
    status: ArtifactStatus,
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    max_vertices: int = DEFAULT_MAX_VERTICES,
    stock_thickness_mm: float | None = None,
) -> MeshPreviewInfo:
    """Plan a preview mesh. This does not allocate triangle buffers."""

    _require_height_array(array)
    _require_artwork(artwork_width_mm, artwork_height_mm)
    height_px, width_px = int(array.shape[0]), int(array.shape[1])
    plan = plan_preview_grid(height_px, width_px, max_vertices)
    warnings: list[str] = []
    if array.dtype != np.float32:
        warnings.append(DTYPE_WARNING)
    if float(array.min()) < 0.0:
        warnings.append("Some samples are below 0 mm.")
    if stock_thickness_mm is not None and float(array.max()) > float(stock_thickness_mm):
        warnings.append("Some samples are above the recorded stock thickness.")
    if plan.downsampling_required:
        warnings.append(DOWNSAMPLE_WARNING)
    return MeshPreviewInfo(
        project_id=project_id,
        status=status,
        surface="top",
        source_width_px=width_px,
        source_height_px=height_px,
        preview_width_px=plan.mesh_width_px,
        preview_height_px=plan.mesh_height_px,
        vertex_count=plan.vertex_count,
        triangle_count=plan.triangle_count,
        artwork_width_mm=float(artwork_width_mm),
        artwork_height_mm=float(artwork_height_mm),
        z_minimum_mm=float(array.min()),
        z_maximum_mm=float(array.max()),
        estimated_bytes=plan.estimated_bytes,
        downsampling_required=plan.downsampling_required,
        max_vertices=max_vertices,
        coordinate_convention=COORDINATE_CONVENTION,
        winding=WINDING,
        sampling_method=SAMPLING_METHOD,
        validation=PhysicalHeightValidation(ok=True, errors=[], warnings=warnings),
    )


def _require_height_array(array: np.ndarray) -> None:
    numeric = isinstance(array, np.ndarray) and (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    )
    if not numeric or array.ndim != 2 or array.size == 0:
        raise InvalidMeshRequest(STRUCTURE_DETAIL)
    if int(array.shape[0]) < MIN_GRID or int(array.shape[1]) < MIN_GRID:
        raise InvalidMeshRequest(AXIS_DETAIL)
    if not np.isfinite(array).all():
        raise InvalidMeshRequest(NON_FINITE_DETAIL)


def _require_artwork(width_mm: float, height_mm: float) -> None:
    if not math.isfinite(width_mm) or not math.isfinite(height_mm) or width_mm <= 0 or height_mm <= 0:
        raise InvalidMeshRequest(ARTWORK_DETAIL)


def _require_budget(max_vertices: int) -> None:
    if isinstance(max_vertices, bool) or not isinstance(max_vertices, int) or max_vertices < 4:
        raise InvalidMeshRequest(BUDGET_DETAIL)


def _fit_grid(height_px: int, width_px: int, max_vertices: int) -> tuple[int, int]:
    aspect = width_px / height_px
    mesh_height = int(math.floor(math.sqrt(max_vertices / aspect)))
    mesh_height = min(height_px, max(MIN_GRID, mesh_height))
    mesh_width = min(width_px, max(MIN_GRID, max_vertices // mesh_height))
    while mesh_height * mesh_width > max_vertices:
        if mesh_width >= mesh_height and mesh_width > MIN_GRID:
            mesh_width -= 1
        elif mesh_height > MIN_GRID:
            mesh_height -= 1
        else:
            raise InvalidMeshRequest(BUDGET_DETAIL)
    return mesh_height, mesh_width


def _axis_coordinates(indices: np.ndarray, length: int, extent_mm: float) -> np.ndarray:
    coordinates = indices.astype(VERTEX_DTYPE) * (extent_mm / (length - 1))
    coordinates[0] = 0.0
    coordinates[-1] = extent_mm
    return coordinates


def _triangles(height_px: int, width_px: int) -> np.ndarray:
    cells_y = height_px - 1
    cells_x = width_px - 1
    rows = np.arange(cells_y, dtype=INDEX_DTYPE)[:, None]
    columns = np.arange(cells_x, dtype=INDEX_DTYPE)[None, :]
    upper_left = rows * width_px + columns
    upper_right = upper_left + 1
    lower_left = upper_left + width_px
    lower_right = lower_left + 1
    triangles = np.empty((cells_y, cells_x, 2, 3), dtype=INDEX_DTYPE)
    triangles[:, :, 0, 0] = upper_left
    triangles[:, :, 0, 1] = upper_right
    triangles[:, :, 0, 2] = lower_left
    triangles[:, :, 1, 0] = upper_right
    triangles[:, :, 1, 1] = lower_right
    triangles[:, :, 1, 2] = lower_left
    flat = triangles.reshape(-1, 3)
    if int(flat.min()) < 0 or int(flat.max()) >= height_px * width_px:
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    return flat


def _face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    crossed = np.cross(second - first, third - first)
    lengths = np.linalg.norm(crossed, axis=1)
    if lengths.size == 0 or not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    normals = crossed / lengths[:, None]
    if not np.isfinite(normals).all() or np.any(normals[:, 2] <= 0.0):
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    return normals
