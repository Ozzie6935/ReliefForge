"""Watertight relief solid for a saved physical-height array.

Coordinates are millimeters. X increases left to right, Y increases top to
bottom, and Z increases upward. The top is the selected physical-height
samples. The bottom is the same XY grid at Z = 0. Sidewalls use those
vertex indices, so shared corners are watertight by index.

This does not write a mesh file, an STL, or a project record.
A passing edge-manifold check is not an export-ready certificate.
Non-adjacent triangle intersections are tested only for meshes with at
most SELF_INTERSECTION_TRIANGLE_CAP triangles.
"""

from dataclasses import dataclass
from uuid import UUID

import numpy as np

from app.projects.errors import InvalidMeshRequest
from app.relief.diagnostics import ABOVE_STOCK_WARNING, DTYPE_WARNING, STRUCTURE_DETAIL
from app.relief.mesh import (
    AXIS_DETAIL,
    DEFAULT_MAX_VERTICES,
    DOWNSAMPLE_WARNING,
    INDEX_DTYPE,
    MIN_GRID,
    NON_FINITE_DETAIL,
    SAMPLING_METHOD,
    VERTEX_DTYPE,
    ArtifactStatus,
    _axis_coordinates,
    _require_artwork,
    _triangles,
    endpoint_indices,
    mesh_memory_bytes,
    plan_preview_grid,
)
from app.schemas.relief import MeshSolidInfo, PhysicalHeightValidation

HEIGHT_SEPARATION_TOLERANCE_MM = 1e-9
AREA_EPSILON_MM2 = 1e-12
COPLANAR_DISTANCE_MM = 1e-12
INTERSECTION_GAP_MM = 1e-9
SELF_INTERSECTION_TRIANGLE_CAP = 128

NON_POSITIVE_DETAIL = (
    "Physical height must be greater than 1e-9 mm at every sample, including the perimeter. "
    "Non-positive height is not replaced."
)
TOPOLOGY_DETAIL = "Solid topology validation failed."
SOLID_KIND = "watertight-relief"
SOLID_COORDINATE_CONVENTION = (
    "The solid covers the artwork rectangle only. "
    "X increases left to right and Y increases top to bottom. "
    "Z increases upward. The top surface is the selected physical_height_mm sample. "
    "The flat bottom is Z = 0 mm. "
    "Column 0 and row 0 are 0 mm. The last column and last row are the artwork edges. "
    "Stock size, center offsets, and a vertical flip are not applied."
)
SOLID_WINDING = (
    "Indexed triangles use outward normals. "
    "Top cells use (row, column), (row, column + 1), (row + 1, column) and "
    "(row, column + 1), (row + 1, column + 1), (row + 1, column), which has positive Z. "
    "Bottom cells reverse that winding, which has negative Z. "
    "Each perimeter segment is two triangles whose normal points out of the artwork rectangle."
)
SOLID_MEMORY_NOTE = (
    "estimated_mesh_bytes is the float64 XYZ vertex array, the int32 triangle index array, "
    "and the float64 face-normal array. "
    "It does not include the loaded physical-height array, int8 face-role codes, "
    "or temporary edge and intersection tables used while validating a built mesh."
)
TOPOLOGY_NOT_VALIDATED = (
    "Counts and the bounding box are estimates from the selected grid. "
    "Edge incidence, winding, volume, and self-intersection checks have not been run."
)
SELF_INTERSECTION_LIMITATION = (
    "Pairwise non-adjacent triangle intersection checks run only for meshes with "
    f"at most {SELF_INTERSECTION_TRIANGLE_CAP} triangles. "
    "A passing edge-manifold result is not a universal manifold or export-ready certificate."
)

ROLE_TOP = 0
ROLE_BOTTOM = 1
ROLE_X0 = 2
ROLE_X1 = 3
ROLE_Y0 = 4
ROLE_Y1 = 5


@dataclass(frozen=True)
class SolidGridPlan:
    """Triangle counts for a solid. This plan does not allocate a mesh."""

    source_height_px: int
    source_width_px: int
    mesh_height_px: int
    mesh_width_px: int
    top_vertex_count: int
    bottom_vertex_count: int
    vertex_count: int
    top_triangle_count: int
    bottom_triangle_count: int
    sidewall_triangle_count: int
    triangle_count: int
    downsampling_required: bool
    max_vertices: int

    @property
    def estimated_mesh_bytes(self) -> int:
        return mesh_memory_bytes(self.vertex_count, self.triangle_count)


@dataclass(frozen=True)
class TopologyReport:
    """Result of the checks that actually ran. This is not an export approval."""

    passed: bool
    errors: tuple[str, ...]
    vertex_count: int
    triangle_count: int
    edge_manifold: bool
    boundary_edge_count: int
    non_manifold_edge_count: int
    inconsistent_winding_edge_count: int
    component_count: int
    signed_volume_mm3: float
    zero_area_triangle_count: int
    normals_consistent: bool
    bounding_box_mm: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    self_intersection_checked: bool
    self_intersection_count: int | None
    limitation: str


@dataclass(frozen=True)
class SolidMesh:
    """Closed relief solid. Vertices are float64 XYZ millimeters."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    face_roles: np.ndarray
    source_height_px: int
    source_width_px: int
    mesh_height_px: int
    mesh_width_px: int
    row_indices: np.ndarray
    column_indices: np.ndarray
    downsampling_required: bool
    max_vertices: int
    topology: TopologyReport
    warnings: tuple[str, ...]

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def top_vertex_count(self) -> int:
        return self.mesh_height_px * self.mesh_width_px

    @property
    def bounding_box_mm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lowest = self.vertices.min(axis=0)
        highest = self.vertices.max(axis=0)
        return (
            (float(lowest[0]), float(lowest[1]), float(lowest[2])),
            (float(highest[0]), float(highest[1]), float(highest[2])),
        )

    @property
    def signed_volume_mm3(self) -> float:
        return self.topology.signed_volume_mm3


def plan_solid_grid(
    height_px: int,
    width_px: int,
    max_vertices: int = DEFAULT_MAX_VERTICES,
) -> SolidGridPlan:
    """Choose the top grid and derive solid counts without building triangles."""

    plan = plan_preview_grid(height_px, width_px, max_vertices)
    return _solid_plan(plan)


def build_solid_mesh(
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    max_vertices: int = DEFAULT_MAX_VERTICES,
    stock_thickness_mm: float | None = None,
) -> SolidMesh:
    """Build a closed solid. The input array is not modified or clamped."""

    _require_solid_shape(array)
    _require_artwork(artwork_width_mm, artwork_height_mm)
    errors, warnings = _height_findings(array, stock_thickness_mm)
    if errors:
        raise InvalidMeshRequest(errors[0])
    height_px, width_px = int(array.shape[0]), int(array.shape[1])
    grid_plan = plan_solid_grid(height_px, width_px, max_vertices)
    if grid_plan.downsampling_required:
        warnings.append(DOWNSAMPLE_WARNING)
    rows = endpoint_indices(height_px, grid_plan.mesh_height_px)
    columns = endpoint_indices(width_px, grid_plan.mesh_width_px)
    sampled = np.asarray(array[np.ix_(rows, columns)], dtype=VERTEX_DTYPE)
    vertices = _solid_vertices(
        sampled,
        rows,
        columns,
        height_px,
        width_px,
        float(artwork_width_mm),
        float(artwork_height_mm),
    )
    triangles, roles = _solid_triangles(grid_plan.mesh_height_px, grid_plan.mesh_width_px)
    report = validate_solid_topology(
        vertices,
        triangles,
        roles,
        mesh_height_px=grid_plan.mesh_height_px,
        mesh_width_px=grid_plan.mesh_width_px,
        artwork_width_mm=float(artwork_width_mm),
        artwork_height_mm=float(artwork_height_mm),
    )
    if not report.passed:
        raise InvalidMeshRequest(TOPOLOGY_DETAIL + " " + "; ".join(report.errors))
    return SolidMesh(
        vertices=vertices,
        triangles=triangles,
        normals=_unit_normals(vertices, triangles),
        face_roles=roles,
        source_height_px=height_px,
        source_width_px=width_px,
        mesh_height_px=grid_plan.mesh_height_px,
        mesh_width_px=grid_plan.mesh_width_px,
        row_indices=rows,
        column_indices=columns,
        downsampling_required=grid_plan.downsampling_required,
        max_vertices=max_vertices,
        topology=report,
        warnings=tuple(warnings),
    )


def build_solid_on_indices(
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    stock_thickness_mm: float | None = None,
) -> SolidMesh:
    """Closed solid on explicit source samples.

    X and Y come from those source indices, not from a uniformly spaced
    output grid and not from the preview vertex budget. The input array
    is not modified.
    """

    _require_solid_shape(array)
    _require_artwork(artwork_width_mm, artwork_height_mm)
    errors, warnings = _height_findings(array, stock_thickness_mm)
    if errors:
        raise InvalidMeshRequest(errors[0])
    height_px, width_px = int(array.shape[0]), int(array.shape[1])
    rows = _require_source_indices(row_indices, height_px)
    columns = _require_source_indices(column_indices, width_px)
    sampled = np.array(array[np.ix_(rows, columns)], dtype=VERTEX_DTYPE, copy=True)
    mesh_height, mesh_width = int(rows.shape[0]), int(columns.shape[0])
    vertices = _solid_vertices(
        sampled,
        rows,
        columns,
        height_px,
        width_px,
        float(artwork_width_mm),
        float(artwork_height_mm),
    )
    triangles, roles = _solid_triangles(mesh_height, mesh_width)
    report = validate_solid_topology(
        vertices,
        triangles,
        roles,
        mesh_height_px=mesh_height,
        mesh_width_px=mesh_width,
        artwork_width_mm=float(artwork_width_mm),
        artwork_height_mm=float(artwork_height_mm),
    )
    if not report.passed:
        raise InvalidMeshRequest(TOPOLOGY_DETAIL + " " + "; ".join(report.errors))
    return SolidMesh(
        vertices=vertices,
        triangles=triangles,
        normals=_unit_normals(vertices, triangles),
        face_roles=roles,
        source_height_px=height_px,
        source_width_px=width_px,
        mesh_height_px=mesh_height,
        mesh_width_px=mesh_width,
        row_indices=rows,
        column_indices=columns,
        downsampling_required=mesh_height != height_px or mesh_width != width_px,
        max_vertices=mesh_height * mesh_width,
        topology=report,
        warnings=tuple(warnings),
    )


def describe_solid(
    *,
    project_id: UUID,
    status: ArtifactStatus,
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    max_vertices: int = DEFAULT_MAX_VERTICES,
    stock_thickness_mm: float | None = None,
) -> MeshSolidInfo:
    """Estimate a solid from the selected grid. This does not build triangles."""

    _require_solid_shape(array)
    _require_artwork(artwork_width_mm, artwork_height_mm)
    height_px, width_px = int(array.shape[0]), int(array.shape[1])
    grid_plan = plan_solid_grid(height_px, width_px, max_vertices)
    errors, warnings = _height_findings(array, stock_thickness_mm)
    if grid_plan.downsampling_required:
        warnings.append(DOWNSAMPLE_WARNING)
    finite = np.isfinite(array)
    source_min = float(array.min()) if bool(finite.all()) else None
    source_max = float(array.max()) if bool(finite.all()) else None
    selected_min, selected_max = _selected_extents(array, grid_plan)
    available = not errors
    box_min = None
    box_max = None
    if available and selected_max is not None:
        box_min = [0.0, 0.0, 0.0]
        box_max = [float(artwork_width_mm), float(artwork_height_mm), selected_max]
    return MeshSolidInfo(
        project_id=project_id,
        status=status,
        solid=SOLID_KIND,
        topology_validated=False,
        geometry_construction_available=available,
        coordinate_convention=SOLID_COORDINATE_CONVENTION,
        winding=SOLID_WINDING,
        source_width_px=width_px,
        source_height_px=height_px,
        mesh_width_px=grid_plan.mesh_width_px,
        mesh_height_px=grid_plan.mesh_height_px,
        top_vertex_count=grid_plan.top_vertex_count,
        bottom_vertex_count=grid_plan.bottom_vertex_count,
        vertex_count=grid_plan.vertex_count,
        top_triangle_count=grid_plan.top_triangle_count,
        bottom_triangle_count=grid_plan.bottom_triangle_count,
        sidewall_triangle_count=grid_plan.sidewall_triangle_count,
        triangle_count=grid_plan.triangle_count,
        artwork_width_mm=float(artwork_width_mm),
        artwork_height_mm=float(artwork_height_mm),
        bounding_box_minimum_mm=box_min,
        bounding_box_maximum_mm=box_max,
        source_z_minimum_mm=source_min,
        source_z_maximum_mm=source_max,
        selected_z_minimum_mm=selected_min,
        selected_z_maximum_mm=selected_max,
        estimated_mesh_bytes=grid_plan.estimated_mesh_bytes,
        memory_note=SOLID_MEMORY_NOTE,
        downsampling_required=grid_plan.downsampling_required,
        max_vertices=max_vertices,
        sampling_method=SAMPLING_METHOD,
        topology_note=TOPOLOGY_NOT_VALIDATED,
        validation=PhysicalHeightValidation(ok=available, errors=errors, warnings=warnings),
    )


def validate_solid_topology(
    vertices: np.ndarray,
    triangles: np.ndarray,
    face_roles: np.ndarray,
    *,
    mesh_height_px: int,
    mesh_width_px: int,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> TopologyReport:
    """Check indices, area, edge pairs, shell connectivity, volume, and normals."""

    errors: list[str] = []
    vertex_count = int(vertices.shape[0]) if getattr(vertices, "ndim", 0) == 2 else 0
    triangle_count = int(triangles.shape[0]) if getattr(triangles, "ndim", 0) == 2 else 0
    expected_vertices = 2 * mesh_height_px * mesh_width_px
    if (
        getattr(vertices, "ndim", 0) != 2
        or vertices.shape[1] != 3
        or vertex_count != expected_vertices
        or getattr(triangles, "ndim", 0) != 2
        or triangles.shape[1] != 3
        or face_roles.shape != (triangle_count,)
    ):
        errors.append("Triangle or vertex buffers do not match the solid grid.")
        return _failed_report(vertex_count, triangle_count, errors)

    if triangle_count and (
        int(triangles.min()) < 0 or int(triangles.max()) >= vertex_count
    ):
        errors.append("A triangle index is outside the vertex buffer.")
        return _failed_report(vertex_count, triangle_count, errors)
    repeated = triangle_count and (
        bool(np.any(triangles[:, 0] == triangles[:, 1]))
        or bool(np.any(triangles[:, 1] == triangles[:, 2]))
        or bool(np.any(triangles[:, 2] == triangles[:, 0]))
    )
    if repeated:
        errors.append("A triangle repeats a vertex index.")

    crossed = _cross_products(vertices, triangles)
    lengths = np.linalg.norm(crossed, axis=1)
    zero_area = int(np.count_nonzero(~np.isfinite(lengths) | (lengths <= 2.0 * AREA_EPSILON_MM2)))
    if zero_area:
        errors.append("The mesh has a zero-area triangle.")

    boundary, non_manifold, inconsistent, components = _edge_topology(triangles, vertex_count)
    if boundary or non_manifold:
        errors.append("An undirected edge does not belong to exactly two triangles.")
    if inconsistent:
        errors.append("Adjacent triangles do not traverse their shared edge in opposite directions.")
    if components != 1:
        errors.append("The mesh is not a single connected shell.")

    if not _roles_match(face_roles, mesh_height_px, mesh_width_px):
        errors.append("Face roles do not match the top, bottom, and four sidewalls.")

    normals = _normals_from_cross(crossed, lengths)
    normals_ok = _normals_have_expected_directions(normals, face_roles)
    if not normals_ok:
        errors.append("Face normals do not point outward.")

    box = _bounding_box(vertices)
    if not _box_matches(vertices, artwork_width_mm, artwork_height_mm):
        errors.append("The bounding box does not match the artwork rectangle and Z = 0 bottom.")

    volume = signed_volume_mm3(vertices, triangles)
    if not np.isfinite(volume) or volume <= 0.0:
        errors.append("Signed enclosed volume is not positive.")

    intersection_count = count_nonadjacent_intersections(vertices, triangles)
    checked = intersection_count is not None
    if intersection_count:
        errors.append("Non-adjacent triangles intersect.")

    return TopologyReport(
        passed=not errors,
        errors=tuple(errors),
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        edge_manifold=boundary == 0 and non_manifold == 0 and inconsistent == 0,
        boundary_edge_count=boundary,
        non_manifold_edge_count=non_manifold,
        inconsistent_winding_edge_count=inconsistent,
        component_count=components,
        signed_volume_mm3=volume,
        zero_area_triangle_count=zero_area,
        normals_consistent=normals_ok,
        bounding_box_mm=box,
        self_intersection_checked=checked,
        self_intersection_count=intersection_count,
        limitation=SELF_INTERSECTION_LIMITATION,
    )


def signed_volume_mm3(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Positive when triangle winding follows the outward right-hand rule."""

    if triangles.size == 0:
        return 0.0
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    return float(np.einsum("ij,ij->", first, np.cross(second, third)) / 6.0)


def count_nonadjacent_intersections(vertices: np.ndarray, triangles: np.ndarray) -> int | None:
    """Count intersecting triangle pairs that share no vertex.

    Returns None when the mesh is larger than the explicit check cap.
    """

    triangle_count = int(triangles.shape[0])
    if triangle_count > SELF_INTERSECTION_TRIANGLE_CAP:
        return None
    hits = 0
    points = vertices[triangles]
    for left in range(triangle_count):
        for right in range(left + 1, triangle_count):
            if np.intersect1d(triangles[left], triangles[right]).size:
                continue
            if _aabb_separated(points[left], points[right]):
                continue
            if _triangles_intersect(points[left], points[right]):
                hits += 1
    return hits


def _solid_plan(plan) -> SolidGridPlan:
    height = plan.mesh_height_px
    width = plan.mesh_width_px
    top_vertices = plan.vertex_count
    top_triangles = plan.triangle_count
    side_triangles = 4 * (width - 1) + 4 * (height - 1)
    return SolidGridPlan(
        source_height_px=plan.source_height_px,
        source_width_px=plan.source_width_px,
        mesh_height_px=height,
        mesh_width_px=width,
        top_vertex_count=top_vertices,
        bottom_vertex_count=top_vertices,
        vertex_count=2 * top_vertices,
        top_triangle_count=top_triangles,
        bottom_triangle_count=top_triangles,
        sidewall_triangle_count=side_triangles,
        triangle_count=2 * top_triangles + side_triangles,
        downsampling_required=plan.downsampling_required,
        max_vertices=plan.max_vertices,
    )


def _require_source_indices(indices: np.ndarray, length: int) -> np.ndarray:
    """Keep a strictly increasing source selection that includes both endpoints."""

    chosen = np.asarray(indices, dtype=np.int64)
    if chosen.ndim != 1 or int(chosen.shape[0]) < MIN_GRID:
        raise InvalidMeshRequest(AXIS_DETAIL)
    if int(chosen[0]) != 0 or int(chosen[-1]) != length - 1:
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    if bool(np.any(chosen < 0)) or bool(np.any(chosen >= length)) or bool(np.any(np.diff(chosen) <= 0)):
        raise InvalidMeshRequest(DEGENERATE_DETAIL)
    return chosen


def _require_solid_shape(array: np.ndarray) -> None:
    numeric = isinstance(array, np.ndarray) and (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    )
    if not numeric or array.ndim != 2 or array.size == 0:
        raise InvalidMeshRequest(STRUCTURE_DETAIL)
    if int(array.shape[0]) < MIN_GRID or int(array.shape[1]) < MIN_GRID:
        raise InvalidMeshRequest(AXIS_DETAIL)


def _height_findings(
    array: np.ndarray,
    stock_thickness_mm: float | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if array.dtype != np.float32:
        warnings.append(DTYPE_WARNING)
    finite = np.isfinite(array)
    if not bool(finite.all()):
        errors.append(NON_FINITE_DETAIL)
    elif bool(np.any(np.asarray(array, dtype=np.float64) <= HEIGHT_SEPARATION_TOLERANCE_MM)):
        errors.append(NON_POSITIVE_DETAIL)
    if stock_thickness_mm is not None and bool(finite.any()):
        if float(np.max(array[finite])) > float(stock_thickness_mm):
            warnings.append(ABOVE_STOCK_WARNING)
    return errors, warnings


def _selected_extents(array: np.ndarray, plan: SolidGridPlan) -> tuple[float | None, float | None]:
    if not bool(np.isfinite(array).all()):
        return None, None
    rows = endpoint_indices(plan.source_height_px, plan.mesh_height_px)
    columns = endpoint_indices(plan.source_width_px, plan.mesh_width_px)
    sampled = np.asarray(array[np.ix_(rows, columns)], dtype=np.float64)
    return float(sampled.min()), float(sampled.max())


def _solid_vertices(
    sampled: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    source_height: int,
    source_width: int,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> np.ndarray:
    mesh_height, mesh_width = sampled.shape
    xs = _axis_coordinates(columns, source_width, artwork_width_mm)
    ys = _axis_coordinates(rows, source_height, artwork_height_mm)
    count = mesh_height * mesh_width
    vertices = np.empty((2 * count, 3), dtype=VERTEX_DTYPE)
    top = vertices[:count].reshape(mesh_height, mesh_width, 3)
    top[:, :, 0] = xs
    top[:, :, 1] = ys[:, None]
    top[:, :, 2] = sampled
    vertices[count:, 0] = vertices[:count, 0]
    vertices[count:, 1] = vertices[:count, 1]
    vertices[count:, 2] = 0.0
    return vertices


def _solid_triangles(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    top = _triangles(height, width)
    offset = np.int64(height * width)
    bottom = np.empty_like(top)
    bottom[:, 0] = top[:, 0].astype(np.int64) + offset
    bottom[:, 1] = top[:, 2].astype(np.int64) + offset
    bottom[:, 2] = top[:, 1].astype(np.int64) + offset
    sides, side_roles = _sidewall_triangles(height, width, offset)
    triangles = np.concatenate((top, bottom, sides), axis=0)
    roles = np.concatenate(
        (
            np.full(top.shape[0], ROLE_TOP, dtype=np.int8),
            np.full(bottom.shape[0], ROLE_BOTTOM, dtype=np.int8),
            side_roles,
        )
    )
    return triangles, roles


def _sidewall_triangles(height: int, width: int, offset: np.int64) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(height - 1, dtype=np.int64)
    left_top = rows * width
    left_next = left_top + width
    left = _wall_pairs(left_top, left_next, left_top + offset, left_next + offset, "x0")

    right_top = rows * width + (width - 1)
    right_next = right_top + width
    right = _wall_pairs(right_top, right_next, right_top + offset, right_next + offset, "x1")

    columns = np.arange(width - 1, dtype=np.int64)
    front = _wall_pairs(columns, columns + 1, columns + offset, columns + 1 + offset, "y0")

    back_top = (height - 1) * width + columns
    back = _wall_pairs(back_top, back_top + 1, back_top + offset, back_top + 1 + offset, "y1")

    triangles = np.concatenate((left[0], right[0], front[0], back[0]), axis=0).astype(INDEX_DTYPE, copy=False)
    roles = np.concatenate((left[1], right[1], front[1], back[1]))
    return triangles, roles


def _wall_pairs(
    top_a: np.ndarray,
    top_b: np.ndarray,
    bottom_a: np.ndarray,
    bottom_b: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(top_a.shape[0])
    pairs = np.empty((count, 2, 3), dtype=np.int64)
    if side == "x0":
        pairs[:, 0, 0] = top_a
        pairs[:, 0, 1] = top_b
        pairs[:, 0, 2] = bottom_a
        pairs[:, 1, 0] = top_b
        pairs[:, 1, 1] = bottom_b
        pairs[:, 1, 2] = bottom_a
        role = ROLE_X0
    elif side == "x1":
        pairs[:, 0, 0] = top_a
        pairs[:, 0, 1] = bottom_a
        pairs[:, 0, 2] = top_b
        pairs[:, 1, 0] = top_b
        pairs[:, 1, 1] = bottom_a
        pairs[:, 1, 2] = bottom_b
        role = ROLE_X1
    elif side == "y0":
        pairs[:, 0, 0] = top_a
        pairs[:, 0, 1] = bottom_a
        pairs[:, 0, 2] = top_b
        pairs[:, 1, 0] = top_b
        pairs[:, 1, 1] = bottom_a
        pairs[:, 1, 2] = bottom_b
        role = ROLE_Y0
    elif side == "y1":
        pairs[:, 0, 0] = top_a
        pairs[:, 0, 1] = top_b
        pairs[:, 0, 2] = bottom_a
        pairs[:, 1, 0] = top_b
        pairs[:, 1, 1] = bottom_b
        pairs[:, 1, 2] = bottom_a
        role = ROLE_Y1
    else:
        raise InvalidMeshRequest(TOPOLOGY_DETAIL)
    return pairs.reshape(-1, 3), np.full(count * 2, role, dtype=np.int8)


def _cross_products(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    return np.cross(second - first, third - first)


def _normals_from_cross(crossed: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(crossed)
    usable = np.isfinite(lengths) & (lengths > 2.0 * AREA_EPSILON_MM2)
    normals[usable] = crossed[usable] / lengths[usable, None]
    return normals


def _unit_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    crossed = _cross_products(vertices, triangles)
    lengths = np.linalg.norm(crossed, axis=1)
    if lengths.size == 0 or not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 2.0 * AREA_EPSILON_MM2)):
        raise InvalidMeshRequest(TOPOLOGY_DETAIL)
    return crossed / lengths[:, None]


def _roles_match(face_roles: np.ndarray, height: int, width: int) -> bool:
    expected = {
        ROLE_TOP: 2 * (height - 1) * (width - 1),
        ROLE_BOTTOM: 2 * (height - 1) * (width - 1),
        ROLE_X0: 2 * (height - 1),
        ROLE_X1: 2 * (height - 1),
        ROLE_Y0: 2 * (width - 1),
        ROLE_Y1: 2 * (width - 1),
    }
    for role, count in expected.items():
        if int(np.count_nonzero(face_roles == role)) != count:
            return False
    return int(face_roles.size) == sum(expected.values())


def _normals_have_expected_directions(normals: np.ndarray, face_roles: np.ndarray) -> bool:
    checks = (
        (ROLE_TOP, 2, 1.0),
        (ROLE_BOTTOM, 2, -1.0),
        (ROLE_X0, 0, -1.0),
        (ROLE_X1, 0, 1.0),
        (ROLE_Y0, 1, -1.0),
        (ROLE_Y1, 1, 1.0),
    )
    for role, axis, sign in checks:
        chosen = normals[face_roles == role]
        if chosen.size == 0 or bool(np.any(chosen[:, axis] * sign <= 0.0)):
            return False
    return True


def _bounding_box(
    vertices: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lowest = vertices.min(axis=0)
    highest = vertices.max(axis=0)
    return (
        (float(lowest[0]), float(lowest[1]), float(lowest[2])),
        (float(highest[0]), float(highest[1]), float(highest[2])),
    )


def _box_matches(vertices: np.ndarray, artwork_width_mm: float, artwork_height_mm: float) -> bool:
    top_count = vertices.shape[0] // 2
    top = vertices[:top_count]
    bottom = vertices[top_count:]
    if not np.array_equal(bottom[:, :2], top[:, :2]):
        return False
    if not bool(np.all(bottom[:, 2] == 0.0)):
        return False
    if not bool(np.all(top[:, 2] > HEIGHT_SEPARATION_TOLERANCE_MM)):
        return False
    lowest = vertices.min(axis=0)
    highest = vertices.max(axis=0)
    return bool(
        np.isclose(lowest[0], 0.0, rtol=0.0, atol=1e-9)
        and np.isclose(lowest[1], 0.0, rtol=0.0, atol=1e-9)
        and np.isclose(lowest[2], 0.0, rtol=0.0, atol=1e-9)
        and np.isclose(highest[0], artwork_width_mm, rtol=0.0, atol=1e-9)
        and np.isclose(highest[1], artwork_height_mm, rtol=0.0, atol=1e-9)
        and highest[2] > HEIGHT_SEPARATION_TOLERANCE_MM
    )


def _edge_topology(triangles: np.ndarray, vertex_count: int) -> tuple[int, int, int, int]:
    tri64 = triangles.astype(np.int64, copy=False)
    pairs = np.stack((tri64[:, [0, 1]], tri64[:, [1, 2]], tri64[:, [2, 0]]), axis=1).reshape(-1, 2)
    signs = np.where(pairs[:, 0] < pairs[:, 1], np.int8(1), np.int8(-1))
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    key = low * np.int64(vertex_count) + high
    owners = np.repeat(np.arange(tri64.shape[0], dtype=np.int64), 3)
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    signs = signs[order]
    owners = owners[order]
    if key.size == 0:
        return 0, 0, 0, 0
    splits = np.flatnonzero(key[1:] != key[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), splits))
    ends = np.concatenate((splits, np.array([key.size], dtype=np.int64)))
    counts = ends - starts
    boundary = int(np.count_nonzero(counts == 1))
    non_manifold = int(np.count_nonzero(counts > 2))
    paired = starts[counts == 2]
    if paired.size:
        sign_sum = signs[paired].astype(np.int16) + signs[paired + 1].astype(np.int16)
        inconsistent = int(np.count_nonzero(sign_sum != 0))
    else:
        inconsistent = 0
    components = _component_count(int(tri64.shape[0]), owners, paired)
    return boundary, non_manifold, inconsistent, components


def _component_count(triangle_count: int, owners: np.ndarray, paired_starts: np.ndarray) -> int:
    parent = np.arange(triangle_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for start in paired_starts:
        left = find(int(owners[int(start)]))
        right = find(int(owners[int(start) + 1]))
        if left != right:
            parent[right] = left
    return len({find(index) for index in range(triangle_count)})


def _failed_report(vertex_count: int, triangle_count: int, errors: list[str]) -> TopologyReport:
    return TopologyReport(
        passed=False,
        errors=tuple(errors),
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        edge_manifold=False,
        boundary_edge_count=0,
        non_manifold_edge_count=0,
        inconsistent_winding_edge_count=0,
        component_count=0,
        signed_volume_mm3=0.0,
        zero_area_triangle_count=0,
        normals_consistent=False,
        bounding_box_mm=None,
        self_intersection_checked=False,
        self_intersection_count=None,
        limitation=SELF_INTERSECTION_LIMITATION,
    )


def _aabb_separated(left: np.ndarray, right: np.ndarray) -> bool:
    for axis in range(3):
        if float(left[:, axis].max()) < float(right[:, axis].min()) - INTERSECTION_GAP_MM:
            return True
        if float(right[:, axis].max()) < float(left[:, axis].min()) - INTERSECTION_GAP_MM:
            return True
    return False


def _triangles_intersect(left: np.ndarray, right: np.ndarray) -> bool:
    left_distances = _plane_distances(left, right)
    right_distances = _plane_distances(right, left)
    if left_distances is None or right_distances is None:
        return False
    if bool(np.all(left_distances == 0.0)):
        return _coplanar_overlap(left, right)
    if bool(np.all(left_distances > 0.0) or np.all(left_distances < 0.0)):
        return False
    if bool(np.all(right_distances > 0.0) or np.all(right_distances < 0.0)):
        return False
    left_normal = _unit_plane_normal(left)
    right_normal = _unit_plane_normal(right)
    if left_normal is None or right_normal is None:
        return False
    direction = np.cross(left_normal, right_normal)
    scale = float(np.linalg.norm(direction))
    if scale <= COPLANAR_DISTANCE_MM:
        return False
    direction /= scale
    left_span = _intersection_span(right, left_distances, direction)
    right_span = _intersection_span(left, right_distances, direction)
    if left_span is None or right_span is None:
        return False
    return max(left_span[0], right_span[0]) <= min(left_span[1], right_span[1]) + INTERSECTION_GAP_MM


def _unit_plane_normal(triangle: np.ndarray) -> np.ndarray | None:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    length = float(np.linalg.norm(normal))
    if length <= 2.0 * AREA_EPSILON_MM2:
        return None
    return normal / length


def _plane_distances(plane: np.ndarray, other: np.ndarray) -> np.ndarray | None:
    normal = _unit_plane_normal(plane)
    if normal is None:
        return None
    distances = (other - plane[0]) @ normal
    distances[np.abs(distances) <= COPLANAR_DISTANCE_MM] = 0.0
    return distances


def _intersection_span(
    triangle: np.ndarray,
    distances: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, float] | None:
    projected: list[float] = []
    for index in range(3):
        nxt = (index + 1) % 3
        start = float(distances[index])
        end = float(distances[nxt])
        if start == 0.0:
            projected.append(float(np.dot(triangle[index], direction)))
        if start * end < 0.0:
            weight = start / (start - end)
            point = triangle[index] + weight * (triangle[nxt] - triangle[index])
            projected.append(float(np.dot(point, direction)))
    if not projected:
        return None
    if len(projected) == 1:
        return projected[0], projected[0]
    return min(projected), max(projected)


def _coplanar_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    normal = np.cross(left[1] - left[0], left[2] - left[0])
    drop = int(np.argmax(np.abs(normal)))
    keep = [axis for axis in (0, 1, 2) if axis != drop]
    left_2d = left[:, keep]
    right_2d = right[:, keep]
    for i in range(3):
        for j in range(3):
            if _segments_intersect(left_2d[i], left_2d[(i + 1) % 3], right_2d[j], right_2d[(j + 1) % 3]):
                return True
    return any(_point_in_triangle(point, right_2d) for point in left_2d) or any(
        _point_in_triangle(point, left_2d) for point in right_2d
    )


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = _near_zero(_orient(a, b, c))
    o2 = _near_zero(_orient(a, b, d))
    o3 = _near_zero(_orient(c, d, a))
    o4 = _near_zero(_orient(c, d, b))
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if o1 == 0.0 and _on_segment(a, b, c):
        return True
    if o2 == 0.0 and _on_segment(a, b, d):
        return True
    if o3 == 0.0 and _on_segment(c, d, a):
        return True
    if o4 == 0.0 and _on_segment(c, d, b):
        return True
    return False


def _near_zero(value: float) -> float:
    return 0.0 if abs(value) <= 1e-12 else value


def _on_segment(a: np.ndarray, b: np.ndarray, point: np.ndarray) -> bool:
    return bool(
        min(float(a[0]), float(b[0])) - INTERSECTION_GAP_MM
        <= float(point[0])
        <= max(float(a[0]), float(b[0])) + INTERSECTION_GAP_MM
        and min(float(a[1]), float(b[1])) - INTERSECTION_GAP_MM
        <= float(point[1])
        <= max(float(a[1]), float(b[1])) + INTERSECTION_GAP_MM
    )


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    o1 = _near_zero(_orient(triangle[0], triangle[1], point))
    o2 = _near_zero(_orient(triangle[1], triangle[2], point))
    o3 = _near_zero(_orient(triangle[2], triangle[0], point))
    return (o1 >= 0.0 and o2 >= 0.0 and o3 >= 0.0) or (o1 <= 0.0 and o2 <= 0.0 and o3 <= 0.0)
