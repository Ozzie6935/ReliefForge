"""Validated binary STL for one Current physical-height artifact.

Binary STL has no reliable unit field. Relief Forge writes millimeters.
The header identifies this file and states that it is not a machining
certificate. It does not contain a timestamp or a personal name.

Coordinates match the export-readiness planner. A selected source column
c is placed at c * artwork_width_mm / (source_width - 1), and a selected
source row r at r * artwork_height_mm / (source_height - 1). Intermediate
samples are not moved onto a uniformly spaced output grid. The preview
vertex budget is not used, and native resolution is not reduced here.

The saved physical-height array is not modified. Nothing is written into
the project directory.
"""

import struct
from dataclasses import dataclass

import numpy as np

from app.projects.errors import (
    ArtifactDependencyRejected,
    InvalidExportRequest,
    InvalidMeshRequest,
    PhysicalHeightNotFound,
)
from app.projects.storage import is_safe_slug
from app.relief.diagnostics import ABOVE_STOCK_WARNING
from app.relief.export_plan import (
    ABOVE_STOCK_DETAIL,
    DEFAULT_EXPORT_LIMITS,
    EXPORT_MEMORY_HEADROOM_BYTES,
    NATIVE_REJECTS_GRID,
    NON_FINITE_DETAIL,
    NON_POSITIVE_DETAIL,
    ExportMode,
    ExportResourceLimits,
    binary_stl_bytes,
    estimated_peak_memory_bytes,
    solid_triangle_count,
    solid_vertex_count,
    validate_reduced_grid,
)
from app.relief.mesh import endpoint_indices
from app.relief.solid import (
    AREA_EPSILON_MM2,
    HEIGHT_SEPARATION_TOLERANCE_MM,
    SELF_INTERSECTION_LIMITATION,
    TOPOLOGY_DETAIL,
    SolidMesh,
    build_solid_on_indices,
)

STL_HEADER_TEXT = b"Relief Forge binary STL, millimeters. Not a machining certificate."
STL_PACK_CHUNK = 4096
FLOAT32_ABSOLUTE_TOLERANCE_MM = 1e-4
FLOAT32_RELATIVE_TOLERANCE = 1e-5

FLOAT32_OVERFLOW_DETAIL = "STL coordinates overflow float32. The export was not written."
FLOAT32_QUANTIZATION_DETAIL = (
    "STL coordinates do not round-trip through float32. The export was not written."
)
FLOAT32_DEGENERATE_DETAIL = (
    "Float32 rounding introduced a degenerate STL triangle. The export was not written."
)
FLOAT32_NORMAL_DETAIL = (
    "Float32 rounding reversed an outward STL normal. The export was not written."
)
SERIALIZATION_DETAIL = "Binary STL serialization failed. No STL was returned."
GRID_MISMATCH_DETAIL = "The built solid does not match the planned export grid."
NOT_READY_DETAIL = "Export readiness did not pass. No STL was written."

STL_RECORD = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ]
)

_ROLE_DIRECTION = (
    (0, 2, 1.0),
    (1, 2, -1.0),
    (2, 0, -1.0),
    (3, 0, 1.0),
    (4, 1, -1.0),
    (5, 1, 1.0),
)


@dataclass(frozen=True)
class StlDocument:
    """One validated STL buffer. The array owns the bytes; the view does not copy them."""

    payload: np.ndarray
    triangle_count: int
    mode: ExportMode
    grid_height_px: int
    grid_width_px: int
    self_intersection_checked: bool
    limitation: str

    def __post_init__(self) -> None:
        if self.payload.dtype != np.uint8 or not self.payload.flags.c_contiguous:
            raise InvalidExportRequest(SERIALIZATION_DETAIL)

    @property
    def body(self) -> memoryview:
        return memoryview(self.payload)


@dataclass(frozen=True)
class StlDownload:
    """In-memory STL response. The project directory is not a destination."""

    document: StlDocument
    filename: str

    def headers(self) -> dict[str, str]:
        checked = "true" if self.document.self_intersection_checked else "false"
        return {
            "Content-Disposition": f'attachment; filename="{self.filename}"',
            "X-Relief-Stl-Units": "mm",
            "X-Relief-Export-Certificate": "none",
            "X-Relief-Self-Intersection-Checked": checked,
            "X-Relief-Topology-Limitation": self.document.limitation,
            "Cache-Control": "no-store",
        }


def raise_blocked_export(report) -> None:
    """Refuse an STL using the first blocking readiness result. This does not write a file."""

    if not report.blocking_errors:
        raise InvalidExportRequest(NOT_READY_DETAIL)
    issue = report.blocking_errors[0]
    if issue.code in {"physical_missing", "physical_file_missing"}:
        raise PhysicalHeightNotFound(issue.message)
    if issue.code in {"provenance_stale", "provenance_unknown"}:
        raise ArtifactDependencyRejected(issue.message)
    raise InvalidExportRequest(issue.message)


def stl_filename(slug: str) -> str:
    """Attachment name from the project slug. Unsafe characters are not copied through."""

    if is_safe_slug(slug):
        return f"{slug}-relief.stl"
    return "relief-export.stl"


def stl_header() -> bytes:
    """80 deterministic ASCII bytes. Unused bytes are NUL, not a timestamp."""

    if len(STL_HEADER_TEXT) > 80:
        raise InvalidExportRequest(SERIALIZATION_DETAIL)
    return STL_HEADER_TEXT + b"\0" * (80 - len(STL_HEADER_TEXT))


def assert_export_budget(
    source_height_px: int,
    source_width_px: int,
    export_height_px: int,
    export_width_px: int,
    source_bytes: int,
    limits: ExportResourceLimits = DEFAULT_EXPORT_LIMITS,
) -> tuple[int, int, int]:
    """Reject an over-limit grid before any solid array is allocated."""

    vertices = solid_vertex_count(export_height_px, export_width_px)
    triangles = solid_triangle_count(export_height_px, export_width_px)
    stl_bytes = binary_stl_bytes(triangles)
    peak = estimated_peak_memory_bytes(
        source_height_px * source_width_px,
        vertices,
        triangles,
        source_bytes,
    )
    if triangles > limits.max_triangles:
        raise InvalidExportRequest(
            _limit_message("triangle count", triangles, limits.max_triangles)
        )
    if stl_bytes > limits.max_stl_bytes:
        raise InvalidExportRequest(
            _limit_message("binary STL size", stl_bytes, limits.max_stl_bytes)
        )
    if peak > limits.max_peak_memory_bytes:
        raise InvalidExportRequest(
            _limit_message("peak memory", peak, limits.max_peak_memory_bytes)
        )
    return vertices, triangles, stl_bytes


def generate_binary_stl(
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    *,
    stock_thickness_mm: float,
    mode: ExportMode,
    export_height_px: int | None = None,
    export_width_px: int | None = None,
    limits: ExportResourceLimits = DEFAULT_EXPORT_LIMITS,
) -> StlDocument:
    """Build, validate, and serialize one STL. The input array is not modified."""

    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise InvalidExportRequest(NON_FINITE_DETAIL)
    source_height, source_width = int(array.shape[0]), int(array.shape[1])
    export_height, export_width = _export_shape(
        source_height,
        source_width,
        mode,
        export_height_px,
        export_width_px,
    )
    vertices, triangles, stl_bytes = assert_export_budget(
        source_height,
        source_width,
        export_height,
        export_width,
        int(array.nbytes),
        limits,
    )
    _require_export_heights(array, stock_thickness_mm)
    rows = endpoint_indices(source_height, export_height)
    columns = endpoint_indices(source_width, export_width)
    try:
        mesh = build_solid_on_indices(
            array,
            artwork_width_mm,
            artwork_height_mm,
            rows,
            columns,
            stock_thickness_mm=float(stock_thickness_mm),
        )
    except InvalidMeshRequest as exc:
        raise InvalidExportRequest(str(exc)) from exc
    if ABOVE_STOCK_WARNING in mesh.warnings:
        raise InvalidExportRequest(ABOVE_STOCK_DETAIL)
    if not mesh.topology.passed:
        raise InvalidExportRequest(TOPOLOGY_DETAIL)
    if mesh.vertex_count != vertices or mesh.triangle_count != triangles:
        raise InvalidExportRequest(GRID_MISMATCH_DETAIL)
    if mesh.mesh_height_px != export_height or mesh.mesh_width_px != export_width:
        raise InvalidExportRequest(GRID_MISMATCH_DETAIL)
    actual = (
        int(array.nbytes)
        + int(mesh.vertices.nbytes)
        + int(mesh.triangles.nbytes)
        + int(mesh.normals.nbytes)
        + int(mesh.face_roles.nbytes)
        + stl_bytes
        + EXPORT_MEMORY_HEADROOM_BYTES
    )
    if actual > limits.max_peak_memory_bytes:
        raise InvalidExportRequest(
            _limit_message("peak memory", actual, limits.max_peak_memory_bytes)
        )
    payload = _serialize_mesh(
        mesh,
        float(artwork_width_mm),
        float(artwork_height_mm),
    )
    return StlDocument(
        payload=payload,
        triangle_count=mesh.triangle_count,
        mode=mode,
        grid_height_px=mesh.mesh_height_px,
        grid_width_px=mesh.mesh_width_px,
        self_intersection_checked=mesh.topology.self_intersection_checked,
        limitation=SELF_INTERSECTION_LIMITATION,
    )


def require_float32_coordinates(
    values: np.ndarray,
    *,
    absolute_tolerance_mm: float = FLOAT32_ABSOLUTE_TOLERANCE_MM,
    relative_tolerance: float = FLOAT32_RELATIVE_TOLERANCE,
) -> np.ndarray:
    """Little-endian float32 copy, or a refusal when the conversion is unsafe."""

    original = np.asarray(values, dtype=np.float64)
    if original.size == 0 or not bool(np.isfinite(original).all()):
        raise InvalidExportRequest(FLOAT32_OVERFLOW_DETAIL)
    with np.errstate(over="ignore"):
        converted = np.ascontiguousarray(original, dtype="<f4")
    if not bool(np.isfinite(converted).all()):
        raise InvalidExportRequest(FLOAT32_OVERFLOW_DETAIL)
    restored = converted.astype(np.float64)
    limit = np.maximum(absolute_tolerance_mm, np.abs(original) * relative_tolerance)
    if bool(np.any(np.abs(restored - original) > limit)):
        raise InvalidExportRequest(FLOAT32_QUANTIZATION_DETAIL)
    return converted


def _export_shape(
    source_height: int,
    source_width: int,
    mode: ExportMode,
    export_height: int | None,
    export_width: int | None,
) -> tuple[int, int]:
    if mode == "native":
        if export_height is not None or export_width is not None:
            raise InvalidExportRequest(NATIVE_REJECTS_GRID)
        return source_height, source_width
    if export_height is None or export_width is None:
        raise InvalidExportRequest("Reduced export requires both height_px and width_px.")
    validate_reduced_grid(source_height, source_width, export_height, export_width)
    return export_height, export_width


def _require_export_heights(array: np.ndarray, stock_thickness_mm: float) -> None:
    if not bool(np.isfinite(array).all()):
        raise InvalidExportRequest(NON_FINITE_DETAIL)
    if bool(np.any(array <= HEIGHT_SEPARATION_TOLERANCE_MM)):
        raise InvalidExportRequest(NON_POSITIVE_DETAIL)
    if not np.isfinite(stock_thickness_mm) or bool(np.any(array > float(stock_thickness_mm))):
        raise InvalidExportRequest(ABOVE_STOCK_DETAIL)


def _limit_message(kind: str, estimate: int, limit: int) -> str:
    return (
        f"Estimated {kind} {estimate} exceeds the export planning limit {limit}. "
        "Native resolution is not reduced automatically."
    )


def _serialize_mesh(
    mesh: SolidMesh,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> np.ndarray:
    count = mesh.triangle_count
    size = binary_stl_bytes(count)
    payload = np.zeros(size, dtype=np.uint8)
    header = stl_header()
    payload[:80] = np.frombuffer(header, dtype=np.uint8)
    payload[80:84] = np.frombuffer(struct.pack("<I", count), dtype=np.uint8)
    offset = 84
    triangles = mesh.triangles
    roles = mesh.face_roles
    vertices = mesh.vertices
    _require_indices(triangles, mesh.vertex_count)
    for start in range(0, count, STL_PACK_CHUNK):
        end = min(start + STL_PACK_CHUNK, count)
        corners = require_float32_coordinates(vertices[triangles[start:end]])
        if corners.shape != (end - start, 3, 3):
            raise InvalidExportRequest(SERIALIZATION_DETAIL)
        _reject_degenerate_float32(corners)
        normals = _outward_float32_normals(corners, roles[start:end])
        block = np.zeros(end - start, dtype=STL_RECORD)
        block["normal"] = normals
        block["v0"] = corners[:, 0]
        block["v1"] = corners[:, 1]
        block["v2"] = corners[:, 2]
        raw = block.tobytes()
        payload[offset : offset + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        offset += len(raw)
    if offset != size:
        raise InvalidExportRequest(SERIALIZATION_DETAIL)
    _audit_serialized(payload, count, artwork_width_mm, artwork_height_mm)
    return payload


def _require_indices(triangles: np.ndarray, vertex_count: int) -> None:
    chosen = np.asarray(triangles)
    if chosen.ndim != 2 or chosen.shape[1] != 3 or chosen.shape[0] == 0:
        raise InvalidExportRequest(TOPOLOGY_DETAIL)
    if int(chosen.min()) < 0 or int(chosen.max()) >= vertex_count:
        raise InvalidExportRequest(TOPOLOGY_DETAIL)
    if (
        bool(np.any(chosen[:, 0] == chosen[:, 1]))
        or bool(np.any(chosen[:, 1] == chosen[:, 2]))
        or bool(np.any(chosen[:, 2] == chosen[:, 0]))
    ):
        raise InvalidExportRequest(TOPOLOGY_DETAIL)


def _reject_degenerate_float32(corners: np.ndarray) -> None:
    first = corners[:, 1].astype(np.float64) - corners[:, 0].astype(np.float64)
    second = corners[:, 2].astype(np.float64) - corners[:, 0].astype(np.float64)
    lengths = np.linalg.norm(np.cross(first, second), axis=1)
    if not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 2.0 * AREA_EPSILON_MM2)):
        raise InvalidExportRequest(FLOAT32_DEGENERATE_DETAIL)


def _outward_float32_normals(corners: np.ndarray, roles: np.ndarray) -> np.ndarray:
    first = corners[:, 1].astype(np.float64) - corners[:, 0].astype(np.float64)
    second = corners[:, 2].astype(np.float64) - corners[:, 0].astype(np.float64)
    crossed = np.cross(first, second)
    lengths = np.linalg.norm(crossed, axis=1)
    if not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 2.0 * AREA_EPSILON_MM2)):
        raise InvalidExportRequest(FLOAT32_DEGENERATE_DETAIL)
    normals = crossed / lengths[:, None]
    for role, axis, sign in _ROLE_DIRECTION:
        chosen = normals[np.asarray(roles) == role]
        if chosen.size and bool(np.any(chosen[:, axis] * sign <= 0.0)):
            raise InvalidExportRequest(FLOAT32_NORMAL_DETAIL)
    with np.errstate(over="ignore"):
        stored = np.ascontiguousarray(normals, dtype="<f4")
    if not bool(np.isfinite(stored).all()):
        raise InvalidExportRequest(FLOAT32_OVERFLOW_DETAIL)
    return stored


def _audit_serialized(
    payload: np.ndarray,
    count: int,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> None:
    """Re-read the STL bytes and refuse a file that the writer cannot stand behind."""

    if int(payload.shape[0]) != binary_stl_bytes(count) or int(STL_RECORD.itemsize) != 50:
        raise InvalidExportRequest(SERIALIZATION_DETAIL)
    header = bytes(payload[:80])
    if header != stl_header() or struct.unpack_from("<I", payload, 80)[0] != count:
        raise InvalidExportRequest(SERIALIZATION_DETAIL)
    records = np.frombuffer(payload, dtype=STL_RECORD, count=count, offset=84)
    if not bool(np.all(records["attr"] == 0)):
        raise InvalidExportRequest(SERIALIZATION_DETAIL)
    lowest = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    highest = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    volume = 0.0
    for start in range(0, count, STL_PACK_CHUNK):
        part = records[start : min(start + STL_PACK_CHUNK, count)]
        v0 = part["v0"].astype(np.float64)
        v1 = part["v1"].astype(np.float64)
        v2 = part["v2"].astype(np.float64)
        stored = part["normal"].astype(np.float64)
        if not (
            bool(np.isfinite(v0).all())
            and bool(np.isfinite(v1).all())
            and bool(np.isfinite(v2).all())
            and bool(np.isfinite(stored).all())
        ):
            raise InvalidExportRequest(FLOAT32_OVERFLOW_DETAIL)
        crossed = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(crossed, axis=1)
        if not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 2.0 * AREA_EPSILON_MM2)):
            raise InvalidExportRequest(FLOAT32_DEGENERATE_DETAIL)
        unit = crossed / lengths[:, None]
        if bool(np.any(np.sum(unit * stored, axis=1) <= 0.0)):
            raise InvalidExportRequest(FLOAT32_NORMAL_DETAIL)
        volume += float(np.einsum("ij,ij->", v0, crossed) / 6.0)
        for corners in (v0, v1, v2):
            lowest = np.minimum(lowest, corners.min(axis=0))
            highest = np.maximum(highest, corners.max(axis=0))
    if not np.isfinite(volume) or volume <= 0.0:
        raise InvalidExportRequest(TOPOLOGY_DETAIL)
    width_limit = max(FLOAT32_ABSOLUTE_TOLERANCE_MM, abs(artwork_width_mm) * FLOAT32_RELATIVE_TOLERANCE)
    height_limit = max(
        FLOAT32_ABSOLUTE_TOLERANCE_MM,
        abs(artwork_height_mm) * FLOAT32_RELATIVE_TOLERANCE,
    )
    if not (
        abs(float(lowest[0])) <= FLOAT32_ABSOLUTE_TOLERANCE_MM
        and abs(float(lowest[1])) <= FLOAT32_ABSOLUTE_TOLERANCE_MM
        and abs(float(lowest[2])) <= FLOAT32_ABSOLUTE_TOLERANCE_MM
        and abs(float(highest[0]) - artwork_width_mm) <= width_limit
        and abs(float(highest[1]) - artwork_height_mm) <= height_limit
        and float(highest[2]) > HEIGHT_SEPARATION_TOLERANCE_MM
    ):
        raise InvalidExportRequest(SERIALIZATION_DETAIL)
