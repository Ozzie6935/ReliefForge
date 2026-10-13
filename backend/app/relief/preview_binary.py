"""Little-endian binary preview of one closed relief solid.

The geometry is built once by build_solid_mesh. This module only checks
that the float64 mesh can be stored as float32 and uint32, then packs a
fixed header plus three typed arrays. It does not write a project file.

Browser decoding, little-endian:
- bytes 0:3 are the ASCII magic RFMP
- uint32 at byte 4 is format version 1
- uint32 at byte 8 is the endian marker 1 (on disk as 01 00 00 00)
- header_bytes at byte 12 is 256
- uint32 vertex_count at byte 16, triangle_count at byte 20
- Float32Array positions at positions_offset (byte 92), length positions_bytes / 4
  Order is xyz, xyz, ... in millimeters. X increases left to right,
  Y increases top to bottom, Z increases upward. Do not swap Y.
- Uint32Array indices at indices_offset (byte 100), length indices_bytes / 4
  Order is i0,i1,i2, i0,i1,i2, ... with the solid's outward winding.
  Use the indices as-is. Do not reverse triangle order.
- Float32Array face normals at normals_offset (byte 108), length normals_bytes / 4
  One xyz unit normal per triangle, in the same order as the index triples.
  These are face normals, not vertex normals. Do not bind them as a
  vertex normal attribute on indexed BufferGeometry.
- export_ready at byte 124 is always 0. A closed preview is not export-ready.
- topology_level at byte 120:
  1 edge-manifold checks ran; pairwise self-intersection did not
  2 those checks plus pairwise non-adjacent intersections (at most 128 triangles)
- status at byte 40 is the physical-height status: 1 unknown, 2 current, 3 stale

All multi-byte numbers are little-endian. Array sections are 4-byte aligned.
"""

import struct
from dataclasses import dataclass

import numpy as np

from app.projects.errors import InvalidMeshRequest
from app.relief.mesh import DEFAULT_MAX_VERTICES, ArtifactStatus
from app.relief.solid import (
    SELF_INTERSECTION_LIMITATION,
    SolidMesh,
    build_solid_mesh,
    plan_solid_grid,
)

MAGIC = b"RFMP"
FORMAT_VERSION = 1
ENDIAN_MARKER = 1
HEADER_BYTES = 256
SERVER_MAX_TOP_VERTICES = DEFAULT_MAX_VERTICES
MAX_PREVIEW_PAYLOAD_BYTES = 32 * 1024 * 1024
FLOAT32_ABSOLUTE_TOLERANCE = 1e-4
FLOAT32_RELATIVE_TOLERANCE = 1e-5

POSITION_COMPONENTS = 3
INDEX_COMPONENTS = 3
NORMAL_COMPONENTS = 3
NORMAL_SCOPE_FACE = 1
EXPORT_READY_VALUE = 0

FLAG_DOWNSAMPLED = 1
FLAG_INTERSECTION_CHECKED = 2
FLAG_TOPOLOGY_PASSED = 4

STATUS_UNKNOWN = 1
STATUS_CURRENT = 2
STATUS_STALE = 3
STATUS_CODE = {"unknown": STATUS_UNKNOWN, "current": STATUS_CURRENT, "stale": STATUS_STALE}
STATUS_NAME = {code: name for name, code in STATUS_CODE.items()}

TOPOLOGY_EDGE_MANIFOLD = 1
TOPOLOGY_BOUNDED_INTERSECTION = 2
TOPOLOGY_LEVEL_NAME = {
    TOPOLOGY_EDGE_MANIFOLD: "edge-manifold",
    TOPOLOGY_BOUNDED_INTERSECTION: "bounded-intersection",
}
TOPOLOGY_LEVEL_CHECKS = {
    TOPOLOGY_EDGE_MANIFOLD: (
        "indices, triangle area, edge incidence, opposite winding, single shell, "
        "positive volume, bounding box, and outward normals"
    ),
    TOPOLOGY_BOUNDED_INTERSECTION: (
        "indices, triangle area, edge incidence, opposite winding, single shell, "
        "positive volume, bounding box, outward normals, and pairwise non-adjacent "
        "triangle intersections"
    ),
}

OFFSET_MAGIC = 0
OFFSET_VERSION = 4
OFFSET_ENDIAN = 8
OFFSET_HEADER_BYTES = 12
OFFSET_VERTEX_COUNT = 16
OFFSET_TRIANGLE_COUNT = 20
OFFSET_POSITION_COMPONENTS = 24
OFFSET_INDEX_COMPONENTS = 28
OFFSET_NORMAL_COMPONENTS = 32
OFFSET_FLAGS = 36
OFFSET_STATUS = 40
OFFSET_SOURCE_WIDTH = 44
OFFSET_SOURCE_HEIGHT = 48
OFFSET_MESH_WIDTH = 52
OFFSET_MESH_HEIGHT = 56
OFFSET_ARTWORK_WIDTH = 60
OFFSET_ARTWORK_HEIGHT = 64
OFFSET_BBOX_MIN_X = 68
OFFSET_BBOX_MIN_Y = 72
OFFSET_BBOX_MIN_Z = 76
OFFSET_BBOX_MAX_X = 80
OFFSET_BBOX_MAX_Y = 84
OFFSET_BBOX_MAX_Z = 88
OFFSET_POSITIONS_OFFSET = 92
OFFSET_POSITIONS_BYTES = 96
OFFSET_INDICES_OFFSET = 100
OFFSET_INDICES_BYTES = 104
OFFSET_NORMALS_OFFSET = 108
OFFSET_NORMALS_BYTES = 112
OFFSET_PAYLOAD_BYTES = 116
OFFSET_TOPOLOGY_LEVEL = 120
OFFSET_EXPORT_READY = 124
OFFSET_MAX_VERTICES = 128
OFFSET_NORMAL_SCOPE = 132
OFFSET_CONVENTION = 136
CONVENTION_BYTES = 64
OFFSET_NORMAL_KIND = 200
NORMAL_KIND_BYTES = 32
OFFSET_RESERVED = 232
RESERVED_BYTES = 24

COORDINATE_TOKEN = b"artwork-xy-y-down-z-up"
NORMAL_KIND_TOKEN = b"face"
SERVER_BUDGET_DETAIL = (
    f"max_vertices cannot exceed {SERVER_MAX_TOP_VERTICES}. "
    "The preview builds at most that many top-grid vertices."
)
FLOAT32_DETAIL = (
    "Mesh values cannot be represented as finite float32 within 0.0001 "
    "or a relative tolerance of 1e-5."
)
INDEX_DETAIL = "Triangle indices are outside the vertex buffer or cannot be stored as uint32."
PAYLOAD_DETAIL = "Mesh preview payload of {payload_bytes} bytes exceeds the {limit} byte response limit."


@dataclass(frozen=True)
class PreviewMemoryEstimate:
    """Byte accounting for one preview. Peak is an approximation, not measured RSS."""

    source_bytes: int
    mesh_array_bytes: int
    serialized_bytes: int
    temporary_bytes: int
    validation_temporary_bytes: int
    peak_bytes: int

    def header_value(self) -> str:
        return (
            f"source={self.source_bytes};"
            f"mesh_arrays={self.mesh_array_bytes};"
            f"serialized={self.serialized_bytes};"
            f"temporary={self.temporary_bytes};"
            f"validation_temporary={self.validation_temporary_bytes};"
            f"peak={self.peak_bytes}"
        )


@dataclass(frozen=True)
class MeshPreviewBinary:
    """One complete preview body. Nothing is returned until this exists."""

    body: bytes
    status: str
    topology_level: int
    self_intersection_checked: bool
    downsampling_required: bool
    memory: PreviewMemoryEstimate

    def headers(self) -> dict[str, str]:
        level_name = TOPOLOGY_LEVEL_NAME[self.topology_level]
        return {
            "X-Relief-Mesh-Format": f"RFMP/{FORMAT_VERSION}",
            "X-Relief-Endian": "little",
            "X-Relief-Provenance-Status": self.status,
            "X-Relief-Topology-Level": str(self.topology_level),
            "X-Relief-Topology-Name": level_name,
            "X-Relief-Topology-Checks": TOPOLOGY_LEVEL_CHECKS[self.topology_level],
            "X-Relief-Topology-Limitation": SELF_INTERSECTION_LIMITATION,
            "X-Relief-Self-Intersection-Checked": "true" if self.self_intersection_checked else "false",
            "X-Relief-Downsampling": "true" if self.downsampling_required else "false",
            "X-Relief-Export-Ready": "false",
            "X-Relief-Payload-Bytes": str(len(self.body)),
            "X-Relief-Memory-Estimate": self.memory.header_value(),
            "Cache-Control": "no-store",
        }


@dataclass(frozen=True)
class DecodedMeshPreview:
    version: int
    vertex_count: int
    triangle_count: int
    positions: np.ndarray
    indices: np.ndarray
    face_normals: np.ndarray
    status: str
    source_width_px: int
    source_height_px: int
    mesh_width_px: int
    mesh_height_px: int
    artwork_width_mm: float
    artwork_height_mm: float
    bounding_box_mm: tuple[tuple[float, float, float], tuple[float, float, float]]
    downsampling_required: bool
    self_intersection_checked: bool
    topology_passed: bool
    topology_level: int
    export_ready: int
    max_vertices: int
    normal_scope: int
    convention: str
    normal_kind: str
    payload_bytes: int


def preview_payload_size(vertex_count: int, triangle_count: int) -> int:
    """Header plus float32 positions, uint32 indices, and float32 face normals."""

    return (
        HEADER_BYTES
        + vertex_count * POSITION_COMPONENTS * 4
        + triangle_count * INDEX_COMPONENTS * 4
        + triangle_count * NORMAL_COMPONENTS * 4
    )


def validation_temporary_bytes(triangle_count: int) -> int:
    """Main temporary NumPy arrays inside the solid topology check.

    Counts edge-pair, key, owner, and argsort buffers plus face cross products
    and unit normals. This is not measured process RSS.
    """

    edges = triangle_count * 3
    return edges * 16 + edges * 8 * 3 + triangle_count * 24 * 2


def estimate_preview_memory(
    *,
    source_bytes: int,
    vertex_count: int,
    triangle_count: int,
) -> PreviewMemoryEstimate:
    mesh_array_bytes = vertex_count * 3 * 8 + triangle_count * 3 * 4 + triangle_count * 3 * 8 + triangle_count
    serialized_bytes = preview_payload_size(vertex_count, triangle_count)
    temporary_bytes = serialized_bytes - HEADER_BYTES
    validation_bytes = validation_temporary_bytes(triangle_count)
    encode_extra = temporary_bytes + serialized_bytes
    peak_bytes = source_bytes + mesh_array_bytes + max(validation_bytes, encode_extra)
    return PreviewMemoryEstimate(
        source_bytes=source_bytes,
        mesh_array_bytes=mesh_array_bytes,
        serialized_bytes=serialized_bytes,
        temporary_bytes=temporary_bytes,
        validation_temporary_bytes=validation_bytes,
        peak_bytes=peak_bytes,
    )


def build_preview_bytes(
    array: np.ndarray,
    artwork_width_mm: float,
    artwork_height_mm: float,
    *,
    status: ArtifactStatus,
    max_vertices: int = DEFAULT_MAX_VERTICES,
    stock_thickness_mm: float | None = None,
    source_bytes: int | None = None,
) -> MeshPreviewBinary:
    """Plan, build one solid, and pack it. The input array is not modified."""

    _require_server_budget(max_vertices)
    height_px, width_px = _planned_shape(array)
    plan = plan_solid_grid(height_px, width_px, max_vertices)
    _require_payload_limit(preview_payload_size(plan.vertex_count, plan.triangle_count))
    mesh = build_solid_mesh(
        array,
        artwork_width_mm,
        artwork_height_mm,
        max_vertices=max_vertices,
        stock_thickness_mm=stock_thickness_mm,
    )
    if mesh.vertex_count != plan.vertex_count or mesh.triangle_count != plan.triangle_count:
        raise InvalidMeshRequest("The built solid does not match the planned preview size.")
    accounted_source = int(array.nbytes if source_bytes is None else source_bytes)
    return encode_solid_preview(
        mesh,
        status=status,
        artwork_width_mm=float(artwork_width_mm),
        artwork_height_mm=float(artwork_height_mm),
        source_bytes=accounted_source,
    )


def encode_solid_preview(
    mesh: SolidMesh,
    *,
    status: ArtifactStatus,
    artwork_width_mm: float,
    artwork_height_mm: float,
    source_bytes: int,
) -> MeshPreviewBinary:
    """Pack an already validated solid. This does not build another mesh."""

    if status not in STATUS_CODE:
        raise InvalidMeshRequest(
            "Mesh preview requires a physical-height status of current, stale, or unknown."
        )
    if not mesh.topology.passed:
        raise InvalidMeshRequest("Mesh preview requires a solid that passed its topology checks.")
    positions = require_preview_float32(mesh.vertices, "Vertex coordinates")
    normals = require_preview_float32(mesh.normals, "Face normals")
    indices = _require_uint32_indices(mesh.triangles, mesh.vertex_count)
    if positions.shape != (mesh.vertex_count, POSITION_COMPONENTS):
        raise InvalidMeshRequest(FLOAT32_DETAIL)
    if normals.shape != (mesh.triangle_count, NORMAL_COMPONENTS):
        raise InvalidMeshRequest(FLOAT32_DETAIL)
    artwork = require_preview_float32(
        np.array([artwork_width_mm, artwork_height_mm], dtype=np.float64),
        "Artwork dimensions",
    )
    lowest, highest = mesh.bounding_box_mm
    box = require_preview_float32(np.array([*lowest, *highest], dtype=np.float64), "Bounding box")
    payload_bytes = preview_payload_size(mesh.vertex_count, mesh.triangle_count)
    _require_payload_limit(payload_bytes)
    positions_bytes = int(positions.nbytes)
    indices_bytes = int(indices.nbytes)
    normals_bytes = int(normals.nbytes)
    positions_offset = HEADER_BYTES
    indices_offset = positions_offset + positions_bytes
    normals_offset = indices_offset + indices_bytes
    if (positions_offset | indices_offset | normals_offset) & 3:
        raise InvalidMeshRequest("Mesh preview arrays are not 4-byte aligned.")
    topology_level = (
        TOPOLOGY_BOUNDED_INTERSECTION
        if mesh.topology.self_intersection_checked
        else TOPOLOGY_EDGE_MANIFOLD
    )
    flags = FLAG_TOPOLOGY_PASSED
    if mesh.downsampling_required:
        flags |= FLAG_DOWNSAMPLED
    if mesh.topology.self_intersection_checked:
        flags |= FLAG_INTERSECTION_CHECKED

    body = bytearray(payload_bytes)
    body[OFFSET_MAGIC : OFFSET_MAGIC + 4] = MAGIC
    _write_u32(body, OFFSET_VERSION, FORMAT_VERSION)
    _write_u32(body, OFFSET_ENDIAN, ENDIAN_MARKER)
    _write_u32(body, OFFSET_HEADER_BYTES, HEADER_BYTES)
    _write_u32(body, OFFSET_VERTEX_COUNT, mesh.vertex_count)
    _write_u32(body, OFFSET_TRIANGLE_COUNT, mesh.triangle_count)
    _write_u32(body, OFFSET_POSITION_COMPONENTS, POSITION_COMPONENTS)
    _write_u32(body, OFFSET_INDEX_COMPONENTS, INDEX_COMPONENTS)
    _write_u32(body, OFFSET_NORMAL_COMPONENTS, NORMAL_COMPONENTS)
    _write_u32(body, OFFSET_FLAGS, flags)
    _write_u32(body, OFFSET_STATUS, STATUS_CODE[status])
    _write_u32(body, OFFSET_SOURCE_WIDTH, mesh.source_width_px)
    _write_u32(body, OFFSET_SOURCE_HEIGHT, mesh.source_height_px)
    _write_u32(body, OFFSET_MESH_WIDTH, mesh.mesh_width_px)
    _write_u32(body, OFFSET_MESH_HEIGHT, mesh.mesh_height_px)
    _write_f32(body, OFFSET_ARTWORK_WIDTH, float(artwork[0]))
    _write_f32(body, OFFSET_ARTWORK_HEIGHT, float(artwork[1]))
    for offset, value in zip(
        (
            OFFSET_BBOX_MIN_X,
            OFFSET_BBOX_MIN_Y,
            OFFSET_BBOX_MIN_Z,
            OFFSET_BBOX_MAX_X,
            OFFSET_BBOX_MAX_Y,
            OFFSET_BBOX_MAX_Z,
        ),
        box,
        strict=True,
    ):
        _write_f32(body, offset, float(value))
    _write_u32(body, OFFSET_POSITIONS_OFFSET, positions_offset)
    _write_u32(body, OFFSET_POSITIONS_BYTES, positions_bytes)
    _write_u32(body, OFFSET_INDICES_OFFSET, indices_offset)
    _write_u32(body, OFFSET_INDICES_BYTES, indices_bytes)
    _write_u32(body, OFFSET_NORMALS_OFFSET, normals_offset)
    _write_u32(body, OFFSET_NORMALS_BYTES, normals_bytes)
    _write_u32(body, OFFSET_PAYLOAD_BYTES, payload_bytes)
    _write_u32(body, OFFSET_TOPOLOGY_LEVEL, topology_level)
    _write_u32(body, OFFSET_EXPORT_READY, EXPORT_READY_VALUE)
    _write_u32(body, OFFSET_MAX_VERTICES, mesh.max_vertices)
    _write_u32(body, OFFSET_NORMAL_SCOPE, NORMAL_SCOPE_FACE)
    _write_token(body, OFFSET_CONVENTION, CONVENTION_BYTES, COORDINATE_TOKEN)
    _write_token(body, OFFSET_NORMAL_KIND, NORMAL_KIND_BYTES, NORMAL_KIND_TOKEN)
    body[positions_offset : positions_offset + positions_bytes] = positions.tobytes()
    body[indices_offset : indices_offset + indices_bytes] = indices.tobytes()
    body[normals_offset : normals_offset + normals_bytes] = normals.tobytes()
    memory = estimate_preview_memory(
        source_bytes=source_bytes,
        vertex_count=mesh.vertex_count,
        triangle_count=mesh.triangle_count,
    )
    return MeshPreviewBinary(
        body=bytes(body),
        status=status,
        topology_level=topology_level,
        self_intersection_checked=mesh.topology.self_intersection_checked,
        downsampling_required=mesh.downsampling_required,
        memory=memory,
    )


def decode_mesh_preview(buffer: bytes) -> DecodedMeshPreview:
    """Read a preview payload. This does not construct geometry."""

    if len(buffer) < HEADER_BYTES:
        raise InvalidMeshRequest("Mesh preview payload is shorter than the header.")
    if buffer[OFFSET_MAGIC : OFFSET_MAGIC + 4] != MAGIC:
        raise InvalidMeshRequest("Mesh preview magic is not RFMP.")
    if _read_u32(buffer, OFFSET_VERSION) != FORMAT_VERSION:
        raise InvalidMeshRequest("Mesh preview version is not 1.")
    if _read_u32(buffer, OFFSET_ENDIAN) != ENDIAN_MARKER or buffer[OFFSET_ENDIAN : OFFSET_ENDIAN + 4] != b"\x01\x00\x00\x00":
        raise InvalidMeshRequest("Mesh preview endian marker is not little-endian 1.")
    if _read_u32(buffer, OFFSET_HEADER_BYTES) != HEADER_BYTES:
        raise InvalidMeshRequest("Mesh preview header size is not 256 bytes.")
    payload_bytes = _read_u32(buffer, OFFSET_PAYLOAD_BYTES)
    if payload_bytes != len(buffer):
        raise InvalidMeshRequest("Mesh preview payload length does not match the header.")
    if buffer[OFFSET_RESERVED : OFFSET_RESERVED + RESERVED_BYTES] != bytes(RESERVED_BYTES):
        raise InvalidMeshRequest("Mesh preview reserved header bytes are not zero.")
    vertex_count = _read_u32(buffer, OFFSET_VERTEX_COUNT)
    triangle_count = _read_u32(buffer, OFFSET_TRIANGLE_COUNT)
    positions_offset = _read_u32(buffer, OFFSET_POSITIONS_OFFSET)
    positions_bytes = _read_u32(buffer, OFFSET_POSITIONS_BYTES)
    indices_offset = _read_u32(buffer, OFFSET_INDICES_OFFSET)
    indices_bytes = _read_u32(buffer, OFFSET_INDICES_BYTES)
    normals_offset = _read_u32(buffer, OFFSET_NORMALS_OFFSET)
    normals_bytes = _read_u32(buffer, OFFSET_NORMALS_BYTES)
    expected_positions = vertex_count * POSITION_COMPONENTS * 4
    expected_indices = triangle_count * INDEX_COMPONENTS * 4
    expected_normals = triangle_count * NORMAL_COMPONENTS * 4
    if (
        positions_offset != HEADER_BYTES
        or positions_bytes != expected_positions
        or indices_offset != positions_offset + positions_bytes
        or indices_bytes != expected_indices
        or normals_offset != indices_offset + indices_bytes
        or normals_bytes != expected_normals
        or positions_offset + positions_bytes + indices_bytes + normals_bytes != payload_bytes
    ):
        raise InvalidMeshRequest("Mesh preview section offsets do not match the vertex and triangle counts.")
    positions = np.frombuffer(buffer, dtype="<f4", count=vertex_count * 3, offset=positions_offset).reshape(
        vertex_count, 3
    )
    indices = np.frombuffer(buffer, dtype="<u4", count=triangle_count * 3, offset=indices_offset).reshape(
        triangle_count, 3
    )
    normals = np.frombuffer(buffer, dtype="<f4", count=triangle_count * 3, offset=normals_offset).reshape(
        triangle_count, 3
    )
    flags = _read_u32(buffer, OFFSET_FLAGS)
    status_code = _read_u32(buffer, OFFSET_STATUS)
    if status_code not in STATUS_NAME:
        raise InvalidMeshRequest("Mesh preview status code is not current, stale, or unknown.")
    box_values = [
        _read_f32(buffer, OFFSET_BBOX_MIN_X),
        _read_f32(buffer, OFFSET_BBOX_MIN_Y),
        _read_f32(buffer, OFFSET_BBOX_MIN_Z),
        _read_f32(buffer, OFFSET_BBOX_MAX_X),
        _read_f32(buffer, OFFSET_BBOX_MAX_Y),
        _read_f32(buffer, OFFSET_BBOX_MAX_Z),
    ]
    return DecodedMeshPreview(
        version=FORMAT_VERSION,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        positions=positions,
        indices=indices,
        face_normals=normals,
        status=STATUS_NAME[status_code],
        source_width_px=_read_u32(buffer, OFFSET_SOURCE_WIDTH),
        source_height_px=_read_u32(buffer, OFFSET_SOURCE_HEIGHT),
        mesh_width_px=_read_u32(buffer, OFFSET_MESH_WIDTH),
        mesh_height_px=_read_u32(buffer, OFFSET_MESH_HEIGHT),
        artwork_width_mm=_read_f32(buffer, OFFSET_ARTWORK_WIDTH),
        artwork_height_mm=_read_f32(buffer, OFFSET_ARTWORK_HEIGHT),
        bounding_box_mm=(
            (box_values[0], box_values[1], box_values[2]),
            (box_values[3], box_values[4], box_values[5]),
        ),
        downsampling_required=bool(flags & FLAG_DOWNSAMPLED),
        self_intersection_checked=bool(flags & FLAG_INTERSECTION_CHECKED),
        topology_passed=bool(flags & FLAG_TOPOLOGY_PASSED),
        topology_level=_read_u32(buffer, OFFSET_TOPOLOGY_LEVEL),
        export_ready=_read_u32(buffer, OFFSET_EXPORT_READY),
        max_vertices=_read_u32(buffer, OFFSET_MAX_VERTICES),
        normal_scope=_read_u32(buffer, OFFSET_NORMAL_SCOPE),
        convention=_read_token(buffer, OFFSET_CONVENTION, CONVENTION_BYTES),
        normal_kind=_read_token(buffer, OFFSET_NORMAL_KIND, NORMAL_KIND_BYTES),
        payload_bytes=payload_bytes,
    )


def require_preview_float32(values: np.ndarray, label: str) -> np.ndarray:
    """Return a little-endian float32 copy, or reject an unsafe conversion."""

    numeric = isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number)
    if not numeric or values.size == 0 or not bool(np.isfinite(values).all()):
        raise InvalidMeshRequest(f"{label} are non-finite. {FLOAT32_DETAIL}")
    with np.errstate(over="ignore"):
        converted = np.ascontiguousarray(values, dtype="<f4")
    if not bool(np.isfinite(converted).all()):
        raise InvalidMeshRequest(f"{label} overflow float32. {FLOAT32_DETAIL}")
    restored = converted.astype(np.float64)
    original = np.asarray(values, dtype=np.float64)
    limit = np.maximum(FLOAT32_ABSOLUTE_TOLERANCE, np.abs(original) * FLOAT32_RELATIVE_TOLERANCE)
    if bool(np.any(np.abs(restored - original) > limit)):
        raise InvalidMeshRequest(f"{label} do not round-trip through float32. {FLOAT32_DETAIL}")
    return converted


def _require_server_budget(max_vertices: int) -> None:
    if isinstance(max_vertices, bool) or not isinstance(max_vertices, int) or max_vertices < 4:
        from app.relief.mesh import BUDGET_DETAIL

        raise InvalidMeshRequest(BUDGET_DETAIL)
    if max_vertices > SERVER_MAX_TOP_VERTICES:
        raise InvalidMeshRequest(SERVER_BUDGET_DETAIL)


def _planned_shape(array: np.ndarray) -> tuple[int, int]:
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        from app.relief.diagnostics import STRUCTURE_DETAIL

        raise InvalidMeshRequest(STRUCTURE_DETAIL)
    return int(array.shape[0]), int(array.shape[1])


def _require_payload_limit(payload_bytes: int) -> None:
    if payload_bytes > MAX_PREVIEW_PAYLOAD_BYTES:
        raise InvalidMeshRequest(
            PAYLOAD_DETAIL.format(payload_bytes=payload_bytes, limit=MAX_PREVIEW_PAYLOAD_BYTES)
        )


def _require_uint32_indices(triangles: np.ndarray, vertex_count: int) -> np.ndarray:
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise InvalidMeshRequest(INDEX_DETAIL)
    as_int = triangles.astype(np.int64, copy=False)
    if bool(np.any(as_int < 0)) or bool(np.any(as_int >= vertex_count)):
        raise InvalidMeshRequest(INDEX_DETAIL)
    converted = np.ascontiguousarray(as_int, dtype="<u4")
    if not np.array_equal(converted.astype(np.int64), as_int):
        raise InvalidMeshRequest(INDEX_DETAIL)
    return converted


def _write_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, int(value))


def _write_f32(buffer: bytearray, offset: int, value: float) -> None:
    struct.pack_into("<f", buffer, offset, float(value))


def _write_token(buffer: bytearray, offset: int, size: int, token: bytes) -> None:
    if len(token) >= size:
        raise InvalidMeshRequest("Mesh preview header token does not fit.")
    buffer[offset : offset + size] = token + bytes(size - len(token))


def _read_u32(buffer: bytes, offset: int) -> int:
    return struct.unpack_from("<I", buffer, offset)[0]


def _read_f32(buffer: bytes, offset: int) -> float:
    return float(struct.unpack_from("<f", buffer, offset)[0])


def _read_token(buffer: bytes, offset: int, size: int) -> str:
    raw = buffer[offset : offset + size].split(b"\x00", 1)[0]
    return raw.decode("ascii")
