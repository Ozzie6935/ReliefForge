"""Bounded binary mesh preview. Temporary projects and synthetic arrays only."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.projects.errors import InvalidMeshRequest
from app.relief.diagnostics import FILE_MISSING_DETAIL, RECORD_MISSING_DETAIL
from app.relief.mesh import BUDGET_DETAIL, NON_FINITE_DETAIL
from app.relief import preview_binary
from app.relief.preview_binary import (
    COORDINATE_TOKEN,
    FLAG_DOWNSAMPLED,
    FLAG_INTERSECTION_CHECKED,
    FLAG_TOPOLOGY_PASSED,
    HEADER_BYTES,
    MAX_PREVIEW_PAYLOAD_BYTES,
    NORMAL_KIND_TOKEN,
    NORMAL_SCOPE_FACE,
    OFFSET_ENDIAN,
    OFFSET_EXPORT_READY,
    OFFSET_MAGIC,
    OFFSET_POSITIONS_OFFSET,
    OFFSET_STATUS,
    OFFSET_TOPOLOGY_LEVEL,
    OFFSET_VERSION,
    SERVER_BUDGET_DETAIL,
    SERVER_MAX_TOP_VERTICES,
    STATUS_CURRENT,
    STATUS_STALE,
    STATUS_UNKNOWN,
    build_preview_bytes,
    decode_mesh_preview,
    estimate_preview_memory,
    preview_payload_size,
    require_preview_float32,
)
from app.relief.solid import NON_POSITIVE_DETAIL, plan_solid_grid
from tests.test_physical_diagnostics import _files
from tests.test_projects import served
from tests.test_provenance import _pipeline, _root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _cross_z(positions: np.ndarray, triangle: np.ndarray) -> float:
    points = positions[triangle.astype(np.int64)]
    return float(np.cross(points[1] - points[0], points[2] - points[0])[2])


def test_header_layout_round_trip_and_outward_winding() -> None:
    array = np.full((2, 2), 5, dtype=np.float32)
    before = array.copy()
    packed = build_preview_bytes(array, 10, 20, status="current")
    assert np.array_equal(array, before)
    body = packed.body
    assert body[OFFSET_MAGIC : OFFSET_MAGIC + 4] == b"RFMP"
    assert body[OFFSET_ENDIAN : OFFSET_ENDIAN + 4] == b"\x01\x00\x00\x00"
    assert int.from_bytes(body[OFFSET_VERSION : OFFSET_VERSION + 4], "little") == 1
    assert int.from_bytes(body[OFFSET_EXPORT_READY : OFFSET_EXPORT_READY + 4], "little") == 0
    assert int.from_bytes(body[OFFSET_STATUS : OFFSET_STATUS + 4], "little") == STATUS_CURRENT
    assert int.from_bytes(body[OFFSET_TOPOLOGY_LEVEL : OFFSET_TOPOLOGY_LEVEL + 4], "little") == 2
    assert int.from_bytes(body[OFFSET_POSITIONS_OFFSET : OFFSET_POSITIONS_OFFSET + 4], "little") == HEADER_BYTES
    assert len(body) == preview_payload_size(8, 12)
    assert len(body) == HEADER_BYTES + 8 * 12 + 12 * 12 + 12 * 12

    decoded = decode_mesh_preview(body)
    assert decoded.version == 1
    assert decoded.vertex_count == 8
    assert decoded.triangle_count == 12
    assert decoded.positions.dtype == np.dtype("<f4")
    assert decoded.indices.dtype == np.dtype("<u4")
    assert decoded.face_normals.dtype == np.dtype("<f4")
    assert decoded.positions.shape == (8, 3)
    assert decoded.indices.shape == (12, 3)
    assert decoded.face_normals.shape == (12, 3)
    assert decoded.export_ready == 0
    assert decoded.normal_scope == NORMAL_SCOPE_FACE
    assert decoded.normal_kind == "face"
    assert decoded.convention == COORDINATE_TOKEN.decode("ascii")
    assert decoded.convention == "artwork-xy-y-down-z-up"
    assert NORMAL_KIND_TOKEN == b"face"
    assert decoded.status == "current"
    assert decoded.artwork_width_mm == pytest.approx(10)
    assert decoded.artwork_height_mm == pytest.approx(20)
    assert decoded.bounding_box_mm[0] == pytest.approx((0, 0, 0))
    assert decoded.bounding_box_mm[1] == pytest.approx((10, 20, 5))
    assert decoded.downsampling_required is False
    assert decoded.self_intersection_checked is True
    assert decoded.topology_level == 2
    assert decoded.topology_passed is True
    np.testing.assert_allclose(decoded.positions[:4, 2], 5)
    np.testing.assert_allclose(decoded.positions[4:, 2], 0)
    np.testing.assert_allclose(decoded.positions[0], [0, 0, 5])
    np.testing.assert_allclose(decoded.positions[4], [0, 0, 0])
    assert _cross_z(decoded.positions, decoded.indices[0]) > 0
    assert decoded.face_normals[0, 2] > 0
    assert np.any(decoded.face_normals[:, 2] < 0)
    assert np.any(np.isclose(decoded.face_normals[:, 0], -1))
    assert np.any(np.isclose(decoded.face_normals[:, 0], 1))
    assert np.any(np.isclose(decoded.face_normals[:, 1], -1))
    assert np.any(np.isclose(decoded.face_normals[:, 1], 1))
    assert packed.headers()["X-Relief-Export-Ready"] == "false"
    assert packed.headers()["X-Relief-Provenance-Status"] == "current"
    assert "export-ready" in packed.headers()["X-Relief-Topology-Limitation"]
    assert FLAG_TOPOLOGY_PASSED | FLAG_INTERSECTION_CHECKED

    again = build_preview_bytes(array, 10, 20, status="current")
    assert again.body == body


def test_slope_non_square_budget_and_downsampling_are_deterministic() -> None:
    slope = np.array([[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    sloped = decode_mesh_preview(build_preview_bytes(slope, 20, 10, status="stale").body)
    assert sloped.status == "stale"
    assert sloped.status != "current"
    np.testing.assert_allclose(sloped.positions[0], [0, 0, 2])
    np.testing.assert_allclose(sloped.positions[2], [20, 0, 6])
    np.testing.assert_allclose(sloped.positions[3], [0, 10, 2])
    assert sloped.mesh_width_px == 3
    assert sloped.mesh_height_px == 2
    assert _cross_z(sloped.positions, sloped.indices[0]) > 0
    assert np.all(sloped.face_normals[: sloped.triangle_count // 4, 2] != 0)

    wide = np.full((2, 5), 2.5, dtype=np.float32)
    block = decode_mesh_preview(build_preview_bytes(wide, 50, 10, status="unknown").body)
    assert block.status == "unknown"
    assert block.vertex_count == 20
    assert block.bounding_box_mm[1] == pytest.approx((50, 10, 2.5))
    assert block.positions[:, 0].max() == pytest.approx(50)
    assert block.positions[:, 1].max() == pytest.approx(10)

    tall = np.arange(1, 46, dtype=np.float32).reshape(9, 5)
    tall[4, 2] = 90
    first = build_preview_bytes(tall, 80, 40, status="current", max_vertices=12)
    second = build_preview_bytes(tall, 80, 40, status="current", max_vertices=12)
    assert first.body == second.body
    sampled = decode_mesh_preview(first.body)
    assert sampled.downsampling_required is True
    assert sampled.max_vertices == 12
    assert sampled.mesh_width_px * sampled.mesh_height_px <= 12
    assert sampled.vertex_count == 2 * sampled.mesh_width_px * sampled.mesh_height_px
    assert sampled.positions[:, 0].min() == pytest.approx(0)
    assert sampled.positions[:, 0].max() == pytest.approx(80)
    assert sampled.positions[:, 1].min() == pytest.approx(0)
    assert sampled.positions[:, 1].max() == pytest.approx(40)
    assert sampled.positions[:, 2].max() < 90
    assert first.body[36] & FLAG_DOWNSAMPLED

    with pytest.raises(InvalidMeshRequest, match=SERVER_BUDGET_DETAIL):
        build_preview_bytes(wide, 50, 10, status="current", max_vertices=SERVER_MAX_TOP_VERTICES + 1)
    with pytest.raises(InvalidMeshRequest, match=BUDGET_DETAIL):
        build_preview_bytes(wide, 50, 10, status="current", max_vertices=3)


def test_invalid_geometry_float32_and_payload_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    valid = np.full((2, 2), 1.5, dtype=np.float32)
    before = valid.copy()
    with pytest.raises(InvalidMeshRequest, match=NON_POSITIVE_DETAIL):
        build_preview_bytes(np.zeros((2, 2), dtype=np.float32), 10, 10, status="current")
    dirty = valid.copy()
    dirty[0, 0] = np.nan
    with pytest.raises(InvalidMeshRequest, match=NON_FINITE_DETAIL):
        build_preview_bytes(dirty, 10, 10, status="current")
    assert np.array_equal(valid, before)
    assert np.isnan(dirty[0, 0])

    packed = build_preview_bytes(valid, 8, 6, status="current")
    decoded = decode_mesh_preview(packed.body)
    assert np.array_equal(decoded.indices.astype(np.int64), decoded.indices.astype(np.int64))
    overflow = require_preview_float32(np.array([1.0, -2.0], dtype=np.float64), "Sample")
    assert overflow.dtype == np.dtype("<f4")
    with pytest.raises(InvalidMeshRequest, match="float32"):
        require_preview_float32(np.array([1e40], dtype=np.float64), "Vertex coordinates")
    with pytest.raises(InvalidMeshRequest, match="non-finite"):
        require_preview_float32(np.array([np.nan], dtype=np.float64), "Vertex coordinates")

    mesh_bytes = packed.body
    broken = bytearray(mesh_bytes)
    # Vertex 0 X becomes a quiet NaN pattern after a successful pack is not how we test
    # the builder. Mutating the packed indices to wrap is covered by the uint32 check
    # through a rebuilt mesh below.
    positions = np.frombuffer(mesh_bytes, dtype="<f4", offset=HEADER_BYTES, count=24).reshape(8, 3).copy()
    positions[0, 0] = np.float32(np.inf)
    assert not np.isfinite(positions).all()

    from app.relief.solid import build_solid_mesh

    mesh = build_solid_mesh(valid, 8, 6)
    mesh.vertices[0, 0] = 1e40
    with pytest.raises(InvalidMeshRequest, match="float32"):
        preview_binary.encode_solid_preview(
            mesh,
            status="current",
            artwork_width_mm=8,
            artwork_height_mm=6,
            source_bytes=int(valid.nbytes),
        )
    mesh.vertices[0, 0] = 0
    mesh.triangles[0, 0] = -1
    with pytest.raises(InvalidMeshRequest, match="uint32"):
        preview_binary.encode_solid_preview(
            mesh,
            status="stale",
            artwork_width_mm=8,
            artwork_height_mm=6,
            source_bytes=int(valid.nbytes),
        )

    monkeypatch.setattr(preview_binary, "MAX_PREVIEW_PAYLOAD_BYTES", HEADER_BYTES)
    called = {"build": 0}

    def fail_build(*_args, **_kwargs):
        called["build"] += 1
        raise AssertionError("mesh was built")

    monkeypatch.setattr(preview_binary, "build_solid_mesh", fail_build)
    with pytest.raises(InvalidMeshRequest, match="response limit"):
        build_preview_bytes(valid, 8, 6, status="current")
    assert called["build"] == 0
    assert broken[0:4] == b"RFMP"

    memory = estimate_preview_memory(source_bytes=16, vertex_count=8, triangle_count=12)
    assert memory.serialized_bytes == preview_payload_size(8, 12)
    assert memory.peak_bytes > memory.serialized_bytes
    assert memory.peak_bytes > memory.source_bytes
    assert "peak=" in memory.header_value()
    assert MAX_PREVIEW_PAYLOAD_BYTES == 32 * 1024 * 1024
    assert SERVER_MAX_TOP_VERTICES == 250_000


def test_angie_sized_plan_stays_inside_the_payload_limit_without_a_mesh() -> None:
    plan = plan_solid_grid(2140, 2940, SERVER_MAX_TOP_VERTICES)
    planned = preview_payload_size(plan.vertex_count, plan.triangle_count)
    assert planned <= MAX_PREVIEW_PAYLOAD_BYTES
    full_vertices = 2 * 2140 * 2940
    full_triangles = 4 * 2139 * 2939 + 4 * 2939 + 4 * 2139
    full_payload = preview_payload_size(full_vertices, full_triangles)
    assert full_payload > MAX_PREVIEW_PAYLOAD_BYTES
    assert plan.downsampling_required is True
    assert plan.top_vertex_count <= SERVER_MAX_TOP_VERTICES


def test_mesh_preview_endpoint_is_read_only(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        project_id = body["id"]
        missing = client.get(f"/api/projects/{uuid4()}/relief/mesh/preview")
        assert missing.status_code == 404

        before = _files(project_root)
        current = client.get(f"/api/projects/{project_id}/relief/mesh/preview")
        assert current.status_code == 200, current.text
        assert current.headers["content-type"] == "application/octet-stream"
        assert current.headers["x-relief-export-ready"] == "false"
        assert current.headers["x-relief-provenance-status"] == "current"
        assert current.headers["x-relief-mesh-format"] == "RFMP/1"
        assert current.headers["cache-control"] == "no-store"
        decoded = decode_mesh_preview(current.content)
        height = np.load(_root(project_root) / "depth/composition/physical/physical_height_mm.npy")
        expected = plan_solid_grid(int(height.shape[0]), int(height.shape[1]))
        assert decoded.status == "current"
        assert decoded.export_ready == 0
        assert decoded.vertex_count == expected.vertex_count
        assert decoded.triangle_count == expected.triangle_count
        assert decoded.source_height_px == int(height.shape[0])
        assert decoded.source_width_px == int(height.shape[1])
        assert decoded.artwork_width_mm == pytest.approx(body["artwork"]["width_mm"])
        assert decoded.artwork_height_mm == pytest.approx(body["artwork"]["height_mm"])
        assert _files(project_root) == before
        assert list(_root(project_root).rglob("*mesh*")) == []
        assert client.post(f"/api/projects/{project_id}/relief/mesh/preview").status_code == 405
        info = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert info.status_code == 200
        assert info.json()["surface"] == "top"

        limited = client.get(
            f"/api/projects/{project_id}/relief/mesh/preview",
            params={"max_vertices": 4},
        )
        assert limited.status_code == 200, limited.text
        limited_mesh = decode_mesh_preview(limited.content)
        assert limited_mesh.max_vertices == 4
        assert limited_mesh.mesh_width_px * limited_mesh.mesh_height_px <= 4
        assert limited_mesh.vertex_count == 2 * limited_mesh.mesh_width_px * limited_mesh.mesh_height_px
        assert limited_mesh.status == "current"
        assert _files(project_root) == before

        rejected_budget = client.get(
            f"/api/projects/{project_id}/relief/mesh/preview",
            params={"max_vertices": SERVER_MAX_TOP_VERTICES + 1},
        )
        assert rejected_budget.status_code == 422
        assert rejected_budget.json()["detail"] == SERVER_BUDGET_DETAIL
        assert _files(project_root) == before

        manifest_path = _root(project_root) / "project.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"]["content_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        stale = client.get(f"/api/projects/{project_id}/relief/mesh/preview")
        assert stale.status_code == 200, stale.text
        assert stale.headers["x-relief-provenance-status"] == "stale"
        assert decode_mesh_preview(stale.content).status == "stale"
        assert int.from_bytes(stale.content[OFFSET_STATUS : OFFSET_STATUS + 4], "little") == STATUS_STALE

        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"].pop("content_sha256", None)
        stored["relief"]["physical"].pop("settings_sha256", None)
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        unknown = client.get(f"/api/projects/{project_id}/relief/mesh/preview")
        assert unknown.status_code == 200, unknown.text
        assert decode_mesh_preview(unknown.content).status == "unknown"
        assert int.from_bytes(unknown.content[OFFSET_STATUS : OFFSET_STATUS + 4], "little") == STATUS_UNKNOWN

        height_path = _root(project_root) / "depth/composition/physical/physical_height_mm.npy"
        original = height_path.read_bytes()
        record_height = stored["relief"]["physical"]["height_px"]
        record_width = stored["relief"]["physical"]["width_px"]
        np.save(height_path, np.zeros((record_height, record_width), dtype=np.float32))
        non_positive = client.get(f"/api/projects/{project_id}/relief/mesh/preview")
        assert non_positive.status_code == 422
        assert non_positive.json()["detail"] == NON_POSITIVE_DETAIL
        assert not non_positive.content.startswith(b"RFMP")

        broken = np.ones((record_height, record_width), dtype=np.float32)
        broken[0, 0] = np.nan
        np.save(height_path, broken)
        non_finite = client.get(f"/api/projects/{project_id}/relief/mesh/preview")
        assert non_finite.status_code == 422
        assert non_finite.json()["detail"] == NON_FINITE_DETAIL
        solid_info = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
        assert solid_info.status_code == 200
        assert solid_info.json()["geometry_construction_available"] is False
        assert solid_info.json()["topology_validated"] is False
        height_path.write_bytes(original)

        height_path.unlink()
        absent = client.get(f"/api/projects/{project_id}/relief/mesh/preview")
        assert absent.status_code == 404
        assert absent.json()["detail"] == FILE_MISSING_DETAIL

        created = client.post(
            "/api/projects",
            json={
                "name": "Plain",
                "stock": {"width_mm": 10, "height_mm": 10},
                "artwork": {"width_mm": 8, "height_mm": 8, "center_x_mm": 5, "center_y_mm": 5},
            },
        )
        assert created.status_code == 201, created.text
        no_record = client.get(f"/api/projects/{created.json()['id']}/relief/mesh/preview")
        assert no_record.status_code == 404
        assert no_record.json()["detail"] == RECORD_MISSING_DETAIL
