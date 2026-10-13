"""Read-only top-surface mesh planning. Temporary projects only."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.projects.errors import InvalidMeshRequest
from app.relief.diagnostics import FILE_MISSING_DETAIL, RECORD_MISSING_DETAIL, SHAPE_DETAIL, STRUCTURE_DETAIL
from app.relief.mesh import (
    ARTWORK_DETAIL,
    AXIS_DETAIL,
    BUDGET_DETAIL,
    DEFAULT_MAX_VERTICES,
    NON_FINITE_DETAIL,
    build_top_surface,
    describe_mesh_preview,
    endpoint_indices,
    mesh_memory_bytes,
    plan_preview_grid,
)
from tests.test_physical_diagnostics import _files
from tests.test_projects import served
from tests.test_provenance import _pipeline, _root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def test_flat_plane_has_positive_z_normals_and_exact_bounds() -> None:
    array = np.full((3, 4), 5, dtype=np.float32)
    mesh = build_top_surface(array, 30, 20)
    assert mesh.vertex_count == 12
    assert mesh.triangle_count == 12
    assert mesh.vertices.shape == (12, 3)
    assert mesh.triangles.shape == (12, 3)
    assert mesh.normals.shape == (12, 3)
    assert mesh.downsampling_required is False
    assert mesh.triangles.dtype == np.int32
    assert int(mesh.triangles.min()) == 0
    assert int(mesh.triangles.max()) == 11
    assert len(np.unique(mesh.vertices[:, :2], axis=0)) == 12
    np.testing.assert_allclose(mesh.vertices[:, 0].reshape(3, 4)[0], [0, 10, 20, 30])
    np.testing.assert_allclose(mesh.vertices[:, 1].reshape(3, 4)[:, 0], [0, 10, 20])
    assert np.all(mesh.vertices[:, 2] == 5)
    assert mesh.bounding_box_mm == ((0.0, 0.0, 5.0), (30.0, 20.0, 5.0))
    assert np.all(mesh.normals[:, 2] > 0)
    np.testing.assert_allclose(mesh.normals, np.tile([0.0, 0.0, 1.0], (12, 1)))
    assert mesh.triangles[0].tolist() == [0, 1, 4]
    assert mesh.triangles[1].tolist() == [1, 5, 4]
    assert (
        mesh.vertices.nbytes + mesh.triangles.nbytes + mesh.normals.nbytes
        == mesh_memory_bytes(12, 12)
    )


def test_sloped_plane_and_peak_keep_image_orientation() -> None:
    slope = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    mesh = build_top_surface(slope, 20, 10)
    assert mesh.vertices[0].tolist() == [0.0, 0.0, 0.0]
    assert mesh.vertices[2].tolist() == [20.0, 0.0, 2.0]
    assert mesh.vertices[3].tolist() == [0.0, 10.0, 0.0]
    assert np.all(mesh.normals[:, 2] > 0)
    assert np.allclose(mesh.normals, mesh.normals[0])
    crossed = np.cross(mesh.vertices[1] - mesh.vertices[0], mesh.vertices[3] - mesh.vertices[0])
    assert crossed[2] > 0

    peak = np.ones((3, 3), dtype=np.float32)
    peak[0, 2] = 9
    peak[1, 1] = 4
    surfaced = build_top_surface(peak, 8, 6)
    assert surfaced.vertices[2].tolist() == [8.0, 0.0, 9.0]
    assert surfaced.vertices[4, 2] == 4
    assert surfaced.bounding_box_mm[0][2] == 1
    assert surfaced.bounding_box_mm[1] == (8.0, 6.0, 9.0)
    assert surfaced.row_indices.tolist() == [0, 1, 2]
    assert surfaced.column_indices.tolist() == [0, 1, 2]


def test_non_square_grid_uses_artwork_extents_without_a_stock_offset() -> None:
    array = np.arange(10, dtype=np.float32).reshape(2, 5)
    mesh = build_top_surface(array, 40, 15)
    assert mesh.source_height_px == 2
    assert mesh.source_width_px == 5
    assert mesh.vertex_count == 10
    assert mesh.triangle_count == 8
    xs = mesh.vertices[:, 0].reshape(2, 5)
    ys = mesh.vertices[:, 1].reshape(2, 5)
    np.testing.assert_allclose(xs[0], [0, 10, 20, 30, 40])
    np.testing.assert_allclose(ys[:, 0], [0, 15])
    assert mesh.bounding_box_mm[0][:2] == (0.0, 0.0)
    assert mesh.bounding_box_mm[1][:2] == (40.0, 15.0)
    np.testing.assert_allclose(mesh.vertices[:, 2], array.reshape(-1))


def test_invalid_arrays_and_artwork_are_rejected_without_filling() -> None:
    finite = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    dirty = finite.copy()
    dirty[1, 1] = np.nan
    before = dirty.copy()
    with pytest.raises(InvalidMeshRequest, match=NON_FINITE_DETAIL):
        build_top_surface(dirty, 10, 10)
    assert np.array_equal(dirty, before, equal_nan=True)
    dirty[0, 0] = np.inf
    with pytest.raises(InvalidMeshRequest, match=NON_FINITE_DETAIL):
        build_top_surface(dirty, 10, 10)
    assert np.isnan(dirty[1, 1])
    assert np.isinf(dirty[0, 0])

    with pytest.raises(InvalidMeshRequest, match=AXIS_DETAIL):
        build_top_surface(np.zeros((1, 5), dtype=np.float32), 10, 10)
    with pytest.raises(InvalidMeshRequest, match=AXIS_DETAIL):
        build_top_surface(np.zeros((5, 1), dtype=np.float32), 10, 10)
    with pytest.raises(InvalidMeshRequest, match=STRUCTURE_DETAIL):
        build_top_surface(np.zeros((0, 4), dtype=np.float32), 10, 10)
    with pytest.raises(InvalidMeshRequest, match=STRUCTURE_DETAIL):
        build_top_surface(np.zeros((4,), dtype=np.float32), 10, 10)
    with pytest.raises(InvalidMeshRequest, match=ARTWORK_DETAIL):
        build_top_surface(finite, 0, 10)
    with pytest.raises(InvalidMeshRequest, match=ARTWORK_DETAIL):
        build_top_surface(finite, 10, -1)
    with pytest.raises(InvalidMeshRequest, match=ARTWORK_DETAIL):
        build_top_surface(finite, float("nan"), 10)
    with pytest.raises(InvalidMeshRequest, match=BUDGET_DETAIL):
        build_top_surface(finite, 10, 10, max_vertices=3)
    assert finite.tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_downsampling_is_deterministic_and_keeps_artwork_endpoints() -> None:
    array = np.arange(45, dtype=np.float32).reshape(9, 5)
    first = build_top_surface(array, 80, 40, max_vertices=12)
    second = build_top_surface(array, 80, 40, max_vertices=12)
    assert first.downsampling_required is True
    assert first.vertex_count <= 12
    assert first.vertex_count == second.vertex_count
    assert first.row_indices.tolist() == second.row_indices.tolist()
    assert first.column_indices.tolist() == second.column_indices.tolist()
    assert int(first.row_indices[0]) == 0
    assert int(first.row_indices[-1]) == 8
    assert int(first.column_indices[0]) == 0
    assert int(first.column_indices[-1]) == 4
    assert first.vertices[0, 0] == 0
    assert first.vertices[:, 0].max() == 80
    assert first.vertices[:, 1].min() == 0
    assert first.vertices[:, 1].max() == 40
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.triangles, second.triangles)
    assert np.all(first.normals[:, 2] > 0)
    sampled = array[np.ix_(first.row_indices, first.column_indices)].reshape(-1)
    np.testing.assert_allclose(first.vertices[:, 2], sampled)


def test_large_preview_plan_stays_inside_the_vertex_budget() -> None:
    plan = plan_preview_grid(2140, 2940, DEFAULT_MAX_VERTICES)
    assert plan.downsampling_required is True
    assert plan.vertex_count <= DEFAULT_MAX_VERTICES
    assert plan.vertex_count == plan.mesh_height_px * plan.mesh_width_px
    assert plan.triangle_count == 2 * (plan.mesh_height_px - 1) * (plan.mesh_width_px - 1)
    assert plan.estimated_bytes == mesh_memory_bytes(plan.vertex_count, plan.triangle_count)
    rows = endpoint_indices(2140, plan.mesh_height_px)
    columns = endpoint_indices(2940, plan.mesh_width_px)
    assert int(rows[0]) == 0 and int(rows[-1]) == 2139
    assert int(columns[0]) == 0 and int(columns[-1]) == 2939
    assert len(np.unique(rows)) == len(rows)
    assert len(np.unique(columns)) == len(columns)
    again = plan_preview_grid(2140, 2940)
    assert (again.mesh_height_px, again.mesh_width_px) == (plan.mesh_height_px, plan.mesh_width_px)

    small = plan_preview_grid(3, 4, 250_000)
    assert small.downsampling_required is False
    assert small.vertex_count == 12
    assert small.triangle_count == 12


def test_preview_description_reports_status_without_triangle_buffers() -> None:
    array = np.array([[1.0, 2.0], [3.0, 8.0]], dtype=np.float32)
    info = describe_mesh_preview(
        project_id=uuid4(),
        status="stale",
        array=array,
        artwork_width_mm=12,
        artwork_height_mm=9,
        stock_thickness_mm=19,
    )
    assert info.status == "stale"
    assert info.surface == "top"
    assert info.downsampling_required is False
    assert info.vertex_count == 4
    assert info.triangle_count == 2
    assert info.z_minimum_mm == 1
    assert info.z_maximum_mm == 8
    assert info.artwork_width_mm == 12
    assert info.artwork_height_mm == 9
    assert info.estimated_bytes == mesh_memory_bytes(4, 2)
    assert info.validation.ok is True
    assert "vertices" not in info.model_dump()
    payload = info.model_dump(mode="json")
    assert payload["status"] == "stale"
    assert len(json.dumps(payload)) < 20_000


def test_mesh_preview_endpoint_is_read_only(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        project_id = body["id"]
        missing = client.get(f"/api/projects/{uuid4()}/relief/mesh/preview-info")
        assert missing.status_code == 404

        before = _files(project_root)
        current = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert current.status_code == 200, current.text
        payload = current.json()
        height = np.load(_root(project_root) / "depth/composition/physical/physical_height_mm.npy")
        expected = plan_preview_grid(int(height.shape[0]), int(height.shape[1]), DEFAULT_MAX_VERTICES)
        assert payload["status"] == "current"
        assert payload["project_id"] == project_id
        assert payload["surface"] == "top"
        assert payload["source_height_px"] == int(height.shape[0])
        assert payload["source_width_px"] == int(height.shape[1])
        assert payload["preview_height_px"] == expected.mesh_height_px
        assert payload["preview_width_px"] == expected.mesh_width_px
        assert payload["vertex_count"] == expected.vertex_count
        assert payload["triangle_count"] == expected.triangle_count
        assert payload["downsampling_required"] == expected.downsampling_required
        assert payload["artwork_width_mm"] == body["artwork"]["width_mm"]
        assert payload["artwork_height_mm"] == body["artwork"]["height_mm"]
        assert payload["z_minimum_mm"] == float(height.min())
        assert payload["z_maximum_mm"] == float(height.max())
        assert payload["estimated_bytes"] == expected.estimated_bytes
        assert "vertices" not in payload
        assert "triangles" not in payload
        assert payload["validation"]["ok"] is True
        assert _files(project_root) == before
        assert list(_root(project_root).rglob("*mesh*")) == []
        assert client.post(f"/api/projects/{project_id}/relief/mesh/preview-info").status_code == 405

        limited = client.get(
            f"/api/projects/{project_id}/relief/mesh/preview-info",
            params={"max_vertices": 4},
        )
        assert limited.status_code == 200, limited.text
        limited_body = limited.json()
        assert limited_body["vertex_count"] <= 4
        assert limited_body["max_vertices"] == 4
        assert limited_body["status"] == "current"
        assert _files(project_root) == before

        manifest_path = _root(project_root) / "project.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"]["content_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        stale = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert stale.status_code == 200, stale.text
        assert stale.json()["status"] == "stale"
        assert stale.json()["status"] != "current"

        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"].pop("content_sha256", None)
        stored["relief"]["physical"].pop("settings_sha256", None)
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        unknown = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert unknown.status_code == 200, unknown.text
        assert unknown.json()["status"] == "unknown"

        height_path = _root(project_root) / "depth/composition/physical/physical_height_mm.npy"
        original = height_path.read_bytes()
        np.save(height_path, np.zeros((1, 4), dtype=np.float32))
        rejected = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == SHAPE_DETAIL
        np.save(height_path, np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32))
        # Shape must match the record before the non-finite check.
        record_height = stored["relief"]["physical"]["height_px"]
        record_width = stored["relief"]["physical"]["width_px"]
        broken = np.zeros((record_height, record_width), dtype=np.float32)
        broken[0, 0] = np.nan
        np.save(height_path, broken)
        non_finite = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert non_finite.status_code == 422
        assert non_finite.json()["detail"] == NON_FINITE_DETAIL
        height_path.write_bytes(original)

        height_path.unlink()
        absent = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
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
        no_record = client.get(f"/api/projects/{created.json()['id']}/relief/mesh/preview-info")
        assert no_record.status_code == 404
        assert no_record.json()["detail"] == RECORD_MISSING_DETAIL
