"""Watertight solid construction. Temporary projects and synthetic arrays only."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.projects.errors import InvalidMeshRequest
from app.relief.diagnostics import (
    ABOVE_STOCK_WARNING,
    FILE_MISSING_DETAIL,
    RECORD_MISSING_DETAIL,
    SHAPE_DETAIL,
    STRUCTURE_DETAIL,
)
from app.relief.mesh import (
    ARTWORK_DETAIL,
    AXIS_DETAIL,
    BUDGET_DETAIL,
    DEFAULT_MAX_VERTICES,
    NON_FINITE_DETAIL,
    build_top_surface,
    mesh_memory_bytes,
)
from app.relief.solid import (
    HEIGHT_SEPARATION_TOLERANCE_MM,
    NON_POSITIVE_DETAIL,
    ROLE_BOTTOM,
    ROLE_TOP,
    ROLE_X0,
    ROLE_X1,
    ROLE_Y0,
    ROLE_Y1,
    SELF_INTERSECTION_TRIANGLE_CAP,
    build_solid_mesh,
    count_nonadjacent_intersections,
    describe_solid,
    plan_solid_grid,
    validate_solid_topology,
)
from tests.test_physical_diagnostics import _files
from tests.test_projects import served
from tests.test_provenance import _pipeline, _root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _triangle_formula(height: int, width: int) -> int:
    return 4 * (height - 1) * (width - 1) + 4 * (width - 1) + 4 * (height - 1)


def test_constant_block_is_a_closed_outward_solid() -> None:
    array = np.full((2, 2), 5, dtype=np.float32)
    before = array.copy()
    mesh = build_solid_mesh(array, 10, 20)
    assert np.array_equal(array, before)
    assert mesh.vertex_count == 8
    assert mesh.top_vertex_count == 4
    assert mesh.triangle_count == 12
    assert mesh.triangle_count == _triangle_formula(2, 2)
    assert int(np.count_nonzero(mesh.face_roles == ROLE_TOP)) == 2
    assert int(np.count_nonzero(mesh.face_roles == ROLE_BOTTOM)) == 2
    assert int(np.count_nonzero(mesh.face_roles == ROLE_X0)) == 2
    assert int(np.count_nonzero(mesh.face_roles == ROLE_X1)) == 2
    assert int(np.count_nonzero(mesh.face_roles == ROLE_Y0)) == 2
    assert int(np.count_nonzero(mesh.face_roles == ROLE_Y1)) == 2
    assert mesh.vertices.shape == (8, 3)
    assert np.array_equal(mesh.vertices[4:, :2], mesh.vertices[:4, :2])
    assert np.all(mesh.vertices[4:, 2] == 0)
    assert np.all(mesh.vertices[:4, 2] == 5)
    assert len(np.unique(np.round(mesh.vertices, 9), axis=0)) == 8
    assert mesh.bounding_box_mm == ((0.0, 0.0, 0.0), (10.0, 20.0, 5.0))
    assert mesh.signed_volume_mm3 == pytest.approx(10 * 20 * 5)
    assert mesh.topology.passed is True
    assert mesh.topology.edge_manifold is True
    assert mesh.topology.boundary_edge_count == 0
    assert mesh.topology.non_manifold_edge_count == 0
    assert mesh.topology.inconsistent_winding_edge_count == 0
    assert mesh.topology.component_count == 1
    assert mesh.topology.zero_area_triangle_count == 0
    assert mesh.topology.self_intersection_checked is True
    assert mesh.topology.self_intersection_count == 0
    assert "export-ready" in mesh.topology.limitation
    assert np.allclose(mesh.normals[mesh.face_roles == ROLE_TOP], [0.0, 0.0, 1.0])
    assert np.allclose(mesh.normals[mesh.face_roles == ROLE_BOTTOM], [0.0, 0.0, -1.0])
    assert np.allclose(mesh.normals[mesh.face_roles == ROLE_X0], [-1.0, 0.0, 0.0])
    assert np.allclose(mesh.normals[mesh.face_roles == ROLE_X1], [1.0, 0.0, 0.0])
    assert np.allclose(mesh.normals[mesh.face_roles == ROLE_Y0], [0.0, -1.0, 0.0])
    assert np.allclose(mesh.normals[mesh.face_roles == ROLE_Y1], [0.0, 1.0, 0.0])
    assert (
        mesh.vertices.nbytes + mesh.triangles.nbytes + mesh.normals.nbytes
        == mesh_memory_bytes(mesh.vertex_count, mesh.triangle_count)
    )
    open_top = build_top_surface(np.zeros((2, 2), dtype=np.float32), 10, 20)
    assert open_top.triangle_count == 2
    assert np.all(open_top.normals[:, 2] > 0)


def test_sloped_peak_and_non_square_solids_keep_image_orientation() -> None:
    slope = np.array([[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    sloped = build_solid_mesh(slope, 20, 10)
    assert sloped.mesh_height_px == 2
    assert sloped.mesh_width_px == 3
    assert sloped.triangle_count == _triangle_formula(2, 3)
    assert sloped.vertices[0].tolist() == [0.0, 0.0, 2.0]
    assert sloped.vertices[2].tolist() == [20.0, 0.0, 6.0]
    assert sloped.vertices[3].tolist() == [0.0, 10.0, 2.0]
    assert sloped.signed_volume_mm3 == pytest.approx(20 * 10 * 4)
    assert np.all(sloped.normals[sloped.face_roles == ROLE_TOP, 2] > 0)
    assert np.all(sloped.normals[sloped.face_roles == ROLE_BOTTOM, 2] < 0)
    assert np.all(sloped.normals[sloped.face_roles == ROLE_X0, 0] < 0)
    assert np.all(sloped.normals[sloped.face_roles == ROLE_X1, 0] > 0)
    assert np.all(sloped.normals[sloped.face_roles == ROLE_Y0, 1] < 0)
    assert np.all(sloped.normals[sloped.face_roles == ROLE_Y1, 1] > 0)
    assert sloped.topology.passed is True
    assert sloped.topology.self_intersection_count == 0

    peak = np.ones((3, 3), dtype=np.float32)
    peak[1, 1] = 4
    raised = build_solid_mesh(peak, 9, 6)
    assert raised.vertices[0].tolist() == [0.0, 0.0, 1.0]
    assert raised.vertices[4, 2] == 4
    assert raised.vertices[6].tolist() == [0.0, 6.0, 1.0]
    assert raised.bounding_box_mm == ((0.0, 0.0, 0.0), (9.0, 6.0, 4.0))
    assert 9 * 6 * 1 < raised.signed_volume_mm3 < 9 * 6 * 4
    assert raised.topology.component_count == 1
    assert raised.topology.self_intersection_checked is True

    wide = np.full((2, 5), 2.5, dtype=np.float32)
    block = build_solid_mesh(wide, 50, 10)
    assert block.vertex_count == 20
    assert block.triangle_count == _triangle_formula(2, 5)
    assert block.bounding_box_mm == ((0.0, 0.0, 0.0), (50.0, 10.0, 2.5))
    assert block.signed_volume_mm3 == pytest.approx(50 * 10 * 2.5)
    assert block.topology.boundary_edge_count == 0
    assert block.topology.inconsistent_winding_edge_count == 0


def test_non_positive_and_non_finite_heights_are_rejected_without_changes() -> None:
    valid = np.full((2, 3), 1.5, dtype=np.float32)
    for value in (0.0, -0.25):
        array = valid.copy()
        array[0, 2] = value
        before = array.copy()
        with pytest.raises(InvalidMeshRequest, match=NON_POSITIVE_DETAIL):
            build_solid_mesh(array, 12, 8)
        assert np.array_equal(array, before)
    at_tolerance = np.full((2, 2), HEIGHT_SEPARATION_TOLERANCE_MM, dtype=np.float64)
    with pytest.raises(InvalidMeshRequest, match=NON_POSITIVE_DETAIL):
        build_solid_mesh(at_tolerance, 4, 4)
    above = np.full((2, 2), np.nextafter(HEIGHT_SEPARATION_TOLERANCE_MM, 1.0), dtype=np.float64)
    built = build_solid_mesh(above, 4, 4)
    assert built.topology.passed is True
    assert built.signed_volume_mm3 > 0

    dirty = np.array([[1.0, 1.0], [1.0, np.nan]], dtype=np.float32)
    before = dirty.copy()
    with pytest.raises(InvalidMeshRequest, match=NON_FINITE_DETAIL):
        build_solid_mesh(dirty, 10, 10)
    assert np.array_equal(dirty, before, equal_nan=True)
    dirty[0, 0] = np.inf
    with pytest.raises(InvalidMeshRequest, match=NON_FINITE_DETAIL):
        build_solid_mesh(dirty, 10, 10)
    assert np.isnan(dirty[1, 1])
    assert np.isinf(dirty[0, 0])

    with pytest.raises(InvalidMeshRequest, match=AXIS_DETAIL):
        build_solid_mesh(np.ones((1, 4), dtype=np.float32), 10, 10)
    with pytest.raises(InvalidMeshRequest, match=STRUCTURE_DETAIL):
        build_solid_mesh(np.ones((4,), dtype=np.float32), 10, 10)
    with pytest.raises(InvalidMeshRequest, match=ARTWORK_DETAIL):
        build_solid_mesh(valid, 0, 10)
    with pytest.raises(InvalidMeshRequest, match=BUDGET_DETAIL):
        build_solid_mesh(valid, 10, 10, max_vertices=3)
    assert valid.tolist() == [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]


def test_above_stock_height_is_warned_and_not_clamped() -> None:
    array = np.full((2, 2), 8, dtype=np.float32)
    before = array.copy()
    mesh = build_solid_mesh(array, 10, 6, stock_thickness_mm=5)
    assert ABOVE_STOCK_WARNING in mesh.warnings
    assert np.all(mesh.vertices[:4, 2] == 8)
    assert mesh.bounding_box_mm[1][2] == 8
    assert np.array_equal(array, before)


def test_downsampling_is_deterministic_and_keeps_artwork_endpoints() -> None:
    array = np.arange(1, 46, dtype=np.float32).reshape(9, 5)
    array[4, 2] = 90
    first = build_solid_mesh(array, 80, 40, max_vertices=12)
    second = build_solid_mesh(array, 80, 40, max_vertices=12)
    assert first.downsampling_required is True
    assert first.top_vertex_count <= 12
    assert first.vertex_count == 2 * first.top_vertex_count
    assert first.triangle_count == _triangle_formula(first.mesh_height_px, first.mesh_width_px)
    assert first.row_indices.tolist() == second.row_indices.tolist()
    assert first.column_indices.tolist() == second.column_indices.tolist()
    assert int(first.row_indices[0]) == 0
    assert int(first.row_indices[-1]) == 8
    assert int(first.column_indices[0]) == 0
    assert int(first.column_indices[-1]) == 4
    assert first.vertices[:, 0].min() == 0
    assert first.vertices[:, 0].max() == 80
    assert first.vertices[:, 1].min() == 0
    assert first.vertices[:, 1].max() == 40
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.triangles, second.triangles)
    assert first.topology.passed is True
    assert first.bounding_box_mm[1][2] < 90


def test_validation_catches_an_open_mesh_and_real_intersections() -> None:
    mesh = build_solid_mesh(np.full((3, 3), 2, dtype=np.float32), 12, 9)
    opened = validate_solid_topology(
        mesh.vertices,
        mesh.triangles[:-1],
        mesh.face_roles[:-1],
        mesh_height_px=3,
        mesh_width_px=3,
        artwork_width_mm=12,
        artwork_height_mm=9,
    )
    assert opened.passed is False
    assert opened.boundary_edge_count > 0
    assert opened.component_count >= 1

    pair = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    separate = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
        dtype=np.float64,
    )
    overlap = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 0], [1.2, 0.2, 0], [0.2, 1.2, 0]],
        dtype=np.float64,
    )
    crossing = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.2, -1], [0.5, 0.8, -1], [0.5, 0.5, 1]],
        dtype=np.float64,
    )
    shared = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    assert count_nonadjacent_intersections(separate, pair) == 0
    assert count_nonadjacent_intersections(overlap, pair) == 1
    assert count_nonadjacent_intersections(crossing, pair) == 1
    assert count_nonadjacent_intersections(shared, np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)) == 0
    skipped = np.zeros((SELF_INTERSECTION_TRIANGLE_CAP + 1, 3), dtype=np.int32)
    assert count_nonadjacent_intersections(np.zeros((3, 3)), skipped) is None

    dense = build_solid_mesh(np.full((8, 8), 1.25, dtype=np.float32), 16, 16)
    assert dense.triangle_count > SELF_INTERSECTION_TRIANGLE_CAP
    assert dense.topology.self_intersection_checked is False
    assert dense.topology.self_intersection_count is None
    assert dense.topology.passed is True
    assert dense.topology.edge_manifold is True
    assert dense.signed_volume_mm3 == pytest.approx(16 * 16 * 1.25)
    assert "export-ready" in dense.topology.limitation


def test_solid_plan_estimates_memory_without_building_a_large_mesh() -> None:
    plan = plan_solid_grid(2140, 2940, DEFAULT_MAX_VERTICES)
    assert plan.downsampling_required is True
    assert plan.top_vertex_count <= DEFAULT_MAX_VERTICES
    assert plan.vertex_count == 2 * plan.mesh_height_px * plan.mesh_width_px
    assert plan.bottom_vertex_count == plan.top_vertex_count
    assert plan.top_triangle_count == 2 * (plan.mesh_height_px - 1) * (plan.mesh_width_px - 1)
    assert plan.bottom_triangle_count == plan.top_triangle_count
    assert plan.sidewall_triangle_count == 4 * (plan.mesh_width_px - 1) + 4 * (plan.mesh_height_px - 1)
    assert plan.triangle_count == _triangle_formula(plan.mesh_height_px, plan.mesh_width_px)
    assert plan.estimated_mesh_bytes == mesh_memory_bytes(plan.vertex_count, plan.triangle_count)
    assert plan.estimated_mesh_bytes < 80_000_000

    small = plan_solid_grid(3, 4)
    mesh = build_solid_mesh(np.full((3, 4), 2, dtype=np.float32), 30, 12)
    assert mesh.vertex_count == small.vertex_count
    assert mesh.triangle_count == small.triangle_count
    assert (
        mesh.vertices.nbytes + mesh.triangles.nbytes + mesh.normals.nbytes
        == small.estimated_mesh_bytes
    )
    assert mesh.signed_volume_mm3 == pytest.approx(30 * 12 * 2)


def test_solid_description_preserves_status_and_does_not_claim_validation() -> None:
    array = np.array([[1.0, 2.0], [3.0, 8.0]], dtype=np.float32)
    info = describe_solid(
        project_id=uuid4(),
        status="stale",
        array=array,
        artwork_width_mm=12,
        artwork_height_mm=9,
        stock_thickness_mm=19,
    )
    assert info.status == "stale"
    assert info.status != "current"
    assert info.solid == "watertight-relief"
    assert info.topology_validated is False
    assert info.geometry_construction_available is True
    assert "have not been run" in info.topology_note
    assert info.vertex_count == 8
    assert info.triangle_count == _triangle_formula(2, 2)
    assert info.bounding_box_minimum_mm == [0.0, 0.0, 0.0]
    assert info.bounding_box_maximum_mm == [12.0, 9.0, 8.0]
    assert info.source_z_maximum_mm == 8
    assert info.selected_z_maximum_mm == 8
    assert info.estimated_mesh_bytes == mesh_memory_bytes(8, 12)
    assert "loaded physical-height array" in info.memory_note
    payload = info.model_dump(mode="json")
    assert "vertices" not in payload
    assert "triangles" not in payload
    assert payload["topology_validated"] is False

    peak = np.ones((5, 5), dtype=np.float32)
    peak[2, 2] = 9
    omitted = describe_solid(
        project_id=uuid4(),
        status="unknown",
        array=peak,
        artwork_width_mm=20,
        artwork_height_mm=20,
        max_vertices=4,
        stock_thickness_mm=4,
    )
    assert omitted.status == "unknown"
    assert omitted.downsampling_required is True
    assert omitted.selected_z_maximum_mm == 1
    assert omitted.source_z_maximum_mm == 9
    assert omitted.bounding_box_maximum_mm[2] == 1
    assert ABOVE_STOCK_WARNING in omitted.validation.warnings
    assert omitted.geometry_construction_available is True

    rejected = describe_solid(
        project_id=uuid4(),
        status="current",
        array=np.zeros((2, 2), dtype=np.float32),
        artwork_width_mm=10,
        artwork_height_mm=10,
    )
    assert rejected.status == "current"
    assert rejected.geometry_construction_available is False
    assert rejected.topology_validated is False
    assert rejected.bounding_box_minimum_mm is None
    assert rejected.validation.ok is False
    assert NON_POSITIVE_DETAIL in rejected.validation.errors
    assert rejected.triangle_count == _triangle_formula(2, 2)

    broken = np.ones((2, 2), dtype=np.float32)
    broken[1, 0] = np.nan
    unavailable = describe_solid(
        project_id=uuid4(),
        status="unknown",
        array=broken,
        artwork_width_mm=10,
        artwork_height_mm=10,
    )
    assert unavailable.status == "unknown"
    assert unavailable.geometry_construction_available is False
    assert unavailable.source_z_minimum_mm is None
    assert NON_FINITE_DETAIL in unavailable.validation.errors
    assert "vertices" not in unavailable.model_dump()


def test_solid_info_endpoint_is_read_only(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        project_id = body["id"]
        missing = client.get(f"/api/projects/{uuid4()}/relief/mesh/solid-info")
        assert missing.status_code == 404

        before = _files(project_root)
        preview = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert preview.status_code == 200
        assert preview.json()["surface"] == "top"
        current = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
        assert current.status_code == 200, current.text
        payload = current.json()
        height = np.load(_root(project_root) / "depth/composition/physical/physical_height_mm.npy")
        expected = plan_solid_grid(int(height.shape[0]), int(height.shape[1]), DEFAULT_MAX_VERTICES)
        assert payload["status"] == "current"
        assert payload["project_id"] == project_id
        assert payload["solid"] == "watertight-relief"
        assert payload["topology_validated"] is False
        assert payload["geometry_construction_available"] is True
        assert "have not been run" in payload["topology_note"]
        assert payload["source_height_px"] == int(height.shape[0])
        assert payload["source_width_px"] == int(height.shape[1])
        assert payload["mesh_height_px"] == expected.mesh_height_px
        assert payload["mesh_width_px"] == expected.mesh_width_px
        assert payload["vertex_count"] == expected.vertex_count
        assert payload["triangle_count"] == expected.triangle_count
        assert payload["estimated_mesh_bytes"] == expected.estimated_mesh_bytes
        assert payload["bounding_box_minimum_mm"] == [0.0, 0.0, 0.0]
        assert payload["bounding_box_maximum_mm"][0] == body["artwork"]["width_mm"]
        assert payload["bounding_box_maximum_mm"][1] == body["artwork"]["height_mm"]
        assert "vertices" not in payload
        assert "triangles" not in payload
        assert _files(project_root) == before
        assert list(_root(project_root).rglob("*mesh*")) == []
        assert client.post(f"/api/projects/{project_id}/relief/mesh/solid-info").status_code == 405

        limited = client.get(
            f"/api/projects/{project_id}/relief/mesh/solid-info",
            params={"max_vertices": 4},
        )
        assert limited.status_code == 200, limited.text
        limited_body = limited.json()
        assert limited_body["top_vertex_count"] <= 4
        assert limited_body["vertex_count"] == 2 * limited_body["top_vertex_count"]
        assert limited_body["status"] == "current"
        assert limited_body["topology_validated"] is False
        assert _files(project_root) == before

        manifest_path = _root(project_root) / "project.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"]["content_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        stale = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
        assert stale.status_code == 200, stale.text
        assert stale.json()["status"] == "stale"
        assert stale.json()["status"] != "current"
        assert stale.json()["topology_validated"] is False

        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"].pop("content_sha256", None)
        stored["relief"]["physical"].pop("settings_sha256", None)
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        unknown = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
        assert unknown.status_code == 200, unknown.text
        assert unknown.json()["status"] == "unknown"

        height_path = _root(project_root) / "depth/composition/physical/physical_height_mm.npy"
        original = height_path.read_bytes()
        np.save(height_path, np.zeros((1, 4), dtype=np.float32))
        rejected = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == SHAPE_DETAIL

        record_height = stored["relief"]["physical"]["height_px"]
        record_width = stored["relief"]["physical"]["width_px"]
        zeros = np.zeros((record_height, record_width), dtype=np.float32)
        np.save(height_path, zeros)
        non_positive = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
        assert non_positive.status_code == 200, non_positive.text
        non_positive_body = non_positive.json()
        assert non_positive_body["geometry_construction_available"] is False
        assert non_positive_body["topology_validated"] is False
        assert non_positive_body["status"] != "current"
        assert NON_POSITIVE_DETAIL in non_positive_body["validation"]["errors"]

        broken = np.ones((record_height, record_width), dtype=np.float32)
        broken[0, 0] = np.nan
        np.save(height_path, broken)
        non_finite = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
        assert non_finite.status_code == 200, non_finite.text
        assert NON_FINITE_DETAIL in non_finite.json()["validation"]["errors"]
        assert non_finite.json()["topology_validated"] is False
        preview_non_finite = client.get(f"/api/projects/{project_id}/relief/mesh/preview-info")
        assert preview_non_finite.status_code == 422
        assert preview_non_finite.json()["detail"] == NON_FINITE_DETAIL
        height_path.write_bytes(original)

        height_path.unlink()
        absent = client.get(f"/api/projects/{project_id}/relief/mesh/solid-info")
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
        no_record = client.get(f"/api/projects/{created.json()['id']}/relief/mesh/solid-info")
        assert no_record.status_code == 404
        assert no_record.json()["detail"] == RECORD_MISSING_DETAIL
