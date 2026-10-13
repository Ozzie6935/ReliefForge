"""Read-only STL export planning. Temporary projects and synthetic arrays only."""

import json
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.projects.errors import InvalidExportRequest
from app.relief.export_plan import (
    ABOVE_STOCK_DETAIL,
    FILE_MISSING_DETAIL,
    MAX_EXPORT_PEAK_MEMORY_BYTES,
    MAX_EXPORT_STL_BYTES,
    MAX_EXPORT_TRIANGLES,
    NATIVE_REJECTS_GRID,
    NON_FINITE_DETAIL,
    NON_POSITIVE_DETAIL,
    RECORD_MISSING_DETAIL,
    REDUCED_NOT_SMALLER,
    REDUCED_REQUIRES_GRID,
    ExportResourceLimits,
    axis_positions,
    binary_stl_bytes,
    build_export_readiness,
    solid_triangle_count,
    solid_vertex_count,
    validate_export_request,
)
from app.relief.mesh import _axis_coordinates
from app.relief.physical import HEIGHT_PATH, PREVIEW_PATH, STATS_PATH
from app.relief.solid import plan_solid_grid
from app.schemas.relief import PhysicalHeightRecord
from tests.test_physical_diagnostics import _files
from tests.test_projects import create_project, sample_body, served
from tests.test_provenance import _pipeline, _root


def _record(
    width: int,
    height: int,
    stock: float = 19,
    base: float = 5,
    peak: float = 17,
) -> PhysicalHeightRecord:
    moment = datetime(2026, 10, 8, tzinfo=timezone.utc)
    return PhysicalHeightRecord(
        stock_thickness_mm=stock,
        base_thickness_mm=base,
        maximum_surface_height_mm=peak,
        generated_at=moment,
        height_path=HEIGHT_PATH,
        preview_path=PREVIEW_PATH,
        stats_path=STATS_PATH,
        source_relief_sha256="abc",
        source_relief_generated_at=moment,
        width_px=width,
        height_px=height,
        dtype="float32",
        height_min_mm=base,
        height_max_mm=peak,
        maximum_cut_depth_mm=stock - base,
        relief_amplitude_mm=peak - base,
    )


def _assess(array: np.ndarray | None, **kwargs):
    height = 4 if array is None else int(array.shape[0])
    width = 6 if array is None else int(array.shape[1])
    record = kwargs.pop("record", _record(width, height))
    return build_export_readiness(
        project_id=kwargs.pop("project_id", uuid4()),
        provenance_status=kwargs.pop("provenance_status", "current"),
        mode=kwargs.pop("mode", "native"),
        artwork_width_mm=kwargs.pop("artwork_width_mm", 90.0),
        artwork_height_mm=kwargs.pop("artwork_height_mm", 40.0),
        stock_width_mm=kwargs.pop("stock_width_mm", 120.0),
        stock_height_mm=kwargs.pop("stock_height_mm", 80.0),
        center_x_mm=kwargs.pop("center_x_mm", 50.0),
        center_y_mm=kwargs.pop("center_y_mm", 30.0),
        record=record,
        file_present=kwargs.pop("file_present", array is not None),
        array=array,
        **kwargs,
    )


def _codes(report) -> list[str]:
    return [issue.code for issue in report.blocking_errors]


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def test_native_resolution_matches_the_solid_count_and_exact_extents() -> None:
    array = np.full((5, 9), 8, dtype=np.float32)
    report = _assess(
        array,
        artwork_width_mm=100,
        artwork_height_mm=40,
        stock_width_mm=140,
        stock_height_mm=70,
    )
    solid = plan_solid_grid(5, 9, max_vertices=10_000)
    assert report.mode == "native"
    assert report.export_target == "artwork-local"
    assert report.export_certificate == "none"
    assert report.ready_for_export is True
    assert report.source_grid is not None and report.planned_grid is not None
    assert report.planned_grid.height_px == 5
    assert report.planned_grid.width_px == 9
    assert report.estimated_vertices == solid.vertex_count == solid_vertex_count(5, 9)
    assert report.estimated_triangles == solid.triangle_count == solid_triangle_count(5, 9)
    assert report.estimated_stl_bytes == binary_stl_bytes(solid.triangle_count)
    assert report.spacing_mm is not None
    assert report.spacing_mm.uniform is True
    assert report.spacing_mm.x_mm == pytest.approx(100 / 8)
    assert report.spacing_mm.y_mm == pytest.approx(40 / 4)
    assert report.spacing_mm.x_mm * 8 == pytest.approx(100)
    assert report.spacing_mm.y_mm * 4 == pytest.approx(40)
    columns = np.arange(9)
    rows = np.arange(5)
    assert np.array_equal(axis_positions(columns, 9, 100), _axis_coordinates(columns, 9, 100))
    assert np.array_equal(axis_positions(rows, 5, 40), _axis_coordinates(rows, 5, 40))
    assert float(axis_positions(columns, 9, 100)[0]) == 0
    assert float(axis_positions(columns, 9, 100)[-1]) == 100
    assert float(axis_positions(rows, 5, 40)[-1]) == 40
    assert report.stock_width_mm == 140
    assert report.stock_height_mm == 70
    assert report.artwork_width_mm == 100
    assert report.artwork_height_mm == 40
    assert report.placement.recorded is True
    assert report.placement.applied is False
    assert report.placement.center_x_mm == 50
    assert report.placement.center_y_mm == 30
    assert report.placement.origin == "front-left-stock"
    assert report.z_minimum_mm == 8
    assert report.z_maximum_mm == 8
    assert report.topology_validated is False
    assert report.self_intersection_checked is False
    assert report.zero_area_measured is False
    assert "self_intersection_not_checked" in [item.code for item in report.warnings]
    assert "reduced_omits_features" not in [item.code for item in report.warnings]


def test_reduced_grid_is_explicit_and_keeps_artwork_extents() -> None:
    array = np.linspace(2, 6, 30, dtype=np.float32).reshape(5, 6)
    reduced = _assess(array, mode="reduced", export_height_px=3, export_width_px=4, artwork_width_mm=100)
    native = _assess(array, artwork_width_mm=100)
    assert reduced.mode == "reduced"
    assert reduced.planned_grid is not None
    assert (reduced.planned_grid.height_px, reduced.planned_grid.width_px) == (3, 4)
    assert (native.planned_grid.height_px, native.planned_grid.width_px) == (5, 6)
    assert reduced.estimated_triangles == solid_triangle_count(3, 4)
    assert reduced.estimated_triangles < native.estimated_triangles
    assert reduced.spacing_mm is not None
    assert reduced.spacing_mm.x_mm * 3 == pytest.approx(100)
    assert reduced.spacing_mm.y_mm * 2 == pytest.approx(40)
    positions = axis_positions(np.array([0, 1, 3, 5]), 6, 100)
    assert float(positions[0]) == 0
    assert float(positions[-1]) == 100
    assert reduced.spacing_mm.uniform is False
    assert "reduced_omits_features" in [item.code for item in reduced.warnings]
    assert reduced.ready_for_export is True
    with pytest.raises(InvalidExportRequest) as equal:
        _assess(array, mode="reduced", export_height_px=5, export_width_px=6)
    assert str(equal.value) == REDUCED_NOT_SMALLER
    with pytest.raises(InvalidExportRequest) as larger:
        _assess(array, mode="reduced", export_height_px=5, export_width_px=7)
    assert str(larger.value) == REDUCED_NOT_SMALLER


def test_native_does_not_downgrade_when_a_limit_is_exceeded() -> None:
    array = np.full((8, 8), 4, dtype=np.float32)
    report = _assess(
        array,
        limits=ExportResourceLimits(
            max_triangles=10,
            max_stl_bytes=10,
            max_peak_memory_bytes=10,
        ),
    )
    assert report.mode == "native"
    assert report.planned_grid is not None
    assert (report.planned_grid.height_px, report.planned_grid.width_px) == (8, 8)
    assert report.estimated_triangles == solid_triangle_count(8, 8)
    assert report.ready_for_export is False
    assert _codes(report) == ["triangle_limit", "stl_byte_limit", "memory_limit"]
    assert any("not reduced automatically" in action for action in report.corrective_actions)


def test_each_resource_limit_is_reported_on_its_own() -> None:
    array = np.full((4, 4), 3, dtype=np.float32)
    triangles = solid_triangle_count(4, 4)
    stl_only = _assess(
        array,
        limits=ExportResourceLimits(
            max_triangles=triangles + 10,
            max_stl_bytes=10,
            max_peak_memory_bytes=MAX_EXPORT_PEAK_MEMORY_BYTES,
        ),
    )
    memory_only = _assess(
        array,
        limits=ExportResourceLimits(
            max_triangles=MAX_EXPORT_TRIANGLES,
            max_stl_bytes=MAX_EXPORT_STL_BYTES,
            max_peak_memory_bytes=1,
        ),
    )
    assert _codes(stl_only) == ["stl_byte_limit"]
    assert _codes(memory_only) == ["memory_limit"]
    assert stl_only.planned_grid.width_px == 4
    assert memory_only.estimated_peak_memory_bytes is not None
    assert memory_only.estimated_peak_memory_bytes > 1


def test_height_failures_block_and_above_stock_is_not_a_warning() -> None:
    stock = 10
    record = _record(3, 2, stock=stock, base=2, peak=9)
    above = _assess(np.full((2, 3), 10.1, dtype=np.float32), record=record)
    equal = _assess(np.full((2, 3), 10, dtype=np.float32), record=record)
    zero = _assess(np.zeros((2, 3), dtype=np.float32), record=record)
    negative = _assess(np.full((2, 3), -1, dtype=np.float32), record=record)
    tiny = _assess(np.full((2, 3), np.float32(1e-12), dtype=np.float32), record=record)
    nan = np.full((2, 3), 4, dtype=np.float32)
    nan[0, 1] = np.nan
    infinite = np.full((2, 3), 4, dtype=np.float32)
    infinite[1, 2] = np.inf
    assert above.ready_for_export is False
    assert _codes(above) == ["above_stock"]
    assert above.blocking_errors[0].message == ABOVE_STOCK_DETAIL
    assert "above_stock" not in [item.code for item in above.warnings]
    assert equal.ready_for_export is True
    assert _codes(zero) == ["non_positive_height"]
    assert zero.blocking_errors[0].message == NON_POSITIVE_DETAIL
    assert _codes(negative) == ["non_positive_height"]
    assert _codes(tiny) == ["non_positive_height"]
    assert _codes(_assess(nan, record=record)) == ["non_finite_height"]
    assert _assess(nan, record=record).blocking_errors[0].message == NON_FINITE_DETAIL
    assert _codes(_assess(infinite, record=record)) == ["non_finite_height"]
    assert _assess(np.full((2, 3), 4, dtype=np.float64), record=record).ready_for_export is True


def test_provenance_and_missing_artifacts_are_not_export_ready() -> None:
    array = np.full((3, 3), 6, dtype=np.float32)
    stale = _assess(array, provenance_status="stale")
    unknown = _assess(array, provenance_status="unknown")
    missing = build_export_readiness(
        project_id=uuid4(),
        provenance_status="missing",
        mode="native",
        artwork_width_mm=10,
        artwork_height_mm=8,
        stock_width_mm=12,
        stock_height_mm=9,
        center_x_mm=None,
        center_y_mm=None,
        record=None,
        file_present=False,
    )
    absent = _assess(None, file_present=False, provenance_status="missing")
    assert stale.ready_for_export is False
    assert _codes(stale) == ["provenance_stale"]
    assert unknown.ready_for_export is False
    assert _codes(unknown) == ["provenance_unknown"]
    assert "backfilled" in stale.corrective_actions[0] or "not backfilled" in stale.corrective_actions[0]
    assert missing.ready_for_export is False
    assert _codes(missing) == ["physical_missing"]
    assert missing.blocking_errors[0].message == RECORD_MISSING_DETAIL
    assert missing.placement.recorded is False
    assert missing.placement.center_x_mm is None
    assert missing.estimated_triangles is None
    assert absent.ready_for_export is False
    assert _codes(absent) == ["physical_file_missing"]
    assert absent.blocking_errors[0].message == FILE_MISSING_DETAIL
    assert absent.estimated_triangles == solid_triangle_count(4, 6)
    assert absent.z_minimum_mm is None
    one_axis = _assess(np.full((1, 4), 3, dtype=np.float32), record=_record(4, 1))
    assert _codes(one_axis) == ["grid_axis"]
    assert one_axis.estimated_triangles is None


def test_angie_sized_native_plan_does_not_allocate_a_mesh() -> None:
    source = Path(__file__).resolve().parents[1] / "app" / "relief" / "export_plan.py"
    text = source.read_text(encoding="utf-8")
    assert "plan_preview_grid" not in text
    assert "DEFAULT_MAX_VERTICES" not in text
    assert "250_000" not in text
    height, width = 2940, 2140
    np.zeros(1)
    tracemalloc.start()
    report = build_export_readiness(
        project_id=uuid4(),
        provenance_status="current",
        mode="native",
        artwork_width_mm=300,
        artwork_height_mm=420,
        stock_width_mm=320,
        stock_height_mm=450,
        center_x_mm=160,
        center_y_mm=210,
        record=_record(width, height, stock=19, base=5, peak=17),
        file_present=True,
        array=None,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    triangles = solid_triangle_count(height, width)
    assert report.mode == "native"
    assert report.planned_grid is not None
    assert (report.planned_grid.height_px, report.planned_grid.width_px) == (height, width)
    assert report.estimated_vertices == solid_vertex_count(height, width)
    assert report.estimated_triangles == triangles
    assert triangles > 10_000_000
    assert report.estimated_stl_bytes == binary_stl_bytes(triangles)
    assert report.estimated_stl_bytes > MAX_EXPORT_STL_BYTES
    assert report.estimated_peak_memory_bytes is not None
    assert report.estimated_peak_memory_bytes > MAX_EXPORT_PEAK_MEMORY_BYTES
    assert report.ready_for_export is False
    assert _codes(report) == ["triangle_limit", "stl_byte_limit", "memory_limit"]
    assert peak < 5_000_000
    assert report.limits.max_triangles == MAX_EXPORT_TRIANGLES


def test_request_validation_rejects_a_silent_grid_change() -> None:
    with pytest.raises(InvalidExportRequest) as native:
        validate_export_request("native", 2, None)
    assert str(native.value) == NATIVE_REJECTS_GRID
    with pytest.raises(InvalidExportRequest) as reduced:
        validate_export_request("reduced", 4, None)
    assert str(reduced.value) == REDUCED_REQUIRES_GRID


def test_readiness_endpoint_is_read_only_and_keeps_native_resolution(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        project_id = body["id"]
        before = _files(project_root)
        native = client.get(f"/api/projects/{project_id}/relief/export/readiness")
        assert native.status_code == 200, native.text
        payload = native.json()
        assert payload["provenance_status"] == "current"
        assert payload["ready_for_export"] is True
        assert payload["export_target"] == "artwork-local"
        assert payload["export_certificate"] == "none"
        assert payload["mode"] == "native"
        assert payload["source_grid"] == payload["planned_grid"]
        assert payload["source_grid"]["height_px"] >= 2
        assert payload["artwork_width_mm"] == 90
        assert payload["artwork_height_mm"] == 70.5
        assert payload["stock_width_mm"] == 100
        assert payload["stock_height_mm"] == 80
        assert payload["stock_thickness_mm"] == 19
        assert payload["placement"]["recorded"] is True
        assert payload["placement"]["applied"] is False
        assert payload["placement"]["center_x_mm"] == 50
        assert payload["placement"]["center_y_mm"] == 40
        assert payload["topology_validated"] is False
        assert payload["self_intersection_checked"] is False
        assert payload["z_maximum_mm"] <= payload["stock_thickness_mm"]
        assert _files(project_root) == before
        assert list(_root(project_root).rglob("*.stl")) == []
        assert client.post(f"/api/projects/{project_id}/relief/export/readiness").status_code == 405

        rejected = client.get(
            f"/api/projects/{project_id}/relief/export/readiness",
            params={"height_px": 2, "width_px": 2},
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == NATIVE_REJECTS_GRID

        source_h = payload["source_grid"]["height_px"]
        source_w = payload["source_grid"]["width_px"]
        reduced = client.get(
            f"/api/projects/{project_id}/relief/export/readiness",
            params={"mode": "reduced", "height_px": 2, "width_px": 2},
        )
        assert reduced.status_code == 200, reduced.text
        reduced_body = reduced.json()
        assert reduced_body["mode"] == "reduced"
        assert reduced_body["planned_grid"] == {"height_px": 2, "width_px": 2}
        assert reduced_body["source_grid"]["height_px"] == source_h
        assert reduced_body["source_grid"]["width_px"] == source_w
        assert reduced_body["estimated_triangles"] < payload["estimated_triangles"]
        assert "reduced_omits_features" in [item["code"] for item in reduced_body["warnings"]]
        assert _files(project_root) == before

        same = client.get(
            f"/api/projects/{project_id}/relief/export/readiness",
            params={"mode": "reduced", "height_px": source_h, "width_px": source_w},
        )
        assert same.status_code == 422
        assert same.json()["detail"] == REDUCED_NOT_SMALLER

        manifest_path = _root(project_root) / "project.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"]["content_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        stale = client.get(f"/api/projects/{project_id}/relief/export/readiness")
        assert stale.status_code == 200, stale.text
        assert stale.json()["provenance_status"] == "stale"
        assert stale.json()["ready_for_export"] is False
        assert stale.json()["planned_grid"] == payload["planned_grid"]
        assert "provenance_stale" in [item["code"] for item in stale.json()["blocking_errors"]]

        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"].pop("content_sha256", None)
        stored["relief"]["physical"].pop("settings_sha256", None)
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        unknown = client.get(f"/api/projects/{project_id}/relief/export/readiness")
        assert unknown.status_code == 200, unknown.text
        assert unknown.json()["provenance_status"] == "unknown"
        assert unknown.json()["ready_for_export"] is False

        height_path = _root(project_root) / "depth/composition/physical/physical_height_mm.npy"
        height_path.unlink()
        absent = client.get(f"/api/projects/{project_id}/relief/export/readiness")
        assert absent.status_code == 200, absent.text
        assert absent.json()["ready_for_export"] is False
        assert "physical_file_missing" in [item["code"] for item in absent.json()["blocking_errors"]]

        plain = sample_body()
        plain["name"] = "Plain Export"
        created = create_project(client, plain)
        no_record = client.get(f"/api/projects/{created['id']}/relief/export/readiness")
        assert no_record.status_code == 200, no_record.text
        assert no_record.json()["provenance_status"] == "missing"
        assert no_record.json()["ready_for_export"] is False
        assert no_record.json()["blocking_errors"][0]["code"] == "physical_missing"
        assert list(project_root.rglob("*.stl")) == []
