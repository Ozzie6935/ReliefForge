"""Read-only physical-height diagnostics. Temporary projects only."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.relief.diagnostics import (
    FILE_MISSING_DETAIL,
    HISTOGRAM_BINS,
    NO_FINITE_SAMPLES,
    RECORD_MISSING_DETAIL,
    SHAPE_DETAIL,
    STRUCTURE_DETAIL,
    diagnose_physical_height,
)
from app.relief.physical import HEIGHT_PATH, PREVIEW_PATH, STATS_PATH, npy_bytes
from app.schemas.relief import PhysicalHeightRecord
from tests.test_physical_height import _generate
from tests.test_projects import served
from tests.test_provenance import _pipeline, _root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _record(width: int, height: int, stock: float = 19, base: float = 5, peak: float = 17) -> PhysicalHeightRecord:
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
        height_min_mm=0,
        height_max_mm=1,
        maximum_cut_depth_mm=stock - base,
        relief_amplitude_mm=peak - base,
    )


def _diagnose(array: np.ndarray, **kwargs):
    record = _record(int(array.shape[1]), int(array.shape[0]), **kwargs)
    return diagnose_physical_height(uuid4(), "current", record, npy_bytes(array))


def _files(project_root: Path) -> dict[str, bytes]:
    root = _root(project_root)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_gradient_percentiles_and_histogram_are_exact() -> None:
    array = np.arange(10, dtype=np.float32).reshape(2, 5)
    report = _diagnose(array)
    finite = array.astype(np.float64).reshape(-1)
    expected = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99], method="linear")
    assert report.width_px == 5
    assert report.height_px == 2
    assert report.metrics is not None
    assert report.metrics.sample_count == 10
    assert report.metrics.minimum_mm == 0
    assert report.metrics.maximum_mm == 9
    assert report.metrics.mean_mm == float(finite.mean())
    assert report.metrics.median_mm == float(np.median(finite))
    assert report.metrics.standard_deviation_mm == float(np.std(finite, ddof=0))
    assert report.metrics.range_mm == 9
    assert report.metrics.percentiles_mm.p1 == float(expected[0])
    assert report.metrics.percentiles_mm.p5 == float(expected[1])
    assert report.metrics.percentiles_mm.p25 == float(expected[2])
    assert report.metrics.percentiles_mm.p50 == float(expected[3])
    assert report.metrics.percentiles_mm.p75 == float(expected[4])
    assert report.metrics.percentiles_mm.p95 == float(expected[5])
    assert report.metrics.percentiles_mm.p99 == float(expected[6])
    assert report.histogram is not None
    assert report.histogram.bins == HISTOGRAM_BINS
    assert sum(report.histogram.counts) == report.histogram.sample_count == 10
    assert len(report.histogram.edges_mm) == HISTOGRAM_BINS + 1
    assert report.mapping.stock_thickness_mm == 19
    assert report.mapping.maximum_theoretical_cut_depth_mm == 14
    assert report.mapping.maximum_surface_height_mm == 17
    assert report.metrics.maximum_mm != report.mapping.maximum_surface_height_mm
    assert report.validation.ok is True


def test_constant_and_zero_arrays_have_a_defined_histogram() -> None:
    constant = _diagnose(np.full((3, 3), 5, dtype=np.float32))
    assert constant.metrics is not None and constant.histogram is not None
    assert constant.metrics.minimum_mm == constant.metrics.maximum_mm == 5
    assert constant.metrics.standard_deviation_mm == 0
    assert constant.metrics.range_mm == 0
    assert constant.histogram.counts[0] == 9
    assert sum(constant.histogram.counts) == 9
    assert constant.histogram.edges_mm == [5.0] * (HISTOGRAM_BINS + 1)

    zeros = _diagnose(np.zeros((2, 4), dtype=np.float32))
    assert zeros.metrics is not None
    assert zeros.width_px == 4
    assert zeros.height_px == 2
    assert zeros.metrics.mean_mm == 0
    assert zeros.metrics.below_zero_count == 0
    assert zeros.metrics.range_mm == 0
    assert zeros.histogram is not None
    assert sum(zeros.histogram.counts) == zeros.histogram.sample_count == 8


def test_non_finite_values_are_counted_and_not_replaced() -> None:
    array = np.array([[1.0, np.nan], [np.inf, 3.0]], dtype=np.float32)
    report = _diagnose(array, stock=10)
    assert report.validation.ok is True
    assert report.metrics is not None
    assert report.metrics.non_finite_count == 2
    assert report.metrics.sample_count == 4
    assert report.metrics.minimum_mm == 1
    assert report.metrics.maximum_mm == 3
    assert report.metrics.mean_mm == 2
    assert report.histogram is not None
    assert sum(report.histogram.counts) == report.histogram.sample_count == 2
    assert NO_FINITE_SAMPLES not in report.validation.errors

    empty = _diagnose(np.array([[np.nan, np.inf]], dtype=np.float32))
    assert empty.metrics is None
    assert empty.histogram is None
    assert empty.validation.ok is False
    assert empty.validation.errors == [NO_FINITE_SAMPLES]


def test_below_zero_and_above_stock_are_separate_from_recorded_settings() -> None:
    array = np.array([[-2.0, 0.0], [8.0, 25.0]], dtype=np.float32)
    report = _diagnose(array, stock=19, base=5, peak=17)
    assert report.metrics is not None
    assert report.metrics.below_zero_count == 1
    assert report.metrics.above_stock_count == 1
    assert report.metrics.minimum_mm == -2
    assert report.metrics.maximum_mm == 25
    assert report.mapping.stock_thickness_mm == 19
    assert report.mapping.base_thickness_mm == 5
    assert report.mapping.maximum_surface_height_mm == 17
    assert report.validation.ok is True


def test_invalid_shape_and_empty_arrays_are_rejected() -> None:
    record = _record(4, 2)
    with pytest.raises(Exception) as shape:
        diagnose_physical_height(uuid4(), "current", record, npy_bytes(np.zeros((2, 3), dtype=np.float32)))
    assert SHAPE_DETAIL in str(shape.value)
    with pytest.raises(Exception) as empty:
        diagnose_physical_height(uuid4(), "current", record, npy_bytes(np.zeros((0, 4), dtype=np.float32)))
    assert STRUCTURE_DETAIL in str(empty.value)
    with pytest.raises(Exception) as flat:
        diagnose_physical_height(uuid4(), "current", record, npy_bytes(np.zeros((4,), dtype=np.float32)))
    assert STRUCTURE_DETAIL in str(flat.value)


def test_diagnostics_endpoint_reports_provenance_without_writing(
    project_root: Path,
) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        missing = client.get(f"/api/projects/{uuid4()}/relief/physical-height/diagnostics")
        assert missing.status_code == 404

        before = _files(project_root)
        current = client.get(f"/api/projects/{body['id']}/relief/physical-height/diagnostics")
        assert current.status_code == 200, current.text
        payload = current.json()
        assert payload["status"] == "current"
        assert payload["project_id"] == body["id"]
        assert payload["mapping"]["stock_thickness_mm"] == 19
        assert payload["mapping"]["maximum_theoretical_cut_depth_mm"] == 14
        assert sum(payload["histogram"]["counts"]) == payload["histogram"]["sample_count"]
        assert payload["histogram"]["sample_count"] == payload["metrics"]["sample_count"]
        assert _files(project_root) == before
        assert not (_root(project_root) / "diagnostics").exists()
        assert list(_root(project_root).rglob("*diagnostics*")) == []

        manifest_path = _root(project_root) / "project.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"]["content_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        stale_bytes = _files(project_root)
        stale = client.get(f"/api/projects/{body['id']}/relief/physical-height/diagnostics")
        assert stale.status_code == 200, stale.text
        assert stale.json()["status"] == "stale"
        assert _files(project_root) == stale_bytes

        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"].pop("content_sha256", None)
        stored["relief"]["physical"].pop("settings_sha256", None)
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        unknown_bytes = _files(project_root)
        unknown = client.get(f"/api/projects/{body['id']}/relief/physical-height/diagnostics")
        assert unknown.status_code == 200, unknown.text
        assert unknown.json()["status"] == "unknown"
        assert _files(project_root) == unknown_bytes

        height = _root(project_root) / "depth" / "composition" / "physical" / "physical_height_mm.npy"
        original = height.read_bytes()
        np.save(height, np.zeros((1, 1), dtype=np.float32))
        invalid_bytes = _files(project_root)
        invalid = client.get(f"/api/projects/{body['id']}/relief/physical-height/diagnostics")
        assert invalid.status_code == 422
        assert invalid.json()["detail"] == SHAPE_DETAIL
        assert _files(project_root) == invalid_bytes
        height.write_bytes(original)

        manifest_text = manifest_path.read_text(encoding="utf-8")
        height.unlink()
        absent = client.get(f"/api/projects/{body['id']}/relief/physical-height/diagnostics")
        assert absent.status_code == 404
        assert absent.json()["detail"] == FILE_MISSING_DETAIL
        assert manifest_path.read_text(encoding="utf-8") == manifest_text
        assert not height.exists()

        ungenerated = client.post(f"/api/projects", json={
            "name": "Plain",
            "stock": {"width_mm": 10, "height_mm": 10},
            "artwork": {"width_mm": 8, "height_mm": 8, "center_x_mm": 5, "center_y_mm": 5},
        })
        assert ungenerated.status_code == 201, ungenerated.text
        no_record = client.get(
            f"/api/projects/{ungenerated.json()['id']}/relief/physical-height/diagnostics"
        )
        assert no_record.status_code == 404
        assert no_record.json()["detail"] == RECORD_MISSING_DETAIL


def test_generate_route_remains_separate_from_diagnostics(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        before = _files(project_root)
        rejected = _generate(client, body["id"], {"stock_thickness_mm": 1, "base_thickness_mm": 5, "maximum_surface_height_mm": 4})
        assert rejected.status_code == 422
        assert _files(project_root) == before
