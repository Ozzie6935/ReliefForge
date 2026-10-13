"""Depth conditioning tests.

Numerical tests call the pure function. Integration tests use a temporary
project directory and a fake depth model. They do not load Depth Anything
weights or write into the repository projects/ folder.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.config import MAX_SOURCE_IMAGE_BYTES
from app.depth.conditioning import bilateral_settings, condition_depth
from app.depth.histogram import HISTOGRAM_BINS, depth_histogram
from app.depth.preview import unit_interval_preview
from app.schemas.conditioning import DepthConditioningParameters
from tests.test_depth import FakeDepthModel, _use_model
from tests.test_projects import create_project, image_bytes, served


def _raw() -> np.ndarray:
    return np.linspace(0.2, 4.0, 12, dtype=np.float32).reshape(3, 4)


def _defaults(**overrides: object) -> DepthConditioningParameters:
    return DepthConditioningParameters(**overrides)


def test_orientation_keeps_larger_raw_values_nearer() -> None:
    raw = np.array([[0.2, 1.0], [2.0, 4.0]], dtype=np.float32)
    result = condition_depth(raw, _defaults())
    assert result[1, 1] > result[0, 0]
    inverted = condition_depth(raw, _defaults(invert=True))
    assert inverted[1, 1] < inverted[0, 0]


def test_percentile_clipping_limits_outliers() -> None:
    values = np.linspace(0.0, 100.0, 1000, dtype=np.float32).reshape(25, 40)
    result = condition_depth(values, _defaults(far_percentile=1.0, near_percentile=99.0))
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    assert np.all(result[values <= low] <= 1e-5)
    assert np.all(result[values >= high] >= 1.0 - 1e-5)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_normalization_gamma_and_contrast_identities() -> None:
    raw = _raw()
    result = condition_depth(raw, _defaults(gamma=1.0, contrast=1.0, smoothing=0.0))
    low = np.float32(np.percentile(raw, 1.0))
    high = np.float32(np.percentile(raw, 99.0))
    expected = np.clip((np.clip(raw, low, high) - low) / (high - low), 0.0, 1.0)
    assert result.dtype == np.float32
    assert np.allclose(result, expected.astype(np.float32))
    assert np.array_equal(result, condition_depth(raw, _defaults()))
    contrasted = condition_depth(raw, _defaults(contrast=2.0))
    assert not np.array_equal(result, contrasted)
    darkened = condition_depth(raw, _defaults(gamma=2.0))
    assert not np.array_equal(result, darkened)


def test_smoothing_zero_bypasses_the_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    def rejected(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("smoothing was not bypassed")

    monkeypatch.setitem(sys.modules, "cv2", type("Cv2", (), {"bilateralFilter": staticmethod(rejected)}))
    raw = _raw()
    result = condition_depth(raw, _defaults(smoothing=0.0, edge_preservation=0.0))
    assert result.shape == raw.shape
    assert bilateral_settings(0.0, 75.0) is None
    diameter, tight, _space = bilateral_settings(100.0, 100.0)
    _diameter, open_color, _space = bilateral_settings(100.0, 0.0)
    assert diameter <= 7
    assert tight < open_color


def test_smoothing_stays_finite_and_in_range() -> None:
    raw = np.linspace(0.2, 5.0, 64, dtype=np.float32).reshape(8, 8)
    result = condition_depth(raw, _defaults(smoothing=100.0, edge_preservation=75.0))
    assert result.dtype == np.float32
    assert result.shape == raw.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_output_is_finite_float32_in_unit_interval_and_input_is_unchanged() -> None:
    raw = _raw()
    original = raw.copy()
    first = condition_depth(raw, _defaults())
    second = condition_depth(raw, _defaults())
    assert first.dtype == np.float32
    assert first.shape == raw.shape
    assert np.isfinite(first).all()
    assert first.min() >= 0.0
    assert first.max() <= 1.0
    assert np.array_equal(first, second)
    assert np.array_equal(raw, original)


def test_constant_input_stays_finite() -> None:
    raw = np.full((5, 5), 2.5, dtype=np.float32)
    result = condition_depth(raw, _defaults())
    assert np.isfinite(result).all()
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_invalid_percentile_ordering_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DepthConditioningParameters(far_percentile=99.0, near_percentile=1.0)


def test_preview_does_not_mutate_conditioned_array() -> None:
    conditioned = condition_depth(_raw(), _defaults())
    original = conditioned.copy()
    preview = unit_interval_preview(conditioned)
    assert preview.startswith(b"\x89PNG")
    assert np.array_equal(conditioned, original)


def test_histogram_bins_account_for_every_value() -> None:
    raw = _raw()
    conditioned = condition_depth(raw, _defaults())
    raw_series = depth_histogram(raw)
    conditioned_series = depth_histogram(conditioned, value_range=(0.0, 1.0))
    assert raw_series.bins == HISTOGRAM_BINS == 256
    assert conditioned_series.bins == 256
    assert sum(raw_series.counts) == raw.size
    assert sum(conditioned_series.counts) == conditioned.size


def test_second_result_comes_from_raw_not_from_the_first_result() -> None:
    raw = _raw()
    original = raw.copy()
    first = condition_depth(raw, _defaults(gamma=2.0))
    second = condition_depth(raw, _defaults(gamma=1.0))
    reapplied = condition_depth(first, _defaults(gamma=1.0))
    assert not np.array_equal(second, reapplied)
    assert np.array_equal(raw, original)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _project_with_raw(project_root: Path, client) -> dict:
    model = FakeDepthModel()
    _use_model(project_root, model)
    created = create_project(client)
    source = image_bytes("PNG", (6, 4))
    upload = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("scan.png", source, "image/png")},
    )
    assert upload.status_code == 200
    generated = client.post(f"/api/projects/{created['id']}/depth")
    assert generated.status_code == 200, generated.text
    return generated.json()


def test_conditioning_without_raw_depth_is_rejected(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _use_model(project_root, FakeDepthModel())
        created = create_project(client)
        response = client.post(
            f"/api/projects/{created['id']}/depth/conditioning",
            json={},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "Raw depth has not been generated."


def test_conditioning_persists_artifacts_metadata_and_leaves_raw_unchanged(
    project_root: Path,
) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project = _project_with_raw(project_root, client)
        raw_path = project_root / "sample-relief" / "depth" / "raw_depth.npy"
        raw_preview = project_root / "sample-relief" / "depth" / "preview.png"
        before_raw = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        before_preview = raw_preview.read_bytes()
        before_updated = project["updated_at"]
        response = client.post(
            f"/api/projects/{project['id']}/depth/conditioning",
            json={
                "far_percentile": 1.0,
                "near_percentile": 99.0,
                "gamma": 1.0,
                "contrast": 1.0,
                "smoothing": 0.0,
                "edge_preservation": 75.0,
                "invert": False,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["updated_at"] != before_updated
        conditioning = body["depth"]["conditioning"]
        assert conditioning["parameters"]["far_percentile"] == 1.0
        assert conditioning["parameters"]["near_percentile"] == 99.0
        assert conditioning["parameters"]["gamma"] == 1.0
        assert conditioning["dtype"] == "float32"
        assert conditioning["width_px"] == 6
        assert conditioning["height_px"] == 4
        assert conditioning["conditioned_filename"] == "conditioned_depth.npy"
        assert conditioning["preview_filename"] == "preview.png"
        assert 0.0 <= conditioning["range_min"] <= conditioning["range_max"] <= 1.0
        conditioned_path = (
            project_root / "sample-relief" / "depth" / "processing" / "conditioned_depth.npy"
        )
        loaded = np.load(conditioned_path)
        assert loaded.dtype == np.float32
        assert loaded.shape == (4, 6)
        assert loaded.min() >= 0.0
        assert loaded.max() <= 1.0
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before_raw
        assert raw_preview.read_bytes() == before_preview
        preview = client.get(f"/api/projects/{project['id']}/depth/conditioning/preview")
        assert preview.status_code == 200
        assert preview.content.startswith(b"\x89PNG")
        refreshed = client.get(f"/api/projects/{project['id']}")
        assert refreshed.status_code == 200
        assert refreshed.json()["depth"]["conditioning"]["parameters"] == conditioning["parameters"]
        assert hashlib.sha256(conditioned_path.read_bytes()).hexdigest() == hashlib.sha256(
            conditioned_path.read_bytes()
        ).hexdigest()
        histogram = client.get(f"/api/projects/{project['id']}/depth/histogram")
        assert histogram.status_code == 200
        payload = histogram.json()
        assert payload["raw"]["bins"] == 256
        assert sum(payload["raw"]["counts"]) == 24
        assert payload["conditioned"]["bins"] == 256
        assert sum(payload["conditioned"]["counts"]) == 24
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before_raw


def test_second_conditioning_starts_from_raw(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project = _project_with_raw(project_root, client)
        raw_path = project_root / "sample-relief" / "depth" / "raw_depth.npy"
        before = raw_path.read_bytes()
        first = client.post(
            f"/api/projects/{project['id']}/depth/conditioning",
            json={"gamma": 2.0},
        )
        assert first.status_code == 200, first.text
        first_array = np.load(
            project_root / "sample-relief" / "depth" / "processing" / "conditioned_depth.npy"
        )
        second = client.post(
            f"/api/projects/{project['id']}/depth/conditioning",
            json={"gamma": 1.0},
        )
        assert second.status_code == 200, second.text
        second_array = np.load(
            project_root / "sample-relief" / "depth" / "processing" / "conditioned_depth.npy"
        )
        raw = np.load(raw_path)
        expected = condition_depth(raw, DepthConditioningParameters(gamma=1.0))
        reapplied = condition_depth(first_array, DepthConditioningParameters(gamma=1.0))
        assert np.array_equal(second_array, expected)
        assert not np.array_equal(second_array, reapplied)
        assert raw_path.read_bytes() == before


def test_invalid_conditioning_parameters_are_rejected(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project = _project_with_raw(project_root, client)
        response = client.post(
            f"/api/projects/{project['id']}/depth/conditioning",
            json={"gamma": 9.0},
        )
        assert response.status_code == 422
