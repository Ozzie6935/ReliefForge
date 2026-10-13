"""Read-only physical-height profiles. Temporary projects only."""

import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.projects.errors import InvalidProfileRequest
from app.relief.profile import (
    NON_FINITE_COORDINATES,
    OUT_OF_BOUNDS,
    SAMPLE_COUNT_DETAIL,
    SLOPE_NOTE,
    ZERO_LENGTH,
    build_physical_profile,
    pixel_to_artwork_mm,
)
from tests.test_projects import served
from tests.test_provenance import _pipeline, _root


def _profile(array: np.ndarray, **kwargs):
    height, width = array.shape
    return build_physical_profile(
        project_id=uuid4(),
        status=kwargs.pop("status", "current"),
        array=array,
        artwork_width_mm=kwargs.pop("artwork_width_mm", 100.0),
        artwork_height_mm=kwargs.pop("artwork_height_mm", 40.0),
        **kwargs,
    )


def _files(project_root: Path) -> dict[str, bytes]:
    root = _root(project_root)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_horizontal_and_vertical_profiles_match_stored_pixels() -> None:
    array = np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float32)
    horizontal = _profile(array, mode="horizontal", y_px=1, sample_count=3, artwork_width_mm=30, artwork_height_mm=10)
    assert [sample.x_px for sample in horizontal.samples] == [0, 1, 2]
    assert [sample.y_px for sample in horizontal.samples] == [1, 1, 1]
    assert [sample.height_mm for sample in horizontal.samples] == [10, 11, 12]
    assert horizontal.samples[0].distance_mm == 0
    assert horizontal.samples[-1].distance_mm == 30
    assert horizontal.samples[1].distance_mm == 15
    assert pixel_to_artwork_mm(2, 3, 30) == 30
    assert pixel_to_artwork_mm(0, 3, 30) == 0

    vertical = _profile(array, mode="vertical", x_px=2, sample_count=2, artwork_width_mm=30, artwork_height_mm=10)
    assert [sample.y_px for sample in vertical.samples] == [0, 1]
    assert [sample.height_mm for sample in vertical.samples] == [2, 12]
    assert vertical.samples[-1].distance_mm == 10
    assert vertical.line.start_x_px == vertical.line.end_x_px == 2


def test_diagonal_uses_artwork_scale_and_bilinear_midpoint() -> None:
    array = np.array([[0, 10], [20, 30]], dtype=np.float32)
    profile = _profile(
        array,
        mode="line",
        start_x_px=0,
        start_y_px=0,
        end_x_px=1,
        end_y_px=1,
        sample_count=3,
        artwork_width_mm=100,
        artwork_height_mm=40,
    )
    length = math.hypot(100, 40)
    assert profile.samples[0].height_mm == 0
    assert profile.samples[-1].height_mm == 30
    assert profile.samples[0].x_px == 0 and profile.samples[0].y_px == 0
    assert profile.samples[-1].x_px == 1 and profile.samples[-1].y_px == 1
    assert profile.samples[1].x_px == pytest.approx(0.5)
    assert profile.samples[1].y_px == pytest.approx(0.5)
    assert profile.samples[1].height_mm == pytest.approx(15)
    assert profile.samples[1].distance_mm == pytest.approx(length / 2)
    assert profile.summary.length_mm == pytest.approx(length)
    steps = [
        profile.samples[index + 1].distance_mm - profile.samples[index].distance_mm
        for index in range(2)
    ]
    assert steps[0] == pytest.approx(steps[1])


def test_constant_height_and_known_slope() -> None:
    constant = _profile(np.full((2, 4), 6, dtype=np.float32), mode="horizontal", y_px=0, sample_count=4)
    assert constant.summary.minimum_mm == constant.summary.maximum_mm == 6
    assert constant.summary.range_mm == 0
    assert constant.summary.maximum_adjacent_delta_mm == 0
    assert constant.summary.maximum_abs_slope_mm_per_mm == 0

    stepped = _profile(
        np.array([[0, 10, 10, 40]], dtype=np.float32),
        mode="horizontal",
        y_px=0,
        sample_count=4,
        artwork_width_mm=3,
        artwork_height_mm=8,
    )
    assert stepped.summary.maximum_adjacent_delta_mm == 30
    assert stepped.summary.maximum_abs_slope_mm_per_mm == pytest.approx(30)
    assert "machining safety" in SLOPE_NOTE
    assert stepped.slope_note == SLOPE_NOTE


def test_single_pixel_axis_and_rejected_geometry() -> None:
    column = np.array([[1], [2], [3], [4]], dtype=np.float32)
    vertical = _profile(column, mode="vertical", x_px=0, sample_count=4, artwork_width_mm=12, artwork_height_mm=30)
    assert [sample.x_px for sample in vertical.samples] == [0, 0, 0, 0]
    assert vertical.samples[-1].distance_mm == 30
    assert pixel_to_artwork_mm(0, 1, 12) == 0
    with pytest.raises(InvalidProfileRequest, match=ZERO_LENGTH):
        _profile(column, mode="horizontal", y_px=1, sample_count=2)

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    with pytest.raises(InvalidProfileRequest, match=ZERO_LENGTH):
        _profile(array, mode="line", start_x_px=1, start_y_px=1, end_x_px=1, end_y_px=1, sample_count=2)
    with pytest.raises(InvalidProfileRequest, match=OUT_OF_BOUNDS):
        _profile(array, mode="horizontal", y_px=2, sample_count=2)
    with pytest.raises(InvalidProfileRequest, match=OUT_OF_BOUNDS):
        _profile(array, mode="vertical", x_px=-0.01, sample_count=2)
    with pytest.raises(InvalidProfileRequest, match=NON_FINITE_COORDINATES):
        _profile(array, mode="horizontal", y_px=float("nan"), sample_count=2)
    with pytest.raises(InvalidProfileRequest, match=NON_FINITE_COORDINATES):
        _profile(array, mode="line", start_x_px=0, start_y_px=0, end_x_px=float("inf"), end_y_px=1, sample_count=2)
    with pytest.raises(InvalidProfileRequest, match=SAMPLE_COUNT_DETAIL):
        _profile(array, mode="horizontal", y_px=0, sample_count=1)
    with pytest.raises(InvalidProfileRequest, match=SAMPLE_COUNT_DETAIL):
        _profile(array, mode="horizontal", y_px=0, sample_count=4097)


def test_non_finite_heights_become_null_and_stay_out_of_the_slope() -> None:
    array = np.array([[5, np.nan, 9], [1, 2, 3]], dtype=np.float32)
    profile = _profile(array, mode="horizontal", y_px=0, sample_count=3, artwork_width_mm=2, artwork_height_mm=4)
    assert profile.samples[0].height_mm == 5
    assert profile.samples[1].height_mm is None
    assert profile.samples[2].height_mm == 9
    assert profile.summary.valid_count == 2
    assert profile.summary.invalid_count == 1
    assert profile.summary.maximum_adjacent_delta_mm is None
    encoded = json.dumps(profile.model_dump(mode="json"), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded

    infinite = np.array([[0, np.inf], [4, 8]], dtype=np.float32)
    across = _profile(infinite, mode="horizontal", y_px=0, sample_count=2)
    assert across.samples[0].height_mm == 0
    assert across.samples[1].height_mm is None
    assert json.dumps(across.model_dump(mode="json"), allow_nan=False)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def test_profile_endpoint_reports_provenance_without_writing(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        before = _files(project_root)
        current = client.get(
            f"/api/projects/{body['id']}/relief/physical-height/profile",
            params={"mode": "horizontal", "y_px": 0, "sample_count": 4},
        )
        assert current.status_code == 200, current.text
        payload = current.json()
        assert payload["status"] == "current"
        assert payload["mode"] == "horizontal"
        assert payload["units"] == "mm"
        assert payload["artwork_width_mm"] == body["artwork"]["width_mm"]
        assert payload["samples"][0]["distance_mm"] == 0
        assert payload["samples"][-1]["distance_mm"] == pytest.approx(body["artwork"]["width_mm"])
        assert payload["summary"]["sample_count"] == 4
        assert "center" not in payload["coordinate_convention"] or "not applied" in payload["coordinate_convention"]
        assert _files(project_root) == before
        assert list(_root(project_root).rglob("*diagnostics*")) == []
        assert list(_root(project_root).rglob("*profile*")) == []

        manifest_path = _root(project_root) / "project.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"]["content_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        stale_files = _files(project_root)
        stale = client.get(
            f"/api/projects/{body['id']}/relief/physical-height/profile",
            params={"mode": "vertical", "x_px": 0, "sample_count": 2},
        )
        assert stale.status_code == 200, stale.text
        assert stale.json()["status"] == "stale"
        assert _files(project_root) == stale_files

        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored["relief"]["physical"].pop("content_sha256", None)
        stored["relief"]["physical"].pop("settings_sha256", None)
        manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        unknown_files = _files(project_root)
        unknown = client.get(
            f"/api/projects/{body['id']}/relief/physical-height/profile",
            params={"mode": "line", "start_x_px": 0, "start_y_px": 0, "end_x_px": 1, "end_y_px": 1, "sample_count": 2},
        )
        assert unknown.status_code == 200, unknown.text
        assert unknown.json()["status"] == "unknown"
        assert _files(project_root) == unknown_files

        rejected = client.get(
            f"/api/projects/{body['id']}/relief/physical-height/profile",
            params={"mode": "horizontal", "y_px": 1000},
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == OUT_OF_BOUNDS
        assert _files(project_root) == unknown_files

        height = _root(project_root) / "depth" / "composition" / "physical" / "physical_height_mm.npy"
        height.unlink()
        missing_file = client.get(
            f"/api/projects/{body['id']}/relief/physical-height/profile",
            params={"mode": "horizontal", "y_px": 0},
        )
        assert missing_file.status_code == 404
        assert missing_file.json()["detail"] == "Physical height array was not found."
        assert not height.exists()

        created = client.post(
            "/api/projects",
            json={
                "name": "Plain",
                "stock": {"width_mm": 10, "height_mm": 10},
                "artwork": {"width_mm": 8, "height_mm": 8, "center_x_mm": 5, "center_y_mm": 5},
            },
        )
        assert created.status_code == 201, created.text
        missing_record = client.get(
            f"/api/projects/{created.json()['id']}/relief/physical-height/profile",
            params={"mode": "horizontal", "y_px": 0},
        )
        assert missing_record.status_code == 404
        assert missing_record.json()["detail"] == "Physical height has not been generated."
