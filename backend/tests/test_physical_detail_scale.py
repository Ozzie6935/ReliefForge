"""Optional physical fine-detail scale. Temporary projects and synthetic arrays only."""

import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.projects.provenance import (
    canonical_json,
    canonical_sha256,
    composition_settings_document,
    composition_source_sha256,
)
from app.relief.composition import (
    DETAIL_SCALE_MAX_MM,
    DETAIL_SCALE_MIN_MM,
    CompositionError,
    compose_relief,
    extract_physical_source_detail,
    extract_source_detail,
    prepare_detail_signals,
)
from app.relief.physical import map_physical_height
from app.schemas.project import Project
from app.schemas.relief import ReliefRegionRecord, ReliefRegionUpdate
from tests.test_provenance import _bytes, _manifest, _pipeline, _statuses, _write_manifest
from tests.test_relief import _depth, _region
from tests.test_source_detail import _pattern, _update


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


WEDDING = {"height": 1023, "width": 1013, "height_mm": 279.4, "width_mm": 279.4}
ANGIE = {"height": 2140, "width": 2940, "height_mm": 258.839, "width_mm": 355.6}
PERIODS_MM = (0.50, 1.00, 2.00, 4.00)
SCALES_MM = (0.25, 0.50, 1.00)


def _record_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "00000000-0000-4000-8000-000000000001",
        "name": "Subject",
        "role": "primary",
        "mask_filename": "00000000-0000-4000-8000-000000000001.png",
        "created_at": "2026-10-08T00:00:00Z",
        "updated_at": "2026-10-08T00:00:00Z",
    }
    body.update(overrides)
    return body


def _compose_body(region: dict, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": region["id"],
        "name": region["name"],
        "role": region["role"],
        "elevation": region["elevation"],
        "depth_range": region["depth_range"],
        "feather": region["feather"],
        "order": region["order"],
        "detail_amount": 0.0,
        "detail_scale_mm": None,
    }
    body.update(overrides)
    return {"regions": [body]}


def test_pixel_mode_matches_the_existing_extractor() -> None:
    rgb = _pattern()
    shipped = extract_source_detail(rgb)
    pixel, physical = prepare_detail_signals(
        rgb,
        [_region(np.ones(rgb.shape[:2], dtype=bool), detail_amount=0.02)],
        artwork_width_mm=90.0,
        artwork_height_mm=70.0,
    )
    assert physical is None
    assert pixel is not None
    assert np.array_equal(pixel, shipped)
    depth = np.full(rgb.shape[:2], 0.4, dtype=np.float32)
    mask = np.ones(depth.shape, dtype=bool)
    current = compose_relief(depth, [_region(mask, detail_amount=0.02, detail_scale_mm=None)], pixel)
    previous = compose_relief(depth, [_region(mask, detail_amount=0.02)], shipped)
    assert np.array_equal(current, previous)


def test_zero_amount_ignores_a_stored_scale_and_skips_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_extract(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("detail extraction ran")

    monkeypatch.setattr("app.relief.composition.extract_source_detail", fail_extract)
    monkeypatch.setattr("app.relief.composition.extract_physical_source_detail", fail_extract)
    depth = _depth()
    mask = np.ones(depth.shape, dtype=bool)
    signal = np.ones(depth.shape, dtype=np.float32)
    scaled = compose_relief(
        depth,
        [_region(mask, elevation=0.2, depth_range=1.4, feather=3, detail_scale_mm=0.5)],
        signal,
        {0.5: signal},
    )
    plain = compose_relief(
        depth,
        [_region(mask, elevation=0.2, depth_range=1.4, feather=3)],
    )
    assert np.array_equal(scaled, plain)
    identity = compose_relief(depth, [_region(mask, detail_scale_mm=1.0)], signal, {1.0: signal})
    assert np.array_equal(identity, depth)


def test_missing_scale_loads_as_null() -> None:
    record = ReliefRegionRecord.model_validate(_record_body())
    assert record.detail_scale_mm is None
    request = ReliefRegionUpdate.model_validate(_update())
    assert request.detail_scale_mm is None
    explicit = ReliefRegionUpdate.model_validate(_update(detail_scale_mm=None))
    assert explicit.detail_scale_mm is None


def test_invalid_scales_are_rejected_without_clamping() -> None:
    for amount in (float("nan"), float("inf"), float("-inf"), -0.1, 0.0, 0.249, 1.001, 2.0):
        with pytest.raises(ValidationError):
            ReliefRegionUpdate.model_validate(_update(detail_scale_mm=amount))
        with pytest.raises(ValidationError):
            ReliefRegionRecord.model_validate(_record_body(detail_scale_mm=amount))
    assert ReliefRegionUpdate.model_validate(_update(detail_scale_mm=DETAIL_SCALE_MIN_MM)).detail_scale_mm == 0.25
    assert ReliefRegionUpdate.model_validate(_update(detail_scale_mm=DETAIL_SCALE_MAX_MM)).detail_scale_mm == 1.0
    depth = _depth()
    mask = np.ones(depth.shape, dtype=bool)
    with pytest.raises(CompositionError):
        compose_relief(depth, [_region(mask, detail_amount=0.0, detail_scale_mm=0.1)])
    with pytest.raises(CompositionError):
        extract_physical_source_detail(
            _pattern(),
            artwork_width_mm=100.0,
            artwork_height_mm=80.0,
            detail_scale_mm=0.0,
        )


def test_invalid_artwork_and_tiny_grids_fail_physical_mode() -> None:
    rgb = _pattern()
    for width, height in ((0.0, 10.0), (-5.0, 10.0), (float("nan"), 10.0), (10.0, float("inf")), (10.0, 0.0)):
        with pytest.raises(CompositionError):
            extract_physical_source_detail(
                rgb,
                artwork_width_mm=width,
                artwork_height_mm=height,
                detail_scale_mm=0.5,
            )
    with pytest.raises(CompositionError):
        extract_physical_source_detail(
            np.zeros((1, 8, 3), dtype=np.uint8),
            artwork_width_mm=10.0,
            artwork_height_mm=10.0,
            detail_scale_mm=0.5,
        )
    with pytest.raises(ValueError):
        composition_settings_document(
            Path("/tmp"),
            [],
            version=3,
            artwork_width_mm=None,
            artwork_height_mm=10.0,
        )


def test_constant_physical_source_is_zero_and_output_stays_in_range() -> None:
    rgb = np.full((12, 9, 3), 80, dtype=np.uint8)
    detail = extract_physical_source_detail(
        rgb,
        artwork_width_mm=90.0,
        artwork_height_mm=60.0,
        detail_scale_mm=0.5,
    )
    assert detail.dtype == np.float32
    assert np.all(detail == 0)
    depth = np.full(rgb.shape[:2], 0.2, dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[2:8, 2:7] = True
    varied = extract_physical_source_detail(
        _pattern(12, 9),
        artwork_width_mm=90.0,
        artwork_height_mm=60.0,
        detail_scale_mm=1.0,
    )
    result = compose_relief(
        depth,
        [_region(mask, detail_amount=0.05, detail_scale_mm=1.0, elevation=0.1, depth_range=1.2)],
        physical_details={1.0: varied},
    )
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
    assert np.array_equal(result[~mask], depth[~mask])


def test_identical_scales_share_one_extract_and_different_scales_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    rgb = _pattern(48, 64)
    calls = {"pixel": 0, "physical": 0}
    real_pixel = extract_source_detail
    real_physical = extract_physical_source_detail

    def count_pixel(image: np.ndarray) -> np.ndarray:
        calls["pixel"] += 1
        return real_pixel(image)

    def count_physical(image: np.ndarray, **kwargs: object) -> np.ndarray:
        calls["physical"] += 1
        return real_physical(image, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("app.relief.composition.extract_source_detail", count_pixel)
    monkeypatch.setattr("app.relief.composition.extract_physical_source_detail", count_physical)
    left = np.zeros(rgb.shape[:2], dtype=bool)
    right = np.zeros(rgb.shape[:2], dtype=bool)
    left[:, :20] = True
    right[:, 10:] = True
    pixel, physical = prepare_detail_signals(
        rgb,
        [
            _region(left, region_id="a", detail_amount=0.02, detail_scale_mm=0.5, order=0),
            _region(right, region_id="b", detail_amount=0.04, detail_scale_mm=0.5, order=1),
        ],
        artwork_width_mm=120.0,
        artwork_height_mm=80.0,
    )
    assert calls == {"pixel": 0, "physical": 1}
    assert pixel is None
    assert physical is not None
    assert set(physical) == {0.5}
    fine, coarse = (
        extract_physical_source_detail(rgb, artwork_width_mm=120.0, artwork_height_mm=80.0, detail_scale_mm=scale)
        for scale in (0.25, 1.0)
    )
    assert not np.array_equal(fine, coarse)
    depth = np.full(rgb.shape[:2], 0.45, dtype=np.float32)
    mixed_pixel, mixed_physical = prepare_detail_signals(
        rgb,
        [
            _region(left, region_id="a", detail_amount=0.02, detail_scale_mm=None, order=0),
            _region(right, region_id="b", detail_amount=0.02, detail_scale_mm=0.25, order=1),
            _region(right, region_id="c", detail_amount=0.02, detail_scale_mm=1.0, order=2),
        ],
        artwork_width_mm=120.0,
        artwork_height_mm=80.0,
    )
    assert calls["pixel"] == 1
    assert calls["physical"] == 3
    assert mixed_pixel is not None and np.array_equal(mixed_pixel, real_pixel(rgb))
    assert mixed_physical is not None
    composed = compose_relief(
        depth,
        [
            _region(left, region_id="a", detail_amount=0.02, elevation=0.05, order=0),
            _region(right, region_id="b", detail_amount=0.02, detail_scale_mm=0.25, feather=8, order=1),
        ],
        mixed_pixel,
        mixed_physical,
    )
    assert composed.dtype == np.float32
    assert float(composed.min()) >= 0.0 and float(composed.max()) <= 1.0


def test_existing_region_math_is_unchanged_when_scale_is_null() -> None:
    depth = np.linspace(0.2, 0.8, 16, dtype=np.float32).reshape(4, 4)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[1:3, 1:3] = True
    signal = np.linspace(-0.5, 0.5, depth.size, dtype=np.float32).reshape(depth.shape)
    current = compose_relief(
        depth,
        [_region(mask, elevation=-0.1, depth_range=1.5, feather=5, detail_amount=0.03, detail_scale_mm=None)],
        signal,
    )
    previous = compose_relief(
        depth,
        [_region(mask, elevation=-0.1, depth_range=1.5, feather=5, detail_amount=0.03)],
        signal,
    )
    assert np.array_equal(current, previous)


def test_physical_height_formula_is_unchanged() -> None:
    relief = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    height = map_physical_height(relief, 12.7, 35.56)
    expected = np.float32(12.7) + relief * (np.float32(35.56) - np.float32(12.7))
    assert np.array_equal(height, expected)


def _pitch(spec: dict[str, float]) -> tuple[float, float]:
    return (
        spec["width_mm"] / (spec["width"] - 1),
        spec["height_mm"] / (spec["height"] - 1),
    )


def _grating(spec: dict[str, float], period_mm: float, axis: str) -> np.ndarray:
    pitch_x, pitch_y = _pitch(spec)
    if axis == "x":
        position = (np.arange(spec["width"], dtype=np.float64) + 0.25) * pitch_x
        wave = np.sin(2 * np.pi * position / period_mm)
        gray = np.broadcast_to(128 + 40 * wave, (96, int(spec["width"]))).copy()
    else:
        position = (np.arange(spec["height"], dtype=np.float64) + 0.25) * pitch_y
        wave = np.sin(2 * np.pi * position / period_mm)
        gray = np.broadcast_to((128 + 40 * wave)[:, None], (int(spec["height"]), 96)).copy()
    return np.repeat(np.clip(np.rint(gray), 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def _gain(detail: np.ndarray, rgb: np.ndarray, axis: str) -> float:
    if axis == "x":
        row = detail.shape[0] // 2
        got = detail[row, 32:-32].astype(np.float64)
        luma = rgb[row, 32:-32, 0].astype(np.float64) / 255.0
    else:
        column = detail.shape[1] // 2
        got = detail[32:-32, column].astype(np.float64)
        luma = rgb[32:-32, column, 0].astype(np.float64) / 255.0
    luma = luma - luma.mean()
    luma_rms = float(np.sqrt(np.mean(luma ** 2)))
    detail_rms = float(np.sqrt(np.mean(got ** 2)))
    return detail_rms / luma_rms


def test_half_millimeter_gain_agrees_across_the_two_grid_sizes() -> None:
    measured: list[dict[str, object]] = []
    for scale in SCALES_MM:
        for axis in ("x", "y"):
            for period in PERIODS_MM:
                gains = {}
                for name, spec in (("wedding", WEDDING), ("angie", ANGIE)):
                    rgb = _grating(spec, period, axis)
                    detail = extract_physical_source_detail(
                        rgb,
                        artwork_width_mm=spec["width_mm"],
                        artwork_height_mm=spec["height_mm"],
                        detail_scale_mm=scale,
                    )
                    gains[name] = _gain(detail, rgb, axis)
                measured.append(
                    {
                        "scale_mm": scale,
                        "axis": axis,
                        "period_mm": period,
                        "wedding_gain": gains["wedding"],
                        "angie_gain": gains["angie"],
                        "difference": abs(gains["wedding"] - gains["angie"]),
                    }
                )
    half = [row for row in measured if row["scale_mm"] == 0.50]
    destination = Path("/tmp/rf115")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "gain.json").write_text(json.dumps(measured), encoding="utf-8")
    offenders = [row for row in half if float(row["difference"]) > 0.001]
    assert offenders == []


def test_full_frame_extract_is_finite() -> None:
    report: dict[str, object] = {}
    for name, spec in (("wedding", WEDDING), ("angie", ANGIE)):
        rgb = np.zeros((int(spec["height"]), int(spec["width"]), 3), dtype=np.uint8)
        rgb[::2, ::2] = 220
        rgb[1::2, 1::2] = 40
        modes: dict[str, object] = {}
        tracemalloc.start()
        started = time.perf_counter()
        pixel = extract_source_detail(rgb)
        modes["pixel_2"] = {
            "seconds": time.perf_counter() - started,
            "peak_bytes": tracemalloc.get_traced_memory()[1],
        }
        tracemalloc.stop()
        assert pixel.dtype == np.float32 and np.isfinite(pixel).all()
        for scale in SCALES_MM:
            tracemalloc.start()
            started = time.perf_counter()
            detail = extract_physical_source_detail(
                rgb,
                artwork_width_mm=spec["width_mm"],
                artwork_height_mm=spec["height_mm"],
                detail_scale_mm=scale,
            )
            modes[f"mm_{scale:.2f}"] = {
                "seconds": time.perf_counter() - started,
                "peak_bytes": tracemalloc.get_traced_memory()[1],
                "dtype": str(detail.dtype),
                "min": float(detail.min()),
                "max": float(detail.max()),
            }
            tracemalloc.stop()
            assert detail.dtype == np.float32
            assert np.isfinite(detail).all()
            assert float(detail.min()) >= -1.0
            assert float(detail.max()) <= 1.0
        calls = {"n": 0}
        real = extract_physical_source_detail

        def counted(image: np.ndarray, **kwargs: object) -> np.ndarray:
            calls["n"] += 1
            return real(image, **kwargs)  # type: ignore[arg-type]

        # Local patch avoids leaking into other tests. The prepare function
        # looks the extractor up on the composition module.
        import app.relief.composition as composition

        previous = composition.extract_physical_source_detail
        composition.extract_physical_source_detail = counted
        try:
            mask = np.ones(rgb.shape[:2], dtype=bool)
            prepare_detail_signals(
                rgb,
                [
                    _region(mask, region_id="a", detail_amount=0.01, detail_scale_mm=0.5),
                    _region(mask, region_id="b", detail_amount=0.02, detail_scale_mm=0.5),
                ],
                artwork_width_mm=spec["width_mm"],
                artwork_height_mm=spec["height_mm"],
            )
        finally:
            composition.extract_physical_source_detail = previous
        assert calls["n"] == 1
        modes["reused_0.50_calls"] = calls["n"]
        report[name] = modes
    destination = Path("/tmp/rf115")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "perf.json").write_text(json.dumps(report), encoding="utf-8")


def test_version_two_hash_has_no_scale_and_version_three_records_artwork(
    project_root: Path,
) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        relief = body["relief"]
        assert relief["settings_version"] == 2
        assert relief["regions"][0]["detail_scale_mm"] is None
        directory = project_root / body["slug"]
        project = Project.model_validate({key: value for key, value in body.items() if key != "_region"})
        assert project.relief is not None
        document = composition_settings_document(directory, project.relief.regions, version=2)
        encoded = canonical_json(document)
        assert "detail_scale_mm" not in encoded
        assert "artwork_width_mm" not in encoded
        assert relief["settings_sha256"] == canonical_sha256(document)
        region = body["_region"]
        composed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json=_compose_body(region, detail_amount=0.02, detail_scale_mm=0.5),
        )
        assert composed.status_code == 200, composed.text
        payload = composed.json()["relief"]
        assert payload["settings_version"] == 3
        assert payload["status"] == "current"
        assert payload["regions"][0]["detail_scale_mm"] == 0.5
        updated = Project.model_validate(composed.json())
        assert updated.relief is not None
        source_hash = composition_source_sha256(directory)
        version3 = composition_settings_document(
            directory,
            updated.relief.regions,
            version=3,
            source_sha256=source_hash,
            artwork_width_mm=updated.artwork.width_mm,
            artwork_height_mm=updated.artwork.height_mm,
        )
        assert version3["version"] == 3
        assert version3["artwork_width_mm"] == updated.artwork.width_mm
        assert version3["artwork_height_mm"] == updated.artwork.height_mm
        assert version3["source_sha256"] == source_hash
        assert version3["regions"][0]["detail_scale_mm"] == 0.5
        assert payload["settings_sha256"] == canonical_sha256(version3)
        zero = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json=_compose_body(region, detail_amount=0.0, detail_scale_mm=0.5),
        )
        assert zero.status_code == 200, zero.text
        zero_relief = zero.json()["relief"]
        assert zero_relief["settings_version"] == 3
        zero_project = Project.model_validate(zero.json())
        assert zero_project.relief is not None
        zero_document = composition_settings_document(
            directory,
            zero_project.relief.regions,
            version=3,
            artwork_width_mm=zero_project.artwork.width_mm,
            artwork_height_mm=zero_project.artwork.height_mm,
        )
        assert "source_sha256" not in zero_document
        assert zero_relief["settings_sha256"] == canonical_sha256(zero_document)


def test_scale_and_artwork_edits_stale_relief_and_physical_only(project_root: Path) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        before = _bytes(project_root)
        composed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json=_compose_body(region, detail_amount=0.02, detail_scale_mm=0.5),
        )
        assert composed.status_code == 200, composed.text
        composed_body = composed.json()
        assert _statuses(composed_body)["relief"] == "current"
        assert composed_body["relief"]["settings_version"] == 3
        previous_relief_hash = body["relief"]["physical"]["source_relief_sha256"]
        if composed_body["relief"]["content_sha256"] == previous_relief_hash:
            assert _statuses(composed_body)["physical"] == "current"
        else:
            assert _statuses(composed_body)["physical"] == "stale"
        assert _bytes(project_root)["depth/raw_depth.npy"] == before["depth/raw_depth.npy"]
        assert _bytes(project_root)["depth/processing/conditioned_depth.npy"] == before[
            "depth/processing/conditioned_depth.npy"
        ]

        manifest = _manifest(project_root)
        relief_bytes = (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes()
        manifest["relief"]["regions"][0]["detail_scale_mm"] = 1.0
        _write_manifest(project_root, manifest)
        changed = client.get(f"/api/projects/{body['id']}")
        statuses = _statuses(changed.json())
        assert statuses["raw"] == "current"
        assert statuses["conditioned"] == "current"
        assert statuses["relief"] == "stale"
        assert statuses["physical"] == "stale"
        assert (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes() == relief_bytes
        assert _bytes(project_root)["depth/raw_depth.npy"] == before["depth/raw_depth.npy"]
        assert _bytes(project_root)["depth/processing/conditioned_depth.npy"] == before[
            "depth/processing/conditioned_depth.npy"
        ]

        manifest = _manifest(project_root)
        manifest["relief"]["regions"][0]["detail_scale_mm"] = 0.5
        manifest["artwork"]["width_mm"] = manifest["artwork"]["width_mm"] + 1.0
        _write_manifest(project_root, manifest)
        resized = client.get(f"/api/projects/{body['id']}")
        resized_status = _statuses(resized.json())
        assert resized_status["raw"] == "current"
        assert resized_status["conditioned"] == "current"
        assert resized_status["relief"] == "stale"
        assert resized_status["physical"] == "stale"

        manifest = _manifest(project_root)
        manifest["relief"].pop("settings_sha256")
        _write_manifest(project_root, manifest)
        unknown = client.get(f"/api/projects/{body['id']}")
        assert _statuses(unknown.json())["relief"] == "unknown"
        assert _statuses(unknown.json())["relief"] != "current"


def test_version_two_scale_and_source_drift_stay_exact(project_root: Path) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        manifest = _manifest(project_root)
        manifest["relief"]["regions"][0]["detail_scale_mm"] = 0.5
        _write_manifest(project_root, manifest)
        stale = client.get(f"/api/projects/{body['id']}")
        assert _statuses(stale.json())["relief"] == "stale"
        assert _statuses(stale.json())["physical"] == "stale"
        assert _statuses(stale.json())["raw"] == "current"
        assert _statuses(stale.json())["conditioned"] == "current"

        manifest = _manifest(project_root)
        manifest["relief"]["regions"][0]["detail_scale_mm"] = None
        original_width = manifest["artwork"]["width_mm"]
        manifest["artwork"]["width_mm"] = original_width + 2.0
        _write_manifest(project_root, manifest)
        pixel_mode = client.get(f"/api/projects/{body['id']}")
        assert _statuses(pixel_mode.json())["relief"] == "current"

        region = body["_region"]
        composed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json=_compose_body(region, detail_amount=0.02, detail_scale_mm=0.5),
        )
        assert composed.status_code == 200, composed.text
        directory = project_root / body["slug"]
        source = directory / "source" / "original.png"
        mutated = bytearray(source.read_bytes())
        mutated[-1] ^= 0x5A
        source.write_bytes(mutated)
        new_hash = hashlib.sha256(mutated).hexdigest()
        manifest = _manifest(project_root)
        manifest["source"]["content_sha256"] = new_hash
        manifest["depth"]["source_sha256"] = new_hash
        _write_manifest(project_root, manifest)
        drifted = client.get(f"/api/projects/{body['id']}")
        drifted_status = _statuses(drifted.json())
        assert drifted_status["source"] == "current"
        assert drifted_status["raw"] == "current"
        assert drifted_status["conditioned"] == "current"
        assert drifted_status["relief"] == "stale"
        assert drifted_status["physical"] == "stale"


def test_zero_detail_physical_scale_does_not_read_source_or_hash_it(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("source was read")

    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        monkeypatch.setattr("app.relief.service.ReliefService._read_source_rgb", fail_read)
        composed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json=_compose_body(region, detail_amount=0.0, detail_scale_mm=0.75),
        )
        assert composed.status_code == 200, composed.text
        relief = composed.json()["relief"]
        assert relief["settings_version"] == 3
        assert relief["status"] == "current"
        directory = project_root / body["slug"]
        project = Project.model_validate(composed.json())
        assert project.relief is not None
        document = composition_settings_document(
            directory,
            project.relief.regions,
            version=3,
            artwork_width_mm=project.artwork.width_mm,
            artwork_height_mm=project.artwork.height_mm,
        )
        assert "source_sha256" not in document


def test_api_rejects_an_out_of_range_scale(project_root: Path) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        before = _bytes(project_root)
        rejected = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json=_compose_body(region, detail_amount=0.02, detail_scale_mm=1.01),
        )
        assert rejected.status_code == 422
        assert _bytes(project_root)["depth/composition/relief_depth.npy"] == before[
            "depth/composition/relief_depth.npy"
        ]


def served_client(project_root: Path):
    from tests.test_projects import served
    from app.core.config import MAX_SOURCE_IMAGE_BYTES

    return served(project_root, MAX_SOURCE_IMAGE_BYTES)
