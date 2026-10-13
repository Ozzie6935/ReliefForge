"""Source-guided fine detail. Temporary projects only."""

import hashlib
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.config import MAX_SOURCE_IMAGE_BYTES
from app.projects.provenance import (
    canonical_sha256,
    composition_settings_document,
    composition_source_sha256,
)
from app.relief.composition import (
    DETAIL_AMOUNT_MAX,
    CompositionError,
    compose_relief,
    extract_source_detail,
)
from app.relief.physical import map_physical_height
from app.schemas.project import Project
from app.schemas.relief import ReliefRegionRecord, ReliefRegionUpdate
from tests.test_projects import image_bytes, served
from tests.test_provenance import _bytes, _manifest, _pipeline, _statuses, _write_manifest
from tests.test_relief import _depth, _region


def _pattern(height: int = 32, width: int = 32) -> np.ndarray:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[::2, ::2] = 255
    rgb[1::2, 1::2] = 20
    return rgb


def _mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[8:24, 8:24] = True
    return mask


def _update(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": str(uuid4()),
        "name": "Subject",
        "role": "primary",
        "elevation": 0.0,
        "depth_range": 1.0,
        "feather": 0.0,
        "order": 0,
    }
    body.update(overrides)
    return body


def test_zero_detail_matches_the_previous_composition() -> None:
    depth = _depth()
    mask = np.zeros(depth.shape, dtype=bool)
    mask[1:3, 1:3] = True
    signal = np.linspace(-1, 1, depth.size, dtype=np.float32).reshape(depth.shape)
    previous = compose_relief(depth, [_region(mask, elevation=0.2, depth_range=1.5, feather=4)])
    current = compose_relief(
        depth,
        [_region(mask, elevation=0.2, depth_range=1.5, feather=4, detail_amount=0.0)],
        signal,
    )
    identity = compose_relief(depth, [_region(mask, detail_amount=0.0)], signal)
    assert np.array_equal(current, previous)
    assert np.array_equal(identity, depth)


def test_missing_detail_amount_defaults_to_zero() -> None:
    record = ReliefRegionRecord.model_validate(
        {
            "id": str(uuid4()),
            "name": "Bride",
            "role": "primary",
            "mask_filename": "mask.png",
            "created_at": "2026-10-08T00:00:00Z",
            "updated_at": "2026-10-08T00:00:00Z",
        }
    )
    assert record.detail_amount == 0.0
    request = ReliefRegionUpdate.model_validate(_update())
    assert request.detail_amount == 0.0


def test_constant_source_produces_zero_detail() -> None:
    rgb = np.full((12, 9, 3), 80, dtype=np.uint8)
    detail = extract_source_detail(rgb)
    assert detail.dtype == np.float32
    assert detail.shape == (12, 9)
    assert np.all(detail == 0)


def test_fine_pattern_changes_only_the_mask() -> None:
    rgb = _pattern()
    detail = extract_source_detail(rgb)
    assert detail.dtype == np.float32
    assert np.isfinite(detail).all()
    assert float(detail.min()) < 0.0
    assert float(detail.max()) > 0.0
    assert float(np.max(np.abs(detail))) <= 1.0
    depth = np.full(rgb.shape[:2], 0.45, dtype=np.float32)
    mask = _mask(depth.shape)
    result = compose_relief(depth, [_region(mask, detail_amount=0.04)], detail)
    assert np.array_equal(result[~mask], depth[~mask])
    change = np.abs(result[mask] - depth[mask])
    assert float(change.max()) > 0.0
    assert float(change.max()) <= 0.04
    again = extract_source_detail(rgb)
    assert np.array_equal(again, detail)


def test_feather_spreads_detail_without_a_single_pixel_cliff() -> None:
    depth = np.full((48, 96), 0.4, dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[:, 40:70] = True
    detail = np.ones(depth.shape, dtype=np.float32)
    result = compose_relief(depth, [_region(mask, feather=20, detail_amount=0.05)], detail)
    row = result[24]
    assert abs(float(row[0]) - 0.4) < 1e-5
    assert float(row[55]) > 0.44
    assert float(np.max(np.abs(np.diff(row)))) < 0.02
    hard = compose_relief(depth, [_region(mask, feather=0, detail_amount=0.05)], detail)
    assert np.array_equal(hard[:, :40], depth[:, :40])
    assert np.array_equal(hard[:, 70:], depth[:, 70:])


def test_elevation_depth_range_order_and_overlap_still_apply() -> None:
    depth = np.full((8, 8), 0.4, dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[2:6, 2:6] = True
    detail = np.ones(depth.shape, dtype=np.float32)
    lifted = compose_relief(depth, [_region(mask, elevation=0.2, detail_amount=0.05)], detail)
    assert lifted[4, 4] == np.float32(0.4) + np.float32(0.2) + np.float32(0.05)
    assert lifted[0, 0] == np.float32(0.4)
    flat = compose_relief(depth, [_region(mask, depth_range=0.0, elevation=0.1)], None)
    assert flat[4, 4] == np.float32(0.5)
    left = np.zeros(depth.shape, dtype=bool)
    left[:, :5] = True
    right = np.zeros(depth.shape, dtype=bool)
    right[:, 3:] = True
    first = compose_relief(
        depth,
        [
            _region(left, region_id="early", elevation=0.1, order=0, detail_amount=0.0),
            _region(right, region_id="late", elevation=0.0, order=1, detail_amount=0.05),
        ],
        detail,
    )
    swapped = compose_relief(
        depth,
        [
            _region(right, region_id="late", elevation=0.0, order=0, detail_amount=0.05),
            _region(left, region_id="early", elevation=0.1, order=1, detail_amount=0.0),
        ],
        detail,
    )
    overlap = (slice(None), slice(3, 5))
    assert not np.array_equal(first[overlap], swapped[overlap])
    assert np.array_equal(first[:, :3], depth[:, :3] + np.float32(0.1))
    assert np.max(first[:, 5:]) > 0.4


def test_output_is_finite_float32_in_unit_interval() -> None:
    depth = np.full((16, 16), 0.95, dtype=np.float32)
    mask = np.ones(depth.shape, dtype=bool)
    detail = np.ones(depth.shape, dtype=np.float32)
    result = compose_relief(depth, [_region(mask, elevation=0.2, detail_amount=DETAIL_AMOUNT_MAX)], detail)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_invalid_detail_amounts_are_rejected() -> None:
    for amount in (float("nan"), float("inf"), float("-inf"), -0.01, 0.050001, 1.0):
        with pytest.raises(ValidationError):
            ReliefRegionUpdate.model_validate(_update(detail_amount=amount))
        with pytest.raises(ValidationError):
            ReliefRegionRecord.model_validate(
                {
                    "id": str(uuid4()),
                    "name": "Subject",
                    "role": "primary",
                    "mask_filename": "mask.png",
                    "detail_amount": amount,
                    "created_at": "2026-10-08T00:00:00Z",
                    "updated_at": "2026-10-08T00:00:00Z",
                }
            )
    depth = _depth()
    mask = np.ones(depth.shape, dtype=bool)
    with pytest.raises(CompositionError):
        compose_relief(depth, [_region(mask, detail_amount=0.2)], np.zeros(depth.shape, dtype=np.float32))


def test_detail_grid_must_match_the_depth_grid() -> None:
    depth = _depth()
    mask = np.ones(depth.shape, dtype=bool)
    with pytest.raises(CompositionError):
        compose_relief(
            depth,
            [_region(mask, detail_amount=0.01)],
            np.zeros((3, 3), dtype=np.float32),
        )
    rgb = _pattern(8, 8)
    assert extract_source_detail(rgb).shape == (8, 8)


def test_physical_mapping_formula_is_unchanged() -> None:
    relief = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    height = map_physical_height(relief, 12.7, 35.56)
    expected = np.float32(12.7) + relief * (np.float32(35.56) - np.float32(12.7))
    assert np.array_equal(height, expected)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def test_legacy_zero_detail_fingerprint_stays_current(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        body = _pipeline(project_root, client)
        versioned_hash = body["relief"]["settings_sha256"]
        manifest = _manifest(project_root)
        for region in manifest["relief"]["regions"]:
            region.pop("detail_amount", None)
        manifest["relief"].pop("settings_version", None)
        _write_manifest(project_root, manifest)
        project = Project.model_validate_json(
            (project_root / body["slug"] / "project.json").read_text(encoding="utf-8")
        )
        assert project.relief is not None
        assert project.relief.settings_version is None
        assert project.relief.regions[0].detail_amount == 0.0
        legacy_hash = canonical_sha256(
            composition_settings_document(project_root / body["slug"], project.relief.regions)
        )
        assert legacy_hash != versioned_hash
        manifest = _manifest(project_root)
        manifest["relief"]["settings_sha256"] = legacy_hash
        _write_manifest(project_root, manifest)
        current = client.get(f"/api/projects/{body['id']}")
        assert current.status_code == 200
        assert _statuses(current.json())["relief"] == "current"
        assert _statuses(current.json())["physical"] == "current"
        assert current.json()["relief"]["regions"][0]["detail_amount"] == 0.0
        manifest = _manifest(project_root)
        manifest["relief"]["regions"][0]["detail_amount"] = 0.02
        _write_manifest(project_root, manifest)
        stale = client.get(f"/api/projects/{body['id']}")
        assert _statuses(stale.json())["relief"] == "stale"
        assert _statuses(stale.json())["physical"] == "stale"
        manifest["relief"].pop("settings_sha256")
        _write_manifest(project_root, manifest)
        unknown = client.get(f"/api/projects/{body['id']}")
        assert _statuses(unknown.json())["relief"] == "unknown"


def test_nonzero_detail_includes_the_source_and_stales_only_downstream(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        before_files = _bytes(project_root)
        composed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": region["name"],
                        "role": region["role"],
                        "elevation": region["elevation"],
                        "depth_range": region["depth_range"],
                        "feather": region["feather"],
                        "order": region["order"],
                        "detail_amount": 0.02,
                    }
                ]
            },
        )
        assert composed.status_code == 200, composed.text
        relief = composed.json()["relief"]
        assert relief["settings_version"] == 2
        assert relief["status"] == "current"
        directory = project_root / body["slug"]
        source_hash = composition_source_sha256(directory)
        assert source_hash is not None
        project = Project.model_validate(composed.json())
        assert project.relief is not None
        document = composition_settings_document(
            directory,
            project.relief.regions,
            version=2,
            source_sha256=source_hash,
        )
        assert document["source_sha256"] == source_hash
        assert document["regions"][0]["detail_amount"] == 0.02
        assert relief["settings_sha256"] == canonical_sha256(document)
        raw_key = "depth/raw_depth.npy"
        conditioned_key = "depth/processing/conditioned_depth.npy"
        assert _bytes(project_root)[raw_key] == before_files[raw_key]
        assert _bytes(project_root)[conditioned_key] == before_files[conditioned_key]

        manifest = _manifest(project_root)
        manifest["relief"]["regions"][0]["detail_amount"] = 0.03
        _write_manifest(project_root, manifest)
        changed = client.get(f"/api/projects/{body['id']}")
        statuses = _statuses(changed.json())
        assert statuses["raw"] == "current"
        assert statuses["conditioned"] == "current"
        assert statuses["relief"] == "stale"
        assert statuses["physical"] == "stale"
        assert _bytes(project_root)[raw_key] == before_files[raw_key]
        assert _bytes(project_root)[conditioned_key] == before_files[conditioned_key]

        source = directory / "source" / "original.png"
        mutated = bytearray(source.read_bytes())
        mutated[-1] ^= 0x5A
        source.write_bytes(mutated)
        new_hash = hashlib.sha256(mutated).hexdigest()
        manifest = _manifest(project_root)
        manifest["relief"]["regions"][0]["detail_amount"] = 0.02
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


def test_zero_detail_does_not_read_the_source_image(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_detail(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("source detail was requested")

    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        monkeypatch.setattr("app.relief.service.ReliefService._read_source_rgb", fail_detail)
        composed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": region["name"],
                        "role": region["role"],
                        "elevation": region["elevation"],
                        "depth_range": region["depth_range"],
                        "feather": region["feather"],
                        "order": region["order"],
                    }
                ]
            },
        )
        assert composed.status_code == 200, composed.text
        assert composed.json()["relief"]["regions"][0]["detail_amount"] == 0.0


def test_source_size_mismatch_is_rejected(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        directory = project_root / body["slug"]
        replacement = image_bytes("PNG", (5, 5))
        (directory / "source" / "original.png").write_bytes(replacement)
        digest = hashlib.sha256(replacement).hexdigest()
        manifest = _manifest(project_root)
        manifest["source"]["content_sha256"] = digest
        manifest["source"]["width_px"] = 5
        manifest["source"]["height_px"] = 5
        manifest["depth"]["source_sha256"] = digest
        _write_manifest(project_root, manifest)
        rejected = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": region["name"],
                        "role": region["role"],
                        "elevation": 0.0,
                        "depth_range": 1.0,
                        "feather": 0.0,
                        "order": 0,
                        "detail_amount": 0.02,
                    }
                ]
            },
        )
        assert rejected.status_code == 422


def test_api_rejects_detail_above_the_maximum(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        rejected = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": "Subject",
                        "role": "primary",
                        "elevation": 0.0,
                        "depth_range": 1.0,
                        "feather": 0.0,
                        "order": 0,
                        "detail_amount": 0.06,
                    }
                ]
            },
        )
        assert rejected.status_code == 422
