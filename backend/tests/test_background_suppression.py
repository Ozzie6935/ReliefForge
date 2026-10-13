"""Opt-in uncovered-background suppression. Temporary projects and synthetic arrays only."""

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.projects.provenance import canonical_json, canonical_sha256, composition_settings_document
from app.relief.composition import CompositionError, apply_background, compose_relief, feather_weights
from app.relief.physical import map_physical_height
from app.schemas.project import Project
from app.schemas.relief import ReliefCompositionRequest, ReliefRecord
from tests.test_physical_detail_scale import served_client
from tests.test_provenance import _bytes, _manifest, _pipeline, _statuses, _write_manifest
from tests.test_relief import _region
from tests.test_source_detail import _update


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _edge_steps(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pieces = []
    pairs = (
        (mask[:, :-1] & ~mask[:, 1:], field[:, :-1], field[:, 1:]),
        (mask[:, 1:] & ~mask[:, :-1], field[:, 1:], field[:, :-1]),
        (mask[:-1, :] & ~mask[1:, :], field[:-1, :], field[1:, :]),
        (mask[1:, :] & ~mask[:-1, :], field[1:, :], field[:-1, :]),
    )
    for inside, left, right in pairs:
        if bool(inside.any()):
            pieces.append(np.abs(left[inside] - right[inside]))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def test_identity_background_is_byte_identical_and_keeps_feather_halos() -> None:
    depth = np.linspace(0.15, 0.85, 80 * 60, dtype=np.float32).reshape(60, 80)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[:, 20:55] = True
    legacy = compose_relief(depth, [_region(mask, elevation=0.2, depth_range=1.3, feather=12)])
    plain = compose_relief(depth, [_region(mask, elevation=0.2, depth_range=1.3)])
    assert not np.array_equal(legacy, plain)
    kept = apply_background(depth, legacy, [mask])
    assert kept is legacy
    explicit = apply_background(depth, legacy, [mask], depth_range=1.0, elevation=0.0, feather=0.0)
    assert explicit is legacy


def test_flat_background_is_the_uncovered_mean_and_feather_zero_keeps_interiors() -> None:
    depth = np.linspace(0.2, 0.9, 40 * 30, dtype=np.float32).reshape(30, 40)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[:, 12:] = True
    composed = compose_relief(depth, [_region(mask, elevation=0.2)])
    result = apply_background(depth, composed, [mask], depth_range=0.0, elevation=0.0, feather=0.0)
    expected = np.float32(np.mean(depth[~mask], dtype=np.float64))
    assert np.all(result[~mask] == expected)
    assert np.array_equal(result[mask], composed[mask])
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0 and float(result.max()) <= 1.0


def test_negative_elevation_lowers_the_uncovered_plane_and_clips() -> None:
    depth = np.full((20, 20), 0.55, dtype=np.float32)
    depth[:, :8] = 0.12
    mask = np.zeros(depth.shape, dtype=bool)
    mask[:, 8:] = True
    composed = compose_relief(depth, [_region(mask, elevation=0.1)])
    lowered = apply_background(depth, composed, [mask], depth_range=0.0, elevation=-0.2, feather=0.0)
    level = np.float32(np.mean(depth[~mask], dtype=np.float64))
    expected = np.clip(level - np.float32(0.2), 0.0, 1.0).astype(np.float32)
    assert float(lowered[~mask][0]) < float(composed[~mask][0])
    assert np.all(lowered[~mask] == expected)
    assert np.array_equal(lowered[mask], composed[mask])
    near_zero = np.full((12, 12), 0.05, dtype=np.float32)
    near_mask = np.zeros(near_zero.shape, dtype=bool)
    near_mask[:, 6:] = True
    clipped = apply_background(
        near_zero,
        compose_relief(near_zero, [_region(near_mask)]),
        [near_mask],
        depth_range=0.0,
        elevation=-0.2,
        feather=0.0,
    )
    assert float(clipped[~near_mask].max()) == 0.0


def test_positive_feather_changes_both_sides_and_leaves_the_deep_interior() -> None:
    depth = np.full((96, 96), 0.3, dtype=np.float32)
    depth[24:72, 24:72] = 0.7
    mask = np.zeros(depth.shape, dtype=bool)
    mask[24:72, 24:72] = True
    composed = compose_relief(depth, [_region(mask, elevation=0.15)])
    result = apply_background(depth, composed, [mask], depth_range=0.0, elevation=-0.25, feather=2.0)
    background_weight = np.float32(1.0) - feather_weights(mask, 2.0)
    interior = background_weight == 0.0
    assert bool(interior[48, 48] and mask[48, 48])
    assert np.array_equal(result[interior], composed[interior])
    assert not bool(mask[23, 48]) and result[23, 48] != composed[23, 48]
    assert bool(mask[24, 48]) and result[24, 48] != composed[24, 48]
    assert bool(background_weight[23, 48] > 0.0 and background_weight[24, 48] > 0.0)
    assert result.dtype == np.float32 and np.isfinite(result).all()


def test_empty_and_full_unions_skip_the_pass() -> None:
    depth = np.full((16, 16), 0.4, dtype=np.float32)
    composed = compose_relief(depth, [])
    assert apply_background(depth, composed, [], depth_range=0.0, elevation=-0.4, feather=10.0) is composed
    full = np.ones(depth.shape, dtype=bool)
    covered = compose_relief(depth, [_region(full, elevation=0.2)])
    assert apply_background(depth, covered, [full], depth_range=0.0) is covered
    empty = np.zeros(depth.shape, dtype=bool)
    untouched = compose_relief(depth, [_region(empty, elevation=0.2)])
    assert apply_background(depth, untouched, [empty], depth_range=0.0) is untouched


def test_overlap_pixels_stay_on_the_existing_region_rule() -> None:
    depth = np.full((32, 32), 0.4, dtype=np.float32)
    left = np.zeros(depth.shape, dtype=bool)
    right = np.zeros(depth.shape, dtype=bool)
    left[:, 8:20] = True
    right[:, 14:26] = True
    overlap = left & right
    composed = compose_relief(
        depth,
        [
            _region(left, elevation=0.2, order=0, region_id="a"),
            _region(right, elevation=0.45, order=1, region_id="b"),
        ],
    )
    reversed_order = compose_relief(
        depth,
        [
            _region(right, elevation=0.45, order=0, region_id="b"),
            _region(left, elevation=0.2, order=1, region_id="a"),
        ],
    )
    assert not np.array_equal(composed[overlap], reversed_order[overlap])
    suppressed = apply_background(depth, composed, [left, right], depth_range=0.0, elevation=-0.1, feather=0.0)
    union = left | right
    assert np.array_equal(suppressed[union], composed[union])
    assert np.array_equal(suppressed[overlap], composed[overlap])
    assert apply_background(depth, composed, [left, right]) is composed


def test_invalid_background_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ReliefCompositionRequest.model_validate(
            {"regions": [], "background_depth_range": float("nan")}
        )
    for payload in (
        {"background_depth_range": 2.01},
        {"background_elevation": -1.01},
        {"background_feather": 100.1},
        {"background_elevation": float("inf")},
        {"background_feather": float("-inf")},
    ):
        with pytest.raises(ValidationError):
            ReliefCompositionRequest.model_validate({"regions": [], **payload})
        with pytest.raises(ValidationError):
            ReliefRecord.model_validate({"regions": [], **payload})
    depth = np.full((4, 4), 0.4, dtype=np.float32)
    composed = compose_relief(depth, [])
    with pytest.raises(CompositionError):
        apply_background(depth, composed, [], depth_range=2.5)
    omitted = ReliefCompositionRequest.model_validate({"regions": []})
    assert omitted.background_depth_range == 1.0
    assert omitted.background_elevation == 0.0
    assert omitted.background_feather == 0.0
    historical = ReliefRecord.model_validate(
        {
            "regions": [
                {
                    "id": "00000000-0000-4000-8000-000000000010",
                    "name": "Subject",
                    "role": "primary",
                    "mask_filename": "00000000-0000-4000-8000-000000000010.png",
                    "created_at": "2026-10-08T00:00:00Z",
                    "updated_at": "2026-10-08T00:00:00Z",
                }
            ]
        }
    )
    assert historical.background_depth_range == 1.0
    assert historical.background_elevation == 0.0
    assert historical.background_feather == 0.0
    assert historical.settings_version is None


def test_identity_keeps_version_two_and_three_and_non_identity_writes_version_four(project_root: Path) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        before = _bytes(project_root)
        identity = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={"regions": [_update_from(region)]},
        )
        assert identity.status_code == 200, identity.text
        relief = identity.json()["relief"]
        assert relief["settings_version"] == 2
        assert relief["background_depth_range"] == 1.0
        directory = project_root / body["slug"]
        project = Project.model_validate(identity.json())
        assert project.relief is not None
        encoded = canonical_json(composition_settings_document(directory, project.relief.regions, version=2))
        assert "background_depth_range" not in encoded
        assert "detail_scale_mm" not in encoded
        assert "artwork_width_mm" not in encoded
        assert relief["settings_sha256"] == canonical_sha256(
            composition_settings_document(directory, project.relief.regions, version=2)
        )
        assert _bytes(project_root)["depth/composition/relief_depth.npy"] == before["depth/composition/relief_depth.npy"]

        scaled = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={"regions": [_update_from(region, detail_scale_mm=0.5)]},
        )
        assert scaled.status_code == 200, scaled.text
        assert scaled.json()["relief"]["settings_version"] == 3
        scaled_project = Project.model_validate(scaled.json())
        assert scaled_project.relief is not None
        version3 = canonical_json(
            composition_settings_document(
                directory,
                scaled_project.relief.regions,
                version=3,
                artwork_width_mm=scaled_project.artwork.width_mm,
                artwork_height_mm=scaled_project.artwork.height_mm,
            )
        )
        assert "background_depth_range" not in version3
        assert "detail_scale_mm" in version3

        suppressed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [_update_from(region)],
                "background_depth_range": 0.0,
                "background_elevation": -0.2,
                "background_feather": 0.0,
            },
        )
        assert suppressed.status_code == 200, suppressed.text
        payload = suppressed.json()["relief"]
        assert payload["settings_version"] == 4
        assert payload["status"] == "current"
        version4_project = Project.model_validate(suppressed.json())
        assert version4_project.relief is not None
        document = composition_settings_document(
            directory,
            version4_project.relief.regions,
            version=4,
            background_depth_range=0.0,
            background_elevation=-0.2,
            background_feather=0.0,
        )
        encoded4 = canonical_json(document)
        assert '"version":4' in encoded4
        assert "background_depth_range" in encoded4
        assert "detail_amount" in encoded4
        assert "detail_scale_mm" in encoded4
        assert "artwork_width_mm" not in encoded4
        assert "source_sha256" not in encoded4
        assert payload["settings_sha256"] == canonical_sha256(document)
        assert _bytes(project_root)["depth/raw_depth.npy"] == before["depth/raw_depth.npy"]
        assert _bytes(project_root)["depth/processing/conditioned_depth.npy"] == before[
            "depth/processing/conditioned_depth.npy"
        ]


def test_version_four_source_and_artwork_follow_existing_rules(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_read(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("source was read")

    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        directory = project_root / body["slug"]
        monkeypatch.setattr("app.relief.service.ReliefService._read_source_rgb", fail_read)
        quiet = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [_update_from(region, detail_amount=0.0, detail_scale_mm=0.5)],
                "background_depth_range": 0.0,
            },
        )
        assert quiet.status_code == 200, quiet.text
        assert quiet.json()["relief"]["settings_version"] == 4
        project = Project.model_validate(quiet.json())
        assert project.relief is not None
        document = composition_settings_document(
            directory,
            project.relief.regions,
            version=4,
            artwork_width_mm=project.artwork.width_mm,
            artwork_height_mm=project.artwork.height_mm,
            background_depth_range=0.0,
            background_elevation=0.0,
            background_feather=0.0,
        )
        encoded = canonical_json(document)
        assert "artwork_width_mm" in encoded
        assert "source_sha256" not in encoded

        monkeypatch.undo()
        detailed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [_update_from(region, detail_amount=0.02)],
                "background_elevation": -0.1,
            },
        )
        assert detailed.status_code == 200, detailed.text
        detailed_project = Project.model_validate(detailed.json())
        assert detailed_project.relief is not None
        detailed_document = composition_settings_document(
            directory,
            detailed_project.relief.regions,
            version=4,
            source_sha256=detailed_project.relief.settings_sha256 and None,
            background_depth_range=1.0,
            background_elevation=-0.1,
            background_feather=0.0,
        )
        # Rebuild with the real source hash by asking the saved record's judge path.
        from app.projects.provenance import composition_source_sha256

        detailed_document = composition_settings_document(
            directory,
            detailed_project.relief.regions,
            version=4,
            source_sha256=composition_source_sha256(directory),
            background_depth_range=1.0,
            background_elevation=-0.1,
            background_feather=0.0,
        )
        assert "source_sha256" in canonical_json(detailed_document)
        assert "artwork_width_mm" not in canonical_json(detailed_document)
        assert detailed.json()["relief"]["settings_sha256"] == canonical_sha256(detailed_document)


def test_background_edits_stale_relief_and_physical_without_touching_depth(project_root: Path) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        before = _bytes(project_root)
        composed = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={"regions": [_update_from(region)], "background_depth_range": 0.0},
        )
        assert composed.status_code == 200, composed.text
        assert _statuses(composed.json())["relief"] == "current"
        assert composed.json()["relief"]["settings_version"] == 4
        assert _statuses(composed.json())["physical"] == "stale"
        assert _bytes(project_root)["depth/raw_depth.npy"] == before["depth/raw_depth.npy"]
        assert _bytes(project_root)["depth/processing/conditioned_depth.npy"] == before[
            "depth/processing/conditioned_depth.npy"
        ]
        relief_bytes = (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes()
        manifest = _manifest(project_root)
        manifest["relief"]["background_elevation"] = -0.25
        _write_manifest(project_root, manifest)
        changed = client.get(f"/api/projects/{body['id']}")
        statuses = _statuses(changed.json())
        assert statuses["relief"] == "stale"
        assert statuses["physical"] == "stale"
        assert statuses["raw"] == "current"
        assert statuses["conditioned"] == "current"
        assert (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes() == relief_bytes

        manifest = _manifest(project_root)
        manifest["relief"]["background_depth_range"] = 1.0
        manifest["relief"]["background_elevation"] = 0.0
        manifest["relief"]["background_feather"] = 0.0
        _write_manifest(project_root, manifest)
        restored = client.get(f"/api/projects/{body['id']}")
        assert _statuses(restored.json())["relief"] == "stale"
        assert _statuses(restored.json())["relief"] != "current"
        assert restored.json()["relief"]["settings_version"] == 4

        manifest = _manifest(project_root)
        manifest["relief"].pop("settings_sha256")
        _write_manifest(project_root, manifest)
        unknown = client.get(f"/api/projects/{body['id']}")
        assert _statuses(unknown.json())["relief"] == "unknown"
        assert _statuses(unknown.json())["relief"] != "current"


def test_version_two_record_with_a_non_identity_background_is_stale(project_root: Path) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        relief_bytes = (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes()
        manifest = _manifest(project_root)
        assert manifest["relief"]["settings_version"] == 2
        manifest["relief"]["background_depth_range"] = 0.0
        _write_manifest(project_root, manifest)
        stale = client.get(f"/api/projects/{body['id']}")
        assert _statuses(stale.json())["relief"] == "stale"
        assert _statuses(stale.json())["physical"] == "stale"
        assert _statuses(stale.json())["raw"] == "current"
        assert _statuses(stale.json())["conditioned"] == "current"
        assert (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes() == relief_bytes

        restored = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={"regions": [_update_from(region, detail_scale_mm=0.5)]},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["relief"]["settings_version"] == 3
        version3_bytes = (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes()
        manifest = _manifest(project_root)
        manifest["relief"]["background_elevation"] = -0.2
        _write_manifest(project_root, manifest)
        version3 = client.get(f"/api/projects/{body['id']}")
        assert _statuses(version3.json())["relief"] == "stale"
        assert _statuses(version3.json())["physical"] == "stale"
        assert (project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy").read_bytes() == version3_bytes

        historical = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={"regions": [_update_from(region)]},
        )
        assert historical.status_code == 200, historical.text
        directory = project_root / body["slug"]
        project = Project.model_validate(historical.json())
        assert project.relief is not None
        manifest = _manifest(project_root)
        manifest["relief"]["settings_version"] = None
        manifest["relief"]["settings_sha256"] = canonical_sha256(
            composition_settings_document(directory, project.relief.regions, version=None)
        )
        _write_manifest(project_root, manifest)
        current = client.get(f"/api/projects/{body['id']}")
        assert _statuses(current.json())["relief"] == "current"
        null_bytes = (directory / "depth" / "composition" / "relief_depth.npy").read_bytes()
        manifest = _manifest(project_root)
        manifest["relief"]["background_depth_range"] = 0.0
        _write_manifest(project_root, manifest)
        null_stale = client.get(f"/api/projects/{body['id']}")
        assert _statuses(null_stale.json())["relief"] == "stale"
        assert _statuses(null_stale.json())["physical"] == "stale"
        assert (directory / "depth" / "composition" / "relief_depth.npy").read_bytes() == null_bytes


def test_invalid_api_value_does_not_replace_published_files(project_root: Path) -> None:
    with served_client(project_root) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        before = _bytes(project_root)
        manifest_before = (project_root / body["slug"] / "project.json").read_bytes()
        rejected = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={"regions": [_update_from(region)], "background_depth_range": 2.5},
        )
        assert rejected.status_code == 422
        assert _bytes(project_root)["depth/composition/relief_depth.npy"] == before["depth/composition/relief_depth.npy"]
        assert (project_root / body["slug"] / "project.json").read_bytes() == manifest_before


def test_physical_formula_is_unchanged() -> None:
    relief = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    height = map_physical_height(relief, 12.7, 35.56)
    expected = np.float32(12.7) + relief * np.float32(35.56 - 12.7)
    assert np.array_equal(height, expected)


def test_artistic_counterfactuals_keep_subject_shape(tmp_path: Path) -> None:
    wedding = np.full((80, 80), 0.22, dtype=np.float32)
    subject = np.zeros(wedding.shape, dtype=bool)
    subject[10:70, 10:70] = True
    columns = np.linspace(0.55, 0.99, 60, dtype=np.float32)
    wedding[10:70, 10:70] = columns[None, :]
    composed = compose_relief(wedding, [_region(subject)])
    inside = composed[subject]
    span_before = float(np.percentile(inside, 95) - np.percentile(inside, 5))
    wedding_before = _edge_steps(composed, subject)
    suppressed = apply_background(wedding, composed, [subject], depth_range=0.0, elevation=-0.15, feather=0.0)
    wedding_after = _edge_steps(suppressed, subject)
    assert np.array_equal(suppressed[subject], composed[subject])
    span_after = float(np.percentile(suppressed[subject], 95) - np.percentile(suppressed[subject], 5))
    assert span_after == span_before
    assert span_before > 0.3
    assert float(np.percentile(inside, 95)) > 0.9

    flat = np.full((48, 72), 0.2, dtype=np.float32)
    flat[:, :40] = np.linspace(0.35, 0.95, 40, dtype=np.float32)[None, :]
    secondary = np.zeros(flat.shape, dtype=bool)
    secondary[:, 40:] = True
    flat[secondary] = np.float32(0.04)
    raised = compose_relief(flat, [_region(secondary, elevation=0.35, region_id="secondary")])
    before_steps = _edge_steps(raised, secondary)
    lowered = apply_background(flat, raised, [secondary], depth_range=0.0, elevation=-0.3, feather=0.0)
    after_steps = _edge_steps(lowered, secondary)
    assert np.array_equal(lowered[secondary], raised[secondary])
    assert float(np.std(lowered[secondary])) == float(np.std(raised[secondary]))
    assert float(np.std(lowered[secondary])) < 1e-6
    report = {
        "wedding_like_internal_p95_p5": span_before,
        "wedding_like_span_unchanged": span_after == span_before,
        "wedding_like_step_before_median": float(np.median(wedding_before)),
        "wedding_like_step_before_max": float(wedding_before.max()),
        "wedding_like_step_after_median": float(np.median(wedding_after)),
        "wedding_like_step_after_max": float(wedding_after.max()),
        "wedding_like_hard_cliff_max_increased": bool(wedding_after.max() > wedding_before.max()),
        "flat_subject_step_before_median": float(np.median(before_steps)),
        "flat_subject_step_before_max": float(before_steps.max()),
        "flat_subject_step_after_median": float(np.median(after_steps)),
        "flat_subject_step_after_max": float(after_steps.max()),
        "flat_subject_hard_cliff_max_increased": bool(after_steps.max() > before_steps.max()),
    }
    destination = Path("/tmp/rf122")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "counterfactual.json").write_text(json.dumps(report), encoding="utf-8")
    assert report["wedding_like_hard_cliff_max_increased"] is True
    assert report["flat_subject_hard_cliff_max_increased"] is False


def test_background_pass_performance() -> None:
    report = {}
    shapes = {"wedding": (1023, 1013), "angie": (2140, 2940)}
    for name, shape in shapes.items():
        depth = np.full(shape, 0.35, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        mask[:, shape[1] // 4 : 3 * shape[1] // 4] = True
        depth[mask] = 0.7
        composed = compose_relief(depth, [_region(mask, elevation=0.2)])
        modes = {}
        for feather in (0.0, 25.0):
            apply_background(depth, composed, [mask], depth_range=0.0, elevation=-0.1, feather=feather)
            tracemalloc.start()
            started = time.perf_counter()
            result = apply_background(depth, composed, [mask], depth_range=0.0, elevation=-0.1, feather=feather)
            modes[f"feather_{int(feather)}"] = {
                "seconds": time.perf_counter() - started,
                "peak_bytes": tracemalloc.get_traced_memory()[1],
                "dtype": str(result.dtype),
                "min": float(result.min()),
                "max": float(result.max()),
            }
            tracemalloc.stop()
            assert result.dtype == np.float32
            assert np.isfinite(result).all()
            assert 0.0 <= float(result.min()) and float(result.max()) <= 1.0
        report[name] = modes
    destination = Path("/tmp/rf122")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "perf.json").write_text(json.dumps(report), encoding="utf-8")


def _update_from(region: dict, **overrides: object) -> dict[str, object]:
    body = _update(
        id=region["id"],
        name=region["name"],
        role=region["role"],
        elevation=region["elevation"],
        depth_range=region["depth_range"],
        feather=region["feather"],
        order=region["order"],
        detail_amount=region.get("detail_amount") or 0.0,
        detail_scale_mm=region.get("detail_scale_mm"),
    )
    body.update(overrides)
    return body
