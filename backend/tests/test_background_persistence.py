"""Accept and delete must keep global background settings. Temporary projects only."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.projects.provenance import (
    canonical_sha256,
    composition_settings_document,
    composition_source_sha256,
)
from app.relief.artifacts import read_mask_png
from app.relief.composition import RegionContribution, apply_background, compose_relief, prepare_detail_signals
from app.schemas.project import Project
from tests.test_physical_detail_scale import served_client
from tests.test_physical_height import _generate
from tests.test_relief import FakeSegmentation, _bind, _conditioned

BACKGROUND = {
    "background_depth_range": 0.0,
    "background_elevation": -0.20,
    "background_feather": 25.0,
}


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def test_version_four_background_survives_accept_and_delete(project_root: Path) -> None:
    with served_client(project_root) as client:
        project, _segmentation = _open(project_root, client)
        first = _accept_box(client, project["id"], _box(0, 0, 3, 2), "Subject")
        region = first["relief"]["regions"][0]
        composed = _compose(
            client,
            project["id"],
            [_region_body(region, elevation=0.25)],
            BACKGROUND,
        )
        _assert_background(composed, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, composed)
        assert composed["relief"]["status"] == "current"
        raw_before, conditioned_before = _depth_bytes(project_root, composed["slug"])
        physical = _generate(client, project["id"])
        assert physical.status_code == 200, physical.text
        physical_bytes = _physical_bytes(project_root, composed["slug"])
        relief_before = _relief_bytes(project_root, composed["slug"])
        manifest_before = _manifest_bytes(project_root, composed["slug"])

        rejected = client.post(
            f"/api/projects/{project['id']}/relief/composition",
            json={
                "regions": [_region_body(region, elevation=0.25)],
                "background_depth_range": 2.5,
                "background_elevation": -0.20,
                "background_feather": 25,
            },
        )
        assert rejected.status_code == 422
        assert _relief_bytes(project_root, composed["slug"]) == relief_before
        assert _manifest_bytes(project_root, composed["slug"]) == manifest_before

        second = _accept_box(client, project["id"], _box(4, 2, 6, 4), "Second")
        _assert_background(second, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, second)
        assert second["relief"]["status"] == "current"
        assert len(second["relief"]["regions"]) == 2
        assert _relief_bytes(project_root, second["slug"]) != relief_before
        assert _depth_bytes(project_root, second["slug"]) == (raw_before, conditioned_before)
        assert second["relief"]["physical"]["status"] == "stale"
        assert _physical_bytes(project_root, second["slug"]) == physical_bytes

        removed = second["relief"]["regions"][1]["id"]
        deleted = client.delete(f"/api/projects/{project['id']}/relief/regions/{removed}")
        assert deleted.status_code == 200, deleted.text
        restored = deleted.json()
        _assert_background(restored, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, restored)
        assert restored["relief"]["status"] == "current"
        assert [item["id"] for item in restored["relief"]["regions"]] == [region["id"]]
        assert _relief_bytes(project_root, restored["slug"]) == relief_before
        assert _depth_bytes(project_root, restored["slug"]) == (raw_before, conditioned_before)
        assert restored["relief"]["physical"]["status"] == "current"
        assert _physical_bytes(project_root, restored["slug"]) == physical_bytes


def test_identity_background_accept_and_delete_keep_elevation_and_physical(project_root: Path) -> None:
    with served_client(project_root) as client:
        project, _segmentation = _open(project_root, client)
        first = _accept_box(client, project["id"], _box(0, 0, 4, 4), "Subject")
        region = first["relief"]["regions"][0]
        composed = _compose(client, project["id"], [_region_body(region, elevation=0.25)], {})
        assert composed["relief"]["settings_version"] == 2
        _assert_background(composed, 2, {"background_depth_range": 1.0, "background_elevation": 0.0, "background_feather": 0.0})
        _assert_matches_recomputed_relief(project_root, composed)
        physical = _generate(client, project["id"])
        assert physical.status_code == 200, physical.text
        relief_before = _relief_bytes(project_root, composed["slug"])
        conditioned = np.load(project_root / composed["slug"] / "depth" / "processing" / "conditioned_depth.npy")
        elevated = np.load(project_root / composed["slug"] / "depth" / "composition" / "relief_depth.npy")
        assert not np.array_equal(elevated, conditioned)

        second = _accept_box(client, project["id"], _box(2, 0, 6, 4), "Identity")
        _assert_background(second, 2, {"background_depth_range": 1.0, "background_elevation": 0.0, "background_feather": 0.0})
        _assert_matches_recomputed_relief(project_root, second)
        assert second["relief"]["status"] == "current"
        assert _relief_bytes(project_root, second["slug"]) == relief_before
        assert second["relief"]["physical"]["status"] == "current"
        assert second["relief"]["regions"][0]["elevation"] == 0.25
        assert second["relief"]["regions"][1]["elevation"] == 0.0

        deleted = client.delete(
            f"/api/projects/{project['id']}/relief/regions/{second['relief']['regions'][1]['id']}"
        )
        assert deleted.status_code == 200, deleted.text
        body = deleted.json()
        _assert_background(body, 2, {"background_depth_range": 1.0, "background_elevation": 0.0, "background_feather": 0.0})
        _assert_matches_recomputed_relief(project_root, body)
        assert body["relief"]["status"] == "current"
        assert body["relief"]["physical"]["status"] == "current"
        assert _relief_bytes(project_root, body["slug"]) == relief_before
        assert body["relief"]["regions"][0]["elevation"] == 0.25


def test_background_and_physical_scale_survive_accept_and_delete(project_root: Path) -> None:
    with served_client(project_root) as client:
        project, _segmentation = _open(project_root, client)
        first = _accept_box(client, project["id"], _box(0, 0, 3, 2), "Subject")
        region = first["relief"]["regions"][0]
        composed = _compose(
            client,
            project["id"],
            [_region_body(region, detail_scale_mm=0.5)],
            BACKGROUND,
        )
        _assert_background(composed, 4, BACKGROUND)
        assert composed["relief"]["regions"][0]["detail_scale_mm"] == 0.5
        _assert_matches_recomputed_relief(project_root, composed)
        encoded = _document_text(project_root, composed)
        assert "artwork_width_mm" in encoded
        assert "source_sha256" not in encoded

        second = _accept_box(client, project["id"], _box(4, 2, 6, 4), "Second")
        _assert_background(second, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, second)
        assert second["relief"]["status"] == "current"
        scales = {item["name"]: item["detail_scale_mm"] for item in second["relief"]["regions"]}
        assert scales == {"Subject": 0.5, "Second": None}
        encoded = _document_text(project_root, second)
        assert "artwork_width_mm" in encoded
        assert "source_sha256" not in encoded

        deleted = client.delete(
            f"/api/projects/{project['id']}/relief/regions/{second['relief']['regions'][1]['id']}"
        )
        assert deleted.status_code == 200, deleted.text
        restored = deleted.json()
        _assert_background(restored, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, restored)
        assert restored["relief"]["regions"][0]["detail_scale_mm"] == 0.5


def test_version_four_source_hash_survives_accept(project_root: Path) -> None:
    with served_client(project_root) as client:
        project, _segmentation = _open(project_root, client)
        first = _accept_box(client, project["id"], _box(0, 0, 3, 3), "Subject")
        region = first["relief"]["regions"][0]
        composed = _compose(
            client,
            project["id"],
            [_region_body(region, detail_amount=0.02)],
            BACKGROUND,
        )
        _assert_background(composed, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, composed)
        encoded = _document_text(project_root, composed)
        assert "source_sha256" in encoded
        assert "artwork_width_mm" not in encoded
        raw_before, conditioned_before = _depth_bytes(project_root, composed["slug"])

        second = _accept_box(client, project["id"], _box(4, 1, 6, 3), "Second")
        _assert_background(second, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, second)
        assert second["relief"]["status"] == "current"
        amounts = {item["name"]: item["detail_amount"] for item in second["relief"]["regions"]}
        assert amounts["Subject"] == 0.02
        assert amounts["Second"] == 0.0
        encoded = _document_text(project_root, second)
        assert "source_sha256" in encoded
        omitted = canonical_sha256(_document(project_root, second, include_source=False))
        assert second["relief"]["settings_sha256"] != omitted
        assert _depth_bytes(project_root, second["slug"]) == (raw_before, conditioned_before)


def test_deleting_the_last_region_does_not_invent_a_background_plane(project_root: Path) -> None:
    with served_client(project_root) as client:
        project, _segmentation = _open(project_root, client)
        first = _accept_box(client, project["id"], _box(0, 0, 3, 2), "Subject")
        region = first["relief"]["regions"][0]
        composed = _compose(
            client,
            project["id"],
            [_region_body(region, elevation=0.25)],
            BACKGROUND,
        )
        suppressed = np.load(project_root / composed["slug"] / "depth" / "composition" / "relief_depth.npy")
        conditioned = np.load(project_root / composed["slug"] / "depth" / "processing" / "conditioned_depth.npy")
        assert not np.array_equal(suppressed, conditioned)
        raw_before, conditioned_before = _depth_bytes(project_root, composed["slug"])

        deleted = client.delete(f"/api/projects/{project['id']}/relief/regions/{region['id']}")
        assert deleted.status_code == 200, deleted.text
        body = deleted.json()
        _assert_background(body, 4, BACKGROUND)
        _assert_matches_recomputed_relief(project_root, body)
        assert body["relief"]["regions"] == []
        assert body["relief"]["status"] == "current"
        restored = np.load(project_root / body["slug"] / "depth" / "composition" / "relief_depth.npy")
        assert np.array_equal(restored, conditioned)
        assert not np.array_equal(restored, suppressed)
        assert _depth_bytes(project_root, body["slug"]) == (raw_before, conditioned_before)


def _open(project_root: Path, client) -> tuple[dict, FakeSegmentation]:
    segmentation = FakeSegmentation()
    project = _conditioned(project_root, client)
    _bind(project_root, segmentation)
    return project, segmentation


def _box(x0: float, y0: float, x1: float, y1: float) -> dict[str, float]:
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _accept_box(client, project_id: str, box: dict[str, float], name: str) -> dict:
    predicted = client.post(
        f"/api/projects/{project_id}/segmentation/predict",
        json={"box": box, "name": name, "role": "primary"},
    )
    assert predicted.status_code == 200, predicted.text
    accepted = client.post(
        f"/api/projects/{project_id}/relief/regions",
        json={"name": name, "role": "primary", "proposal_id": predicted.json()["proposal_id"]},
    )
    assert accepted.status_code == 200, accepted.text
    return accepted.json()


def _region_body(region: dict, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": region["id"],
        "name": region["name"],
        "role": region["role"],
        "elevation": region["elevation"],
        "depth_range": region["depth_range"],
        "feather": region["feather"],
        "order": region["order"],
        "detail_amount": region.get("detail_amount") or 0.0,
        "detail_scale_mm": region.get("detail_scale_mm"),
    }
    body.update(overrides)
    return body


def _compose(client, project_id: str, regions: list[dict], background: dict) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/relief/composition",
        json={"regions": regions, **background},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _assert_background(body: dict, version: int, background: dict[str, float]) -> None:
    relief = body["relief"]
    assert relief["settings_version"] == version
    assert relief["background_depth_range"] == background["background_depth_range"]
    assert relief["background_elevation"] == background["background_elevation"]
    assert relief["background_feather"] == background["background_feather"]


def _document(project_root: Path, body: dict, *, include_source: bool = True) -> dict:
    project = Project.model_validate(body)
    assert project.relief is not None
    directory = project_root / project.slug
    relief = project.relief
    source = None
    if include_source and any(region.detail_amount > 0.0 for region in relief.regions):
        source = composition_source_sha256(directory)
    return composition_settings_document(
        directory,
        relief.regions,
        version=relief.settings_version,
        source_sha256=source,
        artwork_width_mm=project.artwork.width_mm,
        artwork_height_mm=project.artwork.height_mm,
        background_depth_range=relief.background_depth_range,
        background_elevation=relief.background_elevation,
        background_feather=relief.background_feather,
    )


def _document_text(project_root: Path, body: dict) -> str:
    from app.projects.provenance import canonical_json

    return canonical_json(_document(project_root, body))


def _assert_matches_recomputed_relief(project_root: Path, body: dict) -> None:
    project = Project.model_validate(body)
    assert project.relief is not None
    relief = project.relief
    directory = project_root / project.slug
    assert relief.settings_sha256 == canonical_sha256(_document(project_root, body))
    conditioned = np.load(directory / "depth" / "processing" / "conditioned_depth.npy")
    contributions: list[RegionContribution] = []
    masks: list[np.ndarray] = []
    for region in relief.regions:
        mask = read_mask_png(directory / "depth" / "composition" / "masks" / region.mask_filename)
        masks.append(mask)
        contributions.append(
            RegionContribution(
                region_id=str(region.id),
                mask=mask,
                elevation=region.elevation,
                depth_range=region.depth_range,
                feather=region.feather,
                order=region.order,
                detail_amount=region.detail_amount,
                detail_scale_mm=region.detail_scale_mm,
            )
        )
    detail = None
    physical_details = None
    if any(region.detail_amount > 0.0 for region in relief.regions):
        with Image.open(directory / "source" / project.source.filename) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        detail, physical_details = prepare_detail_signals(
            rgb,
            contributions,
            artwork_width_mm=project.artwork.width_mm,
            artwork_height_mm=project.artwork.height_mm,
        )
    composed = compose_relief(conditioned, contributions, detail, physical_details)
    expected = apply_background(
        conditioned,
        composed,
        masks,
        depth_range=relief.background_depth_range,
        elevation=relief.background_elevation,
        feather=relief.background_feather,
    )
    stored = np.load(directory / "depth" / "composition" / "relief_depth.npy")
    assert np.array_equal(stored, expected)
    assert stored.dtype == np.float32
    assert np.isfinite(stored).all()


def _depth_bytes(project_root: Path, slug: str) -> tuple[bytes, bytes]:
    root = project_root / slug
    return (
        (root / "depth" / "raw_depth.npy").read_bytes(),
        (root / "depth" / "processing" / "conditioned_depth.npy").read_bytes(),
    )


def _relief_bytes(project_root: Path, slug: str) -> bytes:
    return (project_root / slug / "depth" / "composition" / "relief_depth.npy").read_bytes()


def _physical_bytes(project_root: Path, slug: str) -> bytes:
    return (project_root / slug / "depth" / "composition" / "physical" / "physical_height_mm.npy").read_bytes()


def _manifest_bytes(project_root: Path, slug: str) -> bytes:
    return (project_root / slug / "project.json").read_bytes()
