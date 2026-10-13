"""Edit an accepted region mask through a proposal copy. No SAM or depth inference."""

import hashlib
import json
from pathlib import Path

import pytest

from app.projects.storage import ProjectStore
from tests.test_projects import served
from tests.test_relief import FakeSegmentation, _bind, _conditioned


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accept(client, project_id: str, segmentation: FakeSegmentation) -> dict:
    predicted = client.post(
        f"/api/projects/{project_id}/segmentation/predict",
        json={"positive_points": [{"x": 1, "y": 1}], "name": "Subject", "role": "primary"},
    )
    assert predicted.status_code == 200, predicted.text
    assert predicted.json()["origin_type"] == "new_region"
    accepted = client.post(
        f"/api/projects/{project_id}/relief/regions",
        json={"name": "Subject", "role": "primary", "proposal_id": predicted.json()["proposal_id"]},
    )
    assert accepted.status_code == 200, accepted.text
    composed = client.post(
        f"/api/projects/{project_id}/relief/composition",
        json={
            "regions": [
                {
                    "id": accepted.json()["relief"]["regions"][0]["id"],
                    "name": "Subject",
                    "role": "primary",
                    "elevation": 0.25,
                    "depth_range": 1.4,
                    "feather": 6,
                    "order": 0,
                }
            ]
        },
    )
    assert composed.status_code == 200, composed.text
    assert segmentation.predicts == 1
    return composed.json()


def _paths(project_root: Path, region_id: str):
    root = project_root / "sample-relief"
    return {
        "mask": root / "depth" / "composition" / "masks" / f"{region_id}.png",
        "raw": root / "depth" / "raw_depth.npy",
        "conditioned": root / "depth" / "processing" / "conditioned_depth.npy",
        "relief": root / "depth" / "composition" / "relief_depth.npy",
    }


def _open(client, project_id: str, region_id: str):
    opened = client.post(f"/api/projects/{project_id}/relief/regions/{region_id}/mask-edit")
    assert opened.status_code == 200, opened.text
    return opened.json()


def test_edit_copies_the_accepted_mask_without_inference(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        saved = _accept(client, project["id"], segmentation)
        region = saved["relief"]["regions"][0]
        paths = _paths(project_root, region["id"])
        before = {name: _sha256(path) for name, path in paths.items()}
        predicts = segmentation.predicts
        proposal = _open(client, project["id"], region["id"])
        folder = (
            project_root
            / "sample-relief"
            / "depth"
            / "composition"
            / "proposals"
            / proposal["proposal_id"]
        )
        assert proposal["origin_type"] == "accepted_region"
        assert proposal["region_id"] == region["id"]
        assert proposal["name"] == "Subject"
        assert proposal["role"] == "primary"
        assert proposal["model_id"] == "accepted-region"
        for name in ("ai_mask.png", "working_mask.png", "session_base_mask.png"):
            assert (folder / name).read_bytes() == paths["mask"].read_bytes()
        metadata = json.loads((folder / "proposal.json").read_text(encoding="utf-8"))
        assert metadata["origin_type"] == "accepted_region"
        assert metadata["region_id"] == region["id"]
        assert segmentation.predicts == predicts
        assert {name: _sha256(path) for name, path in paths.items()} == before


def test_brush_wand_and_undo_change_only_the_working_copy(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        region = _accept(client, project["id"], segmentation)["relief"]["regions"][0]
        paths = _paths(project_root, region["id"])
        permanent = paths["mask"].read_bytes()
        proposal = _open(client, project["id"], region["id"])
        folder = (
            project_root
            / "sample-relief"
            / "depth"
            / "composition"
            / "proposals"
            / proposal["proposal_id"]
        )
        painted = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal['proposal_id']}/strokes",
            json={"mode": "add", "brush_radius_px": 0.6, "points": [{"x": 3.5, "y": 1.5}]},
        )
        assert painted.status_code == 200, painted.text
        assert (folder / "working_mask.png").read_bytes() != permanent
        assert (folder / "ai_mask.png").read_bytes() == permanent
        assert (folder / "session_base_mask.png").read_bytes() == permanent
        assert paths["mask"].read_bytes() == permanent
        wand = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal['proposal_id']}/magic-wand",
            json={"mode": "add", "x": 0.5, "y": 0.5, "tolerance": 100, "contiguous": True},
        )
        assert wand.status_code == 200, wand.text
        assert (folder / "working_mask.png").read_bytes() != permanent
        assert (folder / "ai_mask.png").read_bytes() == permanent
        assert paths["mask"].read_bytes() == permanent
        restored = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal['proposal_id']}/refinement-session/restore",
        )
        assert restored.status_code == 200, restored.text
        assert (folder / "working_mask.png").read_bytes() == permanent
        reset = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal['proposal_id']}/strokes",
            json={"mode": "add", "brush_radius_px": 0.6, "points": [{"x": 4.5, "y": 2.5}]},
        )
        assert reset.status_code == 200, reset.text
        returned = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal['proposal_id']}/reset",
        )
        assert returned.status_code == 200, returned.text
        assert (folder / "working_mask.png").read_bytes() == permanent
        assert (folder / "session_base_mask.png").read_bytes() == permanent
        assert paths["mask"].read_bytes() == permanent


def test_discard_leaves_the_permanent_mask_and_save_updates_only_that_region(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        saved = _accept(client, project["id"], segmentation)
        region = saved["relief"]["regions"][0]
        paths = _paths(project_root, region["id"])
        permanent = paths["mask"].read_bytes()
        relief_hash = _sha256(paths["relief"])
        groom = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 2, "y": 2}], "name": "Groom", "role": "secondary"},
        )
        assert groom.status_code == 200, groom.text
        proposal = _open(client, project["id"], region["id"])
        current = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert current.status_code == 200, current.text
        assert current.json()["proposal_id"] == groom.json()["proposal_id"]
        assert current.json()["origin_type"] == "new_region"
        discarded = client.post(
            f"/api/projects/{project['id']}/relief/regions/{region['id']}/mask-edit/discard",
            json={"proposal_id": proposal["proposal_id"]},
        )
        assert discarded.status_code == 204, discarded.text
        after_discard = client.get(f"/api/projects/{project['id']}").json()["relief"]["regions"][0]
        assert after_discard["id"] == region["id"]
        assert after_discard["name"] == region["name"]
        assert after_discard["role"] == region["role"]
        assert paths["mask"].read_bytes() == permanent
        assert (project_root / "sample-relief" / "depth" / "composition" / "proposals" / groom.json()["proposal_id"]).is_dir()
        again = _open(client, project["id"], region["id"])
        folder = (
            project_root
            / "sample-relief"
            / "depth"
            / "composition"
            / "proposals"
            / again["proposal_id"]
        )
        painted = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{again['proposal_id']}/strokes",
            json={"mode": "add", "brush_radius_px": 0.6, "points": [{"x": 3.5, "y": 1.5}]},
        )
        assert painted.status_code == 200, painted.text
        edited = (folder / "working_mask.png").read_bytes()
        stored = client.post(
            f"/api/projects/{project['id']}/relief/regions/{region['id']}/mask-edit/save",
            json={"proposal_id": again["proposal_id"], "name": "Bride", "role": "secondary"},
        )
        assert stored.status_code == 200, stored.text
        body = stored.json()
        assert len(body["relief"]["regions"]) == 1
        updated = body["relief"]["regions"][0]
        assert updated["id"] == region["id"]
        assert updated["order"] == region["order"]
        assert updated["elevation"] == region["elevation"]
        assert updated["depth_range"] == region["depth_range"]
        assert updated["feather"] == region["feather"]
        assert updated["name"] == "Bride"
        assert updated["role"] == "secondary"
        assert updated["mask_filename"] == region["mask_filename"]
        assert paths["mask"].read_bytes() == edited
        assert paths["mask"].read_bytes() != permanent
        assert _sha256(paths["relief"]) == relief_hash
        assert body["relief"]["generated_at"] == saved["relief"]["generated_at"]
        assert not folder.exists()
        assert (project_root / "sample-relief" / "depth" / "composition" / "proposals" / groom.json()["proposal_id"]).is_dir()


def test_failed_save_restores_the_previous_mask(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        region = _accept(client, project["id"], segmentation)["relief"]["regions"][0]
        paths = _paths(project_root, region["id"])
        permanent = paths["mask"].read_bytes()
        proposal = _open(client, project["id"], region["id"])
        painted = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal['proposal_id']}/strokes",
            json={"mode": "add", "brush_radius_px": 0.6, "points": [{"x": 3.5, "y": 1.5}]},
        )
        assert painted.status_code == 200, painted.text

        def fail_manifest(self, project_record, record):
            raise RuntimeError("disk full")

        monkeypatch.setattr(ProjectStore, "save_relief", fail_manifest)
        failed = client.post(
            f"/api/projects/{project['id']}/relief/regions/{region['id']}/mask-edit/save",
            json={"proposal_id": proposal["proposal_id"], "name": "Subject", "role": "primary"},
        )
        assert failed.status_code == 500, failed.text
        assert paths["mask"].read_bytes() == permanent
        stored = client.get(f"/api/projects/{project['id']}").json()
        assert len(stored["relief"]["regions"]) == 1
        assert stored["relief"]["regions"][0]["id"] == region["id"]
        folder = (
            project_root
            / "sample-relief"
            / "depth"
            / "composition"
            / "proposals"
            / proposal["proposal_id"]
        )
        assert folder.is_dir()


def test_new_region_accept_and_ai_reset_stay_unchanged(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        first = _accept(client, project["id"], segmentation)
        predicted = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 2, "y": 1}], "name": "Guest", "role": "secondary"},
        )
        assert predicted.status_code == 200, predicted.text
        proposal_id = predicted.json()["proposal_id"]
        assert predicted.json()["origin_type"] == "new_region"
        folder = project_root / "sample-relief" / "depth" / "composition" / "proposals" / proposal_id
        ai = (folder / "ai_mask.png").read_bytes()
        client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "add", "brush_radius_px": 0.6, "points": [{"x": 0.5, "y": 0.5}]},
        )
        reset = client.post(f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/reset")
        assert reset.status_code == 200, reset.text
        assert (folder / "working_mask.png").read_bytes() == ai
        assert (folder / "session_base_mask.png").read_bytes() == ai
        refused = client.post(
            f"/api/projects/{project['id']}/relief/regions/{first['relief']['regions'][0]['id']}/mask-edit/save",
            json={"proposal_id": proposal_id, "name": "Guest", "role": "secondary"},
        )
        assert refused.status_code == 422, refused.text
        accepted = client.post(
            f"/api/projects/{project['id']}/relief/regions",
            json={"name": "Guest", "role": "secondary", "proposal_id": proposal_id},
        )
        assert accepted.status_code == 200, accepted.text
        regions = accepted.json()["relief"]["regions"]
        assert len(regions) == 2
        assert regions[0]["id"] == first["relief"]["regions"][0]["id"]
        assert regions[1]["name"] == "Guest"
