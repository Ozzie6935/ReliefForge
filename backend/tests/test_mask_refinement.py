"""Proposal refinement API. Segmentation uses a fake model and does not load SAM."""

import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from app.relief.artifacts import write_mask_png
from app.segmentation.refinement import apply_stroke
from tests.test_projects import served
from tests.test_relief import FakeSegmentation, _bind, _conditioned


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proposal(project_root, proposal_id: str):
    folder = project_root / "sample-relief" / "depth" / "composition" / "proposals" / proposal_id
    return folder / "ai_mask.png", folder / "working_mask.png"


def _predict(client, project_id: str):
    predicted = client.post(
        f"/api/projects/{project_id}/segmentation/predict",
        json={"positive_points": [{"x": 1, "y": 1}]},
    )
    assert predicted.status_code == 200, predicted.text
    return predicted.json()["proposal_id"]


def test_proposal_starts_with_matching_ai_and_working_masks(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        source = project_root / "sample-relief" / "source" / "original.png"
        raw = project_root / "sample-relief" / "depth" / "raw_depth.npy"
        conditioned = project_root / "sample-relief" / "depth" / "processing" / "conditioned_depth.npy"
        source_before = source.read_bytes()
        raw_before = raw.read_bytes()
        conditioned_before = conditioned.read_bytes()
        proposal_id = _predict(client, project["id"])
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        assert ai_mask.is_file()
        assert working_mask.is_file()
        assert ai_mask.read_bytes() == working_mask.read_bytes()
        assert segmentation.predicts == 1

        added = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "add", "brush_radius_px": 1.2, "points": [{"x": 4.5, "y": 2.5}]},
        )
        assert added.status_code == 200, added.text
        assert ai_mask.read_bytes() != working_mask.read_bytes()
        ai_hash = _sha256(ai_mask)
        with Image.open(working_mask) as handle:
            working_values = np.array(handle.convert("L"))
        assert working_values[2, 4] == 255
        assert set(np.unique(working_values)).issubset({0, 255})

        erased = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "erase", "brush_radius_px": 1.2, "points": [{"x": 4.5, "y": 2.5}]},
        )
        assert erased.status_code == 200, erased.text
        with Image.open(working_mask) as handle:
            erased_values = np.array(handle.convert("L"))
        assert erased_values[2, 4] == 0
        assert _sha256(ai_mask) == ai_hash
        assert segmentation.predicts == 1

        reset = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/reset"
        )
        assert reset.status_code == 200, reset.text
        assert working_mask.read_bytes() == ai_mask.read_bytes()
        assert _sha256(ai_mask) == ai_hash
        assert segmentation.predicts == 1
        assert source.read_bytes() == source_before
        assert raw.read_bytes() == raw_before
        assert conditioned.read_bytes() == conditioned_before


def test_accept_uses_working_mask_and_cancel_removes_the_proposal(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        painted = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "add", "brush_radius_px": 1.2, "points": [{"x": 4.5, "y": 2.5}]},
        )
        assert painted.status_code == 200, painted.text
        accepted = client.post(
            f"/api/projects/{project['id']}/relief/regions",
            json={"name": "Subject", "role": "primary", "proposal_id": proposal_id},
        )
        assert accepted.status_code == 200, accepted.text
        region = accepted.json()["relief"]["regions"][0]
        mask = client.get(f"/api/projects/{project['id']}/relief/regions/{region['id']}/mask")
        assert mask.status_code == 200
        with Image.open(io.BytesIO(mask.content)) as handle:
            values = np.array(handle.convert("L"))
        assert values[2, 4] == 255
        assert values[1, 1] == 255
        proposal_dir = project_root / "sample-relief" / "depth" / "composition" / "proposals" / proposal_id
        assert not proposal_dir.exists()

        again = _predict(client, project["id"])
        missing = client.delete(f"/api/projects/{project['id']}/segmentation/proposals/{uuid4()}")
        assert missing.status_code == 404
        removed = client.delete(f"/api/projects/{project['id']}/segmentation/proposals/{again}")
        assert removed.status_code == 204
        assert not (project_root / "sample-relief" / "depth" / "composition" / "proposals" / again).exists()
        preview = client.get(f"/api/projects/{project['id']}/segmentation/proposals/{again}")
        assert preview.status_code == 404


def test_invalid_stroke_and_refinement_failure_do_not_call_the_model(project_root, monkeypatch) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        predicts = segmentation.predicts
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        working_before = working_mask.read_bytes()
        invalid = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "add", "brush_radius_px": 0, "points": [{"x": 1, "y": 1}]},
        )
        assert invalid.status_code == 422
        huge = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "add", "brush_radius_px": 100, "points": [{"x": 1, "y": 1}]},
        )
        assert huge.status_code == 422
        assert huge.json()["detail"] == "The mask stroke is invalid."
        empty = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "add", "brush_radius_px": 1, "points": []},
        )
        assert empty.status_code == 422
        absent = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{uuid4()}/strokes",
            json={"mode": "erase", "brush_radius_px": 1, "points": [{"x": 1, "y": 1}]},
        )
        assert absent.status_code == 404

        def explode(mask, stroke):
            raise RuntimeError("refinement failed")

        monkeypatch.setattr("app.segmentation.service.apply_stroke", explode)
        failed = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/strokes",
            json={"mode": "add", "brush_radius_px": 1, "points": [{"x": 1, "y": 1}]},
        )
        assert failed.status_code == 500
        assert failed.json()["detail"] == "Mask refinement failed."
        assert segmentation.predicts == predicts
        assert ai_mask.read_bytes() == ai_before
        assert working_mask.read_bytes() == working_before


def test_legacy_proposal_file_migrates_without_changing_bytes(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = uuid4()
        legacy = (
            project_root
            / "sample-relief"
            / "depth"
            / "composition"
            / "proposals"
            / f"{proposal_id}.png"
        )
        mask = np.zeros((4, 6), dtype=bool)
        mask[1, 2] = True
        write_mask_png(legacy, mask)
        original = legacy.read_bytes()
        current = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert current.status_code == 200, current.text
        assert current.json()["proposal_id"] == str(proposal_id)
        ai_mask, working_mask = _proposal(project_root, str(proposal_id))
        assert ai_mask.read_bytes() == original
        assert working_mask.read_bytes() == original
        assert not legacy.exists()
        preview = client.get(f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}")
        assert preview.status_code == 200
        assert preview.content == original


def test_proposal_draft_survives_reload_reset_and_legacy_files(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        created = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={
                "positive_points": [{"x": 1, "y": 1}],
                "name": "Bride",
                "role": "primary",
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        proposal_id = body["proposal_id"]
        assert body["name"] == "Bride"
        assert body["role"] == "primary"
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        working_before = working_mask.read_bytes()
        metadata_path = ai_mask.parent / "proposal.json"
        stored = metadata_path.read_text(encoding="utf-8")
        assert '"name": "Bride"' in stored
        assert '"role": "primary"' in stored

        reloaded = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert reloaded.status_code == 200, reloaded.text
        assert reloaded.json()["name"] == "Bride"
        assert reloaded.json()["role"] == "primary"

        renamed = client.put(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/draft",
            json={"name": "Veil", "role": "secondary"},
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["name"] == "Veil"
        assert renamed.json()["role"] == "secondary"
        again = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert again.json()["name"] == "Veil"
        assert again.json()["role"] == "secondary"
        first_created = json.loads(stored)["created_at"]
        assert json.loads(metadata_path.read_text(encoding="utf-8"))["created_at"] == first_created
        assert ai_mask.read_bytes() == ai_before
        assert working_mask.read_bytes() == working_before

        reset = client.post(f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/reset")
        assert reset.status_code == 200, reset.text
        assert reset.json()["name"] == "Veil"
        assert reset.json()["role"] == "secondary"
        assert ai_mask.read_bytes() == working_mask.read_bytes() == ai_before

        invalid = client.put(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/draft",
            json={"name": "Veil", "role": "nope"},
        )
        assert invalid.status_code == 422

        legacy_id = uuid4()
        legacy = (
            project_root
            / "sample-relief"
            / "depth"
            / "composition"
            / "proposals"
            / f"{legacy_id}.png"
        )
        mask = np.zeros((4, 6), dtype=bool)
        mask[0, 0] = True
        # A second proposal is not current while the first still exists.
        removed = client.delete(f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}")
        assert removed.status_code == 204
        write_mask_png(legacy, mask)
        current = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert current.status_code == 200, current.text
        assert current.json()["proposal_id"] == str(legacy_id)
        assert current.json()["name"] == ""
        assert current.json()["role"] is None
        migrated = _proposal(project_root, str(legacy_id))
        assert not (migrated[0].parent / "proposal.json").exists()


def _paint(client, project_id: str, proposal_id: str, x: float, y: float) -> None:
    painted = client.post(
        f"/api/projects/{project_id}/segmentation/proposals/{proposal_id}/strokes",
        json={"mode": "add", "brush_radius_px": 0.6, "points": [{"x": x, "y": y}]},
    )
    assert painted.status_code == 200, painted.text


def _begin_session(client, project_id: str, proposal_id: str) -> None:
    started = client.post(
        f"/api/projects/{project_id}/segmentation/proposals/{proposal_id}/refinement-session"
    )
    assert started.status_code == 200, started.text


def _restore_session(client, project_id: str, proposal_id: str) -> None:
    restored = client.post(
        f"/api/projects/{project_id}/segmentation/proposals/{proposal_id}/refinement-session/restore"
    )
    assert restored.status_code == 200, restored.text


def test_session_baseline_preserves_refinement_across_a_fresh_undo(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        _paint(client, project["id"], proposal_id, 3.5, 1.5)
        after_a = working_mask.read_bytes()
        assert after_a != ai_before
        _begin_session(client, project["id"], proposal_id)
        baseline = ai_mask.parent / "session_base_mask.png"
        assert baseline.read_bytes() == after_a
        assert working_mask.read_bytes() == after_a
        _paint(client, project["id"], proposal_id, 4.5, 2.5)
        assert working_mask.read_bytes() != after_a
        _restore_session(client, project["id"], proposal_id)
        assert working_mask.read_bytes() == after_a
        assert working_mask.read_bytes() != ai_before
        assert ai_mask.read_bytes() == ai_before
        assert baseline.read_bytes() == after_a


def test_session_undo_replays_only_current_strokes(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        _paint(client, project["id"], proposal_id, 3.5, 1.5)
        after_a = working_mask.read_bytes()
        _begin_session(client, project["id"], proposal_id)
        _paint(client, project["id"], proposal_id, 4.5, 2.5)
        after_b = working_mask.read_bytes()
        _paint(client, project["id"], proposal_id, 0.5, 0.5)
        after_c = working_mask.read_bytes()
        assert len({after_a, after_b, after_c, ai_before}) == 4
        _restore_session(client, project["id"], proposal_id)
        _paint(client, project["id"], proposal_id, 4.5, 2.5)
        assert working_mask.read_bytes() == after_b
        _restore_session(client, project["id"], proposal_id)
        assert working_mask.read_bytes() == after_a
        assert working_mask.read_bytes() != ai_before
        assert ai_mask.read_bytes() == ai_before


def test_session_redo_replays_the_undone_stroke(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        _paint(client, project["id"], proposal_id, 3.5, 1.5)
        _begin_session(client, project["id"], proposal_id)
        _paint(client, project["id"], proposal_id, 4.5, 2.5)
        _paint(client, project["id"], proposal_id, 0.5, 0.5)
        after_c = working_mask.read_bytes()
        _restore_session(client, project["id"], proposal_id)
        _paint(client, project["id"], proposal_id, 4.5, 2.5)
        assert working_mask.read_bytes() != after_c
        _paint(client, project["id"], proposal_id, 0.5, 0.5)
        assert working_mask.read_bytes() == after_c
        assert ai_mask.read_bytes() == ai_before


def test_new_session_undo_keeps_refinement_from_before_the_session(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        _paint(client, project["id"], proposal_id, 3.5, 1.5)
        refined = working_mask.read_bytes()
        _begin_session(client, project["id"], proposal_id)
        _paint(client, project["id"], proposal_id, 4.5, 2.5)
        _restore_session(client, project["id"], proposal_id)
        assert working_mask.read_bytes() == refined
        assert working_mask.read_bytes() != ai_before
        assert ai_mask.read_bytes() == ai_before


def test_reset_to_ai_replaces_working_mask_and_session_baseline(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        _paint(client, project["id"], proposal_id, 3.5, 1.5)
        assert working_mask.read_bytes() != ai_before
        _begin_session(client, project["id"], proposal_id)
        reset = client.post(f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/reset")
        assert reset.status_code == 200, reset.text
        baseline = ai_mask.parent / "session_base_mask.png"
        assert working_mask.read_bytes() == ai_before
        assert baseline.read_bytes() == ai_before
        assert ai_mask.read_bytes() == ai_before


def test_restore_without_a_session_does_not_change_masks(project_root) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        _paint(client, project["id"], proposal_id, 3.5, 1.5)
        ai_before = ai_mask.read_bytes()
        working_before = working_mask.read_bytes()
        missing = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/refinement-session/restore"
        )
        assert missing.status_code == 409
        assert ai_mask.read_bytes() == ai_before
        assert working_mask.read_bytes() == working_before


def test_apply_stroke_import_is_the_pure_function() -> None:
    assert apply_stroke.__module__ == "app.segmentation.refinement"


def test_resume_keeps_the_existing_draft_and_its_masks(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        proposal_id = _predict(client, project["id"])
        named = client.put(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/draft",
            json={"name": "Groom", "role": "primary"},
        )
        assert named.status_code == 200, named.text
        ai_mask, working_mask = _proposal(project_root, proposal_id)
        ai_before = ai_mask.read_bytes()
        working_before = working_mask.read_bytes()
        predicts_before = segmentation.predicts
        current = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert current.status_code == 200, current.text
        assert current.json()["proposal_id"] == proposal_id
        assert current.json()["name"] == "Groom"
        assert current.json()["role"] == "primary"
        assert current.json()["origin_type"] == "new_region"
        _begin_session(client, project["id"], proposal_id)
        assert ai_mask.read_bytes() == ai_before
        assert working_mask.read_bytes() == working_before
        assert segmentation.predicts == predicts_before
        again = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert again.json()["proposal_id"] == proposal_id
        folders = [
            child.name
            for child in (project_root / "sample-relief" / "depth" / "composition" / "proposals").iterdir()
        ]
        assert folders == [proposal_id]


def test_accepting_a_draft_follows_the_existing_region_and_removes_the_draft(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        bride_id = _predict(client, project["id"])
        bride = client.post(
            f"/api/projects/{project['id']}/relief/regions",
            json={"name": "Bride", "role": "primary", "proposal_id": bride_id},
        )
        assert bride.status_code == 200, bride.text
        groom_id = _predict(client, project["id"])
        assert groom_id != bride_id
        groom = client.post(
            f"/api/projects/{project['id']}/relief/regions",
            json={"name": "Groom", "role": "primary", "proposal_id": groom_id},
        )
        assert groom.status_code == 200, groom.text
        regions = groom.json()["relief"]["regions"]
        assert [(region["name"], region["order"]) for region in regions] == [("Bride", 0), ("Groom", 1)]
        assert len({region["id"] for region in regions}) == 2
        missing = client.get(f"/api/projects/{project['id']}/segmentation/proposal")
        assert missing.status_code == 404
