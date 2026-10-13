"""Magic Wand selection. These tests use synthetic images and a fake SAM model."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.segmentation.magic_wand import MagicWandError, paint_selection, select_similar
from tests.test_projects import served
from tests.test_relief import FakeSegmentation, _bind, _conditioned


def _rgb(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def test_magic_wand_is_deterministic_and_matches_the_source_shape() -> None:
    image = _rgb(6, 8, (20, 40, 200))
    image[1:4, 2:5] = (210, 40, 40)
    first = select_similar(image, 3.2, 2.4, 18, contiguous=True)
    second = select_similar(image, 3.2, 2.4, 18, contiguous=True)
    assert first.shape == (6, 8)
    assert first.dtype == np.bool_
    assert np.array_equal(first, second)


def test_magic_wand_selects_a_contiguous_similar_region_only() -> None:
    image = _rgb(5, 7, (20, 30, 220))
    image[1:4, 1:4] = (200, 30, 30)
    image[1:3, 5:7] = (198, 32, 28)
    selected = select_similar(image, 2.5, 2.5, 18, contiguous=True)
    assert selected[2, 2]
    assert selected[1:4, 1:4].all()
    assert not selected[1:3, 5:7].any()
    assert not selected[0, 0]


def test_magic_wand_rejects_a_dissimilar_neighbor() -> None:
    image = _rgb(3, 3, (20, 20, 220))
    image[1, 1] = (220, 20, 20)
    selected = select_similar(image, 1.5, 1.5, 18, contiguous=True)
    assert selected[1, 1]
    assert int(selected.sum()) == 1


def test_magic_wand_uses_8_connectivity() -> None:
    image = _rgb(3, 3, (10, 10, 180))
    image[0, 0] = (220, 20, 20)
    image[1, 1] = (218, 22, 18)
    image[2, 2] = (216, 24, 16)
    selected = select_similar(image, 0.2, 0.2, 18, contiguous=True)
    assert selected[0, 0]
    assert selected[1, 1]
    assert selected[2, 2]
    assert int(selected.sum()) == 3


def test_magic_wand_add_and_subtract_keep_a_binary_mask_and_the_inputs() -> None:
    image = _rgb(4, 4, (30, 30, 30))
    image[0:2, 0:2] = (240, 240, 240)
    before = image.copy()
    selected = select_similar(image, 0.4, 0.4, 12, contiguous=True)
    mask = np.zeros((4, 4), dtype=np.uint8)
    added = paint_selection(mask, selected, "add")
    assert added[0, 0] == 255
    assert added[3, 3] == 0
    assert set(np.unique(added)).issubset({0, 255})
    assert np.array_equal(mask, np.zeros_like(mask))
    subtracted = paint_selection(added, selected, "subtract")
    assert subtracted[0, 0] == 0
    assert np.array_equal(image, before)
    assert np.array_equal(mask, np.zeros_like(mask))


def test_magic_wand_accepts_a_boundary_seed() -> None:
    image = _rgb(3, 5, (80, 90, 100))
    image[2, 4] = (240, 240, 240)
    selected = select_similar(image, 4.9, 2.9, 8, contiguous=True)
    assert selected[2, 4]
    assert int(selected.sum()) == 1
    corner = select_similar(image, -4, -8, 8, contiguous=True)
    assert corner[0, 0]


@pytest.mark.parametrize(
    "tolerance",
    [-0.1, 100.1, float("nan"), float("inf")],
)
def test_magic_wand_rejects_invalid_tolerance(tolerance: float) -> None:
    image = _rgb(2, 2, (10, 10, 10))
    with pytest.raises(MagicWandError):
        select_similar(image, 0, 0, tolerance, contiguous=True)


@pytest.mark.parametrize("x,y", [(float("nan"), 0), (0, float("inf")), (True, 0)])
def test_magic_wand_rejects_nonfinite_coordinates(x: float, y: float) -> None:
    image = _rgb(2, 2, (10, 10, 10))
    with pytest.raises(MagicWandError):
        select_similar(image, x, y, 10, contiguous=True)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proposal_paths(project_root: Path, proposal_id: str) -> tuple[Path, Path]:
    folder = project_root / "sample-relief" / "depth" / "composition" / "proposals" / proposal_id
    return folder / "ai_mask.png", folder / "working_mask.png"


def test_magic_wand_api_changes_only_the_working_mask(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        predicted = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 1, "y": 1}]},
        )
        assert predicted.status_code == 200, predicted.text
        proposal_id = predicted.json()["proposal_id"]
        segmentation.predicts = 0
        ai_mask, working_mask = _proposal_paths(project_root, proposal_id)
        ai_before = _sha256(ai_mask)
        working_before = working_mask.read_bytes()

        added = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/magic-wand",
            json={"mode": "add", "x": 0.5, "y": 0.5, "tolerance": 100, "contiguous": True},
        )
        assert added.status_code == 200, added.text
        assert working_mask.read_bytes() != working_before
        with Image.open(working_mask) as handle:
            values = np.array(handle.convert("L"))
        assert values[0, 0] == 255
        assert set(np.unique(values)).issubset({0, 255})
        assert _sha256(ai_mask) == ai_before

        removed = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/magic-wand",
            json={"mode": "subtract", "x": 0.5, "y": 0.5, "tolerance": 100, "contiguous": True},
        )
        assert removed.status_code == 200, removed.text
        with Image.open(working_mask) as handle:
            erased = np.array(handle.convert("L"))
        assert erased[0, 0] == 0
        assert _sha256(ai_mask) == ai_before
        assert source.read_bytes() == source_before
        assert raw.read_bytes() == raw_before
        assert conditioned.read_bytes() == conditioned_before
        assert segmentation.predicts == 0

        client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/refinement-session",
        )
        baseline = working_mask.read_bytes()
        again = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/magic-wand",
            json={"mode": "add", "x": 0.5, "y": 0.5, "tolerance": 100, "contiguous": True},
        )
        assert again.status_code == 200, again.text
        painted = working_mask.read_bytes()
        restored = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/refinement-session/restore",
        )
        assert restored.status_code == 200, restored.text
        assert working_mask.read_bytes() == baseline
        replayed = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/magic-wand",
            json={"mode": "add", "x": 0.5, "y": 0.5, "tolerance": 100, "contiguous": True},
        )
        assert replayed.status_code == 200, replayed.text
        assert working_mask.read_bytes() == painted
        assert _sha256(ai_mask) == ai_before
        assert segmentation.predicts == 0


def test_magic_wand_api_rejects_a_missing_proposal_and_a_bad_payload(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, FakeSegmentation())
        missing = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/00000000-0000-0000-0000-000000000099/magic-wand",
            json={"mode": "add", "x": 0, "y": 0, "tolerance": 10, "contiguous": True},
        )
        assert missing.status_code == 404
        predicted = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 1, "y": 1}]},
        )
        proposal_id = predicted.json()["proposal_id"]
        invalid = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/magic-wand",
            json={"mode": "add", "x": 0, "y": 0, "tolerance": 140, "contiguous": True},
        )
        assert invalid.status_code == 422
        nonfinite = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/magic-wand",
            json={"mode": "add", "x": "NaN", "y": 0, "tolerance": 10, "contiguous": True},
        )
        assert nonfinite.status_code == 422


def test_magic_wand_processing_failure_is_a_controlled_error(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("wand failed")

    monkeypatch.setattr("app.segmentation.service.select_similar", explode)
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, FakeSegmentation())
        predicted = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 1, "y": 1}]},
        )
        proposal_id = predicted.json()["proposal_id"]
        ai_mask, working_mask = _proposal_paths(project_root, proposal_id)
        before = working_mask.read_bytes()
        ai_before = ai_mask.read_bytes()
        failed = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal_id}/magic-wand",
            json={"mode": "add", "x": 1, "y": 1, "tolerance": 10, "contiguous": True},
        )
        assert failed.status_code == 500
        assert failed.json()["detail"] == "Magic Wand failed."
        assert working_mask.read_bytes() == before
        assert ai_mask.read_bytes() == ai_before
