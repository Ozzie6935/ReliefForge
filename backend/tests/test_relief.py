"""Relief composition and segmentation tests.

Segmentation uses a fake model. These tests do not load SAM 2 weights and
do not write into the repository projects/ folder.
"""

from pathlib import Path

import io

import numpy as np
import pytest
from PIL import Image

from app.api.deps import get_depth_service, get_relief_service, get_segmentation_service
from app.core.config import MAX_SOURCE_IMAGE_BYTES
from app.depth.service import DepthService
from app.main import app
from app.projects.errors import SegmentationModelUnavailable
from app.projects.storage import ProjectStore
from app.relief.composition import CompositionError, RegionContribution, compose_relief, feather_weights
from app.relief.service import ReliefService
from app.segmentation.model import SegmentationPrompt
from app.segmentation.registry import MODEL_ID, get_segmentation_model, reset_segmentation_model_cache
from app.segmentation.sam2 import SAM2SegmentationModel
from app.segmentation.service import SegmentationService
from tests.test_depth import FakeDepthModel
from tests.test_projects import create_project, image_bytes, served


def _depth() -> np.ndarray:
    return np.linspace(0.2, 0.8, 16, dtype=np.float32).reshape(4, 4)


def _region(mask: np.ndarray, **kwargs: object) -> RegionContribution:
    values = {
        "region_id": "a",
        "mask": mask,
        "elevation": 0.0,
        "depth_range": 1.0,
        "feather": 0.0,
        "order": 0,
    }
    values.update(kwargs)
    return RegionContribution(**values)


def test_zero_regions_match_conditioned_depth() -> None:
    depth = _depth()
    original = depth.copy()
    result = compose_relief(depth, [])
    assert np.array_equal(result, original)
    assert np.array_equal(depth, original)
    assert result.dtype == np.float32


def test_identity_parameters_preserve_the_region() -> None:
    depth = _depth()
    mask = np.zeros(depth.shape, dtype=bool)
    mask[1:3, 1:3] = True
    result = compose_relief(depth, [_region(mask)])
    assert np.array_equal(result, depth)


def test_elevation_moves_the_region_forward_or_back() -> None:
    depth = np.full((4, 4), 0.4, dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[0, 0] = True
    forward = compose_relief(depth, [_region(mask, elevation=0.2)])
    backward = compose_relief(depth, [_region(mask, elevation=-0.2)])
    assert forward[0, 0] == np.float32(0.6)
    assert backward[0, 0] == np.float32(0.2)
    assert forward[1, 1] == np.float32(0.4)
    assert np.array_equal(depth, np.full((4, 4), 0.4, dtype=np.float32))


def test_depth_range_compresses_or_preserves_local_variation() -> None:
    depth = np.array([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32)
    mask = np.ones(depth.shape, dtype=bool)
    flat = compose_relief(depth, [_region(mask, depth_range=0.0)])
    kept = compose_relief(depth, [_region(mask, depth_range=1.0)])
    assert np.allclose(flat, 0.5)
    assert np.array_equal(kept, depth)


def test_feather_zero_is_hard_and_positive_feather_blends(monkeypatch: pytest.MonkeyPatch) -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    def rejected(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("feather was not bypassed")

    monkeypatch.setitem(
        __import__("sys").modules,
        "cv2",
        type("Cv2", (), {"GaussianBlur": staticmethod(rejected)}),
    )
    hard = feather_weights(mask, 0.0)
    assert np.array_equal(hard, mask.astype(np.float32))


def test_positive_feather_blends_the_mask_edge() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True
    soft = feather_weights(mask, 8.0)
    assert soft.dtype == np.float32
    assert soft[6, 6] > soft[4, 6] > 0.0
    assert soft[3, 6] > 0.0
    assert soft[0, 0] < soft[6, 6]
    assert soft.min() >= 0.0
    assert soft.max() <= 1.0


def test_output_is_finite_float32_and_deterministic() -> None:
    depth = _depth()
    mask = np.ones(depth.shape, dtype=bool)
    regions = [_region(mask, elevation=0.3, depth_range=1.5, feather=0.0)]
    first = compose_relief(depth, regions)
    second = compose_relief(depth, regions)
    assert first.dtype == np.float32
    assert first.shape == depth.shape
    assert np.isfinite(first).all()
    assert first.min() >= 0.0
    assert first.max() <= 1.0
    assert np.array_equal(first, second)


def test_overlap_follows_region_order() -> None:
    depth = np.full((2, 2), 0.2, dtype=np.float32)
    mask = np.ones(depth.shape, dtype=bool)
    early = _region(mask, region_id="early", elevation=0.1, order=0)
    late = _region(mask, region_id="late", elevation=0.5, order=1)
    forward = compose_relief(depth, [late, early])
    assert np.allclose(forward, 0.7)
    swapped = compose_relief(depth, [_region(mask, region_id="late", elevation=0.1, order=1), _region(mask, region_id="early", elevation=0.5, order=0)])
    assert np.allclose(swapped, 0.3)


def test_mask_shape_mismatch_is_rejected() -> None:
    depth = _depth()
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(CompositionError):
        compose_relief(depth, [_region(mask)])


def test_registry_reuses_one_model_and_rejects_unknown_ids() -> None:
    reset_segmentation_model_cache()
    first = get_segmentation_model(MODEL_ID)
    second = get_segmentation_model(MODEL_ID)
    assert first is second
    assert first.model_id == "sam2.1-hiera-tiny"
    with pytest.raises(SegmentationModelUnavailable):
        get_segmentation_model("other-model")
    reset_segmentation_model_cache()


def test_missing_checkpoint_is_reported_before_building(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.segmentation.sam2.sam2_tiny_checkpoint",
        lambda: tmp_path / "missing.pt",
    )
    model = SAM2SegmentationModel()
    with pytest.raises(SegmentationModelUnavailable):
        model.load()


class FakeSegmentation:
    model_id = "sam2.1-hiera-tiny"
    model_version = "test"

    def __init__(self) -> None:
        self.loads = 0
        self.predicts = 0
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self.loads += 1

    def predict(self, image: Image.Image, prompt: SegmentationPrompt, cache_key: str | None = None) -> np.ndarray:
        self.load()
        self.predicts += 1
        mask = np.zeros((image.height, image.width), dtype=bool)
        if prompt.positive_points:
            x, y = prompt.positive_points[0]
            mask[int(y), int(x)] = True
        if prompt.box is not None:
            x0, y0, x1, y1 = prompt.box
            mask[int(y0) : int(y1), int(x0) : int(x1)] = True
        for x, y in prompt.negative_points:
            mask[int(y), int(x)] = False
        return mask


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def _bind(project_root: Path, segmentation: FakeSegmentation) -> None:
    store = ProjectStore(project_root)
    app.dependency_overrides[get_depth_service] = lambda: DepthService(store, FakeDepthModel())
    app.dependency_overrides[get_segmentation_service] = lambda: SegmentationService(store, segmentation)
    app.dependency_overrides[get_relief_service] = lambda: ReliefService(store)


def _conditioned(project_root: Path, client) -> dict:
    _bind(project_root, FakeSegmentation())
    created = create_project(client)
    upload = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("scan.png", image_bytes("PNG", (6, 4)), "image/png")},
    )
    assert upload.status_code == 200
    generated = client.post(f"/api/projects/{created['id']}/depth")
    assert generated.status_code == 200, generated.text
    conditioned = client.post(f"/api/projects/{created['id']}/depth/conditioning", json={})
    assert conditioned.status_code == 200, conditioned.text
    return conditioned.json()


def test_predict_validates_points_and_does_not_persist_a_region(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        missing = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 99, "y": 1}]},
        )
        assert missing.status_code == 422
        empty = client.post(f"/api/projects/{project['id']}/segmentation/predict", json={})
        assert empty.status_code == 422
        predicted = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 1, "y": 2}], "negative_points": [{"x": 0, "y": 0}]},
        )
        assert predicted.status_code == 200, predicted.text
        proposal = predicted.json()
        stored = client.get(f"/api/projects/{project['id']}").json()
        assert stored["relief"] is None
        preview = client.get(
            f"/api/projects/{project['id']}/segmentation/proposals/{proposal['proposal_id']}"
        )
        assert preview.status_code == 200
        assert preview.content.startswith(b"\x89PNG")
        again = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 2, "y": 2}]},
        )
        assert again.status_code == 200
        assert segmentation.loads == 1
        assert segmentation.predicts == 2


def test_accept_delete_and_recompose_from_conditioned_depth(project_root: Path) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project = _conditioned(project_root, client)
        _bind(project_root, segmentation)
        raw_path = project_root / "sample-relief" / "depth" / "raw_depth.npy"
        conditioned_path = project_root / "sample-relief" / "depth" / "processing" / "conditioned_depth.npy"
        raw_before = raw_path.read_bytes()
        conditioned_before = conditioned_path.read_bytes()
        bare = client.post(
            f"/api/projects/{project['id']}/relief/regions",
            json={"name": "Subject", "role": "primary", "proposal_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert bare.status_code == 404
        predicted = client.post(
            f"/api/projects/{project['id']}/segmentation/predict",
            json={"positive_points": [{"x": 1, "y": 1}], "box": {"x0": 0, "y0": 0, "x1": 3, "y1": 2}},
        )
        assert predicted.status_code == 200, predicted.text
        proposal_id = predicted.json()["proposal_id"]
        accepted = client.post(
            f"/api/projects/{project['id']}/relief/regions",
            json={"name": "Subject", "role": "primary", "proposal_id": proposal_id},
        )
        assert accepted.status_code == 200, accepted.text
        body = accepted.json()
        region = body["relief"]["regions"][0]
        assert region["name"] == "Subject"
        assert region["role"] == "primary"
        assert region["elevation"] == 0.0
        assert region["depth_range"] == 1.0
        assert region["order"] == 0
        mask = client.get(f"/api/projects/{project['id']}/relief/regions/{region['id']}/mask")
        assert mask.status_code == 200
        with Image.open(io.BytesIO(mask.content)) as handle:
            values = np.array(handle.convert("L"))
        assert values.shape == (4, 6)
        relief_path = project_root / "sample-relief" / "depth" / "composition" / "relief_depth.npy"
        relief = np.load(relief_path)
        conditioned = np.load(conditioned_path)
        assert relief.dtype == np.float32
        assert relief.shape == conditioned.shape
        assert relief.min() >= 0.0 and relief.max() <= 1.0
        identity = compose_relief(conditioned, [])
        assert np.array_equal(relief, identity)
        preview = client.get(f"/api/projects/{project['id']}/relief/preview")
        assert preview.status_code == 200
        moved = client.post(
            f"/api/projects/{project['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": "Subject",
                        "role": "secondary",
                        "elevation": 0.25,
                        "depth_range": 1.0,
                        "feather": 0.0,
                        "order": 0,
                    }
                ]
            },
        )
        assert moved.status_code == 200, moved.text
        moved_relief = np.load(relief_path)
        mask_array = values >= 128
        expected = compose_relief(
            conditioned,
            [
                RegionContribution(
                    region_id=region["id"],
                    mask=mask_array,
                    elevation=0.25,
                    depth_range=1.0,
                    feather=0.0,
                    order=0,
                )
            ],
        )
        assert np.array_equal(moved_relief, expected)
        restored = client.post(
            f"/api/projects/{project['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": "Subject",
                        "role": "secondary",
                        "elevation": 0.0,
                        "depth_range": 1.0,
                        "feather": 0.0,
                        "order": 0,
                    }
                ]
            },
        )
        assert restored.status_code == 200, restored.text
        restored_relief = np.load(relief_path)
        assert np.array_equal(restored_relief, conditioned)
        reapplied = compose_relief(
            moved_relief,
            [
                RegionContribution(
                    region_id=region["id"],
                    mask=mask_array,
                    elevation=0.0,
                    depth_range=1.0,
                    feather=0.0,
                    order=0,
                )
            ],
        )
        assert not np.array_equal(restored_relief, reapplied)
        removed = client.delete(f"/api/projects/{project['id']}/relief/regions/{region['id']}")
        assert removed.status_code == 200, removed.text
        assert removed.json()["relief"]["regions"] == []
        assert np.array_equal(np.load(relief_path), conditioned)
        assert not (project_root / "sample-relief" / "depth" / "composition" / "masks" / f"{region['id']}.png").exists()
        assert raw_path.read_bytes() == raw_before
        assert conditioned_path.read_bytes() == conditioned_before


def test_conditioning_is_required_before_segmentation(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _bind(project_root, FakeSegmentation())
        created = create_project(client)
        response = client.post(
            f"/api/projects/{created['id']}/segmentation/predict",
            json={"positive_points": [{"x": 1, "y": 1}]},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "Conditioned depth has not been created."
