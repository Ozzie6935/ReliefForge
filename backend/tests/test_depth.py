"""Depth generation tests.

These use a fake depth model and a temporary project directory. They do not
load Depth Anything weights or write into the repository projects/ folder.
"""

import threading
import time
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from PIL import Image

from app.api.deps import get_depth_service
from app.core.config import MAX_SOURCE_IMAGE_BYTES
from app.depth.depth_anything import DepthAnythingV2Model
from app.depth.model import DepthModel
from app.depth.preview import grayscale_preview
from app.depth.registry import get_depth_model, reset_depth_model_cache
from app.depth.service import DepthService
from app.main import app
from app.projects.errors import DepthModelUnavailable
from app.projects.storage import ProjectStore
from tests.test_projects import create_project, image_bytes, served


class FakeDepthModel:
    model_id = "depth-anything-v2-small"
    model_version = "test"
    input_size = 518
    device = "cpu"

    def __init__(self) -> None:
        self.loads = 0

    def load(self) -> None:
        self.loads += 1

    def infer(self, image: Image.Image) -> np.ndarray:
        count = image.width * image.height
        return np.linspace(0.25, 1.5, count, dtype=np.float32).reshape(image.height, image.width)


class FailingDepthModel(FakeDepthModel):
    def infer(self, image: Image.Image) -> np.ndarray:
        raise RuntimeError("secret /tmp/checkpoint failed")


class SlowDepthModel(FakeDepthModel):
    current = 0
    peak = 0
    gate = threading.Lock()

    def infer(self, image: Image.Image) -> np.ndarray:
        with SlowDepthModel.gate:
            SlowDepthModel.current += 1
            SlowDepthModel.peak = max(SlowDepthModel.peak, SlowDepthModel.current)
        time.sleep(0.15)
        with SlowDepthModel.gate:
            SlowDepthModel.current -= 1
        return super().infer(image)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def test_depth_model_exposes_the_contract() -> None:
    model: DepthModel = FakeDepthModel()
    assert model.model_id == "depth-anything-v2-small"
    image = Image.new("RGB", (4, 3), (1, 2, 3))
    model.load()
    depth = model.infer(image)
    assert depth.dtype == np.float32
    assert depth.shape == (3, 4)


def test_registry_reuses_one_unloaded_model() -> None:
    reset_depth_model_cache()
    first = get_depth_model("depth-anything-v2-small")
    second = get_depth_model("depth-anything-v2-small")
    assert first is second
    assert isinstance(first, DepthAnythingV2Model)
    assert first._ready is False
    with pytest.raises(DepthModelUnavailable):
        get_depth_model("some-other-model")


def test_missing_checkpoint_does_not_mark_the_model_loaded(tmp_path: Path) -> None:
    model = DepthAnythingV2Model(tmp_path / "missing.pth")
    with pytest.raises(DepthModelUnavailable):
        model.load()
    assert model._ready is False


def test_model_build_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "depth_anything_v2_vits.pth"
    checkpoint.write_bytes(b"placeholder")
    model = DepthAnythingV2Model(checkpoint)
    calls = {"count": 0}

    def fake_build(self: DepthAnythingV2Model) -> None:
        calls["count"] += 1
        self._ready = True

    monkeypatch.setattr(DepthAnythingV2Model, "_build", fake_build)
    model.load()
    model.load()
    assert calls["count"] == 1


def test_preview_does_not_modify_the_raw_array() -> None:
    depth = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32)
    original = depth.copy()
    preview = grayscale_preview(depth)
    assert np.array_equal(depth, original)
    assert preview.startswith(b"\x89PNG")


def _use_model(project_root: Path, model: FakeDepthModel) -> None:
    app.dependency_overrides[get_depth_service] = lambda: DepthService(
        ProjectStore(project_root),
        model,
    )


def test_generate_depth_persists_float32_and_preview(project_root: Path) -> None:
    model = FakeDepthModel()
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _use_model(project_root, model)
        created = create_project(client)
        source = image_bytes("PNG", (6, 4))
        upload = client.post(
            f"/api/projects/{created['id']}/source",
            files={"file": ("scan.png", source, "image/png")},
        )
        assert upload.status_code == 200
        source_path = project_root / "sample-relief" / "source" / "original.png"
        before = source_path.read_bytes()
        response = client.post(f"/api/projects/{created['id']}/depth")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["created_at"] == created["created_at"]
        assert body["updated_at"] != created["updated_at"]
        assert body["depth"]["model_id"] == "depth-anything-v2-small"
        assert body["depth"]["device"] == "cpu"
        assert body["depth"]["input_size"] == 518
        assert body["depth"]["width_px"] == 6
        assert body["depth"]["height_px"] == 4
        assert body["depth"]["raw_filename"] == "raw_depth.npy"
        assert body["depth"]["preview_filename"] == "preview.png"
        raw_path = project_root / "sample-relief" / "depth" / "raw_depth.npy"
        loaded = np.load(raw_path)
        assert loaded.dtype == np.float32
        assert loaded.shape == (4, 6)
        expected = np.linspace(0.25, 1.5, 24, dtype=np.float32).reshape(4, 6)
        assert np.array_equal(loaded, expected)
        preview = client.get(f"/api/projects/{created['id']}/depth/preview")
        assert preview.status_code == 200
        assert preview.headers["content-type"].startswith("image/png")
        assert preview.content.startswith(b"\x89PNG")
        assert np.array_equal(np.load(raw_path), expected)
        assert source_path.read_bytes() == before
        assert model.loads == 1


def test_second_generation_reuses_the_same_model(project_root: Path) -> None:
    model = FakeDepthModel()
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _use_model(project_root, model)
        created = create_project(client)
        client.post(
            f"/api/projects/{created['id']}/source",
            files={"file": ("scan.png", image_bytes("PNG", (2, 2)), "image/png")},
        )
        assert client.post(f"/api/projects/{created['id']}/depth").status_code == 200
        assert client.post(f"/api/projects/{created['id']}/depth").status_code == 200
        assert model.loads == 2


def test_project_without_source_is_rejected(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _use_model(project_root, FakeDepthModel())
        created = create_project(client)
        response = client.post(f"/api/projects/{created['id']}/depth")
        assert response.status_code == 404
        assert not (project_root / "sample-relief" / "depth").exists()


def test_missing_project_is_rejected(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _use_model(project_root, FakeDepthModel())
        response = client.post(f"/api/projects/{uuid4()}/depth")
        assert response.status_code == 404


def test_failed_inference_does_not_record_depth(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _use_model(project_root, FailingDepthModel())
        created = create_project(client)
        client.post(
            f"/api/projects/{created['id']}/source",
            files={"file": ("scan.png", image_bytes("PNG", (2, 2)), "image/png")},
        )
        response = client.post(f"/api/projects/{created['id']}/depth")
        assert response.status_code == 500
        assert response.json()["detail"] == "Depth generation failed."
        assert "secret" not in response.text
        stored = client.get(f"/api/projects/{created['id']}").json()
        assert stored["depth"] is None
        assert not (project_root / "sample-relief" / "depth").exists()


def test_preview_missing_returns_404(project_root: Path) -> None:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        _use_model(project_root, FakeDepthModel())
        created = create_project(client)
        response = client.get(f"/api/projects/{created['id']}/depth/preview")
        assert response.status_code == 404


def test_concurrent_generation_does_not_overlap(project_root: Path) -> None:
    SlowDepthModel.current = 0
    SlowDepthModel.peak = 0
    model = SlowDepthModel()
    service = DepthService(ProjectStore(project_root), model)
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        uploaded = client.post(
            f"/api/projects/{created['id']}/source",
            files={"file": ("scan.png", image_bytes("PNG", (2, 2)), "image/png")},
        )
        assert uploaded.status_code == 200
        project_id = UUID(created["id"])
        errors: list[BaseException] = []

        def run() -> None:
            try:
                service.generate(project_id)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                errors.append(exc)

        first = threading.Thread(target=run)
        second = threading.Thread(target=run)
        first.start()
        second.start()
        first.join()
        second.join()
        assert errors == []
        assert SlowDepthModel.peak == 1
