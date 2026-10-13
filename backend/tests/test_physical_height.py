"""Physical finished-surface height. These tests use a temporary project root."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.depth.preview import grayscale_preview
from app.projects.errors import InvalidPhysicalHeight
from app.projects.storage import ProjectStore
from app.relief.physical import (
    cut_depth_mm,
    fixed_preview_png,
    map_physical_height,
    maximum_cut_depth_mm,
    relief_amplitude_mm,
    validate_physical_settings,
    write_physical_bundle,
)
from app.schemas.relief import PhysicalHeightRequest
from tests.test_projects import served
from tests.test_region_mask_edit import _accept
from tests.test_relief import FakeSegmentation, _bind, _conditioned

SETTINGS = {
    "stock_thickness_mm": 19,
    "base_thickness_mm": 5,
    "maximum_surface_height_mm": 17,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compose(client, project_id: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/relief/composition",
        json={"regions": []},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _generate(client, project_id: str, body: dict | None = None):
    return client.post(
        f"/api/projects/{project_id}/relief/physical-height",
        json=SETTINGS if body is None else body,
    )


def _root(project_root: Path) -> Path:
    return project_root / "sample-relief"


def test_mapping_is_exact_monotonic_and_leaves_the_relief_unchanged() -> None:
    relief = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    before = relief.tobytes()
    height = map_physical_height(relief, 5, 17)
    assert relief.tobytes() == before
    assert height.dtype == np.float32
    assert height.shape == relief.shape
    assert np.array_equal(height[1, 0], np.float32(11))
    assert np.array_equal(height, np.float32(5) + relief * np.float32(12))
    flat = height.reshape(-1)
    assert flat[0] < flat[1] < flat[2] < flat[3]
    again = map_physical_height(relief, 5, 17)
    assert np.array_equal(height, again)
    cuts = cut_depth_mm(19, height)
    assert float(cuts.min()) >= 19 - 17
    assert float(cuts.max()) <= 19 - 5
    assert maximum_cut_depth_mm(19, 5) == 14
    assert relief_amplitude_mm(17, 5) == 12
    assert float(cuts[0, 0]) == 14
    assert float(cuts[1, 0]) == 8


def test_invalid_relief_and_settings_are_rejected() -> None:
    valid = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(InvalidPhysicalHeight):
        map_physical_height(valid.astype(np.float64), 5, 17)
    with pytest.raises(InvalidPhysicalHeight):
        map_physical_height(valid.reshape(-1), 5, 17)
    with pytest.raises(InvalidPhysicalHeight):
        map_physical_height(valid.reshape(2, 1, 2), 5, 17)
    with pytest.raises(InvalidPhysicalHeight):
        map_physical_height(np.zeros((0, 2), dtype=np.float32), 5, 17)
    for value in (np.float32(np.nan), np.float32(np.inf), np.float32(-0.01), np.float32(1.01)):
        broken = valid.copy()
        broken[0, 0] = value
        changed = broken.tobytes()
        with pytest.raises(InvalidPhysicalHeight):
            map_physical_height(broken, 5, 17)
        assert broken.tobytes() == changed
    with pytest.raises(InvalidPhysicalHeight):
        map_physical_height(valid, 0, 17)
    with pytest.raises(InvalidPhysicalHeight):
        map_physical_height(valid, 17, 17)
    with pytest.raises(InvalidPhysicalHeight):
        validate_physical_settings(19, 8, 20)
    with pytest.raises(ValidationError):
        PhysicalHeightRequest(
            stock_thickness_mm=19,
            base_thickness_mm=5,
            maximum_surface_height_mm=17,
            stock_width_mm=304.8,
        )


def test_preview_uses_fixed_base_to_peak_scale() -> None:
    relief = np.array([[0.0, 0.25], [0.5, 0.75]], dtype=np.float32)
    height = map_physical_height(relief, 5, 17)
    preview = fixed_preview_png(height, 5, 17)
    stretched = grayscale_preview(height)
    assert preview != stretched
    with Image.open(io.BytesIO(preview)) as handle:
        pixels = np.array(handle)
    assert pixels.dtype == np.uint8
    assert pixels[0, 0] == 0
    assert pixels[1, 0] == 128
    assert pixels[0, 1] != 0
    assert pixels[1, 1] != 255
    assert pixels[0, 1] < pixels[1, 1]


def test_failed_replace_restores_the_previous_physical_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "panel"
    (directory / "depth" / "composition").mkdir(parents=True)
    height = np.full((2, 3), 11, dtype=np.float32)
    write_physical_bundle(directory, height, fixed_preview_png(height, 5, 17), '{"schema":1}\n')
    before = {
        name: (directory / "depth" / "composition" / "physical" / name).read_bytes()
        for name in ("physical_height_mm.npy", "preview.png", "stats.json")
    }
    failed = {"done": False}
    original = Path.replace

    def wrapped(self: Path, target: Path) -> Path:
        if self.name == ".stats.partial.json" and not failed["done"]:
            failed["done"] = True
            raise OSError("disk full")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", wrapped)
    replacement = map_physical_height(np.full((2, 3), 0.25, dtype=np.float32), 5, 17)
    with pytest.raises(OSError):
        write_physical_bundle(directory, replacement, fixed_preview_png(replacement, 5, 17), '{"schema":2}\n')
    folder = directory / "depth" / "composition" / "physical"
    for name, data in before.items():
        assert (folder / name).read_bytes() == data
    assert list(folder.glob(".*partial*")) == []


def test_physical_height_endpoints(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segmentation = FakeSegmentation()
    with served(project_root, 8_000_000) as client:
        project = _conditioned(project_root, client)
        missing = _generate(client, project["id"])
        assert missing.status_code == 409
        assert missing.json()["detail"] == "Composed relief has not been created."
        assert client.get(f"/api/projects/{project['id']}/relief/physical-height/preview").status_code == 404
        assert client.post(f"/api/projects/{project['id']}/relief/stl").status_code == 404
        assert client.post(f"/api/projects/{project['id']}/gcode").status_code == 404

        _bind(project_root, segmentation)
        composed = _accept(client, project["id"], segmentation)
        root = _root(project_root)
        relief_path = root / "depth" / "composition" / "relief_depth.npy"
        protected = {
            "source": _sha256(root / "source" / "original.png"),
            "raw": _sha256(root / "depth" / "raw_depth.npy"),
            "conditioned": _sha256(root / "depth" / "processing" / "conditioned_depth.npy"),
            "relief": _sha256(relief_path),
            "mask": _sha256(root / "depth" / "composition" / "masks" / f"{composed['relief']['regions'][0]['id']}.png"),
        }
        relief_before = relief_path.read_bytes()
        np.save(relief_path, np.zeros(np.load(relief_path).shape, dtype=np.float64))
        invalid_array = _generate(client, project["id"])
        assert invalid_array.status_code == 422
        assert "float32" in invalid_array.json()["detail"]
        relief_path.write_bytes(relief_before)
        assert not (root / "depth" / "composition" / "physical").exists()
        created = _generate(client, project["id"])
        assert created.status_code == 200, created.text
        physical = created.json()["relief"]["physical"]
        assert physical["status"] == "current"
        assert physical["height_path"] == "depth/composition/physical/physical_height_mm.npy"
        assert physical["preview_path"] == "depth/composition/physical/preview.png"
        assert physical["stats_path"] == "depth/composition/physical/stats.json"
        assert physical["source_relief_sha256"] == protected["relief"]
        assert physical["maximum_cut_depth_mm"] == 14
        assert physical["relief_amplitude_mm"] == 12
        assert physical["dtype"] == "float32"
        assert created.json()["stock"]["width_mm"] != physical["stock_thickness_mm"]

        folder = root / "depth" / "composition" / "physical"
        height = np.load(folder / "physical_height_mm.npy")
        relief = np.load(relief_path)
        assert height.dtype == np.float32
        assert height.shape == relief.shape
        assert np.array_equal(height, np.float32(5) + relief * np.float32(12))
        assert relief_path.read_bytes() == relief_before
        manifest = json.loads((root / "project.json").read_text(encoding="utf-8"))
        assert "status" not in manifest["relief"]["physical"]
        stats = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
        assert stats["source_relief_sha256"] == protected["relief"]
        assert stats["height_min_mm"] == physical["height_min_mm"]
        assert "status" not in stats

        preview = client.get(f"/api/projects/{project['id']}/relief/physical-height/preview")
        assert preview.status_code == 200
        assert preview.content == fixed_preview_png(height, 5, 17)
        assert preview.content != grayscale_preview(height)
        repeated = _generate(client, project["id"])
        assert repeated.status_code == 200, repeated.text
        assert np.array_equal(np.load(folder / "physical_height_mm.npy"), height)

        stored = (folder / "physical_height_mm.npy").read_bytes()
        preview_bytes = (folder / "preview.png").read_bytes()
        stats_bytes = (folder / "stats.json").read_bytes()
        manifest_bytes = (root / "project.json").read_bytes()

        def fail_manifest(self: ProjectStore, directory: Path, saved: object) -> None:
            raise OSError("manifest")

        monkeypatch.setattr(ProjectStore, "_write_manifest", fail_manifest)
        failed = _generate(
            client,
            project["id"],
            {
                "stock_thickness_mm": 19,
                "base_thickness_mm": 4,
                "maximum_surface_height_mm": 16,
            },
        )
        assert failed.status_code == 500
        assert (folder / "physical_height_mm.npy").read_bytes() == stored
        assert (folder / "preview.png").read_bytes() == preview_bytes
        assert (folder / "stats.json").read_bytes() == stats_bytes
        assert (root / "project.json").read_bytes() == manifest_bytes
        assert list(folder.glob(".*partial*")) == []
        assert not (root / ".project.json.partial").exists()
        monkeypatch.undo()

        region = composed["relief"]["regions"][0]
        opened = client.post(f"/api/projects/{project['id']}/relief/regions/{region['id']}/mask-edit")
        assert opened.status_code == 200, opened.text
        painted = client.post(
            f"/api/projects/{project['id']}/segmentation/proposals/{opened.json()['proposal_id']}/strokes",
            json={"mode": "add", "brush_radius_px": 0.6, "points": [{"x": 4.2, "y": 2.2}]},
        )
        assert painted.status_code == 200, painted.text
        edited = client.post(
            f"/api/projects/{project['id']}/relief/regions/{region['id']}/mask-edit/save",
            json={"proposal_id": opened.json()["proposal_id"], "name": "Subject", "role": "primary"},
        )
        assert edited.status_code == 200, edited.text
        assert edited.json()["relief"]["physical"]["status"] == "stale"
        assert relief_path.read_bytes() == relief_before
        assert (folder / "physical_height_mm.npy").read_bytes() == stored
        stale_preview = client.get(f"/api/projects/{project['id']}/relief/physical-height/preview")
        assert stale_preview.status_code == 200
        assert stale_preview.content == preview_bytes

        recomposed = client.post(
            f"/api/projects/{project['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": "Subject",
                        "role": "secondary",
                        "elevation": 0.5,
                        "depth_range": 1.4,
                        "feather": 6,
                        "order": 0,
                    }
                ]
            },
        )
        assert recomposed.status_code == 200, recomposed.text
        assert recomposed.json()["relief"]["physical"]["status"] == "stale"
        assert (folder / "physical_height_mm.npy").read_bytes() == stored
        assert relief_path.read_bytes() != relief_before

        assert _sha256(root / "source" / "original.png") == protected["source"]
        assert _sha256(root / "depth" / "raw_depth.npy") == protected["raw"]
        assert _sha256(root / "depth" / "processing" / "conditioned_depth.npy") == protected["conditioned"]

        invalid = {
            "stock_thickness_mm": 19,
            "base_thickness_mm": 17,
            "maximum_surface_height_mm": 5,
        }
        rejected = _generate(client, project["id"], invalid)
        assert rejected.status_code == 422
        with pytest.raises(InvalidPhysicalHeight):
            validate_physical_settings(19, float("nan"), 17)
        not_a_number = client.post(
            f"/api/projects/{project['id']}/relief/physical-height",
            json={
                "stock_thickness_mm": 19,
                "base_thickness_mm": "five",
                "maximum_surface_height_mm": 17,
            },
        )
        assert not_a_number.status_code == 422
        assert (folder / "physical_height_mm.npy").read_bytes() == stored


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"
