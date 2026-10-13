"""Provenance and processing guards. Temporary projects only."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.api.deps import get_depth_service
from app.depth.service import DepthService
from app.main import app
from app.projects.provenance import (
    CONDITIONED_STALE_DETAIL,
    CONDITIONED_UNKNOWN_DETAIL,
    RAW_STALE_DETAIL,
    RAW_UNKNOWN_DETAIL,
    RELIEF_STALE_DETAIL,
    RELIEF_UNKNOWN_DETAIL,
    canonical_sha256,
    judge_stage,
)
from app.projects.storage import ProjectStore
from app.schemas.project import Project
from tests.test_depth import FakeDepthModel
from tests.test_physical_height import SETTINGS, _generate
from tests.test_projects import image_bytes, served
from tests.test_region_mask_edit import _accept
from tests.test_relief import FakeSegmentation, _bind, _conditioned


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


class ShiftedDepthModel(FakeDepthModel):
    def infer(self, image):
        return super().infer(image) + np.float32(0.1)


def _root(project_root: Path) -> Path:
    return project_root / "sample-relief"


def _manifest(project_root: Path) -> dict:
    return json.loads((_root(project_root) / "project.json").read_text(encoding="utf-8"))


def _write_manifest(project_root: Path, payload: dict) -> None:
    path = _root(project_root) / "project.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _strip_provenance(value: object) -> None:
    if isinstance(value, dict):
        for key in (
            "content_sha256",
            "source_sha256",
            "settings_sha256",
            "raw_sha256",
            "conditioned_sha256",
        ):
            value.pop(key, None)
        for child in value.values():
            _strip_provenance(child)
    elif isinstance(value, list):
        for child in value:
            _strip_provenance(child)


def _statuses(body: dict) -> dict[str, str | None]:
    relief = body.get("relief") or {}
    physical = relief.get("physical") or {}
    depth = body.get("depth") or {}
    conditioning = depth.get("conditioning") or {}
    return {
        "source": None if body.get("source") is None else body["source"].get("status"),
        "raw": None if body.get("depth") is None else depth.get("status"),
        "conditioned": None if not conditioning else conditioning.get("status"),
        "relief": None if body.get("relief") is None else relief.get("status"),
        "physical": None if not physical else physical.get("status"),
    }


def _pipeline(project_root: Path, client) -> dict:
    project = _conditioned(project_root, client)
    segmentation = FakeSegmentation()
    _bind(project_root, segmentation)
    composed = _accept(client, project["id"], segmentation)
    created = _generate(client, project["id"])
    assert created.status_code == 200, created.text
    return created.json() | {"_region": composed["relief"]["regions"][0]}


def _bytes(project_root: Path) -> dict[str, bytes]:
    root = _root(project_root)
    paths = [
        "source/original.png",
        "depth/raw_depth.npy",
        "depth/preview.png",
        "depth/processing/conditioned_depth.npy",
        "depth/processing/preview.png",
        "depth/composition/relief_depth.npy",
        "depth/composition/preview.png",
        "depth/composition/physical/physical_height_mm.npy",
        "depth/composition/physical/preview.png",
        "depth/composition/physical/stats.json",
    ]
    found = {rel: (root / rel).read_bytes() for rel in paths}
    masks = root / "depth" / "composition" / "masks"
    for path in masks.iterdir():
        found[f"depth/composition/masks/{path.name}"] = path.read_bytes()
    return found


def test_canonical_json_is_stable_and_ignores_key_order() -> None:
    first = canonical_sha256({"b": 1.0, "a": False, "nested": {"z": 2, "y": "mm"}})
    second = canonical_sha256({"nested": {"y": "mm", "z": 2}, "a": False, "b": 1.0})
    assert first == second
    assert first != canonical_sha256({"a": False, "b": 1.0, "nested": {"y": "mm", "z": 3}})


def test_judge_precedence_does_not_promote_unknown_ancestors() -> None:
    assert judge_stage(
        present=False,
        content_recorded=True,
        content_ok=False,
        settings_recorded=True,
        settings_ok=False,
        input_recorded=True,
        input_ok=False,
        require_settings=True,
        require_input=True,
        upstream="stale",
    ) == "missing"
    assert judge_stage(
        present=True,
        content_recorded=True,
        content_ok=False,
        settings_recorded=False,
        settings_ok=True,
        input_recorded=False,
        input_ok=True,
        require_settings=True,
        require_input=True,
        upstream="unknown",
    ) == "stale"
    assert judge_stage(
        present=True,
        content_recorded=False,
        content_ok=True,
        settings_recorded=False,
        settings_ok=True,
        input_recorded=False,
        input_ok=True,
        require_settings=True,
        require_input=True,
        upstream="stale",
    ) == "unknown"
    matched = dict(
        present=True,
        content_recorded=True,
        content_ok=True,
        settings_recorded=True,
        settings_ok=True,
        input_recorded=True,
        input_ok=True,
        require_settings=True,
        require_input=True,
    )
    assert judge_stage(**matched, upstream="unknown") == "unknown"
    assert judge_stage(**matched, upstream="stale") == "stale"
    assert judge_stage(**matched, upstream="current") == "current"
    assert judge_stage(**matched, upstream=None) == "current"


def test_new_pipeline_is_current_and_status_is_not_stored(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        assert _statuses(body) == {
            "source": "current",
            "raw": "current",
            "conditioned": "current",
            "relief": "current",
            "physical": "current",
        }
        stored = _manifest(project_root)
        assert "status" not in json.dumps(stored)
        assert stored["source"]["content_sha256"]
        assert stored["depth"]["source_sha256"] == stored["source"]["content_sha256"]
        assert stored["depth"]["conditioning"]["raw_sha256"] == stored["depth"]["content_sha256"]
        assert stored["relief"]["conditioned_sha256"] == stored["depth"]["conditioning"]["content_sha256"]
        assert stored["relief"]["physical"]["content_sha256"]
        assert stored["relief"]["physical"]["source_relief_sha256"] == stored["relief"]["content_sha256"]
        opened = client.get(f"/api/projects/{body['id']}")
        assert opened.status_code == 200
        assert _manifest(project_root) == stored


def test_source_replacement_keeps_downstream_bytes_and_marks_verified_stages_stale(
    project_root: Path,
) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        before = _bytes(project_root)
        replaced = client.post(
            f"/api/projects/{body['id']}/source",
            files={"file": ("other.png", image_bytes("PNG", (5, 5)), "image/png")},
        )
        assert replaced.status_code == 200, replaced.text
        after = _bytes(project_root)
        for rel, data in before.items():
            if rel == "source/original.png":
                assert after[rel] != data
            else:
                assert after[rel] == data
        assert _statuses(replaced.json()) == {
            "source": "current",
            "raw": "stale",
            "conditioned": "stale",
            "relief": "stale",
            "physical": "stale",
        }
        blocked = client.post(f"/api/projects/{body['id']}/depth/conditioning", json={})
        assert blocked.status_code == 409
        assert blocked.json()["detail"] == RAW_STALE_DETAIL
        assert _bytes(project_root)["depth/processing/conditioned_depth.npy"] == before[
            "depth/processing/conditioned_depth.npy"
        ]
        refused = _generate(client, body["id"])
        assert refused.status_code == 409
        assert refused.json()["detail"] == RELIEF_STALE_DETAIL
        assert _bytes(project_root)["depth/composition/physical/physical_height_mm.npy"] == before[
            "depth/composition/physical/physical_height_mm.npy"
        ]


def test_legacy_replacement_stays_unknown_when_lineage_was_not_recorded(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        payload = _manifest(project_root)
        _strip_provenance(payload)
        _write_manifest(project_root, payload)
        before = _bytes(project_root)
        before_manifest = (_root(project_root) / "project.json").read_bytes()
        opened = client.get(f"/api/projects/{body['id']}")
        assert opened.status_code == 200, opened.text
        assert (_root(project_root) / "project.json").read_bytes() == before_manifest
        assert _statuses(opened.json()) == {
            "source": "unknown",
            "raw": "unknown",
            "conditioned": "unknown",
            "relief": "unknown",
            "physical": "unknown",
        }
        replaced = client.post(
            f"/api/projects/{body['id']}/source",
            files={"file": ("other.png", image_bytes("PNG", (5, 5)), "image/png")},
        )
        assert replaced.status_code == 200, replaced.text
        after = _bytes(project_root)
        for rel, data in before.items():
            if rel != "source/original.png":
                assert after[rel] == data
        assert _statuses(replaced.json())["source"] == "current"
        assert _statuses(replaced.json())["raw"] == "unknown"
        assert _statuses(replaced.json())["conditioned"] == "unknown"
        assert _statuses(replaced.json())["relief"] == "unknown"
        assert _statuses(replaced.json())["physical"] == "unknown"
        blocked = client.post(f"/api/projects/{body['id']}/depth/conditioning", json={})
        assert blocked.status_code == 409
        assert blocked.json()["detail"] == RAW_UNKNOWN_DETAIL


def test_raw_regeneration_retains_conditioning_and_does_not_claim_it_is_current(
    project_root: Path,
) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        before = _bytes(project_root)
        conditioning = _manifest(project_root)["depth"]["conditioning"]
        store = ProjectStore(project_root)
        app.dependency_overrides[get_depth_service] = lambda: DepthService(store, ShiftedDepthModel())
        regenerated = client.post(f"/api/projects/{body['id']}/depth")
        assert regenerated.status_code == 200, regenerated.text
        stored = _manifest(project_root)["depth"]["conditioning"]
        assert stored["generated_at"] == conditioning["generated_at"]
        assert stored["raw_sha256"] == conditioning["raw_sha256"]
        assert stored["raw_sha256"] != _manifest(project_root)["depth"]["content_sha256"]
        assert _bytes(project_root)["depth/processing/conditioned_depth.npy"] == before[
            "depth/processing/conditioned_depth.npy"
        ]
        assert _bytes(project_root)["depth/composition/relief_depth.npy"] == before[
            "depth/composition/relief_depth.npy"
        ]
        assert _bytes(project_root)["depth/composition/physical/physical_height_mm.npy"] == before[
            "depth/composition/physical/physical_height_mm.npy"
        ]
        assert regenerated.json()["depth"]["status"] == "current"
        assert regenerated.json()["depth"]["conditioning"]["status"] == "stale"
        assert regenerated.json()["relief"]["status"] == "stale"
        assert regenerated.json()["relief"]["physical"]["status"] == "stale"


def test_legacy_conditioning_remains_unknown_after_raw_regeneration(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        payload = _manifest(project_root)
        _strip_provenance(payload["depth"]["conditioning"])
        _write_manifest(project_root, payload)
        before = (_root(project_root) / "depth" / "processing" / "conditioned_depth.npy").read_bytes()
        regenerated = client.post(f"/api/projects/{body['id']}/depth")
        assert regenerated.status_code == 200, regenerated.text
        assert regenerated.json()["depth"]["conditioning"]["status"] == "unknown"
        assert regenerated.json()["depth"]["conditioning"]["status"] != "current"
        assert (_root(project_root) / "depth" / "processing" / "conditioned_depth.npy").read_bytes() == before


def test_byte_changes_timestamp_edits_and_missing_files(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        raw_path = _root(project_root) / "depth" / "raw_depth.npy"
        original = raw_path.read_bytes()
        generated_at = _manifest(project_root)["depth"]["generated_at"]
        flipped = bytearray(original)
        flipped[-1] ^= 0x01
        raw_path.write_bytes(bytes(flipped))
        changed = client.get(f"/api/projects/{body['id']}")
        assert changed.json()["depth"]["status"] == "stale"
        assert changed.json()["depth"]["generated_at"] == generated_at
        raw_path.write_bytes(original)

        payload = _manifest(project_root)
        payload["depth"]["generated_at"] = "2020-01-01T00:00:00Z"
        payload["depth"]["conditioning"]["generated_at"] = "2020-01-01T00:00:00Z"
        payload["relief"]["conditioned_generated_at"] = "2020-01-01T00:00:00Z"
        payload["relief"]["generated_at"] = "2020-01-01T00:00:00Z"
        _write_manifest(project_root, payload)
        same_bytes = client.get(f"/api/projects/{body['id']}")
        assert _statuses(same_bytes.json()) == {
            "source": "current",
            "raw": "current",
            "conditioned": "current",
            "relief": "current",
            "physical": "current",
        }

        mask = next((_root(project_root) / "depth" / "composition" / "masks").iterdir())
        mask_bytes = mask.read_bytes()
        mask.write_bytes(mask_bytes + b"\x00")
        masked = client.get(f"/api/projects/{body['id']}")
        assert masked.json()["relief"]["status"] == "stale"
        assert masked.json()["relief"]["physical"]["status"] == "stale"
        mask.write_bytes(mask_bytes)

        payload = _manifest(project_root)
        payload["relief"]["regions"][0]["elevation"] = 0.9
        _write_manifest(project_root, payload)
        moved = client.get(f"/api/projects/{body['id']}")
        assert moved.json()["relief"]["status"] == "stale"
        relief_bytes = (_root(project_root) / "depth" / "composition" / "relief_depth.npy").read_bytes()
        payload = _manifest(project_root)
        payload["relief"]["regions"][0]["elevation"] = 0.25
        _write_manifest(project_root, payload)
        assert (_root(project_root) / "depth" / "composition" / "relief_depth.npy").read_bytes() == relief_bytes

        raw_path.unlink()
        missing = client.get(f"/api/projects/{body['id']}")
        assert missing.json()["depth"]["status"] == "missing"
        assert missing.json()["depth"]["status"] != "current"
        raw_path.write_bytes(original)


def test_guards_reject_unknown_composition_and_keep_previews(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        region = body["_region"]
        preview = client.get(f"/api/projects/{body['id']}/relief/physical-height/preview")
        assert preview.status_code == 200
        preview_bytes = preview.content
        relief_preview = client.get(f"/api/projects/{body['id']}/relief/preview")
        assert relief_preview.status_code == 200
        payload = _manifest(project_root)
        _strip_provenance(payload["relief"])
        _write_manifest(project_root, payload)
        unknown = _generate(client, body["id"])
        assert unknown.status_code == 409
        assert unknown.json()["detail"] == RELIEF_UNKNOWN_DETAIL
        assert client.get(f"/api/projects/{body['id']}/relief/physical-height/preview").content == preview_bytes
        compose = client.post(
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
        assert compose.status_code == 200, compose.text
        stale_relief = (_root(project_root) / "depth" / "composition" / "relief_depth.npy").read_bytes()
        payload = _manifest(project_root)
        _strip_provenance(payload["depth"]["conditioning"])
        _write_manifest(project_root, payload)
        blocked = client.post(
            f"/api/projects/{body['id']}/relief/composition",
            json={
                "regions": [
                    {
                        "id": region["id"],
                        "name": region["name"],
                        "role": region["role"],
                        "elevation": 0.2,
                        "depth_range": region["depth_range"],
                        "feather": region["feather"],
                        "order": region["order"],
                    }
                ]
            },
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"] == CONDITIONED_UNKNOWN_DETAIL
        assert (_root(project_root) / "depth" / "composition" / "relief_depth.npy").read_bytes() == stale_relief
        predicted = client.post(
            f"/api/projects/{body['id']}/segmentation/predict",
            json={"positive_points": [{"x": 1, "y": 1}], "name": "Keep", "role": "primary"},
        )
        assert predicted.status_code == 200, predicted.text


def test_manifest_failure_restores_raw_and_conditioned_files(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with served(project_root, 8_000_000) as client:
        body = _conditioned(project_root, client)
        raw = (_root(project_root) / "depth" / "raw_depth.npy").read_bytes()
        conditioned = (_root(project_root) / "depth" / "processing" / "conditioned_depth.npy").read_bytes()
        manifest = (_root(project_root) / "project.json").read_bytes()

        def fail_manifest(self: ProjectStore, directory: Path, saved: object) -> None:
            raise OSError("manifest")

        monkeypatch.setattr(ProjectStore, "_write_manifest", fail_manifest)
        with pytest.raises(OSError, match="manifest"):
            client.post(f"/api/projects/{body['id']}/depth")
        assert (_root(project_root) / "depth" / "raw_depth.npy").read_bytes() == raw
        assert (_root(project_root) / "depth" / "processing" / "conditioned_depth.npy").read_bytes() == conditioned
        assert (_root(project_root) / "project.json").read_bytes() == manifest
        with pytest.raises(OSError, match="manifest"):
            client.post(
                f"/api/projects/{body['id']}/depth/conditioning",
                json={"contrast": 1.5},
            )
        assert (_root(project_root) / "depth" / "processing" / "conditioned_depth.npy").read_bytes() == conditioned
        assert (_root(project_root) / "project.json").read_bytes() == manifest
        monkeypatch.undo()


def test_artwork_edit_and_preview_png_do_not_change_verified_status(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        updated = client.put(
            f"/api/projects/{body['id']}",
            json={
                "name": body["name"],
                "stock": body["stock"],
                "artwork": {
                    "width_mm": body["artwork"]["width_mm"] + 1,
                    "height_mm": body["artwork"]["height_mm"],
                    "center_x_mm": body["artwork"]["center_x_mm"],
                    "center_y_mm": body["artwork"]["center_y_mm"],
                },
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["relief"]["physical"]["status"] == "current"
        preview = _root(project_root) / "depth" / "composition" / "physical" / "preview.png"
        preview.write_bytes(preview.read_bytes() + b"\x00")
        opened = client.get(f"/api/projects/{body['id']}")
        assert opened.json()["relief"]["physical"]["status"] == "current"
        height = _root(project_root) / "depth" / "composition" / "physical" / "physical_height_mm.npy"
        original = height.read_bytes()
        flipped = bytearray(original)
        flipped[-1] ^= 0x01
        height.write_bytes(bytes(flipped))
        stale = client.get(f"/api/projects/{body['id']}")
        assert stale.json()["relief"]["physical"]["status"] == "stale"
        height.write_bytes(original)


def test_legacy_manifest_without_provenance_fields_still_loads(project_root: Path) -> None:
    document = {
        "schema_version": 1,
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "Legacy",
        "slug": "legacy",
        "units": "mm",
        "stock": {"width_mm": 10, "height_mm": 10},
        "artwork": {
            "width_mm": 8,
            "height_mm": 8,
            "center_x_mm": 5,
            "center_y_mm": 5,
        },
    }
    Project.model_validate(document)
    directory = project_root / "legacy"
    directory.mkdir(parents=True)
    (directory / "project.json").write_text(json.dumps(document), encoding="utf-8")
    loaded = ProjectStore(project_root).list_projects()
    assert [project.name for project in loaded] == ["Legacy"]
    assert loaded[0].source is None
    assert loaded[0].depth is None
    assert loaded[0].relief is None


def test_conditioning_stale_detail_is_stable() -> None:
    assert "Regenerate raw depth" in RAW_STALE_DETAIL
    assert "Apply conditioning" in CONDITIONED_STALE_DETAIL
    assert "Apply composition" in RELIEF_STALE_DETAIL
    assert SETTINGS["stock_thickness_mm"] == 19
