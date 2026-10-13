"""Project duplication copies bytes and leaves the source project unchanged."""

import json
import os
import shutil
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import default_projects_dir
from app.projects.provenance import resolve_artifact_statuses
from app.projects.storage import ProjectStore
from tests.test_projects import MAX_SOURCE_IMAGE_BYTES, create_project, image_bytes, served

PROTECTED = {
    "wedding-relief-benchmark/project.json": "22a6080d6dcf673f293bcfd9a5cbfc2f4f8f54de60d9ea285dbdbd0fd520a7c8",
    "wedding-relief-benchmark/source/original.jpg": "9e35ce50c4d82ef767440ce8f742a2663df5ed32bff84ffdfbac5c59c68e0eef",
    "wedding-relief-benchmark/depth/raw_depth.npy": "9ab85025fdff498113f3b3c618f154d920d285a8624d8fee4820b1b00058d0e8",
    "wedding-relief-benchmark/depth/preview.png": "8f62306428bd1069856db1c6d043edcfee7f5643c2c15b8ebb447c41397e920d",
    "wedding-relief-benchmark/depth/processing/conditioned_depth.npy": "d8a21fa6cc5aaf4525779c9ffe1beb8d661f62940e973f329a8c675ffa22e78b",
    "wedding-relief-benchmark/depth/processing/preview.png": "09221ecef9056d61773bda7c1217a2352acdb5e7e5d1943852e088f5b4b038ee",
    "wedding-relief-benchmark/depth/composition/relief_depth.npy": "d8a21fa6cc5aaf4525779c9ffe1beb8d661f62940e973f329a8c675ffa22e78b",
    "wedding-relief-benchmark/depth/composition/preview.png": "09221ecef9056d61773bda7c1217a2352acdb5e7e5d1943852e088f5b4b038ee",
    "wedding-relief-benchmark/depth/composition/masks/92c35f7a-0814-458b-a456-9621ed32cccc.png": "be79bf17e4ab1cbca6336991d94f453a87c9902838644f55b32675a45eae696b",
    "wedding-relief-benchmark/depth/composition/masks/ea280aed-8759-4c9b-915c-074ac67727af.png": "ff7274f0d492b5e3284c994d1da0272268db7bf735ca0bfca861f515eb5f504c",
    "angie-dance/project.json": "3faae690ae95ee15ee5340165f00002ed59085688ef4b905358c85aee5ef7ab7",
    "angie-dance/source/original.jpg": "8018d9d884e5ef13f50a9524d362a4062f6b1c4ab98744fc79cba1b715b0e230",
    "angie-dance/depth/raw_depth.npy": "a3605c4ddab77a4cf36c4c6581caa56f9d514af72a11ae921a310aa9e4a3be2f",
    "angie-dance/depth/preview.png": "844fb6eae5c571b1edeac6eb8797eb1af44f2d3b7234bd4604fbca3de6392e79",
    "angie-dance/depth/processing/conditioned_depth.npy": "486e67922808df819daa461f004bd59c12a31bfa26b4f836fb62c483c3a102e4",
    "angie-dance/depth/processing/preview.png": "9a6d494932f4d3053a176aa03190a7dc80b9c0701772ba9d198b4018f602695d",
    "angie-dance/depth/composition/relief_depth.npy": "92fc23249f9e713328f411d37747ae9c5211c2a4dd12b29e2eb60c7e7da6e8de",
    "angie-dance/depth/composition/preview.png": "881ceeca3be7ed43058c77f6b3cbf931de16c5d0ad5fe148161222052b8cc4c0",
    "angie-dance/depth/composition/masks/9b6d7dce-c3d1-47f1-a58e-7a8c2d80e19c.png": "64182cb37f661095a16ac39f52be9a81cb6abc30db68c63f5fb96df1b44560b5",
    "angie-dance/depth/composition/masks/f691285d-f58f-48c5-a07c-9d0562703da1.png": "7c61dd2b16e84a1fd8c0c7b27f924b245cb10c2dcaa43f43511d76f69208f97e",
}


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _duplicate(client: TestClient, project_id: str, name: str):
    return client.post(f"/api/projects/{project_id}/duplicate", json={"name": name})


def test_duplicate_is_independent_and_leaves_the_original_unchanged(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        upload = client.post(
            f"/api/projects/{created['id']}/source",
            files={"file": ("scan.png", image_bytes("PNG", (6, 4)), "image/png")},
        )
        assert upload.status_code == 200
        original_dir = project_root / created["slug"]
        note = original_dir / "notes.txt"
        note.write_text("keep this extra file\n", encoding="utf-8")
        before = _files(original_dir)
        response = _duplicate(client, created["id"], "Sample Relief Copy")
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["duplicated"] is True
        assert body["source_project_id"] == created["id"]
        assert body["name"] == "Sample Relief Copy"
        assert body["slug"] == "sample-relief-copy"
        assert body["id"] != created["id"]
        assert _files(original_dir) == before
        copied = _files(project_root / body["slug"])
        assert copied["source/original.png"] == before["source/original.png"]
        assert copied["notes.txt"] == before["notes.txt"]
        manifest = json.loads(copied["project.json"])
        original_manifest = json.loads(before["project.json"])
        assert manifest["id"] == body["id"]
        assert manifest["slug"] == body["slug"]
        assert manifest["name"] == body["name"]
        assert manifest["source"]["content_sha256"] == original_manifest["source"]["content_sha256"]
        for relative, payload in before.items():
            if relative == "project.json":
                continue
            assert copied[relative] == payload
            source_stat = (original_dir / relative).stat()
            copy_stat = (project_root / body["slug"] / relative).stat()
            assert copy_stat.st_ino != source_stat.st_ino or copy_stat.st_dev != source_stat.st_dev
        loaded = client.get(f"/api/projects/{body['id']}")
        assert loaded.status_code == 200
        assert loaded.json()["source"]["status"] == "current"
        listed = client.get("/api/projects")
        names = {item["name"] for item in listed.json()}
        assert names == {"Sample Relief", "Sample Relief Copy"}


def test_duplicate_preserves_recorded_provenance_and_regions(tmp_path: Path) -> None:
    from tests.test_stl_export_e2e import _establish

    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project, _height = _establish(project_root, client)
        original_dir = project_root / "disposable-export"
        before = _files(original_dir)
        response = _duplicate(client, project["id"], "Disposable Export Copy")
        assert response.status_code == 201, response.text
        body = response.json()
        assert _files(original_dir) == before
        store = ProjectStore(project_root)
        original = store.get_by_id(UUID(project["id"]))
        copied = store.get_by_id(UUID(body["id"]))
        assert resolve_artifact_statuses(original_dir, original) == {
            "source": "current",
            "raw": "current",
            "conditioned": "current",
            "relief": "current",
            "physical": "current",
        }
        assert resolve_artifact_statuses(project_root / body["slug"], copied) == {
            "source": "current",
            "raw": "current",
            "conditioned": "current",
            "relief": "current",
            "physical": "current",
        }
        assert [(region.name, region.elevation) for region in copied.relief.regions] == [
            (region.name, region.elevation) for region in original.relief.regions
        ]
        assert copied.depth.content_sha256 == original.depth.content_sha256
        assert copied.relief.content_sha256 == original.relief.content_sha256
        assert copied.relief.physical.content_sha256 == original.relief.physical.content_sha256


def test_historical_shape_stays_unknown_and_masks_match(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        directory = project_root / created["slug"]
        manifest = json.loads((directory / "project.json").read_text(encoding="utf-8"))
        array = np.zeros((4, 6), dtype=np.float32)
        array[1, 5] = np.float32(1.0)
        for relative in (
            "depth/raw_depth.npy",
            "depth/processing/conditioned_depth.npy",
            "depth/composition/relief_depth.npy",
        ):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
        mask_dir = directory / "depth" / "composition" / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / "11111111-1111-4111-8111-111111111111.png"
        mask_path.write_bytes(b"\x89PNG-mask")
        manifest["source"] = {
            "filename": "original.jpg",
            "original_filename": "photo.jpg",
            "media_type": "image/jpeg",
            "width_px": 6,
            "height_px": 4,
        }
        (directory / "source").mkdir(exist_ok=True)
        (directory / "source" / "original.jpg").write_bytes(b"jpeg-bytes")
        manifest["depth"] = {
            "model_id": "depth-anything-v2-small",
            "model_version": "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
            "device": "mps",
            "input_size": 518,
            "width_px": 6,
            "height_px": 4,
            "raw_filename": "raw_depth.npy",
            "preview_filename": "preview.png",
            "generated_at": "2026-10-04T02:41:17.603408Z",
            "conditioning": {
                "parameters": {
                    "far_percentile": 1.0,
                    "near_percentile": 99.0,
                    "gamma": 1.0,
                    "contrast": 1.0,
                    "smoothing": 0.0,
                    "edge_preservation": 75.0,
                    "invert": False,
                },
                "width_px": 6,
                "height_px": 4,
                "dtype": "float32",
                "range_min": 0.0,
                "range_max": 1.0,
                "conditioned_filename": "conditioned_depth.npy",
                "preview_filename": "preview.png",
                "generated_at": "2026-10-04T02:42:40.123624Z",
            },
        }
        manifest["relief"] = {
            "regions": [
                {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "name": "Bride",
                    "role": "primary",
                    "mask_filename": "11111111-1111-4111-8111-111111111111.png",
                    "elevation": 0.0,
                    "depth_range": 1.0,
                    "feather": 0.0,
                    "order": 0,
                    "created_at": "2026-10-07T05:33:11.482737Z",
                    "updated_at": "2026-10-08T00:40:41.431671Z",
                }
            ],
            "conditioned_generated_at": "2026-10-04T02:42:40.123624Z",
            "width_px": 6,
            "height_px": 4,
            "dtype": "float32",
            "range_min": 0.0,
            "range_max": 1.0,
            "relief_filename": "relief_depth.npy",
            "preview_filename": "preview.png",
            "generated_at": "2026-10-08T00:40:41.452944Z",
        }
        (directory / "project.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        before = (directory / "project.json").read_bytes()
        response = _duplicate(client, created["id"], "Historical Copy")
        assert response.status_code == 201, response.text
        body = response.json()
        assert (directory / "project.json").read_bytes() == before
        store = ProjectStore(project_root)
        copied = store.get_by_id(UUID(body["id"]))
        statuses = resolve_artifact_statuses(project_root / body["slug"], copied)
        assert statuses == {
            "source": "unknown",
            "raw": "unknown",
            "conditioned": "unknown",
            "relief": "unknown",
            "physical": "missing",
        }
        assert copied.relief.regions[0].name == "Bride"
        assert copied.relief.regions[0].elevation == 0.0
        copied_mask = (
            project_root
            / body["slug"]
            / "depth"
            / "composition"
            / "masks"
            / "11111111-1111-4111-8111-111111111111.png"
        )
        assert copied_mask.read_bytes() == b"\x89PNG-mask"
        stored = json.loads((project_root / body["slug"] / "project.json").read_text(encoding="utf-8"))
        assert "content_sha256" not in json.dumps(stored["depth"])
        assert "status" not in json.dumps(stored)


def test_duplicate_rejects_collision_symlink_and_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        other = create_project(
            client,
            {
                "name": "Other Relief",
                "stock": created["stock"],
                "artwork": created["artwork"],
            },
        )
        assert other["slug"] == "other-relief"
        missing = _duplicate(client, "00000000-0000-4000-8000-000000000000", "Missing Copy")
        assert missing.status_code == 404
        invalid = _duplicate(client, created["id"], "!!!")
        assert invalid.status_code == 422
        blank = _duplicate(client, created["id"], "   ")
        assert blank.status_code == 422
        extra = client.post(
            f"/api/projects/{created['id']}/duplicate",
            json={"name": "Elsewhere", "slug": "elsewhere"},
        )
        assert extra.status_code == 422
        first = _duplicate(client, created["id"], "Other Relief")
        assert first.status_code == 409
        assert not (project_root / "sample-relief-copy").exists()

        directory = project_root / created["slug"]
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        escape = directory / "escape"
        escape.symlink_to(outside)
        linked = _duplicate(client, created["id"], "Linked Copy")
        assert linked.status_code == 400
        assert not (project_root / "linked-copy").exists()
        assert list(project_root.glob(".duplicating-*")) == []
        escape.unlink()

        monkeypatch.setattr("app.projects.duplicate.MAX_DUPLICATE_FILES", 0)
        limited = _duplicate(client, created["id"], "Too Many")
        assert limited.status_code == 413
        assert "too many files" in limited.json()["detail"]
        assert list(project_root.glob(".duplicating-*")) == []
        monkeypatch.setattr("app.projects.duplicate.MAX_DUPLICATE_FILES", 10_000)
        monkeypatch.setattr("app.projects.duplicate.MAX_DUPLICATE_BYTES", 1)
        oversized = _duplicate(client, created["id"], "Too Large")
        assert oversized.status_code == 413
        assert "too large" in oversized.json()["detail"]


def test_failed_publish_removes_the_staging_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        directory = project_root / created["slug"]
        before = _files(directory)

        def fail_rename(source: str, destination: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("app.projects.duplicate.os.rename", fail_rename)
        response = _duplicate(client, created["id"], "Broken Copy")
        assert response.status_code == 400
        assert not (project_root / "broken-copy").exists()
        assert list(project_root.glob(".duplicating-*")) == []
        assert _files(directory) == before


def test_benchmark_directories_duplicate_only_inside_a_temporary_store(tmp_path: Path) -> None:
    import hashlib

    real = default_projects_dir()
    assert real == Path(__file__).resolve().parents[2] / "projects"
    before = {
        relative: hashlib.sha256((real / relative).read_bytes()).hexdigest()
        for relative in PROTECTED
    }
    project_root = tmp_path / "projects"
    project_root.mkdir()
    for slug in ("wedding-relief-benchmark", "angie-dance"):
        shutil.copytree(real / slug, project_root / slug, symlinks=False)
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        listed = {item["slug"]: item for item in client.get("/api/projects").json()}
        for slug, name in (
            ("wedding-relief-benchmark", "Wedding Relief Recovery Draft"),
            ("angie-dance", "Angie Dance Recovery Draft"),
        ):
            original = listed[slug]
            before_copy = _files(project_root / slug)
            response = _duplicate(client, original["id"], name)
            assert response.status_code == 201, response.text
            body = response.json()
            assert body["source_project_id"] == original["id"]
            assert body["id"] != original["id"]
            assert _files(project_root / slug) == before_copy
            store = ProjectStore(project_root)
            copied = store.get_by_id(UUID(body["id"]))
            statuses = resolve_artifact_statuses(project_root / body["slug"], copied)
            assert statuses == {
                "source": "unknown",
                "raw": "unknown",
                "conditioned": "unknown",
                "relief": "unknown",
                "physical": "missing",
            }
            assert [region.name for region in copied.relief.regions] == [
                region.name for region in store.get_by_id(UUID(original["id"])).relief.regions
            ]
            copied_files = _files(project_root / body["slug"])
            for relative, payload in before_copy.items():
                if relative == "project.json":
                    continue
                assert copied_files[relative] == payload
    after = {
        relative: hashlib.sha256((real / relative).read_bytes()).hexdigest()
        for relative in PROTECTED
    }
    assert after == before == PROTECTED
    assert not (real / "wedding-relief-recovery-draft").exists()
    assert not (real / "angie-dance-recovery-draft").exists()


def test_publish_refuses_to_replace_an_existing_directory(tmp_path: Path) -> None:
    from app.projects.duplicate import _publish_staging
    from app.projects.errors import ProjectConflict

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "project.json").write_text('{"keep": true}\n', encoding="utf-8")
    empty = tmp_path / "empty-dest"
    empty.mkdir()
    occupied = tmp_path / "occupied-dest"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ProjectConflict):
        _publish_staging(staging, empty)
    with pytest.raises(ProjectConflict):
        _publish_staging(staging, occupied)

    assert list(empty.iterdir()) == []
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (staging / "project.json").read_text(encoding="utf-8") == '{"keep": true}\n'


def test_concurrent_same_slug_publishes_one_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from app.main import app
    from app.projects import duplicate as duplicate_module

    project_root = tmp_path / "projects"
    barrier = threading.Barrier(2)
    original = duplicate_module._publish_staging

    def gated(staging: Path, destination: Path) -> None:
        barrier.wait(timeout=5)
        original(staging, destination)

    monkeypatch.setattr(duplicate_module, "_publish_staging", gated)
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        source = project_root / created["slug"]
        (source / "notes.txt").write_text("stable\n", encoding="utf-8")
        before = _files(source)
        results: list[int] = []
        errors: list[BaseException] = []

        def once() -> None:
            try:
                with TestClient(app) as local:
                    response = local.post(
                        f"/api/projects/{created['id']}/duplicate",
                        json={"name": "Race Copy"},
                    )
                    results.append(response.status_code)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=once) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

    assert errors == []
    assert sorted(results) == [201, 409]
    published = [path for path in project_root.iterdir() if path.name != created["slug"]]
    assert [path.name for path in published] == ["race-copy"]
    assert _files(source) == before
    assert (project_root / "race-copy" / "notes.txt").read_text(encoding="utf-8") == "stable\n"
    assert list(project_root.glob(".duplicating-*")) == []


def test_nested_symlink_fifo_and_socket_are_rejected(tmp_path: Path) -> None:
    import socket

    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        directory = project_root / created["slug"]
        before = _files(directory)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        nested = directory / "level-one" / "level-two"
        nested.mkdir(parents=True)
        (nested / "linked").symlink_to(outside, target_is_directory=True)
        linked = _duplicate(client, created["id"], "Nested Link")
        assert linked.status_code == 400
        assert not (project_root / "nested-link").exists()
        assert list(project_root.glob(".duplicating-*")) == []
        (nested / "linked").unlink()

        (nested / "escape").symlink_to(outside / "secret.txt")
        escaped = _duplicate(client, created["id"], "Nested File")
        assert escaped.status_code == 400
        assert not (project_root / "nested-file").exists()
        (nested / "escape").unlink()

        os.mkfifo(directory / "queue")
        fifo = _duplicate(client, created["id"], "Fifo Copy")
        assert fifo.status_code == 400
        assert not (project_root / "fifo-copy").exists()
        (directory / "queue").unlink()

        endpoint = socket.socket(socket.AF_UNIX)
        previous = os.getcwd()
        os.chdir(directory)
        try:
            endpoint.bind("device.sock")
            socket_copy = _duplicate(client, created["id"], "Socket Copy")
            assert socket_copy.status_code == 400
            assert not (project_root / "socket-copy").exists()
        finally:
            os.chdir(previous)
            endpoint.close()
        assert _files(directory) == before
        assert list(project_root.glob(".duplicating-*")) == []
        assert not any((project_root / "nested-link").rglob("secret.txt"))


def test_partial_copy_removes_staging_and_leaves_existing_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        other = create_project(
            client,
            {
                "name": "Other Relief",
                "stock": created["stock"],
                "artwork": created["artwork"],
            },
        )
        source = project_root / created["slug"]
        other_dir = project_root / other["slug"]
        (other_dir / "keep.txt").write_text("keep", encoding="utf-8")
        before = _files(source)
        other_before = _files(other_dir)

        def fail_read(descriptor: int, size: int) -> bytes:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("app.projects.duplicate.os.read", fail_read)
        response = _duplicate(client, created["id"], "Partial Copy")
        assert response.status_code == 400
        assert response.json()["detail"] == "The project could not be duplicated."
        assert not (project_root / "partial-copy").exists()
        assert list(project_root.glob(".duplicating-*")) == []
        assert _files(source) == before
        assert _files(other_dir) == other_before


def test_source_change_during_copy_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.projects import duplicate as duplicate_module

    project_root = tmp_path / "projects"
    original = duplicate_module._copy_regular_file

    def change_after_manifest(root: Path, destination: Path, expected: duplicate_module._FileSnapshot) -> str:
        digest = original(root, destination, expected)
        if expected.relative == Path("project.json"):
            (root / "notes.txt").write_text("changed during copy\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(duplicate_module, "_copy_regular_file", change_after_manifest)
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        created = create_project(client)
        source = project_root / created["slug"]
        (source / "notes.txt").write_text("original\n", encoding="utf-8")
        manifest = (source / "project.json").read_bytes()
        response = _duplicate(client, created["id"], "Changed Copy")
        assert response.status_code == 400
        assert response.json()["detail"] == "The project changed while it was being copied."
        assert not (project_root / "changed-copy").exists()
        assert list(project_root.glob(".duplicating-*")) == []
        assert (source / "project.json").read_bytes() == manifest
