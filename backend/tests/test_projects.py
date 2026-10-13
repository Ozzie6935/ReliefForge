"""Project persistence and source-image tests.

Storage is a temporary directory. These tests must not write into the
repository projects/ folder.
"""

import copy
import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import get_project_service
from app.core.config import MAX_SOURCE_IMAGE_BYTES, default_projects_dir, repository_root
from app.main import app
from app.projects.errors import UnsafeProjectPath
from app.projects.service import ProjectService
from app.projects.storage import ProjectStore, slugify
from app.schemas.project import ProjectCreate

SAMPLE = {
    "name": "Sample Relief",
    "stock": {"width_mm": 100.0, "height_mm": 80.0},
    "artwork": {
        "width_mm": 90.0,
        "height_mm": 70.5,
        "center_x_mm": 50.0,
        "center_y_mm": 40.0,
    },
}


def sample_body() -> dict:
    return copy.deepcopy(SAMPLE)


def image_bytes(image_format: str, size: tuple[int, int] = (12, 7)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format=image_format)
    return buffer.getvalue()


@contextmanager
def served(project_root: Path, max_image_bytes: int) -> Iterator[TestClient]:
    def override() -> ProjectService:
        return ProjectService(
            ProjectStore(project_root),
            max_image_bytes=max_image_bytes,
        )

    app.dependency_overrides[get_project_service] = override
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


@pytest.fixture
def client(project_root: Path) -> Iterator[TestClient]:
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as test_client:
        yield test_client


def create_project(client: TestClient, body: dict | None = None) -> dict:
    response = client.post("/api/projects", json=body if body is not None else sample_body())
    assert response.status_code == 201, response.text
    return response.json()


def test_default_projects_dir_does_not_depend_on_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_projects_dir() == repository_root() / "projects"
    assert default_projects_dir().parent == Path(__file__).resolve().parents[2]


def test_upload_limit_is_fifty_megabytes() -> None:
    assert MAX_SOURCE_IMAGE_BYTES == 50 * 1024 * 1024


def test_slug_generation() -> None:
    assert slugify("Sample Relief") == "sample-relief"
    assert slugify("Hello, World!") == "hello-world"


def test_project_creation_returns_success(client: TestClient) -> None:
    response = client.post("/api/projects", json=sample_body())
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Sample Relief"
    assert body["schema_version"] == 1
    assert body["source"] is None


def test_generated_id_is_a_uuid(client: TestClient) -> None:
    body = create_project(client)
    parsed = UUID(body["id"])
    assert str(parsed) == body["id"]


def test_project_json_is_persisted(client: TestClient, project_root: Path) -> None:
    body = create_project(client)
    manifest = json.loads((project_root / body["slug"] / "project.json").read_text())
    assert manifest["id"] == body["id"]
    assert manifest["name"] == "Sample Relief"
    assert manifest["slug"] == "sample-relief"


def test_persisted_units_are_millimeters(client: TestClient, project_root: Path) -> None:
    body = create_project(client)
    assert body["units"] == "mm"
    manifest = json.loads((project_root / "sample-relief" / "project.json").read_text())
    assert manifest["units"] == "mm"


def test_positive_dimensions_are_stored(client: TestClient) -> None:
    body = create_project(client)
    assert body["stock"] == {"width_mm": 100.0, "height_mm": 80.0, "thickness_mm": 38.1}
    assert body["artwork"]["width_mm"] == 90.0
    assert body["artwork"]["height_mm"] == 70.5


def test_project_can_be_loaded_again_from_disk(client: TestClient) -> None:
    created = create_project(client)
    response = client.get(f"/api/projects/{created['id']}")
    assert response.status_code == 200
    assert response.json() == created


def test_project_reloads_from_a_fresh_service(project_root: Path) -> None:
    created = ProjectService(ProjectStore(project_root)).create_project(
        ProjectCreate.model_validate(sample_body())
    )
    loaded = ProjectService(ProjectStore(project_root)).get_project(created.id)
    assert loaded == created


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("stock", "width_mm", 0),
        ("stock", "height_mm", -1),
        ("artwork", "width_mm", -5),
        ("artwork", "height_mm", 0),
    ],
)
def test_invalid_dimensions_are_rejected(
    client: TestClient,
    project_root: Path,
    section: str,
    field: str,
    value: float,
) -> None:
    body = sample_body()
    body[section][field] = value
    response = client.post("/api/projects", json=body)
    assert response.status_code == 422
    assert list(project_root.glob("*/project.json")) == []


def test_center_is_a_coordinate_not_a_dimension(client: TestClient) -> None:
    body = sample_body()
    body["artwork"]["center_x_mm"] = 0
    body["artwork"]["center_y_mm"] = -2.5
    created = create_project(client, body)
    assert created["artwork"]["center_x_mm"] == 0
    assert created["artwork"]["center_y_mm"] == -2.5


def test_missing_project_returns_404(client: TestClient) -> None:
    response = client.get(f"/api/projects/{uuid4()}")
    assert response.status_code == 404


def test_blank_project_name_is_rejected(client: TestClient, project_root: Path) -> None:
    body = sample_body()
    body["name"] = "   "
    response = client.post("/api/projects", json=body)
    assert response.status_code == 422
    assert list(project_root.glob("*")) == []


def test_valid_png_upload_preserves_bytes_and_pixels(
    client: TestClient,
    project_root: Path,
) -> None:
    created = create_project(client)
    payload = image_bytes("PNG", (12, 7))
    response = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("Photo.PNG", payload, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"]["filename"] == "original.png"
    assert body["source"]["original_filename"] == "Photo.PNG"
    assert body["source"]["media_type"] == "image/png"
    assert body["source"]["width_px"] == 12
    assert body["source"]["height_px"] == 7
    assert body["artwork"]["width_mm"] == 90.0
    stored = (project_root / "sample-relief" / "source" / "original.png").read_bytes()
    assert stored == payload
    manifest = json.loads((project_root / "sample-relief" / "project.json").read_text())
    assert manifest["source"]["width_px"] == 12
    assert manifest["source"]["height_px"] == 7


def test_valid_jpeg_upload(client: TestClient, project_root: Path) -> None:
    created = create_project(client)
    payload = image_bytes("JPEG", (8, 5))
    response = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("scan.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"]["filename"] == "original.jpg"
    assert body["source"]["media_type"] == "image/jpeg"
    assert body["source"]["width_px"] == 8
    assert body["source"]["height_px"] == 5
    stored = (project_root / "sample-relief" / "source" / "original.jpg").read_bytes()
    assert stored == payload


def test_get_source_returns_the_persisted_image(client: TestClient) -> None:
    created = create_project(client)
    payload = image_bytes("PNG")
    uploaded = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("art.png", payload, "image/png")},
    )
    assert uploaded.status_code == 200
    response = client.get(f"/api/projects/{created['id']}/source")
    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("image/png")


def test_missing_source_returns_404(client: TestClient) -> None:
    created = create_project(client)
    response = client.get(f"/api/projects/{created['id']}/source")
    assert response.status_code == 404


def test_fake_jpg_is_rejected(client: TestClient, project_root: Path) -> None:
    created = create_project(client)
    response = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("fake.jpg", b"this is not a jpeg", "image/jpeg")},
    )
    assert response.status_code == 400
    assert not (project_root / "sample-relief" / "source").exists()
    manifest = json.loads((project_root / "sample-relief" / "project.json").read_text())
    assert manifest["source"] is None


def test_corrupt_image_is_rejected(client: TestClient, project_root: Path) -> None:
    created = create_project(client)
    corrupt = image_bytes("JPEG")[:24]
    response = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("broken.jpg", corrupt, "image/jpeg")},
    )
    assert response.status_code == 400
    manifest = json.loads((project_root / "sample-relief" / "project.json").read_text())
    assert manifest["source"] is None


def test_unsupported_image_format_is_rejected(client: TestClient, project_root: Path) -> None:
    created = create_project(client)
    response = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("picture.gif", image_bytes("GIF"), "image/gif")},
    )
    assert response.status_code == 415
    assert not (project_root / "sample-relief" / "source").exists()


def test_source_upload_for_missing_project_returns_404(client: TestClient) -> None:
    response = client.post(
        f"/api/projects/{uuid4()}/source",
        files={"file": ("art.png", image_bytes("PNG"), "image/png")},
    )
    assert response.status_code == 404


def test_png_disguised_as_jpg_is_stored_from_contents(
    client: TestClient,
    project_root: Path,
) -> None:
    created = create_project(client)
    payload = image_bytes("PNG", (4, 3))
    response = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("mislabeled.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"]["filename"] == "original.png"
    assert body["source"]["original_filename"] == "mislabeled.jpg"
    assert body["source"]["width_px"] == 4
    assert body["source"]["height_px"] == 3
    assert (project_root / "sample-relief" / "source" / "original.png").read_bytes() == payload
    assert not (project_root / "sample-relief" / "source" / "original.jpg").exists()


def test_unsafe_project_name_cannot_escape_project_root(
    tmp_path: Path,
    project_root: Path,
    client: TestClient,
) -> None:
    body = sample_body()
    body["name"] = "../../tmp/pwn"
    response = client.post("/api/projects", json=body)
    assert response.status_code == 201, response.text
    assert response.json()["slug"] == "tmp-pwn"
    assert (project_root / "tmp-pwn" / "project.json").is_file()
    assert not (tmp_path / "tmp").exists()
    assert {path.name for path in tmp_path.iterdir()} == {"projects"}

    blank = sample_body()
    blank["name"] = "../.."
    rejected = client.post("/api/projects", json=blank)
    assert rejected.status_code == 422

    store = ProjectStore(project_root)
    with pytest.raises(UnsafeProjectPath):
        store.project_directory("../outside")
    with pytest.raises(UnsafeProjectPath):
        store.project_directory("foo/bar")


def test_symlink_directory_is_not_used_as_project_storage(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    store = ProjectStore(root)
    with pytest.raises(UnsafeProjectPath):
        store.project_directory("linked")
    assert list(outside.iterdir()) == []


def test_slug_collision_does_not_overwrite(client: TestClient, project_root: Path) -> None:
    first = create_project(client)
    other = sample_body()
    other["name"] = "sample-relief"
    other["stock"]["width_mm"] = 10
    response = client.post("/api/projects", json=other)
    assert response.status_code == 409
    manifest = json.loads((project_root / "sample-relief" / "project.json").read_text())
    assert manifest["id"] == first["id"]
    assert manifest["name"] == "Sample Relief"
    assert manifest["stock"]["width_mm"] == 100.0
    assert [path.name for path in project_root.iterdir() if path.is_dir()] == ["sample-relief"]


def test_oversized_upload_is_rejected(project_root: Path) -> None:
    with served(project_root, max_image_bytes=32) as client:
        created = create_project(client)
        response = client.post(
            f"/api/projects/{created['id']}/source",
            files={"file": ("big.png", b"x" * 33, "image/png")},
        )
        assert response.status_code == 413
        manifest = json.loads((project_root / "sample-relief" / "project.json").read_text())
        assert manifest["source"] is None
        assert not (project_root / "sample-relief" / "source").exists()


def test_list_projects_empty(client: TestClient) -> None:
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert response.json() == []


def test_list_projects_returns_persisted_project(
    client: TestClient,
    project_root: Path,
) -> None:
    created = create_project(client)
    response = client.get("/api/projects")
    assert response.status_code == 200
    body = response.json()
    assert body == [created]
    assert str(project_root) not in response.text
    assert "project.json" not in response.text


def test_list_projects_returns_multiple(client: TestClient, project_root: Path) -> None:
    first = create_project(client)
    other = sample_body()
    other["name"] = "Second Panel"
    second = create_project(client, other)
    response = client.get("/api/projects")
    assert response.status_code == 200
    body = response.json()
    assert {item["id"] for item in body} == {first["id"], second["id"]}
    assert str(project_root) not in response.text


def test_malformed_project_does_not_break_listing(
    client: TestClient,
    project_root: Path,
) -> None:
    created = create_project(client)
    broken = project_root / "broken-record"
    broken.mkdir()
    (broken / "project.json").write_text("{not json", encoding="utf-8")
    invalid = project_root / "invalid-record"
    invalid.mkdir()
    (invalid / "project.json").write_text(
        '{"schema_version": 99, "name": "Bad"}',
        encoding="utf-8",
    )
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert response.json() == [created]
    assert str(project_root) not in response.text


def test_tampered_source_filename_cannot_escape(project_root: Path) -> None:
    service = ProjectService(ProjectStore(project_root))
    created = service.create_project(ProjectCreate.model_validate(sample_body()))
    manifest_path = project_root / created.slug / "project.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = {
        "filename": "../../secret.png",
        "original_filename": "secret.png",
        "media_type": "image/png",
        "width_px": 1,
        "height_px": 1,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outside = project_root.parent / "secret.png"
    outside.write_bytes(b"secret")
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        response = client.get(f"/api/projects/{created.id}/source")
    assert response.status_code == 404
    assert outside.read_bytes() == b"secret"


def test_update_keeps_uuid_directory_and_created_at(
    client: TestClient,
    project_root: Path,
) -> None:
    created = create_project(client)
    directory = project_root / "sample-relief"
    body = sample_body()
    body["name"] = "Renamed Relief"
    body["stock"]["width_mm"] = 120.0
    body["artwork"]["height_mm"] = 66.0
    response = client.put(f"/api/projects/{created['id']}", json=body)
    assert response.status_code == 200
    updated = response.json()
    assert updated["id"] == created["id"]
    assert updated["slug"] == "sample-relief"
    assert updated["name"] == "Renamed Relief"
    assert updated["stock"]["width_mm"] == 120.0
    assert updated["artwork"]["height_mm"] == 66.0
    assert updated["created_at"] == created["created_at"]
    assert updated["updated_at"] != created["updated_at"]
    assert directory.is_dir()
    assert (directory / "project.json").is_file()
    assert not (project_root / "renamed-relief").exists()
    reloaded = client.get(f"/api/projects/{created['id']}")
    assert reloaded.status_code == 200
    assert reloaded.json()["name"] == "Renamed Relief"
    assert reloaded.json()["stock"]["width_mm"] == 120.0
    assert reloaded.json()["created_at"] == created["created_at"]


def test_update_rejects_non_positive_dimensions(client: TestClient) -> None:
    created = create_project(client)
    body = sample_body()
    body["stock"]["width_mm"] = 0
    response = client.put(f"/api/projects/{created['id']}", json=body)
    assert response.status_code == 422
    stored = client.get(f"/api/projects/{created['id']}").json()
    assert stored["stock"]["width_mm"] == created["stock"]["width_mm"]


def test_source_upload_updates_updated_at_only(client: TestClient) -> None:
    created = create_project(client)
    response = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("scan.png", image_bytes("PNG"), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["created_at"] == created["created_at"]
    assert body["updated_at"] != created["updated_at"]
    assert body["source"]["original_filename"] == "scan.png"


def test_schema_v1_project_without_timestamps_stays_loadable(
    client: TestClient,
    project_root: Path,
) -> None:
    project_id = str(uuid4())
    legacy = {
        "schema_version": 1,
        "id": project_id,
        "name": "Legacy Panel",
        "slug": "legacy-panel",
        "units": "mm",
        "stock": {"width_mm": 10.0, "height_mm": 8.0},
        "artwork": {
            "width_mm": 9.0,
            "height_mm": 7.0,
            "center_x_mm": 5.0,
            "center_y_mm": 4.0,
        },
        "source": {
            "filename": "original.png",
            "original_filename": "scan.png",
            "media_type": "image/png",
            "width_px": 4,
            "height_px": 3,
        },
    }
    directory = project_root / "legacy-panel"
    directory.mkdir(parents=True)
    manifest = directory / "project.json"
    text = json.dumps(legacy)
    manifest.write_text(text, encoding="utf-8")
    response = client.get(f"/api/projects/{project_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["id"] == project_id
    assert body["name"] == "Legacy Panel"
    assert body["stock"]["width_mm"] == 10.0
    assert body["source"]["original_filename"] == "scan.png"
    assert body["source"]["width_px"] == 4
    assert body["source"]["height_px"] == 3
    assert body["created_at"]
    assert body["updated_at"]
    assert manifest.read_text(encoding="utf-8") == text


def test_projects_are_ordered_by_updated_at_descending(
    client: TestClient,
    project_root: Path,
) -> None:
    older = create_project(client)
    newer_body = sample_body()
    newer_body["name"] = "Newer Panel"
    newer = create_project(client, newer_body)

    def stamp(slug: str, updated_at: str) -> None:
        path = project_root / slug / "project.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["created_at"] = "2020-01-01T00:00:00Z"
        data["updated_at"] = updated_at
        path.write_text(json.dumps(data), encoding="utf-8")

    stamp("sample-relief", "2020-01-02T00:00:00Z")
    stamp("newer-panel", "2024-06-01T00:00:00Z")
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [newer["id"], older["id"]]


def test_delete_project_removes_only_that_directory(
    client: TestClient,
    project_root: Path,
) -> None:
    first = create_project(client)
    other = sample_body()
    other["name"] = "Second Panel"
    second = create_project(client, other)
    response = client.delete(f"/api/projects/{first['id']}")
    assert response.status_code == 204
    assert response.content == b""
    assert not (project_root / "sample-relief").exists()
    assert (project_root / "second-panel" / "project.json").is_file()
    assert client.get(f"/api/projects/{first['id']}").status_code == 404
    assert client.get(f"/api/projects/{second['id']}").status_code == 200


def test_delete_missing_project_returns_404(client: TestClient) -> None:
    response = client.delete(f"/api/projects/{uuid4()}")
    assert response.status_code == 404


def test_delete_rejects_symlink_and_does_not_escape(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("safe", encoding="utf-8")
    (outside / "project.json").write_text("{}", encoding="utf-8")
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    store = ProjectStore(root)
    from app.schemas.project import Artwork, Project, Stock

    project = Project(
        id=uuid4(),
        name="Linked",
        slug="linked",
        stock=Stock(width_mm=10, height_mm=8),
        artwork=Artwork(width_mm=9, height_mm=7, center_x_mm=5, center_y_mm=4),
    )
    with pytest.raises(UnsafeProjectPath):
        store.delete_project(project)
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "safe"


def test_delete_does_not_follow_a_symlink_out_of_the_project(
    client: TestClient,
    project_root: Path,
    tmp_path: Path,
) -> None:
    created = create_project(client)
    outside = tmp_path / "keep.txt"
    outside.write_text("safe", encoding="utf-8")
    (project_root / "sample-relief" / "linked-file").symlink_to(outside)
    response = client.delete(f"/api/projects/{created['id']}")
    assert response.status_code == 204
    assert outside.read_text(encoding="utf-8") == "safe"
    assert not (project_root / "sample-relief").exists()
