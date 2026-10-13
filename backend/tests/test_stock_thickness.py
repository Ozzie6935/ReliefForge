"""Stock thickness default and project override.

These tests use a temporary project directory. They do not read or write
the repository projects folder.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import MAX_SOURCE_IMAGE_BYTES
from app.projects.defaults import DEFAULT_STOCK_THICKNESS_MM
from tests.test_projects import create_project, image_bytes, sample_body, served


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with served(tmp_path / "projects", MAX_SOURCE_IMAGE_BYTES) as test_client:
        yield test_client


def test_new_project_stores_the_application_default(client: TestClient, tmp_path: Path) -> None:
    assert DEFAULT_STOCK_THICKNESS_MM == 38.10
    created = create_project(client)
    assert created["stock"]["thickness_mm"] == 38.10
    manifest = json.loads((tmp_path / "projects" / created["slug"] / "project.json").read_text())
    assert manifest["stock"]["thickness_mm"] == 38.10
    assert manifest["schema_version"] == 1


def test_custom_thickness_is_stored_and_survives_an_update_that_omits_it(
    client: TestClient,
) -> None:
    body = sample_body()
    body["stock"]["thickness_mm"] = 50.80
    created = create_project(client, body)
    assert created["stock"]["thickness_mm"] == 50.80

    renamed = sample_body()
    renamed["name"] = "Renamed Relief"
    response = client.put(f"/api/projects/{created['id']}", json=renamed)
    assert response.status_code == 200
    assert response.json()["stock"]["thickness_mm"] == 50.80
    assert response.json()["stock"]["width_mm"] == 100.0

    reloaded = client.get(f"/api/projects/{created['id']}")
    assert reloaded.json()["stock"]["thickness_mm"] == 50.80


def test_explicit_update_replaces_only_the_project_thickness(client: TestClient) -> None:
    created = create_project(client)
    body = sample_body()
    body["stock"]["thickness_mm"] = 19.05
    response = client.put(f"/api/projects/{created['id']}", json=body)
    assert response.status_code == 200
    assert response.json()["stock"]["thickness_mm"] == 19.05
    assert response.json()["stock"]["width_mm"] == 100.0


def test_legacy_manifest_without_thickness_loads_and_is_not_rewritten(
    client: TestClient,
    tmp_path: Path,
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
    }
    directory = tmp_path / "projects" / "legacy-panel"
    directory.mkdir(parents=True)
    manifest = directory / "project.json"
    text = json.dumps(legacy)
    manifest.write_text(text, encoding="utf-8")

    response = client.get(f"/api/projects/{project_id}")
    assert response.status_code == 200
    assert response.json()["stock"]["thickness_mm"] is None
    assert response.json()["stock"]["width_mm"] == 10.0
    assert manifest.read_text(encoding="utf-8") == text

    listed = client.get("/api/projects")
    assert listed.status_code == 200
    assert manifest.read_text(encoding="utf-8") == text


def test_legacy_source_upload_does_not_insert_the_default_thickness(
    client: TestClient,
    tmp_path: Path,
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
    }
    directory = tmp_path / "projects" / "legacy-panel"
    directory.mkdir(parents=True)
    (directory / "project.json").write_text(json.dumps(legacy), encoding="utf-8")
    response = client.post(
        f"/api/projects/{project_id}/source",
        files={"file": ("scan.png", image_bytes("PNG"), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["stock"]["thickness_mm"] is None
    stored = json.loads((directory / "project.json").read_text(encoding="utf-8"))
    assert "thickness_mm" not in stored["stock"]
    assert stored["stock"]["width_mm"] == 10.0


def test_non_positive_thickness_is_rejected_and_does_not_replace_a_saved_value(
    client: TestClient,
) -> None:
    created = create_project(client, {**sample_body(), "stock": {**sample_body()["stock"], "thickness_mm": 50.8}})
    for value in (0, -1):
        body = sample_body()
        body["stock"]["thickness_mm"] = value
        response = client.put(f"/api/projects/{created['id']}", json=body)
        assert response.status_code == 422
    stored = client.get(f"/api/projects/{created['id']}").json()
    assert stored["stock"]["thickness_mm"] == 50.8
