"""Isolated end-to-end STL export.

The disposable project lives under pytest's temporary directory. Depth and
segmentation models are test doubles. Conditioning, composition, physical
height, readiness, and STL generation use the production services.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.api.deps import get_depth_service, get_relief_service
from app.core.config import MAX_SOURCE_IMAGE_BYTES
from app.depth.service import DepthService
from app.main import app
from app.projects.provenance import file_sha256
from app.projects.storage import ProjectStore
from app.relief.export_plan import ExportResourceLimits
from app.relief.service import ReliefService
from app.relief.stl_export import STL_HEADER_TEXT
from tests.test_physical_height import _compose
from tests.test_projects import create_project, image_bytes, served
from tests.test_stl_export import _edge_report, _parse_stl, _signed_volume

ARTWORK_WIDTH_MM = 90.0
ARTWORK_HEIGHT_MM = 40.0
STOCK_WIDTH_MM = 120.0
STOCK_HEIGHT_MM = 70.0
STOCK_THICKNESS_MM = 12.0
BASE_MM = 2.0
PEAK_MM = 8.0
SOURCE_HEIGHT = 4
SOURCE_WIDTH = 6
REDUCED_HEIGHT = 3
REDUCED_WIDTH = 4

PROJECT_BODY = {
    "name": "Disposable Export",
    "stock": {"width_mm": STOCK_WIDTH_MM, "height_mm": STOCK_HEIGHT_MM},
    "artwork": {
        "width_mm": ARTWORK_WIDTH_MM,
        "height_mm": ARTWORK_HEIGHT_MM,
        "center_x_mm": 60.0,
        "center_y_mm": 35.0,
    },
}
PHYSICAL_BODY = {
    "stock_thickness_mm": STOCK_THICKNESS_MM,
    "base_thickness_mm": BASE_MM,
    "maximum_surface_height_mm": PEAK_MM,
}


class AsymmetricDepth:
    """Mostly far pixels, with a raised cluster on the right. This is not a production model."""

    model_id = "depth-anything-v2-small"
    model_version = "e2e"
    input_size = 518
    device = "cpu"

    def load(self) -> None:
        return None

    def infer(self, image: Image.Image) -> np.ndarray:
        if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
            raise RuntimeError("disposable artwork must stay 6 by 4 pixels")
        raw = np.zeros((SOURCE_HEIGHT, SOURCE_WIDTH), dtype=np.float32)
        raw[1, 4] = np.float32(0.4)
        raw[1, 5] = np.float32(1.0)
        raw[2, 5] = np.float32(0.55)
        return raw


def _triangle_count(height_px: int, width_px: int) -> int:
    return (
        4 * (height_px - 1) * (width_px - 1)
        + 4 * (width_px - 1)
        + 4 * (height_px - 1)
    )


def _stl_bytes(triangle_count: int) -> int:
    return 84 + 50 * triangle_count


def _near(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-4, 1e-5 * max(abs(actual), abs(expected)))


def _selected_indices(length: int, count: int) -> list[int]:
    if count == length:
        return list(range(length))
    return [index * (length - 1) // (count - 1) for index in range(count)]


def _positions(indices: list[int], source_length: int, extent_mm: float) -> list[float]:
    coordinates = [index * extent_mm / (source_length - 1) for index in indices]
    coordinates[0] = 0.0
    coordinates[-1] = float(extent_mm)
    return coordinates


def _project_dir(project_root: Path) -> Path:
    return project_root / "disposable-export"


def _height_path(project_root: Path) -> Path:
    return _project_dir(project_root) / "depth" / "composition" / "physical" / "physical_height_mm.npy"


def _manifest_path(project_root: Path) -> Path:
    return _project_dir(project_root) / "project.json"


def _write_manifest(project_root: Path, payload: dict) -> None:
    _manifest_path(project_root).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _establish(project_root: Path, client) -> tuple[dict, np.ndarray]:
    store = ProjectStore(project_root)
    app.dependency_overrides[get_depth_service] = lambda: DepthService(store, AsymmetricDepth())
    app.dependency_overrides[get_relief_service] = lambda: ReliefService(store)
    created = create_project(client, PROJECT_BODY)
    upload = client.post(
        f"/api/projects/{created['id']}/source",
        files={"file": ("scan.png", image_bytes("PNG", (SOURCE_WIDTH, SOURCE_HEIGHT)), "image/png")},
    )
    assert upload.status_code == 200, upload.text
    generated = client.post(f"/api/projects/{created['id']}/depth")
    assert generated.status_code == 200, generated.text
    conditioned = client.post(f"/api/projects/{created['id']}/depth/conditioning", json={})
    assert conditioned.status_code == 200, conditioned.text
    composed = _compose(client, created["id"])
    assert composed["relief"]["status"] == "current"
    physical = client.post(
        f"/api/projects/{created['id']}/relief/physical-height",
        json=PHYSICAL_BODY,
    )
    assert physical.status_code == 200, physical.text
    body = physical.json()
    assert body["relief"]["physical"]["status"] == "current"
    stored = json.loads(_manifest_path(project_root).read_text(encoding="utf-8"))
    assert "status" not in stored["relief"]["physical"]
    height = np.load(_height_path(project_root), allow_pickle=False)
    assert height.shape == (SOURCE_HEIGHT, SOURCE_WIDTH)
    assert height.dtype == np.float32
    assert _near(float(height.min()), BASE_MM)
    assert _near(float(height.max()), PEAK_MM)
    assert float(height.max()) < STOCK_THICKNESS_MM
    assert int(np.count_nonzero(np.isclose(height, BASE_MM))) > height.size // 2
    assert not np.array_equal(height, np.fliplr(height))
    assert not np.array_equal(height, np.flipud(height))
    return body, height


def _readiness(client, project_id: str, **params) -> dict:
    response = client.get(f"/api/projects/{project_id}/relief/export/readiness", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _post_stl(client, project_id: str, body: dict):
    return client.post(f"/api/projects/{project_id}/relief/export/stl", json=body)


def _assert_readiness(report: dict, *, mode: str, height_px: int, width_px: int, height: np.ndarray) -> None:
    triangles = _triangle_count(height_px, width_px)
    assert report["provenance_status"] == "current"
    assert report["ready_for_export"] is True
    assert report["mode"] == mode
    assert report["source_grid"] == {"height_px": SOURCE_HEIGHT, "width_px": SOURCE_WIDTH}
    assert report["planned_grid"] == {"height_px": height_px, "width_px": width_px}
    assert report["artwork_width_mm"] == ARTWORK_WIDTH_MM
    assert report["artwork_height_mm"] == ARTWORK_HEIGHT_MM
    assert report["stock_width_mm"] == STOCK_WIDTH_MM
    assert report["stock_height_mm"] == STOCK_HEIGHT_MM
    assert report["stock_thickness_mm"] == STOCK_THICKNESS_MM
    assert report["base_thickness_mm"] == BASE_MM
    assert report["maximum_surface_height_mm"] == PEAK_MM
    assert report["placement"]["applied"] is False
    assert report["estimated_triangles"] == triangles
    assert report["estimated_stl_bytes"] == _stl_bytes(triangles)
    assert _near(report["z_minimum_mm"], float(height.min()))
    assert _near(report["z_maximum_mm"], float(height.max()))
    assert report["blocking_errors"] == []


def _assert_stl(payload: bytes, headers, height: np.ndarray, export_height: int, export_width: int) -> None:
    assert headers["x-relief-stl-units"] == "mm"
    assert headers["x-relief-export-certificate"] == "none"
    assert headers["content-disposition"] == 'attachment; filename="disposable-export-relief.stl"'
    assert len(payload) >= 84
    count = int.from_bytes(payload[80:84], "little")
    assert len(payload) == _stl_bytes(count)
    assert count == _triangle_count(export_height, export_width)
    assert payload[: len(STL_HEADER_TEXT)] == STL_HEADER_TEXT
    assert payload[len(STL_HEADER_TEXT) : 80] == b"\0" * (80 - len(STL_HEADER_TEXT))
    triangles = _parse_stl(payload)
    assert len(triangles) == count
    rows = _selected_indices(SOURCE_HEIGHT, export_height)
    columns = _selected_indices(SOURCE_WIDTH, export_width)
    xs = _positions(columns, SOURCE_WIDTH, ARTWORK_WIDTH_MM)
    ys = _positions(rows, SOURCE_HEIGHT, ARTWORK_HEIGHT_MM)
    sampled = height[np.ix_(rows, columns)]
    seen = {(row, column): False for row in range(export_height) for column in range(export_width)}
    for normal, corners, attribute in triangles:
        assert attribute == 0
        assert all(math.isfinite(value) for value in normal)
        geometric = _cross(corners[0], corners[1], corners[2])
        assert sum(left * right for left, right in zip(normal, geometric, strict=True)) > 0
        for corner in corners:
            assert all(math.isfinite(value) for value in corner)
            column = next(index for index, value in enumerate(xs) if _near(corner[0], value))
            row = next(index for index, value in enumerate(ys) if _near(corner[1], value))
            assert _near(corner[2], 0.0) or _near(corner[2], float(sampled[row, column]))
            if _near(corner[2], float(sampled[row, column])):
                seen[(row, column)] = True
    assert all(seen.values())
    zs = [corner[2] for _normal, corners, _attr in triangles for corner in corners]
    assert _near(min(zs), 0.0)
    assert _near(max(zs), float(sampled.max()))
    assert max(corner[0] for _n, corners, _a in triangles for corner in corners) <= ARTWORK_WIDTH_MM + 1e-4
    assert max(corner[1] for _n, corners, _a in triangles for corner in corners) <= ARTWORK_HEIGHT_MM + 1e-4
    boundary, inconsistent = _edge_report(triangles)
    assert boundary == 0
    assert inconsistent == 0
    assert _signed_volume(triangles) > 0
    if (export_height, export_width) != (SOURCE_HEIGHT, SOURCE_WIDTH):
        assert xs == [0.0, 18.0, 54.0, 90.0]
        assert ys[0] == 0.0 and ys[-1] == ARTWORK_HEIGHT_MM
        assert _near(ys[1], 40.0 / 3.0)
        steps = [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]
        assert steps != [steps[0]] * len(steps)
        selected_z = {round(float(value), 5) for value in sampled.ravel()}
        omitted = [
            round(float(value), 5)
            for value in height.ravel()
            if round(float(value), 5) not in selected_z
        ]
        assert omitted
        present = {round(corner[2], 5) for _n, corners, _a in triangles for corner in corners}
        assert 0.0 in present
        for value in omitted:
            assert value not in present


def _cross(first, second, third) -> tuple[float, float, float]:
    ax, ay, az = second[0] - first[0], second[1] - first[1], second[2] - first[2]
    bx, by, bz = third[0] - first[0], third[1] - first[1], third[2] - first[2]
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _assert_rejected(response, status: int, detail: str | None = None) -> None:
    assert response.status_code == status, response.text
    assert "octet-stream" not in response.headers.get("content-type", "")
    if detail is not None:
        body = response.json()["detail"]
        text = body if isinstance(body, str) else json.dumps(body)
        assert detail in text
    assert not response.content.startswith(STL_HEADER_TEXT)


def test_dev_origin_can_read_export_headers() -> None:
    from fastapi.testclient import TestClient

    response = TestClient(app).get("/api/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert response.status_code == 200
    exposed = response.headers["access-control-expose-headers"]
    for name in (
        "Content-Disposition",
        "X-Relief-Stl-Units",
        "X-Relief-Export-Certificate",
        "X-Relief-Self-Intersection-Checked",
        "X-Relief-Topology-Limitation",
    ):
        assert name in exposed


def test_disposable_native_and_reduced_stl(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project, height = _establish(project_root, client)
        project_id = project["id"]
        native = _readiness(client, project_id, mode="native")
        _assert_readiness(native, mode="native", height_px=SOURCE_HEIGHT, width_px=SOURCE_WIDTH, height=height)
        assert native["spacing_mm"]["uniform"] is True
        reduced = _readiness(
            client,
            project_id,
            mode="reduced",
            height_px=REDUCED_HEIGHT,
            width_px=REDUCED_WIDTH,
        )
        _assert_readiness(
            reduced,
            mode="reduced",
            height_px=REDUCED_HEIGHT,
            width_px=REDUCED_WIDTH,
            height=height,
        )
        assert reduced["spacing_mm"]["uniform"] is False
        native_stl = _post_stl(client, project_id, {"mode": "native"})
        assert native_stl.status_code == 200, native_stl.text
        _assert_stl(native_stl.content, native_stl.headers, height, SOURCE_HEIGHT, SOURCE_WIDTH)
        reduced_stl = _post_stl(
            client,
            project_id,
            {"mode": "reduced", "height_px": REDUCED_HEIGHT, "width_px": REDUCED_WIDTH},
        )
        assert reduced_stl.status_code == 200, reduced_stl.text
        _assert_stl(reduced_stl.content, reduced_stl.headers, height, REDUCED_HEIGHT, REDUCED_WIDTH)
        assert list(project_root.rglob("*.stl")) == []


def test_failed_exports_do_not_return_an_stl(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project, height = _establish(project_root, client)
        project_id = project["id"]
        ready = _post_stl(client, project_id, {"mode": "native"})
        assert ready.status_code == 200

        equal = _post_stl(
            client,
            project_id,
            {"mode": "reduced", "height_px": SOURCE_HEIGHT, "width_px": SOURCE_WIDTH},
        )
        _assert_rejected(equal, 422, "smaller than the saved physical-height array")
        native_grid = _post_stl(
            client,
            project_id,
            {"mode": "native", "height_px": REDUCED_HEIGHT, "width_px": REDUCED_WIDTH},
        )
        _assert_rejected(native_grid, 422, "does not accept height_px")
        tiny = _post_stl(client, project_id, {"mode": "reduced", "height_px": 1, "width_px": 4})
        _assert_rejected(tiny, 422)

        manifest = json.loads(_manifest_path(project_root).read_text(encoding="utf-8"))
        original = json.loads(json.dumps(manifest))
        manifest["relief"]["physical"]["content_sha256"] = "0" * 64
        _write_manifest(project_root, manifest)
        stale = _post_stl(client, project_id, {"mode": "native"})
        _assert_rejected(stale, 409, "stale")
        stale_plan = _readiness(client, project_id, mode="native")
        assert stale_plan["provenance_status"] == "stale"
        assert stale_plan["ready_for_export"] is False

        unknown_manifest = json.loads(json.dumps(original))
        unknown_manifest["relief"]["physical"].pop("content_sha256")
        _write_manifest(project_root, unknown_manifest)
        unknown = _post_stl(client, project_id, {"mode": "native"})
        _assert_rejected(unknown, 409, "unknown")
        unknown_plan = _readiness(client, project_id, mode="native")
        assert unknown_plan["provenance_status"] == "unknown"
        assert unknown_plan["ready_for_export"] is False

        _write_manifest(project_root, original)
        raised = np.array(height, copy=True)
        raised[1, 5] = np.float32(STOCK_THICKNESS_MM + 1.0)
        np.save(_height_path(project_root), raised)
        digest = file_sha256(_height_path(project_root))
        assert digest is not None
        above = json.loads(json.dumps(original))
        above["relief"]["physical"]["content_sha256"] = digest
        _write_manifest(project_root, above)
        above_plan = _readiness(client, project_id, mode="native")
        assert above_plan["provenance_status"] == "current"
        assert above_plan["ready_for_export"] is False
        assert any(issue["code"] == "above_stock" for issue in above_plan["blocking_errors"])
        above_stl = _post_stl(client, project_id, {"mode": "native"})
        _assert_rejected(above_stl, 422, "exceeds the recorded stock thickness")

        missing = create_project(client, {**PROJECT_BODY, "name": "Disposable Missing"})
        missing_plan = _readiness(client, missing["id"], mode="native")
        assert missing_plan["provenance_status"] == "missing"
        assert missing_plan["ready_for_export"] is False
        missing_stl = _post_stl(client, missing["id"], {"mode": "native"})
        _assert_rejected(missing_stl, 404, "Physical height has not been generated.")
        assert list(project_root.rglob("*.stl")) == []


def test_readiness_rejection_does_not_build_a_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "projects"
    with served(project_root, MAX_SOURCE_IMAGE_BYTES) as client:
        project, _height = _establish(project_root, client)
        project_id = project["id"]
        calls = {"mesh": 0}

        def refuse_mesh(*_args, **_kwargs):
            calls["mesh"] += 1
            raise AssertionError("a readiness rejection allocated a solid mesh")

        monkeypatch.setattr("app.relief.stl_export.build_solid_on_indices", refuse_mesh)
        from app.relief import export_plan

        def tight_plan(*args, **kwargs):
            kwargs["limits"] = ExportResourceLimits(max_triangles=10)
            return export_plan.build_export_readiness(*args, **kwargs)

        monkeypatch.setattr("app.relief.service.build_export_readiness", tight_plan)
        limited = _readiness(client, project_id, mode="native")
        assert limited["provenance_status"] == "current"
        assert limited["ready_for_export"] is False
        assert any(issue["code"] == "triangle_limit" for issue in limited["blocking_errors"])
        rejected = _post_stl(client, project_id, {"mode": "native"})
        _assert_rejected(rejected, 422, "triangle count")
        reduced = _post_stl(
            client,
            project_id,
            {"mode": "reduced", "height_px": REDUCED_HEIGHT, "width_px": REDUCED_WIDTH},
        )
        _assert_rejected(reduced, 422, "triangle count")

        manifest = json.loads(_manifest_path(project_root).read_text(encoding="utf-8"))
        manifest["relief"]["physical"]["content_sha256"] = "0" * 64
        _write_manifest(project_root, manifest)
        stale = _post_stl(client, project_id, {"mode": "native"})
        _assert_rejected(stale, 409, "stale")
        assert calls["mesh"] == 0
        assert list(project_root.rglob("*.stl")) == []
