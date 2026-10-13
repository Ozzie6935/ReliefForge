"""Validated binary STL export. Temporary projects and synthetic arrays only."""

import json
import struct
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from app.projects.errors import (
    ArtifactDependencyRejected,
    InvalidExportRequest,
    PhysicalHeightNotFound,
)
from app.relief.export_plan import (
    ABOVE_STOCK_DETAIL,
    FILE_MISSING_DETAIL,
    NATIVE_REJECTS_GRID,
    NON_POSITIVE_DETAIL,
    RECORD_MISSING_DETAIL,
    REDUCED_NOT_SMALLER,
    STALE_DETAIL,
    UNKNOWN_DETAIL,
    ExportResourceLimits,
    axis_positions,
    binary_stl_bytes,
    solid_triangle_count,
)
from app.relief.mesh import endpoint_indices
from app.relief.solid import SELF_INTERSECTION_LIMITATION, build_solid_mesh
from app.relief.stl_export import (
    FLOAT32_DEGENERATE_DETAIL,
    FLOAT32_OVERFLOW_DETAIL,
    FLOAT32_QUANTIZATION_DETAIL,
    STL_HEADER_TEXT,
    StlDownload,
    _reject_degenerate_float32,
    _require_indices,
    assert_export_budget,
    generate_binary_stl,
    raise_blocked_export,
    require_float32_coordinates,
    stl_filename,
    stl_header,
)
from tests.test_export_readiness import _assess
from tests.test_physical_diagnostics import _files
from tests.test_projects import create_project, sample_body, served
from tests.test_provenance import _pipeline, _root


def _parse_stl(data: bytes) -> list[tuple[tuple[float, float, float], tuple[tuple[float, ...], ...], int]]:
    """Independent binary STL reader. This does not use the exporter's record dtype."""

    assert isinstance(data, (bytes, bytearray, memoryview))
    raw = bytes(data)
    assert len(raw) >= 84
    count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + 50 * count
    triangles = []
    for index in range(count):
        offset = 84 + index * 50
        values = struct.unpack_from("<12fH", raw, offset)
        normal = values[0:3]
        corners = (values[3:6], values[6:9], values[9:12])
        triangles.append((normal, corners, values[12]))
    return triangles


def _signed_volume(triangles) -> float:
    total = 0.0
    for _normal, (first, second, third), _attr in triangles:
        total += (
            first[0] * (second[1] * third[2] - second[2] * third[1])
            - first[1] * (second[0] * third[2] - second[2] * third[0])
            + first[2] * (second[0] * third[1] - second[1] * third[0])
        )
    return total / 6.0


def _cross(first, second, third) -> tuple[float, float, float]:
    ax, ay, az = second[0] - first[0], second[1] - first[1], second[2] - first[2]
    bx, by, bz = third[0] - first[0], third[1] - first[1], third[2] - first[2]
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _edge_report(triangles) -> tuple[int, int]:
    partners: dict[tuple[bytes, bytes], list[int]] = {}
    for normal, corners, _attr in triangles:
        keys = [struct.pack("<3f", *corner) for corner in corners]
        for start, end in ((0, 1), (1, 2), (2, 0)):
            left, right = keys[start], keys[end]
            sign = 1 if left < right else -1
            pair = (left, right) if left < right else (right, left)
            partners.setdefault(pair, []).append(sign)
    boundary = 0
    inconsistent = 0
    for signs in partners.values():
        if len(signs) != 2:
            boundary += 1
        elif sum(signs) != 0:
            inconsistent += 1
    return boundary, inconsistent


def _export(array: np.ndarray, **kwargs):
    return generate_binary_stl(
        array,
        kwargs.pop("artwork_width_mm", 100.0),
        kwargs.pop("artwork_height_mm", 40.0),
        stock_thickness_mm=kwargs.pop("stock_thickness_mm", 19.0),
        mode=kwargs.pop("mode", "native"),
        export_height_px=kwargs.pop("export_height_px", None),
        export_width_px=kwargs.pop("export_width_px", None),
        limits=kwargs.pop("limits", ExportResourceLimits()),
    )


def _body(document) -> bytes:
    return bytes(document.body)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "projects"


def test_binary_stl_header_count_and_record_size() -> None:
    array = np.full((3, 4), 5, dtype=np.float32)
    original = array.tobytes()
    document = _export(array, artwork_width_mm=30, artwork_height_mm=12, stock_thickness_mm=8)
    raw = _body(document)
    assert raw[:80] == stl_header()
    assert raw.startswith(STL_HEADER_TEXT)
    assert b"\x00" in raw[:80]
    assert STL_HEADER_TEXT.decode("ascii") == raw[: len(STL_HEADER_TEXT)].decode("ascii")
    assert b"202" not in raw[:80]
    count = struct.unpack_from("<I", raw, 80)[0]
    assert count == solid_triangle_count(3, 4)
    assert len(raw) == binary_stl_bytes(count) == 84 + 50 * count
    parsed = _parse_stl(raw)
    assert len(parsed) == count
    assert all(attr == 0 for _normal, _corners, attr in parsed)
    assert array.tobytes() == original
    assert document.self_intersection_checked is True
    assert document.limitation == SELF_INTERSECTION_LIMITATION
    assert "certificate" in STL_HEADER_TEXT.decode("ascii").lower() or "Not a machining" in STL_HEADER_TEXT.decode(
        "ascii"
    )


def test_flat_block_has_outward_normals_closed_shell_and_exact_volume() -> None:
    width, height, thickness = 30.0, 12.0, 5.0
    array = np.full((3, 4), thickness, dtype=np.float32)
    raw = _body(_export(array, artwork_width_mm=width, artwork_height_mm=height, stock_thickness_mm=8))
    parsed = _parse_stl(raw)
    boundary, inconsistent = _edge_report(parsed)
    assert boundary == 0
    assert inconsistent == 0
    assert _signed_volume(parsed) == pytest.approx(width * height * thickness, rel=1e-5)
    xs, ys, zs = [], [], []
    for normal, corners, _attr in parsed:
        computed = _cross(*corners)
        assert sum(axis * stored for axis, stored in zip(computed, normal, strict=True)) > 0
        for corner in corners:
            xs.append(corner[0])
            ys.append(corner[1])
            zs.append(corner[2])
    assert min(xs) == pytest.approx(0)
    assert max(xs) == pytest.approx(width)
    assert min(ys) == pytest.approx(0)
    assert max(ys) == pytest.approx(height)
    assert min(zs) == pytest.approx(0)
    assert max(zs) == pytest.approx(thickness)
    top = [normal for normal, corners, _attr in parsed if min(corner[2] for corner in corners) > 1]
    assert top
    assert all(normal[2] > 0.9 for normal in top)
    bottom = [normal for normal, corners, _attr in parsed if max(corner[2] for corner in corners) < 0.1]
    assert bottom
    assert all(normal[2] < -0.9 for normal in bottom)


def test_native_resolution_is_not_the_preview_budget() -> None:
    source = Path(__file__).resolve().parents[1] / "app" / "relief" / "stl_export.py"
    text = source.read_text(encoding="utf-8")
    assert "plan_preview_grid" not in text
    assert "plan_solid_grid" not in text
    assert "build_solid_mesh" not in text
    assert "DEFAULT_MAX_VERTICES" not in text
    assert "250_000" not in text
    array = np.arange(20, dtype=np.float32).reshape(4, 5) * 0.1 + 2
    document = _export(array, artwork_width_mm=80, artwork_height_mm=50)
    assert (document.grid_height_px, document.grid_width_px) == (4, 5)
    assert document.triangle_count == solid_triangle_count(4, 5)
    parsed = _parse_stl(_body(document))
    found = {(round(corner[0], 4), round(corner[1], 4), round(corner[2], 4)) for _n, corners, _a in parsed for corner in corners}
    for row in range(4):
        for column in range(5):
            point = (column * 80 / 4, row * 50 / 3, float(array[row, column]))
            assert any(
                abs(point[0] - item[0]) < 1e-3 and abs(point[1] - item[1]) < 1e-3 and abs(point[2] - item[2]) < 1e-3
                for item in found
            )
    assert max(item[0] for item in found) == pytest.approx(80)
    assert max(item[1] for item in found) == pytest.approx(50)


def test_reduced_grid_keeps_selected_source_coordinates() -> None:
    source = np.array([2, 3, 4, 5, 6, 7], dtype=np.float32)
    array = np.vstack([source, source])
    document = _export(
        array,
        mode="reduced",
        export_height_px=2,
        export_width_px=4,
        artwork_width_mm=100,
        artwork_height_mm=30,
        stock_thickness_mm=20,
    )
    assert (document.grid_height_px, document.grid_width_px) == (2, 4)
    assert document.triangle_count == solid_triangle_count(2, 4)
    columns = endpoint_indices(6, 4)
    expected_x = axis_positions(columns, 6, 100)
    uniform = np.linspace(0, 100, 4)
    assert list(columns) == [0, 1, 3, 5]
    assert float(expected_x[1]) == pytest.approx(20)
    assert float(uniform[1]) == pytest.approx(100 / 3)
    parsed = _parse_stl(_body(document))
    xs = sorted({round(corner[0], 5) for _n, corners, _a in parsed for corner in corners})
    assert xs == pytest.approx(list(expected_x))
    assert not any(abs(item - float(uniform[1])) < 1 for item in xs)
    zs = {round(corner[2], 5) for _n, corners, _a in parsed for corner in corners}
    assert 3 in zs
    assert 5 in zs
    assert 4 not in zs
    assert 6 not in zs
    assert 0 in zs


def test_sloped_and_asymmetric_heights_stay_on_their_samples() -> None:
    slope = np.array(
        [
            [2, 2, 2],
            [4, 4, 4],
            [6, 6, 6],
            [8, 8, 8],
        ],
        dtype=np.float32,
    )
    parsed = _parse_stl(
        _body(_export(slope, artwork_width_mm=20, artwork_height_mm=30, stock_thickness_mm=12))
    )
    top = [
        corner
        for _normal, corners, _attr in parsed
        for corner in corners
        if corner[2] > 0.5
    ]
    assert any(abs(corner[1]) < 1e-4 and abs(corner[2] - 2) < 1e-4 for corner in top)
    assert any(abs(corner[1] - 30) < 1e-3 and abs(corner[2] - 8) < 1e-4 for corner in top)
    assert _signed_volume(parsed) == pytest.approx(20 * 30 * 5, rel=1e-4)

    peak = np.full((3, 5), 3, dtype=np.float32)
    peak[0, 0] = 9
    parsed = _parse_stl(
        _body(_export(peak, artwork_width_mm=40, artwork_height_mm=16, stock_thickness_mm=12))
    )
    corners = [corner for _normal, points, _attr in parsed for corner in points]
    assert any(abs(corner[0]) < 1e-4 and abs(corner[1]) < 1e-4 and abs(corner[2] - 9) < 1e-4 for corner in corners)
    assert any(abs(corner[0] - 40) < 1e-3 and abs(corner[1] - 16) < 1e-3 and abs(corner[2] - 3) < 1e-4 for corner in corners)
    assert not any(
        abs(corner[0] - 40) < 1e-3 and abs(corner[1] - 16) < 1e-3 and abs(corner[2] - 9) < 1e-3
        for corner in corners
    )


def test_output_is_deterministic_and_float32_safe() -> None:
    array = np.full((3, 5), 4.25, dtype=np.float32)
    array[1, 2] = 6.5
    first = _body(_export(array, artwork_width_mm=90, artwork_height_mm=70.5))
    second = _body(_export(array, artwork_width_mm=90, artwork_height_mm=70.5))
    assert first == second
    parsed = _parse_stl(first)
    assert any(abs(corner[2] - 6.5) < 1e-4 for _n, corners, _a in parsed for corner in corners)
    safe = require_float32_coordinates(np.array([0.0, 12.5, 70.5], dtype=np.float64))
    assert safe.dtype == np.float32
    assert float(safe[2]) == pytest.approx(70.5)
    with pytest.raises(InvalidExportRequest) as overflow:
        require_float32_coordinates(np.array([1e40, 0.0, 1.0]))
    assert str(overflow.value) == FLOAT32_OVERFLOW_DETAIL
    with pytest.raises(InvalidExportRequest) as quantized:
        require_float32_coordinates(np.array([0.1]), absolute_tolerance_mm=0.0, relative_tolerance=0.0)
    assert str(quantized.value) == FLOAT32_QUANTIZATION_DETAIL
    collapsed = np.array(
        [
            [[16777216.0, 0.0, 1.0], [16777216.4, 0.0, 1.0], [16777216.0, 1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    converted = require_float32_coordinates(collapsed)
    with pytest.raises(InvalidExportRequest) as degenerate:
        _reject_degenerate_float32(converted)
    assert str(degenerate.value) == FLOAT32_DEGENERATE_DETAIL
    with pytest.raises(InvalidExportRequest):
        _require_indices(np.array([[0, 0, 1]], dtype=np.int32), 4)


def test_invalid_geometry_and_provenance_do_not_return_an_stl(monkeypatch) -> None:
    array = np.full((3, 3), 4, dtype=np.float32)
    before = array.tobytes()
    with pytest.raises(InvalidExportRequest) as above:
        _export(array, stock_thickness_mm=3)
    assert str(above.value) == ABOVE_STOCK_DETAIL
    with pytest.raises(InvalidExportRequest) as flat:
        _export(np.zeros((3, 3), dtype=np.float32))
    assert str(flat.value) == NON_POSITIVE_DETAIL
    assert array.tobytes() == before

    def refuse(*_args, **_kwargs):
        raise AssertionError("over-limit export allocated a solid")

    monkeypatch.setattr("app.relief.stl_export.build_solid_on_indices", refuse)
    with pytest.raises(InvalidExportRequest) as limited:
        _export(array, limits=ExportResourceLimits(max_triangles=1, max_stl_bytes=10, max_peak_memory_bytes=10))
    assert "not reduced automatically" in str(limited.value)

    stale = _assess(array, provenance_status="stale")
    unknown = _assess(array, provenance_status="unknown")
    missing = _assess(None, file_present=False, provenance_status="missing")
    with pytest.raises(ArtifactDependencyRejected) as stale_error:
        raise_blocked_export(stale)
    assert str(stale_error.value) == STALE_DETAIL
    with pytest.raises(ArtifactDependencyRejected) as unknown_error:
        raise_blocked_export(unknown)
    assert str(unknown_error.value) == UNKNOWN_DETAIL
    with pytest.raises(PhysicalHeightNotFound) as missing_error:
        raise_blocked_export(missing)
    assert str(missing_error.value) == FILE_MISSING_DETAIL


def test_angie_sized_native_budget_does_not_allocate_a_mesh() -> None:
    tracemalloc.start()
    with pytest.raises(InvalidExportRequest) as rejected:
        assert_export_budget(2940, 2140, 2940, 2140, 2940 * 2140 * 4)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert "not reduced automatically" in str(rejected.value)
    assert peak < 5_000_000
    with pytest.raises(InvalidExportRequest) as native_grid:
        _export(np.full((4, 4), 3, dtype=np.float32), mode="native", export_height_px=2, export_width_px=2)
    assert str(native_grid.value) == NATIVE_REJECTS_GRID
    with pytest.raises(InvalidExportRequest) as same:
        _export(
            np.full((4, 5), 3, dtype=np.float32),
            mode="reduced",
            export_height_px=4,
            export_width_px=5,
        )
    assert str(same.value) == REDUCED_NOT_SMALLER


def test_larger_mesh_reports_the_self_intersection_limit_without_a_certificate() -> None:
    array = np.full((8, 8), 3, dtype=np.float32)
    document = _export(array, artwork_width_mm=16, artwork_height_mm=16, stock_thickness_mm=6)
    assert document.triangle_count > 128
    assert document.self_intersection_checked is False
    assert "not a universal manifold" in document.limitation
    download = StlDownload(document=document, filename=stl_filename("sample-relief"))
    headers = download.headers()
    assert headers["X-Relief-Stl-Units"] == "mm"
    assert headers["X-Relief-Export-Certificate"] == "none"
    assert headers["X-Relief-Self-Intersection-Checked"] == "false"
    assert headers["Content-Disposition"] == 'attachment; filename="sample-relief-relief.stl"'
    assert stl_filename("../secret") == "relief-export.stl"
    assert build_solid_mesh(array, 16, 16).triangle_count == document.triangle_count


def test_download_is_validated_and_writes_nothing(project_root: Path) -> None:
    with served(project_root, 8_000_000) as client:
        body = _pipeline(project_root, client)
        project_id = body["id"]
        before = _files(project_root)
        readiness = client.get(f"/api/projects/{project_id}/relief/export/readiness")
        assert readiness.status_code == 200, readiness.text
        estimate = readiness.json()
        assert estimate["ready_for_export"] is True
        assert estimate["provenance_status"] == "current"
        response = client.post(
            f"/api/projects/{project_id}/relief/export/stl",
            json={"mode": "native"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["x-relief-stl-units"] == "mm"
        assert response.headers["x-relief-export-certificate"] == "none"
        assert response.headers["cache-control"] == "no-store"
        disposition = response.headers["content-disposition"]
        assert disposition.startswith("attachment; filename=")
        assert disposition.endswith('-relief.stl"')
        assert "/" not in disposition
        raw = response.content
        assert raw[: len(STL_HEADER_TEXT)] == STL_HEADER_TEXT
        parsed = _parse_stl(raw)
        assert len(parsed) == estimate["estimated_triangles"]
        assert len(raw) == estimate["estimated_stl_bytes"]
        assert _signed_volume(parsed) > 0
        boundary, inconsistent = _edge_report(parsed)
        assert boundary == 0
        assert inconsistent == 0
        assert _files(project_root) == before
        assert list(project_root.rglob("*.stl")) == []
        assert client.get(f"/api/projects/{project_id}/relief/export/stl").status_code == 405

        rejected = client.post(
            f"/api/projects/{project_id}/relief/export/stl",
            json={"mode": "native", "height_px": 2, "width_px": 2},
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == NATIVE_REJECTS_GRID
        assert _files(project_root) == before

        reduced = client.post(
            f"/api/projects/{project_id}/relief/export/stl",
            json={"mode": "reduced", "height_px": 2, "width_px": 2},
        )
        assert reduced.status_code == 200, reduced.text
        assert len(_parse_stl(reduced.content)) == solid_triangle_count(2, 2)
        assert len(reduced.content) < len(raw)
        assert _files(project_root) == before

        stored = json.loads((_root(project_root) / "project.json").read_text(encoding="utf-8"))
        stored["relief"]["physical"]["content_sha256"] = "0" * 64
        (_root(project_root) / "project.json").write_text(json.dumps(stored), encoding="utf-8")
        stale = client.post(
            f"/api/projects/{project_id}/relief/export/stl",
            json={"mode": "native"},
        )
        assert stale.status_code == 409
        assert stale.json()["detail"] == STALE_DETAIL
        assert not stale.content.startswith(STL_HEADER_TEXT)
        assert list(project_root.rglob("*.stl")) == []

        stored = json.loads((_root(project_root) / "project.json").read_text(encoding="utf-8"))
        stored["relief"]["physical"].pop("content_sha256", None)
        stored["relief"]["physical"].pop("settings_sha256", None)
        (_root(project_root) / "project.json").write_text(json.dumps(stored), encoding="utf-8")
        unknown = client.post(
            f"/api/projects/{project_id}/relief/export/stl",
            json={"mode": "native"},
        )
        assert unknown.status_code == 409
        assert unknown.json()["detail"] == UNKNOWN_DETAIL

        height_path = _root(project_root) / "depth/composition/physical/physical_height_mm.npy"
        height_bytes = height_path.read_bytes()
        height_path.unlink()
        absent = client.post(
            f"/api/projects/{project_id}/relief/export/stl",
            json={"mode": "native"},
        )
        assert absent.status_code == 404
        assert absent.json()["detail"] == FILE_MISSING_DETAIL
        height_path.write_bytes(height_bytes)

        plain = sample_body()
        plain["name"] = "Plain STL"
        created = create_project(client, plain)
        missing = client.post(
            f"/api/projects/{created['id']}/relief/export/stl",
            json={"mode": "native"},
        )
        assert missing.status_code == 404
        assert missing.json()["detail"] == RECORD_MISSING_DETAIL
        assert list(project_root.rglob("*.stl")) == []
