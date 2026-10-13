"""Map a composed relief to finished-surface height in millimeters.

This does not change the source image, depth, masks, proposals, or the
composed relief array.
"""

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from app.projects.errors import InvalidPhysicalHeight, UnsafeProjectPath
from app.relief.artifacts import atomic_replace_bytes, composition_directory
from app.schemas.project import Project
from app.schemas.relief import PhysicalHeightRecord, ReliefRecord

PHYSICAL_DIRECTORY = "physical"
HEIGHT_FILENAME = "physical_height_mm.npy"
PREVIEW_FILENAME = "preview.png"
STATS_FILENAME = "stats.json"
HEIGHT_PATH = "depth/composition/physical/physical_height_mm.npy"
PREVIEW_PATH = "depth/composition/physical/preview.png"
STATS_PATH = "depth/composition/physical/stats.json"
RELIEF_FILENAME = "relief_depth.npy"

PhysicalStatus = Literal["missing", "unknown", "current", "stale"]


def validate_physical_settings(stock_mm: float, base_mm: float, peak_mm: float) -> None:
    """Require finite thicknesses with 0 < base < peak <= stock."""

    for value in (stock_mm, base_mm, peak_mm):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidPhysicalHeight("parameters")
        if not np.isfinite(value):
            raise InvalidPhysicalHeight("parameters")
    if not (0 < float(base_mm) < float(peak_mm) <= float(stock_mm)):
        raise InvalidPhysicalHeight("parameters")


def validate_relief_array(relief: np.ndarray) -> None:
    """Accept only a finite 2D float32 array in [0, 1]. No clipping."""

    if not isinstance(relief, np.ndarray):
        raise InvalidPhysicalHeight("relief")
    if relief.dtype != np.float32 or relief.ndim != 2 or relief.size == 0:
        raise InvalidPhysicalHeight("relief")
    if not np.isfinite(relief).all():
        raise InvalidPhysicalHeight("relief")
    lowest = float(np.min(relief))
    highest = float(np.max(relief))
    if lowest < 0.0 or highest > 1.0:
        raise InvalidPhysicalHeight("relief")


def map_physical_height(relief: np.ndarray, base_mm: float, peak_mm: float) -> np.ndarray:
    """H = B + R * (P - B), measured up from the stock bottom."""

    validate_relief_array(relief)
    if (
        isinstance(base_mm, bool)
        or isinstance(peak_mm, bool)
        or not isinstance(base_mm, (int, float))
        or not isinstance(peak_mm, (int, float))
        or not np.isfinite(base_mm)
        or not np.isfinite(peak_mm)
        or not (0 < float(base_mm) < float(peak_mm))
    ):
        raise InvalidPhysicalHeight("parameters")
    base = np.float32(base_mm)
    peak = np.float32(peak_mm)
    return base + relief * (peak - base)


def cut_depth_mm(stock_mm: float, height: np.ndarray) -> np.ndarray:
    """Distance from the stock top down to the finished surface."""

    if height.dtype != np.float32:
        raise InvalidPhysicalHeight("relief")
    return np.float32(stock_mm) - height


def maximum_cut_depth_mm(stock_mm: float, base_mm: float) -> float:
    return float(stock_mm) - float(base_mm)


def relief_amplitude_mm(peak_mm: float, base_mm: float) -> float:
    return float(peak_mm) - float(base_mm)


def fixed_preview_png(height: np.ndarray, base_mm: float, peak_mm: float) -> bytes:
    """8-bit grayscale on a fixed base-to-peak scale. No auto contrast."""

    if height.dtype != np.float32 or height.ndim != 2:
        raise InvalidPhysicalHeight("relief")
    base = np.float32(base_mm)
    peak = np.float32(peak_mm)
    amplitude = peak - base
    if not np.isfinite(amplitude) or float(amplitude) <= 0.0:
        raise InvalidPhysicalHeight("parameters")
    scaled = (height - base) / amplitude
    if not np.isfinite(scaled).all():
        raise InvalidPhysicalHeight("relief")
    if float(np.min(scaled)) < 0.0 or float(np.max(scaled)) > 1.0:
        raise InvalidPhysicalHeight("relief")
    rendered = np.rint(scaled * np.float32(255))
    if not np.isfinite(rendered).all():
        raise InvalidPhysicalHeight("relief")
    if float(np.min(rendered)) < 0.0 or float(np.max(rendered)) > 255.0:
        raise InvalidPhysicalHeight("relief")
    image = Image.fromarray(rendered.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OSError(path)
    return sha256_bytes(path.read_bytes())


def load_relief_array(data: bytes) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(data))
    except Exception as exc:
        raise InvalidPhysicalHeight("relief") from exc
    validate_relief_array(array)
    return array


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def physical_directory(project_directory: Path) -> Path:
    composition = composition_directory(project_directory)
    folder = composition / PHYSICAL_DIRECTORY
    if folder.is_symlink():
        raise UnsafeProjectPath
    return folder


def relief_file(project_directory: Path) -> Path:
    path = composition_directory(project_directory) / RELIEF_FILENAME
    if path.is_symlink():
        raise UnsafeProjectPath
    return path


def snapshot_physical(project_directory: Path) -> dict[str, bytes | None]:
    folder = physical_directory(project_directory)
    snapshot: dict[str, bytes | None] = {}
    for name in (HEIGHT_FILENAME, PREVIEW_FILENAME, STATS_FILENAME):
        path = folder / name
        if path.is_symlink():
            raise UnsafeProjectPath
        snapshot[name] = path.read_bytes() if path.is_file() else None
    return snapshot


def restore_physical(project_directory: Path, snapshot: dict[str, bytes | None]) -> None:
    folder = physical_directory(project_directory)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in snapshot.items():
        path = folder / name
        if path.is_symlink():
            raise UnsafeProjectPath
        partials = (
            folder / f".{name}.partial",
            folder / ".physical_height_mm.partial.npy",
            folder / ".preview.partial.png",
            folder / ".stats.partial.json",
        )
        for partial in partials:
            if partial.is_file() and not partial.is_symlink():
                partial.unlink()
        if data is None:
            if path.is_file():
                path.unlink()
            continue
        atomic_replace_bytes(path, data)


def write_physical_bundle(
    project_directory: Path,
    height: np.ndarray,
    preview_png: bytes,
    stats_text: str,
) -> None:
    """Write the three physical files, or put the previous files back."""

    if height.dtype != np.float32 or height.ndim != 2 or not np.isfinite(height).all():
        raise InvalidPhysicalHeight("relief")
    snapshot = snapshot_physical(project_directory)
    folder = physical_directory(project_directory)
    folder.mkdir(parents=True, exist_ok=True)
    height_path = folder / HEIGHT_FILENAME
    preview_path = folder / PREVIEW_FILENAME
    stats_path = folder / STATS_FILENAME
    temps = (
        folder / ".physical_height_mm.partial.npy",
        folder / ".preview.partial.png",
        folder / ".stats.partial.json",
    )
    try:
        for path in (height_path, preview_path, stats_path, *temps):
            if path.is_symlink():
                raise UnsafeProjectPath
        temps[0].write_bytes(npy_bytes(height))
        temps[1].write_bytes(preview_png)
        temps[2].write_text(stats_text, encoding="utf-8")
        temps[0].replace(height_path)
        temps[1].replace(preview_path)
        temps[2].replace(stats_path)
        for path in (height_path, preview_path, stats_path):
            if not path.resolve().is_relative_to(project_directory.resolve()):
                raise UnsafeProjectPath
    except Exception:
        for temp in temps:
            if temp.is_file() and not temp.is_symlink():
                temp.unlink()
        restore_physical(project_directory, snapshot)
        raise


def build_physical_record(
    relief: ReliefRecord,
    relief_sha256: str,
    height: np.ndarray,
    stock_mm: float,
    base_mm: float,
    peak_mm: float,
    generated_at: datetime,
) -> PhysicalHeightRecord:
    if relief.generated_at is None:
        raise InvalidPhysicalHeight("relief")
    return PhysicalHeightRecord(
        stock_thickness_mm=float(stock_mm),
        base_thickness_mm=float(base_mm),
        maximum_surface_height_mm=float(peak_mm),
        generated_at=generated_at,
        height_path=HEIGHT_PATH,
        preview_path=PREVIEW_PATH,
        stats_path=STATS_PATH,
        source_relief_sha256=relief_sha256,
        source_relief_generated_at=relief.generated_at,
        width_px=int(height.shape[1]),
        height_px=int(height.shape[0]),
        dtype="float32",
        height_min_mm=float(np.min(height)),
        height_max_mm=float(np.max(height)),
        maximum_cut_depth_mm=maximum_cut_depth_mm(stock_mm, base_mm),
        relief_amplitude_mm=relief_amplitude_mm(peak_mm, base_mm),
    )


def stats_text(record: PhysicalHeightRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"status"})
    return json.dumps(payload, indent=2) + "\n"


def resolve_physical_status(directory: Path, project: Project) -> PhysicalStatus:
    from app.projects.provenance import resolve_artifact_statuses

    return resolve_artifact_statuses(directory, project)["physical"]


def attach_physical_status(directory: Path, project: Project) -> Project:
    from app.projects.provenance import attach_artifact_status

    return attach_artifact_status(directory, project)


def _drop_status(value: object) -> None:
    if isinstance(value, dict):
        value.pop("status", None)
        for child in value.values():
            _drop_status(child)
    elif isinstance(value, list):
        for child in value:
            _drop_status(child)


def manifest_payload(project: Project) -> dict[str, object]:
    """Drop response-only status and an empty physical record before saving."""

    payload = project.model_dump(mode="json")
    _drop_status(payload)
    relief = payload.get("relief")
    if isinstance(relief, dict):
        physical = relief.get("physical")
        if physical is None:
            relief.pop("physical", None)
    stock = payload.get("stock")
    if isinstance(stock, dict) and stock.get("thickness_mm") is None:
        stock.pop("thickness_mm", None)
    return payload
