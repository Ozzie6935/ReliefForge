"""Persist raw depth and its preview inside a project directory."""

from datetime import datetime
from pathlib import Path

import numpy as np

from app.projects.errors import (
    ConditionedPreviewNotFound,
    DepthArtifactNotFound,
    RawDepthNotFound,
    UnsafeProjectPath,
)
from app.schemas.project import DepthArtifact, Project

RAW_DEPTH_FILENAME = "raw_depth.npy"
PREVIEW_FILENAME = "preview.png"
PROCESSING_DIRECTORY = "processing"
CONDITIONED_FILENAME = "conditioned_depth.npy"
CONDITIONED_PREVIEW_FILENAME = "preview.png"


def write_depth_files(directory: Path, depth: np.ndarray, preview_png: bytes) -> None:
    if depth.dtype != np.float32 or depth.ndim != 2:
        raise UnsafeProjectPath
    if not np.isfinite(depth).all():
        raise UnsafeProjectPath
    depth_dir = directory / "depth"
    if depth_dir.is_symlink():
        raise UnsafeProjectPath
    depth_dir.mkdir(exist_ok=True)
    raw_path = depth_dir / RAW_DEPTH_FILENAME
    preview_path = depth_dir / PREVIEW_FILENAME
    if raw_path.is_symlink() or preview_path.is_symlink():
        raise UnsafeProjectPath
    raw_temporary = depth_dir / ".raw_depth.partial.npy"
    preview_temporary = depth_dir / ".preview.partial.png"
    try:
        np.save(raw_temporary, depth)
        preview_temporary.write_bytes(preview_png)
        raw_temporary.replace(raw_path)
        preview_temporary.replace(preview_path)
    except Exception:
        raw_temporary.unlink(missing_ok=True)
        preview_temporary.unlink(missing_ok=True)
        raise
    if not raw_path.resolve().is_relative_to(directory.resolve()):
        raise UnsafeProjectPath


def write_conditioned_files(
    directory: Path,
    depth: np.ndarray,
    preview_png: bytes,
) -> None:
    """Write depth/processing artifacts. Raw depth and its preview stay untouched."""

    if depth.dtype != np.float32 or depth.ndim != 2:
        raise UnsafeProjectPath
    if not np.isfinite(depth).all():
        raise UnsafeProjectPath
    depth_dir = directory / "depth"
    if depth_dir.is_symlink() or not depth_dir.is_dir():
        raise RawDepthNotFound
    processing = depth_dir / PROCESSING_DIRECTORY
    if processing.is_symlink():
        raise UnsafeProjectPath
    processing.mkdir(exist_ok=True)
    conditioned_path = processing / CONDITIONED_FILENAME
    preview_path = processing / CONDITIONED_PREVIEW_FILENAME
    if conditioned_path.is_symlink() or preview_path.is_symlink():
        raise UnsafeProjectPath
    conditioned_temporary = processing / ".conditioned_depth.partial.npy"
    preview_temporary = processing / ".preview.partial.png"
    try:
        np.save(conditioned_temporary, depth)
        preview_temporary.write_bytes(preview_png)
        conditioned_temporary.replace(conditioned_path)
        preview_temporary.replace(preview_path)
    except Exception:
        conditioned_temporary.unlink(missing_ok=True)
        preview_temporary.unlink(missing_ok=True)
        raise
    for path in (conditioned_path, preview_path):
        if not path.resolve().is_relative_to(directory.resolve()):
            raise UnsafeProjectPath


def raw_depth_file(directory: Path, project: Project) -> Path:
    artifact = project.depth
    if artifact is None or artifact.raw_filename != RAW_DEPTH_FILENAME:
        raise RawDepthNotFound
    path = directory / "depth" / artifact.raw_filename
    if path.is_symlink() or not path.is_file():
        raise RawDepthNotFound
    if not path.resolve().is_relative_to(directory.resolve()):
        raise RawDepthNotFound
    return path


def conditioned_preview_file(directory: Path, project: Project) -> Path:
    artifact = project.depth
    conditioning = None if artifact is None else artifact.conditioning
    if (
        conditioning is None
        or conditioning.conditioned_filename != CONDITIONED_FILENAME
        or conditioning.preview_filename != CONDITIONED_PREVIEW_FILENAME
    ):
        raise ConditionedPreviewNotFound
    path = directory / "depth" / PROCESSING_DIRECTORY / conditioning.preview_filename
    if path.is_symlink() or not path.is_file():
        raise ConditionedPreviewNotFound
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ConditionedPreviewNotFound
    return path


def conditioned_depth_file(directory: Path, project: Project) -> Path:
    artifact = project.depth
    conditioning = None if artifact is None else artifact.conditioning
    if conditioning is None or conditioning.conditioned_filename != CONDITIONED_FILENAME:
        raise ConditionedPreviewNotFound
    path = directory / "depth" / PROCESSING_DIRECTORY / conditioning.conditioned_filename
    if path.is_symlink() or not path.is_file():
        raise ConditionedPreviewNotFound
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ConditionedPreviewNotFound
    return path


def preview_file(directory: Path, project: Project) -> Path:
    artifact = project.depth
    if artifact is None or artifact.preview_filename != PREVIEW_FILENAME:
        raise DepthArtifactNotFound
    path = directory / "depth" / artifact.preview_filename
    if path.is_symlink() or not path.is_file():
        raise DepthArtifactNotFound
    if not path.resolve().is_relative_to(directory.resolve()):
        raise DepthArtifactNotFound
    return path


def artifact_for(
    *,
    model_id: str,
    model_version: str,
    device: str,
    input_size: int,
    width_px: int,
    height_px: int,
    generated_at: datetime,
) -> DepthArtifact:
    if device not in {"mps", "cpu"}:
        raise UnsafeProjectPath
    return DepthArtifact(
        model_id=model_id,
        model_version=model_version,
        device=device,
        input_size=input_size,
        width_px=width_px,
        height_px=height_px,
        raw_filename=RAW_DEPTH_FILENAME,
        preview_filename=PREVIEW_FILENAME,
        generated_at=generated_at,
    )
