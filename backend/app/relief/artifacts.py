"""Files for relief composition. Raw and conditioned depth are not written here."""

import io
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
from PIL import Image

from app.projects.errors import (
    RefinementSessionNotFound,
    ReliefRegionNotFound,
    SegmentationProposalNotFound,
    UnsafeProjectPath,
)

COMPOSITION_DIRECTORY = "composition"
MASKS_DIRECTORY = "masks"
PROPOSALS_DIRECTORY = "proposals"
AI_MASK_FILENAME = "ai_mask.png"
WORKING_MASK_FILENAME = "working_mask.png"
SESSION_BASE_FILENAME = "session_base_mask.png"
PROPOSAL_METADATA_FILENAME = "proposal.json"
PROPOSAL_ROLES = {"primary", "secondary", "supporting", "background", "custom"}
RELIEF_FILENAME = "relief_depth.npy"
RELIEF_PREVIEW_FILENAME = "preview.png"


def composition_directory(project_directory: Path) -> Path:
    depth_dir = project_directory / "depth"
    if depth_dir.is_symlink() or not depth_dir.is_dir():
        raise UnsafeProjectPath
    composition = depth_dir / COMPOSITION_DIRECTORY
    if composition.is_symlink():
        raise UnsafeProjectPath
    return composition


def atomic_replace_bytes(path: Path, data: bytes) -> None:
    """Replace a file by writing a sibling and renaming it into place."""

    if path.is_symlink():
        raise UnsafeProjectPath
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.is_symlink():
        raise UnsafeProjectPath
    temporary.write_bytes(data)
    temporary.replace(path)


def copy_mask_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise SegmentationProposalNotFound
    atomic_replace_bytes(destination, source.read_bytes())


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise UnsafeProjectPath
    if path.is_symlink():
        raise UnsafeProjectPath
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    atomic_replace_bytes(path, buffer.getvalue())


def read_mask_png(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ReliefRegionNotFound
    with Image.open(path) as handle:
        values = np.array(handle.convert("L"))
    return values >= 128


def write_relief_files(project_directory: Path, relief: np.ndarray, preview_png: bytes) -> None:
    if relief.dtype != np.float32 or relief.ndim != 2 or not np.isfinite(relief).all():
        raise UnsafeProjectPath
    composition = composition_directory(project_directory)
    composition.mkdir(parents=True, exist_ok=True)
    relief_path = composition / RELIEF_FILENAME
    preview_path = composition / RELIEF_PREVIEW_FILENAME
    if relief_path.is_symlink() or preview_path.is_symlink():
        raise UnsafeProjectPath
    relief_temporary = composition / ".relief_depth.partial.npy"
    preview_temporary = composition / ".preview.partial.png"
    try:
        np.save(relief_temporary, relief)
        preview_temporary.write_bytes(preview_png)
        relief_temporary.replace(relief_path)
        preview_temporary.replace(preview_path)
    except Exception:
        relief_temporary.unlink(missing_ok=True)
        preview_temporary.unlink(missing_ok=True)
        raise
    for path in (relief_path, preview_path):
        if not path.resolve().is_relative_to(project_directory.resolve()):
            raise UnsafeProjectPath


def mask_path(project_directory: Path, region_id: UUID) -> Path:
    filename = f"{region_id}.png"
    path = composition_directory(project_directory) / MASKS_DIRECTORY / filename
    if path.is_symlink():
        raise UnsafeProjectPath
    return path


def proposal_directory(project_directory: Path, proposal_id: UUID) -> Path:
    path = composition_directory(project_directory) / PROPOSALS_DIRECTORY / str(proposal_id)
    if path.is_symlink():
        raise UnsafeProjectPath
    return path


def _legacy_proposal_path(project_directory: Path, proposal_id: UUID) -> Path:
    path = composition_directory(project_directory) / PROPOSALS_DIRECTORY / f"{proposal_id}.png"
    if path.is_symlink():
        raise UnsafeProjectPath
    return path


def _require_inside(path: Path, project_directory: Path) -> None:
    if not path.resolve().is_relative_to(project_directory.resolve()):
        raise SegmentationProposalNotFound


def migrate_legacy_proposal(project_directory: Path, proposal_id: UUID) -> None:
    """Copy a single-file proposal into ai_mask.png and working_mask.png."""

    legacy = _legacy_proposal_path(project_directory, proposal_id)
    folder = proposal_directory(project_directory, proposal_id)
    if not legacy.is_file():
        return
    _require_inside(legacy, project_directory)
    if folder.exists():
        return
    folder.mkdir(parents=True, exist_ok=False)
    data = legacy.read_bytes()
    ai_mask = folder / AI_MASK_FILENAME
    working_mask = folder / WORKING_MASK_FILENAME
    ai_mask.write_bytes(data)
    working_mask.write_bytes(data)
    legacy.unlink()


@dataclass(frozen=True)
class ProposalMetadata:
    name: str
    role: str | None
    created_at: str
    origin_type: str = "new_region"
    region_id: str | None = None


def read_proposal_metadata(project_directory: Path, proposal_id: UUID) -> ProposalMetadata | None:
    """Return stored draft fields. A missing or unreadable file means none were saved."""

    path = proposal_directory(project_directory, proposal_id) / PROPOSAL_METADATA_FILENAME
    if not path.is_file() or path.is_symlink():
        return None
    _require_inside(path, project_directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    name = payload.get("name")
    role = payload.get("role")
    created_at = payload.get("created_at")
    if not isinstance(name, str) or not isinstance(created_at, str):
        return None
    if role is not None and role not in PROPOSAL_ROLES:
        role = None
    origin_type, region_id = _origin_fields(payload)
    return ProposalMetadata(
        name=name.strip(),
        role=role,
        created_at=created_at,
        origin_type=origin_type,
        region_id=region_id,
    )


def write_proposal_metadata(
    project_directory: Path,
    proposal_id: UUID,
    name: str,
    role: str | None,
    *,
    origin_type: str | None = None,
    region_id: str | None = None,
) -> ProposalMetadata:
    folder = proposal_directory(project_directory, proposal_id)
    if not folder.is_dir() or folder.is_symlink():
        raise SegmentationProposalNotFound
    _require_inside(folder, project_directory)
    if role is not None and role not in PROPOSAL_ROLES:
        raise UnsafeProjectPath
    existing = read_proposal_metadata(project_directory, proposal_id)
    created_at = existing.created_at if existing is not None else datetime.now(timezone.utc).isoformat()
    if existing is not None:
        stored_origin = existing.origin_type
        stored_region = existing.region_id
    else:
        stored_origin = origin_type if origin_type in ("new_region", "accepted_region") else "new_region"
        stored_region = region_id if stored_origin == "accepted_region" else None
    metadata = ProposalMetadata(
        name=name.strip(),
        role=role,
        created_at=created_at,
        origin_type=stored_origin,
        region_id=stored_region,
    )
    path = folder / PROPOSAL_METADATA_FILENAME
    if path.is_symlink():
        raise UnsafeProjectPath
    temporary = path.with_name(".proposal.partial.json")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": metadata.name,
                "role": metadata.role,
                "created_at": metadata.created_at,
                "origin_type": metadata.origin_type,
                "region_id": metadata.region_id,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return metadata


def _origin_fields(payload: dict) -> tuple[str, str | None]:
    origin = payload.get("origin_type", "new_region")
    region_id = payload.get("region_id")
    if origin != "accepted_region" or not isinstance(region_id, str):
        return "new_region", None
    try:
        UUID(region_id)
    except ValueError:
        return "new_region", None
    return "accepted_region", region_id


def write_proposal_masks(project_directory: Path, proposal_id: UUID, mask: np.ndarray) -> None:
    folder = proposal_directory(project_directory, proposal_id)
    if folder.exists():
        raise UnsafeProjectPath
    folder.mkdir(parents=True, exist_ok=False)
    ai_mask = folder / AI_MASK_FILENAME
    write_mask_png(ai_mask, mask)
    working_mask = folder / WORKING_MASK_FILENAME
    copy_mask_file(ai_mask, working_mask)
    _require_inside(ai_mask, project_directory)
    _require_inside(working_mask, project_directory)


def existing_proposal(project_directory: Path, proposal_id: UUID) -> Path:
    """Return the refined working mask, migrating an older single-file proposal."""

    migrate_legacy_proposal(project_directory, proposal_id)
    path = proposal_directory(project_directory, proposal_id) / WORKING_MASK_FILENAME
    if not path.is_file() or path.is_symlink():
        raise SegmentationProposalNotFound
    _require_inside(path, project_directory)
    return path


def existing_ai_mask(project_directory: Path, proposal_id: UUID) -> Path:
    migrate_legacy_proposal(project_directory, proposal_id)
    path = proposal_directory(project_directory, proposal_id) / AI_MASK_FILENAME
    if not path.is_file() or path.is_symlink():
        raise SegmentationProposalNotFound
    _require_inside(path, project_directory)
    return path


def replace_working_mask(project_directory: Path, proposal_id: UUID, mask: np.ndarray) -> None:
    path = existing_proposal(project_directory, proposal_id)
    write_mask_png(path, mask)


def start_refinement_session(project_directory: Path, proposal_id: UUID) -> None:
    """Snapshot the current working mask. This does not modify the working mask."""

    working = existing_proposal(project_directory, proposal_id)
    baseline = working.parent / SESSION_BASE_FILENAME
    _require_inside(working.parent, project_directory)
    copy_mask_file(working, baseline)


def restore_session_baseline(project_directory: Path, proposal_id: UUID) -> None:
    """Replace the working mask with the current session baseline."""

    working = existing_proposal(project_directory, proposal_id)
    baseline = working.parent / SESSION_BASE_FILENAME
    if not baseline.is_file() or baseline.is_symlink():
        raise RefinementSessionNotFound
    _require_inside(baseline, project_directory)
    copy_mask_file(baseline, working)


def reset_working_mask(project_directory: Path, proposal_id: UUID) -> None:
    ai_mask = existing_ai_mask(project_directory, proposal_id)
    working = existing_proposal(project_directory, proposal_id)
    copy_mask_file(ai_mask, working)
    copy_mask_file(working, working.parent / SESSION_BASE_FILENAME)


def remove_proposal(project_directory: Path, proposal_id: UUID) -> None:
    folder = proposal_directory(project_directory, proposal_id)
    legacy = _legacy_proposal_path(project_directory, proposal_id)
    found = False
    if folder.is_dir() and not folder.is_symlink():
        _require_inside(folder, project_directory)
        shutil.rmtree(folder)
        found = True
    if legacy.is_file() and not legacy.is_symlink():
        _require_inside(legacy, project_directory)
        legacy.unlink()
        found = True
    if not found:
        raise SegmentationProposalNotFound


def list_proposal_ids(project_directory: Path) -> list[UUID]:
    folder = composition_directory(project_directory) / PROPOSALS_DIRECTORY
    if not folder.is_dir() or folder.is_symlink():
        return []
    found: set[UUID] = set()
    for child in folder.iterdir():
        if child.is_symlink():
            continue
        name = child.stem if child.is_file() and child.suffix == ".png" else child.name
        if child.is_dir() or (child.is_file() and child.suffix == ".png"):
            try:
                found.add(UUID(name))
            except ValueError:
                continue
    return sorted(found, key=str)


def existing_mask(project_directory: Path, region_id: UUID, filename: str) -> Path:
    expected = f"{region_id}.png"
    if filename != expected:
        raise ReliefRegionNotFound
    path = mask_path(project_directory, region_id)
    if not path.is_file():
        raise ReliefRegionNotFound
    if not path.resolve().is_relative_to(project_directory.resolve()):
        raise ReliefRegionNotFound
    return path


def clear_other_proposals(project_directory: Path, keep: UUID) -> None:
    folder = composition_directory(project_directory) / PROPOSALS_DIRECTORY
    if not folder.is_dir() or folder.is_symlink():
        return
    kept_names = {str(keep), f"{keep}.png"}
    for child in folder.iterdir():
        if child.name in kept_names or child.is_symlink():
            continue
        if child.is_dir():
            shutil.rmtree(child)
        elif child.is_file() and child.suffix == ".png":
            child.unlink()
