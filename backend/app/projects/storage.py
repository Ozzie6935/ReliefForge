"""Filesystem persistence for projects.

The directory layout is projects/<slug>/project.json and, after upload,
projects/<slug>/source/original.<ext>.
"""

import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np

from pydantic import ValidationError

from app.projects.errors import (
    ProjectConflict,
    ProjectNotFound,
    RawDepthNotFound,
    SourceImageNotFound,
    UnsafeProjectPath,
)
from app.depth.artifacts import write_conditioned_files, write_depth_files
from app.projects.provenance import (
    file_sha256,
    fingerprint_conditioning,
    fingerprint_depth,
    sha256_bytes,
)
from app.relief.artifacts import atomic_replace_bytes
from app.relief.physical import attach_physical_status, manifest_payload
from app.schemas.conditioning import ConditioningRecord
from app.schemas.project import DepthArtifact, Project, SourceImage
from app.schemas.relief import ReliefRecord

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SOURCE_MEDIA_TYPES = {
    "original.jpg": "image/jpeg",
    "original.png": "image/png",
}


def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")


def is_safe_slug(slug: str) -> bool:
    return bool(_SLUG_PATTERN.fullmatch(slug))


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_project(self, project: Project) -> None:
        directory = self.project_directory(project.slug)
        if directory.exists():
            raise ProjectConflict
        directory.mkdir(parents=False)
        try:
            self._write_manifest(directory, project)
        except Exception:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
            raise

    def list_projects(self) -> list[Project]:
        projects = self._iter_projects()
        projects.sort(
            key=lambda project: (
                -(project.updated_at.timestamp() if project.updated_at else 0.0),
                project.name.casefold(),
            )
        )
        return projects

    def update_project(self, project: Project) -> None:
        """Rewrite metadata in the existing directory. The slug is not changed."""

        directory = self.project_directory(project.slug)
        if not (directory / "project.json").is_file():
            raise ProjectNotFound
        self._write_manifest(directory, project)

    def delete_project(self, project: Project) -> None:
        directory = self.project_directory(project.slug)
        resolved = directory.resolve()
        root = self.root.resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise UnsafeProjectPath
        if not (directory / "project.json").is_file():
            raise ProjectNotFound
        shutil.rmtree(directory)

    def get_by_id(self, project_id: UUID) -> Project:
        for project in self._iter_projects():
            if project.id == project_id:
                return project
        raise ProjectNotFound

    def save_source(self, project: Project, data: bytes, source: SourceImage) -> Project:
        if _SOURCE_MEDIA_TYPES.get(source.filename) != source.media_type:
            raise UnsafeProjectPath
        directory = self.project_directory(project.slug)
        source_dir = directory / "source"
        source_dir.mkdir(exist_ok=True)
        destination = source_dir / source.filename
        temporary = source_dir / f".{source.filename}.partial"
        recorded = source.model_copy(
            update={"content_sha256": sha256_bytes(data), "status": None}
        )
        updated = project.model_copy(
            update={
                "source": recorded,
                "created_at": project.created_at or datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        manifest_updated = False
        try:
            temporary.write_bytes(data)
            self._write_manifest(directory, updated)
            manifest_updated = True
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            if manifest_updated:
                self._write_manifest(directory, project)
            raise
        for filename in _SOURCE_MEDIA_TYPES:
            if filename == source.filename:
                continue
            leftover = source_dir / filename
            if leftover.is_file() and not leftover.is_symlink():
                leftover.unlink()
        return self._present(directory, updated)

    def save_depth(
        self,
        project: Project,
        depth: np.ndarray,
        preview_png: bytes,
        artifact: DepthArtifact,
    ) -> Project:
        """Write depth files, then the manifest. A failed write restores the previous raw files."""

        directory = self.project_directory(project.slug)
        depth_dir = directory / "depth"
        raw_path = depth_dir / "raw_depth.npy"
        preview_path = depth_dir / "preview.png"
        snapshot = {
            raw_path: self._read_snapshot(raw_path),
            preview_path: self._read_snapshot(preview_path),
        }
        try:
            write_depth_files(directory, depth, preview_png)
            source_hash = file_sha256(self.source_path(project))
            raw_hash = file_sha256(raw_path)
            if source_hash is None or raw_hash is None or project.source is None:
                raise RawDepthNotFound
            prior = None if project.depth is None else project.depth.conditioning
            if prior is not None:
                prior = prior.model_copy(update={"status": None})
            recorded = fingerprint_depth(artifact, raw_hash, source_hash).model_copy(
                update={"conditioning": prior}
            )
            source = project.source.model_copy(
                update={"content_sha256": source_hash, "status": None}
            )
            updated = project.model_copy(
                update={
                    "source": source,
                    "depth": recorded,
                    "created_at": project.created_at or datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._write_manifest(directory, updated)
        except Exception:
            self._restore_snapshot(snapshot)
            raise
        return self._present(directory, updated)

    def save_conditioning(
        self,
        project: Project,
        depth: np.ndarray,
        preview_png: bytes,
        record: ConditioningRecord,
    ) -> Project:
        """Write conditioned artifacts, then attach their metadata.

        ``raw_depth.npy`` and the raw preview are not part of this write.
        """

        if project.depth is None:
            raise RawDepthNotFound
        directory = self.project_directory(project.slug)
        processing = directory / "depth" / "processing"
        conditioned_path = processing / "conditioned_depth.npy"
        preview_path = processing / "preview.png"
        raw_path = directory / "depth" / "raw_depth.npy"
        snapshot = {
            conditioned_path: self._read_snapshot(conditioned_path),
            preview_path: self._read_snapshot(preview_path),
        }
        try:
            write_conditioned_files(directory, depth, preview_png)
            content_hash = file_sha256(conditioned_path)
            raw_hash = file_sha256(raw_path)
            if content_hash is None or raw_hash is None:
                raise RawDepthNotFound
            recorded = fingerprint_conditioning(record, content_hash, raw_hash)
            depth_record = project.depth.model_copy(update={"conditioning": recorded, "status": None})
            updated = project.model_copy(
                update={
                    "depth": depth_record,
                    "created_at": project.created_at or datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._write_manifest(directory, updated)
        except Exception:
            self._restore_snapshot(snapshot)
            raise
        return self._present(directory, updated)

    def save_relief(self, project: Project, record: ReliefRecord) -> Project:
        """Attach relief metadata after the composition files have been written."""

        directory = self.project_directory(project.slug)
        updated = project.model_copy(
            update={
                "relief": record,
                "created_at": project.created_at or datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._write_manifest(directory, updated)
        return self._present(directory, updated)

    def _read_snapshot(self, path: Path) -> bytes | None:
        if path.is_symlink():
            raise UnsafeProjectPath
        if not path.is_file():
            return None
        return path.read_bytes()

    def _restore_snapshot(self, snapshot: dict[Path, bytes | None]) -> None:
        for path, data in snapshot.items():
            if path.is_symlink():
                raise UnsafeProjectPath
            parent = path.parent
            if parent.is_dir() and not parent.is_symlink():
                for partial in parent.glob(".*partial*"):
                    if partial.is_file() and not partial.is_symlink():
                        partial.unlink()
            if data is None:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
                continue
            parent.mkdir(parents=True, exist_ok=True)
            atomic_replace_bytes(path, data)

    def _discard_unrecorded_conditioning(self, directory: Path, project: Project) -> None:
        if project.depth is not None and project.depth.conditioning is not None:
            return
        processing = directory / "depth" / "processing"
        if processing.is_symlink() or not processing.is_dir():
            return
        for name in ("conditioned_depth.npy", "preview.png"):
            path = processing / name
            if path.is_file() and not path.is_symlink():
                path.unlink()
        try:
            processing.rmdir()
        except OSError:
            return

    def _discard_unrecorded_depth(self, directory: Path, project: Project) -> None:
        if project.depth is not None:
            return
        depth_dir = directory / "depth"
        if depth_dir.is_symlink() or not depth_dir.is_dir():
            return
        for name in ("raw_depth.npy", "preview.png"):
            path = depth_dir / name
            if path.is_file() and not path.is_symlink():
                path.unlink()
        try:
            depth_dir.rmdir()
        except OSError:
            return

    def source_path(self, project: Project) -> Path:
        if project.source is None:
            raise SourceImageNotFound
        filename = project.source.filename
        if _SOURCE_MEDIA_TYPES.get(filename) != project.source.media_type:
            raise SourceImageNotFound
        directory = self.project_directory(project.slug)
        direct = directory / "source" / filename
        if direct.is_symlink() or not direct.is_file():
            raise SourceImageNotFound
        if not direct.resolve().is_relative_to(directory.resolve()):
            raise SourceImageNotFound
        return direct

    def project_directory(self, slug: str) -> Path:
        if not is_safe_slug(slug):
            raise UnsafeProjectPath
        root = self.root.resolve()
        candidate = root / slug
        if candidate.is_symlink():
            raise UnsafeProjectPath
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise UnsafeProjectPath
        return candidate

    def _iter_projects(self) -> list[Project]:
        projects: list[Project] = []
        for child in self.root.iterdir():
            if not child.is_dir() or child.is_symlink():
                continue
            manifest = child / "project.json"
            if not manifest.is_file():
                continue
            try:
                project = Project.model_validate_json(manifest.read_text(encoding="utf-8"))
            except (OSError, ValidationError, json.JSONDecodeError, ValueError):
                continue
            if child.name != project.slug or not is_safe_slug(project.slug):
                continue
            projects.append(self._present(child, project))
        return projects

    def _with_timestamps(self, directory: Path, project: Project) -> Project:
        """Fill missing lifecycle times for the response. This does not write."""

        if project.created_at is not None and project.updated_at is not None:
            return project
        manifest = directory / "project.json"
        try:
            stat = manifest.stat()
        except OSError:
            return project
        created_stamp = getattr(stat, "st_birthtime", stat.st_mtime)
        created = project.created_at or datetime.fromtimestamp(created_stamp, timezone.utc)
        updated = project.updated_at or datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        return project.model_copy(update={"created_at": created, "updated_at": updated})

    def _present(self, directory: Path, project: Project) -> Project:
        return attach_physical_status(directory, self._with_timestamps(directory, project))

    def _write_manifest(self, directory: Path, project: Project) -> None:
        if directory.resolve() != self.project_directory(project.slug).resolve():
            raise UnsafeProjectPath
        target = directory / "project.json"
        temporary = directory / ".project.json.partial"
        text = json.dumps(manifest_payload(project), indent=2) + "\n"
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
