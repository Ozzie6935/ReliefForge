"""Copy one project directory to a new identity.

The copy is byte-for-byte. Provenance fingerprints are kept when they are
already recorded. Missing fingerprints stay missing, so Unknown and Stale are
not promoted. This module does not generate depth, relief, or physical height.

Publication claims the destination with mkdir, which fails if that directory
already exists, including when it is empty. rename then moves the staging
directory onto the placeholder this process just created. A second concurrent
duplicate of the same slug gets HTTP 409 and does not replace the winner.
"""

import errno
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.core.config import MAX_DUPLICATE_BYTES, MAX_DUPLICATE_FILES
from app.projects.errors import (
    ProjectConflict,
    ProjectDuplicateFailed,
    ProjectDuplicateTooLarge,
    ProjectError,
    UnsafeProjectPath,
)
from app.projects.storage import ProjectStore, is_safe_slug
from app.schemas.project import Project

_STAGING_PREFIX = ".duplicating-"
_FILE_LIMIT_DETAIL = "The project has too many files to duplicate."
_BYTE_LIMIT_DETAIL = "The project is too large to duplicate."
_CHANGED_DETAIL = "The project changed while it was being copied."
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_OCCUPIED = {errno.EEXIST, errno.ENOTEMPTY}


class _FileSnapshot:
    def __init__(self, relative: Path, info: os.stat_result) -> None:
        self.relative = relative
        self.dev = info.st_dev
        self.ino = info.st_ino
        self.size = info.st_size
        self.mtime_ns = info.st_mtime_ns

    def matches(self, info: os.stat_result) -> bool:
        return (
            info.st_dev == self.dev
            and info.st_ino == self.ino
            and info.st_size == self.size
            and info.st_mtime_ns == self.mtime_ns
        )


def duplicate_project_directory(
    store: ProjectStore,
    source: Project,
    *,
    name: str,
    slug: str,
    new_id: UUID,
    now: datetime,
) -> None:
    """Publish a new project directory. The source directory is only read."""

    if not is_safe_slug(slug) or not is_safe_slug(source.slug):
        raise UnsafeProjectPath
    source_dir = _source_directory(store, source.slug)
    destination = _destination_directory(store, slug)
    if destination.exists() or destination.is_symlink():
        raise ProjectConflict
    files, directories = _inventory(source_dir)
    staging = _staging_directory(store)
    try:
        staging.mkdir(parents=False, exist_ok=False)
        for relative in directories:
            _require_directory_chain(source_dir, relative)
            _child(staging, relative).mkdir(parents=True, exist_ok=True)
        digests: dict[str, str] = {}
        for snapshot in files:
            digests[snapshot.relative.as_posix()] = _copy_regular_file(
                source_dir,
                _child(staging, snapshot.relative),
                snapshot,
            )
        _rewrite_identity(staging / "project.json", name=name, slug=slug, new_id=new_id, now=now)
        _require_same_tree(source_dir, files, digests)
        _publish_staging(staging, destination)
    except ProjectError:
        if staging.exists() or staging.is_symlink():
            _remove_staging(store, staging)
        raise
    except Exception as exc:
        if staging.exists() or staging.is_symlink():
            _remove_staging(store, staging)
        raise ProjectDuplicateFailed from exc


def _source_directory(store: ProjectStore, slug: str) -> Path:
    directory = store.project_directory(slug)
    root = store.root.resolve()
    if directory.is_symlink() or not directory.is_dir():
        raise UnsafeProjectPath
    resolved = directory.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise UnsafeProjectPath
    manifest = directory / "project.json"
    info = _lstat_regular_or_dir(manifest)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise UnsafeProjectPath
    return directory


def _destination_directory(store: ProjectStore, slug: str) -> Path:
    destination = store.project_directory(slug)
    root = store.root.resolve()
    if destination.is_symlink():
        raise UnsafeProjectPath
    resolved = destination.resolve()
    if resolved == root or destination.parent.resolve() != root or not resolved.is_relative_to(root):
        raise UnsafeProjectPath
    return destination


def _inventory(directory: Path) -> tuple[list[_FileSnapshot], list[Path]]:
    """List regular files and real directories. Symlinks and other types are rejected."""

    root = directory.resolve()
    files: list[_FileSnapshot] = []
    directories: list[Path] = []
    total = 0
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        folder = Path(current)
        _require_real_directory(folder, root)
        if folder != directory:
            directories.append(folder.relative_to(directory))
        kept: list[str] = []
        for name in dirnames:
            child = folder / name
            info = _lstat_regular_or_dir(child)
            if info is None or not stat.S_ISDIR(info.st_mode):
                raise UnsafeProjectPath
            _require_inside(child, root)
            kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            child = folder / name
            info = _lstat_regular_or_dir(child)
            if info is None or not stat.S_ISREG(info.st_mode):
                raise UnsafeProjectPath
            _require_inside(child, root)
            total += info.st_size
            files.append(_FileSnapshot(child.relative_to(directory), info))
            if len(files) > MAX_DUPLICATE_FILES:
                raise ProjectDuplicateTooLarge(_FILE_LIMIT_DETAIL)
            if total > MAX_DUPLICATE_BYTES:
                raise ProjectDuplicateTooLarge(_BYTE_LIMIT_DETAIL)
    if not any(item.relative == Path("project.json") for item in files):
        raise UnsafeProjectPath
    return files, directories


def _lstat_regular_or_dir(path: Path) -> os.stat_result | None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise UnsafeProjectPath from exc
    if stat.S_ISLNK(info.st_mode):
        raise UnsafeProjectPath
    if stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode):
        return info
    raise UnsafeProjectPath


def _require_real_directory(path: Path, root: Path) -> None:
    info = _lstat_regular_or_dir(path)
    if info is None or not stat.S_ISDIR(info.st_mode):
        raise UnsafeProjectPath
    _require_inside(path, root)


def _require_inside(path: Path, root: Path) -> None:
    if path.is_symlink():
        raise UnsafeProjectPath
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise UnsafeProjectPath
    if any(part in {"", ".."} for part in resolved.relative_to(root).parts):
        raise UnsafeProjectPath


def _staging_directory(store: ProjectStore) -> Path:
    root = store.root.resolve()
    staging = root / f"{_STAGING_PREFIX}{uuid4().hex}"
    resolved = staging.resolve()
    if staging.exists() or staging.is_symlink() or resolved == root or not resolved.is_relative_to(root):
        raise UnsafeProjectPath
    if not staging.name.startswith(_STAGING_PREFIX):
        raise UnsafeProjectPath
    return staging


def _child(staging: Path, relative: Path) -> Path:
    if relative.is_absolute() or any(part in {"", ".."} for part in relative.parts):
        raise UnsafeProjectPath
    target = staging.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(staging.resolve()):
        raise UnsafeProjectPath
    return target


def _open_directory(root: Path) -> int:
    try:
        return os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise UnsafeProjectPath from exc


def _require_directory_chain(root: Path, relative: Path) -> None:
    if relative.is_absolute() or any(part in {"", ".."} for part in relative.parts):
        raise UnsafeProjectPath
    directory = _open_directory(root)
    try:
        for part in relative.parts:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
    except OSError as exc:
        raise UnsafeProjectPath from exc
    finally:
        os.close(directory)


def _open_nofollow(root: Path, relative: Path) -> int:
    """Open one file without following a symlink at any path component."""

    if relative.is_absolute() or any(part in {"", ".."} for part in relative.parts):
        raise UnsafeProjectPath
    directory = _open_directory(root)
    try:
        parts = relative.parts
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(parts[-1], _READ_FLAGS, dir_fd=directory)
    except OSError as exc:
        raise UnsafeProjectPath from exc
    finally:
        os.close(directory)


def _copy_regular_file(root: Path, destination: Path, expected: _FileSnapshot) -> str:
    descriptor = _open_nofollow(root, expected.relative)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafeProjectPath
        if not expected.matches(info):
            raise ProjectDuplicateFailed(_CHANGED_DETAIL)
        hasher = hashlib.sha256()
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        try:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                _write_all(output, chunk)
                hasher.update(chunk)
        finally:
            os.close(output)
        if not expected.matches(os.fstat(descriptor)):
            raise ProjectDuplicateFailed(_CHANGED_DETAIL)
        copied = destination.lstat()
        if (
            stat.S_ISLNK(copied.st_mode)
            or not stat.S_ISREG(copied.st_mode)
            or copied.st_nlink != 1
            or (copied.st_dev == info.st_dev and copied.st_ino == info.st_ino)
        ):
            raise UnsafeProjectPath
        return hasher.hexdigest()
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, chunk: bytes) -> None:
    view = memoryview(chunk)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ProjectDuplicateFailed
        view = view[written:]


def _require_same_tree(
    source_dir: Path,
    files: list[_FileSnapshot],
    digests: dict[str, str],
) -> None:
    """Reject the copy when the source tree changed after the bytes were read."""

    again, _directories = _inventory(source_dir)
    expected = {item.relative.as_posix() for item in files}
    if {item.relative.as_posix() for item in again} != expected:
        raise ProjectDuplicateFailed(_CHANGED_DETAIL)
    for snapshot in files:
        key = snapshot.relative.as_posix()
        digest = _hash_regular_file(source_dir, snapshot)
        if digest != digests[key]:
            raise ProjectDuplicateFailed(_CHANGED_DETAIL)


def _hash_regular_file(root: Path, expected: _FileSnapshot) -> str:
    try:
        descriptor = _open_nofollow(root, expected.relative)
    except UnsafeProjectPath as exc:
        raise ProjectDuplicateFailed(_CHANGED_DETAIL) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not expected.matches(info):
            raise ProjectDuplicateFailed(_CHANGED_DETAIL)
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
        if not expected.matches(os.fstat(descriptor)):
            raise ProjectDuplicateFailed(_CHANGED_DETAIL)
        return hasher.hexdigest()
    finally:
        os.close(descriptor)


def _publish_staging(staging: Path, destination: Path) -> None:
    """Claim destination with mkdir, then rename staging onto that empty placeholder.

    mkdir fails when the directory already exists, so rename never replaces
    another project's directory. The placeholder is removed if rename fails.
    """

    if destination.exists() or destination.is_symlink() or staging.is_symlink():
        raise ProjectConflict
    created = False
    try:
        os.mkdir(destination)
        created = True
        os.rename(staging, destination)
    except FileExistsError as exc:
        if created:
            _remove_empty_placeholder(destination)
        raise ProjectConflict from exc
    except OSError as exc:
        if created:
            _remove_empty_placeholder(destination)
        if exc.errno in _OCCUPIED:
            raise ProjectConflict from exc
        raise


def _remove_empty_placeholder(destination: Path) -> None:
    try:
        os.rmdir(destination)
    except OSError:
        return


def _rewrite_identity(
    manifest_path: Path,
    *,
    name: str,
    slug: str,
    new_id: UUID,
    now: datetime,
) -> None:
    """Replace project identity. Nested artifact records and hashes stay as copied."""

    info = _lstat_regular_or_dir(manifest_path)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise UnsafeProjectPath
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnsafeProjectPath
    payload["id"] = str(new_id)
    payload["name"] = name
    payload["slug"] = slug
    stamp = now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["created_at"] = stamp
    payload["updated_at"] = stamp
    project = Project.model_validate(payload)
    if project.id != new_id or project.slug != slug or project.name != name:
        raise UnsafeProjectPath
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary = manifest_path.with_name(".project.json.partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, manifest_path)


def _remove_staging(store: ProjectStore, staging: Path) -> None:
    root = store.root.resolve()
    if not staging.name.startswith(_STAGING_PREFIX):
        raise UnsafeProjectPath
    if staging.is_symlink():
        raise UnsafeProjectPath
    resolved = staging.resolve() if staging.exists() else staging
    if not resolved.is_relative_to(root) or resolved == root:
        raise UnsafeProjectPath
    if staging.exists():
        shutil.rmtree(staging)
