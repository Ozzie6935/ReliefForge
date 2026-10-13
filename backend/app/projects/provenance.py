"""Read-time freshness for source, depth, relief, and physical height.

Computed status is not written to project.json. Timestamps are kept, but a
matching timestamp is not enough to call an artifact current. Preview PNGs
are not processing inputs.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from app.depth.artifacts import (
    CONDITIONED_FILENAME,
    PROCESSING_DIRECTORY,
    RAW_DEPTH_FILENAME,
)
from app.projects.errors import UnsafeProjectPath
from app.relief.artifacts import MASKS_DIRECTORY, RELIEF_FILENAME, composition_directory
from app.schemas.conditioning import ConditioningRecord, DepthConditioningParameters
from app.schemas.project import DepthArtifact, Project, SourceImage
from app.schemas.relief import PhysicalHeightRecord, ReliefRecord, ReliefRegionRecord

ArtifactStatus = Literal["missing", "unknown", "current", "stale"]

RAW_STALE_DETAIL = (
    "Raw depth is stale. Regenerate raw depth from the current source before conditioning."
)
RAW_UNKNOWN_DETAIL = (
    "Raw depth lineage is unknown. Regenerate raw depth from the current source before conditioning."
)
CONDITIONED_STALE_DETAIL = (
    "Conditioned depth is stale. Apply conditioning again before composing relief."
)
CONDITIONED_UNKNOWN_DETAIL = (
    "Conditioned depth lineage is unknown. Apply conditioning again before composing relief."
)
RELIEF_STALE_DETAIL = (
    "Composed relief is stale. Apply composition again before generating physical height."
)
RELIEF_UNKNOWN_DETAIL = (
    "Composed relief lineage is unknown. Apply composition again before generating physical height."
)

_SOURCE_FILES = {"original.jpg": "image/jpeg", "original.png": "image/png"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def canonical_json(value: object) -> str:
    """Stable JSON: sorted keys, compact separators, no NaN."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def depth_settings_document(artifact: DepthArtifact) -> dict[str, object]:
    return {
        "device": artifact.device,
        "height_px": artifact.height_px,
        "input_size": artifact.input_size,
        "model_id": artifact.model_id,
        "model_version": artifact.model_version,
        "width_px": artifact.width_px,
    }


def conditioning_settings_document(parameters: DepthConditioningParameters) -> dict[str, object]:
    return parameters.model_dump(mode="json")


def physical_settings_document(record: PhysicalHeightRecord) -> dict[str, object]:
    return {
        "base_thickness_mm": record.base_thickness_mm,
        "maximum_surface_height_mm": record.maximum_surface_height_mm,
        "stock_thickness_mm": record.stock_thickness_mm,
    }


COMPOSITION_SETTINGS_VERSION = 2
PHYSICAL_SCALE_SETTINGS_VERSION = 3
BACKGROUND_SETTINGS_VERSION = 4


def composition_source_sha256(directory: Path) -> str | None:
    """Hash of the stored source file used by fine detail. No timestamp is used."""

    source_dir = directory / "source"
    for filename in ("original.jpg", "original.png"):
        path = source_dir / filename
        if path.is_symlink():
            raise UnsafeProjectPath
        digest = file_sha256(path)
        if digest is not None:
            return digest
    return None


def _positive_artwork_mm(value: float | None) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _background_identity(depth_range: float, elevation: float, feather: float) -> bool:
    return depth_range == 1.0 and elevation == 0.0 and feather == 0.0


def composition_settings_document(
    directory: Path,
    regions: list[ReliefRegionRecord],
    *,
    version: int | None = None,
    source_sha256: str | None = None,
    artwork_width_mm: float | None = None,
    artwork_height_mm: float | None = None,
    background_depth_range: float | None = None,
    background_elevation: float | None = None,
    background_feather: float | None = None,
) -> dict[str, object]:
    """Mask bytes plus the settings compose_relief actually reads.

    Version None is the historical document: elevation, depth range, feather,
    id, mask hash, and order. It does not contain ``detail_amount``. Records
    saved before fine detail use that document, so a default of zero does not
    change their settings hash.

    Version 2 is written when a composition uses fine detail in pixel mode.
    It includes ``detail_amount`` on every region and no scale key. When any
    amount is above zero it also includes the source-file hash, because that
    composition read source pixels. A zero-detail version 2 document does not
    add that source dependency.

    Version 3 is written only when a region has a physical detail scale. It
    adds ``detail_scale_mm`` on every region and the artwork width and height
    in millimeters. Those dimensions set the pixel pitch. A null scale on a
    version-3 region means that region stayed in pixel mode.

    Version 4 is written only when background settings are not the identity
    depth range 1, elevation 0, and feather 0. It adds those three fields.
    Region entries include detail amount and detail scale. Artwork millimeters
    are included only when a scale is set. The source hash is included only
    when some detail amount is above zero. Versions null, 2, and 3 do not
    gain background keys.
    """

    if version not in {
        None,
        COMPOSITION_SETTINGS_VERSION,
        PHYSICAL_SCALE_SETTINGS_VERSION,
        BACKGROUND_SETTINGS_VERSION,
    }:
        raise ValueError(version)
    has_scale = any(region.detail_scale_mm is not None for region in regions)
    if version not in {PHYSICAL_SCALE_SETTINGS_VERSION, BACKGROUND_SETTINGS_VERSION} and has_scale:
        raise ValueError(version)
    needs_artwork = version == PHYSICAL_SCALE_SETTINGS_VERSION or (
        version == BACKGROUND_SETTINGS_VERSION and has_scale
    )
    if needs_artwork and not (
        _positive_artwork_mm(artwork_width_mm) and _positive_artwork_mm(artwork_height_mm)
    ):
        raise ValueError(version)
    if version == BACKGROUND_SETTINGS_VERSION and (
        background_depth_range is None
        or background_elevation is None
        or background_feather is None
    ):
        raise ValueError(version)
    items: list[dict[str, object]] = []
    ordered = sorted(regions, key=lambda region: (region.order, str(region.id)))
    masks = composition_directory(directory) / MASKS_DIRECTORY
    for region in ordered:
        filename = region.mask_filename
        if filename != f"{region.id}.png" or "/" in filename or "\\" in filename:
            raise OSError(filename)
        path = masks / filename
        if path.is_symlink() or not path.is_file():
            raise OSError(filename)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise UnsafeProjectPath
        item: dict[str, object] = {
            "depth_range": region.depth_range,
            "elevation": region.elevation,
            "feather": region.feather,
            "id": str(region.id),
            "mask_sha256": sha256_bytes(path.read_bytes()),
            "order": region.order,
        }
        if version == COMPOSITION_SETTINGS_VERSION:
            item["detail_amount"] = region.detail_amount
        elif version in {PHYSICAL_SCALE_SETTINGS_VERSION, BACKGROUND_SETTINGS_VERSION}:
            item["detail_amount"] = region.detail_amount
            item["detail_scale_mm"] = region.detail_scale_mm
        items.append(item)
    document: dict[str, object] = {"regions": items}
    if version == COMPOSITION_SETTINGS_VERSION:
        document["version"] = COMPOSITION_SETTINGS_VERSION
        if any(region.detail_amount > 0.0 for region in ordered):
            document["source_sha256"] = source_sha256
    elif version == PHYSICAL_SCALE_SETTINGS_VERSION:
        document["artwork_height_mm"] = artwork_height_mm
        document["artwork_width_mm"] = artwork_width_mm
        document["version"] = PHYSICAL_SCALE_SETTINGS_VERSION
        if any(region.detail_amount > 0.0 for region in ordered):
            document["source_sha256"] = source_sha256
    elif version == BACKGROUND_SETTINGS_VERSION:
        document["background_depth_range"] = background_depth_range
        document["background_elevation"] = background_elevation
        document["background_feather"] = background_feather
        document["version"] = BACKGROUND_SETTINGS_VERSION
        if has_scale:
            document["artwork_height_mm"] = artwork_height_mm
            document["artwork_width_mm"] = artwork_width_mm
        if any(region.detail_amount > 0.0 for region in ordered):
            document["source_sha256"] = source_sha256
    return document


def judge_stage(
    *,
    present: bool,
    content_recorded: bool,
    content_ok: bool,
    settings_recorded: bool,
    settings_ok: bool,
    input_recorded: bool,
    input_ok: bool,
    require_settings: bool,
    require_input: bool,
    upstream: ArtifactStatus | None,
) -> ArtifactStatus:
    """Precedence: missing file, then a proven mismatch, then unknown lineage.

    An upstream Unknown result does not become Current. It also does not become
    a proven Stale result when this stage has no fingerprint that can disagree.
    A recorded fingerprint that does not match is Stale even if the ancestor
    is Unknown.
    """

    if not present:
        return "missing"
    if content_recorded and not content_ok:
        return "stale"
    if settings_recorded and not settings_ok:
        return "stale"
    if input_recorded and not input_ok:
        return "stale"
    if (
        not content_recorded
        or (require_settings and not settings_recorded)
        or (require_input and not input_recorded)
    ):
        return "unknown"
    if upstream is None or upstream == "current":
        return "current"
    if upstream == "stale" or upstream == "missing":
        return "stale"
    return "unknown"


def _source_path(directory: Path, source: SourceImage | None) -> Path | None:
    if source is None or _SOURCE_FILES.get(source.filename) != source.media_type:
        return None
    return directory / "source" / source.filename


def resolve_artifact_statuses(directory: Path, project: Project) -> dict[str, ArtifactStatus]:
    source_path = _source_path(directory, project.source)
    source_hash = None if source_path is None else file_sha256(source_path)
    source_recorded = project.source is not None and project.source.content_sha256 is not None
    source_status = judge_stage(
        present=project.source is not None and source_hash is not None,
        content_recorded=source_recorded,
        content_ok=bool(project.source and source_hash == project.source.content_sha256),
        settings_recorded=False,
        settings_ok=True,
        input_recorded=False,
        input_ok=True,
        require_settings=False,
        require_input=False,
        upstream=None,
    )

    raw_path = None if project.depth is None else directory / "depth" / RAW_DEPTH_FILENAME
    raw_hash = None if raw_path is None else file_sha256(raw_path)
    depth = project.depth
    raw_settings_ok = False
    if depth is not None and depth.settings_sha256 is not None:
        raw_settings_ok = depth.settings_sha256 == canonical_sha256(depth_settings_document(depth))
    raw_input_ok = bool(depth and raw_hash is not None and depth.source_sha256 == source_hash)
    raw_status = judge_stage(
        present=depth is not None and raw_hash is not None,
        content_recorded=depth is not None and depth.content_sha256 is not None,
        content_ok=bool(depth and raw_hash == depth.content_sha256),
        settings_recorded=depth is not None and depth.settings_sha256 is not None,
        settings_ok=raw_settings_ok,
        input_recorded=depth is not None and depth.source_sha256 is not None,
        input_ok=raw_input_ok,
        require_settings=True,
        require_input=True,
        upstream=source_status,
    )

    conditioning = None if depth is None else depth.conditioning
    conditioned_path = (
        None
        if conditioning is None
        else directory / "depth" / PROCESSING_DIRECTORY / CONDITIONED_FILENAME
    )
    conditioned_hash = None if conditioned_path is None else file_sha256(conditioned_path)
    conditioned_settings_ok = False
    if conditioning is not None and conditioning.settings_sha256 is not None:
        conditioned_settings_ok = conditioning.settings_sha256 == canonical_sha256(
            conditioning_settings_document(conditioning.parameters)
        )
    conditioned_input_ok = bool(
        conditioning and conditioned_hash is not None and conditioning.raw_sha256 == raw_hash
    )
    conditioned_status = judge_stage(
        present=conditioning is not None and conditioned_hash is not None,
        content_recorded=conditioning is not None and conditioning.content_sha256 is not None,
        content_ok=bool(conditioning and conditioned_hash == conditioning.content_sha256),
        settings_recorded=conditioning is not None and conditioning.settings_sha256 is not None,
        settings_ok=conditioned_settings_ok,
        input_recorded=conditioning is not None and conditioning.raw_sha256 is not None,
        input_ok=conditioned_input_ok,
        require_settings=True,
        require_input=True,
        upstream=raw_status,
    )

    relief = project.relief
    relief_file = directory / "depth" / "composition" / RELIEF_FILENAME
    if relief_file.is_symlink():
        raise UnsafeProjectPath
    relief_path = (
        relief_file
        if relief is not None and relief.generated_at is not None and relief.relief_filename == RELIEF_FILENAME
        else None
    )
    relief_hash = None if relief_path is None else file_sha256(relief_path)
    relief_settings_ok = False
    relief_settings_recorded = relief is not None and relief.settings_sha256 is not None
    if relief_settings_recorded and relief is not None and relief.settings_sha256 is not None:
        try:
            relief_settings_ok = relief.settings_sha256 == canonical_sha256(
                _relief_settings_document(
                    directory,
                    relief,
                    project.artwork.width_mm,
                    project.artwork.height_mm,
                )
            )
        except (OSError, UnsafeProjectPath, ValueError):
            relief_settings_ok = False
    relief_input_ok = bool(
        relief and relief_hash is not None and relief.conditioned_sha256 == conditioned_hash
    )
    relief_status = judge_stage(
        present=relief_hash is not None,
        content_recorded=relief is not None and relief.content_sha256 is not None,
        content_ok=bool(relief and relief_hash == relief.content_sha256),
        settings_recorded=relief_settings_recorded,
        settings_ok=relief_settings_ok,
        input_recorded=relief is not None and relief.conditioned_sha256 is not None,
        input_ok=relief_input_ok,
        require_settings=True,
        require_input=True,
        upstream=conditioned_status,
    )

    physical = None if relief is None else relief.physical
    physical_status = _physical_status(directory, physical, relief_hash, relief_status)
    return {
        "source": source_status,
        "raw": raw_status,
        "conditioned": conditioned_status,
        "relief": relief_status,
        "physical": physical_status,
    }


def _physical_status(
    directory: Path,
    record: PhysicalHeightRecord | None,
    relief_hash: str | None,
    relief_status: ArtifactStatus,
) -> ArtifactStatus:
    if record is None:
        return "missing"
    path = directory / "depth" / "composition" / "physical" / "physical_height_mm.npy"
    if path.is_symlink():
        raise UnsafeProjectPath
    content_hash = file_sha256(path)
    settings_ok = False
    if record.settings_sha256 is not None:
        settings_ok = record.settings_sha256 == canonical_sha256(physical_settings_document(record))
    return judge_stage(
        present=content_hash is not None,
        content_recorded=record.content_sha256 is not None,
        content_ok=content_hash == record.content_sha256,
        settings_recorded=record.settings_sha256 is not None,
        settings_ok=settings_ok,
        input_recorded=True,
        input_ok=relief_hash is not None and record.source_relief_sha256 == relief_hash,
        require_settings=True,
        require_input=True,
        upstream=relief_status,
    )


def attach_artifact_status(directory: Path, project: Project) -> Project:
    """Attach response-only status. This does not write the manifest."""

    statuses = resolve_artifact_statuses(directory, project)
    updates: dict[str, object] = {}
    if project.source is not None:
        updates["source"] = project.source.model_copy(update={"status": statuses["source"]})
    if project.depth is not None:
        conditioning = project.depth.conditioning
        if conditioning is not None:
            conditioning = conditioning.model_copy(update={"status": statuses["conditioned"]})
        updates["depth"] = project.depth.model_copy(
            update={"status": statuses["raw"], "conditioning": conditioning}
        )
    if project.relief is not None:
        physical = project.relief.physical
        if physical is not None:
            physical = physical.model_copy(update={"status": statuses["physical"]})
        updates["relief"] = project.relief.model_copy(
            update={"status": statuses["relief"], "physical": physical}
        )
    if not updates:
        return project
    return project.model_copy(update=updates)


def fingerprint_depth(artifact: DepthArtifact, raw_sha256: str, source_sha256: str) -> DepthArtifact:
    return artifact.model_copy(
        update={
            "content_sha256": raw_sha256,
            "source_sha256": source_sha256,
            "settings_sha256": canonical_sha256(depth_settings_document(artifact)),
            "status": None,
        }
    )


def fingerprint_conditioning(
    record: ConditioningRecord,
    content_sha256: str,
    raw_sha256: str,
) -> ConditioningRecord:
    return record.model_copy(
        update={
            "content_sha256": content_sha256,
            "raw_sha256": raw_sha256,
            "settings_sha256": canonical_sha256(conditioning_settings_document(record.parameters)),
            "status": None,
        }
    )


def _relief_settings_document(
    directory: Path,
    relief: ReliefRecord,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> dict[str, object]:
    """Select the document shape that was current when this record was written.

    A missing settings version is the historical shape. Nonzero detail, a
    physical scale, or a non-identity background on that record cannot match,
    because the historical hash never recorded them. Version 2 records detail
    amount and, when needed, the source hash. It does not record a scale or a
    background. A version-2 record that now has either cannot match. Version 3
    records the scale and the artwork millimeters, and rejects a non-identity
    background. Version 4 records the background fields.
    """

    version = relief.settings_version
    identity = _background_identity(
        relief.background_depth_range,
        relief.background_elevation,
        relief.background_feather,
    )
    if version is None:
        if any(region.detail_amount != 0.0 for region in relief.regions):
            raise ValueError(version)
        if any(region.detail_scale_mm is not None for region in relief.regions):
            raise ValueError(version)
        if not identity:
            raise ValueError(version)
        return composition_settings_document(directory, relief.regions)
    if version == COMPOSITION_SETTINGS_VERSION:
        if any(region.detail_scale_mm is not None for region in relief.regions):
            raise ValueError(version)
        if not identity:
            raise ValueError(version)
        source_sha256 = None
        if any(region.detail_amount > 0.0 for region in relief.regions):
            source_sha256 = composition_source_sha256(directory)
        return composition_settings_document(
            directory,
            relief.regions,
            version=version,
            source_sha256=source_sha256,
        )
    if version == BACKGROUND_SETTINGS_VERSION:
        source_sha256 = None
        if any(region.detail_amount > 0.0 for region in relief.regions):
            source_sha256 = composition_source_sha256(directory)
        return composition_settings_document(
            directory,
            relief.regions,
            version=version,
            source_sha256=source_sha256,
            artwork_width_mm=artwork_width_mm,
            artwork_height_mm=artwork_height_mm,
            background_depth_range=relief.background_depth_range,
            background_elevation=relief.background_elevation,
            background_feather=relief.background_feather,
        )
    if version != PHYSICAL_SCALE_SETTINGS_VERSION:
        raise ValueError(version)
    if not identity:
        raise ValueError(version)
    source_sha256 = None
    if any(region.detail_amount > 0.0 for region in relief.regions):
        source_sha256 = composition_source_sha256(directory)
    return composition_settings_document(
        directory,
        relief.regions,
        version=version,
        source_sha256=source_sha256,
        artwork_width_mm=artwork_width_mm,
        artwork_height_mm=artwork_height_mm,
    )


def fingerprint_relief(
    record: ReliefRecord,
    directory: Path,
    content_sha256: str,
    conditioned_sha256: str,
    artwork_width_mm: float,
    artwork_height_mm: float,
) -> ReliefRecord:
    """Stamp version 2, version 3 for a physical scale, or version 4 for background.

    Historical records are not rewritten here. Identity background settings
    keep version 2, or version 3 when a scale is selected. Those documents do
    not gain background keys.
    """

    if not _background_identity(
        record.background_depth_range,
        record.background_elevation,
        record.background_feather,
    ):
        version: int = BACKGROUND_SETTINGS_VERSION
    elif any(region.detail_scale_mm is not None for region in record.regions):
        version = PHYSICAL_SCALE_SETTINGS_VERSION
    else:
        version = COMPOSITION_SETTINGS_VERSION
    source_sha256 = None
    if any(region.detail_amount > 0.0 for region in record.regions):
        source_sha256 = composition_source_sha256(directory)
    return record.model_copy(
        update={
            "content_sha256": content_sha256,
            "conditioned_sha256": conditioned_sha256,
            "settings_sha256": canonical_sha256(
                composition_settings_document(
                    directory,
                    record.regions,
                    version=version,
                    source_sha256=source_sha256,
                    artwork_width_mm=artwork_width_mm,
                    artwork_height_mm=artwork_height_mm,
                    background_depth_range=record.background_depth_range,
                    background_elevation=record.background_elevation,
                    background_feather=record.background_feather,
                )
            ),
            "settings_version": version,
            "status": None,
        }
    )


def fingerprint_physical(record: PhysicalHeightRecord, content_sha256: str) -> PhysicalHeightRecord:
    return record.model_copy(
        update={
            "content_sha256": content_sha256,
            "settings_sha256": canonical_sha256(physical_settings_document(record)),
            "status": None,
        }
    )
