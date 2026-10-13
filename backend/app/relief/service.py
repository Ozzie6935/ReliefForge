"""Persist relief regions and compose them from conditioned depth."""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
from PIL import Image

from app.depth.artifacts import conditioned_depth_file
from app.depth.preview import unit_interval_preview
from app.projects.errors import (
    ArtifactDependencyRejected,
    ComposedReliefNotFound,
    ConditionedDepthNotFound,
    ConditionedPreviewNotFound,
    InvalidPhysicalHeight,
    InvalidReliefRequest,
    MaskSaveFailed,
    PhysicalHeightFailed,
    PhysicalHeightNotFound,
    PhysicalPreviewNotFound,
    ReliefCompositionFailed,
    ReliefRegionNotFound,
    SegmentationProposalNotFound,
)
from app.projects.provenance import (
    CONDITIONED_STALE_DETAIL,
    CONDITIONED_UNKNOWN_DETAIL,
    RELIEF_STALE_DETAIL,
    RELIEF_UNKNOWN_DETAIL,
    file_sha256,
    fingerprint_physical,
    fingerprint_relief,
)
from app.projects.storage import ProjectStore
from app.relief.export_plan import (
    STALE_DETAIL,
    UNKNOWN_DETAIL,
    build_export_readiness,
    validate_export_request,
    validate_reduced_grid,
)
from app.relief.stl_export import StlDownload, generate_binary_stl, raise_blocked_export, stl_filename
from app.relief.diagnostics import (
    FILE_MISSING_DETAIL,
    RECORD_MISSING_DETAIL,
    diagnose_physical_height,
    load_physical_array,
    require_physical_array,
)
from app.relief.mesh import DEFAULT_MAX_VERTICES, describe_mesh_preview
from app.relief.preview_binary import MeshPreviewBinary, build_preview_bytes
from app.relief.solid import describe_solid
from app.relief.profile import ProfileMode, build_physical_profile
from app.relief.artifacts import (
    AI_MASK_FILENAME,
    SESSION_BASE_FILENAME,
    WORKING_MASK_FILENAME,
    atomic_replace_bytes,
    copy_mask_file,
    existing_mask,
    existing_proposal,
    list_proposal_ids,
    mask_path,
    proposal_directory,
    read_mask_png,
    read_proposal_metadata,
    remove_proposal,
    write_mask_png,
    write_proposal_metadata,
    write_relief_files,
)
from app.relief.composition import (
    CompositionError,
    RegionContribution,
    apply_background,
    compose_relief,
    prepare_detail_signals,
)
from app.relief.physical import (
    HEIGHT_FILENAME,
    build_physical_record,
    fixed_preview_png,
    load_relief_array,
    map_physical_height,
    npy_bytes,
    physical_directory,
    relief_file,
    resolve_physical_status,
    restore_physical,
    sha256_bytes,
    snapshot_physical,
    stats_text,
    validate_physical_settings,
    write_physical_bundle,
)
from app.schemas.project import Project
from app.schemas.relief import (
    ExportReadiness,
    MeshPreviewInfo,
    MeshSolidInfo,
    PhysicalHeightDiagnostics,
    PhysicalHeightProfile,
    PhysicalHeightRequest,
    RegionMaskEditDiscard,
    RegionMaskEditSave,
    ReliefCompositionRequest,
    ReliefRecord,
    ReliefRegionCreate,
    ReliefRegionRecord,
    SegmentationProposal,
)


class ReliefService:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def get_relief(self, project_id: UUID) -> ReliefRecord:
        project = self._require_conditioned(project_id)
        if project.relief is not None:
            return project.relief
        return ReliefRecord(regions=[])

    def accept_region(self, project_id: UUID, request: ReliefRegionCreate) -> Project:
        project = self._require_conditioned(project_id)
        directory = self.store.project_directory(project.slug)
        proposal = existing_proposal(directory, request.proposal_id)
        mask = read_mask_png(proposal)
        conditioned = self._load_conditioned(project)
        if mask.shape != conditioned.shape:
            raise ReliefCompositionFailed
        current = list(project.relief.regions) if project.relief is not None else []
        now = datetime.now(timezone.utc)
        region_id = uuid4()
        order = 0 if not current else max(region.order for region in current) + 1
        record = ReliefRegionRecord(
            id=region_id,
            name=request.name.strip(),
            role=request.role,
            mask_filename=f"{region_id}.png",
            elevation=0.0,
            depth_range=1.0,
            feather=0.0,
            order=order,
            created_at=now,
            updated_at=now,
        )
        if not record.name:
            raise InvalidReliefRequest
        destination = mask_path(directory, region_id)
        write_mask_png(destination, mask)
        remove_proposal(directory, request.proposal_id)
        try:
            return self._compose(project, [*current, record])
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def apply_composition(self, project_id: UUID, request: ReliefCompositionRequest) -> Project:
        project = self._require_conditioned(project_id)
        current = [] if project.relief is None else list(project.relief.regions)
        by_id = {region.id: region for region in current}
        if {item.id for item in request.regions} != set(by_id):
            raise ReliefRegionNotFound
        if len({item.order for item in request.regions}) != len(request.regions):
            raise InvalidReliefRequest
        now = datetime.now(timezone.utc)
        updated: list[ReliefRegionRecord] = []
        for item in request.regions:
            if not item.name.strip():
                raise InvalidReliefRequest
            previous = by_id[item.id]
            updated.append(
                previous.model_copy(
                    update={
                        "name": item.name.strip(),
                        "role": item.role,
                        "elevation": item.elevation,
                        "depth_range": item.depth_range,
                        "feather": item.feather,
                        "order": item.order,
                        "detail_amount": item.detail_amount,
                        "detail_scale_mm": item.detail_scale_mm,
                        "updated_at": now,
                    }
                )
            )
        return self._compose(
            project,
            updated,
            background_depth_range=request.background_depth_range,
            background_elevation=request.background_elevation,
            background_feather=request.background_feather,
        )

    def begin_region_mask_edit(self, project_id: UUID, region_id: UUID) -> SegmentationProposal:
        """Copy an accepted region mask into a proposal. The permanent mask is not written."""

        project = self._require_conditioned(project_id)
        region = self._region(project, region_id)
        directory = self.store.project_directory(project.slug)
        accepted = existing_mask(directory, region.id, region.mask_filename)
        mask = read_mask_png(accepted)
        source = self.store.source_path(project)
        with Image.open(source) as handle:
            if mask.shape != (handle.height, handle.width):
                raise InvalidReliefRequest
        self._remove_region_edits(directory, region.id)
        proposal_id = uuid4()
        folder = proposal_directory(directory, proposal_id)
        folder.mkdir(parents=True, exist_ok=False)
        try:
            payload = accepted.read_bytes()
            for name in (AI_MASK_FILENAME, WORKING_MASK_FILENAME, SESSION_BASE_FILENAME):
                atomic_replace_bytes(folder / name, payload)
            write_proposal_metadata(
                directory,
                proposal_id,
                region.name,
                region.role,
                origin_type="accepted_region",
                region_id=str(region.id),
            )
        except Exception:
            if folder.exists():
                shutil.rmtree(folder)
            raise
        if accepted.read_bytes() != payload:
            shutil.rmtree(folder)
            raise MaskSaveFailed
        return SegmentationProposal(
            proposal_id=proposal_id,
            width_px=mask.shape[1],
            height_px=mask.shape[0],
            model_id="accepted-region",
            name=region.name,
            role=region.role,
            origin_type="accepted_region",
            region_id=region.id,
        )

    def save_region_mask_edit(self, project_id: UUID, region_id: UUID, request: RegionMaskEditSave) -> Project:
        """Replace one accepted mask with the edit working mask. Composition is not rebuilt."""

        project = self._require_conditioned(project_id)
        region = self._region(project, region_id)
        directory = self.store.project_directory(project.slug)
        self._require_region_edit(directory, request.proposal_id, region.id)
        working = existing_proposal(directory, request.proposal_id)
        permanent = existing_mask(directory, region.id, region.mask_filename)
        edited = read_mask_png(working)
        saved = read_mask_png(permanent)
        if edited.shape != saved.shape:
            raise InvalidReliefRequest
        name = request.name.strip()
        if not name:
            raise InvalidReliefRequest
        previous = permanent.read_bytes()
        relief = project.relief
        if relief is None:
            raise ReliefRegionNotFound
        updated_region = region.model_copy(
            update={
                "name": name,
                "role": request.role,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        regions = [updated_region if item.id == region.id else item for item in relief.regions]
        record = relief.model_copy(update={"regions": regions})
        try:
            copy_mask_file(working, permanent)
            saved_project = self.store.save_relief(project, record)
        except Exception as exc:
            atomic_replace_bytes(permanent, previous)
            raise MaskSaveFailed from exc
        remove_proposal(directory, request.proposal_id)
        return saved_project

    def discard_region_mask_edit(self, project_id: UUID, region_id: UUID, request: RegionMaskEditDiscard) -> None:
        """Remove an edit proposal. The accepted region mask stays as it is."""

        project = self.store.get_by_id(project_id)
        region = self._region(project, region_id)
        directory = self.store.project_directory(project.slug)
        permanent = existing_mask(directory, region.id, region.mask_filename)
        before = permanent.read_bytes()
        self._require_region_edit(directory, request.proposal_id, region.id)
        remove_proposal(directory, request.proposal_id)
        if permanent.read_bytes() != before:
            raise MaskSaveFailed

    def _remove_region_edits(self, directory: Path, region_id: UUID) -> None:
        for proposal_id in list_proposal_ids(directory):
            metadata = read_proposal_metadata(directory, proposal_id)
            if metadata is None or metadata.origin_type != "accepted_region":
                continue
            if metadata.region_id == str(region_id):
                remove_proposal(directory, proposal_id)

    def _require_region_edit(self, directory: Path, proposal_id: UUID, region_id: UUID) -> None:
        metadata = read_proposal_metadata(directory, proposal_id)
        if (
            metadata is None
            or metadata.origin_type != "accepted_region"
            or metadata.region_id != str(region_id)
        ):
            raise InvalidReliefRequest
        existing_proposal(directory, proposal_id)

    def delete_region(self, project_id: UUID, region_id: UUID) -> Project:
        project = self._require_conditioned(project_id)
        current = [] if project.relief is None else list(project.relief.regions)
        kept = [region for region in current if region.id != region_id]
        if len(kept) == len(current):
            raise ReliefRegionNotFound
        directory = self.store.project_directory(project.slug)
        removed = next(region for region in current if region.id == region_id)
        path = existing_mask(directory, removed.id, removed.mask_filename)
        updated = self._compose(project, kept)
        path.unlink()
        return updated

    def mask_file(self, project_id: UUID, region_id: UUID) -> Path:
        project = self.store.get_by_id(project_id)
        region = self._region(project, region_id)
        directory = self.store.project_directory(project.slug)
        return existing_mask(directory, region.id, region.mask_filename)

    def preview_file(self, project_id: UUID) -> Path:
        project = self.store.get_by_id(project_id)
        relief = project.relief
        if relief is None or relief.preview_filename != "preview.png" or relief.generated_at is None:
            raise ConditionedPreviewNotFound
        path = self.store.project_directory(project.slug) / "depth" / "composition" / "preview.png"
        if path.is_symlink() or not path.is_file():
            raise ConditionedPreviewNotFound
        if not path.resolve().is_relative_to(self.store.project_directory(project.slug).resolve()):
            raise ConditionedPreviewNotFound
        return path

    def generate_physical_height(self, project_id: UUID, request: PhysicalHeightRequest) -> Project:
        """Map the saved composed relief. Upstream files are not rewritten."""

        validate_physical_settings(
            request.stock_thickness_mm,
            request.base_thickness_mm,
            request.maximum_surface_height_mm,
        )
        project = self.store.get_by_id(project_id)
        relief = project.relief
        if (
            relief is None
            or relief.generated_at is None
            or relief.relief_filename != "relief_depth.npy"
        ):
            raise ComposedReliefNotFound
        directory = self.store.project_directory(project.slug)
        path = relief_file(directory)
        if not path.is_file():
            raise ComposedReliefNotFound
        original = path.read_bytes()
        try:
            array = load_relief_array(original)
            before = array.tobytes()
            height = map_physical_height(
                array,
                request.base_thickness_mm,
                request.maximum_surface_height_mm,
            )
            preview = fixed_preview_png(
                height,
                request.base_thickness_mm,
                request.maximum_surface_height_mm,
            )
        except InvalidPhysicalHeight:
            raise
        except Exception as exc:
            raise InvalidPhysicalHeight("relief") from exc
        if array.tobytes() != before or path.read_bytes() != original:
            raise PhysicalHeightFailed
        if relief.status == "stale":
            raise ArtifactDependencyRejected(RELIEF_STALE_DETAIL)
        if relief.status != "current":
            raise ArtifactDependencyRejected(RELIEF_UNKNOWN_DETAIL)
        record = fingerprint_physical(
            build_physical_record(
                relief,
                sha256_bytes(original),
                height,
                request.stock_thickness_mm,
                request.base_thickness_mm,
                request.maximum_surface_height_mm,
                datetime.now(timezone.utc),
            ),
            sha256_bytes(npy_bytes(height)),
        )
        snapshot = snapshot_physical(directory)
        try:
            write_physical_bundle(directory, height, preview, stats_text(record))
            written = file_sha256(directory / "depth" / "composition" / "physical" / "physical_height_mm.npy")
            if path.read_bytes() != original or written != record.content_sha256:
                raise PhysicalHeightFailed
            saved = self.store.save_relief(project, relief.model_copy(update={"physical": record}))
        except Exception as exc:
            restore_physical(directory, snapshot)
            partial = directory / ".project.json.partial"
            if partial.is_file() and not partial.is_symlink():
                partial.unlink()
            if isinstance(exc, (InvalidPhysicalHeight, ComposedReliefNotFound)):
                raise
            raise PhysicalHeightFailed from exc
        return saved

    def physical_preview_file(self, project_id: UUID) -> Path:
        project = self.store.get_by_id(project_id)
        relief = project.relief
        physical = None if relief is None else relief.physical
        if physical is None or physical.preview_path != "depth/composition/physical/preview.png":
            raise PhysicalPreviewNotFound
        directory = self.store.project_directory(project.slug)
        path = directory / physical.preview_path
        if path.is_symlink() or not path.is_file():
            raise PhysicalPreviewNotFound
        if not path.resolve().is_relative_to(directory.resolve()):
            raise PhysicalPreviewNotFound
        return path

    def physical_height_diagnostics(self, project_id: UUID) -> PhysicalHeightDiagnostics:
        """Read the saved height array. This does not generate or rewrite it."""

        project, record, status, data = self._read_physical_height(project_id)
        return diagnose_physical_height(project.id, status, record, data)

    def physical_height_profile(
        self,
        project_id: UUID,
        *,
        mode: ProfileMode,
        y_px: float | None = None,
        x_px: float | None = None,
        start_x_px: float | None = None,
        start_y_px: float | None = None,
        end_x_px: float | None = None,
        end_y_px: float | None = None,
        sample_count: int = 256,
    ) -> PhysicalHeightProfile:
        """Sample the saved height array. This does not generate or rewrite it."""

        project, record, status, data = self._read_physical_height(project_id)
        array = load_physical_array(data)
        require_physical_array(array, record)
        return build_physical_profile(
            project_id=project.id,
            status=status,
            array=array,
            artwork_width_mm=project.artwork.width_mm,
            artwork_height_mm=project.artwork.height_mm,
            mode=mode,
            y_px=y_px,
            x_px=x_px,
            start_x_px=start_x_px,
            start_y_px=start_y_px,
            end_x_px=end_x_px,
            end_y_px=end_y_px,
            sample_count=sample_count,
        )

    def mesh_preview_info(
        self,
        project_id: UUID,
        max_vertices: int = DEFAULT_MAX_VERTICES,
    ) -> MeshPreviewInfo:
        """Describe a top-surface preview. This does not build or store a mesh."""

        project, record, status, data = self._read_physical_height(project_id)
        array = load_physical_array(data)
        require_physical_array(array, record)
        return describe_mesh_preview(
            project_id=project.id,
            status=status,
            array=array,
            artwork_width_mm=project.artwork.width_mm,
            artwork_height_mm=project.artwork.height_mm,
            max_vertices=max_vertices,
            stock_thickness_mm=record.stock_thickness_mm,
        )

    def mesh_solid_info(
        self,
        project_id: UUID,
        max_vertices: int = DEFAULT_MAX_VERTICES,
    ) -> MeshSolidInfo:
        """Describe a watertight solid. This does not build or store a mesh."""

        project, record, status, data = self._read_physical_height(project_id)
        array = load_physical_array(data)
        require_physical_array(array, record)
        return describe_solid(
            project_id=project.id,
            status=status,
            array=array,
            artwork_width_mm=project.artwork.width_mm,
            artwork_height_mm=project.artwork.height_mm,
            max_vertices=max_vertices,
            stock_thickness_mm=record.stock_thickness_mm,
        )

    def mesh_preview_binary(
        self,
        project_id: UUID,
        max_vertices: int = DEFAULT_MAX_VERTICES,
    ) -> MeshPreviewBinary:
        """Build one bounded solid preview. This does not store a mesh or rewrite provenance."""

        project, record, status, data = self._read_physical_height(project_id)
        array = load_physical_array(data)
        require_physical_array(array, record)
        return build_preview_bytes(
            array,
            project.artwork.width_mm,
            project.artwork.height_mm,
            status=status,
            max_vertices=max_vertices,
            stock_thickness_mm=record.stock_thickness_mm,
            source_bytes=array.nbytes + len(data),
        )

    def export_readiness(
        self,
        project_id: UUID,
        mode: Literal["native", "reduced"] = "native",
        height_px: int | None = None,
        width_px: int | None = None,
    ) -> ExportReadiness:
        """Plan an artwork-local STL. This does not write a file or build a mesh."""

        validate_export_request(mode, height_px, width_px)
        project = self.store.get_by_id(project_id)
        relief = project.relief
        record = None if relief is None else relief.physical
        array = None
        file_present = False
        status: Literal["missing", "unknown", "current", "stale"] = "missing"
        if record is not None:
            directory = self.store.project_directory(project.slug)
            status = resolve_physical_status(directory, project)
            if mode == "reduced" and height_px is not None and width_px is not None:
                validate_reduced_grid(record.height_px, record.width_px, height_px, width_px)
            path = physical_directory(directory) / HEIGHT_FILENAME
            if path.is_symlink() or not path.is_file():
                file_present = False
            elif not path.resolve().is_relative_to(directory.resolve()):
                file_present = False
            else:
                array = load_physical_array(path.read_bytes())
                require_physical_array(array, record)
                file_present = True
        return build_export_readiness(
            project_id=project.id,
            provenance_status=status,
            mode=mode,
            artwork_width_mm=project.artwork.width_mm,
            artwork_height_mm=project.artwork.height_mm,
            stock_width_mm=project.stock.width_mm,
            stock_height_mm=project.stock.height_mm,
            center_x_mm=project.artwork.center_x_mm,
            center_y_mm=project.artwork.center_y_mm,
            record=record,
            file_present=file_present,
            array=array,
            export_height_px=height_px,
            export_width_px=width_px,
        )

    def export_stl(
        self,
        project_id: UUID,
        mode: Literal["native", "reduced"] = "native",
        height_px: int | None = None,
        width_px: int | None = None,
    ) -> StlDownload:
        """Validate the saved artifact again, then return one STL. Nothing is stored."""

        validate_export_request(mode, height_px, width_px)
        report = self.export_readiness(
            project_id,
            mode=mode,
            height_px=height_px,
            width_px=width_px,
        )
        if not report.ready_for_export:
            raise_blocked_export(report)
        project, record, status, data = self._read_physical_height(project_id)
        if status != "current":
            if status == "stale":
                raise ArtifactDependencyRejected(STALE_DETAIL)
            if status == "unknown":
                raise ArtifactDependencyRejected(UNKNOWN_DETAIL)
            raise PhysicalHeightNotFound(RECORD_MISSING_DETAIL)
        array = load_physical_array(data)
        require_physical_array(array, record)
        document = generate_binary_stl(
            array,
            project.artwork.width_mm,
            project.artwork.height_mm,
            stock_thickness_mm=record.stock_thickness_mm,
            mode=mode,
            export_height_px=height_px,
            export_width_px=width_px,
        )
        return StlDownload(document=document, filename=stl_filename(project.slug))

    def _read_physical_height(self, project_id: UUID):
        project = self.store.get_by_id(project_id)
        relief = project.relief
        record = None if relief is None else relief.physical
        if record is None or record.height_path != "depth/composition/physical/physical_height_mm.npy":
            raise PhysicalHeightNotFound(RECORD_MISSING_DETAIL)
        directory = self.store.project_directory(project.slug)
        status = resolve_physical_status(directory, project)
        path = physical_directory(directory) / HEIGHT_FILENAME
        if path.is_symlink() or not path.is_file():
            raise PhysicalHeightNotFound(FILE_MISSING_DETAIL)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise PhysicalHeightNotFound(FILE_MISSING_DETAIL)
        return project, record, status, path.read_bytes()

    def _require_current_conditioning(self, project: Project) -> None:
        conditioning = None if project.depth is None else project.depth.conditioning
        if conditioning is None or conditioning.status in {None, "missing"}:
            raise ConditionedDepthNotFound
        if conditioning.status == "stale":
            raise ArtifactDependencyRejected(CONDITIONED_STALE_DETAIL)
        if conditioning.status != "current":
            raise ArtifactDependencyRejected(CONDITIONED_UNKNOWN_DETAIL)

    def _compose(
        self,
        project: Project,
        regions: list[ReliefRegionRecord],
        *,
        background_depth_range: float | None = None,
        background_elevation: float | None = None,
        background_feather: float | None = None,
    ) -> Project:
        self._require_current_conditioning(project)
        stored = project.relief
        if background_depth_range is None:
            background_depth_range = 1.0 if stored is None else stored.background_depth_range
        if background_elevation is None:
            background_elevation = 0.0 if stored is None else stored.background_elevation
        if background_feather is None:
            background_feather = 0.0 if stored is None else stored.background_feather
        directory = self.store.project_directory(project.slug)
        composition = directory / "depth" / "composition"
        relief_path = composition / "relief_depth.npy"
        preview_path = composition / "preview.png"
        snapshot = {
            relief_path: self.store._read_snapshot(relief_path),
            preview_path: self.store._read_snapshot(preview_path),
        }
        conditioned = self._load_conditioned(project)
        contributions: list[RegionContribution] = []
        for region in regions:
            path = existing_mask(directory, region.id, region.mask_filename)
            mask = read_mask_png(path)
            contributions.append(
                RegionContribution(
                    region_id=str(region.id),
                    mask=mask,
                    elevation=region.elevation,
                    depth_range=region.depth_range,
                    feather=region.feather,
                    order=region.order,
                    detail_amount=region.detail_amount,
                    detail_scale_mm=region.detail_scale_mm,
                )
            )
        try:
            detail = None
            physical_details = None
            if any(region.detail_amount > 0.0 for region in regions):
                rgb = self._read_source_rgb(project, conditioned.shape)
                try:
                    detail, physical_details = prepare_detail_signals(
                        rgb,
                        contributions,
                        artwork_width_mm=project.artwork.width_mm,
                        artwork_height_mm=project.artwork.height_mm,
                    )
                except CompositionError as exc:
                    raise InvalidReliefRequest from exc
            relief = compose_relief(conditioned, contributions, detail, physical_details)
            try:
                relief = apply_background(
                    conditioned,
                    relief,
                    [item.mask for item in contributions],
                    depth_range=background_depth_range,
                    elevation=background_elevation,
                    feather=background_feather,
                )
            except CompositionError as exc:
                raise InvalidReliefRequest from exc
            preview = unit_interval_preview(relief)
            write_relief_files(directory, relief, preview)
            content_hash = file_sha256(relief_path)
            conditioned_hash = file_sha256(directory / "depth" / "processing" / "conditioned_depth.npy")
            if content_hash is None or conditioned_hash is None:
                raise ReliefCompositionFailed
            conditioning = project.depth.conditioning if project.depth is not None else None
            previous = None if project.relief is None else project.relief.physical
            record = fingerprint_relief(
                ReliefRecord(
                    regions=sorted(regions, key=lambda item: (item.order, str(item.id))),
                    background_depth_range=background_depth_range,
                    background_elevation=background_elevation,
                    background_feather=background_feather,
                    conditioned_generated_at=None if conditioning is None else conditioning.generated_at,
                    width_px=int(relief.shape[1]),
                    height_px=int(relief.shape[0]),
                    dtype="float32",
                    range_min=float(relief.min()),
                    range_max=float(relief.max()),
                    relief_filename="relief_depth.npy",
                    preview_filename="preview.png",
                    generated_at=datetime.now(timezone.utc),
                    physical=None if previous is None else previous.model_copy(update={"status": None}),
                ),
                directory,
                content_hash,
                conditioned_hash,
                project.artwork.width_mm,
                project.artwork.height_mm,
            )
            return self.store.save_relief(project, record)
        except (ReliefCompositionFailed, InvalidReliefRequest):
            self.store._restore_snapshot(snapshot)
            raise
        except (CompositionError, Exception) as exc:
            self.store._restore_snapshot(snapshot)
            raise ReliefCompositionFailed from exc

    def _require_conditioned(self, project_id: UUID) -> Project:
        project = self.store.get_by_id(project_id)
        if project.depth is None or project.depth.conditioning is None:
            raise ConditionedDepthNotFound
        return project

    def _read_source_rgb(self, project: Project, expected_shape: tuple[int, ...]) -> np.ndarray:
        """Read stored source pixels only when a region asked for fine detail."""

        source = self.store.source_path(project)
        with Image.open(source) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        if rgb.ndim != 3 or rgb.shape[0] != expected_shape[0] or rgb.shape[1] != expected_shape[1]:
            raise InvalidReliefRequest
        if rgb.dtype != np.uint8:
            raise InvalidReliefRequest
        return rgb

    def _load_conditioned(self, project: Project) -> np.ndarray:
        directory = self.store.project_directory(project.slug)
        try:
            path = conditioned_depth_file(directory, project)
        except ConditionedPreviewNotFound as exc:
            raise ConditionedDepthNotFound from exc
        array = np.load(path)
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise ReliefCompositionFailed
        return array

    @staticmethod
    def _region(project: Project, region_id: UUID) -> ReliefRegionRecord:
        regions = [] if project.relief is None else project.relief.regions
        for region in regions:
            if region.id == region_id:
                return region
        raise ReliefRegionNotFound
