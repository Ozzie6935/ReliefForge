"""Propose a mask from source-image prompts. Proposals are not regions."""

from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from PIL import Image

from app.projects.errors import (
    ConditionedDepthNotFound,
    InvalidMagicWand,
    InvalidMaskStroke,
    InvalidSegmentationPrompt,
    MagicWandFailed,
    MaskRefinementFailed,
    SegmentationFailed,
    SegmentationProposalNotFound,
)
from app.projects.storage import ProjectStore
from app.relief.artifacts import (
    clear_other_proposals,
    existing_ai_mask,
    existing_proposal,
    list_proposal_ids,
    read_mask_png,
    read_proposal_metadata,
    remove_proposal,
    replace_working_mask,
    reset_working_mask,
    restore_session_baseline,
    start_refinement_session,
    write_proposal_masks,
    write_proposal_metadata,
)
from app.schemas.relief import (
    ImageBox,
    ImagePoint,
    MagicWandRequest,
    MaskStrokeRequest,
    ProposalDraftUpdate,
    SegmentationProposal,
    SegmentationRequest,
)
from app.segmentation.magic_wand import MagicWandError, paint_selection, select_similar
from app.segmentation.model import SegmentationModel, SegmentationPrompt
from app.segmentation.refinement import MaskStrokeError, Stroke, apply_stroke
from app.segmentation.registry import MODEL_ID, get_segmentation_model


class SegmentationService:
    def __init__(self, store: ProjectStore, model: SegmentationModel | None = None) -> None:
        self.store = store
        self.model = model if model is not None else get_segmentation_model(MODEL_ID)

    def predict(self, project_id: UUID, request: SegmentationRequest) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        if project.depth is None or project.depth.conditioning is None:
            raise ConditionedDepthNotFound
        source = self.store.source_path(project)
        with Image.open(source) as handle:
            image = handle.convert("RGB").copy()
        _require_points_inside(request.positive_points, image.width, image.height)
        _require_points_inside(request.negative_points, image.width, image.height)
        if request.box is not None:
            _require_box_inside(request.box, image.width, image.height)
        prompt = SegmentationPrompt(
            positive_points=tuple((point.x, point.y) for point in request.positive_points),
            negative_points=tuple((point.x, point.y) for point in request.negative_points),
            box=None
            if request.box is None
            else (request.box.x0, request.box.y0, request.box.x1, request.box.y1),
        )
        stat = source.stat()
        cache_key = f"{project.id}:{stat.st_mtime_ns}:{stat.st_size}"
        try:
            mask = self.model.predict(image, prompt, cache_key=cache_key)
        except (ConditionedDepthNotFound, InvalidSegmentationPrompt):
            raise
        except Exception as exc:
            raise SegmentationFailed from exc
        if mask.shape != (image.height, image.width) or mask.dtype != bool:
            raise SegmentationFailed
        proposal_id = uuid4()
        directory = self.store.project_directory(project.slug)
        write_proposal_masks(directory, proposal_id, mask)
        write_proposal_metadata(directory, proposal_id, request.name, request.role)
        clear_other_proposals(directory, proposal_id)
        return self._proposal(directory, proposal_id, image.width, image.height)

    def proposal_file(self, project_id: UUID, proposal_id: UUID) -> Path:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        return existing_proposal(directory, proposal_id)

    def current_proposal(self, project_id: UUID) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        proposals = [
            proposal_id
            for proposal_id in list_proposal_ids(directory)
            if not _is_accepted_region_edit(directory, proposal_id)
        ]
        if len(proposals) != 1:
            raise SegmentationProposalNotFound
        proposal_id = proposals[0]
        path = existing_proposal(directory, proposal_id)
        with Image.open(path) as handle:
            width, height = handle.size
        return self._proposal(directory, proposal_id, width, height)

    def update_proposal_draft(
        self,
        project_id: UUID,
        proposal_id: UUID,
        request: ProposalDraftUpdate,
    ) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        path = existing_proposal(directory, proposal_id)
        with Image.open(path) as handle:
            width, height = handle.size
        write_proposal_metadata(directory, proposal_id, request.name, request.role)
        return self._proposal(directory, proposal_id, width, height)

    def discard_proposal(self, project_id: UUID, proposal_id: UUID) -> None:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        remove_proposal(directory, proposal_id)

    def apply_mask_stroke(
        self,
        project_id: UUID,
        proposal_id: UUID,
        request: MaskStrokeRequest,
    ) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        working = existing_proposal(directory, proposal_id)
        existing_ai_mask(directory, proposal_id)
        mask = _binary_mask(working)
        width, height = _source_size(self.store.source_path(project))
        if mask.shape != (height, width):
            raise MaskRefinementFailed
        stroke = Stroke(
            mode=request.mode,
            brush_radius_px=request.brush_radius_px,
            points=tuple((point.x, point.y) for point in request.points),
        )
        try:
            updated = apply_stroke(mask, stroke)
        except MaskStrokeError as exc:
            raise InvalidMaskStroke from exc
        except Exception as exc:
            raise MaskRefinementFailed from exc
        replace_working_mask(directory, proposal_id, updated == 255)
        return self._proposal(directory, proposal_id, width, height)

    def apply_magic_wand(
        self,
        project_id: UUID,
        proposal_id: UUID,
        request: MagicWandRequest,
    ) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        working = existing_proposal(directory, proposal_id)
        existing_ai_mask(directory, proposal_id)
        source = self.store.source_path(project)
        with Image.open(source) as handle:
            rgb = np.array(handle.convert("RGB"))
        mask = _binary_mask(working)
        if mask.shape != rgb.shape[:2]:
            raise MagicWandFailed
        try:
            selected = select_similar(
                rgb,
                request.x,
                request.y,
                request.tolerance,
                contiguous=request.contiguous,
            )
            updated = paint_selection(mask, selected, request.mode)
        except MagicWandError as exc:
            raise InvalidMagicWand from exc
        except (InvalidMagicWand, MagicWandFailed):
            raise
        except Exception as exc:
            raise MagicWandFailed from exc
        replace_working_mask(directory, proposal_id, updated == 255)
        width, height = rgb.shape[1], rgb.shape[0]
        return self._proposal(directory, proposal_id, width, height)

    def begin_refinement_session(self, project_id: UUID, proposal_id: UUID) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        start_refinement_session(directory, proposal_id)
        width, height = _source_size(self.store.source_path(project))
        return self._proposal(directory, proposal_id, width, height)

    def restore_refinement_baseline(self, project_id: UUID, proposal_id: UUID) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        restore_session_baseline(directory, proposal_id)
        width, height = _source_size(self.store.source_path(project))
        return self._proposal(directory, proposal_id, width, height)

    def reset_proposal(self, project_id: UUID, proposal_id: UUID) -> SegmentationProposal:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        existing_proposal(directory, proposal_id)
        reset_working_mask(directory, proposal_id)
        width, height = _source_size(self.store.source_path(project))
        return self._proposal(directory, proposal_id, width, height)

    def _proposal(self, directory: Path, proposal_id: UUID, width: int, height: int) -> SegmentationProposal:
        metadata = read_proposal_metadata(directory, proposal_id)
        origin = "new_region" if metadata is None else metadata.origin_type
        region_id = None if metadata is None or metadata.region_id is None else UUID(metadata.region_id)
        return SegmentationProposal(
            proposal_id=proposal_id,
            width_px=width,
            height_px=height,
            model_id=self.model.model_id,
            name="" if metadata is None else metadata.name,
            role=None if metadata is None else metadata.role,  # type: ignore[arg-type]
            origin_type=origin,  # type: ignore[arg-type]
            region_id=region_id,
        )


def _is_accepted_region_edit(directory: Path, proposal_id: UUID) -> bool:
    metadata = read_proposal_metadata(directory, proposal_id)
    return metadata is not None and metadata.origin_type == "accepted_region" and metadata.region_id is not None


def _require_points_inside(points: list[ImagePoint], width: int, height: int) -> None:
    for point in points:
        if point.x < 0 or point.y < 0 or point.x >= width or point.y >= height:
            raise InvalidSegmentationPrompt


def _require_box_inside(box: ImageBox, width: int, height: int) -> None:
    if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height:
        raise InvalidSegmentationPrompt


def _source_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as handle:
        return handle.size


def _binary_mask(path: Path) -> np.ndarray:
    try:
        binary = read_mask_png(path)
    except Exception as exc:
        raise MaskRefinementFailed from exc
    return np.where(binary, np.uint8(255), np.uint8(0))
