"""Segmentation routes. The model stays behind SegmentationService."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import get_segmentation_service
from app.projects.errors import (
    ConditionedDepthNotFound,
    InvalidMagicWand,
    InvalidMaskStroke,
    InvalidSegmentationPrompt,
    MagicWandFailed,
    MaskRefinementFailed,
    ProjectError,
    ProjectNotFound,
    SegmentationFailed,
    SegmentationModelUnavailable,
    RefinementSessionNotFound,
    SegmentationProposalNotFound,
    SourceImageNotFound,
)
from app.schemas.relief import (
    MagicWandRequest,
    MaskStrokeRequest,
    ProposalDraftUpdate,
    SegmentationProposal,
    SegmentationRequest,
)
from app.segmentation.service import SegmentationService

router = APIRouter(tags=["segmentation"])


def _http_error(exc: ProjectError) -> HTTPException:
    if isinstance(exc, ProjectNotFound):
        return HTTPException(status_code=404, detail="Project not found.")
    if isinstance(exc, SourceImageNotFound):
        return HTTPException(status_code=404, detail="Source image not found.")
    if isinstance(exc, ConditionedDepthNotFound):
        return HTTPException(status_code=409, detail="Conditioned depth has not been created.")
    if isinstance(exc, InvalidSegmentationPrompt):
        return HTTPException(status_code=422, detail="Segmentation prompts are outside the source image.")
    if isinstance(exc, SegmentationProposalNotFound):
        return HTTPException(status_code=404, detail="Segmentation proposal not found.")
    if isinstance(exc, RefinementSessionNotFound):
        return HTTPException(status_code=409, detail="Refinement session has not been started.")
    if isinstance(exc, InvalidMaskStroke):
        return HTTPException(status_code=422, detail="The mask stroke is invalid.")
    if isinstance(exc, InvalidMagicWand):
        return HTTPException(status_code=422, detail="The Magic Wand selection is invalid.")
    if isinstance(exc, MaskRefinementFailed):
        return HTTPException(status_code=500, detail="Mask refinement failed.")
    if isinstance(exc, MagicWandFailed):
        return HTTPException(status_code=500, detail="Magic Wand failed.")
    if isinstance(exc, SegmentationModelUnavailable):
        return HTTPException(status_code=503, detail="The segmentation model is not installed.")
    if isinstance(exc, SegmentationFailed):
        return HTTPException(status_code=500, detail="Segmentation failed.")
    return HTTPException(status_code=400, detail="Request could not be completed.")


@router.post(
    "/projects/{project_id}/segmentation/predict",
    response_model=SegmentationProposal,
)
def predict_segmentation(
    project_id: UUID,
    request: SegmentationRequest,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.predict(project_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/segmentation/proposal",
    response_model=SegmentationProposal,
)
def get_current_segmentation_proposal(
    project_id: UUID,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.current_proposal(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.put(
    "/projects/{project_id}/segmentation/proposals/{proposal_id}/draft",
    response_model=SegmentationProposal,
)
def update_proposal_draft(
    project_id: UUID,
    proposal_id: UUID,
    request: ProposalDraftUpdate,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.update_proposal_draft(project_id, proposal_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/segmentation/proposals/{proposal_id}/refinement-session",
    response_model=SegmentationProposal,
)
def begin_refinement_session(
    project_id: UUID,
    proposal_id: UUID,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.begin_refinement_session(project_id, proposal_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/segmentation/proposals/{proposal_id}/refinement-session/restore",
    response_model=SegmentationProposal,
)
def restore_refinement_baseline(
    project_id: UUID,
    proposal_id: UUID,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.restore_refinement_baseline(project_id, proposal_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/segmentation/proposals/{proposal_id}/strokes",
    response_model=SegmentationProposal,
)
def refine_segmentation_proposal(
    project_id: UUID,
    proposal_id: UUID,
    request: MaskStrokeRequest,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.apply_mask_stroke(project_id, proposal_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/segmentation/proposals/{proposal_id}/magic-wand",
    response_model=SegmentationProposal,
)
def apply_magic_wand(
    project_id: UUID,
    proposal_id: UUID,
    request: MagicWandRequest,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.apply_magic_wand(project_id, proposal_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/segmentation/proposals/{proposal_id}/reset",
    response_model=SegmentationProposal,
)
def reset_segmentation_proposal(
    project_id: UUID,
    proposal_id: UUID,
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentationProposal:
    try:
        return service.reset_proposal(project_id, proposal_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/segmentation/proposals/{proposal_id}")
def get_segmentation_proposal(
    project_id: UUID,
    proposal_id: UUID,
    service: SegmentationService = Depends(get_segmentation_service),
) -> FileResponse:
    try:
        path = service.proposal_file(project_id, proposal_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return FileResponse(path, media_type="image/png")


@router.delete(
    "/projects/{project_id}/segmentation/proposals/{proposal_id}",
    status_code=204,
)
def discard_segmentation_proposal(
    project_id: UUID,
    proposal_id: UUID,
    service: SegmentationService = Depends(get_segmentation_service),
) -> None:
    try:
        service.discard_proposal(project_id, proposal_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
