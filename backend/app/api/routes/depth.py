"""Raw relative-depth routes. Inference stays in the depth service."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import get_depth_service
from app.depth.service import DepthService
from app.projects.errors import (
    ArtifactDependencyRejected,
    ConditionedPreviewNotFound,
    ConditioningFailed,
    DepthArtifactNotFound,
    DepthInferenceFailed,
    DepthModelUnavailable,
    ProjectError,
    ProjectNotFound,
    RawDepthNotFound,
    SourceImageNotFound,
)
from app.schemas.conditioning import DepthConditioningParameters, DepthHistogram
from app.schemas.project import Project

router = APIRouter(tags=["depth"])


def _http_error(exc: ProjectError) -> HTTPException:
    if isinstance(exc, ProjectNotFound):
        return HTTPException(status_code=404, detail="Project not found.")
    if isinstance(exc, SourceImageNotFound):
        return HTTPException(status_code=404, detail="Source image not found.")
    if isinstance(exc, DepthArtifactNotFound):
        return HTTPException(status_code=404, detail="Depth preview not found.")
    if isinstance(exc, DepthModelUnavailable):
        return HTTPException(status_code=503, detail="The depth model is not installed.")
    if isinstance(exc, DepthInferenceFailed):
        return HTTPException(status_code=500, detail="Depth generation failed.")
    if isinstance(exc, RawDepthNotFound):
        return HTTPException(status_code=409, detail="Raw depth has not been generated.")
    if isinstance(exc, ArtifactDependencyRejected):
        return HTTPException(status_code=409, detail=exc.detail)
    if isinstance(exc, ConditionedPreviewNotFound):
        return HTTPException(status_code=404, detail="Conditioned preview not found.")
    if isinstance(exc, ConditioningFailed):
        return HTTPException(status_code=500, detail="Depth conditioning failed.")
    return HTTPException(status_code=400, detail="Request could not be completed.")


@router.post("/projects/{project_id}/depth", response_model=Project)
def generate_depth(
    project_id: UUID,
    service: DepthService = Depends(get_depth_service),
) -> Project:
    try:
        return service.generate(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/depth/preview")
def get_depth_preview(
    project_id: UUID,
    service: DepthService = Depends(get_depth_service),
) -> FileResponse:
    try:
        path = service.preview_path(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return FileResponse(path, media_type="image/png")


@router.post("/projects/{project_id}/depth/conditioning", response_model=Project)
def apply_conditioning(
    project_id: UUID,
    parameters: DepthConditioningParameters,
    service: DepthService = Depends(get_depth_service),
) -> Project:
    try:
        return service.condition(project_id, parameters)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/depth/conditioning/preview")
def get_conditioned_preview(
    project_id: UUID,
    service: DepthService = Depends(get_depth_service),
) -> FileResponse:
    try:
        path = service.conditioned_preview_path(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return FileResponse(path, media_type="image/png")


@router.get("/projects/{project_id}/depth/histogram", response_model=DepthHistogram)
def get_depth_histogram(
    project_id: UUID,
    service: DepthService = Depends(get_depth_service),
) -> DepthHistogram:
    try:
        return service.histogram(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
