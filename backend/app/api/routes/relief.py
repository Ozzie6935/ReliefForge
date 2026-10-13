"""Relief region routes. Composition stays in the relief service."""

from uuid import UUID

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.api.deps import get_relief_service
from app.projects.errors import (
    ArtifactDependencyRejected,
    ComposedReliefNotFound,
    ConditionedDepthNotFound,
    ConditionedPreviewNotFound,
    InvalidExportRequest,
    InvalidPhysicalHeight,
    InvalidMeshRequest,
    InvalidPhysicalHeightArtifact,
    InvalidProfileRequest,
    InvalidReliefRequest,
    MaskSaveFailed,
    PhysicalHeightFailed,
    PhysicalHeightNotFound,
    PhysicalPreviewNotFound,
    ProjectError,
    ProjectNotFound,
    ReliefCompositionFailed,
    ReliefRegionNotFound,
    SegmentationProposalNotFound,
)
from app.relief.service import ReliefService
from app.schemas.project import Project
from app.schemas.relief import (
    ExportReadiness,
    StlExportRequest,
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
    SegmentationProposal,
)

router = APIRouter(tags=["relief"])


def _http_error(exc: ProjectError) -> HTTPException:
    if isinstance(exc, ProjectNotFound):
        return HTTPException(status_code=404, detail="Project not found.")
    if isinstance(exc, ConditionedDepthNotFound):
        return HTTPException(status_code=409, detail="Conditioned depth has not been created.")
    if isinstance(exc, SegmentationProposalNotFound):
        return HTTPException(status_code=404, detail="Segmentation proposal not found.")
    if isinstance(exc, ReliefRegionNotFound):
        return HTTPException(status_code=404, detail="Relief region not found.")
    if isinstance(exc, ConditionedPreviewNotFound):
        return HTTPException(status_code=404, detail="Relief preview not found.")
    if isinstance(exc, InvalidReliefRequest):
        return HTTPException(status_code=422, detail="Relief parameters are invalid.")
    if isinstance(exc, ComposedReliefNotFound):
        return HTTPException(status_code=409, detail="Composed relief has not been created.")
    if isinstance(exc, ArtifactDependencyRejected):
        return HTTPException(status_code=409, detail=exc.detail)
    if isinstance(exc, InvalidPhysicalHeight):
        if exc.reason == "relief":
            return HTTPException(
                status_code=422,
                detail="Composed relief must be a finite two-dimensional float32 array from 0 to 1.",
            )
        return HTTPException(status_code=422, detail="Physical height parameters are invalid.")
    if isinstance(exc, PhysicalPreviewNotFound):
        return HTTPException(status_code=404, detail="Physical height preview not found.")
    if isinstance(exc, PhysicalHeightNotFound):
        return HTTPException(status_code=404, detail=exc.detail)
    if isinstance(exc, InvalidPhysicalHeightArtifact):
        return HTTPException(status_code=422, detail=exc.detail)
    if isinstance(exc, InvalidProfileRequest):
        return HTTPException(status_code=422, detail=exc.detail)
    if isinstance(exc, InvalidMeshRequest):
        return HTTPException(status_code=422, detail=exc.detail)
    if isinstance(exc, InvalidExportRequest):
        return HTTPException(status_code=422, detail=exc.detail)
    if isinstance(exc, PhysicalHeightFailed):
        return HTTPException(status_code=500, detail="Physical height could not be saved.")
    if isinstance(exc, ReliefCompositionFailed):
        return HTTPException(status_code=500, detail="Relief composition failed.")
    if isinstance(exc, MaskSaveFailed):
        return HTTPException(status_code=500, detail="The region mask could not be saved.")
    return HTTPException(status_code=400, detail="Request could not be completed.")


@router.get("/projects/{project_id}/relief", response_model=ReliefRecord)
def get_relief(
    project_id: UUID,
    service: ReliefService = Depends(get_relief_service),
) -> ReliefRecord:
    try:
        return service.get_relief(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/relief/regions", response_model=Project)
def create_relief_region(
    project_id: UUID,
    request: ReliefRegionCreate,
    service: ReliefService = Depends(get_relief_service),
) -> Project:
    try:
        return service.accept_region(project_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/relief/regions/{region_id}/mask-edit",
    response_model=SegmentationProposal,
)
def begin_region_mask_edit(
    project_id: UUID,
    region_id: UUID,
    service: ReliefService = Depends(get_relief_service),
) -> SegmentationProposal:
    try:
        return service.begin_region_mask_edit(project_id, region_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/relief/regions/{region_id}/mask-edit/save",
    response_model=Project,
)
def save_region_mask_edit(
    project_id: UUID,
    region_id: UUID,
    request: RegionMaskEditSave,
    service: ReliefService = Depends(get_relief_service),
) -> Project:
    try:
        return service.save_region_mask_edit(project_id, region_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/projects/{project_id}/relief/regions/{region_id}/mask-edit/discard",
    status_code=204,
)
def discard_region_mask_edit(
    project_id: UUID,
    region_id: UUID,
    request: RegionMaskEditDiscard,
    service: ReliefService = Depends(get_relief_service),
) -> None:
    try:
        service.discard_region_mask_edit(project_id, region_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/relief/composition", response_model=Project)
def apply_relief_composition(
    project_id: UUID,
    request: ReliefCompositionRequest,
    service: ReliefService = Depends(get_relief_service),
) -> Project:
    try:
        return service.apply_composition(project_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.delete("/projects/{project_id}/relief/regions/{region_id}", response_model=Project)
def delete_relief_region(
    project_id: UUID,
    region_id: UUID,
    service: ReliefService = Depends(get_relief_service),
) -> Project:
    try:
        return service.delete_region(project_id, region_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/relief/regions/{region_id}/mask")
def get_relief_region_mask(
    project_id: UUID,
    region_id: UUID,
    service: ReliefService = Depends(get_relief_service),
) -> FileResponse:
    try:
        path = service.mask_file(project_id, region_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return FileResponse(path, media_type="image/png")


@router.get("/projects/{project_id}/relief/preview")
def get_relief_preview(
    project_id: UUID,
    service: ReliefService = Depends(get_relief_service),
) -> FileResponse:
    try:
        path = service.preview_file(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return FileResponse(path, media_type="image/png")


@router.post("/projects/{project_id}/relief/physical-height", response_model=Project)
def generate_physical_height(
    project_id: UUID,
    request: PhysicalHeightRequest,
    service: ReliefService = Depends(get_relief_service),
) -> Project:
    try:
        return service.generate_physical_height(project_id, request)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/relief/physical-height/diagnostics",
    response_model=PhysicalHeightDiagnostics,
)
def get_physical_height_diagnostics(
    project_id: UUID,
    service: ReliefService = Depends(get_relief_service),
) -> PhysicalHeightDiagnostics:
    try:
        return service.physical_height_diagnostics(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/relief/physical-height/profile",
    response_model=PhysicalHeightProfile,
)
def get_physical_height_profile(
    project_id: UUID,
    mode: Literal["horizontal", "vertical", "line"] = Query(),
    y_px: float | None = None,
    x_px: float | None = None,
    start_x_px: float | None = None,
    start_y_px: float | None = None,
    end_x_px: float | None = None,
    end_y_px: float | None = None,
    sample_count: int = 256,
    service: ReliefService = Depends(get_relief_service),
) -> PhysicalHeightProfile:
    try:
        return service.physical_height_profile(
            project_id,
            mode=mode,
            y_px=y_px,
            x_px=x_px,
            start_x_px=start_x_px,
            start_y_px=start_y_px,
            end_x_px=end_x_px,
            end_y_px=end_y_px,
            sample_count=sample_count,
        )
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/relief/mesh/preview-info",
    response_model=MeshPreviewInfo,
)
def get_mesh_preview_info(
    project_id: UUID,
    max_vertices: int = Query(default=250_000, ge=4),
    service: ReliefService = Depends(get_relief_service),
) -> MeshPreviewInfo:
    try:
        return service.mesh_preview_info(project_id, max_vertices=max_vertices)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get(
    "/projects/{project_id}/relief/mesh/solid-info",
    response_model=MeshSolidInfo,
)
def get_mesh_solid_info(
    project_id: UUID,
    max_vertices: int = Query(default=250_000, ge=4),
    service: ReliefService = Depends(get_relief_service),
) -> MeshSolidInfo:
    try:
        return service.mesh_solid_info(project_id, max_vertices=max_vertices)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/relief/mesh/preview")
def get_mesh_preview(
    project_id: UUID,
    max_vertices: int = Query(default=250_000, ge=4),
    service: ReliefService = Depends(get_relief_service),
) -> Response:
    try:
        preview = service.mesh_preview_binary(project_id, max_vertices=max_vertices)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return Response(
        content=preview.body,
        media_type="application/octet-stream",
        headers=preview.headers(),
    )


@router.get(
    "/projects/{project_id}/relief/export/readiness",
    response_model=ExportReadiness,
)
def get_export_readiness(
    project_id: UUID,
    mode: Literal["native", "reduced"] = Query(default="native"),
    height_px: int | None = Query(default=None, ge=2),
    width_px: int | None = Query(default=None, ge=2),
    service: ReliefService = Depends(get_relief_service),
) -> ExportReadiness:
    try:
        return service.export_readiness(
            project_id,
            mode=mode,
            height_px=height_px,
            width_px=width_px,
        )
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/relief/export/stl")
def post_export_stl(
    project_id: UUID,
    request: StlExportRequest,
    service: ReliefService = Depends(get_relief_service),
) -> Response:
    try:
        download = service.export_stl(
            project_id,
            mode=request.mode,
            height_px=request.height_px,
            width_px=request.width_px,
        )
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return Response(
        content=download.document.body,
        media_type="application/octet-stream",
        headers=download.headers(),
    )


@router.get("/projects/{project_id}/relief/physical-height/preview")
def get_physical_height_preview(
    project_id: UUID,
    service: ReliefService = Depends(get_relief_service),
) -> FileResponse:
    try:
        path = service.physical_preview_file(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return FileResponse(path, media_type="image/png")
