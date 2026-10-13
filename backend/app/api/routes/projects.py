"""Project and source-image routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_project_service
from app.projects.errors import (
    InvalidProjectName,
    InvalidSourceImage,
    ProjectConflict,
    ProjectDuplicateFailed,
    ProjectDuplicateTooLarge,
    ProjectError,
    ProjectNotFound,
    SourceImageNotFound,
    SourceImageTooLarge,
    UnsupportedSourceImage,
    UnsafeProjectPath,
)
from app.projects.service import ProjectService
from app.schemas.project import Project, ProjectCreate, ProjectDuplicateRequest, ProjectDuplicateResult

router = APIRouter(tags=["projects"])


def _http_error(exc: ProjectError) -> HTTPException:
    if isinstance(exc, ProjectNotFound):
        return HTTPException(status_code=404, detail="Project not found.")
    if isinstance(exc, SourceImageNotFound):
        return HTTPException(status_code=404, detail="Source image not found.")
    if isinstance(exc, ProjectConflict):
        return HTTPException(
            status_code=409,
            detail="A project with this name already exists.",
        )
    if isinstance(exc, ProjectDuplicateTooLarge):
        return HTTPException(status_code=413, detail=exc.detail)
    if isinstance(exc, ProjectDuplicateFailed):
        return HTTPException(status_code=400, detail=exc.detail)
    if isinstance(exc, InvalidProjectName):
        return HTTPException(status_code=422, detail="Project name is not valid.")
    if isinstance(exc, SourceImageTooLarge):
        return HTTPException(status_code=413, detail="Image exceeds the upload limit.")
    if isinstance(exc, UnsupportedSourceImage):
        return HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported.",
        )
    if isinstance(exc, InvalidSourceImage):
        return HTTPException(
            status_code=400,
            detail="Image is not a valid JPEG or PNG.",
        )
    if isinstance(exc, UnsafeProjectPath):
        return HTTPException(status_code=400, detail="Project name is not allowed.")
    return HTTPException(status_code=400, detail="Request could not be completed.")


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await upload.read(min(1024 * 1024, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise SourceImageTooLarge
            chunks.append(chunk)
    finally:
        await upload.close()
    return b"".join(chunks)


@router.get("/projects", response_model=list[Project])
def list_projects(service: ProjectService = Depends(get_project_service)) -> list[Project]:
    return service.list_projects()


@router.post("/projects", response_model=Project, status_code=201)
def create_project(
    body: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
) -> Project:
    try:
        return service.create_project(body)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.put("/projects/{project_id}", response_model=Project)
def update_project(
    project_id: UUID,
    body: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
) -> Project:
    try:
        return service.update_project(project_id, body)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service),
) -> Response:
    try:
        service.delete_project(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    return Response(status_code=204)


@router.post(
    "/projects/{project_id}/duplicate",
    response_model=ProjectDuplicateResult,
    status_code=201,
)
def duplicate_project(
    project_id: UUID,
    body: ProjectDuplicateRequest,
    service: ProjectService = Depends(get_project_service),
) -> ProjectDuplicateResult:
    try:
        return service.duplicate_project(project_id, body.name)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}", response_model=Project)
def get_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service),
) -> Project:
    try:
        return service.get_project(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.post("/projects/{project_id}/source", response_model=Project)
async def upload_source_image(
    project_id: UUID,
    file: UploadFile = File(...),
    service: ProjectService = Depends(get_project_service),
) -> Project:
    try:
        data = await _read_limited(file, service.max_image_bytes)
        return service.save_source_image(project_id, data, file.filename)
    except ProjectError as exc:
        raise _http_error(exc) from exc


@router.get("/projects/{project_id}/source")
def get_source_image(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service),
) -> FileResponse:
    try:
        project, path = service.source_file(project_id)
    except ProjectError as exc:
        raise _http_error(exc) from exc
    if project.source is None:
        raise HTTPException(status_code=404, detail="Source image not found.")
    return FileResponse(path, media_type=project.source.media_type)
