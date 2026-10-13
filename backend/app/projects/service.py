"""Project operations used by the API. Routes stay thin."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.core.config import MAX_SOURCE_IMAGE_BYTES
from app.projects.defaults import stock_for_create, stock_for_update
from app.projects.duplicate import duplicate_project_directory
from app.projects.errors import InvalidProjectName, SourceImageTooLarge
from app.projects.images import inspect_source_image, sanitize_original_filename
from app.projects.storage import ProjectStore, is_safe_slug, slugify
from app.schemas.project import (
    Project,
    ProjectCreate,
    ProjectDuplicateResult,
    SourceImage,
)


class ProjectService:
    def __init__(
        self,
        store: ProjectStore,
        max_image_bytes: int = MAX_SOURCE_IMAGE_BYTES,
    ) -> None:
        self.store = store
        self.max_image_bytes = max_image_bytes

    def create_project(self, request: ProjectCreate) -> Project:
        name = request.name.strip()
        slug = slugify(name)
        if not name or not is_safe_slug(slug):
            raise InvalidProjectName
        now = datetime.now(timezone.utc)
        project = Project(
            id=uuid4(),
            name=name,
            slug=slug,
            stock=stock_for_create(request.stock),
            artwork=request.artwork,
            created_at=now,
            updated_at=now,
        )
        self.store.create_project(project)
        return project

    def update_project(self, project_id: UUID, request: ProjectCreate) -> Project:
        current = self.store.get_by_id(project_id)
        name = request.name.strip()
        if not name:
            raise InvalidProjectName
        updated = current.model_copy(
            update={
                "name": name,
                "stock": stock_for_update(request.stock, current.stock),
                "artwork": request.artwork,
                "created_at": current.created_at or datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.store.update_project(updated)
        return updated

    def delete_project(self, project_id: UUID) -> None:
        project = self.store.get_by_id(project_id)
        self.store.delete_project(project)

    def duplicate_project(self, project_id: UUID, name: str) -> ProjectDuplicateResult:
        """Copy every project file to a new id and slug. No processing is run."""

        source = self.store.get_by_id(project_id)
        cleaned = name.strip()
        slug = slugify(cleaned)
        if not cleaned or not is_safe_slug(slug):
            raise InvalidProjectName
        new_id = uuid4()
        duplicate_project_directory(
            self.store,
            source,
            name=cleaned,
            slug=slug,
            new_id=new_id,
            now=datetime.now(timezone.utc),
        )
        return ProjectDuplicateResult(
            id=new_id,
            slug=slug,
            name=cleaned,
            source_project_id=source.id,
            duplicated=True,
        )

    def list_projects(self) -> list[Project]:
        return self.store.list_projects()

    def get_project(self, project_id: UUID) -> Project:
        return self.store.get_by_id(project_id)

    def save_source_image(
        self,
        project_id: UUID,
        data: bytes,
        upload_filename: str | None,
    ) -> Project:
        if len(data) > self.max_image_bytes:
            raise SourceImageTooLarge
        project = self.store.get_by_id(project_id)
        inspected = inspect_source_image(data)
        source = SourceImage(
            filename=inspected.filename,
            original_filename=sanitize_original_filename(upload_filename),
            media_type=inspected.media_type,
            width_px=inspected.width_px,
            height_px=inspected.height_px,
        )
        return self.store.save_source(project, data, source)

    def source_file(self, project_id: UUID) -> tuple[Project, Path]:
        project = self.store.get_by_id(project_id)
        return project, self.store.source_path(project)
