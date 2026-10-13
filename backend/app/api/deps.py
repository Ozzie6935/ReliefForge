"""FastAPI dependencies."""

from app.core.config import default_projects_dir
from app.depth.service import DepthService
from app.projects.service import ProjectService
from app.projects.storage import ProjectStore
from app.relief.service import ReliefService
from app.segmentation.service import SegmentationService


def get_project_service() -> ProjectService:
    return ProjectService(ProjectStore(default_projects_dir()))


def get_depth_service() -> DepthService:
    return DepthService(ProjectStore(default_projects_dir()))


def get_segmentation_service() -> SegmentationService:
    return SegmentationService(ProjectStore(default_projects_dir()))


def get_relief_service() -> ReliefService:
    return ReliefService(ProjectStore(default_projects_dir()))
