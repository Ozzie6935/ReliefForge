"""Application constants for CNC Relief Studio."""

from pathlib import Path

APPLICATION_NAME = "CNC Relief Studio"
API_VERSION = "1"

# Vite's default dev server. Keep this list explicit.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# High-resolution photographs. The upload reader stops at this size.
MAX_SOURCE_IMAGE_BYTES = 50 * 1024 * 1024

# A duplicate copies every project-owned file into a new directory.
# These bounds stop an unexpected tree from filling the disk.
MAX_DUPLICATE_FILES = 10_000
MAX_DUPLICATE_BYTES = 2 * 1024 * 1024 * 1024


def repository_root() -> Path:
    """Repository root containing backend/ and projects/.

    Resolved from this file so launch directory does not matter.
    """

    return Path(__file__).resolve().parents[3]


def default_projects_dir() -> Path:
    return repository_root() / "projects"


def default_models_dir() -> Path:
    return repository_root() / "models"


def depth_anything_v2_small_checkpoint() -> Path:
    return default_models_dir() / "depth-anything-v2" / "small" / "depth_anything_v2_vits.pth"


def sam2_tiny_checkpoint() -> Path:
    return default_models_dir() / "sam2" / "tiny" / "sam2.1_hiera_tiny.pt"
