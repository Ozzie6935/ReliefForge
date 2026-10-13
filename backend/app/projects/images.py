"""Validate source images without modifying their bytes."""

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.projects.errors import InvalidSourceImage, UnsupportedSourceImage

# Stored names are chosen by us. The user's filename is metadata only.
_FORMATS: dict[str, tuple[str, str]] = {
    "JPEG": ("original.jpg", "image/jpeg"),
    "PNG": ("original.png", "image/png"),
}


@dataclass(frozen=True)
class InspectedImage:
    filename: str
    media_type: str
    width_px: int
    height_px: int


def sanitize_original_filename(raw: str | None) -> str:
    if not raw:
        return "upload"
    name = Path(raw.replace("\\", "/")).name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        return "upload"
    return name[:255]


def inspect_source_image(data: bytes) -> InspectedImage:
    """Read format and pixel size. The caller stores ``data`` unchanged."""

    if not data:
        raise InvalidSourceImage

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            image_format = image.format
            width, height = image.size
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:
        raise InvalidSourceImage from exc

    if not image_format or width <= 0 or height <= 0:
        raise InvalidSourceImage
    if image_format not in _FORMATS:
        raise UnsupportedSourceImage

    filename, media_type = _FORMATS[image_format]
    return InspectedImage(
        filename=filename,
        media_type=media_type,
        width_px=width,
        height_px=height,
    )
