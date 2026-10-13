"""Generate and read raw relative depth for one project."""

import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
from PIL import Image

from app.core.compute import detect_compute
from app.depth.artifacts import (
    artifact_for,
    conditioned_depth_file,
    conditioned_preview_file,
    preview_file,
    raw_depth_file,
)
from app.depth.conditioning import LARGER_RAW_VALUE_IS_NEAR, ConditioningError, condition_depth
from app.depth.histogram import HISTOGRAM_BINS, depth_histogram
from app.depth.model import DepthModel
from app.depth.preview import grayscale_preview, unit_interval_preview
from app.depth.registry import get_depth_model
from app.projects.errors import (
    ArtifactDependencyRejected,
    ConditionedPreviewNotFound,
    ConditioningFailed,
    DepthInferenceFailed,
    DepthModelUnavailable,
    RawDepthNotFound,
)
from app.projects.provenance import RAW_STALE_DETAIL, RAW_UNKNOWN_DETAIL
from app.projects.storage import ProjectStore
from app.schemas.conditioning import (
    ConditioningRecord,
    DepthConditioningParameters,
    DepthHistogram,
    HistogramSeries,
)
from app.schemas.project import DepthArtifact, Project

_INFERENCE_LOCK = threading.Lock()


class DepthService:
    def __init__(self, store: ProjectStore, model: DepthModel | None = None) -> None:
        self.store = store
        self.model = model if model is not None else get_depth_model("depth-anything-v2-small")

    def generate(self, project_id: UUID) -> Project:
        project = self.store.get_by_id(project_id)
        source = self.store.source_path(project)
        with Image.open(source) as handle:
            rgb = handle.convert("RGB").copy()
        with _INFERENCE_LOCK:
            try:
                self.model.load()
                depth = self.model.infer(rgb)
            except (DepthModelUnavailable, DepthInferenceFailed):
                raise
            except Exception as exc:
                raise DepthInferenceFailed from exc
            self._require_raw_depth(depth, rgb.width, rgb.height)
            preview = grayscale_preview(depth)
            artifact = self._artifact(depth, rgb.width, rgb.height)
            return self.store.save_depth(project, depth, preview, artifact)

    def condition(self, project_id: UUID, parameters: DepthConditioningParameters) -> Project:
        project = self.store.get_by_id(project_id)
        if project.depth is None or project.depth.status in {None, "missing"}:
            raise RawDepthNotFound
        if project.depth.status == "stale":
            raise ArtifactDependencyRejected(RAW_STALE_DETAIL)
        if project.depth.status != "current":
            raise ArtifactDependencyRejected(RAW_UNKNOWN_DETAIL)
        raw = self._read_raw(project)
        try:
            conditioned = condition_depth(raw, parameters)
            preview = unit_interval_preview(conditioned)
        except (ConditioningError, DepthInferenceFailed) as exc:
            raise ConditioningFailed from exc
        except Exception as exc:
            raise ConditioningFailed from exc
        record = ConditioningRecord(
            parameters=parameters,
            width_px=int(conditioned.shape[1]),
            height_px=int(conditioned.shape[0]),
            dtype="float32",
            range_min=float(conditioned.min()),
            range_max=float(conditioned.max()),
            conditioned_filename="conditioned_depth.npy",
            preview_filename="preview.png",
            generated_at=datetime.now(timezone.utc),
        )
        return self.store.save_conditioning(project, conditioned, preview, record)

    def conditioned_preview_path(self, project_id: UUID) -> Path:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        return conditioned_preview_file(directory, project)

    def histogram(self, project_id: UUID) -> DepthHistogram:
        project = self.store.get_by_id(project_id)
        raw = self._read_raw(project)
        raw_series = depth_histogram(self._oriented_raw(project, raw), bins=HISTOGRAM_BINS)
        conditioned_series = None
        directory = self.store.project_directory(project.slug)
        try:
            path = conditioned_depth_file(directory, project)
        except ConditionedPreviewNotFound:
            path = None
        if path is not None:
            conditioned = np.load(path)
            conditioned_series = depth_histogram(
                conditioned,
                bins=HISTOGRAM_BINS,
                value_range=(0.0, 1.0),
            )
        return DepthHistogram(
            raw=_series(raw_series),
            conditioned=None if conditioned_series is None else _series(conditioned_series),
        )

    def preview_path(self, project_id: UUID) -> Path:
        project = self.store.get_by_id(project_id)
        directory = self.store.project_directory(project.slug)
        return preview_file(directory, project)

    def _artifact(self, depth: np.ndarray, width: int, height: int) -> DepthArtifact:
        device = getattr(self.model, "device", None) or detect_compute().preferred_device
        if device not in {"mps", "cpu"}:
            device = detect_compute().preferred_device
        return artifact_for(
            model_id=self.model.model_id,
            model_version=self.model.model_version,
            device=device,
            input_size=self.model.input_size,
            width_px=width,
            height_px=height,
            generated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _oriented_raw(project: Project, raw: np.ndarray) -> np.ndarray:
        """Model orientation, then the saved diagnostic invert when one exists."""

        oriented = np.array(raw, dtype=np.float32, copy=True)
        if not LARGER_RAW_VALUE_IS_NEAR:
            np.negative(oriented, out=oriented)
        conditioning = None if project.depth is None else project.depth.conditioning
        if conditioning is not None and conditioning.parameters.invert:
            np.negative(oriented, out=oriented)
        return oriented

    def _read_raw(self, project: Project) -> np.ndarray:
        directory = self.store.project_directory(project.slug)
        path = raw_depth_file(directory, project)
        raw = np.load(path)
        if not isinstance(raw, np.ndarray):
            raise ConditioningFailed
        return raw

    @staticmethod
    def _require_raw_depth(depth: np.ndarray, width: int, height: int) -> None:
        if depth.dtype != np.float32 or depth.shape != (height, width):
            raise DepthInferenceFailed
        if not np.isfinite(depth).all():
            raise DepthInferenceFailed


def _series(series) -> HistogramSeries:
    return HistogramSeries(
        bins=series.bins,
        counts=series.counts,
        minimum=series.minimum,
        maximum=series.maximum,
    )
