"""Registry of segmentation models. SAM 2 is loaded only when a model is used."""

import threading

from app.projects.errors import SegmentationModelUnavailable
from app.segmentation.model import SegmentationModel

MODEL_ID = "sam2.1-hiera-tiny"

_LOCK = threading.Lock()
_MODELS: dict[str, SegmentationModel] = {}


def get_segmentation_model(model_id: str) -> SegmentationModel:
    with _LOCK:
        cached = _MODELS.get(model_id)
        if cached is not None:
            return cached
        if model_id != MODEL_ID:
            raise SegmentationModelUnavailable
        from app.segmentation.sam2 import SAM2SegmentationModel

        model = SAM2SegmentationModel()
        _MODELS[model_id] = model
        return model


def reset_segmentation_model_cache() -> None:
    with _LOCK:
        _MODELS.clear()
