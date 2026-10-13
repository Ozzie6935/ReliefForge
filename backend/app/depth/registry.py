"""Small registry of relative-depth models."""

import threading

from app.depth.depth_anything import MODEL_ID, DepthAnythingV2Model
from app.depth.model import DepthModel
from app.projects.errors import DepthModelUnavailable

_LOCK = threading.Lock()
_MODELS: dict[str, DepthModel] = {}


def get_depth_model(model_id: str) -> DepthModel:
    with _LOCK:
        cached = _MODELS.get(model_id)
        if cached is not None:
            return cached
        if model_id != MODEL_ID:
            raise DepthModelUnavailable
        model = DepthAnythingV2Model()
        _MODELS[model_id] = model
        return model


def reset_depth_model_cache() -> None:
    with _LOCK:
        _MODELS.clear()
