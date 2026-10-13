"""Official SAM 2.1 Hiera Tiny behind the segmentation interface.

Upstream: https://github.com/facebookresearch/sam2
Commit: 2b90b9f5ceec907a1c18123530e92e794ad901a4
Checkpoint: facebook/sam2.1-hiera-tiny ``sam2.1_hiera_tiny.pt``
License: Apache-2.0

The tiny checkpoint is the smallest official SAM 2.1 image model, chosen so
interactive point prompts stay practical on Apple Silicon. SAM 2 is imported
only when ``load`` runs.
"""

import threading

import numpy as np
from PIL import Image

from app.core.compute import detect_compute
from app.core.config import sam2_tiny_checkpoint
from app.projects.errors import SegmentationFailed, SegmentationModelUnavailable
from app.segmentation.model import SegmentationPrompt

MODEL_ID = "sam2.1-hiera-tiny"
MODEL_VERSION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_t.yaml"

_PREDICT_LOCK = threading.Lock()


class SAM2SegmentationModel:
    model_id = MODEL_ID
    model_version = MODEL_VERSION

    def __init__(self) -> None:
        self._predictor = None
        self._cache_key: str | None = None
        self.device = "cpu"

    def load(self) -> None:
        if self._predictor is not None:
            return
        checkpoint = sam2_tiny_checkpoint()
        if not checkpoint.is_file():
            raise SegmentationModelUnavailable
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise SegmentationFailed from exc
        device = detect_compute().preferred_device
        try:
            network = build_sam2(
                config_file=CONFIG_NAME,
                ckpt_path=str(checkpoint),
                device=device,
            )
        except Exception as exc:
            raise SegmentationFailed from exc
        self._predictor = SAM2ImagePredictor(network)
        self.device = device

    def predict(
        self,
        image: Image.Image,
        prompt: SegmentationPrompt,
        cache_key: str | None = None,
    ) -> np.ndarray:
        self.load()
        predictor = self._predictor
        if predictor is None:
            raise SegmentationFailed
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        points, labels = _point_arrays(prompt)
        box = None if prompt.box is None else np.asarray(prompt.box, dtype=np.float32)
        ambiguous = points is None or len(points) + (0 if box is None else 1) <= 1
        with _PREDICT_LOCK:
            try:
                if cache_key is None or cache_key != self._cache_key:
                    predictor.set_image(rgb)
                    self._cache_key = cache_key
                masks, scores, _low = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=ambiguous,
                    normalize_coords=True,
                )
            except Exception as exc:
                self._cache_key = None
                raise SegmentationFailed from exc
        if masks.size == 0:
            raise SegmentationFailed
        choice = int(np.argmax(scores))
        mask = np.asarray(masks[choice], dtype=bool)
        if mask.shape != (image.height, image.width):
            raise SegmentationFailed
        return mask


def _point_arrays(
    prompt: SegmentationPrompt,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    coords.extend(prompt.positive_points)
    labels.extend(1 for _ in prompt.positive_points)
    coords.extend(prompt.negative_points)
    labels.extend(0 for _ in prompt.negative_points)
    if not coords:
        return None, None
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )
