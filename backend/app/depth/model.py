"""Depth model contract used by the rest of Relief Forge."""

from typing import Protocol

import numpy as np
from PIL import Image


class DepthModel(Protocol):
    """A relative-depth estimator.

    ``infer`` returns a float32 array shaped (height, width), aligned with
    the source image. Values stay in the model's relative scale.
    """

    model_id: str
    model_version: str
    input_size: int

    def load(self) -> None:
        """Load weights if they are not already loaded."""

    def infer(self, image: Image.Image) -> np.ndarray:
        """Estimate relative depth for one RGB image."""
