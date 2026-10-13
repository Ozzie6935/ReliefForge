"""Segmentation model contract.

A prediction is a boolean mask in source-image pixels. Accepting that mask
is a separate step and is not part of this interface.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentationPrompt:
    positive_points: tuple[tuple[float, float], ...]
    negative_points: tuple[tuple[float, float], ...] = ()
    box: tuple[float, float, float, float] | None = None


class SegmentationModel(Protocol):
    model_id: str
    model_version: str

    def load(self) -> None:
        """Prepare weights. Missing weights raise before the network is built."""

    def predict(
        self,
        image: Image.Image,
        prompt: SegmentationPrompt,
        cache_key: str | None = None,
    ) -> np.ndarray:
        """Return a boolean mask shaped (height, width)."""
