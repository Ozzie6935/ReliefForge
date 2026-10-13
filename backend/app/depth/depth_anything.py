"""Depth Anything V2 Small.

The network code is the official package in ``third_party/depth_anything_v2``.
Device selection uses Relief Forge's compute detection, not the upstream
helper that also checks for CUDA.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.compute import detect_compute
from app.core.config import depth_anything_v2_small_checkpoint
from app.projects.errors import DepthInferenceFailed, DepthModelUnavailable

MODEL_ID = "depth-anything-v2-small"
MODEL_VERSION = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
INPUT_SIZE = 518
_ENCODER = "vits"
_FEATURES = 64
_OUT_CHANNELS = [48, 96, 192, 384]


class DepthAnythingV2Model:
    model_id = MODEL_ID
    model_version = MODEL_VERSION
    input_size = INPUT_SIZE

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint or depth_anything_v2_small_checkpoint()
        self._network = None
        self._device = ""
        self._ready = False

    def load(self) -> None:
        if self._ready:
            return
        if not self.checkpoint.is_file():
            raise DepthModelUnavailable
        self._build()

    def _build(self) -> None:
        import torch
        import torch.nn.functional as F
        from torchvision.transforms import Compose

        network_cls = _load_network_class()
        network = network_cls(
            encoder=_ENCODER,
            features=_FEATURES,
            out_channels=_OUT_CHANNELS,
        )
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        device = detect_compute().preferred_device
        network = network.to(device).eval()
        self._network = network
        self._device = device
        self._interpolate = F.interpolate
        self._compose = Compose
        self._ready = True

    @property
    def device(self) -> str:
        if self._device in {"mps", "cpu"}:
            return self._device
        return detect_compute().preferred_device

    def infer(self, image: Image.Image) -> np.ndarray:
        self.load()
        import cv2
        import torch

        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

        rgb = image.convert("RGB")
        height = rgb.height
        width = rgb.width
        sample = np.asarray(rgb, dtype=np.float32) / 255.0
        transform = self._compose(
            [
                Resize(
                    width=self.input_size,
                    height=self.input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        prepared = transform({"image": sample})["image"]
        tensor = torch.from_numpy(prepared).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            predicted = self._network(tensor)
            restored = self._interpolate(
                predicted[:, None],
                (height, width),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
        depth = restored.detach().to(dtype=torch.float32).cpu().numpy()
        if depth.dtype != np.float32 or depth.shape != (height, width):
            raise DepthInferenceFailed
        if not np.isfinite(depth).all():
            raise DepthInferenceFailed
        return depth


def _load_network_class():
    vendor = Path(__file__).resolve().parents[2] / "third_party"
    vendor_text = str(vendor)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from depth_anything_v2.dpt import DepthAnythingV2

    return DepthAnythingV2
