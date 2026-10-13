"""Download the Depth Anything V2 Small checkpoint.

The API does not download weights on startup. Run this once:

    python scripts/download_depth_anything_v2_small.py

Checkpoint:
    https://huggingface.co/depth-anything/Depth-Anything-V2-Small
    depth_anything_v2_vits.pth

License: Apache-2.0. This is the relative-depth Small model, not a metric model.
"""

import sys
import urllib.request
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import depth_anything_v2_small_checkpoint  # noqa: E402

CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
    "/resolve/main/depth_anything_v2_vits.pth"
)


def main() -> None:
    destination = depth_anything_v2_small_checkpoint()
    if destination.is_file() and "--force" not in sys.argv:
        print(f"Checkpoint already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading Depth Anything V2 Small to {destination}")
    urllib.request.urlretrieve(CHECKPOINT_URL, temporary)
    temporary.replace(destination)
    print(f"Saved {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
