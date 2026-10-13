"""Download the official SAM 2.1 Hiera Tiny checkpoint.

This is an explicit setup step. The API does not download weights on startup.

Upstream: https://github.com/facebookresearch/sam2
Commit: 2b90b9f5ceec907a1c18123530e92e794ad901a4
Checkpoint: https://huggingface.co/facebook/sam2.1-hiera-tiny
License: Apache-2.0
"""

from pathlib import Path
from urllib.request import urlretrieve

CHECKPOINT_URL = (
    "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt"
)


def destination() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "sam2" / "tiny" / "sam2.1_hiera_tiny.pt"


def main() -> None:
    path = destination()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1_000_000:
        print(f"Checkpoint already present: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial")
    print(f"Downloading {CHECKPOINT_URL}")
    urlretrieve(CHECKPOINT_URL, temporary)
    temporary.replace(path)
    print(f"Saved {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
