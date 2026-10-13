"""Build temporary RFMP fixtures with the existing solid packer.

Writes only under frontend/src/fixtures/synthetic-relief/.
Does not read or write projects/.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

from app.relief.preview_binary import build_preview_bytes

ARTWORK_WIDTH_MM = 100.0
ARTWORK_HEIGHT_MM = 70.0
OUT = Path(__file__).resolve().parents[1] / "src" / "fixtures" / "synthetic-relief"

# Upper-left plateau and a smaller lower-right bump. Y grows downward.
UPPER_LEFT = (4.0, 36.0, 3.0, 20.0)
LOWER_RIGHT = (74.0, 90.0, 50.0, 64.0)


def height_field(
    rows: int,
    cols: int,
    base_mm: float,
    upper_mm: float,
    lower_mm: float,
) -> np.ndarray:
    field = np.full((rows, cols), base_mm, dtype=np.float32)
    xs = np.linspace(0.0, ARTWORK_WIDTH_MM, cols)
    ys = np.linspace(0.0, ARTWORK_HEIGHT_MM, rows)
    xx, yy = np.meshgrid(xs, ys)
    x0, x1, y0, y1 = UPPER_LEFT
    field[(xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)] = np.float32(upper_mm)
    x0, x1, y0, y1 = LOWER_RIGHT
    field[(xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)] = np.float32(lower_mm)
    return field


def header_summary(body: bytes) -> dict[str, int | float | bool | str]:
    def u32(offset: int) -> int:
        return struct.unpack_from("<I", body, offset)[0]

    def f32(offset: int) -> float:
        return struct.unpack_from("<f", body, offset)[0]

    flags = u32(36)
    status = {1: "unknown", 2: "current", 3: "stale"}[u32(40)]
    return {
        "bytes": len(body),
        "vertices": u32(16),
        "triangles": u32(20),
        "status": status,
        "sourceWidth": u32(44),
        "sourceHeight": u32(48),
        "meshWidth": u32(52),
        "meshHeight": u32(56),
        "artworkWidthMm": f32(60),
        "artworkHeightMm": f32(64),
        "zMin": f32(76),
        "zMax": f32(88),
        "downsampled": bool(flags & 1),
        "maxVertices": u32(128),
        "exportReady": u32(124),
    }


def main() -> None:
    if "projects" in OUT.parts:
        raise SystemExit("Refusing to write a fixture into projects/.")
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("20000.rfmp", 201, 141, 5.0, 18.0, 9.0, "current", 20_000),
        ("50000.rfmp", 81, 57, 5.0, 18.0, 9.0, "current", 50_000),
        ("150000.rfmp", 121, 85, 5.0, 18.0, 9.0, "unknown", 150_000),
        ("250000.rfmp", 161, 113, 5.0, 18.0, 9.0, "stale", 250_000),
        ("thin.rfmp", 41, 29, 0.4, 1.2, 0.7, "current", 50_000),
        ("tall.rfmp", 41, 29, 40.0, 72.0, 55.0, "current", 50_000),
    ]
    manifest: dict[str, dict[str, int | float | bool | str]] = {}
    for name, rows, cols, base, upper, lower, status, budget in specs:
        field = height_field(rows, cols, base, upper, lower)
        packed = build_preview_bytes(
            field,
            ARTWORK_WIDTH_MM,
            ARTWORK_HEIGHT_MM,
            status=status,
            max_vertices=budget,
        )
        target = OUT / name
        target.write_bytes(packed.body)
        summary = header_summary(packed.body)
        summary["baseMm"] = base
        summary["upperMm"] = upper
        summary["lowerMm"] = lower
        manifest[name] = summary
        print(
            f"{name} mesh {summary['meshWidth']}x{summary['meshHeight']} "
            f"verts {summary['vertices']} tris {summary['triangles']} "
            f"status {summary['status']} downsampled {summary['downsampled']} "
            f"bytes {summary['bytes']}"
        )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
