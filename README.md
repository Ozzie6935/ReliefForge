# CNC Relief Studio

CNC Relief Studio is a local-first application for converting photographs and artwork into CNC-ready 3D bas-relief geometry.

CNC Relief Studio generates relief geometry and related assets. MillMage remains responsible for CAM: tooling, feeds and speeds, toolpaths, G-code, and CNC machine control. This application does not generate G-code and does not communicate with a CNC.

## Milestone 1

Milestone 1 is the local application foundation:

- A React, TypeScript, and Vite frontend.
- A Python and FastAPI backend.
- A frontend API client that calls the backend.
- `GET /api/health`, including local compute capability.
- Placeholder packages for later depth, processing, mask, relief, mesh, and project work.

PyTorch is not a Milestone 1 dependency. If it is not installed, health reports CPU and shows PyTorch as not installed. The process does not crash.

Depth estimation, relief processing, meshes, and STL export are not implemented.

## Requirements

- macOS on Apple Silicon
- Python 3.11 or newer
- Node.js with npm

CUDA is not used. The compute check looks for Apple MPS only when PyTorch happens to be installed, and otherwise falls back to CPU.

## Backend

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

`requirements-dev.txt` installs the API dependencies plus pytest. To install the API only:

```bash
pip install -r requirements.txt
```

The API listens at [http://127.0.0.1:8000](http://127.0.0.1:8000).

Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health).

FastAPI's generated schema is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Tests

With the backend virtual environment active and the working directory set to `backend`:

```bash
pytest
```

Tests do not require PyTorch. Project tests use a temporary directory and do not write into `projects/`.

## Frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev
```

The dev server is at [http://localhost:5173](http://localhost:5173).

The page calls `GET /api/health` on load. The API client defaults to `http://127.0.0.1:8000`. Override that with `VITE_API_BASE_URL` if the API is on another local address. Browser calls are allowed from `http://localhost:5173` and `http://127.0.0.1:5173`.

Production build check:

```bash
npm run build
```

## Milestone 2

Milestone 2 adds local projects and a source image. AI depth estimation is not implemented.

- Projects are stored on disk under `projects/<slug>/`. There is no database.
- `project.json` is the project record. Stored physical sizes are millimeters.
- The create form asks for inches. It converts them with `mm = inches * 25.4` before calling the API.
- `POST /api/projects` creates a project. The identifier is a UUID. The folder name is a slug derived from the project name.
- A second project that produces the same slug is rejected. It does not overwrite the first project.
- `POST /api/projects/{id}/source` accepts a JPEG or PNG. The file is checked with Pillow, not by its extension or Content-Type alone.
- The original bytes are stored as `source/original.jpg` or `source/original.png`. They are not resized, recompressed, or otherwise edited.
- Upload size limit: 50 MB.
- `GET /api/projects/{id}/source` returns that stored file. The React page shows this response, not a local preview.
- Browser scaling of the picture does not change stock size, artwork size, or center coordinates.

Example record before an image is attached:

```json
{
  "schema_version": 1,
  "units": "mm",
  "stock": { "width_mm": 100.0, "height_mm": 80.0 },
  "artwork": {
    "width_mm": 90.0,
    "height_mm": 70.5,
    "center_x_mm": 50.0,
    "center_y_mm": 40.0
  },
  "source": null
}
```

Dimensions are values you enter for each project. They are not built into the application.

The backend finds `projects/` from its own file location, so it still works when `uvicorn` is started from `backend/`.

## Milestone 2.5

Milestone 2.5 is the application shell and project library. Depth estimation is not implemented.

- `/` redirects to `/projects`. Other pages are `/projects/new`, `/projects/{id}`, `/projects/{id}/edit`, and `/settings`.
- `GET /api/projects` lists saved projects, newest `updated_at` first.
- `PUT /api/projects/{id}` updates the name and geometry. The UUID and the project directory stay the same.
- `DELETE /api/projects/{id}` removes that project directory only. The browser asks for the project name before it enables deletion.
- `created_at` and `updated_at` are optional ISO 8601 fields on schema version 1. Older project files, including ones saved before those fields existed, still load. Reading a project does not rewrite it. Creating, editing, or uploading a source image writes the timestamps.
- Display units are inches or millimeters. The choice is stored in browser `localStorage` under `cnc-relief-studio.units`. Stored geometry remains millimeters.
- Health details stay on Settings. The navbar shows a compact backend connection indicator.

## Milestone 3

Milestone 3 adds raw relative AI depth. It does not produce a CNC height map.

`raw_depth.npy` is the model's relative scene depth, stored as float32 at the source image's pixel size. It is not carving depth, relief, or a Z height.

`preview.png` is a grayscale visualization. It is made from a copy of the raw array that is stretched to 0–255 for display. Generating the preview does not change `raw_depth.npy`.

The depth engine uses Depth Anything V2 Small (`vits`), the relative-depth model. Metric-depth checkpoints are not used. The application talks to a `DepthModel` interface. `get_depth_model("depth-anything-v2-small")` is the only registered model. Routes and React do not import Depth Anything directly.

The network implementation is the official package from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) at commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`, kept in `backend/third_party/depth_anything_v2/`. That code and the Small checkpoint are Apache-2.0. The upstream Gradio app is not included.

Install the checkpoint once. The API does not download it on startup:

```bash
cd backend
source .venv/bin/activate
python scripts/download_depth_anything_v2_small.py
```

The file is stored at:

```text
models/depth-anything-v2/small/depth_anything_v2_vits.pth
```

Model weights stay gitignored. If the checkpoint is missing, depth generation returns a service error instead of crashing.

PyTorch is required for depth. Health still starts when it is absent. On this machine, verify Apple MPS with:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

Device choice stays in `app/core/compute.py`: MPS when `torch.backends.mps.is_available()` is true, otherwise CPU. Inference input size is 518, the Depth Anything V2 default. The result is resized back to the source dimensions with the official pipeline. The model is loaded on the first generation and reused. One generation runs at a time.

`POST /api/projects/{id}/depth` reads the saved source image and writes:

```text
projects/<slug>/depth/raw_depth.npy
projects/<slug>/depth/preview.png
```

`GET /api/projects/{id}/depth/preview` returns the grayscale preview. Opening the Depth page shows a saved result and does not run the model again.

## Milestone 4

Milestone 4 conditions the immutable raw depth. It does not estimate depth again, and it does not produce a CNC height map.

`raw_depth.npy` remains the Depth Anything V2 Small result in model units. Conditioning never overwrites it or `depth/preview.png`. Every apply reads that raw file again:

```text
condition(raw_depth, parameters)
```

Depth Anything V2 Small relative output is disparity: a larger value is nearer. The official app labels that output as disparity, and Angie Dance's raw map is low on the rear wall and high in the foreground. The pipeline keeps that order unless Invert is checked. After conditioning, the invariant is:

```text
0.0 = far / back
1.0 = near / front
```

`conditioned_depth.npy` is float32, the same width and height as the raw array, and dimensionless in `[0, 1]`. It is not millimeters, CNC Z, or a finished relief. `depth/processing/preview.png` is a direct grayscale of that `[0, 1]` array. It is visualization only and is not an input to later numerical processing.

Processing order:

1. Load the immutable raw float32 array.
2. Reject non-finite values.
3. Orient so a larger value is nearer.
4. Apply diagnostic invert when requested.
5. Read the far and near percentile bounds.
6. Clip to those bounds.
7. Normalize to `[0, 1]`.
8. Gamma: `output = input ** gamma`. Gamma `1` leaves the value unchanged.
9. Contrast around `0.5`: `output = (input - 0.5) * contrast + 0.5`. Contrast `1` leaves the value unchanged.
10. Optional edge-aware smoothing. Smoothing `0` skips this step. Otherwise a bilateral filter is used. The diameter is `2 * clamp(round(smoothing / 100 * 3), 1, 3) + 1` (3, 5, or 7). `sigma_space` equals that diameter. `sigma_color = 0.02 + (1 - edge_preservation / 100) * 0.18`.
11. Clamp to `[0, 1]` and store float32.

Default parameters:

| Control | Meaning | Default | Range |
| --- | --- | --- | --- |
| Far Clip | Far percentile | 1% | 0–20% |
| Near Clip | Near percentile | 99% | 80–100% |
| Gamma | Exponent after normalization | 1 | 0.25–4 |
| Contrast | Scale around 0.5 | 1 | 0.5–2 |
| Smoothing | Edge-aware filter amount | 0 | 0–100 |
| Edge Preservation | How tightly similar depths are blended | 75 | 0–100 |
| Invert | Diagnostic reversal after orientation | off | on/off |

Far Clip must stay below Near Clip. A degenerate percentile range becomes a finite constant instead of NaN.

Artifacts:

```text
projects/<slug>/depth/raw_depth.npy
projects/<slug>/depth/preview.png
projects/<slug>/depth/processing/conditioned_depth.npy
projects/<slug>/depth/processing/preview.png
```

`POST /api/projects/{id}/depth/conditioning` reads raw depth, writes the conditioned artifacts, and stores the parameters under `depth.conditioning` in `project.json`. The arrays themselves are not stored in that file. `GET /api/projects/{id}/depth/conditioning/preview` returns the conditioned preview. `GET /api/projects/{id}/depth/histogram` returns 256-bin counts for the oriented raw depth and, when present, the conditioned depth. Opening the project loads the saved parameters and preview. It does not run conditioning or Depth Anything again.

## Milestone 5

Milestone 5 composes a dimensionless relief from conditioned depth and named regions. It does not create physical CNC Z, a mesh, or an STL.

```text
source/original.jpg
        ↓
raw_depth.npy
        ↓
conditioned_depth.npy
        ↓
relief composition
        ↓
depth/composition/relief_depth.npy
```

`source/original.jpg`, `raw_depth.npy`, and `conditioned_depth.npy` stay unchanged. A relief with zero regions equals the conditioned depth. With regions, the result is reproducible from that conditioned array, the region masks, and the region parameters.

`relief_depth.npy` is float32, the same height and width as the conditioned depth, and clamped to `[0, 1]`. `0` is back and `1` is front. It is not millimeters and it is not a CNC height map. `depth/composition/preview.png` is a direct grayscale of that range and is visualization only.

Each region stores an id, name, role (`primary`, `secondary`, `supporting`, `background`, or `custom`), mask filename, timestamps, elevation, depth range, feather, and composition order. Masks are lossless 8-bit PNGs under `depth/composition/masks/<region-id>.png`. They are not embedded in `project.json`.

For one region, the local level is the mean of the conditioned depth inside the hard mask:

```text
adjusted = clip(level + (conditioned - level) * depth_range + elevation, 0, 1)
result = (1 - weight) * result + weight * adjusted
```

Elevation `0` keeps the region's existing level. Positive elevation moves it toward the front. Negative elevation moves it toward the back. The control runs from `-1` to `+1`. Depth range `1` keeps internal variation, `0` flattens the region around its local level, and values up to `2` expand that variation. Feather `0` uses a hard mask. Feather above `0` blurs only the mask into a blending weight, with Gaussian sigma `0.24 * feather` pixels. It does not blur the depth. Regions are applied in ascending `order`, then region id. A later region blends over an earlier one.

Segmentation uses a `SegmentationModel` interface. The installed model is official SAM 2.1 Hiera Tiny (`sam2.1-hiera-tiny`), loaded lazily and cached. React and the route modules do not import SAM 2. The checkpoint is gitignored. Download it explicitly:

```bash
cd backend
source .venv/bin/activate
python scripts/download_sam2_tiny.py
```

| Item | Value |
| --- | --- |
| Upstream | https://github.com/facebookresearch/sam2 |
| Commit | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Config | `configs/sam2.1/sam2.1_hiera_t.yaml` |
| Checkpoint | `models/sam2/tiny/sam2.1_hiera_tiny.pt` from `facebook/sam2.1-hiera-tiny` |
| License | Apache-2.0, preserved at `backend/third_party/sam2/UPSTREAM_LICENSE` |

The tiny SAM 2.1 checkpoint is the smallest official image model in that release, so interactive point prompts stay practical on Apple Silicon. The API prefers MPS and falls back to CPU. It does not assume CUDA and does not load the model at startup. A missing checkpoint returns “The segmentation model is not installed.”

`POST /api/projects/{id}/segmentation/predict` takes original-image pixel prompts (positive points, negative points, and an optional box) and writes a temporary proposal PNG. It does not create a region. `POST /api/projects/{id}/relief/regions` accepts that proposal. `POST /api/projects/{id}/relief/composition` updates names, roles, elevation, depth range, feather, and order, then recomposes from conditioned depth. Deleting a region removes its mask and recomposes. Conditioned depth is required; the API does not create it. If conditioning is applied again later, the masks remain and the relief is marked stale until composition is applied again.

## Layout

```text
frontend/     React application
backend/      FastAPI application
projects/     Local user projects (not committed)
models/       Local model weights (not committed)
docs/         Project notes
```

`projects/` in the repository root holds local user projects and is not committed. `backend/app/depth/` estimates raw relative depth and conditions that array into a dimensionless `[0, 1]` map. `backend/app/relief/` composes that map with region masks. Mesh export is not implemented.
