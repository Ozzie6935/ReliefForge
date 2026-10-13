# RELIEF FORGE — MILESTONE 8.5.15

Controlled Angie Physical-Height Generation

- Result: PASS
- Files changed: Angie Dance — Recovery only. Added `depth/composition/physical/physical_height_mm.npy`, `depth/composition/physical/preview.png`, and `depth/composition/physical/stats.json`. Modified `project.json`. Deleted none. No product code changes.
- Tests: 34 passed, 0 failed
- Protected benchmarks: 0 mismatches, 0 additions, 0 deletions (20 of 20), before and after generation
- Findings: Physical height is Current for T 38.10 mm, B 12.70 mm, P 35.56 mm. Array SHA-256 `a3666f76b1700f65c416450b2a94764650fa54efec01192be7520d227effee45`. Native STL planning is blocked by triangle, file-size, and memory limits. The explicit 585 × 426 reduced grid is ready. No mesh or STL was generated.

<details>
<summary>A. Pre-flight</summary>

Target: Angie Dance — Recovery
ID: 85664a3f-7ff9-4332-a369-1183c90b7524
Slug: angie-dance-recovery

Protected original benchmarks before generation: 0 mismatches, 0 additions, 0 deletions.

Wedding Relief — Recovery: 13 files, composite `d6362e40aeb709bb`. Unchanged from the established snapshot.

Angie Dance — Recovery provenance before generation:

- Source: Current
- Raw Depth: Current
- Conditioned Depth: Current
- Composed Relief: Current
- Physical Height: Missing

Composed relief SHA-256: `92fc23249f9e713328f411d37747ae9c5211c2a4dd12b29e2eb60c7e7da6e8de`
Shape: 2140 rows × 2940 columns
Dtype: float32
Samples: 6,291,600
Finite: 6,291,600
Nonfinite: 0
Minimum: 0
Maximum: 1

Accepted masks, unchanged:

- Angie `9b6d7dce-c3d1-47f1-a58e-7a8c2d80e19c.png` SHA-256 `64182cb37f661095a16ac39f52be9a81cb6abc30db68c63f5fb96df1b44560b5`
- Capricorn `f691285d-f58f-48c5-a07c-9d0562703da1.png` SHA-256 `7c61dd2b16e84a1fd8c0c7b27f924b245cb10c2dcaa43f43511d76f69208f97e`

Artistic settings, unchanged:

- Angie, id `9b6d7dce-c3d1-47f1-a58e-7a8c2d80e19c`, role primary, elevation 0.20, depth range 1, feather 0, order 0, updated_at `2026-10-12T14:56:56.910399Z`
- Capricorn, id `f691285d-f58f-48c5-a07c-9d0562703da1`, role secondary, elevation 0.35, depth range 1, feather 0, order 1, updated_at `2026-10-12T14:56:56.910399Z`

No physical-height directory and no physical-height record existed. Generation did not overwrite an existing physical artifact.

</details>

<details>
<summary>B. Geometry validation</summary>

Stock width: 381.0 mm (15 in)
Stock height: 279.4 mm (11 in)
Artwork width: 355.6 mm (14 in)
Artwork height: 258.839 mm (10.1905 in)
Artwork center X: 190.5 mm (7.5 in)
Artwork center Y: 139.7 mm (5.5 in)

The artwork fits inside the stock.

Margins:

- Left: 12.70 mm
- Right: 12.70 mm
- Front: 10.2805 mm
- Back: 10.2805 mm

Project stock thickness is absent. The manifest has no `thickness_mm` key. The application default is 38.10 mm. Artwork XY dimensions and placement were not modified.

</details>

<details>
<summary>C. Approved physical parameters</summary>

Approved values:

- T = 38.10 mm
- B = 12.70 mm
- P = 35.56 mm

Order check: 0 < 12.70 < 35.56 <= 38.10. Passed.

- Amplitude P − B = 22.86 mm
- Maximum cut depth T − B = 25.40 mm
- Top clearance T − P = 2.54 mm

Production request body sent to `POST /api/projects/85664a3f-7ff9-4332-a369-1183c90b7524/relief/physical-height`:

```json
{"stock_thickness_mm": 38.1, "base_thickness_mm": 12.7, "maximum_surface_height_mm": 35.56}
```

Those JSON numbers are the approved 38.10, 12.70, and 35.56. The schema accepts only `stock_thickness_mm`, `base_thickness_mm`, and `maximum_surface_height_mm`. No extra fields were sent. UI example defaults of 5 mm and 17 mm were not used.

Mapping contract used by production: H = B + R × (P − B), measured up from the stock bottom. Cut depth from the stock top is T − H. The height array is artwork-local and does not embed a stock XY offset.

</details>

<details>
<summary>D. Production API result</summary>

Endpoint: `POST /api/projects/{project_id}/relief/physical-height`
HTTP status: 200
Generated at: `2026-10-12T15:12:07.886512Z`

The production service loaded the Current composed relief, mapped it, wrote the physical bundle, and saved the provenance record. No internal generator was called directly, and the array was not written by hand.

Response physical record:

- stock_thickness_mm: 38.10
- base_thickness_mm: 12.70
- maximum_surface_height_mm: 35.56
- height_path: `depth/composition/physical/physical_height_mm.npy`
- preview_path: `depth/composition/physical/preview.png`
- stats_path: `depth/composition/physical/stats.json`
- source_relief_sha256: `92fc23249f9e713328f411d37747ae9c5211c2a4dd12b29e2eb60c7e7da6e8de`
- source_relief_generated_at: `2026-10-12T14:56:57.024417Z`
- width_px: 2940
- height_px: 2140
- dtype: float32
- height_min_mm: 12.699999809265137
- height_max_mm: 35.560001373291016
- maximum_cut_depth_mm: 25.40
- relief_amplitude_mm: 22.86
- content_sha256: `a3666f76b1700f65c416450b2a94764650fa54efec01192be7520d227effee45`
- settings_sha256: `fff996d6d314b183e8858606dbb6f961b743ecc6fcae930aacbb9ab52a5ad0de`
- response status: current

The API response includes `stock.thickness_mm: null` because the project still has no saved stock thickness. That null is not stored in the manifest.

</details>

<details>
<summary>E. Physical-height numerical validation</summary>

Loaded `depth/composition/physical/physical_height_mm.npy` after production wrote it.

- Shape: 2140 × 2940
- Dtype: float32
- Samples: 6,291,600
- Finite: 6,291,600
- Nonfinite: 0
- Bytes: 25,166,400
- Minimum: 12.699999809265137 mm
- Maximum: 35.560001373291016 mm
- Mean: 18.787433624267578 mm
- Standard deviation: 6.976763725280762 mm
- SHA-256: `a3666f76b1700f65c416450b2a94764650fa54efec01192be7520d227effee45`

Independent float64 reference: H = 12.70 + R × (35.56 − 12.70), using the saved composed relief.

Absolute error of the float32 production array against that reference:

- Maximum: 2.325773237998874e-06 mm
- Mean: 4.781422560104837e-07 mm
- RMSE: 6.453598746062829e-07 mm
- Percentile 0: 0 mm
- Percentile 50: 3.445148486491689e-07 mm
- Percentile 90: 1.1527538319455743e-06 mm
- Percentile 95: 1.3732910133512632e-06 mm
- Percentile 99: 1.9574165293079204e-06 mm
- Percentile 99.9: 2.2089481319653714e-06 mm
- Percentile 100: 2.325773237998874e-06 mm
- Nonfinite samples in the error: 0

Float32 spacing at 35.56 mm is 3.814697265625e-06 mm. The acceptance tolerance used here is 4 times that spacing, 1.52587890625e-05 mm. Every absolute error is inside that tolerance.

Float32 values of the approved settings:

- float32(12.70) = 12.699999809265137
- float32(35.56) = 35.560001373291016
- float32(38.10) = 38.099998474121094

Checks, all passed:

- H >= B within the float32 tolerance
- H <= P within the float32 tolerance
- H <= T within the float32 tolerance
- Observed minimum equals float32(B), difference 0
- Observed maximum equals float32(P), difference 0
- The minimum height occurs where composed relief is 0
- The maximum height occurs where composed relief is 1
- 62,897 relief samples equal 0, and all of them map to one height, float32(B)
- 63,126 relief samples equal 1, and all of them map to one height, float32(P)

</details>

<details>
<summary>F. Cut-depth validation</summary>

Independent calculation: D = 38.10 − H, with H promoted to float64. No toolpath was generated.

- Minimum cut depth: 2.539998626708986 mm
- Maximum cut depth: 25.400000190734865 mm
- Mean cut depth: 19.31256529656586 mm
- Finite samples: 6,291,600
- Nonfinite samples: 0

Expected span from the approved settings is 2.54 mm at the peak to 25.40 mm at the base.

With a 1e-05 mm tolerance:

- Every cut depth is >= 0
- Every cut depth is <= T − B (25.40 mm)

</details>

<details>
<summary>G. Provenance</summary>

After production generation:

- Source: Current
- Raw Depth: Current
- Conditioned Depth: Current
- Composed Relief: Current
- Physical Height: Current

The saved physical record references composed relief SHA-256 `92fc23249f9e713328f411d37747ae9c5211c2a4dd12b29e2eb60c7e7da6e8de`, generated at `2026-10-12T14:56:57.024417Z`. That matches the composed relief file.

Saved settings:

- T 38.10 mm
- B 12.70 mm
- P 35.56 mm
- generated_at `2026-10-12T15:12:07.886512Z`
- content_sha256 `a3666f76b1700f65c416450b2a94764650fa54efec01192be7520d227effee45`
- settings_sha256 `fff996d6d314b183e8858606dbb6f961b743ecc6fcae930aacbb9ab52a5ad0de`

The on-disk record does not store a status field. Current is computed when the project is read. Provenance was not backfilled by hand.

Preview SHA-256: `8c49605e856cea0d68d2afa1a62db62598a9bea584348ba36f7cab067565b2aa`
Stats SHA-256: `fa88c66ca90b5574a3b19409bdb96d4b4cef4a8971fbfd70cefa85cbbfde2a8b`

</details>

<details>
<summary>H. Artwork preservation</summary>

Unchanged on Angie Dance — Recovery:

- Source image `8018d9d884e5ef13f50a9524d362a4062f6b1c4ab98744fc79cba1b715b0e230`
- Raw depth `a3605c4ddab77a4cf36c4c6581caa56f9d514af72a11ae921a310aa9e4a3be2f`
- Raw preview `844fb6eae5c571b1edeac6eb8797eb1af44f2d3b7234bd4604fbca3de6392e79`
- Conditioned depth `486e67922808df819daa461f004bd59c12a31bfa26b4f836fb62c483c3a102e4`
- Conditioned preview `9a6d494932f4d3053a176aa03190a7dc80b9c0701772ba9d198b4018f602695d`
- Composed relief `92fc23249f9e713328f411d37747ae9c5211c2a4dd12b29e2eb60c7e7da6e8de`
- Composed preview `881ceeca3be7ed43058c77f6b3cbf931de16c5d0ad5fe148161222052b8cc4c0`
- Angie mask `64182cb37f661095a16ac39f52be9a81cb6abc30db68c63f5fb96df1b44560b5`
- Capricorn mask `7c61dd2b16e84a1fd8c0c7b27f924b245cb10c2dcaa43f43511d76f69208f97e`
- Region ids, names, roles, elevations, depth ranges, feather, order, created_at, and updated_at
- Stock 381.0 mm × 279.4 mm
- Artwork 355.6 mm × 258.839 mm at center 190.5 mm, 139.7 mm

Original Angie Dance: unchanged.
Original Wedding Relief Benchmark: unchanged.
Wedding Relief — Recovery: 13 files, unchanged, composite `d6362e40aeb709bb`.

</details>

<details>
<summary>I. Export-readiness results</summary>

Read-only `GET /api/projects/85664a3f-7ff9-4332-a369-1183c90b7524/relief/export/readiness`. The export endpoint was not called. No mesh or STL was generated.

Limits: 2,000,000 triangles, 104,857,600 STL bytes, 536,870,912 peak-memory bytes.

Native mode, planned grid 2140 rows × 2940 columns:

- Ready: false
- Provenance: current
- Triangles: 25,166,396
- Estimated STL bytes: 1,258,319,884
- Estimated peak memory: 2,525,027,772 bytes
- Blocking errors: `triangle_limit`, `stl_byte_limit`, `memory_limit`
- Warnings: `placement_not_applied`, `topology_not_validated`, `self_intersection_not_checked`, `zero_area_not_measured`

Reduced mode, explicit grid 426 rows × 585 columns:

- Ready: true
- Provenance: current
- Triangles: 996,836
- Estimated STL bytes: 49,841,884
- Estimated peak memory: 156,410,892 bytes
- Blocking errors: none
- Warnings: the four native planning warnings, plus `reduced_omits_features`

Native resolution is not reduced automatically. The reduced plan omits saved samples between the chosen indices.

</details>

<details>
<summary>J. Tests</summary>

No product code was changed. Tests use temporary projects.

- `tests/test_physical_height.py`: 5 passed
- `tests/test_physical_diagnostics.py`: 7 passed
- `tests/test_provenance.py`: 13 passed
- `tests/test_export_readiness.py`: 9 passed

Total: 34 passed, 0 failed.

</details>

<details>
<summary>K. Protected hashes</summary>

Post-flight: 0 mismatches, 0 additions, 0 deletions.

wedding-relief-benchmark/project.json 22a6080d6dcf673f293bcfd9a5cbfc2f4f8f54de60d9ea285dbdbd0fd520a7c8
wedding-relief-benchmark/source/original.jpg 9e35ce50c4d82ef767440ce8f742a2663df5ed32bff84ffdfbac5c59c68e0eef
wedding-relief-benchmark/depth/raw_depth.npy 9ab85025fdff498113f3b3c618f154d920d285a8624d8fee4820b1b00058d0e8
wedding-relief-benchmark/depth/preview.png 8f62306428bd1069856db1c6d043edcfee7f5643c2c15b8ebb447c41397e920d
wedding-relief-benchmark/depth/processing/conditioned_depth.npy d8a21fa6cc5aaf4525779c9ffe1beb8d661f62940e973f329a8c675ffa22e78b
wedding-relief-benchmark/depth/processing/preview.png 09221ecef9056d61773bda7c1217a2352acdb5e7e5d1943852e088f5b4b038ee
wedding-relief-benchmark/depth/composition/relief_depth.npy d8a21fa6cc5aaf4525779c9ffe1beb8d661f62940e973f329a8c675ffa22e78b
wedding-relief-benchmark/depth/composition/preview.png 09221ecef9056d61773bda7c1217a2352acdb5e7e5d1943852e088f5b4b038ee
wedding-relief-benchmark/depth/composition/masks/92c35f7a-0814-458b-a456-9621ed32cccc.png be79bf17e4ab1cbca6336991d94f453a87c9902838644f55b32675a45eae696b
wedding-relief-benchmark/depth/composition/masks/ea280aed-8759-4c9b-915c-074ac67727af.png ff7274f0d492b5e3284c994d1da0272268db7bf735ca0bfca861f515eb5f504c
angie-dance/project.json 3faae690ae95ee15ee5340165f00002ed59085688ef4b905358c85aee5ef7ab7
angie-dance/source/original.jpg 8018d9d884e5ef13f50a9524d362a4062f6b1c4ab98744fc79cba1b715b0e230
angie-dance/depth/raw_depth.npy a3605c4ddab77a4cf36c4c6581caa56f9d514af72a11ae921a310aa9e4a3be2f
angie-dance/depth/preview.png 844fb6eae5c571b1edeac6eb8797eb1af44f2d3b7234bd4604fbca3de6392e79
angie-dance/depth/processing/conditioned_depth.npy 486e67922808df819daa461f004bd59c12a31bfa26b4f836fb62c483c3a102e4
angie-dance/depth/processing/preview.png 9a6d494932f4d3053a176aa03190a7dc80b9c0701772ba9d198b4018f602695d
angie-dance/depth/composition/relief_depth.npy 92fc23249f9e713328f411d37747ae9c5211c2a4dd12b29e2eb60c7e7da6e8de
angie-dance/depth/composition/preview.png 881ceeca3be7ed43058c77f6b3cbf931de16c5d0ad5fe148161222052b8cc4c0
angie-dance/depth/composition/masks/9b6d7dce-c3d1-47f1-a58e-7a8c2d80e19c.png 64182cb37f661095a16ac39f52be9a81cb6abc30db68c63f5fb96df1b44560b5
angie-dance/depth/composition/masks/f691285d-f58f-48c5-a07c-9d0562703da1.png 7c61dd2b16e84a1fd8c0c7b27f924b245cb10c2dcaa43f43511d76f69208f97e

</details>

<details>
<summary>L. Differences, blockers, limitations</summary>

Angie Dance — Recovery file changes against the pre-flight snapshot:

Added:

- `depth/composition/physical/physical_height_mm.npy` SHA-256 `a3666f76b1700f65c416450b2a94764650fa54efec01192be7520d227effee45`
- `depth/composition/physical/preview.png` SHA-256 `8c49605e856cea0d68d2afa1a62db62598a9bea584348ba36f7cab067565b2aa`
- `depth/composition/physical/stats.json` SHA-256 `fa88c66ca90b5574a3b19409bdb96d4b4cef4a8971fbfd70cefa85cbbfde2a8b`

Modified:

- `project.json` from `dea783b5a8403ac19284cd5d9e529a009d9735f7da85af545a5c6121b73f9732` to `df30914dac21bc155fd57dc6dccb18bd59dd5fd21cbbe0096c24ecfecfde1de7`

Deleted: none.

`project.json` changed because production saved the physical-height record: thicknesses, generated time, file paths, source-relief fingerprint, grid size, observed height range, content fingerprint, and settings fingerprint. Region records, stock XY, and artwork placement stayed the same. Project stock thickness remains absent on disk.

Float32 cannot store 12.70 mm and 35.56 mm as those exact decimal strings. The stored endpoints are 12.699999809265137 mm and 35.560001373291016 mm. The largest difference from the float64 reference is 0.0000023258 mm.

Native export planning is blocked. A future STL requires an explicit reduced grid. The 585 × 426 plan is ready and was not exported. Readiness warnings say topology, self-intersection, and zero-area triangles were not measured.

No raw depth, conditioning, composition, mesh, STL, G-code, CAM, or machine control was run. Original benchmarks were not modified.

</details>
