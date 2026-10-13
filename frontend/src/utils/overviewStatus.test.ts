import assert from "node:assert/strict"
import { describe, it } from "node:test"

import type { PhysicalHeightRecord, Project } from "../types/project.ts"
import { overviewMeasurements, overviewStatusLabel, overviewWorkflow } from "./overviewStatus.ts"

const physical: PhysicalHeightRecord = {
  stock_thickness_mm: 19,
  base_thickness_mm: 5,
  maximum_surface_height_mm: 17,
  generated_at: "2026-10-08T00:00:00.000Z",
  height_path: "depth/composition/physical/physical_height_mm.npy",
  preview_path: "depth/composition/physical/preview.png",
  stats_path: "depth/composition/physical/stats.json",
  source_relief_sha256: "abc",
  source_relief_generated_at: "2026-10-08T00:00:00.000Z",
  width_px: 10,
  height_px: 10,
  dtype: "float32",
  height_min_mm: 5,
  height_max_mm: 17,
  maximum_cut_depth_mm: 14,
  relief_amplitude_mm: 12,
  status: "current",
}

function project(overrides: Partial<Project> = {}): Project {
  return {
    schema_version: 1,
    id: "project",
    name: "Sample",
    slug: "sample",
    units: "mm",
    stock: { width_mm: 304.8, height_mm: 304.8 },
    artwork: { width_mm: 279.4, height_mm: 279.4, center_x_mm: 152.4, center_y_mm: 152.4 },
    source: null,
    created_at: "2026-10-08T12:00:00.000Z",
    updated_at: "2026-10-08T13:00:00.000Z",
    ...overrides,
  }
}

describe("overview status", () => {
  it("uses backend status instead of treating saved metadata as available", () => {
    const empty = overviewWorkflow(project())
    assert.deepEqual(
      empty.map((item) => item.state),
      ["missing", "missing", "missing", "missing", "missing"],
    )

    const legacy = overviewWorkflow(
      project({
        source: {
          filename: "original.jpg",
          original_filename: "photo.jpg",
          media_type: "image/jpeg",
          width_px: 1013,
          height_px: 1023,
        },
        depth: {
          model_id: "model",
          model_version: "1",
          device: "cpu",
          input_size: 518,
          width_px: 1013,
          height_px: 1023,
          raw_filename: "raw_depth.npy",
          preview_filename: "preview.png",
          generated_at: "2026-10-08T00:00:00.000Z",
          conditioning: {
            parameters: {
              far_percentile: 1,
              near_percentile: 99,
              gamma: 1,
              contrast: 1,
              smoothing: 0,
              edge_preservation: 75,
              invert: false,
            },
            width_px: 1013,
            height_px: 1023,
            dtype: "float32",
            range_min: 0,
            range_max: 1,
            conditioned_filename: "conditioned_depth.npy",
            preview_filename: "preview.png",
            generated_at: "2026-10-08T00:10:00.000Z",
          },
        },
        relief: {
          regions: [],
          generated_at: "2026-10-08T00:20:00.000Z",
          physical: { ...physical, status: "stale" },
        },
      }),
    )
    assert.deepEqual(
      legacy.map((item) => [item.label, item.state]),
      [
        ["Source", "unknown"],
        ["Raw Depth", "unknown"],
        ["Conditioned Depth", "unknown"],
        ["Relief Composition", "unknown"],
        ["Physical Height", "stale"],
      ],
    )
    assert.equal(overviewStatusLabel("unknown"), "Unknown")
    assert.equal(overviewStatusLabel("current"), "Current")
    assert.notEqual(legacy[0]?.state, "current")
    assert.notEqual(legacy[0]?.state, "stale")
  })

  it("keeps later stages missing when only raw depth metadata exists", () => {
    const items = overviewWorkflow(
      project({
        depth: {
          model_id: "model",
          model_version: "1",
          device: "cpu",
          input_size: 518,
          width_px: 8,
          height_px: 8,
          raw_filename: "raw_depth.npy",
          preview_filename: "preview.png",
          generated_at: "2026-10-08T00:00:00.000Z",
          status: "current",
        },
        relief: { regions: [], generated_at: null, physical: null },
      }),
    )
    assert.equal(items.find((item) => item.id === "raw-depth")?.state, "current")
    assert.equal(items.find((item) => item.id === "conditioned")?.state, "missing")
    assert.equal(items.find((item) => item.id === "composed-relief")?.state, "unknown")
    assert.equal(items.find((item) => item.id === "physical-height")?.state, "missing")
  })

  it("uses the API physical-height status and treats a blank status as unknown", () => {
    const current = overviewWorkflow(project({ relief: { regions: [], physical } }))
    assert.equal(current.find((item) => item.id === "physical-height")?.state, "current")
    const blank = overviewWorkflow(
      project({ relief: { regions: [], physical: { ...physical, status: null } } }),
    )
    assert.equal(blank.find((item) => item.id === "physical-height")?.state, "unknown")
    assert.notEqual(blank.find((item) => item.id === "physical-height")?.state, "stale")
  })

  it("formats stock, artwork, and center in the selected display unit", () => {
    const sample = project({
      source: {
        filename: "original.jpg",
        original_filename: "photo.jpg",
        media_type: "image/jpeg",
        width_px: 1013,
        height_px: 1023,
      },
    })
    const inches = overviewMeasurements(sample, "in")
    assert.equal(inches.name, "Sample")
    assert.equal(inches.stockWidth, "12 in")
    assert.equal(inches.stockHeight, "12 in")
    assert.equal(inches.stockThickness, "1.50 in")
    assert.equal(inches.stockThicknessSource, "Application default")
    assert.equal(inches.artworkWidth, "11 in")
    assert.equal(inches.artworkHeight, "11 in")
    assert.equal(inches.centerX, "6 in")
    assert.equal(inches.centerY, "6 in")
    assert.equal(inches.sourceFilename, "photo.jpg")
    assert.equal(inches.sourceResolution, "1013 × 1023 px")
    assert.equal(typeof inches.created, "string")
    assert.equal(typeof inches.modified, "string")

    const millimeters = overviewMeasurements(project(), "mm")
    assert.equal(millimeters.stockWidth, "304.8 mm")
    assert.equal(millimeters.artworkWidth, "279.4 mm")
    const projectThickness = overviewMeasurements(
      project({ stock: { width_mm: 304.8, height_mm: 304.8, thickness_mm: 50.8 } }),
      "mm",
    )
    assert.equal(projectThickness.stockThickness, "50.80 mm")
    assert.equal(projectThickness.stockThicknessSource, "Project thickness")
    assert.equal(millimeters.centerX, "152.4 mm")
    assert.equal(millimeters.sourceFilename, null)
    assert.equal(millimeters.sourceResolution, null)
  })
})
