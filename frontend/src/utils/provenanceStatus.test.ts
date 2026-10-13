import assert from "node:assert/strict"
import { describe, it } from "node:test"

import type { PhysicalHeightRecord, Project, ReliefRegion } from "../types/project.ts"
import { overviewWorkflow } from "./overviewStatus.ts"
import {
  artifactStatus,
  canApplyComposition,
  canApplyConditioning,
  canGeneratePhysical,
  failedProcessingResult,
  physicalSettingsPending,
  regionSettingsPending,
  statusExplanation,
  statusLabel,
} from "./provenanceStatus.ts"
import { workspaceCanvas } from "./reliefWorkspace.ts"

const region: ReliefRegion = {
  id: "region-1",
  name: "Subject",
  role: "primary",
  mask_filename: "region-1.png",
  elevation: 0.25,
  depth_range: 1.4,
  feather: 6,
  order: 0,
  created_at: "2026-10-08T00:00:00.000Z",
  updated_at: "2026-10-08T00:00:00.000Z",
}

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
  width_px: 8,
  height_px: 8,
  dtype: "float32",
  height_min_mm: 5,
  height_max_mm: 17,
  maximum_cut_depth_mm: 14,
  relief_amplitude_mm: 12,
  status: "current",
}

function project(): Project {
  return {
    schema_version: 1,
    id: "project",
    name: "Legacy",
    slug: "legacy",
    units: "mm",
    stock: { width_mm: 100, height_mm: 100 },
    artwork: { width_mm: 80, height_mm: 80, center_x_mm: 50, center_y_mm: 50 },
    source: {
      filename: "original.jpg",
      original_filename: "photo.jpg",
      media_type: "image/jpeg",
      width_px: 10,
      height_px: 10,
    },
    depth: {
      model_id: "model",
      model_version: "1",
      device: "cpu",
      input_size: 518,
      width_px: 10,
      height_px: 10,
      raw_filename: "raw_depth.npy",
      preview_filename: "preview.png",
      generated_at: "2026-10-04T00:00:00.000Z",
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
        width_px: 10,
        height_px: 10,
        dtype: "float32",
        range_min: 0,
        range_max: 1,
        conditioned_filename: "conditioned_depth.npy",
        preview_filename: "preview.png",
        generated_at: "2026-10-04T00:10:00.000Z",
      },
    },
    relief: {
      regions: [region],
      generated_at: "2026-10-04T00:20:00.000Z",
      physical: null,
    },
  }
}

describe("provenance status", () => {
  it("maps the four statuses and does not style unknown as stale", () => {
    for (const status of ["missing", "unknown", "current", "stale"] as const) {
      assert.equal(artifactStatus({ id: "saved" }, status), status)
      assert.equal(statusLabel(status).length > 0, true)
      assert.equal(statusExplanation(status).length > 0, true)
    }
    assert.equal(statusLabel("unknown"), "Unknown")
    assert.notEqual(statusLabel("unknown"), statusLabel("stale"))
    assert.match(statusExplanation("unknown"), /cannot be verified/)
    assert.doesNotMatch(statusExplanation("unknown"), /corrupt|damaged|outdated/)
  })

  it("treats a null record as missing and a blank status as unknown", () => {
    assert.equal(artifactStatus(null, "current"), "missing")
    assert.equal(artifactStatus(undefined, undefined), "missing")
    assert.equal(artifactStatus({ generated_at: "2026-10-04T00:00:00.000Z" }, undefined), "unknown")
    assert.equal(artifactStatus({ generated_at: "2026-10-04T00:00:00.000Z" }, null), "unknown")
  })

  it("reads overview status from the backend fields", () => {
    const items = overviewWorkflow(project())
    assert.deepEqual(
      items.map((item) => [item.label, item.state]),
      [
        ["Source", "unknown"],
        ["Raw Depth", "unknown"],
        ["Conditioned Depth", "unknown"],
        ["Relief Composition", "unknown"],
        ["Physical Height", "missing"],
      ],
    )
    const verified = project()
    verified.source = { ...verified.source!, status: "current" }
    verified.depth = { ...verified.depth!, status: "current", conditioning: { ...verified.depth!.conditioning!, status: "current" } }
    verified.relief = { ...verified.relief!, status: "stale", physical }
    assert.equal(overviewWorkflow(verified).find((item) => item.id === "physical-height")?.state, "current")
    assert.equal(overviewWorkflow(verified).find((item) => item.id === "composed-relief")?.state, "stale")
  })

  it("enables processing only when the required upstream stage is current", () => {
    assert.equal(canApplyConditioning("current"), true)
    assert.equal(canApplyConditioning("unknown"), false)
    assert.equal(canApplyConditioning("stale"), false)
    assert.equal(canApplyConditioning("missing"), false)
    assert.equal(canApplyComposition("current"), true)
    assert.equal(canApplyComposition("unknown"), false)
    assert.equal(canGeneratePhysical("current"), true)
    assert.equal(canGeneratePhysical("unknown"), false)
    assert.equal(canGeneratePhysical("stale"), false)
  })

  it("keeps saved status when drafts change", () => {
    const saved = project()
    saved.relief = { ...saved.relief!, status: "current" }
    const pending = regionSettingsPending([{ ...region, elevation: 0.9 }], [region])
    assert.equal(pending, true)
    assert.equal(artifactStatus(saved.relief, saved.relief?.status), "current")
    assert.equal(regionSettingsPending([region], [region]), false)
    assert.equal(regionSettingsPending([{ ...region, detail_amount: 0 }], [region]), false)
    assert.equal(regionSettingsPending([{ ...region, detail_amount: 0.02 }], [region]), true)
    assert.equal(regionSettingsPending([{ ...region, detail_scale_mm: null }], [region]), false)
    assert.equal(regionSettingsPending([{ ...region, detail_scale_mm: 0.5 }], [region]), true)
    assert.equal(
      regionSettingsPending([{ ...region, detail_scale_mm: 0.5 }], [{ ...region, detail_scale_mm: 0.5 }]),
      false,
    )
    assert.equal(
      physicalSettingsPending(
        { stock_thickness_mm: 19, base_thickness_mm: 6, maximum_surface_height_mm: 17 },
        physical,
      ),
      true,
    )
    assert.equal(physicalSettingsPending(null, physical), false)
    assert.equal(artifactStatus(physical, physical.status), "current")
  })

  it("preserves project and draft state when a 409 detail is returned", () => {
    const current = project()
    const draft = { contrast: 1.5 }
    const failure = failedProcessingResult({
      error: new Error("Raw depth is stale. Regenerate raw depth from the current source before conditioning."),
      fallback: "Depth conditioning failed.",
      project: current,
      draft,
    })
    assert.equal(failure.succeeded, false)
    assert.equal(failure.previewsCleared, false)
    assert.equal(failure.project, current)
    assert.equal(failure.draft, draft)
    assert.match(failure.message, /Regenerate raw depth/)
    const blank = failedProcessingResult({
      error: new Error("  "),
      fallback: "Relief composition failed.",
      project: current,
      draft,
    })
    assert.equal(blank.message, "Relief composition failed.")
  })

  it("keeps an unknown legacy preview reachable", () => {
    const canvas = workspaceCanvas({
      editing: false,
      view: "physical",
      selectedRegionId: region.id,
      generatedAt: "2026-10-04T00:20:00.000Z",
      physicalGeneratedAt: physical.generated_at,
      physicalStatus: "unknown",
    })
    assert.equal(canvas.mode, "physical")
    const composed = workspaceCanvas({
      editing: false,
      view: "composed",
      selectedRegionId: region.id,
      generatedAt: "2026-10-04T00:20:00.000Z",
      physicalStatus: "missing",
    })
    assert.equal(composed.mode, "composed")
    const mask = workspaceCanvas({
      editing: false,
      view: "mask",
      selectedRegionId: region.id,
      generatedAt: "2026-10-04T00:20:00.000Z",
    })
    assert.equal(mask.mode, "mask")
  })
})
