import assert from "node:assert/strict"
import { describe, it } from "node:test"

import type { PhysicalHeightRecord } from "../types/project.ts"
import {
  PHYSICAL_EXAMPLE_MM,
  PHYSICAL_EXAMPLE_NOTE,
  PHYSICAL_ORDER_ERROR,
  commitPhysicalMillimeters,
  physicalActionLabel,
  physicalCanonicalMillimeters,
  physicalDisplayStatus,
  physicalDraftProblem,
  physicalInputs,
  physicalInputsAfterUnitChange,
  physicalStatusCopy,
  physicalStockNote,
  physicalViewRequests,
} from "./physicalHeight.ts"
import { workspaceCanvas } from "./reliefWorkspace.ts"

const saved: PhysicalHeightRecord = {
  stock_thickness_mm: 20,
  base_thickness_mm: 6,
  maximum_surface_height_mm: 16,
  generated_at: "2026-10-08T00:00:00Z",
  height_path: "depth/composition/physical/physical_height_mm.npy",
  preview_path: "depth/composition/physical/preview.png",
  stats_path: "depth/composition/physical/stats.json",
  source_relief_sha256: "abc",
  source_relief_generated_at: "2026-10-08T00:00:00Z",
  width_px: 4,
  height_px: 2,
  dtype: "float32",
  height_min_mm: 6,
  height_max_mm: 16,
  maximum_cut_depth_mm: 14,
  relief_amplitude_mm: 10,
  status: "current",
}

describe("physical height settings", () => {
  it("uses the application default when nothing is saved", () => {
    assert.deepEqual(physicalCanonicalMillimeters(null), {
      stock: 38.1,
      base: PHYSICAL_EXAMPLE_MM.base,
      peak: PHYSICAL_EXAMPLE_MM.peak,
      stockSource: "application-default",
    })
    assert.equal(PHYSICAL_EXAMPLE_NOTE, "Example values, not machining recommendations.")
    const inputs = physicalInputs(null, "mm")
    assert.equal(inputs.usingExample, true)
    assert.equal(inputs.stockSource, "application-default")
    assert.equal(inputs.stock, "38.10")
    assert.equal(inputs.base, "5")
    assert.equal(inputs.peak, "17")
    assert.match(physicalStockNote("application-default", true), /Application default/)
    assert.match(physicalStockNote("application-default", true), /38\.10 mm/)
  })

  it("uses a project thickness before the application default and a saved physical thickness before both", () => {
    const project = physicalInputs(null, "mm", 50.8)
    assert.equal(project.stockSource, "project")
    assert.equal(project.stock, "50.80")
    assert.equal(project.base, "5")
    assert.equal(project.peak, "17")
    assert.match(physicalStockNote("project", true), /Project stock thickness/)

    const savedInputs = physicalInputs(saved, "mm", 50.8)
    assert.equal(savedInputs.usingExample, false)
    assert.equal(savedInputs.stockSource, "saved-physical")
    assert.equal(savedInputs.stock, "20.00")
    assert.equal(savedInputs.base, "6")
    assert.equal(savedInputs.peak, "16")
    assert.equal(physicalStockNote("saved-physical", true), "Saved physical-height thickness.")
    assert.equal(physicalCanonicalMillimeters(saved, 50.8).stock, 20)
  })

  it("sends canonical millimeters when the inch display is unedited", () => {
    const draft = physicalInputs(null, "in")
    const canonical = physicalCanonicalMillimeters(null)
    const committed = commitPhysicalMillimeters(draft, canonical, "in")
    assert.deepEqual(committed, {
      stock_thickness_mm: 38.1,
      base_thickness_mm: 5,
      maximum_surface_height_mm: 17,
    })
  })

  it("reports an incompatible thickness without changing base or peak", () => {
    const draft = { stock: "10", base: "5", peak: "17" }
    const canonical = physicalCanonicalMillimeters(null)
    assert.equal(physicalDraftProblem(draft, canonical, "mm"), "order")
    assert.equal(PHYSICAL_ORDER_ERROR.includes("cannot exceed stock thickness"), true)
    assert.equal(draft.base, "5")
    assert.equal(draft.peak, "17")
    assert.equal(
      commitPhysicalMillimeters(draft, canonical, "mm"),
      null,
    )
    const switched = physicalInputsAfterUnitChange(
      { ...draft, usingExample: true, stockSource: "application-default" },
      canonical,
      "mm",
      "in",
    )
    assert.equal(switched.base, physicalInputs(null, "in").base)
    assert.equal(switched.peak, physicalInputs(null, "in").peak)
    assert.equal(physicalDraftProblem(switched, canonical, "in"), "order")
  })

  it("labels missing, unknown, current, and stale output", () => {
    assert.equal(physicalDisplayStatus(null), "missing")
    assert.equal(physicalDisplayStatus(saved), "current")
    assert.equal(physicalDisplayStatus({ ...saved, status: "stale" }), "stale")
    assert.equal(physicalDisplayStatus({ ...saved, status: "unknown" }), "unknown")
    assert.equal(physicalDisplayStatus({ ...saved, status: null }), "unknown")
    assert.notEqual(physicalDisplayStatus({ ...saved, status: "unknown" }), "stale")
    assert.notEqual(physicalDisplayStatus(null), "stale")
    assert.equal(physicalActionLabel(null), "Generate")
    assert.equal(physicalActionLabel(saved), "Regenerate")
    assert.match(physicalStatusCopy("stale"), /regeneration is required/)
    assert.match(physicalStatusCopy("unknown"), /cannot be verified/)
    assert.match(physicalStatusCopy("current"), /recorded dependencies/)
  })

  it("shows a stale preview without treating it as current or regenerating it", () => {
    const requests = physicalViewRequests()
    assert.equal(requests.regenerate, false)
    assert.equal(requests.stl, false)
    assert.equal(requests.gcode, false)
    const canvas = workspaceCanvas({
      editing: false,
      view: "physical",
      selectedRegionId: null,
      generatedAt: "2026-10-08T00:00:00Z",
      physicalGeneratedAt: saved.generated_at,
      physicalStatus: "stale",
    })
    assert.equal(canvas.mode, "physical")
    if (canvas.mode === "physical") {
      assert.equal(canvas.status, "stale")
      assert.equal(canvas.readOnly, true)
    }
    const missing = workspaceCanvas({
      editing: false,
      view: "physical",
      selectedRegionId: null,
      generatedAt: null,
      physicalStatus: "missing",
    })
    assert.equal(missing.mode, "physical-empty")
    const unknown = workspaceCanvas({
      editing: false,
      view: "physical",
      selectedRegionId: null,
      generatedAt: "2026-10-08T00:00:00Z",
      physicalGeneratedAt: saved.generated_at,
      physicalStatus: "unknown",
    })
    assert.equal(unknown.mode, "physical")
    if (unknown.mode === "physical") {
      assert.equal(unknown.status, "unknown")
      assert.notEqual(unknown.status, "stale")
    }
  })
})
