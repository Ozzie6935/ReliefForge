import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import type { DepthConditioningParameters } from "../types/project.ts"
import {
  DEPTH_INSPECTOR_CONTROLS,
  DEPTH_VIEWS,
  RELATIVE_DEPTH_LABEL,
  SOURCE_FORMAT_NOTE,
  SOURCE_REPLACE_WARNING,
  conditioningIsPending,
  depthParametersMatch,
  depthViewChange,
  depthWorkspaceFrame,
  meshViewportFrame,
  sourcePreviewState,
  sourceUploadEnabled,
  sourceWorkspaceFrame,
} from "./workspaceLayout.ts"

const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")

const saved: DepthConditioningParameters = {
  far_percentile: 1,
  near_percentile: 99,
  gamma: 1,
  contrast: 1,
  smoothing: 0,
  edge_preservation: 75,
  invert: false,
}

describe("shared workspace layout", () => {
  it("keeps one desktop inspector width and stacks at 960px", () => {
    for (const width of [1920, 1366, 1100]) {
      const frame = depthWorkspaceFrame(width)
      assert.equal(frame.stacked, false)
      assert.equal(frame.shell, "full-height")
      assert.equal(frame.belowNavbar, true)
      assert.equal(frame.pinnedPrimaryAction, "Apply Conditioning")
      assert.equal(frame.previewFit, "contain")
      if (frame.inspectorWidth === "fluid") continue
      assert.equal(frame.inspectorWidth.min, 340)
      assert.equal(frame.inspectorWidth.max, 380)
    }
    for (const width of [960, 800, 700]) {
      const frame = depthWorkspaceFrame(width)
      assert.equal(frame.stacked, true)
      assert.equal(frame.inspectorWidth, "fluid")
    }
    assert.match(css, /\.depth-workspace\.is-split\s*\{[^}]*display:\s*contents/)
    assert.match(css, /\.depth-panel\s*\{[^}]*grid-row:\s*1\s*\/\s*-1/)
    assert.match(css, /\.depth-fit-image\s*\{[^}]*object-fit:\s*contain/)
  })

  it("lists the conditioning controls and labels relative depth as dimensionless", () => {
    assert.deepEqual(DEPTH_INSPECTOR_CONTROLS, [
      "Far Clip",
      "Near Clip",
      "Gamma",
      "Contrast",
      "Smoothing",
      "Edge Preservation",
      "Invert Depth",
      "Reset to Defaults",
    ])
    assert.match(RELATIVE_DEPTH_LABEL, /dimensionless/)
    assert.match(RELATIVE_DEPTH_LABEL, /not a physical CNC height/)
    assert.deepEqual(
      DEPTH_VIEWS.map((view) => view.id),
      ["conditioned", "source", "raw"],
    )
  })

  it("marks unsaved slider edits as pending and leaves a matching draft applied", () => {
    assert.equal(depthParametersMatch(saved, saved), true)
    assert.equal(conditioningIsPending(saved, saved), false)
    assert.equal(conditioningIsPending({ ...saved, gamma: 1.4 }, saved), true)
    assert.equal(conditioningIsPending({ ...saved, invert: true }, saved), true)
  })

  it("keeps a depth view change from generating or applying", () => {
    const change = depthViewChange()
    assert.equal(change.generate, false)
    assert.equal(change.apply, false)
    assert.equal(change.writesArtifacts, false)
  })

  it("uses the shared full-height frame for the source workspace", () => {
    for (const width of [1920, 1366, 1100]) {
      const frame = sourceWorkspaceFrame(width)
      assert.equal(frame.stacked, false)
      assert.equal(frame.pinnedPrimaryAction, "Upload Image")
      assert.equal(frame.previewFit, "contain")
      if (frame.inspectorWidth === "fluid") continue
      assert.equal(frame.inspectorWidth.max, 380)
    }
    assert.equal(sourceWorkspaceFrame(800).stacked, true)
    assert.equal(sourceWorkspaceFrame(960).stacked, true)
    assert.match(css, /\.source-workspace\.is-split[\s\S]*display:\s*contents/)
    assert.match(css, /\.source-panel\s*\{[^}]*grid-row:\s*1\s*\/\s*-1/)
    assert.match(SOURCE_FORMAT_NOTE, /JPEG or PNG/)
    assert.match(SOURCE_FORMAT_NOTE, /50 MB/)
    assert.match(SOURCE_REPLACE_WARNING, /may require depth and relief/)
  })

  it("gives the 3D viewport the remaining desktop width and stacks below 960px", () => {
    for (const width of [1440, 1280, 1024]) {
      const frame = meshViewportFrame(width)
      assert.equal(frame.stacked, false)
      assert.equal(frame.viewportTrack, "minmax(0, 1fr)")
      assert.equal(frame.inspectorMinPx, 340)
      assert.equal(frame.inspectorMaxPx, 380)
      assert.equal(frame.inspectorTrack, "minmax(340px, 380px)")
      assert.equal(frame.horizontalOverflow, "clip")
      assert.equal(frame.canvasFillsViewport, true)
    }
    for (const width of [960, 768]) {
      const frame = meshViewportFrame(width)
      assert.equal(frame.stacked, true)
      assert.equal(frame.viewportTrack, "minmax(0, 1.8fr)")
      assert.equal(frame.inspectorMinPx, null)
      assert.equal(frame.inspectorMaxPx, null)
      assert.equal(frame.inspectorTrack, "minmax(0, 1fr)")
      assert.equal(frame.horizontalOverflow, "clip")
      assert.equal(frame.canvasFillsViewport, true)
    }
    assert.match(css, /@media \(min-width: 961px\) \{[^}]*\.panel:has\(\.mesh-workspace\.is-split\) \{[^}]*minmax\(0, 1fr\) minmax\(340px, 380px\)/)
    assert.match(css, /@media \(max-width: 960px\) \{[^}]*\.panel:has\(\.mesh-workspace\.is-split\) \{[^}]*minmax\(0, 1fr\);[^}]*minmax\(0, 1\.8fr\) minmax\(0, 1fr\)/)
    assert.match(css, /\.mesh-workspace\.is-split \.mesh-viewport \{[^}]*flex:\s*1 1 auto/)
    assert.match(css, /\.mesh-workspace\.is-split \.mesh-viewport \{[^}]*min-width:\s*0/)
    assert.match(css, /\.mesh-workspace\.is-split \.mesh-main \{[^}]*overflow:\s*hidden/)
    assert.match(css, /\.panel:has\(\.mesh-workspace\.is-split\) \{[^}]*overflow-x:\s*clip/)
    assert.match(css, /\.workspace-nav/)
    assert.match(css, /\.export-workspace\.is-split/)
    assert.doesNotMatch(
      css,
      /\.panel:has\(\.export-workspace\.is-split\) \{[^}]*1\.8fr/,
    )
  })

  it("describes source preview states and keeps upload disabled until a file is chosen", () => {
    assert.equal(sourcePreviewState({ hasSource: false, imageLoaded: false, imageFailed: false }), "empty")
    assert.equal(sourcePreviewState({ hasSource: true, imageLoaded: false, imageFailed: false }), "loading")
    assert.equal(sourcePreviewState({ hasSource: true, imageLoaded: true, imageFailed: false }), "loaded")
    assert.equal(sourcePreviewState({ hasSource: true, imageLoaded: false, imageFailed: true }), "failed")
    assert.equal(sourceUploadEnabled(false, false), false)
    assert.equal(sourceUploadEnabled(true, true), false)
    assert.equal(sourceUploadEnabled(true, false), true)
  })
})
