import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import {
  HISTORICAL_OUTPUT_NOTE,
  MISSING_PHYSICAL_MESSAGE,
  PROFILE_SAMPLE_COUNT,
  UNVERIFIED_OUTPUT_NOTE,
  clampPixel,
  defaultProfileSelection,
  diagnosticsArtifactStatus,
  diagnosticsLayout,
  diagnosticsStatusLabel,
  diagnosticsTrustCopy,
  diagnosticsView,
  displayPointToPixel,
  histogramBars,
  histogramInputValid,
  physicalDiagnosticsPath,
  physicalProfilePath,
  pixelToDisplay,
  profileChartBounds,
  profilePathCommands,
  profilePolylineSegments,
  profileRequestAllowed,
  profileSelectionFromPointer,
  profileSelectionKey,
  selectionForMode,
  shouldApplyProfileResult,
  shouldFetchProfileOnPointer,
  showDiagnosticsFigures,
} from "./diagnosticsWorkspace.ts"

const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")
const api = readFileSync(new URL("../api/projects.ts", import.meta.url), "utf8")
const workspace = readFileSync(new URL("../pages/WorkspacePage.tsx", import.meta.url), "utf8")
const section = readFileSync(new URL("../components/DiagnosticsSection.tsx", import.meta.url), "utf8")

const plot = { left: 52, top: 16, width: 560, height: 150 }

describe("diagnostics API requests", () => {
  it("builds read-only diagnostics and profile URLs", () => {
    assert.equal(
      physicalDiagnosticsPath("project/1"),
      "/api/projects/project%2F1/relief/physical-height/diagnostics",
    )
    const horizontal = defaultProfileSelection(101, 51)
    assert.equal(horizontal.mode, "horizontal")
    assert.equal(horizontal.y_px, 25)
    assert.equal(horizontal.start_x_px, 0)
    assert.equal(horizontal.end_x_px, 100)
    const horizontalPath = physicalProfilePath("abc", horizontal)
    assert.equal(
      horizontalPath,
      "/api/projects/abc/relief/physical-height/profile?mode=horizontal&y_px=25&sample_count=256",
    )
    assert.equal(PROFILE_SAMPLE_COUNT, 256)
    const vertical = selectionForMode("vertical", 101, 51, horizontal)
    assert.equal(
      physicalProfilePath("abc", vertical),
      "/api/projects/abc/relief/physical-height/profile?mode=vertical&x_px=50&sample_count=256",
    )
    const line = selectionForMode("line", 101, 51, {
      ...horizontal,
      start_x_px: 1.5,
      start_y_px: 2,
      end_x_px: 8,
      end_y_px: 9,
    })
    assert.equal(
      physicalProfilePath("abc", line),
      "/api/projects/abc/relief/physical-height/profile?mode=line&start_x_px=1.5&start_y_px=2&end_x_px=8&end_y_px=9&sample_count=256",
    )
    assert.match(api, /export function getPhysicalDiagnostics[\s\S]*apiGet/)
    assert.match(api, /export function getPhysicalProfile[\s\S]*apiGet/)
    assert.doesNotMatch(
      api.slice(api.indexOf("export function getPhysicalDiagnostics")),
      /apiPost/,
    )
    assert.doesNotMatch(section, /generatePhysicalHeight|apiPost/)
  })
})

describe("diagnostics states", () => {
  it("keeps a missing physical record on the empty state and does not request a profile", () => {
    const absent = diagnosticsView({
      hasRecord: false,
      phase: "idle",
      errorStatus: null,
      errorMessage: null,
    })
    assert.equal(absent.kind, "absent")
    assert.equal(profileRequestAllowed(absent), false)
    assert.equal(showDiagnosticsFigures(absent), false)
    assert.equal(MISSING_PHYSICAL_MESSAGE, "Physical height has not been generated.")
    assert.match(workspace, /Diagnostics/)
    assert.match(section, /MISSING_PHYSICAL_MESSAGE/)
    assert.match(section, /onOpenRelief/)
  })

  it("labels current, stale, and unknown from the backend status", () => {
    assert.equal(diagnosticsArtifactStatus("current"), "current")
    assert.equal(diagnosticsArtifactStatus("stale"), "stale")
    assert.equal(diagnosticsArtifactStatus("unknown"), "unknown")
    assert.equal(diagnosticsArtifactStatus(null), "unknown")
    assert.equal(diagnosticsStatusLabel("current"), "Current")
    assert.equal(diagnosticsStatusLabel("stale"), "Stale")
    assert.equal(diagnosticsStatusLabel("unknown"), "Unknown")
    assert.equal(diagnosticsStatusLabel("missing"), "Missing")
    assert.equal(diagnosticsTrustCopy("current"), null)
    assert.equal(diagnosticsTrustCopy("stale"), HISTORICAL_OUTPUT_NOTE)
    assert.match(HISTORICAL_OUTPUT_NOTE, /not verified current geometry/)
    assert.equal(diagnosticsTrustCopy("unknown"), UNVERIFIED_OUTPUT_NOTE)
    assert.match(UNVERIFIED_OUTPUT_NOTE, /Unverified/)
  })

  it("distinguishes loading, not-found, invalid, and network failures", () => {
    const loading = diagnosticsView({
      hasRecord: true,
      phase: "loading",
      errorStatus: null,
      errorMessage: null,
    })
    assert.equal(loading.kind, "loading")
    assert.equal(showDiagnosticsFigures(loading), false)
    assert.equal(profileRequestAllowed(loading), false)
    const missing = diagnosticsView({
      hasRecord: true,
      phase: "error",
      errorStatus: 404,
      errorMessage: MISSING_PHYSICAL_MESSAGE,
    })
    assert.equal(missing.kind, "not-found")
    assert.equal(profileRequestAllowed(missing), false)
    const invalid = diagnosticsView({
      hasRecord: true,
      phase: "error",
      errorStatus: 422,
      errorMessage: "Profile coordinates are outside the image.",
    })
    assert.equal(invalid.kind, "invalid")
    if (invalid.kind === "invalid") {
      assert.match(invalid.message, /outside the image/)
    }
    const network = diagnosticsView({
      hasRecord: true,
      phase: "error",
      errorStatus: null,
      errorMessage: "Unable to reach the backend.",
    })
    assert.equal(network.kind, "network")
    const ready = diagnosticsView({
      hasRecord: true,
      phase: "ready",
      errorStatus: null,
      errorMessage: null,
    })
    assert.equal(ready.kind, "ready")
    assert.equal(showDiagnosticsFigures(ready), true)
    assert.equal(profileRequestAllowed(ready), true)
  })
})

describe("diagnostics geometry", () => {
  it("validates histogram edges and counts without inventing bins", () => {
    assert.equal(histogramInputValid(null), false)
    assert.equal(histogramInputValid({ edges_mm: [0, 1], counts: [1, 2] }), false)
    assert.equal(histogramInputValid({ edges_mm: [0, 2, 1], counts: [1, 0] }), false)
    assert.equal(histogramInputValid({ edges_mm: [0, 1, 2], counts: [3, 4] }), true)
    const bars = histogramBars([0, 10, 20], [2, 8], plot)
    assert.ok(bars)
    assert.equal(bars?.length, 2)
    assert.equal(bars?.[1]?.height, plot.height)
    assert.equal(bars?.[0]?.height, plot.height / 4)
    const flat = histogramBars([5, 5, 5], [4, 0], plot)
    assert.equal(flat?.length, 1)
    assert.equal(flat?.[0]?.startMm, 5)
    assert.equal(flat?.[0]?.count, 4)
    assert.equal(histogramBars([0, Number.NaN], [1], plot), null)
  })

  it("maps preview pixels through letterboxing to array pixel centers", () => {
    const mapped = displayPointToPixel({
      pointerX: 50.5,
      pointerY: 0.5,
      boxWidth: 200,
      boxHeight: 100,
      imageWidth: 100,
      imageHeight: 100,
      arrayWidth: 100,
      arrayHeight: 100,
    })
    assert.deepEqual(mapped, { x_px: 0, y_px: 0 })
    assert.equal(
      displayPointToPixel({
        pointerX: 10,
        pointerY: 10,
        boxWidth: 200,
        boxHeight: 100,
        imageWidth: 100,
        imageHeight: 100,
        arrayWidth: 100,
        arrayHeight: 100,
      }),
      null,
    )
    const center = pixelToDisplay({
      x_px: 0,
      y_px: 0,
      boxWidth: 200,
      boxHeight: 100,
      imageWidth: 100,
      imageHeight: 100,
      arrayWidth: 100,
      arrayHeight: 100,
    })
    assert.deepEqual(center, { x: 50.5, y: 0.5 })
    const roundTrip = displayPointToPixel({
      pointerX: 149.5,
      pointerY: 99.5,
      boxWidth: 200,
      boxHeight: 100,
      imageWidth: 100,
      imageHeight: 100,
      arrayWidth: 100,
      arrayHeight: 100,
    })
    assert.deepEqual(roundTrip, { x_px: 99, y_px: 99 })
  })

  it("selects horizontal, vertical, and custom lines and clamps endpoints", () => {
    const horizontal = defaultProfileSelection(10, 8)
    assert.equal(horizontal.mode, "horizontal")
    assert.equal(horizontal.y_px, 3.5)
    assert.equal(horizontal.start_y_px, 3.5)
    assert.equal(horizontal.end_y_px, 3.5)
    const moved = profileSelectionFromPointer({
      selection: horizontal,
      point: { x_px: 4, y_px: 6 },
      handle: "line",
      widthPx: 10,
      heightPx: 8,
    })
    assert.equal(moved.y_px, 6)
    assert.equal(moved.start_x_px, 0)
    assert.equal(moved.end_x_px, 9)
    const vertical = selectionForMode("vertical", 10, 8, moved)
    assert.equal(vertical.mode, "vertical")
    assert.equal(vertical.x_px, 4.5)
    assert.equal(vertical.start_y_px, 0)
    assert.equal(vertical.end_y_px, 7)
    const custom = profileSelectionFromPointer({
      selection: selectionForMode("line", 10, 8, vertical),
      point: { x_px: 40, y_px: -3 },
      handle: "end",
      widthPx: 10,
      heightPx: 8,
    })
    assert.equal(custom.end_x_px, 9)
    assert.equal(custom.end_y_px, 0)
    assert.equal(clampPixel(-4, 10), 0)
    assert.equal(clampPixel(25, 10), 9)
    assert.notEqual(profileSelectionKey(horizontal), profileSelectionKey(moved))
  })

  it("fetches a profile on pointer release and ignores an older in-flight response", () => {
    assert.equal(shouldFetchProfileOnPointer("move"), false)
    assert.equal(shouldFetchProfileOnPointer("up"), true)
    assert.equal(shouldApplyProfileResult(1, 2), false)
    assert.equal(shouldApplyProfileResult(2, 2), true)
  })

  it("breaks the profile chart at null heights instead of treating them as zero", () => {
    const samples = [
      { distance_mm: 0, height_mm: 1 },
      { distance_mm: 1, height_mm: null },
      { distance_mm: 2, height_mm: 4 },
      { distance_mm: 3, height_mm: 0 },
    ]
    const segments = profilePolylineSegments(samples)
    assert.equal(segments.length, 2)
    assert.deepEqual(segments[0], [{ distance_mm: 0, height_mm: 1 }])
    assert.equal(segments[1]?.[0]?.height_mm, 4)
    assert.equal(segments[1]?.[1]?.height_mm, 0)
    const paths = profilePathCommands(samples, plot)
    assert.equal(paths.length, 2)
    assert.equal(paths.some((path) => path.includes("L") && path.split("L").length > 2), false)
    const bounds = profileChartBounds(samples)
    assert.ok(bounds)
    assert.equal(bounds?.minHeight, 0)
    assert.equal(profileChartBounds([{ distance_mm: 0, height_mm: null }]), null)
  })
})

describe("diagnostics layout", () => {
  it("uses the desktop inspector width and stacks at 960 pixels and below", () => {
    for (const width of [1920, 1366, 1100]) {
      const layout = diagnosticsLayout(width)
      assert.equal(layout.stacked, false)
      assert.equal(layout.inspectorMinPx, 340)
      assert.equal(layout.inspectorMaxPx, 380)
    }
    for (const width of [960, 800]) {
      const layout = diagnosticsLayout(width)
      assert.equal(layout.stacked, true)
      assert.equal(layout.inspectorMinPx, null)
    }
    assert.match(css, /\.diagnostics-workspace\.is-split/)
    assert.match(css, /grid-template-columns:\s*minmax\(0,\s*1fr\)\s*minmax\(340px,\s*380px\)/)
    assert.match(css, /@media \(max-width:\s*960px\)[\s\S]*diagnostics-workspace\.is-split/)
    assert.match(css, /object-fit:\s*contain/)
  })
})
