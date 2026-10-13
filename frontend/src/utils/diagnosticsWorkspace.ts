import type { ArtifactStatus, ProfileMode } from "../types/project.ts"
import { artifactStatus, statusLabel } from "./provenanceStatus.ts"
import { inspectorSizing } from "./inspectorLayout.ts"

export const MISSING_PHYSICAL_MESSAGE = "Physical height has not been generated."
export const PROFILE_SAMPLE_COUNT = 256

export const HISTORICAL_OUTPUT_NOTE =
  "Historical output. These figures are not verified current geometry."
export const UNVERIFIED_OUTPUT_NOTE =
  "Unverified output. These figures are not verified current geometry."

export type ProfileSelection = {
  mode: ProfileMode
  x_px: number
  y_px: number
  start_x_px: number
  start_y_px: number
  end_x_px: number
  end_y_px: number
}

export type ContainedRect = { x: number; y: number; width: number; height: number }

export type DiagnosticsView =
  | { kind: "absent" }
  | { kind: "loading" }
  | { kind: "not-found"; message: string }
  | { kind: "invalid"; message: string }
  | { kind: "network"; message: string }
  | { kind: "failed"; message: string }
  | { kind: "ready" }

export function physicalDiagnosticsPath(projectId: string): string {
  return `/api/projects/${encodeURIComponent(projectId)}/relief/physical-height/diagnostics`
}

export function physicalProfilePath(projectId: string, selection: ProfileSelection): string {
  const params = new URLSearchParams()
  params.set("mode", selection.mode)
  if (selection.mode === "horizontal") params.set("y_px", String(selection.y_px))
  if (selection.mode === "vertical") params.set("x_px", String(selection.x_px))
  if (selection.mode === "line") {
    params.set("start_x_px", String(selection.start_x_px))
    params.set("start_y_px", String(selection.start_y_px))
    params.set("end_x_px", String(selection.end_x_px))
    params.set("end_y_px", String(selection.end_y_px))
  }
  params.set("sample_count", String(PROFILE_SAMPLE_COUNT))
  return `/api/projects/${encodeURIComponent(projectId)}/relief/physical-height/profile?${params.toString()}`
}

export function diagnosticsView(input: {
  hasRecord: boolean
  phase: "idle" | "loading" | "ready" | "error"
  errorStatus: number | null
  errorMessage: string | null
}): DiagnosticsView {
  if (!input.hasRecord) return { kind: "absent" }
  if (input.phase === "loading" || input.phase === "idle") return { kind: "loading" }
  if (input.phase === "ready") return { kind: "ready" }
  const message = input.errorMessage ?? "Physical height diagnostics could not be loaded."
  if (input.errorStatus == null) return { kind: "network", message }
  if (input.errorStatus === 404) return { kind: "not-found", message }
  if (input.errorStatus === 422) return { kind: "invalid", message }
  return { kind: "failed", message }
}

export function profileRequestAllowed(view: DiagnosticsView): boolean {
  return view.kind === "ready"
}

export function showDiagnosticsFigures(view: DiagnosticsView): boolean {
  return view.kind === "ready"
}

export function diagnosticsArtifactStatus(status: string | null | undefined): ArtifactStatus {
  return artifactStatus({ record: true }, status)
}

export function diagnosticsTrustCopy(status: ArtifactStatus): string | null {
  if (status === "stale") return HISTORICAL_OUTPUT_NOTE
  if (status === "unknown") return UNVERIFIED_OUTPUT_NOTE
  return null
}

export function diagnosticsStatusLabel(status: ArtifactStatus): string {
  return statusLabel(status)
}

export function diagnosticsLayout(viewportWidth: number): {
  stacked: boolean
  inspectorMinPx: number | null
  inspectorMaxPx: number | null
} {
  const sizing = inspectorSizing(viewportWidth)
  if (sizing.stacked) return { stacked: true, inspectorMinPx: null, inspectorMaxPx: null }
  return {
    stacked: false,
    inspectorMinPx: sizing.width.min,
    inspectorMaxPx: sizing.width.max,
  }
}

export function clampPixel(value: number, count: number): number {
  if (!Number.isFinite(value) || count < 1) return 0
  if (count === 1) return 0
  return Math.min(count - 1, Math.max(0, value))
}

export function defaultProfileSelection(widthPx: number, heightPx: number): ProfileSelection {
  const y = clampPixel((heightPx - 1) / 2, heightPx)
  const x = clampPixel((widthPx - 1) / 2, widthPx)
  return {
    mode: "horizontal",
    x_px: x,
    y_px: y,
    start_x_px: 0,
    start_y_px: y,
    end_x_px: clampPixel(widthPx - 1, widthPx),
    end_y_px: y,
  }
}

export function selectionForMode(
  mode: ProfileMode,
  widthPx: number,
  heightPx: number,
  current: ProfileSelection,
): ProfileSelection {
  if (mode === "horizontal") {
    const y = clampPixel(current.y_px, heightPx)
    return {
      ...current,
      mode,
      y_px: y,
      start_x_px: 0,
      start_y_px: y,
      end_x_px: clampPixel(widthPx - 1, widthPx),
      end_y_px: y,
    }
  }
  if (mode === "vertical") {
    const x = clampPixel(current.x_px, widthPx)
    return {
      ...current,
      mode,
      x_px: x,
      start_x_px: x,
      start_y_px: 0,
      end_x_px: x,
      end_y_px: clampPixel(heightPx - 1, heightPx),
    }
  }
  return {
    ...current,
    mode: "line",
    start_x_px: clampPixel(current.start_x_px, widthPx),
    start_y_px: clampPixel(current.start_y_px, heightPx),
    end_x_px: clampPixel(current.end_x_px, widthPx),
    end_y_px: clampPixel(current.end_y_px, heightPx),
  }
}

export function profileSelectionFromPointer(input: {
  selection: ProfileSelection
  point: { x_px: number; y_px: number } | null
  handle: "start" | "end" | "line"
  widthPx: number
  heightPx: number
}): ProfileSelection {
  if (!input.point) return input.selection
  const x = clampPixel(input.point.x_px, input.widthPx)
  const y = clampPixel(input.point.y_px, input.heightPx)
  if (input.selection.mode === "horizontal") {
    return selectionForMode("horizontal", input.widthPx, input.heightPx, {
      ...input.selection,
      y_px: y,
    })
  }
  if (input.selection.mode === "vertical") {
    return selectionForMode("vertical", input.widthPx, input.heightPx, {
      ...input.selection,
      x_px: x,
    })
  }
  const handle =
    input.handle === "line" ? nearerEndpoint(input.selection, x, y) : input.handle
  if (handle === "end") {
    return { ...input.selection, end_x_px: x, end_y_px: y }
  }
  return { ...input.selection, start_x_px: x, start_y_px: y }
}

function nearerEndpoint(selection: ProfileSelection, x: number, y: number): "start" | "end" {
  const start =
    (selection.start_x_px - x) ** 2 + (selection.start_y_px - y) ** 2
  const end = (selection.end_x_px - x) ** 2 + (selection.end_y_px - y) ** 2
  return start <= end ? "start" : "end"
}

export function nudgeProfileSelection(
  selection: ProfileSelection,
  key: string,
  widthPx: number,
  heightPx: number,
): ProfileSelection | null {
  const step = key === "ArrowUp" || key === "ArrowLeft" ? -1 : 1
  if (key !== "ArrowUp" && key !== "ArrowDown" && key !== "ArrowLeft" && key !== "ArrowRight") {
    return null
  }
  if (selection.mode === "horizontal" && (key === "ArrowUp" || key === "ArrowDown")) {
    return selectionForMode("horizontal", widthPx, heightPx, {
      ...selection,
      y_px: selection.y_px + step,
    })
  }
  if (selection.mode === "vertical" && (key === "ArrowLeft" || key === "ArrowRight")) {
    return selectionForMode("vertical", widthPx, heightPx, {
      ...selection,
      x_px: selection.x_px + step,
    })
  }
  return null
}

/** Profile requests are committed on pointer release, not while the pointer moves. */
export function shouldFetchProfileOnPointer(phase: "move" | "up"): boolean {
  return phase === "up"
}

export function shouldApplyProfileResult(requestId: number, latestRequestId: number): boolean {
  return requestId === latestRequestId
}

export function profileSelectionKey(selection: ProfileSelection): string {
  return [
    selection.mode,
    selection.x_px,
    selection.y_px,
    selection.start_x_px,
    selection.start_y_px,
    selection.end_x_px,
    selection.end_y_px,
  ].join("|")
}

export function containedImageRect(
  boxWidth: number,
  boxHeight: number,
  imageWidth: number,
  imageHeight: number,
): ContainedRect | null {
  if (boxWidth <= 0 || boxHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) return null
  const scale = Math.min(boxWidth / imageWidth, boxHeight / imageHeight)
  const width = imageWidth * scale
  const height = imageHeight * scale
  return {
    x: (boxWidth - width) / 2,
    y: (boxHeight - height) / 2,
    width,
    height,
  }
}

/** Maps a pointer in the preview box to array pixel-center coordinates. */
export function displayPointToPixel(input: {
  pointerX: number
  pointerY: number
  boxWidth: number
  boxHeight: number
  imageWidth: number
  imageHeight: number
  arrayWidth: number
  arrayHeight: number
}): { x_px: number; y_px: number } | null {
  const rect = containedImageRect(
    input.boxWidth,
    input.boxHeight,
    input.imageWidth,
    input.imageHeight,
  )
  if (!rect || input.arrayWidth < 1 || input.arrayHeight < 1) return null
  if (
    input.pointerX < rect.x ||
    input.pointerY < rect.y ||
    input.pointerX > rect.x + rect.width ||
    input.pointerY > rect.y + rect.height
  ) {
    return null
  }
  const xEdge = ((input.pointerX - rect.x) / rect.width) * input.arrayWidth
  const yEdge = ((input.pointerY - rect.y) / rect.height) * input.arrayHeight
  return {
    x_px: clampPixel(xEdge - 0.5, input.arrayWidth),
    y_px: clampPixel(yEdge - 0.5, input.arrayHeight),
  }
}

export function pixelToDisplay(input: {
  x_px: number
  y_px: number
  boxWidth: number
  boxHeight: number
  imageWidth: number
  imageHeight: number
  arrayWidth: number
  arrayHeight: number
}): { x: number; y: number } | null {
  const rect = containedImageRect(
    input.boxWidth,
    input.boxHeight,
    input.imageWidth,
    input.imageHeight,
  )
  if (!rect || input.arrayWidth < 1 || input.arrayHeight < 1) return null
  return {
    x: rect.x + ((input.x_px + 0.5) / input.arrayWidth) * rect.width,
    y: rect.y + ((input.y_px + 0.5) / input.arrayHeight) * rect.height,
  }
}

export type ChartSample = { distance_mm: number; height_mm: number | null }

export function profilePolylineSegments(
  samples: ChartSample[],
): Array<Array<{ distance_mm: number; height_mm: number }>> {
  const segments: Array<Array<{ distance_mm: number; height_mm: number }>> = []
  let current: Array<{ distance_mm: number; height_mm: number }> = []
  for (const sample of samples) {
    if (
      sample.height_mm == null ||
      !Number.isFinite(sample.height_mm) ||
      !Number.isFinite(sample.distance_mm)
    ) {
      if (current.length > 0) segments.push(current)
      current = []
      continue
    }
    current.push({ distance_mm: sample.distance_mm, height_mm: sample.height_mm })
  }
  if (current.length > 0) segments.push(current)
  return segments
}

export type PlotRect = { left: number; top: number; width: number; height: number }

export function profileChartBounds(samples: ChartSample[]): {
  minDistance: number
  maxDistance: number
  minHeight: number
  maxHeight: number
} | null {
  const distances = samples
    .map((sample) => sample.distance_mm)
    .filter((value) => Number.isFinite(value))
  const heights = samples
    .map((sample) => sample.height_mm)
    .filter((value): value is number => value != null && Number.isFinite(value))
  if (distances.length === 0 || heights.length === 0) return null
  let minHeight = Math.min(...heights)
  let maxHeight = Math.max(...heights)
  if (minHeight === maxHeight) {
    minHeight -= 0.5
    maxHeight += 0.5
  }
  let minDistance = Math.min(...distances)
  let maxDistance = Math.max(...distances)
  if (minDistance === maxDistance) maxDistance = minDistance + 1
  return { minDistance, maxDistance, minHeight, maxHeight }
}

export function chartPoint(
  distance: number,
  height: number,
  bounds: { minDistance: number; maxDistance: number; minHeight: number; maxHeight: number },
  plot: PlotRect,
): { x: number; y: number } {
  const distanceSpan = bounds.maxDistance - bounds.minDistance || 1
  const heightSpan = bounds.maxHeight - bounds.minHeight || 1
  return {
    x: plot.left + ((distance - bounds.minDistance) / distanceSpan) * plot.width,
    y: plot.top + (1 - (height - bounds.minHeight) / heightSpan) * plot.height,
  }
}

export function profilePathCommands(
  samples: ChartSample[],
  plot: PlotRect,
): string[] {
  const bounds = profileChartBounds(samples)
  if (!bounds) return []
  return profilePolylineSegments(samples).map((segment) =>
    segment
      .map((point, index) => {
        const mapped = chartPoint(point.distance_mm, point.height_mm, bounds, plot)
        const command = index === 0 ? "M" : "L"
        return `${command}${mapped.x},${mapped.y}`
      })
      .join(" "),
  )
}

export function nearestProfileSample<T extends ChartSample>(
  samples: T[],
  distanceMm: number,
): T | null {
  let best: T | null = null
  let bestDistance = Number.POSITIVE_INFINITY
  for (const sample of samples) {
    if (!Number.isFinite(sample.distance_mm)) continue
    const delta = Math.abs(sample.distance_mm - distanceMm)
    if (delta < bestDistance) {
      best = sample
      bestDistance = delta
    }
  }
  return best
}

export function histogramInputValid(histogram: {
  edges_mm: number[]
  counts: number[]
} | null): boolean {
  if (!histogram) return false
  if (histogram.counts.length === 0) return false
  if (histogram.edges_mm.length !== histogram.counts.length + 1) return false
  if (histogram.edges_mm.some((edge) => !Number.isFinite(edge))) return false
  if (histogram.counts.some((count) => !Number.isFinite(count) || count < 0)) return false
  for (let index = 1; index < histogram.edges_mm.length; index += 1) {
    if (histogram.edges_mm[index]! < histogram.edges_mm[index - 1]!) return false
  }
  return true
}

export type HistogramBar = {
  x: number
  y: number
  width: number
  height: number
  count: number
  startMm: number
  endMm: number
}

export function histogramBars(
  edges: number[],
  counts: number[],
  plot: PlotRect,
): HistogramBar[] | null {
  if (!histogramInputValid({ edges_mm: edges, counts })) return null
  const maxCount = Math.max(...counts, 1)
  const minEdge = edges[0] ?? 0
  const maxEdge = edges[edges.length - 1] ?? minEdge
  if (maxEdge === minEdge) {
    const count = counts[0] ?? 0
    const height = (count / maxCount) * plot.height
    return [
      {
        x: plot.left,
        y: plot.top + plot.height - height,
        width: plot.width,
        height,
        count,
        startMm: minEdge,
        endMm: maxEdge,
      },
    ]
  }
  const span = maxEdge - minEdge
  return counts.map((count, index) => {
    const start = edges[index] ?? minEdge
    const end = edges[index + 1] ?? start
    const x = plot.left + ((start - minEdge) / span) * plot.width
    const width = Math.max(0, ((end - start) / span) * plot.width)
    const height = (count / maxCount) * plot.height
    return {
      x,
      y: plot.top + plot.height - height,
      width,
      height,
      count,
      startMm: start,
      endMm: end,
    }
  })
}

export function formatMillimeters(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—"
  const rounded = Math.round(value * 1000) / 1000
  return `${rounded} mm`
}

export function formatPixel(value: number): string {
  if (!Number.isFinite(value)) return "—"
  const rounded = Math.round(value * 100) / 100
  return String(rounded)
}
