import { ApiError, readErrorDetail } from "../api/client.ts"
import type { ArtifactStatus } from "../types/project.ts"
import { formatLength, type DisplayUnit } from "./units.ts"

export const MISSING_PHYSICAL_MESSAGE = "Physical height has not been generated."
export const STL_UNIT_NOTE =
  "STL coordinates are millimeters. STL has no reliable built-in unit metadata."
export const ARTWORK_LOCAL_NOTE =
  "Artwork-local geometry. Stock placement is not applied. Stock width and stock height are not the export rectangle."
export const Z_CONVENTION_NOTE = "Bottom Z = 0 mm. Physical top Z comes from the saved height."
export const PLAN_NOT_MESH_NOTE =
  "This readiness check is a plan. It does not validate the mesh. Final geometry validation runs during the export request."
export const REDUCED_QUALITY_NOTE =
  "Reduced sampling may omit narrow features or fine facial detail. Triangle count does not promise a quality level."
export const ESTIMATE_NOTE =
  "Triangle count, file size, and peak memory are estimates. Triangle count does not describe surface quality."
export const SUGGESTION_NOTE = "Suggested size. Check readiness before these dimensions are used."
export const CERTIFICATE_NOTE = "An STL is not a machining certificate."

export type ExportMode = "native" | "reduced"

export type ExportSelection =
  | { mode: "native" }
  | { mode: "reduced"; heightPx: number; widthPx: number }

export type ExportIssue = { code: string; message: string }

export type ExportGrid = { height_px: number; width_px: number }

export type ExportReadiness = {
  project_id: string
  provenance_status: ArtifactStatus
  export_target: string
  export_certificate: string
  coordinate_convention: string
  resolution_policy: string
  topology_note: string
  mode: ExportMode
  source_grid: ExportGrid | null
  planned_grid: ExportGrid | null
  spacing_mm: { x_mm: number; y_mm: number; uniform: boolean } | null
  artwork_width_mm: number
  artwork_height_mm: number
  stock_width_mm: number
  stock_height_mm: number
  stock_thickness_mm: number | null
  base_thickness_mm: number | null
  maximum_surface_height_mm: number | null
  placement: {
    recorded: boolean
    applied: boolean
    center_x_mm: number | null
    center_y_mm: number | null
    origin: string | null
  }
  z_minimum_mm: number | null
  z_maximum_mm: number | null
  estimated_vertices: number | null
  estimated_triangles: number | null
  estimated_stl_bytes: number | null
  estimated_peak_memory_bytes: number | null
  topology_expected: string
  topology_validated: boolean
  self_intersection_checked: boolean
  zero_area_measured: boolean
  blocking_errors: ExportIssue[]
  warnings: ExportIssue[]
  corrective_actions: string[]
  ready_for_export: boolean
  limits: {
    max_triangles: number
    max_stl_bytes: number
    max_peak_memory_bytes: number
  }
}

export type StlDownloadResult = {
  body: ArrayBuffer
  filename: string
  units: string | null
  certificate: string | null
  selfIntersectionChecked: string | null
  limitation: string | null
}

export type DownloadAnchor = {
  href: string
  download: string
  click: () => void
  remove: () => void
}

export type DownloadHost = {
  createObjectURL: (blob: Blob) => string
  revokeObjectURL: (url: string) => void
  createElement: (tag: string) => DownloadAnchor
  appendChild: (node: DownloadAnchor) => void
}

export type Fact = { term: string; detail: string }

const STATUSES: readonly ArtifactStatus[] = ["missing", "unknown", "current", "stale"]
const SAFE_STL_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.stl$/

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function nullableNumber(value: unknown): number | null | undefined {
  if (value === null) return null
  const parsed = finiteNumber(value)
  return parsed === null ? undefined : parsed
}

function grid(value: unknown): ExportGrid | null | undefined {
  if (value === null) return null
  if (!isRecord(value)) return undefined
  const height = finiteNumber(value.height_px)
  const width = finiteNumber(value.width_px)
  if (height === null || width === null) return undefined
  return { height_px: height, width_px: width }
}

function issues(value: unknown): ExportIssue[] | null {
  if (!Array.isArray(value)) return null
  const parsed: ExportIssue[] = []
  for (const item of value) {
    if (!isRecord(item) || typeof item.code !== "string" || typeof item.message !== "string") return null
    parsed.push({ code: item.code, message: item.message })
  }
  return parsed
}

function strings(value: unknown): string[] | null {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) return null
  return value as string[]
}

export function parseExportReadiness(value: unknown): ExportReadiness | null {
  if (!isRecord(value)) return null
  const status = value.provenance_status
  const mode = value.mode
  if (typeof value.project_id !== "string") return null
  if (typeof status !== "string" || !STATUSES.includes(status as ArtifactStatus)) return null
  if (mode !== "native" && mode !== "reduced") return null
  if (typeof value.export_target !== "string") return null
  if (typeof value.export_certificate !== "string") return null
  if (typeof value.coordinate_convention !== "string") return null
  if (typeof value.resolution_policy !== "string") return null
  if (typeof value.topology_note !== "string") return null
  if (typeof value.topology_expected !== "string") return null
  if (typeof value.topology_validated !== "boolean") return null
  if (typeof value.self_intersection_checked !== "boolean") return null
  if (typeof value.zero_area_measured !== "boolean") return null
  if (typeof value.ready_for_export !== "boolean") return null
  const source = grid(value.source_grid)
  const planned = grid(value.planned_grid)
  if (source === undefined || planned === undefined) return null
  const artworkWidth = finiteNumber(value.artwork_width_mm)
  const artworkHeight = finiteNumber(value.artwork_height_mm)
  const stockWidth = finiteNumber(value.stock_width_mm)
  const stockHeight = finiteNumber(value.stock_height_mm)
  if (artworkWidth === null || artworkHeight === null || stockWidth === null || stockHeight === null) {
    return null
  }
  const stockThickness = nullableNumber(value.stock_thickness_mm)
  const baseThickness = nullableNumber(value.base_thickness_mm)
  const peak = nullableNumber(value.maximum_surface_height_mm)
  const zMin = nullableNumber(value.z_minimum_mm)
  const zMax = nullableNumber(value.z_maximum_mm)
  const vertices = nullableNumber(value.estimated_vertices)
  const triangles = nullableNumber(value.estimated_triangles)
  const stlBytes = nullableNumber(value.estimated_stl_bytes)
  const peakBytes = nullableNumber(value.estimated_peak_memory_bytes)
  if (
    stockThickness === undefined ||
    baseThickness === undefined ||
    peak === undefined ||
    zMin === undefined ||
    zMax === undefined ||
    vertices === undefined ||
    triangles === undefined ||
    stlBytes === undefined ||
    peakBytes === undefined
  ) {
    return null
  }
  if (!isRecord(value.placement) || typeof value.placement.recorded !== "boolean") return null
  if (typeof value.placement.applied !== "boolean") return null
  const centerX = nullableNumber(value.placement.center_x_mm)
  const centerY = nullableNumber(value.placement.center_y_mm)
  if (centerX === undefined || centerY === undefined) return null
  const origin = value.placement.origin
  if (origin !== null && typeof origin !== "string") return null
  let spacing: ExportReadiness["spacing_mm"] = null
  if (value.spacing_mm !== null) {
    if (!isRecord(value.spacing_mm) || typeof value.spacing_mm.uniform !== "boolean") return null
    const x = finiteNumber(value.spacing_mm.x_mm)
    const y = finiteNumber(value.spacing_mm.y_mm)
    if (x === null || y === null) return null
    spacing = { x_mm: x, y_mm: y, uniform: value.spacing_mm.uniform }
  }
  const blocking = issues(value.blocking_errors)
  const warnings = issues(value.warnings)
  const actions = strings(value.corrective_actions)
  if (!blocking || !warnings || !actions) return null
  if (!isRecord(value.limits)) return null
  const maxTriangles = finiteNumber(value.limits.max_triangles)
  const maxBytes = finiteNumber(value.limits.max_stl_bytes)
  const maxMemory = finiteNumber(value.limits.max_peak_memory_bytes)
  if (maxTriangles === null || maxBytes === null || maxMemory === null) return null
  return {
    project_id: value.project_id,
    provenance_status: status as ArtifactStatus,
    export_target: value.export_target,
    export_certificate: value.export_certificate,
    coordinate_convention: value.coordinate_convention,
    resolution_policy: value.resolution_policy,
    topology_note: value.topology_note,
    mode,
    source_grid: source,
    planned_grid: planned,
    spacing_mm: spacing,
    artwork_width_mm: artworkWidth,
    artwork_height_mm: artworkHeight,
    stock_width_mm: stockWidth,
    stock_height_mm: stockHeight,
    stock_thickness_mm: stockThickness,
    base_thickness_mm: baseThickness,
    maximum_surface_height_mm: peak,
    placement: {
      recorded: value.placement.recorded,
      applied: value.placement.applied,
      center_x_mm: centerX,
      center_y_mm: centerY,
      origin,
    },
    z_minimum_mm: zMin,
    z_maximum_mm: zMax,
    estimated_vertices: vertices,
    estimated_triangles: triangles,
    estimated_stl_bytes: stlBytes,
    estimated_peak_memory_bytes: peakBytes,
    topology_expected: value.topology_expected,
    topology_validated: value.topology_validated,
    self_intersection_checked: value.self_intersection_checked,
    zero_area_measured: value.zero_area_measured,
    blocking_errors: blocking,
    warnings,
    corrective_actions: actions,
    ready_for_export: value.ready_for_export,
    limits: {
      max_triangles: maxTriangles,
      max_stl_bytes: maxBytes,
      max_peak_memory_bytes: maxMemory,
    },
  }
}

export function selectionKey(selection: ExportSelection): string {
  if (selection.mode === "native") return "native"
  return `reduced:${selection.heightPx}x${selection.widthPx}`
}

export function exportReadinessPath(projectId: string, selection: ExportSelection): string {
  const params = new URLSearchParams()
  params.set("mode", selection.mode)
  if (selection.mode === "reduced") {
    params.set("height_px", String(selection.heightPx))
    params.set("width_px", String(selection.widthPx))
  }
  return `/api/projects/${encodeURIComponent(projectId)}/relief/export/readiness?${params.toString()}`
}

export function stlExportPath(projectId: string): string {
  return `/api/projects/${encodeURIComponent(projectId)}/relief/export/stl`
}

export function stlExportBody(
  selection: ExportSelection,
): { mode: "native" } | { mode: "reduced"; height_px: number; width_px: number } {
  if (selection.mode === "native") return { mode: "native" }
  return { mode: "reduced", height_px: selection.heightPx, width_px: selection.widthPx }
}

export function parseGridInput(value: string): number | null {
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const parsed = Number(trimmed)
  if (!Number.isSafeInteger(parsed)) return null
  return parsed
}

export function distinctEndpointIndices(length: number, count: number): boolean {
  if (!Number.isInteger(length) || !Number.isInteger(count)) return false
  if (length < 2 || count < 2 || count > length) return false
  if (count === length) return true
  const indices: number[] = []
  for (let index = 0; index < count; index += 1) {
    indices.push(Math.floor((index * (length - 1)) / (count - 1)))
  }
  if (indices[0] !== 0 || indices[indices.length - 1] !== length - 1) return false
  return new Set(indices).size === count
}

export function reducedGridError(
  source: { heightPx: number; widthPx: number } | null,
  heightPx: number,
  widthPx: number,
): string | null {
  if (!Number.isInteger(heightPx) || !Number.isInteger(widthPx) || heightPx < 2 || widthPx < 2) {
    return "Reduced export requires whole-number width and height of at least 2."
  }
  if (!source) return null
  if (heightPx > source.heightPx || widthPx > source.widthPx) {
    return "Reduced export cannot exceed the saved physical-height dimensions."
  }
  if (heightPx * widthPx >= source.heightPx * source.widthPx) {
    return "Reduced export must be smaller on at least one axis, and neither axis may be larger."
  }
  if (!distinctEndpointIndices(source.heightPx, heightPx) || !distinctEndpointIndices(source.widthPx, widthPx)) {
    return "The requested export grid does not select distinct source samples."
  }
  return null
}

export function suggestReducedGrid(source: {
  heightPx: number
  widthPx: number
}): { heightPx: number; widthPx: number } | null {
  const candidates = [
    {
      heightPx: Math.max(2, Math.floor(source.heightPx / 2)),
      widthPx: Math.max(2, Math.floor(source.widthPx / 2)),
    },
    { heightPx: Math.max(2, source.heightPx - 1), widthPx: source.widthPx },
    { heightPx: source.heightPx, widthPx: Math.max(2, source.widthPx - 1) },
  ]
  for (const candidate of candidates) {
    if (reducedGridError(source, candidate.heightPx, candidate.widthPx) == null) return candidate
  }
  return null
}

export function readinessMatches(
  report: ExportReadiness,
  projectId: string,
  selection: ExportSelection,
): boolean {
  if (report.project_id !== projectId || report.mode !== selection.mode) return false
  if (selection.mode === "native") return true
  return (
    report.planned_grid?.height_px === selection.heightPx &&
    report.planned_grid?.width_px === selection.widthPx
  )
}

export function resolutionDraftDirty(
  draftMode: ExportMode,
  committed: ExportSelection,
  heightText: string,
  widthText: string,
): boolean {
  if (draftMode === "native") return committed.mode !== "native"
  if (committed.mode !== "reduced") return true
  return String(committed.heightPx) !== heightText.trim() || String(committed.widthPx) !== widthText.trim()
}

export function downloadAllowed(input: {
  report: ExportReadiness | null
  projectId: string
  selection: ExportSelection
  draftDirty: boolean
  exporting: boolean
  phase: "loading" | "ready" | "error"
}): boolean {
  if (input.exporting || input.draftDirty || input.phase !== "ready" || input.report == null) return false
  if (!readinessMatches(input.report, input.projectId, input.selection)) return false
  return input.report.ready_for_export
}

export function shouldApplyReadiness(requestId: number, latestId: number): boolean {
  return requestId === latestId
}

export function decideExportClick(exporting: boolean, allowed: boolean): "ignored" | "started" {
  if (!allowed || exporting) return "ignored"
  return "started"
}

export function afterExportFailure(token: number): { refresh: number; downloadEnabled: false } {
  return { refresh: token + 1, downloadEnabled: false }
}

export function showsMissingPhysical(report: ExportReadiness | null): boolean {
  if (!report) return false
  return report.blocking_errors.some(
    (issue) => issue.code === "physical_missing" || issue.message === MISSING_PHYSICAL_MESSAGE,
  )
}

export function issueMessages(issues: ExportIssue[]): string[] {
  return issues.map((issue) => issue.message)
}

export function sanitizeStlFilename(name: string): string {
  const trimmed = name.trim().replace(/^"|"$/g, "")
  if (
    trimmed.length === 0 ||
    trimmed.includes("/") ||
    trimmed.includes("\\") ||
    trimmed.includes("..") ||
    trimmed.includes("\0")
  ) {
    return "relief-export.stl"
  }
  return SAFE_STL_NAME.test(trimmed) ? trimmed : "relief-export.stl"
}

export function filenameFromContentDisposition(header: string | null): string {
  if (!header) return "relief-export.stl"
  const quoted = /filename="([^"]*)"/.exec(header)
  const plain = /filename=([^;]+)/.exec(header)
  return sanitizeStlFilename(quoted?.[1] ?? plain?.[1] ?? "")
}

export function formatCount(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "Unavailable"
  return Math.round(value)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ",")
}

export function formatByteSize(bytes: number | null): string {
  if (bytes == null || !Number.isFinite(bytes)) return "Unavailable"
  const exact = `${formatCount(bytes)} bytes`
  if (bytes < 1024) return exact
  const mib = bytes / (1024 * 1024)
  const rendered = mib >= 100 ? mib.toFixed(0) : mib.toFixed(1)
  return `${exact} (${rendered} MiB)`
}

export function formatMillimeters(value: number): string {
  return `${String(Number(value.toFixed(4)))} mm`
}

export function formatFamiliarMillimeters(value: number | null, unit: DisplayUnit): string {
  if (value == null || !Number.isFinite(value)) return "Unavailable"
  const millimeters = formatMillimeters(value)
  if (unit === "mm") return millimeters
  return `${formatLength(value, "in")} (${millimeters})`
}

export function formatGrid(gridSize: ExportGrid | null): string {
  if (!gridSize) return "Unavailable"
  return `${formatCount(gridSize.width_px)} × ${formatCount(gridSize.height_px)} px (width × height)`
}

export function placementExplanation(report: ExportReadiness, unit: DisplayUnit): string {
  if (
    !report.placement.recorded ||
    report.placement.center_x_mm == null ||
    report.placement.center_y_mm == null
  ) {
    return `${ARTWORK_LOCAL_NOTE} Stock-relative artwork placement is not recorded.`
  }
  const left = formatFamiliarMillimeters(report.placement.center_x_mm, unit)
  const front = formatFamiliarMillimeters(report.placement.center_y_mm, unit)
  return `${ARTWORK_LOCAL_NOTE} The saved artwork center is ${left} from the left and ${front} from the front of the stock. That center is not applied to the STL.`
}

export function geometryFacts(report: ExportReadiness, unit: DisplayUnit): Fact[] {
  const target = report.export_target === "artwork-local" ? "Artwork-local" : report.export_target
  return [
    { term: "Export target", detail: target },
    { term: "Source grid", detail: formatGrid(report.source_grid) },
    { term: "Selected grid", detail: formatGrid(report.planned_grid) },
    { term: "Artwork width", detail: formatFamiliarMillimeters(report.artwork_width_mm, unit) },
    { term: "Artwork height", detail: formatFamiliarMillimeters(report.artwork_height_mm, unit) },
    { term: "Stock width", detail: formatFamiliarMillimeters(report.stock_width_mm, unit) },
    { term: "Stock height", detail: formatFamiliarMillimeters(report.stock_height_mm, unit) },
    { term: "Stock thickness", detail: formatFamiliarMillimeters(report.stock_thickness_mm, unit) },
    { term: "Base thickness", detail: formatFamiliarMillimeters(report.base_thickness_mm, unit) },
    {
      term: "Maximum surface height",
      detail: formatFamiliarMillimeters(report.maximum_surface_height_mm, unit),
    },
    { term: "Measured minimum Z", detail: formatFamiliarMillimeters(report.z_minimum_mm, unit) },
    { term: "Measured maximum Z", detail: formatFamiliarMillimeters(report.z_maximum_mm, unit) },
    { term: "Estimated triangles", detail: formatCount(report.estimated_triangles) },
    { term: "Estimated STL size", detail: formatByteSize(report.estimated_stl_bytes) },
    { term: "Estimated peak memory", detail: formatByteSize(report.estimated_peak_memory_bytes) },
  ]
}

export function planValidationNotes(report: ExportReadiness): string[] {
  return [
    report.ready_for_export
      ? "Export readiness passed for this plan."
      : "Export readiness did not pass for this plan.",
    report.self_intersection_checked
      ? "Full self-intersection validation was run for this plan."
      : "Full self-intersection validation was not run for this plan.",
    report.topology_validated
      ? "Mesh topology was validated for this plan."
      : "Mesh topology was not validated for this plan.",
    report.zero_area_measured
      ? "Zero-area triangles were measured for this plan."
      : "Zero-area triangles were not measured for this plan.",
    report.export_certificate === "none" ? CERTIFICATE_NOTE : "Certificate status was not recognized.",
    PLAN_NOT_MESH_NOTE,
  ]
}

export function exportLayout(viewportWidth: number): {
  stacked: boolean
  inspectorMinPx: number | null
  inspectorMaxPx: number | null
} {
  if (viewportWidth <= 960) return { stacked: true, inspectorMinPx: null, inspectorMaxPx: null }
  return { stacked: false, inspectorMinPx: 340, inspectorMaxPx: 380 }
}

/** 80-byte header, uint32 count, and 50 bytes per triangle. */
export function stlLengthError(body: ArrayBuffer): string | null {
  if (body.byteLength < 84) return "Backend returned an incomplete STL."
  const count = new DataView(body).getUint32(80, true)
  const payload = 50 * count
  const expected = 84 + payload
  if (!Number.isSafeInteger(expected) || body.byteLength !== expected) {
    return "Backend returned an STL whose length does not match its triangle count."
  }
  return null
}

export async function interpretStlResponse(response: Response): Promise<StlDownloadResult> {
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }
  const type = response.headers.get("Content-Type") ?? ""
  if (type.includes("application/json")) {
    throw new ApiError("Backend returned an unreadable response.", response.status)
  }
  const body = await response.arrayBuffer()
  const problem = stlLengthError(body)
  if (problem) throw new ApiError(problem, response.status)
  return {
    body,
    filename: filenameFromContentDisposition(response.headers.get("Content-Disposition")),
    units: response.headers.get("X-Relief-Stl-Units"),
    certificate: response.headers.get("X-Relief-Export-Certificate"),
    selfIntersectionChecked: response.headers.get("X-Relief-Self-Intersection-Checked"),
    limitation: response.headers.get("X-Relief-Topology-Limitation"),
  }
}

export function exportDownloadNote(result: Pick<
  StlDownloadResult,
  "units" | "certificate" | "selfIntersectionChecked" | "limitation"
>): string {
  const units =
    result.units === "mm"
      ? "The file coordinates are millimeters."
      : "The response did not confirm STL units."
  const certificate =
    result.certificate === "none"
      ? "This file is not a machining certificate."
      : "The response did not confirm a certificate status."
  const intersection =
    result.selfIntersectionChecked === "true"
      ? "The server reported that self-intersection was checked for this file."
      : result.selfIntersectionChecked === "false"
        ? "The server reported that full self-intersection validation was not completed for this file."
        : "The response did not say whether self-intersection was checked."
  const limitation = result.limitation?.trim() ?? ""
  return [units, certificate, intersection, limitation].filter((part) => part.length > 0).join(" ")
}

export function saveBlobDownload(bytes: ArrayBuffer, filename: string, host: DownloadHost): string {
  const blob = new Blob([bytes], { type: "application/octet-stream" })
  const url = host.createObjectURL(blob)
  try {
    const anchor = host.createElement("a")
    anchor.href = url
    anchor.download = sanitizeStlFilename(filename)
    host.appendChild(anchor)
    anchor.click()
    anchor.remove()
  } finally {
    host.revokeObjectURL(url)
  }
  return url
}

export function browserDownloadHost(): DownloadHost {
  return {
    createObjectURL: (blob) => URL.createObjectURL(blob),
    revokeObjectURL: (url) => URL.revokeObjectURL(url),
    createElement: (tag) => document.createElement(tag) as HTMLAnchorElement,
    appendChild: (node) => {
      document.body.appendChild(node as unknown as Node)
    },
  }
}
