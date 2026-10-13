/**
 * RFMP v1 preview decoding and display math.
 *
 * Backend millimeters: X left to right, Y top to bottom, Z height upward.
 * Display uses a proper rotation, so triangle winding stays unchanged:
 * display X = backend X, display Y = backend Z, display Z = -backend Y.
 * RFMP normals are one face normal per triangle. They are not vertex attributes.
 */

import type { ArtifactStatus, Project } from "../types/project.ts"
import { overviewWorkflow } from "./overviewStatus.ts"
import { artifactStatus } from "./provenanceStatus.ts"
import { formatLength, type DisplayUnit } from "./units.ts"

export const DISPLAY_TRANSFORM =
  "display X = backend X; display Y = backend Z; display Z = -backend Y"

export const PREVIEW_ONLY_NOTICE =
  "Preview only — not validated for STL export or machining."
export const DOWNSAMPLE_WARNING = "Reduced-resolution previews can omit narrow peaks."
export const STALE_GEOMETRY_NOTE = "Historical geometry."
export const UNKNOWN_GEOMETRY_NOTE = "Unverified lineage."
export const MISSING_PHYSICAL_MESSAGE = "Physical height has not been generated."
export const WEBGL_UNAVAILABLE_MESSAGE = "WebGL is not available in this browser."
export const GPU_FAILURE_MESSAGE = "The 3D view could not be created."
export const SHADING_NOTE = "Shading is a display aid. It does not change physical height."
export const LARGE_PREVIEW_NOTE = "This preview is large and may be slow to draw."

export const DEFAULT_MAX_VERTICES = 50_000
export const SERVER_MAX_VERTICES = 250_000
export const HEADER_BYTES = 256
export const FACE_NORMALS_ARE_VERTEX_ATTRIBUTES = false

export const MESH_QUALITIES = [
  { id: "fast", label: "Fast", maxVertices: 20_000 },
  { id: "standard", label: "Standard", maxVertices: 50_000 },
  { id: "detailed", label: "Detailed", maxVertices: 150_000 },
  { id: "maximum", label: "Maximum", maxVertices: 250_000 },
] as const

export type MeshQualityId = (typeof MESH_QUALITIES)[number]["id"]
export type MeshProvenance = "unknown" | "current" | "stale"
export type TopologyLevel = 1 | 2
export type CameraViewName = "perspective" | "top" | "front" | "side" | "isometric"
export type Vec3 = [number, number, number]

export type DecodedMeshPreview = {
  version: 1
  vertexCount: number
  triangleCount: number
  positions: Float32Array
  indices: Uint32Array
  faceNormals: Float32Array
  status: MeshProvenance
  sourceWidthPx: number
  sourceHeightPx: number
  meshWidthPx: number
  meshHeightPx: number
  artworkWidthMm: number
  artworkHeightMm: number
  boundingBox: { min: Vec3; max: Vec3 }
  downsamplingRequired: boolean
  selfIntersectionChecked: boolean
  topologyPassed: true
  topologyLevel: TopologyLevel
  exportReady: false
  maxVertices: number
  normalScope: 1
  convention: string
  normalKind: "face"
}

export type MeshView =
  | { kind: "absent" }
  | { kind: "webgl"; message: string }
  | { kind: "loading" }
  | { kind: "not-found"; message: string }
  | { kind: "invalid"; message: string }
  | { kind: "network"; message: string }
  | { kind: "malformed"; message: string }
  | { kind: "gpu"; message: string }
  | { kind: "failed"; message: string }
  | { kind: "ready" }

export type CameraPose = {
  view: CameraViewName
  position: Vec3
  target: Vec3
  up: Vec3
  near: number
  far: number
  fov: number
}

export class MeshPreviewDecodeError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "MeshPreviewDecodeError"
  }
}

const STATUS_BY_CODE: Record<number, MeshProvenance> = {
  1: "unknown",
  2: "current",
  3: "stale",
}

export function meshQuality(id: MeshQualityId) {
  const match = MESH_QUALITIES.find((item) => item.id === id)
  if (!match) return MESH_QUALITIES[1]
  return match
}

export function meshPreviewPath(projectId: string, maxVertices: number): string {
  const params = new URLSearchParams()
  params.set("max_vertices", String(maxVertices))
  return `/api/projects/${encodeURIComponent(projectId)}/relief/mesh/preview?${params.toString()}`
}

export function meshRequestKey(projectId: string, maxVertices: number): string {
  return `${projectId}:${maxVertices}`
}

export type MeshPhysicalIdentity = {
  generated_at?: string | null
  status?: string | null
  source_relief_generated_at?: string | null
  source_relief_sha256?: string | null
  content_sha256?: string | null
  settings_sha256?: string | null
} | null | undefined

export type MeshPreviewRequestIdentity = {
  projectId: string
  maxVertices: number
  available: boolean
  generatedAt: string
  physicalStatus: ArtifactStatus
  sourceReliefGeneratedAt: string
  sourceReliefSha256: string
  contentSha256: string
  settingsSha256: string
}

export const MESH_PROVENANCE_CHAIN = [
  { id: "source", title: "Source" },
  { id: "raw-depth", title: "Raw depth", stage: "raw" },
  { id: "conditioned", title: "Conditioned depth", stage: "conditioned" },
  { id: "composed-relief", title: "Composed relief", stage: "relief" },
  { id: "physical-height", title: "Physical height", stage: "physical" },
] as const

function identityText(value: string | null | undefined): string {
  return value ?? ""
}

export function meshPreviewRequestIdentity(input: {
  projectId: string
  maxVertices: number
  physical: MeshPhysicalIdentity
}): MeshPreviewRequestIdentity {
  const physical = input.physical ?? null
  const generatedAt = identityText(physical?.generated_at)
  return {
    projectId: input.projectId,
    maxVertices: input.maxVertices,
    available: physical != null && generatedAt.length > 0,
    generatedAt,
    physicalStatus: artifactStatus(physical, physical?.status),
    sourceReliefGeneratedAt: identityText(physical?.source_relief_generated_at),
    sourceReliefSha256: identityText(physical?.source_relief_sha256),
    contentSha256: identityText(physical?.content_sha256),
    settingsSha256: identityText(physical?.settings_sha256),
  }
}

/** Generation identity. Physical status is recorded separately and does not refetch. */
export function meshPreviewRequestKey(identity: MeshPreviewRequestIdentity): string | null {
  if (!identity.available) return null
  return [
    identity.projectId,
    String(identity.maxVertices),
    identity.generatedAt,
    identity.sourceReliefGeneratedAt,
    identity.sourceReliefSha256,
    identity.contentSha256,
    identity.settingsSha256,
  ].join("\u001f")
}

export function meshPreviewFetchToken(requestKey: string | null, retry: number): string | null {
  if (requestKey == null) return null
  return `${requestKey}\u001f${retry}`
}

export function meshPreviewShouldFetch(input: {
  requestKey: string | null
  webgl: "available" | "unavailable"
}): boolean {
  return input.requestKey != null && input.webgl === "available"
}

export function meshPreviewGenerationChange(
  previousToken: string | null,
  nextToken: string | null,
): { changed: boolean; clearGeometry: boolean; phase: "idle" | "loading" } {
  if (previousToken === nextToken) {
    return { changed: false, clearGeometry: false, phase: nextToken == null ? "idle" : "loading" }
  }
  return {
    changed: true,
    clearGeometry: true,
    phase: nextToken == null ? "idle" : "loading",
  }
}

export function shouldApplyMeshResult(input: {
  requestId: number
  currentId: number
  projectId: string
  requestedProjectId: string
  requestedKey: string
  activeKey: string | null
  aborted?: boolean
}): boolean {
  if (input.aborted || input.activeKey == null) return false
  return (
    input.requestId === input.currentId &&
    input.projectId === input.requestedProjectId &&
    input.requestedKey === input.activeKey
  )
}

export function meshProvenanceChain(project: Pick<Project, "source" | "depth" | "relief">): {
  id: (typeof MESH_PROVENANCE_CHAIN)[number]["id"]
  title: string
  stage?: "raw" | "conditioned" | "relief" | "physical"
  status: ArtifactStatus
}[] {
  const workflow = overviewWorkflow(project)
  return MESH_PROVENANCE_CHAIN.map((item) => {
    const match = workflow.find((entry) => entry.id === item.id)
    return {
      id: item.id,
      title: item.title,
      stage: "stage" in item ? item.stage : undefined,
      status: match?.state ?? "missing",
    }
  })
}

export function previewVerifiedCurrent(
  payloadStatus: MeshProvenance | null,
  physicalStatus: ArtifactStatus,
): boolean {
  return payloadStatus === "current" && physicalStatus === "current"
}

export function previewGeometryCaution(
  payloadStatus: MeshProvenance | null,
  physicalStatus: ArtifactStatus,
): string | null {
  if (payloadStatus == null || physicalStatus === "missing") return null
  if (previewVerifiedCurrent(payloadStatus, physicalStatus)) return null
  if (payloadStatus === "stale" || physicalStatus === "stale") return STALE_GEOMETRY_NOTE
  return UNKNOWN_GEOMETRY_NOTE
}

export function previewAriaStatus(
  payloadStatus: MeshProvenance,
  physicalStatus: ArtifactStatus,
): string {
  if (previewVerifiedCurrent(payloadStatus, physicalStatus)) return "Current"
  if (payloadStatus === "stale" || physicalStatus === "stale") return "Stale"
  return "Unknown"
}

export function meshPreviewView(input: {
  hasRecord: boolean
  webgl: "available" | "unavailable"
  phase: "idle" | "loading" | "ready" | "error"
  errorStatus: number | null
  errorKind: "http" | "malformed" | "gpu" | null
  errorMessage: string | null
}): MeshView {
  if (!input.hasRecord) return { kind: "absent" }
  if (input.webgl === "unavailable") return { kind: "webgl", message: WEBGL_UNAVAILABLE_MESSAGE }
  if (input.phase === "idle" || input.phase === "loading") return { kind: "loading" }
  if (input.phase === "ready") return { kind: "ready" }
  const message = input.errorMessage ?? "The 3D preview could not be loaded."
  if (input.errorKind === "malformed") return { kind: "malformed", message }
  if (input.errorKind === "gpu") return { kind: "gpu", message }
  if (input.errorStatus == null) return { kind: "network", message }
  if (input.errorStatus === 404) return { kind: "not-found", message }
  if (input.errorStatus === 422) return { kind: "invalid", message }
  return { kind: "failed", message }
}

export function retryAllowed(view: MeshView): boolean {
  return (
    view.kind === "webgl" ||
    view.kind === "not-found" ||
    view.kind === "invalid" ||
    view.kind === "network" ||
    view.kind === "malformed" ||
    view.kind === "gpu" ||
    view.kind === "failed"
  )
}

export function provenanceLabel(status: MeshProvenance): string {
  if (status === "current") return "Current"
  if (status === "stale") return "Stale"
  return "Unknown"
}

export function provenanceNote(status: MeshProvenance): string | null {
  if (status === "stale") return STALE_GEOMETRY_NOTE
  if (status === "unknown") return UNKNOWN_GEOMETRY_NOTE
  return null
}

export function topologyExplanation(level: TopologyLevel): string {
  if (level === 2) {
    return "Bounded intersection checks ran for this small mesh. Comprehensive self-intersection checks are not claimed."
  }
  return "Edge-manifold checks ran. Comprehensive self-intersection checks have not been run on this mesh."
}

export function topologyLabel(level: TopologyLevel): string {
  return level === 2 ? "Bounded intersection" : "Edge manifold"
}

export function largePreviewWarning(triangleCount: number): string | null {
  if (triangleCount > 400_000) return LARGE_PREVIEW_NOTE
  return null
}

export function formatMillimeters(value: number): string {
  if (!Number.isFinite(value)) return "—"
  return `${value.toFixed(3)} mm`
}

/** Display only. Canonical millimeters stay on the decoded preview. */
export function formatMeshDimension(millimeters: number, unit: DisplayUnit): string {
  if (!Number.isFinite(millimeters)) return "—"
  if (unit === "in") return formatLength(millimeters, "in")
  // Four decimal places removes float32 header residue and still shows 258.839 mm.
  return `${Number(millimeters.toFixed(4))} mm`
}

export function formatMeshRange(min: number, max: number, unit: DisplayUnit): string {
  return `${formatMeshDimension(min, unit)} to ${formatMeshDimension(max, unit)}`
}

export const DEFAULT_PREVIEW_LIGHTING = {
  ambient: 0.58,
  key: 1.15,
  fill: 0.28,
} as const

export const PREVIEW_LIGHT_LIMITS = {
  ambient: { min: 0, max: 2, step: 0.01 },
  key: { min: 0, max: 3, step: 0.01 },
  fill: { min: 0, max: 2, step: 0.01 },
} as const

export type PreviewLightChannel = keyof typeof DEFAULT_PREVIEW_LIGHTING

export type PreviewLighting = Record<PreviewLightChannel, number>

export function defaultPreviewLighting(): PreviewLighting {
  return {
    ambient: DEFAULT_PREVIEW_LIGHTING.ambient,
    key: DEFAULT_PREVIEW_LIGHTING.key,
    fill: DEFAULT_PREVIEW_LIGHTING.fill,
  }
}

export function previewLightValue(channel: PreviewLightChannel, value: number): number {
  const limits = PREVIEW_LIGHT_LIMITS[channel]
  if (!Number.isFinite(value)) return DEFAULT_PREVIEW_LIGHTING[channel]
  const clamped = Math.min(limits.max, Math.max(limits.min, value))
  return Number(clamped.toFixed(2))
}

export function formatPreviewLight(value: number): string {
  return value.toFixed(2)
}

/** Display-only drop below the stock bottom. It is not a saved or exported height. */
export const BOUNDARY_GUIDE_OFFSET_MM = 0.2

export const STOCK_BOUNDARY_UNAVAILABLE =
  "Stock boundary is unavailable. Saved stock and artwork placement are incomplete."

export const STOCK_BOUNDARY_MISMATCH =
  "Stock boundary is unavailable. Saved artwork size does not match this preview."

export type ArtworkLocalRect = {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export type StockBoundaryGuide =
  | {
      available: true
      leftMm: number
      rightMm: number
      frontMm: number
      backMm: number
      bounds: ArtworkLocalRect
    }
  | { available: false; reason: string }

export function artworkLocalRect(widthMm: number, heightMm: number): ArtworkLocalRect | null {
  if (!Number.isFinite(widthMm) || !Number.isFinite(heightMm) || !(widthMm > 0) || !(heightMm > 0)) return null
  return { xMin: 0, xMax: widthMm, yMin: 0, yMax: heightMm }
}

export function stockBoundaryGuide(input: {
  stockWidthMm: number
  stockHeightMm: number
  artworkWidthMm: number
  artworkHeightMm: number
  centerXmm: number | null
  centerYmm: number | null
  meshArtworkWidthMm: number
  meshArtworkHeightMm: number
}): StockBoundaryGuide {
  const numbers = [
    input.stockWidthMm,
    input.stockHeightMm,
    input.artworkWidthMm,
    input.artworkHeightMm,
    input.centerXmm,
    input.centerYmm,
    input.meshArtworkWidthMm,
    input.meshArtworkHeightMm,
  ]
  if (numbers.some((value) => value == null || !Number.isFinite(value))) {
    return { available: false, reason: STOCK_BOUNDARY_UNAVAILABLE }
  }
  if (
    !(input.stockWidthMm > 0) ||
    !(input.stockHeightMm > 0) ||
    !(input.artworkWidthMm > 0) ||
    !(input.artworkHeightMm > 0)
  ) {
    return { available: false, reason: STOCK_BOUNDARY_UNAVAILABLE }
  }
  if (
    Math.abs(input.artworkWidthMm - input.meshArtworkWidthMm) > 0.01 ||
    Math.abs(input.artworkHeightMm - input.meshArtworkHeightMm) > 0.01
  ) {
    return { available: false, reason: STOCK_BOUNDARY_MISMATCH }
  }
  const centerXmm = input.centerXmm
  const centerYmm = input.centerYmm
  if (centerXmm == null || centerYmm == null) {
    return { available: false, reason: STOCK_BOUNDARY_UNAVAILABLE }
  }
  const leftMm = centerXmm - input.artworkWidthMm / 2
  const rightMm = input.stockWidthMm - centerXmm - input.artworkWidthMm / 2
  const frontMm = centerYmm - input.artworkHeightMm / 2
  const backMm = input.stockHeightMm - centerYmm - input.artworkHeightMm / 2
  return {
    available: true,
    leftMm,
    rightMm,
    frontMm,
    backMm,
    bounds: {
      xMin: -leftMm,
      xMax: input.artworkWidthMm + rightMm,
      yMin: -frontMm,
      yMax: input.artworkHeightMm + backMm,
    },
  }
}

export function displayBoundaryCorners(rect: ArtworkLocalRect): Vec3[] {
  const elevation = -BOUNDARY_GUIDE_OFFSET_MM
  return [
    toDisplayPoint(rect.xMin, rect.yMin, elevation),
    toDisplayPoint(rect.xMax, rect.yMin, elevation),
    toDisplayPoint(rect.xMax, rect.yMax, elevation),
    toDisplayPoint(rect.xMin, rect.yMax, elevation),
  ]
}

export function webglStatus(probe?: () => boolean): "available" | "unavailable" {
  try {
    if (probe) return probe() ? "available" : "unavailable"
    if (typeof document === "undefined") return "unavailable"
    const canvas = document.createElement("canvas")
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl")
    return gl ? "available" : "unavailable"
  } catch {
    return "unavailable"
  }
}

export function meshLayout(viewportWidth: number): {
  stacked: boolean
  inspectorMinPx: number | null
  inspectorMaxPx: number | null
} {
  if (viewportWidth <= 960) return { stacked: true, inspectorMinPx: null, inspectorMaxPx: null }
  return { stacked: false, inspectorMinPx: 340, inspectorMaxPx: 380 }
}

export function solidTriangleCount(meshHeight: number, meshWidth: number): number {
  return (
    4 * (meshHeight - 1) * (meshWidth - 1) + 4 * (meshWidth - 1) + 4 * (meshHeight - 1)
  )
}

export function vertexAttributePlan(): {
  position: "display-xyz"
  index: "rfmp-indices"
  normal: "computed-display-vertex-normals"
  faceNormalBinding: "rejected"
} {
  return {
    position: "display-xyz",
    index: "rfmp-indices",
    normal: "computed-display-vertex-normals",
    faceNormalBinding: "rejected",
  }
}

export function decodeRfmp(input: ArrayBuffer | ArrayBufferView): DecodedMeshPreview {
  const view = ArrayBuffer.isView(input)
    ? new DataView(input.buffer, input.byteOffset, input.byteLength)
    : new DataView(input)
  if (view.byteLength < HEADER_BYTES) {
    throw new MeshPreviewDecodeError("Mesh preview payload is shorter than the 256 byte header.")
  }
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  )
  if (magic !== "RFMP") throw new MeshPreviewDecodeError("Mesh preview magic is not RFMP.")
  if (readU32(view, 4) !== 1) throw new MeshPreviewDecodeError("Mesh preview version is not 1.")
  if (
    view.getUint8(8) !== 1 ||
    view.getUint8(9) !== 0 ||
    view.getUint8(10) !== 0 ||
    view.getUint8(11) !== 0 ||
    readU32(view, 8) !== 1
  ) {
    throw new MeshPreviewDecodeError("Mesh preview endian marker is not little-endian 1.")
  }
  if (readU32(view, 12) !== HEADER_BYTES) {
    throw new MeshPreviewDecodeError("Mesh preview header length is not 256.")
  }
  const payloadBytes = readU32(view, 116)
  if (payloadBytes !== view.byteLength) {
    throw new MeshPreviewDecodeError("Mesh preview payload has unexpected trailing or truncated bytes.")
  }
  for (let index = 0; index < 24; index += 1) {
    if (view.getUint8(232 + index) !== 0) {
      throw new MeshPreviewDecodeError("Mesh preview reserved header bytes are not zero.")
    }
  }
  if (readU32(view, 24) !== 3 || readU32(view, 28) !== 3 || readU32(view, 32) !== 3) {
    throw new MeshPreviewDecodeError("Mesh preview component counts are not 3.")
  }
  const vertexCount = readU32(view, 16)
  const triangleCount = readU32(view, 20)
  const positionsOffset = readU32(view, 92)
  const positionsBytes = readU32(view, 96)
  const indicesOffset = readU32(view, 100)
  const indicesBytes = readU32(view, 104)
  const normalsOffset = readU32(view, 108)
  const normalsBytes = readU32(view, 112)
  const expectedPositions = vertexCount * 12
  const expectedIndices = triangleCount * 12
  const expectedNormals = triangleCount * 12
  if (
    vertexCount < 4 ||
    triangleCount < 4 ||
    positionsOffset !== HEADER_BYTES ||
    positionsBytes !== expectedPositions ||
    indicesOffset !== positionsOffset + positionsBytes ||
    indicesBytes !== expectedIndices ||
    normalsOffset !== indicesOffset + indicesBytes ||
    normalsBytes !== expectedNormals ||
    positionsOffset + positionsBytes + indicesBytes + normalsBytes !== payloadBytes
  ) {
    throw new MeshPreviewDecodeError("Mesh preview section offsets or counts do not match the payload.")
  }
  if (
    positionsOffset % 4 !== 0 ||
    indicesOffset % 4 !== 0 ||
    normalsOffset % 4 !== 0 ||
    (view.byteOffset + positionsOffset) % 4 !== 0
  ) {
    throw new MeshPreviewDecodeError("Mesh preview arrays are not 4-byte aligned.")
  }
  const exportReady = readU32(view, 124)
  if (exportReady !== 0) {
    throw new MeshPreviewDecodeError("Mesh preview export_ready must remain false.")
  }
  const topologyLevel = readU32(view, 120)
  if (topologyLevel !== 1 && topologyLevel !== 2) {
    throw new MeshPreviewDecodeError("Mesh preview topology level is not 1 or 2.")
  }
  const status = STATUS_BY_CODE[readU32(view, 40)]
  if (!status) throw new MeshPreviewDecodeError("Mesh preview provenance status is not known.")
  const flags = readU32(view, 36)
  if ((flags & ~0b111) !== 0) throw new MeshPreviewDecodeError("Mesh preview flags are not recognized.")
  const downsamplingRequired = (flags & 1) !== 0
  const selfIntersectionChecked = (flags & 2) !== 0
  const topologyPassed = (flags & 4) !== 0
  if (!topologyPassed) throw new MeshPreviewDecodeError("Mesh preview topology checks did not pass.")
  if (topologyLevel === 2 && !selfIntersectionChecked) {
    throw new MeshPreviewDecodeError("Topology level does not match the intersection flag.")
  }
  if (topologyLevel === 1 && selfIntersectionChecked) {
    throw new MeshPreviewDecodeError("Topology level does not match the intersection flag.")
  }
  if (readU32(view, 132) !== 1) {
    throw new MeshPreviewDecodeError("Mesh preview normals are not face normals.")
  }
  const convention = readToken(view, 136, 64)
  const normalKind = readToken(view, 200, 32)
  if (convention !== "artwork-xy-y-down-z-up" || normalKind !== "face") {
    throw new MeshPreviewDecodeError("Mesh preview coordinate or normal token is not recognized.")
  }
  const sourceWidthPx = readU32(view, 44)
  const sourceHeightPx = readU32(view, 48)
  const meshWidthPx = readU32(view, 52)
  const meshHeightPx = readU32(view, 56)
  const maxVertices = readU32(view, 128)
  if (
    meshWidthPx < 2 ||
    meshHeightPx < 2 ||
    sourceWidthPx < meshWidthPx ||
    sourceHeightPx < meshHeightPx ||
    vertexCount !== 2 * meshWidthPx * meshHeightPx ||
    triangleCount !== solidTriangleCount(meshHeightPx, meshWidthPx) ||
    maxVertices < 4 ||
    maxVertices > SERVER_MAX_VERTICES
  ) {
    throw new MeshPreviewDecodeError("Mesh preview dimensions are inconsistent.")
  }
  const artworkWidthMm = readF32(view, 60)
  const artworkHeightMm = readF32(view, 64)
  const headerMin: Vec3 = [readF32(view, 68), readF32(view, 72), readF32(view, 76)]
  const headerMax: Vec3 = [readF32(view, 80), readF32(view, 84), readF32(view, 88)]
  if (
    !headerMin.every(Number.isFinite) ||
    !headerMax.every(Number.isFinite) ||
    !(artworkWidthMm > 0) ||
    !(artworkHeightMm > 0) ||
    !(headerMin[0] <= headerMax[0]) ||
    !(headerMin[1] <= headerMax[1]) ||
    !(headerMin[2] < headerMax[2]) ||
    Math.abs(headerMin[0]) > 1e-2 ||
    Math.abs(headerMin[1]) > 1e-2 ||
    Math.abs(headerMin[2]) > 1e-2 ||
    Math.abs(headerMax[0] - artworkWidthMm) > 1e-2 ||
    Math.abs(headerMax[1] - artworkHeightMm) > 1e-2
  ) {
    throw new MeshPreviewDecodeError("Mesh preview bounding box is not valid.")
  }
  let positions: Float32Array
  let indices: Uint32Array
  let faceNormals: Float32Array
  try {
    positions = new Float32Array(view.buffer, view.byteOffset + positionsOffset, vertexCount * 3)
    indices = new Uint32Array(view.buffer, view.byteOffset + indicesOffset, triangleCount * 3)
    faceNormals = new Float32Array(view.buffer, view.byteOffset + normalsOffset, triangleCount * 3)
  } catch {
    throw new MeshPreviewDecodeError("Mesh preview arrays could not be viewed.")
  }
  if (!allFinite(positions) || !allFinite(faceNormals)) {
    throw new MeshPreviewDecodeError("Mesh preview contains non-finite geometry.")
  }
  for (let index = 0; index < indices.length; index += 1) {
    if (indices[index] >= vertexCount) {
      throw new MeshPreviewDecodeError("Mesh preview index is outside the vertex range.")
    }
  }
  const actual = boundsOfPositions(positions)
  if (
    !actual ||
    !closeVec(actual.min, headerMin) ||
    !closeVec(actual.max, headerMax)
  ) {
    throw new MeshPreviewDecodeError("Mesh preview positions do not match the bounding box.")
  }
  if (faceNormals.length !== triangleCount * 3) {
    throw new MeshPreviewDecodeError("Mesh preview face normals do not match the triangle count.")
  }
  return {
    version: 1,
    vertexCount,
    triangleCount,
    positions,
    indices,
    faceNormals,
    status,
    sourceWidthPx,
    sourceHeightPx,
    meshWidthPx,
    meshHeightPx,
    artworkWidthMm,
    artworkHeightMm,
    boundingBox: { min: headerMin, max: headerMax },
    downsamplingRequired,
    selfIntersectionChecked,
    topologyPassed: true,
    topologyLevel,
    exportReady: false,
    maxVertices,
    normalScope: 1,
    convention,
    normalKind: "face",
  }
}

export function toDisplayPoint(x: number, y: number, z: number): Vec3 {
  return [x, z, -y]
}

export function toDisplayPositions(backend: Float32Array): Float32Array {
  const display = new Float32Array(backend.length)
  for (let index = 0; index < backend.length; index += 3) {
    const point = toDisplayPoint(backend[index], backend[index + 1], backend[index + 2])
    display[index] = point[0]
    display[index + 1] = point[1]
    display[index + 2] = point[2]
  }
  return display
}

export function displayHandedness(): number {
  const x = toDisplayPoint(1, 0, 0)
  const y = toDisplayPoint(0, 1, 0)
  const z = toDisplayPoint(0, 0, 1)
  const cross: Vec3 = [
    y[1] * z[2] - y[2] * z[1],
    y[2] * z[0] - y[0] * z[2],
    y[0] * z[1] - y[1] * z[0],
  ]
  return x[0] * cross[0] + x[1] * cross[1] + x[2] * cross[2]
}

export type HeldReliefGeometry<T> = {
  positions: Float32Array
  indices: Uint32Array
  geometry: T
}

export type HeldReliefMaterial<T> = {
  material: T
  flat: boolean
  wireframe: boolean
}

/** Reuse the geometry object while the decoded display buffers are the same objects. */
export function retainReliefGeometry<T>(
  held: HeldReliefGeometry<T> | null,
  positions: Float32Array,
  indices: Uint32Array,
  create: () => T,
): { hold: HeldReliefGeometry<T>; replaced: boolean } {
  if (held && held.positions === positions && held.indices === indices) {
    return { hold: held, replaced: false }
  }
  return { hold: { positions, indices, geometry: create() }, replaced: true }
}

/** Keep one material and update its presentation flags in place. */
export function retainReliefMaterial<T>(
  held: HeldReliefMaterial<T> | null,
  flat: boolean,
  wireframe: boolean,
  create: () => T,
  apply: (material: T, flat: boolean, wireframe: boolean) => void,
): { hold: HeldReliefMaterial<T>; replaced: boolean } {
  if (held) {
    if (held.flat !== flat || held.wireframe !== wireframe) {
      apply(held.material, flat, wireframe)
      held.flat = flat
      held.wireframe = wireframe
    }
    return { hold: held, replaced: false }
  }
  const material = create()
  apply(material, flat, wireframe)
  return { hold: { material, flat, wireframe }, replaced: true }
}

/** Flat shading displays the triangle normal. Smooth shading displays the vertex normal. */
export function reliefShadingSample(
  shading: "smooth" | "flat",
  vertexNormal: Vec3,
  faceNormal: Vec3,
): Vec3 {
  return shading === "flat" ? faceNormal : vertexNormal
}

export function displayVertexNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length)
  for (let index = 0; index < indices.length; index += 3) {
    const a = indices[index] * 3
    const b = indices[index + 1] * 3
    const c = indices[index + 2] * 3
    const e1x = positions[b] - positions[a]
    const e1y = positions[b + 1] - positions[a + 1]
    const e1z = positions[b + 2] - positions[a + 2]
    const e2x = positions[c] - positions[a]
    const e2y = positions[c + 1] - positions[a + 1]
    const e2z = positions[c + 2] - positions[a + 2]
    const nx = e1y * e2z - e1z * e2y
    const ny = e1z * e2x - e1x * e2z
    const nz = e1x * e2y - e1y * e2x
    normals[a] += nx
    normals[a + 1] += ny
    normals[a + 2] += nz
    normals[b] += nx
    normals[b + 1] += ny
    normals[b + 2] += nz
    normals[c] += nx
    normals[c + 1] += ny
    normals[c + 2] += nz
  }
  for (let index = 0; index < normals.length; index += 3) {
    const length = Math.hypot(normals[index], normals[index + 1], normals[index + 2])
    if (length > 1e-12) {
      normals[index] /= length
      normals[index + 1] /= length
      normals[index + 2] /= length
    } else {
      normals[index] = 0
      normals[index + 1] = 1
      normals[index + 2] = 0
    }
  }
  return normals
}

export function frontFaceToward(normal: Vec3, centroid: Vec3, camera: Vec3): boolean {
  return (
    normal[0] * (camera[0] - centroid[0]) +
      normal[1] * (camera[1] - centroid[1]) +
      normal[2] * (camera[2] - centroid[2]) >
    0
  )
}

export function boundsOfPositions(positions: Float32Array): { min: Vec3; max: Vec3 } | null {
  if (positions.length < 3 || positions.length % 3 !== 0) return null
  const min: Vec3 = [Infinity, Infinity, Infinity]
  const max: Vec3 = [-Infinity, -Infinity, -Infinity]
  for (let index = 0; index < positions.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = positions[index + axis]
      if (!Number.isFinite(value)) return null
      min[axis] = Math.min(min[axis], value)
      max[axis] = Math.max(max[axis], value)
    }
  }
  return { min, max }
}

export function fitCamera(
  bounds: { min: Vec3; max: Vec3 },
  aspect: number,
  view: CameraViewName,
): CameraPose {
  const safeAspect = Number.isFinite(aspect) && aspect > 0.05 ? aspect : 1
  const size: Vec3 = [
    Math.max(bounds.max[0] - bounds.min[0], 1e-4),
    Math.max(bounds.max[1] - bounds.min[1], 1e-4),
    Math.max(bounds.max[2] - bounds.min[2], 1e-4),
  ]
  const target: Vec3 = [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
  ]
  const radius = 0.5 * Math.hypot(size[0], size[1], size[2])
  const fov = 45
  const vertical = (fov * Math.PI) / 180
  const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * safeAspect)
  const half = Math.min(vertical, horizontal) / 2
  const distance = Math.max((radius / Math.sin(half)) * 1.2, radius + 1)
  const direction = cameraDirection(view)
  const position: Vec3 = [
    target[0] + direction[0] * distance,
    target[1] + direction[1] * distance,
    target[2] + direction[2] * distance,
  ]
  let near = Math.max(0.05, distance - radius * 1.8)
  let far = distance + radius * 3
  if (far <= near) far = near * 10
  if (far / near > 20_000) near = far / 20_000
  return { view, position, target, up: cameraUp(view), near, far, fov }
}

function cameraDirection(view: CameraViewName): Vec3 {
  if (view === "top") return [0, 1, 0]
  if (view === "front") return [0, 0, -1]
  if (view === "side") return [1, 0, 0]
  if (view === "isometric") {
    const length = Math.hypot(1, 1, 1)
    return [1 / length, 1 / length, -1 / length]
  }
  const length = Math.hypot(1, 0.85, 1)
  return [1 / length, 0.85 / length, 1 / length]
}

function cameraUp(view: CameraViewName): Vec3 {
  if (view === "top") return [0, 0, 1]
  return [0, 1, 0]
}

function readU32(view: DataView, offset: number): number {
  return view.getUint32(offset, true)
}

function readF32(view: DataView, offset: number): number {
  return view.getFloat32(offset, true)
}

function readToken(view: DataView, offset: number, size: number): string {
  let text = ""
  for (let index = 0; index < size; index += 1) {
    const byte = view.getUint8(offset + index)
    if (byte === 0) break
    text += String.fromCharCode(byte)
  }
  return text
}

function allFinite(values: Float32Array): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) return false
  }
  return true
}

function closeVec(left: Vec3, right: Vec3): boolean {
  return left.every((value, index) => Math.abs(value - right[index]) <= 1e-3)
}
