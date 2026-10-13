import {
  ApiError,
  apiDelete,
  apiGet,
  apiGetArrayBuffer,
  apiPostForm,
  apiPostJson,
  apiPostNoContent,
  apiPutJson,
  apiRequest,
  apiUrl,
  readErrorDetail,
} from "./client.ts"
import type {
  DepthConditioningParameters,
  DepthHistogram,
  Project,
  ProjectCreateRequest,
  PhysicalHeightDiagnostics,
  PhysicalHeightProfile,
  ReliefRecord,
  ReliefRegion,
  ReliefRole,
  SegmentationProposal,
} from "../types/project.ts"
import {
  physicalDiagnosticsPath,
  physicalProfilePath,
  type ProfileSelection,
} from "../utils/diagnosticsWorkspace.ts"
import {
  exportReadinessPath,
  interpretStlResponse,
  parseExportReadiness,
  stlExportBody,
  stlExportPath,
  type ExportReadiness,
  type ExportSelection,
  type StlDownloadResult,
} from "../utils/stlExport.ts"

export function listProjects(): Promise<Project[]> {
  return apiGet<Project[]>("/api/projects")
}

export function createProject(request: ProjectCreateRequest): Promise<Project> {
  return apiPostJson<Project>("/api/projects", request)
}

export function getProject(projectId: string): Promise<Project> {
  return apiGet<Project>(`/api/projects/${encodeURIComponent(projectId)}`)
}

export function updateProject(
  projectId: string,
  request: ProjectCreateRequest,
): Promise<Project> {
  return apiPutJson<Project>(`/api/projects/${encodeURIComponent(projectId)}`, request)
}

export function deleteProject(projectId: string): Promise<void> {
  return apiDelete(`/api/projects/${encodeURIComponent(projectId)}`)
}

export type ProjectDuplicateResult = {
  id: string
  slug: string
  name: string
  source_project_id: string
  duplicated: true
}

export function duplicateProject(projectId: string, name: string): Promise<ProjectDuplicateResult> {
  return apiPostJson<ProjectDuplicateResult>(
    `/api/projects/${encodeURIComponent(projectId)}/duplicate`,
    { name },
  )
}

export function uploadSourceImage(projectId: string, file: File): Promise<Project> {
  const form = new FormData()
  form.append("file", file)
  return apiPostForm<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/source`,
    form,
  )
}

export function getSourceImageUrl(projectId: string): string {
  return apiUrl(`/api/projects/${encodeURIComponent(projectId)}/source`)
}

export function generateDepth(projectId: string): Promise<Project> {
  return apiRequest<Project>(`/api/projects/${encodeURIComponent(projectId)}/depth`, {
    method: "POST",
  })
}

export function getDepthPreviewUrl(projectId: string, generatedAt: string): string {
  const version = encodeURIComponent(generatedAt)
  return apiUrl(
    `/api/projects/${encodeURIComponent(projectId)}/depth/preview?v=${version}`,
  )
}

export function applyConditioning(
  projectId: string,
  parameters: DepthConditioningParameters,
): Promise<Project> {
  return apiPostJson<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/depth/conditioning`,
    parameters,
  )
}

export function getConditionedPreviewUrl(projectId: string, generatedAt: string): string {
  const version = encodeURIComponent(generatedAt)
  return apiUrl(
    `/api/projects/${encodeURIComponent(projectId)}/depth/conditioning/preview?v=${version}`,
  )
}

export function getDepthHistogram(projectId: string): Promise<DepthHistogram> {
  return apiGet<DepthHistogram>(
    `/api/projects/${encodeURIComponent(projectId)}/depth/histogram`,
  )
}

export function predictSegmentation(
  projectId: string,
  body: {
    positive_points: Array<{ x: number; y: number }>
    negative_points: Array<{ x: number; y: number }>
    box?: { x0: number; y0: number; x1: number; y1: number } | null
    name?: string
    role?: ReliefRole | null
  },
): Promise<SegmentationProposal> {
  return apiPostJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/predict`,
    body,
  )
}

export function getSegmentationProposalUrl(
  projectId: string,
  proposalId: string,
  revision = 0,
): string {
  return apiUrl(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}?v=${revision}`,
  )
}

export type MaskStrokePayload = {
  mode: "add" | "erase"
  brush_radius_px: number
  points: Array<{ x: number; y: number }>
}

export type MagicWandPayload = {
  mode: "add" | "subtract"
  x: number
  y: number
  tolerance: number
  contiguous: boolean
}

export function applyMagicWand(
  projectId: string,
  proposalId: string,
  wand: MagicWandPayload,
): Promise<SegmentationProposal> {
  return apiPostJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}/magic-wand`,
    wand,
  )
}

export function applyMaskStroke(
  projectId: string,
  proposalId: string,
  stroke: MaskStrokePayload,
): Promise<SegmentationProposal> {
  return apiPostJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}/strokes`,
    stroke,
  )
}

export function updateProposalDraft(
  projectId: string,
  proposalId: string,
  draft: { name: string; role: ReliefRole },
): Promise<SegmentationProposal> {
  return apiPutJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}/draft`,
    draft,
  )
}

export function startRefinementSession(projectId: string, proposalId: string): Promise<SegmentationProposal> {
  return apiPostJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}/refinement-session`,
    {},
  )
}

export function restoreRefinementSession(projectId: string, proposalId: string): Promise<SegmentationProposal> {
  return apiPostJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}/refinement-session/restore`,
    {},
  )
}

export function resetMaskProposal(projectId: string, proposalId: string): Promise<SegmentationProposal> {
  return apiPostJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}/reset`,
    {},
  )
}

export async function getCurrentSegmentationProposal(
  projectId: string,
): Promise<SegmentationProposal | null> {
  try {
    return await apiGet<SegmentationProposal>(
      `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposal`,
    )
  } catch (error) {
    if (error instanceof ApiError && error.message === "Segmentation proposal not found.") return null
    throw error
  }
}

export function discardSegmentationProposal(projectId: string, proposalId: string): Promise<void> {
  return apiDelete(
    `/api/projects/${encodeURIComponent(projectId)}/segmentation/proposals/${encodeURIComponent(proposalId)}`,
  )
}

export function getRelief(projectId: string): Promise<ReliefRecord> {
  return apiGet<ReliefRecord>(`/api/projects/${encodeURIComponent(projectId)}/relief`)
}

export function startRegionMaskEdit(
  projectId: string,
  regionId: string,
): Promise<SegmentationProposal> {
  return apiPostJson<SegmentationProposal>(
    `/api/projects/${encodeURIComponent(projectId)}/relief/regions/${encodeURIComponent(regionId)}/mask-edit`,
    {},
  )
}

export function saveRegionMaskEdit(
  projectId: string,
  regionId: string,
  body: { proposal_id: string; name: string; role: ReliefRole },
): Promise<Project> {
  return apiPostJson<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/relief/regions/${encodeURIComponent(regionId)}/mask-edit/save`,
    body,
  )
}

export function discardRegionMaskEdit(
  projectId: string,
  regionId: string,
  proposalId: string,
): Promise<void> {
  return apiPostNoContent(
    `/api/projects/${encodeURIComponent(projectId)}/relief/regions/${encodeURIComponent(regionId)}/mask-edit/discard`,
    { proposal_id: proposalId },
  )
}

export function createReliefRegion(
  projectId: string,
  body: { name: string; role: ReliefRole; proposal_id: string },
): Promise<Project> {
  return apiPostJson<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/relief/regions`,
    body,
  )
}

export function applyReliefComposition(
  projectId: string,
  regions: Array<
    Pick<
      ReliefRegion,
      | "id"
      | "name"
      | "role"
      | "elevation"
      | "depth_range"
      | "feather"
      | "order"
      | "detail_amount"
      | "detail_scale_mm"
    >
  >,
  background: {
    background_depth_range: number
    background_elevation: number
    background_feather: number
  },
): Promise<Project> {
  return apiPostJson<Project>(`/api/projects/${encodeURIComponent(projectId)}/relief/composition`, {
    regions,
    ...background,
  })
}

export function deleteReliefRegion(projectId: string, regionId: string): Promise<Project> {
  return apiRequest<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/relief/regions/${encodeURIComponent(regionId)}`,
    { method: "DELETE" },
  )
}

export function getReliefMaskUrl(projectId: string, regionId: string): string {
  return apiUrl(
    `/api/projects/${encodeURIComponent(projectId)}/relief/regions/${encodeURIComponent(regionId)}/mask`,
  )
}

export function getReliefPreviewUrl(projectId: string, generatedAt: string): string {
  const version = encodeURIComponent(generatedAt)
  return apiUrl(
    `/api/projects/${encodeURIComponent(projectId)}/relief/preview?v=${version}`,
  )
}

export function generatePhysicalHeight(
  projectId: string,
  body: {
    stock_thickness_mm: number
    base_thickness_mm: number
    maximum_surface_height_mm: number
  },
): Promise<Project> {
  return apiPostJson<Project>(
    `/api/projects/${encodeURIComponent(projectId)}/relief/physical-height`,
    body,
  )
}

export function getPhysicalPreviewUrl(projectId: string, generatedAt: string): string {
  const version = encodeURIComponent(generatedAt)
  return apiUrl(
    `/api/projects/${encodeURIComponent(projectId)}/relief/physical-height/preview?v=${version}`,
  )
}

export function getPhysicalDiagnostics(projectId: string): Promise<PhysicalHeightDiagnostics> {
  return apiGet<PhysicalHeightDiagnostics>(physicalDiagnosticsPath(projectId))
}

export function getMeshPreview(
  projectId: string,
  maxVertices: number,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const params = new URLSearchParams()
  params.set("max_vertices", String(maxVertices))
  return apiGetArrayBuffer(
    `/api/projects/${encodeURIComponent(projectId)}/relief/mesh/preview?${params.toString()}`,
    signal,
  )
}

export function getPhysicalProfile(
  projectId: string,
  selection: ProfileSelection,
): Promise<PhysicalHeightProfile> {
  return apiGet<PhysicalHeightProfile>(physicalProfilePath(projectId, selection))
}

function aborted(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError"
}

export function getExportReadiness(
  projectId: string,
  selection: ExportSelection,
  signal?: AbortSignal,
): Promise<ExportReadiness> {
  return fetchExportReadiness(exportReadinessPath(projectId, selection), signal)
}

async function fetchExportReadiness(path: string, signal?: AbortSignal): Promise<ExportReadiness> {
  let response: Response
  try {
    response = await fetch(apiUrl(path), { method: "GET", signal })
  } catch (error) {
    if (aborted(error)) throw error
    throw new ApiError("Unable to reach the backend.")
  }
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new ApiError("Backend returned an unreadable response.", response.status)
  }
  const parsed = parseExportReadiness(body)
  if (!parsed) throw new ApiError("Backend returned an unreadable response.", response.status)
  return parsed
}

export async function postStlExport(
  projectId: string,
  selection: ExportSelection,
  signal?: AbortSignal,
): Promise<StlDownloadResult> {
  let response: Response
  try {
    response = await fetch(apiUrl(stlExportPath(projectId)), {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/octet-stream" },
      body: JSON.stringify(stlExportBody(selection)),
      signal,
    })
  } catch (error) {
    if (aborted(error)) throw error
    throw new ApiError("Unable to reach the backend.")
  }
  return interpretStlResponse(response)
}
