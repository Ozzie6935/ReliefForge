import type { ArtifactStatus } from "../types/project.ts"
import { editorKind, type EditorKind } from "./editorSession.ts"

export type ReliefView = "source" | "mask" | "composed" | "physical"

export const RELIEF_VIEWS: readonly { id: ReliefView; label: string }[] = [
  { id: "source", label: "Source" },
  { id: "mask", label: "Mask" },
  { id: "composed", label: "Composed Relief" },
  { id: "physical", label: "Physical Height" },
]

export const NO_ACCEPTED_REGION_MESSAGE = "No accepted relief region selected."
export const NO_COMPOSED_RELIEF_MESSAGE = "Apply Composition to create the composed relief."
export const NO_PHYSICAL_HEIGHT_MESSAGE = "Generate physical height from the composed relief."
export const STALE_MASK_NOTICE = "A region mask was saved. Apply Composition to update the relief."

export type WorkspaceRegion = {
  id: string
  name: string
  role: string
  order: number
}

export function defaultReliefView(hasComposedRelief: boolean): ReliefView {
  return hasComposedRelief ? "composed" : "source"
}

export function switchReliefView(next: ReliefView): ReliefView {
  return next
}

export function viewSwitchRequests(): { composition: false; segmentation: false; depth: false } {
  return { composition: false, segmentation: false, depth: false }
}

export const ACCEPTED_REGIONS_HEADING = "Accepted Regions"
export const DRAFT_REGIONS_HEADING = "Draft Regions"
export const NO_ACCEPTED_REGIONS_MESSAGE = "No accepted regions."
export const NO_DRAFT_REGIONS_MESSAGE = "No draft regions."

export type DraftRegion = {
  proposalId: string
  name: string
  role: string
}

export type DraftListEntry = {
  proposalId: string
  name: string
  summary: string
  className: string
  continueLabel: "Continue"
}

export function draftRegionList(
  drafts: DraftRegion[],
  roleLabel: (role: string) => string,
): DraftListEntry[] {
  return drafts.map((draft) => ({
    proposalId: draft.proposalId,
    name: draft.name.trim() || "Untitled",
    summary: `${roleLabel(draft.role)} · Draft`,
    className: "region-list-item is-draft",
    continueLabel: "Continue",
  }))
}

export function addRegionAvailability(draft: { name?: string } | null | undefined): {
  enabled: boolean
  message: string | null
} {
  if (draft === undefined) return { enabled: false, message: null }
  if (draft === null) return { enabled: true, message: null }
  const name = draft.name?.trim() || "current"
  return {
    enabled: false,
    message: `Finish or delete the ${name} draft before adding another region.`,
  }
}

export function continueExistingDraft(proposalId: string): {
  proposalId: string
  createsProposal: false
  callsSam: false
  replacesWorkingMask: false
  replacesAiMask: false
  acceptsRegion: false
  changesAcceptedRegions: false
  recomputesComposition: false
} {
  return {
    proposalId,
    createsProposal: false,
    callsSam: false,
    replacesWorkingMask: false,
    replacesAiMask: false,
    acceptsRegion: false,
    changesAcceptedRegions: false,
    recomputesComposition: false,
  }
}

export function draftIsAccepted(proposalId: string, acceptedIds: readonly string[]): boolean {
  return acceptedIds.includes(proposalId)
}

export function regionsAfterDraftAccepted(
  accepted: WorkspaceRegion[],
  draft: { id: string; name: string; role: string },
): { accepted: WorkspaceRegion[]; drafts: [] } {
  const order = accepted.reduce((highest, region) => Math.max(highest, region.order), -1) + 1
  return {
    accepted: [...accepted, { id: draft.id, name: draft.name, role: draft.role, order }],
    drafts: [],
  }
}

function byOrder(left: { id: string; order: number }, right: { id: string; order: number }): number {
  return left.order - right.order || left.id.localeCompare(right.id)
}

export function initialSelectedRegionId(regions: Array<{ id: string; order: number }>): string | null {
  return [...regions].sort(byOrder)[0]?.id ?? null
}

export function selectedRegionIdAfterChange(
  current: string | null,
  regions: Array<{ id: string; order: number }>,
): string | null {
  if (current && regions.some((region) => region.id === current)) return current
  return initialSelectedRegionId(regions)
}

export function selectAcceptedRegion(regionId: string): {
  selectedId: string
  createsProposal: false
  deletesProposal: false
  recomputesComposition: false
  writesMask: false
} {
  return {
    selectedId: regionId,
    createsProposal: false,
    deletesProposal: false,
    recomputesComposition: false,
    writesMask: false,
  }
}

export type RegionListEntry = {
  id: string
  name: string
  summary: string
  selected: boolean
  className: string
}

export function reliefRegionList(
  regions: WorkspaceRegion[],
  selectedId: string | null,
  roleLabel: (role: string) => string,
): RegionListEntry[] {
  return [...regions].sort(byOrder).map((region) => {
    const selected = region.id === selectedId
    return {
      id: region.id,
      name: region.name,
      summary: `${roleLabel(region.role)} · order ${region.order}`,
      selected,
      className: selected ? "region-list-item is-selected" : "region-list-item",
    }
  })
}

export type WorkspaceCanvas =
  | { mode: "editor" }
  | { mode: "source"; image: "source"; readOnly: true }
  | { mode: "mask"; image: "source"; regionId: string; readOnly: true; overlay: "positive-mask" }
  | { mode: "mask-empty"; message: typeof NO_ACCEPTED_REGION_MESSAGE; readOnly: true }
  | { mode: "composed"; image: "relief-preview"; generatedAt: string; readOnly: true }
  | { mode: "composed-empty"; message: typeof NO_COMPOSED_RELIEF_MESSAGE; readOnly: true }
  | {
      mode: "physical"
      image: "physical-preview"
      generatedAt: string
      status: Exclude<ArtifactStatus, "missing">
      readOnly: true
    }
  | { mode: "physical-empty"; message: typeof NO_PHYSICAL_HEIGHT_MESSAGE; readOnly: true }

export function workspaceCanvas(input: {
  editing: boolean
  view: ReliefView
  selectedRegionId: string | null
  generatedAt: string | null
  physicalGeneratedAt?: string | null
  physicalStatus?: ArtifactStatus
}): WorkspaceCanvas {
  if (input.editing) return { mode: "editor" }
  if (input.view === "source") return { mode: "source", image: "source", readOnly: true }
  if (input.view === "mask") {
    if (!input.selectedRegionId) {
      return { mode: "mask-empty", message: NO_ACCEPTED_REGION_MESSAGE, readOnly: true }
    }
    return {
      mode: "mask",
      image: "source",
      regionId: input.selectedRegionId,
      readOnly: true,
      overlay: "positive-mask",
    }
  }
  if (input.view === "physical") {
    if (
      !input.physicalGeneratedAt ||
      input.physicalStatus === "missing" ||
      input.physicalStatus == null
    ) {
      return { mode: "physical-empty", message: NO_PHYSICAL_HEIGHT_MESSAGE, readOnly: true }
    }
    return {
      mode: "physical",
      image: "physical-preview",
      generatedAt: input.physicalGeneratedAt,
      status: input.physicalStatus,
      readOnly: true,
    }
  }
  if (!input.generatedAt) {
    return { mode: "composed-empty", message: NO_COMPOSED_RELIEF_MESSAGE, readOnly: true }
  }
  return {
    mode: "composed",
    image: "relief-preview",
    generatedAt: input.generatedAt,
    readOnly: true,
  }
}

export function acceptedMaskDisplayUrl(maskUrl: string, updatedAt: string): string {
  const separator = maskUrl.includes("?") ? "&" : "?"
  return `${maskUrl}${separator}v=${encodeURIComponent(updatedAt)}`
}

export function displayedMaskRegionId(canvas: WorkspaceCanvas): string | null {
  return canvas.mode === "mask" ? canvas.regionId : null
}

export function maskViewIsReadOnly(canvas: WorkspaceCanvas): boolean {
  return canvas.mode !== "mask" || canvas.readOnly
}

/** Timestamp hint only. Provenance status comes from the backend, not this comparison. */
export function showStaleCompositionNotice(
  regions: Array<{ updated_at: string }>,
  generatedAt: string | null,
): boolean {
  if (!generatedAt) return false
  return regions.some((region) => {
    const regionTime = Date.parse(region.updated_at)
    const reliefTime = Date.parse(generatedAt)
    return Number.isFinite(regionTime) && Number.isFinite(reliefTime) && regionTime > reliefTime
  })
}

export function workspaceEditorKind(regionId: string | null): EditorKind {
  return editorKind(regionId)
}
