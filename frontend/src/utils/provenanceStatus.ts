import type { ArtifactStatus, PhysicalHeightRecord, ReliefRegion } from "../types/project.ts"

const STATUSES: readonly ArtifactStatus[] = ["missing", "unknown", "current", "stale"]

export function isArtifactStatus(value: string | null | undefined): value is ArtifactStatus {
  return value != null && (STATUSES as readonly string[]).includes(value)
}

/** Null record is Missing. A record with no recognized status is Unknown. */
export function artifactStatus(
  record: object | null | undefined,
  status: string | null | undefined,
): ArtifactStatus {
  if (record == null) return "missing"
  return isArtifactStatus(status) ? status : "unknown"
}

export function statusLabel(status: ArtifactStatus): string {
  if (status === "current") return "Current"
  if (status === "stale") return "Stale"
  if (status === "unknown") return "Unknown"
  return "Missing"
}

export function statusExplanation(status: ArtifactStatus): string {
  if (status === "unknown") return "Existing artifact; historical provenance cannot be verified."
  if (status === "stale") return "Upstream input or saved settings have changed; regeneration is required."
  if (status === "missing") return "No usable artifact is available."
  return "Artifact matches its recorded dependencies."
}

export function statusClassName(status: ArtifactStatus): string {
  return `provenance-status is-${status}`
}

export const RAW_UNKNOWN_GUIDANCE = "Regenerate Raw Depth from the current source."
export const CONDITIONED_UNKNOWN_GUIDANCE =
  "First establish Current raw depth, then Apply Conditioning."
export const RELIEF_UNKNOWN_GUIDANCE =
  "First establish Current conditioned depth, then Apply Composition."
export const PHYSICAL_UNKNOWN_GUIDANCE =
  "First establish Current relief, then regenerate physical height."

export function recoveryGuidance(
  stage: "raw" | "conditioned" | "relief" | "physical",
  status: ArtifactStatus,
): string | null {
  if (status !== "unknown") return null
  if (stage === "raw") return RAW_UNKNOWN_GUIDANCE
  if (stage === "conditioned") return CONDITIONED_UNKNOWN_GUIDANCE
  if (stage === "relief") return RELIEF_UNKNOWN_GUIDANCE
  return PHYSICAL_UNKNOWN_GUIDANCE
}

export function canApplyConditioning(raw: ArtifactStatus): boolean {
  return raw === "current"
}

export function canApplyComposition(conditioned: ArtifactStatus): boolean {
  return conditioned === "current"
}

export function canGeneratePhysical(relief: ArtifactStatus): boolean {
  return relief === "current"
}

export function canRegenerateRaw(hasSource: boolean): boolean {
  return hasSource
}

export function physicalPreviewCaption(status: ArtifactStatus): string {
  if (status === "current") return "Physical height"
  if (status === "missing") return "Physical height"
  return `Saved historical output — ${statusLabel(status)}`
}

export function compositionPreviewCaption(): string {
  return "Last saved composition"
}

type RegionSettings = Pick<
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

function detailAmount(region: Pick<ReliefRegion, "detail_amount">): number {
  return region.detail_amount ?? 0
}

function detailScaleMm(region: Pick<ReliefRegion, "detail_scale_mm">): number | null {
  return region.detail_scale_mm ?? null
}

export function regionSettingsPending(drafts: RegionSettings[], saved: RegionSettings[]): boolean {
  if (drafts.length !== saved.length) return true
  return drafts.some((draft) => {
    const match = saved.find((region) => region.id === draft.id)
    if (!match) return true
    return (
      draft.name !== match.name ||
      draft.role !== match.role ||
      draft.elevation !== match.elevation ||
      draft.depth_range !== match.depth_range ||
      draft.feather !== match.feather ||
      draft.order !== match.order ||
      detailAmount(draft) !== detailAmount(match) ||
      detailScaleMm(draft) !== detailScaleMm(match)
    )
  })
}

export function physicalSettingsPending(
  committed: {
    stock_thickness_mm: number
    base_thickness_mm: number
    maximum_surface_height_mm: number
  } | null,
  saved: Pick<
    PhysicalHeightRecord,
    "stock_thickness_mm" | "base_thickness_mm" | "maximum_surface_height_mm"
  > | null,
): boolean {
  if (!saved || !committed) return false
  return (
    committed.stock_thickness_mm !== saved.stock_thickness_mm ||
    committed.base_thickness_mm !== saved.base_thickness_mm ||
    committed.maximum_surface_height_mm !== saved.maximum_surface_height_mm
  )
}

export function failedProcessingResult<TProject, TDraft>(input: {
  error: unknown
  fallback: string
  project: TProject
  draft: TDraft
}): {
  message: string
  project: TProject
  draft: TDraft
  succeeded: false
  previewsCleared: false
} {
  const message =
    input.error instanceof Error && input.error.message.trim()
      ? input.error.message
      : input.fallback
  return {
    message,
    project: input.project,
    draft: input.draft,
    succeeded: false,
    previewsCleared: false,
  }
}
