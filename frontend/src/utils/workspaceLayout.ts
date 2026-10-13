import type { DepthConditioningParameters } from "../types/project.ts"
import {
  DESKTOP_INSPECTOR_MAX_PX,
  DESKTOP_INSPECTOR_MIN_PX,
  NARROW_LAYOUT_MAX_PX,
  inspectorSizing,
} from "./inspectorLayout.ts"

export { DESKTOP_INSPECTOR_MAX_PX, DESKTOP_INSPECTOR_MIN_PX, NARROW_LAYOUT_MAX_PX, inspectorSizing }

export const RELATIVE_DEPTH_LABEL =
  "Relative depth is dimensionless, from 0 far to 1 near. It is not a physical CNC height."

export const DEPTH_INSPECTOR_CONTROLS = [
  "Far Clip",
  "Near Clip",
  "Gamma",
  "Contrast",
  "Smoothing",
  "Edge Preservation",
  "Invert Depth",
  "Reset to Defaults",
] as const

export const DEPTH_VIEWS = [
  { id: "conditioned", label: "Conditioned" },
  { id: "source", label: "Source" },
  { id: "raw", label: "Raw" },
] as const

export type DepthView = (typeof DEPTH_VIEWS)[number]["id"]

export function depthParametersMatch(
  draft: DepthConditioningParameters,
  saved: DepthConditioningParameters,
): boolean {
  return (
    draft.far_percentile === saved.far_percentile &&
    draft.near_percentile === saved.near_percentile &&
    draft.gamma === saved.gamma &&
    draft.contrast === saved.contrast &&
    draft.smoothing === saved.smoothing &&
    draft.edge_preservation === saved.edge_preservation &&
    draft.invert === saved.invert
  )
}

export function conditioningIsPending(
  draft: DepthConditioningParameters,
  saved: DepthConditioningParameters,
): boolean {
  return !depthParametersMatch(draft, saved)
}

/** Switching the depth preview does not generate, apply, or write artifacts. */
export function depthViewChange(): { generate: false; apply: false; writesArtifacts: false } {
  return { generate: false, apply: false, writesArtifacts: false }
}

export const SOURCE_FORMAT_NOTE =
  "JPEG or PNG. Maximum 50 MB. The original file is stored unchanged."

export const SOURCE_REPLACE_WARNING =
  "Replacing the source image may require depth and relief to be regenerated."

export const SOURCE_SCALE_NOTE =
  "The picture may be scaled to fit this window. That does not change the artwork size."

export type SourcePreviewState = "empty" | "loading" | "loaded" | "failed"

export function sourcePreviewState(input: {
  hasSource: boolean
  imageLoaded: boolean
  imageFailed: boolean
}): SourcePreviewState {
  if (!input.hasSource) return "empty"
  if (input.imageFailed) return "failed"
  if (!input.imageLoaded) return "loading"
  return "loaded"
}

export function sourceUploadEnabled(fileSelected: boolean, uploading: boolean): boolean {
  return fileSelected && !uploading
}

/** 3D tab frame. The viewport track is the larger share. Export keeps its own rows. */
export function meshViewportFrame(viewportWidth: number): {
  stacked: boolean
  viewportTrack: "minmax(0, 1fr)" | "minmax(0, 1.8fr)"
  inspectorMinPx: number | null
  inspectorMaxPx: number | null
  inspectorTrack: "minmax(340px, 380px)" | "minmax(0, 1fr)"
  horizontalOverflow: "clip"
  canvasFillsViewport: true
} {
  const sizing = inspectorSizing(viewportWidth)
  if (sizing.stacked) {
    return {
      stacked: true,
      viewportTrack: "minmax(0, 1.8fr)",
      inspectorMinPx: null,
      inspectorMaxPx: null,
      inspectorTrack: "minmax(0, 1fr)",
      horizontalOverflow: "clip",
      canvasFillsViewport: true,
    }
  }
  return {
    stacked: false,
    viewportTrack: "minmax(0, 1fr)",
    inspectorMinPx: sizing.width.min,
    inspectorMaxPx: sizing.width.max,
    inspectorTrack: "minmax(340px, 380px)",
    horizontalOverflow: "clip",
    canvasFillsViewport: true,
  }
}

export function sourceWorkspaceFrame(viewportWidth: number): {
  shell: "full-height"
  belowNavbar: true
  pinnedPrimaryAction: "Upload Image"
  previewFit: "contain"
  stacked: boolean
  inspectorWidth: "fluid" | { min: number; max: number }
} {
  const sizing = inspectorSizing(viewportWidth)
  return {
    shell: "full-height",
    belowNavbar: true,
    pinnedPrimaryAction: "Upload Image",
    previewFit: "contain",
    stacked: sizing.stacked,
    inspectorWidth: sizing.stacked ? "fluid" : sizing.width,
  }
}

export function depthWorkspaceFrame(viewportWidth: number): {
  shell: "full-height"
  belowNavbar: true
  pinnedPrimaryAction: "Apply Conditioning"
  previewFit: "contain"
  stacked: boolean
  inspectorWidth: "fluid" | { min: number; max: number }
} {
  const sizing = inspectorSizing(viewportWidth)
  return {
    shell: "full-height",
    belowNavbar: true,
    pinnedPrimaryAction: "Apply Conditioning",
    previewFit: "contain",
    stacked: sizing.stacked,
    inspectorWidth: sizing.stacked ? "fluid" : sizing.width,
  }
}
