import { primaryMaskAction, type EditorKind } from "./editorSession.ts"

export const DESKTOP_INSPECTOR_MIN_PX = 340
export const DESKTOP_INSPECTOR_MAX_PX = 380
export const NARROW_LAYOUT_MAX_PX = 960
export const PROJECT_WORKSPACE_LABEL = "Project workspace"
export const RELIEF_REGION_LABEL = "Relief"
export const WORKING_CANVAS_LABEL = "Working canvas"

export type InspectorMode = "editor" | "browse"

export function inspectorMode(editing: boolean): InspectorMode {
  return editing ? "editor" : "browse"
}

export function showsViewSwitch(mode: InspectorMode): boolean {
  return mode === "browse"
}

export const EDITOR_INSPECTOR_SECTIONS = ["identity", "tool", "history", "navigation", "actions"] as const

export const BROWSE_INSPECTOR_SECTIONS = ["accepted", "drafts", "controls", "apply"] as const

export function historyRegion(): {
  container: "history-list"
  scrollsInternally: true
  preservesEntries: true
} {
  return { container: "history-list", scrollsInternally: true, preservesEntries: true }
}

export function editorActionBar(kind: EditorKind): {
  primary: "Accept Mask" | "Save Mask"
  close: "Close"
  deleteProposal: boolean
} {
  return {
    primary: primaryMaskAction(kind),
    close: "Close",
    deleteProposal: kind === "new-region",
  }
}

export function projectHeadingChrome(): {
  label: typeof PROJECT_WORKSPACE_LABEL
  presentation: "pill"
  sameLineAsTitle: true
  longNames: "truncate"
} {
  return {
    label: PROJECT_WORKSPACE_LABEL,
    presentation: "pill",
    sameLineAsTitle: true,
    longNames: "truncate",
  }
}

export function reliefVisibleChrome(): {
  reliefHeading: "accessible-only"
  workingCanvas: "accessible-only"
} {
  return { reliefHeading: "accessible-only", workingCanvas: "accessible-only" }
}

export function inspectorFrame(mode: InspectorMode): {
  architecture: "full-height"
  belowNavbar: true
  pinnedPrimaryActions: boolean
} {
  return {
    architecture: "full-height",
    belowNavbar: true,
    pinnedPrimaryActions: mode === "browse" || mode === "editor",
  }
}

export function inspectorSizing(viewportWidth: number):
  | { stacked: true; width: "fluid" }
  | { stacked: false; width: { min: number; max: number } } {
  if (viewportWidth <= NARROW_LAYOUT_MAX_PX) return { stacked: true, width: "fluid" }
  return {
    stacked: false,
    width: { min: DESKTOP_INSPECTOR_MIN_PX, max: DESKTOP_INSPECTOR_MAX_PX },
  }
}
