export type EditorKind = "new-region" | "accepted-region"

export function editorKind(regionId: string | null): EditorKind {
  return regionId === null ? "new-region" : "accepted-region"
}

export function primaryMaskAction(kind: EditorKind): "Accept Mask" | "Save Mask" {
  return kind === "accepted-region" ? "Save Mask" : "Accept Mask"
}

export function resetMaskAction(kind: EditorKind): "Reset to AI Mask" | "Reset to Saved Mask" {
  return kind === "accepted-region" ? "Reset to Saved Mask" : "Reset to AI Mask"
}

export function resetMaskPrompt(kind: EditorKind): string {
  return kind === "accepted-region"
    ? "Reset all manual refinement and return to the saved region mask?"
    : "Reset all manual refinement and return to the original AI mask?"
}

export type EditBaseline = {
  name: string
  role: string
}

export type EditDraft = EditBaseline & {
  historyCursor: number
}

export const DISCARD_UNSAVED_CHANGES = "Discard unsaved changes?"

const SAM_CONTROL_LABELS = ["Positive Prompt", "Negative Prompt", "Box Prompt", "Generate Mask", "Update Mask"] as const

const ACCEPTED_HIDDEN_HELP = new Set([
  "Positive Prompt",
  "Negative Prompt",
  "Box Prompt",
  "Generate Mask",
  "Reset to AI Mask",
  "Accept Mask",
])

export function editSessionDirty(baseline: EditBaseline, draft: EditDraft): boolean {
  return draft.historyCursor > 0 || draft.name !== baseline.name || draft.role !== baseline.role
}

export function closeShouldConfirmDiscard(kind: EditorKind, dirty: boolean): boolean {
  return kind === "accepted-region" && dirty
}

export function discardPrompt(kind: EditorKind, dirty: boolean): string | null {
  return closeShouldConfirmDiscard(kind, dirty) ? DISCARD_UNSAVED_CHANGES : null
}

export function keepEditing<T>(session: T): T {
  return session
}

export function visibleSamLabels(kind: EditorKind): readonly string[] {
  return generateMaskEnabled(kind) ? SAM_CONTROL_LABELS : []
}

export function initialEditorTool(kind: EditorKind): "positive" | "add" {
  return kind === "accepted-region" ? "add" : "positive"
}

export function toolAfterZoom(kind: EditorKind, previous: string): string {
  if (
    kind === "accepted-region" &&
    (previous === "zoom-area" || previous === "positive" || previous === "negative" || previous === "box")
  ) {
    return "add"
  }
  return previous === "zoom-area" ? "positive" : previous
}

export function maskEditorNote(kind: EditorKind): string {
  return kind === "accepted-region"
    ? "Choose a tool on the left. Brush and Magic Wand edit the working mask."
    : "Choose a tool on the left. Prompt tools place AI hints. Brush and Magic Wand edit the working mask. A successful mask update clears the displayed prompts."
}

export function acceptedRegionInspectorActions(): readonly ["Edit Mask", "Delete"] {
  return ["Edit Mask", "Delete"]
}

export function editOpenRequests(): readonly ["mask-edit"] {
  return ["mask-edit"]
}

export function generateMaskEnabled(kind: EditorKind): boolean {
  return kind === "new-region"
}

export type HelpEntry = {
  term: string
  detail: string
}

const HELP_ENTRIES: readonly HelpEntry[] = [
  { term: "Zoom Area", detail: "Drag a rectangle on the image to fill the canvas with that area." },
  { term: "Fit", detail: "Fit the whole image inside the canvas. This is the smallest zoom." },
  { term: "100%", detail: "Show one source pixel per screen pixel." },
  { term: "Zoom In / Zoom Out", detail: "Zoom around the canvas. Zoom Out stops at Fit. The maximum zoom is 800%." },
  { term: "Pan", detail: "Drag the image inside the canvas. The arrow pad and keyboard arrows pan as well." },
  { term: "Positive Prompt", detail: "Tell AI this belongs to the subject. Shown in green." },
  { term: "Negative Prompt", detail: "Tell AI this does not belong to the subject. Shown in red." },
  { term: "Box Prompt", detail: "Drag a rectangle around the subject for AI." },
  { term: "Add Brush", detail: "Manually add pixels to the mask." },
  { term: "Erase Brush", detail: "Manually remove pixels from the mask." },
  {
    term: "Magic Wand Add",
    detail: "Click to add a contiguous area of similar color. Adjust Tolerance first; the slider does not apply by itself.",
  },
  { term: "Magic Wand Subtract", detail: "Click to remove a contiguous area of similar color." },
  {
    term: "Generate Mask",
    detail:
      "Prompt tools place AI hints. Generate or update the mask from those hints. A successful update clears the displayed prompts. Brush and Magic Wand then edit the working mask.",
  },
  {
    term: "History",
    detail:
      "Session Start is the mask when this visit began. Click an entry to return to it. A new edit after going backward drops the later entries. Reloading keeps the working mask and starts again at Session Start, with Undo disabled.",
  },
  {
    term: "Reset to AI Mask",
    detail: "On a new region, replace the working mask with the original AI mask. This clears session history and cannot be undone.",
  },
  {
    term: "Reset to Saved Mask",
    detail: "While editing an accepted region, restore the mask from when Edit Mask was opened.",
  },
  { term: "Accept Mask", detail: "Save the current working mask as a new permanent relief region." },
  {
    term: "Save Mask",
    detail: "Replace the accepted region mask with the current working mask. The region keeps its id and settings. Apply Composition separately.",
  },
]

export function helpTopics(kind: EditorKind): readonly HelpEntry[] {
  if (generateMaskEnabled(kind)) return HELP_ENTRIES
  return HELP_ENTRIES.filter((entry) => !ACCEPTED_HIDDEN_HELP.has(entry.term))
}

export function compositionNeedsApply(regionUpdatedAt: string, reliefGeneratedAt: string | null): boolean {
  if (!reliefGeneratedAt) return false
  const regionTime = Date.parse(regionUpdatedAt)
  const reliefTime = Date.parse(reliefGeneratedAt)
  return Number.isFinite(regionTime) && Number.isFinite(reliefTime) && regionTime > reliefTime
}
