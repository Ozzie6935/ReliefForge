export type EditorTool =
  | "positive"
  | "negative"
  | "box"
  | "add"
  | "erase"
  | "wand-add"
  | "wand-subtract"
  | "pan"
  | "zoom-area"

/** The one image interaction allowed by the active tool. */
export type ImagePointerAction =
  | "sam-point"
  | "sam-box"
  | "add"
  | "erase"
  | "wand-add"
  | "wand-subtract"
  | "pan"
  | "zoom-area"

export type MaskOperation = "add" | "erase" | "wand-add" | "wand-subtract"

export function imagePointerAction(tool: EditorTool): ImagePointerAction {
  if (tool === "positive" || tool === "negative") return "sam-point"
  if (tool === "box") return "sam-box"
  return tool
}

/** Mask-changing tools. Pan and Zoom Area return null. */
export function maskOperation(tool: EditorTool): MaskOperation | null {
  const action = imagePointerAction(tool)
  if (action === "add" || action === "erase" || action === "wand-add" || action === "wand-subtract") {
    return action
  }
  return null
}

export type SamPromptState<TPoint, TBox> = {
  positive: TPoint[]
  negative: TPoint[]
  box: TBox | null
}

/** Successful SAM generation clears displayed prompts. A failure keeps them. */
export function promptsAfterMaskGeneration<TPoint, TBox>(
  success: boolean,
  prompts: SamPromptState<TPoint, TBox>,
): SamPromptState<TPoint, TBox> {
  if (!success) return prompts
  return { positive: [], negative: [], box: null }
}
