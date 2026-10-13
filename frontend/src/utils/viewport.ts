/**Viewport pan and zoom. These functions do not touch image pixels. */

/** 1 means the image fills the editor box. Zoom out does not go below this. */
export const MIN_VIEW_SCALE = 1
export const MAX_ZOOM_PERCENT = 800
export const MIN_NAVIGATION_DRAG_PX = 5

export type SourceRect = {
  x0: number
  y0: number
  x1: number
  y1: number
}

export type ViewportView = {
  scale: number
  panX: number
  panY: number
  percent: number
}

const PAN_STEP_FRACTION = 0.1

export function contentPlacement(
  layoutWidth: number,
  layoutHeight: number,
  naturalWidth: number,
  naturalHeight: number,
): { scale: number; offsetX: number; offsetY: number } | null {
  if (layoutWidth <= 0 || layoutHeight <= 0 || naturalWidth <= 0 || naturalHeight <= 0) return null
  const scale = Math.min(layoutWidth / naturalWidth, layoutHeight / naturalHeight)
  return {
    scale,
    offsetX: (layoutWidth - naturalWidth * scale) / 2,
    offsetY: (layoutHeight - naturalHeight * scale) / 2,
  }
}

export function clampViewScale(scale: number, contentScale: number): number {
  if (!(contentScale > 0)) return MIN_VIEW_SCALE
  const maxScale = MAX_ZOOM_PERCENT / 100 / contentScale
  return Math.min(maxScale, Math.max(MIN_VIEW_SCALE, scale))
}

export function clampViewportPan(input: {
  panX: number
  panY: number
  scale: number
  layoutWidth: number
  layoutHeight: number
  viewportWidth: number
  viewportHeight: number
}): { panX: number; panY: number } {
  return {
    panX: clampPanAxis(input.panX, input.layoutWidth * input.scale, input.viewportWidth),
    panY: clampPanAxis(input.panY, input.layoutHeight * input.scale, input.viewportHeight),
  }
}

export function panDelta(
  direction: "up" | "down" | "left" | "right",
  viewportWidth: number,
  viewportHeight: number,
): { x: number; y: number } {
  const stepX = viewportWidth * PAN_STEP_FRACTION
  const stepY = viewportHeight * PAN_STEP_FRACTION
  if (direction === "up") return { x: 0, y: stepY }
  if (direction === "down") return { x: 0, y: -stepY }
  if (direction === "left") return { x: stepX, y: 0 }
  return { x: -stepX, y: 0 }
}

export function zoomAreaView(input: {
  selection: SourceRect
  displayWidth: number
  displayHeight: number
  naturalWidth: number
  naturalHeight: number
  layoutWidth: number
  layoutHeight: number
  viewportWidth: number
  viewportHeight: number
}): ViewportView | null {
  if (input.displayWidth < MIN_NAVIGATION_DRAG_PX || input.displayHeight < MIN_NAVIGATION_DRAG_PX) return null
  const placed = contentPlacement(
    input.layoutWidth,
    input.layoutHeight,
    input.naturalWidth,
    input.naturalHeight,
  )
  const selectionWidth = input.selection.x1 - input.selection.x0
  const selectionHeight = input.selection.y1 - input.selection.y0
  if (
    !placed ||
    selectionWidth <= 0 ||
    selectionHeight <= 0 ||
    input.viewportWidth <= 0 ||
    input.viewportHeight <= 0
  ) {
    return null
  }
  const fittedScale =
    Math.min(input.viewportWidth / selectionWidth, input.viewportHeight / selectionHeight) / placed.scale
  const scale = clampViewScale(fittedScale, placed.scale)
  const pixelScale = scale * placed.scale
  const centerX = placed.offsetX + ((input.selection.x0 + input.selection.x1) / 2) * placed.scale
  const centerY = placed.offsetY + ((input.selection.y0 + input.selection.y1) / 2) * placed.scale
  const pan = clampViewportPan({
    panX: input.viewportWidth / 2 - centerX * scale,
    panY: input.viewportHeight / 2 - centerY * scale,
    scale,
    layoutWidth: input.layoutWidth,
    layoutHeight: input.layoutHeight,
    viewportWidth: input.viewportWidth,
    viewportHeight: input.viewportHeight,
  })
  return {
    scale,
    panX: pan.panX,
    panY: pan.panY,
    percent: pixelScale * 100,
  }
}

/** View scale for 100%: one source pixel per CSS pixel. Fit remains scale 1. */
export function sourcePixelScale(fittedDisplayWidth: number, naturalWidth: number): number {
  if (!(fittedDisplayWidth > 0) || !(naturalWidth > 0)) return MIN_VIEW_SCALE
  return naturalWidth / fittedDisplayWidth
}

export function viewAfterResize(input: {
  scale: number
  panX: number
  panY: number
  layoutWidth: number
  layoutHeight: number
  viewportWidth: number
  viewportHeight: number
}): { scale: number; panX: number; panY: number } {
  const fitted = input.scale <= MIN_VIEW_SCALE + 0.001
  const scale = fitted ? MIN_VIEW_SCALE : input.scale
  const pan = clampViewportPan({
    panX: fitted ? 0 : input.panX,
    panY: fitted ? 0 : input.panY,
    scale,
    layoutWidth: input.layoutWidth,
    layoutHeight: input.layoutHeight,
    viewportWidth: input.viewportWidth,
    viewportHeight: input.viewportHeight,
  })
  return { scale, panX: pan.panX, panY: pan.panY }
}

function clampPanAxis(pan: number, visual: number, viewport: number): number {
  if (viewport <= 0) return 0
  if (visual <= viewport + 0.5) {
    return (viewport - visual) / 2
  }
  const margin = Math.min(48, viewport * 0.25)
  return Math.min(Math.max(pan, margin - visual), viewport - margin)
}
