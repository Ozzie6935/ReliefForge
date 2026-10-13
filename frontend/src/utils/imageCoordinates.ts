export type ImagePixel = {
  x: number
  y: number
}

export type ImageFrame = {
  left: number
  top: number
  width: number
  height: number
  naturalWidth: number
  naturalHeight: number
}

function placedImage(frame: ImageFrame): { scale: number; offsetX: number; offsetY: number } | null {
  if (frame.naturalWidth <= 0 || frame.naturalHeight <= 0 || frame.width <= 0 || frame.height <= 0) {
    return null
  }
  const scale = Math.min(frame.width / frame.naturalWidth, frame.height / frame.naturalHeight)
  const displayedWidth = frame.naturalWidth * scale
  const displayedHeight = frame.naturalHeight * scale
  return {
    scale,
    offsetX: (frame.width - displayedWidth) / 2,
    offsetY: (frame.height - displayedHeight) / 2,
  }
}

export function clampClientToDisplayedImage(
  clientX: number,
  clientY: number,
  frame: ImageFrame,
): { x: number; y: number } | null {
  const placed = placedImage(frame)
  if (!placed) return null
  const left = frame.left + placed.offsetX
  const top = frame.top + placed.offsetY
  const right = left + frame.naturalWidth * placed.scale
  const bottom = top + frame.naturalHeight * placed.scale
  const maxX = Math.max(left, right - 1e-4)
  const maxY = Math.max(top, bottom - 1e-4)
  return {
    x: Math.min(Math.max(clientX, left), maxX),
    y: Math.min(Math.max(clientY, top), maxY),
  }
}

export function imagePixelFromClient(clientX: number, clientY: number, frame: ImageFrame): ImagePixel | null {
  const placed = placedImage(frame)
  if (!placed) return null
  const x = (clientX - frame.left - placed.offsetX) / placed.scale
  const y = (clientY - frame.top - placed.offsetY) / placed.scale
  if (x < 0 || y < 0 || x >= frame.naturalWidth || y >= frame.naturalHeight) return null
  return { x, y }
}

export function displayScale(frame: ImageFrame): number | null {
  return placedImage(frame)?.scale ?? null
}

export function elementLayoutFrame(image: {
  offsetWidth: number
  offsetHeight: number
  naturalWidth: number
  naturalHeight: number
}): ImageFrame {
  return {
    left: 0,
    top: 0,
    width: image.offsetWidth,
    height: image.offsetHeight,
    naturalWidth: image.naturalWidth,
    naturalHeight: image.naturalHeight,
  }
}

export function clientOffsetFromPixel(pixel: ImagePixel, frame: ImageFrame): { x: number; y: number } | null {
  const placed = placedImage(frame)
  if (!placed) return null
  return {
    x: placed.offsetX + pixel.x * placed.scale,
    y: placed.offsetY + pixel.y * placed.scale,
  }
}
