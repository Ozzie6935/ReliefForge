import { formatLengthInput, toMillimeters, type DisplayUnit } from "./units.ts"

/** Canonical physical feature width. Inches are display only. */
export const DETAIL_SCALE_MIN_MM = 0.25
export const DETAIL_SCALE_MAX_MM = 1
export const DETAIL_SCALE_INITIAL_MM = 0.5

const EXACT_SCALES_MM = [DETAIL_SCALE_MIN_MM, DETAIL_SCALE_INITIAL_MM, DETAIL_SCALE_MAX_MM] as const

export function parseDetailScaleMm(text: string, unit: DisplayUnit): number | null {
  const trimmed = text.trim()
  if (trimmed === "") return null
  const value = Number(trimmed)
  if (!Number.isFinite(value)) return null
  for (const candidate of EXACT_SCALES_MM) {
    const shown = Number(formatLengthInput(candidate, unit))
    if (value === shown) return candidate
  }
  const millimeters = toMillimeters(value, unit)
  if (millimeters < DETAIL_SCALE_MIN_MM || millimeters > DETAIL_SCALE_MAX_MM) return null
  return millimeters
}

export const DETAIL_SAMPLING_WARNING =
  "This detail scale is smaller than one source pixel on at least one axis. Fine features may be poorly resolved."

export interface DetailSamplingInput {
  detailScaleMm: number | null
  artworkWidthMm?: number | null
  artworkHeightMm?: number | null
  widthPx?: number | null
  heightPx?: number | null
}

export interface DetailSamplingGuidance {
  warning: string | null
  sigmaXPx: number | null
  sigmaYPx: number | null
}

function positiveFinite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
}

function pixelCount(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 2
}

/**
 * Physical-mode sampling check. Null scale is the 2-pixel mode and never warns.
 * Missing or unusable dimensions omit the warning instead of inventing a pitch.
 * Stock size is not an input.
 */
export function detailSamplingGuidance(input: DetailSamplingInput): DetailSamplingGuidance {
  const silent = { warning: null, sigmaXPx: null, sigmaYPx: null }
  if (input.detailScaleMm == null || !positiveFinite(input.detailScaleMm)) return silent
  if (
    !positiveFinite(input.artworkWidthMm) ||
    !positiveFinite(input.artworkHeightMm) ||
    !pixelCount(input.widthPx) ||
    !pixelCount(input.heightPx)
  ) {
    return silent
  }
  const pitchX = input.artworkWidthMm / (input.widthPx - 1)
  const pitchY = input.artworkHeightMm / (input.heightPx - 1)
  const sigmaXPx = input.detailScaleMm / pitchX
  const sigmaYPx = input.detailScaleMm / pitchY
  if (!Number.isFinite(sigmaXPx) || !Number.isFinite(sigmaYPx)) return silent
  const warning = sigmaXPx < 1 || sigmaYPx < 1 ? DETAIL_SAMPLING_WARNING : null
  return { warning, sigmaXPx, sigmaYPx }
}
