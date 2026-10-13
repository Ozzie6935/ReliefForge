import type { ArtifactStatus, PhysicalHeightRecord } from "../types/project.ts"
import { artifactStatus, statusExplanation } from "./provenanceStatus.ts"
import { DEFAULT_STOCK_THICKNESS_MM } from "./projectFormLayout.ts"
import {
  formatLengthInput,
  formatThicknessInput,
  toMillimeters,
  type DisplayUnit,
} from "./units.ts"

/** Example base and peak used until physical height is generated. Stock comes from the project or the application default. */
export const PHYSICAL_EXAMPLE_MM = {
  base: 5,
  peak: 17,
} as const

export const PHYSICAL_EXAMPLE_NOTE = "Example values, not machining recommendations."

export type PhysicalStockSource = "saved-physical" | "project" | "application-default"

export type PhysicalDisplayStatus = ArtifactStatus

export function physicalDisplayStatus(
  record: PhysicalHeightRecord | null | undefined,
): PhysicalDisplayStatus {
  return artifactStatus(record, record?.status)
}

export function physicalActionLabel(record: PhysicalHeightRecord | null | undefined): "Generate" | "Regenerate" {
  return record ? "Regenerate" : "Generate"
}

export function physicalStatusCopy(status: PhysicalDisplayStatus): string {
  return statusExplanation(status)
}

export function physicalCanonicalMillimeters(
  record: PhysicalHeightRecord | null | undefined,
  projectThicknessMm?: number | null,
): { stock: number; base: number; peak: number; stockSource: PhysicalStockSource } {
  if (record) {
    return {
      stock: record.stock_thickness_mm,
      base: record.base_thickness_mm,
      peak: record.maximum_surface_height_mm,
      stockSource: "saved-physical",
    }
  }
  if (projectThicknessMm != null && Number.isFinite(projectThicknessMm) && projectThicknessMm > 0) {
    return {
      stock: projectThicknessMm,
      base: PHYSICAL_EXAMPLE_MM.base,
      peak: PHYSICAL_EXAMPLE_MM.peak,
      stockSource: "project",
    }
  }
  return {
    stock: DEFAULT_STOCK_THICKNESS_MM,
    base: PHYSICAL_EXAMPLE_MM.base,
    peak: PHYSICAL_EXAMPLE_MM.peak,
    stockSource: "application-default",
  }
}

export function physicalStockNote(source: PhysicalStockSource, stockMatchesSource: boolean): string {
  if (!stockMatchesSource) {
    return "This stock thickness is only in the form. Generate physical height to store it there. The application default stays 1.50 in (38.10 mm), and the project thickness changes only when the project is saved."
  }
  if (source === "saved-physical") return "Saved physical-height thickness."
  if (source === "project") return `Project stock thickness. ${PHYSICAL_EXAMPLE_NOTE}`
  return `Application default stock thickness, 1.50 in (38.10 mm). ${PHYSICAL_EXAMPLE_NOTE}`
}

export type PhysicalDraft = {
  stock: string
  base: string
  peak: string
  usingExample: boolean
  stockSource: PhysicalStockSource
}

export function physicalInputs(
  record: PhysicalHeightRecord | null | undefined,
  unit: DisplayUnit,
  projectThicknessMm?: number | null,
): PhysicalDraft {
  const canonical = physicalCanonicalMillimeters(record, projectThicknessMm)
  return {
    stock: formatThicknessInput(canonical.stock, unit),
    base: formatLengthInput(canonical.base, unit),
    peak: formatLengthInput(canonical.peak, unit),
    usingExample: canonical.stockSource !== "saved-physical",
    stockSource: canonical.stockSource,
  }
}

/** Keep an unedited example or saved value exact when the display unit rounds it. */
export function commitLength(text: string, canonicalMm: number, unit: DisplayUnit): number | null {
  const trimmed = text.trim()
  if (!trimmed) return null
  const parsed = Number(trimmed)
  if (!Number.isFinite(parsed)) return null
  if (
    trimmed === formatLengthInput(canonicalMm, unit) ||
    trimmed === formatThicknessInput(canonicalMm, unit)
  ) {
    return canonicalMm
  }
  return toMillimeters(parsed, unit)
}

export function commitPhysicalMillimeters(
  draft: { stock: string; base: string; peak: string },
  canonical: { stock: number; base: number; peak: number },
  unit: DisplayUnit,
): { stock_thickness_mm: number; base_thickness_mm: number; maximum_surface_height_mm: number } | null {
  const stock = commitLength(draft.stock, canonical.stock, unit)
  const base = commitLength(draft.base, canonical.base, unit)
  const peak = commitLength(draft.peak, canonical.peak, unit)
  if (stock == null || base == null || peak == null) return null
  if (!(stock > 0 && base > 0 && peak > base && peak <= stock)) return null
  return {
    stock_thickness_mm: stock,
    base_thickness_mm: base,
    maximum_surface_height_mm: peak,
  }
}

export function physicalDraftProblem(
  draft: { stock: string; base: string; peak: string },
  canonical: { stock: number; base: number; peak: number },
  unit: DisplayUnit,
): "incomplete" | "order" | null {
  const stock = commitLength(draft.stock, canonical.stock, unit)
  const base = commitLength(draft.base, canonical.base, unit)
  const peak = commitLength(draft.peak, canonical.peak, unit)
  if (stock == null || base == null || peak == null) return "incomplete"
  if (!(stock > 0 && base > 0 && peak > base && peak <= stock)) return "order"
  return null
}

export const PHYSICAL_INCOMPLETE_ERROR =
  "Enter a stock thickness, a base thickness, and a maximum surface height."

export const PHYSICAL_ORDER_ERROR =
  "Base thickness must be greater than 0 and less than the maximum surface height. Maximum surface height cannot exceed stock thickness."

export function physicalInputsAfterUnitChange(
  draft: PhysicalDraft,
  canonical: { stock: number; base: number; peak: number },
  from: DisplayUnit,
  to: DisplayUnit,
): PhysicalDraft {
  const stock = commitLength(draft.stock, canonical.stock, from)
  const base = commitLength(draft.base, canonical.base, from)
  const peak = commitLength(draft.peak, canonical.peak, from)
  return {
    stock: stock == null ? draft.stock : formatThicknessInput(stock, to),
    base: base == null ? draft.base : formatLengthInput(base, to),
    peak: peak == null ? draft.peak : formatLengthInput(peak, to),
    usingExample: draft.usingExample,
    stockSource: draft.stockSource,
  }
}

export function physicalViewRequests(): {
  composition: false
  segmentation: false
  depth: false
  stl: false
  gcode: false
  regenerate: false
} {
  return {
    composition: false,
    segmentation: false,
    depth: false,
    stl: false,
    gcode: false,
    regenerate: false,
  }
}
