import type { ProjectCreateRequest } from "../types/project.ts"
import {
  formatLengthInput,
  formatThicknessInput,
  toMillimeters,
  unitLabel,
  type DisplayUnit,
} from "./units.ts"

export const DEFAULT_STOCK_THICKNESS_MM = 38.1

export const PROJECT_FORM_INTRO = "Physical dimensions are independent of image pixels."

export const STOCK_GUIDANCE = "Stock dimensions describe the physical material."

export const ARTWORK_GUIDANCE =
  "Artwork dimensions describe the relief area. Artwork Center X and Y are measured from the front-left stock origin."

export const DISPLAY_UNIT_NOTE = "Changing the display unit does not change the stored geometry."

export const PROJECT_FORM_SECTIONS = [
  { id: "project", title: "Project", fields: ["Project Name"] },
  { id: "stock", title: "Stock dimensions", fields: ["Stock Width", "Stock Height", "Stock Thickness"] },
  {
    id: "artwork",
    title: "Artwork size and placement",
    fields: ["Artwork Width", "Artwork Height", "Artwork Center X", "Artwork Center Y"],
  },
] as const

export type GeometryField = {
  text: string
  mm: number | null
  edited: boolean
  omitIfUnedited?: boolean
}

export type GeometryFields = {
  stockWidth: GeometryField
  stockHeight: GeometryField
  stockThickness: GeometryField
  artworkWidth: GeometryField
  artworkHeight: GeometryField
  centerX: GeometryField
  centerY: GeometryField
}

const FIELD_KEYS = [
  "stockWidth",
  "stockHeight",
  "stockThickness",
  "artworkWidth",
  "artworkHeight",
  "centerX",
  "centerY",
] as const satisfies readonly (keyof GeometryFields)[]

function parseNumber(value: string): number | null {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return null
  return parsed
}

export function defaultStockThicknessField(unit: DisplayUnit, omitIfUnedited: boolean): GeometryField {
  return {
    text: formatThicknessInput(DEFAULT_STOCK_THICKNESS_MM, unit),
    mm: DEFAULT_STOCK_THICKNESS_MM,
    edited: false,
    omitIfUnedited,
  }
}

export function stockThicknessNote(input: {
  creating: boolean
  savedThicknessMm: number | null
  edited: boolean
}): string {
  if (input.creating) {
    return "Application default, 1.50 in (38.10 mm). Saving stores this thickness on the project. It does not change the application default."
  }
  if (input.savedThicknessMm != null && !input.edited) {
    return "Project stock thickness. Saving keeps this value on the project. It does not change the application default."
  }
  if (input.edited) {
    return "This thickness will be stored on the project when you save. The application default stays 1.50 in (38.10 mm)."
  }
  return "Application default, 1.50 in (38.10 mm). This project has no saved stock thickness. Change it and save to store a project thickness."
}

export function resolveMillimeters(
  field: GeometryField,
  unit: DisplayUnit,
  positive: boolean,
): number | null {
  if (!field.edited && field.mm !== null) {
    if (positive && field.mm <= 0) return null
    return field.mm
  }
  const parsed = parseNumber(field.text)
  if (parsed === null || (positive && parsed <= 0)) return null
  return toMillimeters(parsed, unit)
}

function displayText(key: keyof GeometryFields, millimeters: number, unit: DisplayUnit): string {
  if (key === "stockThickness") return formatThicknessInput(millimeters, unit)
  return formatLengthInput(millimeters, unit)
}

export function geometryAfterUnitChange(
  fields: GeometryFields,
  from: DisplayUnit,
  to: DisplayUnit,
): GeometryFields {
  const next = { ...fields }
  for (const key of FIELD_KEYS) {
    const field = fields[key]
    if (!field.edited) {
      next[key] =
        field.mm === null ? field : { ...field, text: displayText(key, field.mm, to) }
      continue
    }
    const parsed = parseNumber(field.text)
    if (parsed === null) {
      next[key] = field
      continue
    }
    const millimeters = toMillimeters(parsed, from)
    next[key] = {
      text: displayText(key, millimeters, to),
      mm: millimeters,
      edited: false,
      omitIfUnedited: false,
    }
  }
  return next
}

export function projectGeometryRequest(
  name: string,
  fields: GeometryFields,
  unit: DisplayUnit,
): { ok: true; request: ProjectCreateRequest } | { ok: false; error: string } {
  const stockWidth = resolveMillimeters(fields.stockWidth, unit, true)
  const stockHeight = resolveMillimeters(fields.stockHeight, unit, true)
  const artworkWidth = resolveMillimeters(fields.artworkWidth, unit, true)
  const artworkHeight = resolveMillimeters(fields.artworkHeight, unit, true)
  const centerX = resolveMillimeters(fields.centerX, unit, false)
  const centerY = resolveMillimeters(fields.centerY, unit, false)
  const keepUnsavedThickness = Boolean(fields.stockThickness.omitIfUnedited) && !fields.stockThickness.edited
  const stockThickness = keepUnsavedThickness ? null : resolveMillimeters(fields.stockThickness, unit, true)
  if (
    !name.trim() ||
    stockWidth === null ||
    stockHeight === null ||
    artworkWidth === null ||
    artworkHeight === null ||
    centerX === null ||
    centerY === null ||
    (!keepUnsavedThickness && stockThickness === null)
  ) {
    return { ok: false, error: `Enter a name and positive ${unitLabel(unit)} dimensions.` }
  }
  return {
    ok: true,
    request: {
      name: name.trim(),
      stock: {
        width_mm: stockWidth,
        height_mm: stockHeight,
        thickness_mm: stockThickness,
      },
      artwork: {
        width_mm: artworkWidth,
        height_mm: artworkHeight,
        center_x_mm: centerX,
        center_y_mm: centerY,
      },
    },
  }
}

export function projectFormColumns(viewportWidth: number): 1 | 2 {
  return viewportWidth <= 640 ? 1 : 2
}

export function claimProjectSubmit(lock: { held: boolean }): boolean {
  if (lock.held) return false
  lock.held = true
  return true
}

export function releaseProjectSubmit(lock: { held: boolean }): void {
  lock.held = false
}
