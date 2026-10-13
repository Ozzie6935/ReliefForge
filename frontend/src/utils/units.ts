/** Canonical geometry is millimeters. Display units are a UI preference. */

export const MILLIMETERS_PER_INCH = 25.4

export const UNIT_PREFERENCE_KEY = "cnc-relief-studio.units"

export type DisplayUnit = "in" | "mm"

export function inchesToMillimeters(inches: number): number {
  const millimeters = inches * MILLIMETERS_PER_INCH
  // Drop binary floating-point residue. This keeps 14 * 25.4 at 355.6
  // and 10.1905 * 25.4 at 258.8387. It does not coarsen the result.
  return Math.round(millimeters * 1_000_000) / 1_000_000
}

export function millimetersToInches(millimeters: number): number {
  return millimeters / MILLIMETERS_PER_INCH
}

export function toMillimeters(value: number, unit: DisplayUnit): number {
  if (unit === "mm") return Math.round(value * 1_000_000) / 1_000_000
  return inchesToMillimeters(value)
}

export function fromMillimeters(millimeters: number, unit: DisplayUnit): number {
  if (unit === "mm") return millimeters
  return millimetersToInches(millimeters)
}

export function readUnitPreference(): DisplayUnit {
  try {
    return window.localStorage.getItem(UNIT_PREFERENCE_KEY) === "mm" ? "mm" : "in"
  } catch {
    return "in"
  }
}

export function writeUnitPreference(unit: DisplayUnit): void {
  window.localStorage.setItem(UNIT_PREFERENCE_KEY, unit)
}

/** Display only. This does not change values sent to the API. */
export function formatInches(millimeters: number): string {
  return formatLength(millimeters, "in")
}

export function formatLength(millimeters: number, unit: DisplayUnit): string {
  if (unit === "mm") {
    const text = String(Number(millimeters.toFixed(6)))
    return `${text} mm`
  }
  const inches = Number(millimetersToInches(millimeters).toFixed(4))
  return `${inches} in`
}

export function formatLengthInput(millimeters: number, unit: DisplayUnit): string {
  if (unit === "mm") return String(Number(millimeters.toFixed(6)))
  return String(Number(millimetersToInches(millimeters).toFixed(4)))
}

/** Two decimal places when the display value lands on a 0.01 grid. */
export function formatThicknessInput(millimeters: number, unit: DisplayUnit): string {
  const value = unit === "mm" ? millimeters : millimetersToInches(millimeters)
  const hundredths = Math.round(value * 100) / 100
  if (Math.abs(value - hundredths) < 1e-6) return hundredths.toFixed(2)
  return formatLengthInput(millimeters, unit)
}

export function formatThickness(millimeters: number, unit: DisplayUnit): string {
  return `${formatThicknessInput(millimeters, unit)} ${unitLabel(unit)}`
}

export function unitLabel(unit: DisplayUnit): string {
  return unit === "mm" ? "mm" : "in"
}
