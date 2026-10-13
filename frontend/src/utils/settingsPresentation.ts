import type { HealthResponse } from "../types/health.ts"
import { UNIT_PREFERENCE_KEY, type DisplayUnit } from "./units.ts"

export const SETTINGS_UNIT_NOTE =
  "Project geometry is stored internally in millimeters regardless of the display unit. Switching units changes how sizes are shown. It does not change the saved project."

export const SETTINGS_STOCK_NOTE =
  "The application default stock thickness is 1.50 in (38.10 mm). Each project can store its own thickness. A project value does not change this default."

export const SETTINGS_UNIT_OPTIONS: readonly { value: DisplayUnit; label: string }[] = [
  { value: "in", label: "Inches" },
  { value: "mm", label: "Millimeters" },
]

export const SETTINGS_UNIT_STORAGE_KEY = UNIT_PREFERENCE_KEY

export type SettingsConnection = "checking" | "connected" | "disconnected"

export function settingsConnectionLabel(connection: SettingsConnection): string {
  if (connection === "checking") return "Checking"
  if (connection === "connected") return "Connected"
  return "Disconnected"
}

export function settingsApiVersion(
  health: HealthResponse | null,
  connection: SettingsConnection,
): string {
  if (connection === "checking") return "Checking"
  return health?.api_version ?? "—"
}

export function settingsDeviceLabel(
  health: HealthResponse | null,
  connection: SettingsConnection,
): string {
  if (connection === "checking") return "Checking"
  if (!health) return "—"
  return health.compute.preferred_device === "mps" ? "MPS" : "CPU"
}

export function settingsInstalledLabel(
  value: boolean | undefined,
  connection: SettingsConnection,
): string {
  if (connection === "checking") return "Checking"
  if (value === undefined) return "—"
  return value ? "Installed" : "Not installed"
}

export function settingsAvailableLabel(
  value: boolean | undefined,
  connection: SettingsConnection,
): string {
  if (connection === "checking") return "Checking"
  if (value === undefined) return "—"
  return value ? "Available" : "Not available"
}
