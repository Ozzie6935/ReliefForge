import type { Project } from "../types/project.ts"
import { formatTimestamp } from "./time.ts"
import { formatLength, type DisplayUnit } from "./units.ts"

export type ProjectLibraryEntry = {
  id: string
  name: string
  stock: string
  artwork: string
  source: string
  modified: string
}

export function projectLibraryEntries(
  projects: readonly Project[],
  unit: DisplayUnit,
): ProjectLibraryEntry[] {
  return projects.map((project) => ({
    id: project.id,
    name: project.name,
    stock: `${formatLength(project.stock.width_mm, unit)} × ${formatLength(project.stock.height_mm, unit)}`,
    artwork: `${formatLength(project.artwork.width_mm, unit)} × ${formatLength(project.artwork.height_mm, unit)}`,
    source: project.source
      ? `Available · ${project.source.width_px} × ${project.source.height_px} px`
      : "No source image",
    modified: formatTimestamp(project.updated_at) ?? "None",
  }))
}

export function projectDeleteConfirmed(typedName: string, projectName: string): boolean {
  return typedName === projectName
}

export function suggestedDuplicateName(name: string): string {
  const trimmed = name.trim()
  const suffix = " copy"
  const room = 200 - suffix.length
  return `${trimmed.slice(0, Math.max(0, room)).trimEnd()}${suffix}`
}

export function duplicateRequestBody(name: string): { name: string } | null {
  const trimmed = name.trim()
  if (trimmed.length < 1 || trimmed.length > 200) return null
  return { name: trimmed }
}

export function duplicateNavigationTarget(id: string | null | undefined): string | null {
  if (!id) return null
  return `/projects/${id}`
}
