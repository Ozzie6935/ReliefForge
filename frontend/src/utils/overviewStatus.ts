import type { ArtifactStatus, Project } from "../types/project.ts"
import { artifactStatus, statusLabel } from "./provenanceStatus.ts"
import { DEFAULT_STOCK_THICKNESS_MM } from "./projectFormLayout.ts"
import { formatTimestamp } from "./time.ts"
import { formatLength, formatThickness, type DisplayUnit } from "./units.ts"

export type OverviewWorkflowItem = {
  id: "source" | "raw-depth" | "conditioned" | "composed-relief" | "physical-height"
  label: string
  state: ArtifactStatus
}

export function overviewStatusLabel(state: ArtifactStatus): string {
  return statusLabel(state)
}

export function overviewWorkflow(project: Pick<Project, "source" | "depth" | "relief">): OverviewWorkflowItem[] {
  return [
    {
      id: "source",
      label: "Source",
      state: artifactStatus(project.source, project.source?.status),
    },
    {
      id: "raw-depth",
      label: "Raw Depth",
      state: artifactStatus(project.depth, project.depth?.status),
    },
    {
      id: "conditioned",
      label: "Conditioned Depth",
      state: artifactStatus(project.depth?.conditioning, project.depth?.conditioning?.status),
    },
    {
      id: "composed-relief",
      label: "Relief Composition",
      state: artifactStatus(project.relief, project.relief?.status),
    },
    {
      id: "physical-height",
      label: "Physical Height",
      state: artifactStatus(project.relief?.physical, project.relief?.physical?.status),
    },
  ]
}

export function overviewMeasurements(
  project: Pick<Project, "name" | "stock" | "artwork" | "source" | "created_at" | "updated_at">,
  unit: DisplayUnit,
) {
  return {
    name: project.name,
    stockWidth: formatLength(project.stock.width_mm, unit),
    stockHeight: formatLength(project.stock.height_mm, unit),
    stockThickness: formatThickness(project.stock.thickness_mm ?? DEFAULT_STOCK_THICKNESS_MM, unit),
    stockThicknessSource: project.stock.thickness_mm == null ? "Application default" : "Project thickness",
    artworkWidth: formatLength(project.artwork.width_mm, unit),
    artworkHeight: formatLength(project.artwork.height_mm, unit),
    centerX: formatLength(project.artwork.center_x_mm, unit),
    centerY: formatLength(project.artwork.center_y_mm, unit),
    sourceFilename: project.source?.original_filename ?? null,
    sourceResolution: project.source
      ? `${project.source.width_px} × ${project.source.height_px} px`
      : null,
    created: formatTimestamp(project.created_at),
    modified: formatTimestamp(project.updated_at),
  }
}
