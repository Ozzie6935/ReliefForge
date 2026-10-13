import { Link } from "react-router"
import type { ArtifactStatus, Project } from "../types/project.ts"
import { useUnits } from "../units/UnitsContext.tsx"
import {
  overviewMeasurements,
  overviewStatusLabel,
  overviewWorkflow,
} from "../utils/overviewStatus.ts"
import { statusExplanation } from "../utils/provenanceStatus.ts"

type OverviewSectionProps = {
  project: Project
  onOpen: (section: "source" | "depth" | "relief") => void
}

function statusClass(state: ArtifactStatus): string {
  if (state === "current") return "value-ok"
  if (state === "stale") return "value-bad"
  if (state === "unknown") return "value-unknown"
  return "value-wait"
}

export function OverviewSection({ project, onOpen }: OverviewSectionProps) {
  const { unit } = useUnits()
  const fields = overviewMeasurements(project, unit)
  const workflow = overviewWorkflow(project)

  return (
    <section className="overview-dashboard" aria-labelledby="overview-heading">
      <div className="overview-toolbar">
        <h2 id="overview-heading" className="section-title">
          Overview
        </h2>
        <div className="card-actions">
          <Link className="button-link button-secondary" to={`/projects/${project.id}/edit`}>
            Edit Project
          </Link>
          <button type="button" className="button-secondary" onClick={() => onOpen("source")}>
            Source
          </button>
          <button type="button" className="button-secondary" onClick={() => onOpen("depth")}>
            Depth
          </button>
          <button type="button" className="button-secondary" onClick={() => onOpen("relief")}>
            Relief
          </button>
        </div>
      </div>
      <div className="overview-grid">
        <article className="status-card">
          <h2 className="section-title">Project</h2>
          <dl>
            <div className="status-row">
              <dt>Name</dt>
              <dd>{fields.name}</dd>
            </div>
            <div className="status-row">
              <dt>Stock width</dt>
              <dd>{fields.stockWidth}</dd>
            </div>
            <div className="status-row">
              <dt>Stock height</dt>
              <dd>{fields.stockHeight}</dd>
            </div>
            <div className="status-row">
              <dt>Stock thickness</dt>
              <dd>
                {fields.stockThickness}
                <span className="field-note">{fields.stockThicknessSource}</span>
              </dd>
            </div>
            <div className="status-row">
              <dt>Artwork width</dt>
              <dd>{fields.artworkWidth}</dd>
            </div>
            <div className="status-row">
              <dt>Artwork height</dt>
              <dd>{fields.artworkHeight}</dd>
            </div>
            <div className="status-row">
              <dt>Artwork center X</dt>
              <dd>{fields.centerX}</dd>
            </div>
            <div className="status-row">
              <dt>Artwork center Y</dt>
              <dd>{fields.centerY}</dd>
            </div>
          </dl>
        </article>
        <article className="status-card">
          <h2 className="section-title">Source and dates</h2>
          <dl>
            <div className="status-row">
              <dt>Source filename</dt>
              <dd>{fields.sourceFilename ?? "No source image"}</dd>
            </div>
            <div className="status-row">
              <dt>Resolution</dt>
              <dd>{fields.sourceResolution ?? "None"}</dd>
            </div>
            <div className="status-row">
              <dt>Created</dt>
              <dd>{fields.created ?? "None"}</dd>
            </div>
            <div className="status-row">
              <dt>Modified</dt>
              <dd>{fields.modified ?? "None"}</dd>
            </div>
          </dl>
        </article>
      </div>
      <article className="status-card">
        <h2 className="section-title">Workflow</h2>
        <p className="field-note">Each step uses the provenance status returned by the API.</p>
        <dl>
          {workflow.map((item) => (
            <div className="status-row" key={item.id}>
              <dt>{item.label}</dt>
              <dd className={statusClass(item.state)}>
                <span className="provenance-label">{overviewStatusLabel(item.state)}</span>
                <span className="field-note">{statusExplanation(item.state)}</span>
              </dd>
            </div>
          ))}
        </dl>
      </article>
    </section>
  )
}
