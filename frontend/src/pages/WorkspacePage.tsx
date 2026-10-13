import { lazy, Suspense, useEffect, useState } from "react"
import { useParams } from "react-router"
import { getProject } from "../api/projects.ts"
import { DepthSection } from "../components/DepthSection.tsx"
import { DiagnosticsSection } from "../components/DiagnosticsSection.tsx"
import { ExportSection } from "../components/ExportSection.tsx"
import { OverviewSection } from "../components/OverviewSection.tsx"
import { ReliefSection } from "../components/ReliefSection.tsx"
import { SourceImageSection } from "../components/SourceImageSection.tsx"
import type { Project } from "../types/project.ts"
import { PROJECT_WORKSPACE_LABEL } from "../utils/inspectorLayout.ts"

type WorkspaceSection = "overview" | "source" | "depth" | "relief" | "diagnostics" | "mesh" | "export"

const MeshPreviewSection = lazy(() => import("../components/MeshPreviewSection.tsx"))

export function WorkspacePage() {
  const { projectId } = useParams()
  const [project, setProject] = useState<Project | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [section, setSection] = useState<WorkspaceSection>("overview")

  useEffect(() => {
    if (!projectId) return
    let cancelled = false
    setProject(null)
    setError(null)
    getProject(projectId)
      .then((loaded) => {
        if (cancelled) return
        setProject(loaded)
      })
      .catch(() => {
        if (cancelled) return
        setError("Unable to open project.")
      })
    return () => {
      cancelled = true
    }
  }, [projectId])

  return (
    <>
      <header className="project-heading">
        <h1 title={project?.name ?? "Project"}>{project?.name ?? "Project"}</h1>
        <p className="subtitle project-kind">{PROJECT_WORKSPACE_LABEL}</p>
      </header>
      <nav className="workspace-nav" aria-label="Workspace">
        <button
          type="button"
          className={section === "overview" ? "active" : undefined}
          aria-current={section === "overview" ? "page" : undefined}
          onClick={() => setSection("overview")}
        >
          Overview
        </button>
        <button
          type="button"
          className={section === "source" ? "active" : undefined}
          aria-current={section === "source" ? "page" : undefined}
          onClick={() => setSection("source")}
        >
          Source
        </button>
        <button
          type="button"
          className={section === "depth" ? "active" : undefined}
          aria-current={section === "depth" ? "page" : undefined}
          onClick={() => setSection("depth")}
        >
          Depth
        </button>
        <button
          type="button"
          className={section === "relief" ? "active" : undefined}
          aria-current={section === "relief" ? "page" : undefined}
          onClick={() => setSection("relief")}
        >
          Relief
        </button>
        <button
          type="button"
          className={section === "diagnostics" ? "active" : undefined}
          aria-current={section === "diagnostics" ? "page" : undefined}
          onClick={() => setSection("diagnostics")}
        >
          Diagnostics
        </button>
        <button
          type="button"
          className={section === "mesh" ? "active" : undefined}
          aria-current={section === "mesh" ? "page" : undefined}
          onClick={() => setSection("mesh")}
        >
          3D
        </button>
        <button
          type="button"
          className={section === "export" ? "active" : undefined}
          aria-current={section === "export" ? "page" : undefined}
          onClick={() => setSection("export")}
        >
          Export
        </button>
      </nav>
      {error ? <p className="status-notice">{error}</p> : null}
      {!project && !error ? <p className="field-note">Checking</p> : null}
      {project && section === "overview" ? (
        <OverviewSection project={project} onOpen={setSection} />
      ) : null}
      {project && section === "source" ? (
        <SourceImageSection project={project} onProjectChange={setProject} />
      ) : null}
      {project && section === "depth" ? (
        <DepthSection project={project} onProjectChange={setProject} />
      ) : null}
      {project && section === "relief" ? (
        <ReliefSection project={project} onProjectChange={setProject} />
      ) : null}
      {project && section === "diagnostics" ? (
        <DiagnosticsSection project={project} onOpenRelief={() => setSection("relief")} />
      ) : null}
      {project && section === "mesh" ? (
        <Suspense fallback={<p className="field-note">Loading 3D preview.</p>}>
          <MeshPreviewSection project={project} onOpenRelief={() => setSection("relief")} />
        </Suspense>
      ) : null}
      {project && section === "export" ? (
        <ExportSection project={project} onOpenRelief={() => setSection("relief")} />
      ) : null}
    </>
  )
}
