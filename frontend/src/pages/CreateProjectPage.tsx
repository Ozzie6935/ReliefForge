import { useState } from "react"
import { useNavigate } from "react-router"
import { createProject } from "../api/projects.ts"
import { ProjectForm } from "../components/ProjectForm.tsx"
import { useHealth } from "../health/HealthContext.tsx"
import type { ProjectCreateRequest } from "../types/project.ts"
import { PROJECT_FORM_INTRO } from "../utils/projectFormLayout.ts"

export function CreateProjectPage() {
  const navigate = useNavigate()
  const { connection } = useHealth()
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function onSubmit(request: ProjectCreateRequest) {
    setPending(true)
    setError(null)
    try {
      const created = await createProject(request)
      navigate(`/projects/${created.id}`)
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : "Could not create the project.")
      setPending(false)
    }
  }

  return (
    <section className="new-project" aria-labelledby="new-project-heading">
      <h1 id="new-project-heading">New Project</h1>
      <p className="subtitle">{PROJECT_FORM_INTRO}</p>
      {connection === "disconnected" ? (
        <p className="status-notice">Connect the backend before creating a project.</p>
      ) : null}
      <section className="status-card">
        <ProjectForm
          submitLabel="Create Project"
          pending={pending}
          disabled={connection !== "connected"}
          error={error}
          cancelHref="/projects"
          cancelLabel="Back to Projects"
          onSubmit={(request) => {
            void onSubmit(request)
          }}
        />
      </section>
    </section>
  )
}
