import { useEffect, useState } from "react"
import { Link, useNavigate, useParams } from "react-router"
import { getProject, updateProject } from "../api/projects.ts"
import { ProjectForm } from "../components/ProjectForm.tsx"
import type { Project, ProjectCreateRequest } from "../types/project.ts"

export function EditProjectPage() {
  const { projectId } = useParams()
  const navigate = useNavigate()
  const [project, setProject] = useState<Project | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!projectId) return
    let cancelled = false
    getProject(projectId)
      .then((loaded) => {
        if (cancelled) return
        setProject(loaded)
        setLoadError(null)
      })
      .catch(() => {
        if (cancelled) return
        setLoadError("Unable to open project.")
      })
    return () => {
      cancelled = true
    }
  }, [projectId])

  async function onSubmit(request: ProjectCreateRequest) {
    if (!project) return
    setPending(true)
    setError(null)
    try {
      const updated = await updateProject(project.id, request)
      navigate(`/projects/${updated.id}`)
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : "Could not save the project.")
      setPending(false)
    }
  }

  return (
    <>
      <h1>Edit Project</h1>
      <p className="subtitle">The project identity stays the same when the name changes.</p>
      {loadError ? <p className="status-notice">{loadError}</p> : null}
      {!project && !loadError ? <p className="field-note">Checking</p> : null}
      {project ? (
        <section className="status-card">
          <ProjectForm
            project={project}
            submitLabel="Save Project"
            pending={pending}
            error={error}
            onSubmit={(request) => {
              void onSubmit(request)
            }}
          />
          <p className="field-note">
            <Link to={`/projects/${project.id}`}>Back to project</Link>
          </p>
        </section>
      ) : null}
    </>
  )
}
