import { useEffect, useRef, useState } from "react"
import { Link, useNavigate } from "react-router"
import { deleteProject, duplicateProject, listProjects } from "../api/projects.ts"
import { DeleteProjectDialog } from "../components/DeleteProjectDialog.tsx"
import { DuplicateProjectDialog } from "../components/DuplicateProjectDialog.tsx"
import { useHealth } from "../health/HealthContext.tsx"
import type { Project } from "../types/project.ts"
import { useUnits } from "../units/UnitsContext.tsx"
import { duplicateNavigationTarget, projectLibraryEntries } from "../utils/projectLibrary.ts"

export function ProjectsPage() {
  const { connection } = useHealth()
  const { unit } = useUnits()
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()
  const [pendingDelete, setPendingDelete] = useState<Project | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [pendingDuplicate, setPendingDuplicate] = useState<Project | null>(null)
  const [duplicateError, setDuplicateError] = useState<string | null>(null)
  const [duplicating, setDuplicating] = useState(false)
  const duplicatingRef = useRef(false)

  useEffect(() => {
    if (connection !== "connected") return
    let cancelled = false
    setLoading(true)
    listProjects()
      .then((loaded) => {
        if (cancelled) return
        setProjects(loaded)
        setError(null)
      })
      .catch((caught: unknown) => {
        if (cancelled) return
        setError(caught instanceof Error ? caught.message : "Could not load projects.")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [connection])

  async function confirmDuplicate(name: string) {
    if (!pendingDuplicate || duplicatingRef.current) return
    duplicatingRef.current = true
    setDuplicating(true)
    setDuplicateError(null)
    try {
      const created = await duplicateProject(pendingDuplicate.id, name)
      const target = duplicateNavigationTarget(created.id)
      if (!target || created.source_project_id !== pendingDuplicate.id) {
        throw new Error("The duplicate could not be opened.")
      }
      navigate(target)
    } catch (caught: unknown) {
      setDuplicateError(caught instanceof Error ? caught.message : "Could not duplicate the project.")
      duplicatingRef.current = false
      setDuplicating(false)
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await deleteProject(pendingDelete.id)
      setProjects((current) => current.filter((item) => item.id !== pendingDelete.id))
      setPendingDelete(null)
    } catch (caught: unknown) {
      setDeleteError(caught instanceof Error ? caught.message : "Could not delete the project.")
    } finally {
      setDeleting(false)
    }
  }

  const entries = projectLibraryEntries(projects, unit)

  return (
    <section className="projects-library" aria-labelledby="projects-heading">
      <h1 id="projects-heading">Projects</h1>
      <p className="subtitle">A Local AI-assisted CNC bas-relief creation</p>
      {connection === "checking" || loading ? <p className="field-note">Checking</p> : null}
      {connection === "disconnected" ? (
        <p className="status-notice">Connect the backend before opening a project.</p>
      ) : null}
      {error ? <p className="status-notice">{error}</p> : null}
      {connection === "connected" && !loading && !error && projects.length === 0 ? (
        <section className="status-card empty-state">
          <h2>No projects yet</h2>
          <p className="field-note">Create a project to start a local relief.</p>
          <Link className="button-link" to="/projects/new">
            Create Project
          </Link>
        </section>
      ) : null}
      {entries.length > 0 ? (
        <ul className="project-library">
          {entries.map((entry) => (
            <li key={entry.id} className="status-card project-card">
              <div className="project-summary">
                <h2>{entry.name}</h2>
                <dl>
                  <div className="status-row">
                    <dt>Stock</dt>
                    <dd>{entry.stock}</dd>
                  </div>
                  <div className="status-row">
                    <dt>Artwork</dt>
                    <dd>{entry.artwork}</dd>
                  </div>
                  <div className="status-row">
                    <dt>Source</dt>
                    <dd>{entry.source}</dd>
                  </div>
                  <div className="status-row">
                    <dt>Modified</dt>
                    <dd>{entry.modified}</dd>
                  </div>
                </dl>
              </div>
              <div className="card-actions">
                <Link className="button-link" to={`/projects/${entry.id}`} aria-label={`Open ${entry.name}`}>
                  Open
                </Link>
                <Link
                  className="button-link button-secondary"
                  to={`/projects/${entry.id}/edit`}
                  aria-label={`Edit ${entry.name}`}
                >
                  Edit
                </Link>
                <button
                  type="button"
                  className="button-secondary"
                  aria-label={`Duplicate ${entry.name}`}
                  onClick={() => {
                    setDuplicateError(null)
                    setPendingDuplicate(projects.find((project) => project.id === entry.id) ?? null)
                  }}
                >
                  Duplicate
                </button>
                <button
                  type="button"
                  className="button-danger"
                  aria-label={`Delete ${entry.name}`}
                  onClick={() => {
                    setDeleteError(null)
                    setPendingDelete(projects.find((project) => project.id === entry.id) ?? null)
                  }}
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
      {pendingDuplicate ? (
        <DuplicateProjectDialog
          projectName={pendingDuplicate.name}
          pending={duplicating}
          error={duplicateError}
          onCancel={() => {
            if (!duplicating) setPendingDuplicate(null)
          }}
          onConfirm={(name) => {
            void confirmDuplicate(name)
          }}
        />
      ) : null}
      {pendingDelete ? (
        <DeleteProjectDialog
          projectName={pendingDelete.name}
          pending={deleting}
          error={deleteError}
          onCancel={() => {
            if (!deleting) setPendingDelete(null)
          }}
          onConfirm={() => {
            void confirmDelete()
          }}
        />
      ) : null}
    </section>
  )
}
