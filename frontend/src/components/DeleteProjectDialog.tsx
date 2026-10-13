import { useState } from "react"
import { projectDeleteConfirmed } from "../utils/projectLibrary.ts"

type DeleteProjectDialogProps = {
  projectName: string
  pending: boolean
  error: string | null
  onCancel: () => void
  onConfirm: () => void
}

export function DeleteProjectDialog({
  projectName,
  pending,
  error,
  onCancel,
  onConfirm,
}: DeleteProjectDialogProps) {
  const [typedName, setTypedName] = useState("")
  const confirmed = projectDeleteConfirmed(typedName, projectName)

  return (
    <div className="modal-backdrop">
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-project-title"
      >
        <h2 id="delete-project-title">Delete "{projectName}"?</h2>
        <p>
          This permanently removes the local project and its artifacts. This cannot
          be undone.
        </p>
        <label>
          Type the project name to confirm
          <input
            value={typedName}
            onChange={(event) => setTypedName(event.target.value)}
            autoFocus
          />
        </label>
        {error ? <p className="status-notice">{error}</p> : null}
        <div className="modal-actions">
          <button type="button" className="button-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className="button-danger"
            disabled={!confirmed || pending}
            onClick={onConfirm}
          >
            {pending ? "Deleting…" : "Delete Project"}
          </button>
        </div>
      </div>
    </div>
  )
}
