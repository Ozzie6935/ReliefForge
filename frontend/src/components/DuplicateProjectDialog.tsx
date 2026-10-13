import { useState } from "react"
import { duplicateRequestBody, suggestedDuplicateName } from "../utils/projectLibrary.ts"

type DuplicateProjectDialogProps = {
  projectName: string
  pending: boolean
  error: string | null
  onCancel: () => void
  onConfirm: (name: string) => void
}

export function DuplicateProjectDialog({
  projectName,
  pending,
  error,
  onCancel,
  onConfirm,
}: DuplicateProjectDialogProps) {
  const [name, setName] = useState(() => suggestedDuplicateName(projectName))
  const body = duplicateRequestBody(name)

  return (
    <div className="modal-backdrop">
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="duplicate-project-title"
      >
        <h2 id="duplicate-project-title">Duplicate "{projectName}"?</h2>
        <p>The original project stays unchanged. Choose a name for the independent copy.</p>
        <label htmlFor="duplicate-project-name">
          New project name
          <input
            id="duplicate-project-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            autoFocus
            disabled={pending}
          />
        </label>
        {error ? <p className="status-notice">{error}</p> : null}
        <div className="modal-actions">
          <button type="button" className="button-secondary" onClick={onCancel} disabled={pending}>
            Cancel
          </button>
          <button
            type="button"
            disabled={body === null || pending}
            onClick={() => {
              if (body) onConfirm(body.name)
            }}
          >
            {pending ? "Duplicating…" : "Duplicate Project"}
          </button>
        </div>
      </div>
    </div>
  )
}
