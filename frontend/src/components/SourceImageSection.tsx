import { useEffect, useState, type FormEvent } from "react"
import { getSourceImageUrl, uploadSourceImage } from "../api/projects.ts"
import type { Project } from "../types/project.ts"
import { useUnits } from "../units/UnitsContext.tsx"
import {
  SOURCE_FORMAT_NOTE,
  SOURCE_REPLACE_WARNING,
  SOURCE_SCALE_NOTE,
  sourcePreviewState,
  sourceUploadEnabled,
} from "../utils/workspaceLayout.ts"
import { formatLength } from "../utils/units.ts"
import { artifactStatus } from "../utils/provenanceStatus.ts"
import { ProvenanceStatus } from "./ProvenanceStatus.tsx"
import { WorkspaceInspector } from "./WorkspaceInspector.tsx"

const MAX_SOURCE_IMAGE_BYTES = 50 * 1024 * 1024
const SOURCE_FORM_ID = "source-upload-form"

type SourceImageSectionProps = {
  project: Project
  onProjectChange: (project: Project) => void
}

export function SourceImageSection({ project, onProjectChange }: SourceImageSectionProps) {
  const { unit } = useUnits()
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageFailed, setImageFailed] = useState(false)
  const source = project.source
  const preview = sourcePreviewState({
    hasSource: source != null,
    imageLoaded,
    imageFailed,
  })

  useEffect(() => {
    setImageLoaded(false)
    setImageFailed(false)
  }, [source?.filename, source?.width_px, source?.height_px])

  async function onUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!file || uploading) return
    if (file.size > MAX_SOURCE_IMAGE_BYTES) {
      setUploadError("Image exceeds the 50 MB upload limit.")
      return
    }
    setUploading(true)
    setUploadError(null)
    setImageFailed(false)
    try {
      const updated = await uploadSourceImage(project.id, file)
      onProjectChange(updated)
      setFile(null)
    } catch (error: unknown) {
      setUploadError(error instanceof Error ? error.message : "Could not upload the image.")
    } finally {
      setUploading(false)
    }
  }

  const status =
    uploading
      ? "Uploading image."
      : preview === "empty"
        ? "No source image is saved for this project."
        : preview === "loading"
          ? "Loading source image."
          : preview === "failed"
            ? "The saved image could not be loaded from the backend."
            : "Source image is available."

  return (
    <section className="source-workspace is-split" aria-labelledby="source-heading">
      <div className="source-main">
        <h2 id="source-heading" className="visually-hidden">
          Source Image
        </h2>
        {uploadError ? <p className="status-notice">{uploadError}</p> : null}
        <div className="depth-preview source-preview">
          <figure>
            <figcaption>Source image</figcaption>
            <div className="depth-fit source-fit">
              {preview === "empty" ? (
                <p className="field-note">No source image is saved for this project.</p>
              ) : null}
              {preview === "loading" ? <p className="field-note">Loading source image.</p> : null}
              {uploading ? <p className="field-note">Uploading image.</p> : null}
              {preview === "failed" ? (
                <p className="status-notice">The saved image could not be loaded from the backend.</p>
              ) : null}
              {source && preview !== "failed" ? (
                <img
                  className={`depth-fit-image source-fit-image${preview === "loaded" ? "" : " is-pending"}`}
                  src={getSourceImageUrl(project.id)}
                  alt={source.original_filename}
                  onLoad={() => setImageLoaded(true)}
                  onError={() => setImageFailed(true)}
                />
              ) : null}
            </div>
          </figure>
        </div>
      </div>
      <WorkspaceInspector
        className="source-panel"
        label="Source image"
        footer={
          <div className="card-actions">
            <button type="submit" form={SOURCE_FORM_ID} disabled={!sourceUploadEnabled(file != null, uploading)}>
              {uploading ? "Uploading…" : "Upload Image"}
            </button>
          </div>
        }
      >
        <h2>Source Image</h2>
        <p className={`source-status is-${preview}${uploading ? " is-uploading" : ""}`}>{status}</p>
        {source ? (
          <ProvenanceStatus title="Source" status={artifactStatus(source, source.status)} />
        ) : (
          <ProvenanceStatus title="Source" status="missing" />
        )}
        <dl className="source-meta">
          <div>
            <dt>Original filename</dt>
            <dd>{source ? source.original_filename : "None"}</dd>
          </div>
          <div>
            <dt>Pixels</dt>
            <dd>{source ? `${source.width_px} × ${source.height_px} px` : "None"}</dd>
          </div>
          <div>
            <dt>Artwork size</dt>
            <dd>
              {formatLength(project.artwork.width_mm, unit)} × {formatLength(project.artwork.height_mm, unit)}
            </dd>
          </div>
        </dl>
        <p className="field-note">{SOURCE_FORMAT_NOTE}</p>
        <p className="field-note">{SOURCE_SCALE_NOTE}</p>
        <form id={SOURCE_FORM_ID} className="source-form" onSubmit={(event) => void onUpload(event)}>
          <label>
            {source ? "Replace Image" : "Choose Image"}
            <input
              type="file"
              accept="image/jpeg,image/png,.jpg,.jpeg,.png"
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null)
                setUploadError(null)
              }}
            />
          </label>
        </form>
        {source ? <p className="status-notice">{SOURCE_REPLACE_WARNING}</p> : null}
        {uploadError ? <p className="status-notice">{uploadError}</p> : null}
      </WorkspaceInspector>
    </section>
  )
}
