import { useEffect, useState, type ReactNode } from "react"
import {
  applyConditioning,
  generateDepth,
  getConditionedPreviewUrl,
  getDepthHistogram,
  getDepthPreviewUrl,
  getSourceImageUrl,
} from "../api/projects.ts"
import { useHealth } from "../health/HealthContext.tsx"
import type {
  DepthConditioningParameters,
  DepthHistogram,
  Project,
} from "../types/project.ts"
import { formatTimestamp } from "../utils/time.ts"
import { DepthHistogramChart } from "./DepthHistogram.tsx"
import { WorkspaceInspector } from "./WorkspaceInspector.tsx"
import {
  artifactStatus,
  canApplyConditioning,
  canRegenerateRaw,
  failedProcessingResult,
} from "../utils/provenanceStatus.ts"
import {
  DEPTH_VIEWS,
  RELATIVE_DEPTH_LABEL,
  conditioningIsPending,
  depthViewChange,
  type DepthView,
} from "../utils/workspaceLayout.ts"
import { ProvenanceStatus } from "./ProvenanceStatus.tsx"

const MODEL_LABELS: Record<string, string> = {
  "depth-anything-v2-small": "Depth Anything V2 Small",
}

const DEFAULT_PARAMETERS: DepthConditioningParameters = {
  far_percentile: 1,
  near_percentile: 99,
  gamma: 1,
  contrast: 1,
  smoothing: 0,
  edge_preservation: 75,
  invert: false,
}

type DepthSectionProps = {
  project: Project
  onProjectChange: (project: Project) => void
}

function deviceLabel(device: string): string {
  return device === "mps" ? "MPS" : "CPU"
}

function savedParameters(project: Project): DepthConditioningParameters {
  return project.depth?.conditioning?.parameters ?? DEFAULT_PARAMETERS
}

function formatNumber(value: number, digits: number): string {
  return value.toFixed(digits)
}

type SliderProps = {
  label: string
  hint: string
  min: number
  max: number
  step: number
  value: number
  display: string
  onChange: (value: number) => void
}

function SliderField({ label, hint, min, max, step, value, display, onChange }: SliderProps) {
  return (
    <label className="conditioning-slider">
      <span className="slider-heading">
        <span>{label}</span>
        <span className="slider-value">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="field-note">{hint}</span>
    </label>
  )
}

export function DepthSection({ project, onProjectChange }: DepthSectionProps) {
  const { health } = useHealth()
  const [generating, setGenerating] = useState(false)
  const [applying, setApplying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previewFailed, setPreviewFailed] = useState(false)
  const [conditionedFailed, setConditionedFailed] = useState(false)
  const [draft, setDraft] = useState<DepthConditioningParameters>(() => savedParameters(project))
  const [histogram, setHistogram] = useState<DepthHistogram | null>(null)
  const [view, setView] = useState<DepthView>("conditioned")
  const depth = project.depth
  const conditioning = depth?.conditioning
  const rawStatus = artifactStatus(depth, depth?.status)
  const conditionedStatus = artifactStatus(conditioning, conditioning?.status)
  const generated = formatTimestamp(depth?.generated_at)
  const conditionedAt = formatTimestamp(conditioning?.generated_at)
  const device = depth?.device ?? health?.compute.preferred_device

  useEffect(() => {
    setDraft(savedParameters(project))
    setConditionedFailed(false)
  }, [project.depth?.conditioning?.generated_at])

  useEffect(() => {
    if (!depth) {
      setHistogram(null)
      return
    }
    let cancelled = false
    getDepthHistogram(project.id)
      .then((data) => {
        if (!cancelled) setHistogram(data)
      })
      .catch(() => {
        if (!cancelled) setHistogram(null)
      })
    return () => {
      cancelled = true
    }
  }, [project.id, depth, depth?.generated_at, conditioning?.generated_at])

  function updateDraft(patch: Partial<DepthConditioningParameters>) {
    setDraft((current) => ({ ...current, ...patch }))
  }

  async function onGenerate() {
    if (generating || !canRegenerateRaw(project.source != null)) return
    setGenerating(true)
    setError(null)
    setPreviewFailed(false)
    try {
      const updated = await generateDepth(project.id)
      onProjectChange(updated)
    } catch (caught: unknown) {
      const failure = failedProcessingResult({
        error: caught,
        fallback: "Depth generation failed.",
        project,
        draft,
      })
      setError(failure.message)
    } finally {
      setGenerating(false)
    }
  }

  async function onApply() {
    if (applying || !canApplyConditioning(rawStatus)) return
    setApplying(true)
    setError(null)
    setConditionedFailed(false)
    try {
      const updated = await applyConditioning(project.id, draft)
      onProjectChange(updated)
    } catch (caught: unknown) {
      const failure = failedProcessingResult({
        error: caught,
        fallback: "Depth conditioning failed.",
        project,
        draft,
      })
      setError(failure.message)
    } finally {
      setApplying(false)
    }
  }

  const generateLabel = generating
    ? depth
      ? "Regenerating raw depth..."
      : "Generating depth..."
    : depth
      ? "Regenerate Raw Depth"
      : "Generate Depth"
  const pending = conditioningIsPending(draft, savedParameters(project))
  const preview = depthPreview({
    view,
    project,
    conditioningGeneratedAt: conditioning?.generated_at ?? null,
    rawGeneratedAt: depth?.generated_at ?? null,
    conditionedFailed,
    previewFailed,
    onConditionedError: () => setConditionedFailed(true),
    onRawError: () => setPreviewFailed(true),
  })

  function chooseView(next: DepthView) {
    const change = depthViewChange()
    if (change.generate || change.apply || change.writesArtifacts) return
    setView(next)
  }

  return (
    <section className="depth-workspace is-split" aria-labelledby="depth-heading">
      <div className="depth-main">
        <h2 id="depth-heading" className="visually-hidden">
          Depth
        </h2>
        <p className="visually-hidden">{RELATIVE_DEPTH_LABEL}</p>
        {error ? <p className="status-notice">{error}</p> : null}
        <div className="depth-preview">
          <div className="view-switch" role="radiogroup" aria-label="Depth view">
            {DEPTH_VIEWS.map((item) => (
              <button
                key={item.id}
                type="button"
                role="radio"
                aria-checked={view === item.id}
                onClick={() => chooseView(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
          <figure>
            <figcaption>{preview.caption}</figcaption>
            <div className="depth-fit">
              {preview.image ?? <p className="field-note">{preview.empty}</p>}
            </div>
          </figure>
        </div>
      </div>
      <WorkspaceInspector
        className="depth-panel"
        label="Depth conditioning"
        footer={
          <div className="card-actions">
            <button
              type="button"
              disabled={!canApplyConditioning(rawStatus) || applying}
              onClick={() => void onApply()}
            >
              {applying ? "Applying conditioning..." : "Apply Conditioning"}
            </button>
          </div>
        }
      >
        <h2 id="conditioning-heading">Depth Conditioning</h2>
        <ProvenanceStatus title="Raw Depth" status={rawStatus} stage="raw" />
        <ProvenanceStatus title="Conditioned Depth" status={conditionedStatus} stage="conditioned" />
        <p className="field-note">{RELATIVE_DEPTH_LABEL}</p>
        <p className="field-note">Each apply reads the saved raw depth. 0 is far and 1 is near.</p>
        {pending ? (
          <p className="status-notice">Conditioning changes are not applied yet. Saved depth status is unchanged.</p>
        ) : null}
        <div className="card-actions">
          <button
            type="button"
            disabled={!canRegenerateRaw(project.source != null) || generating}
            onClick={() => void onGenerate()}
          >
            {generateLabel}
          </button>
        </div>
        <div className="conditioning-panel">
          <SliderField
            label="Far Clip"
            hint="Percentile of far values. Not a physical depth."
            min={0}
            max={20}
            step={0.1}
            value={draft.far_percentile}
            display={`${formatNumber(draft.far_percentile, 1)}%`}
            onChange={(value) => updateDraft({ far_percentile: value })}
          />
          <SliderField
            label="Near Clip"
            hint="Percentile of near values. Not a physical depth."
            min={80}
            max={100}
            step={0.1}
            value={draft.near_percentile}
            display={`${formatNumber(draft.near_percentile, 1)}%`}
            onChange={(value) => updateDraft({ near_percentile: value })}
          />
          <SliderField
            label="Gamma"
            hint="1.0 leaves the depth unchanged."
            min={0.25}
            max={4}
            step={0.05}
            value={draft.gamma}
            display={formatNumber(draft.gamma, 2)}
            onChange={(value) => updateDraft({ gamma: value })}
          />
          <SliderField
            label="Contrast"
            hint="1.0 leaves the depth unchanged. Applied around 0.5."
            min={0.5}
            max={2}
            step={0.05}
            value={draft.contrast}
            display={formatNumber(draft.contrast, 2)}
            onChange={(value) => updateDraft({ contrast: value })}
          />
          <SliderField
            label="Smoothing"
            hint="0 skips smoothing."
            min={0}
            max={100}
            step={1}
            value={draft.smoothing}
            display={formatNumber(draft.smoothing, 0)}
            onChange={(value) => updateDraft({ smoothing: value })}
          />
          <SliderField
            label="Edge Preservation"
            hint="Higher values blend only similar depths."
            min={0}
            max={100}
            step={1}
            value={draft.edge_preservation}
            display={formatNumber(draft.edge_preservation, 0)}
            onChange={(value) => updateDraft({ edge_preservation: value })}
          />
          <label className="conditioning-invert">
            <input
              type="checkbox"
              checked={draft.invert}
              onChange={(event) => updateDraft({ invert: event.target.checked })}
            />
            Invert Depth
          </label>
          <div className="card-actions">
            <button type="button" className="button-secondary" onClick={() => setDraft(DEFAULT_PARAMETERS)}>
              Reset to Defaults
            </button>
          </div>
        </div>
        <section className="inspector-section" aria-label="Depth details">
          <h3 className="region-group-heading">Model</h3>
          <dl className="depth-meta">
            <div>
              <dt>Model</dt>
              <dd>{MODEL_LABELS[depth?.model_id ?? "depth-anything-v2-small"]}</dd>
            </div>
            <div>
              <dt>Compute device</dt>
              <dd>{device ? deviceLabel(device) : "Checking"}</dd>
            </div>
            {depth ? (
              <div>
                <dt>Resolution</dt>
                <dd>
                  {depth.width_px} × {depth.height_px} px
                </dd>
              </div>
            ) : null}
            {generated ? (
              <div>
                <dt>Raw generated</dt>
                <dd>{generated}</dd>
              </div>
            ) : null}
            {conditioning ? (
              <div>
                <dt>Conditioned range</dt>
                <dd>
                  {conditioning.range_min.toFixed(4)} – {conditioning.range_max.toFixed(4)}
                </dd>
              </div>
            ) : null}
            {conditionedAt ? (
              <div>
                <dt>Conditioned</dt>
                <dd>{conditionedAt}</dd>
              </div>
            ) : null}
          </dl>
          <h3 className="region-group-heading">Distributions</h3>
          <div className="histogram-row">
            <DepthHistogramChart
              title="Raw"
              note="Oriented raw depth. Larger is nearer."
              series={histogram?.raw ?? null}
              empty="Raw depth distribution appears after raw depth exists."
            />
            <DepthHistogramChart
              title="Conditioned"
              note="0 is far and 1 is near."
              series={histogram?.conditioned ?? null}
              empty="Conditioned distribution appears after conditioning is applied."
            />
          </div>
        </section>
      </WorkspaceInspector>
    </section>
  )
}

function depthPreview(input: {
  view: DepthView
  project: Project
  conditioningGeneratedAt: string | null
  rawGeneratedAt: string | null
  conditionedFailed: boolean
  previewFailed: boolean
  onConditionedError: () => void
  onRawError: () => void
}): { caption: string; empty: string; image: ReactNode } {
  if (input.view === "source") {
    return {
      caption: "Source image",
      empty: "Add a source image before generating depth.",
      image: input.project.source ? (
        <img className="depth-fit-image" src={getSourceImageUrl(input.project.id)} alt={input.project.source.original_filename} />
      ) : null,
    }
  }
  if (input.view === "raw") {
    return {
      caption: "Raw AI depth",
      empty: input.previewFailed
        ? "The depth preview could not be loaded from the backend."
        : "No raw depth has been generated for this project.",
      image:
        input.rawGeneratedAt && !input.previewFailed ? (
          <img
            className="depth-fit-image"
            src={getDepthPreviewUrl(input.project.id, input.rawGeneratedAt)}
            alt="Raw AI depth preview"
            onError={input.onRawError}
          />
        ) : null,
    }
  }
  return {
    caption: "Conditioned depth",
    empty: input.conditionedFailed
      ? "The conditioned preview could not be loaded from the backend."
      : "Apply conditioning to create a processed preview.",
    image:
      input.conditioningGeneratedAt && !input.conditionedFailed ? (
        <img
          className="depth-fit-image"
          src={getConditionedPreviewUrl(input.project.id, input.conditioningGeneratedAt)}
          alt="Conditioned depth preview"
          onError={input.onConditionedError}
        />
      ) : null,
  }
}
