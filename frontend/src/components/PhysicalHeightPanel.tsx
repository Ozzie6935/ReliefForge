import { useState } from "react"
import { generatePhysicalHeight } from "../api/projects.ts"
import type { Project } from "../types/project.ts"
import {
  PHYSICAL_INCOMPLETE_ERROR,
  PHYSICAL_ORDER_ERROR,
  commitPhysicalMillimeters,
  physicalActionLabel,
  physicalCanonicalMillimeters,
  physicalDisplayStatus,
  physicalDraftProblem,
  physicalInputs,
  physicalInputsAfterUnitChange,
  physicalStockNote,
} from "../utils/physicalHeight.ts"
import {
  artifactStatus,
  canGeneratePhysical,
  failedProcessingResult,
  physicalSettingsPending,
  recoveryGuidance,
  statusExplanation,
} from "../utils/provenanceStatus.ts"
import { ProvenanceStatus } from "./ProvenanceStatus.tsx"
import { formatLength, unitLabel } from "../utils/units.ts"
import { useUnits } from "../units/UnitsContext.tsx"

export function PhysicalHeightPanel({
  project,
  compositionBlocked,
  onProjectChange,
  onError,
}: {
  project: Project
  compositionBlocked: boolean
  onProjectChange: (project: Project) => void
  onError: (message: string | null) => void
}) {
  const { unit } = useUnits()
  const saved = project.relief?.physical ?? null
  const projectThickness = project.stock.thickness_mm ?? null
  const canonical = physicalCanonicalMillimeters(saved, projectThickness)
  const sourceKey = saved
    ? `physical|${saved.stock_thickness_mm}|${saved.base_thickness_mm}|${saved.maximum_surface_height_mm}|${saved.generated_at}`
    : `suggested|${projectThickness ?? "default"}`
  const [draft, setDraft] = useState(() => physicalInputs(saved, unit, projectThickness))
  const [draftUnit, setDraftUnit] = useState(unit)
  const [draftKey, setDraftKey] = useState(sourceKey)
  const [generating, setGenerating] = useState(false)
  if (draftKey !== sourceKey) {
    setDraft(physicalInputs(saved, unit, projectThickness))
    setDraftUnit(unit)
    setDraftKey(sourceKey)
  } else if (draftUnit !== unit) {
    setDraft(physicalInputsAfterUnitChange(draft, canonical, draftUnit, unit))
    setDraftUnit(unit)
  }

  const status = physicalDisplayStatus(saved)
  const reliefStatus = artifactStatus(project.relief, project.relief?.status)
  const suffix = unitLabel(unit)
  const values = commitPhysicalMillimeters(draft, canonical, unit)
  const problem = physicalDraftProblem(draft, canonical, unit)
  const pending = physicalSettingsPending(values, saved)
  const generateEnabled = canGeneratePhysical(reliefStatus) && !compositionBlocked
  const origin = physicalInputs(saved, unit, projectThickness)
  const stockNote = physicalStockNote(canonical.stockSource, draft.stock === origin.stock)

  async function generate() {
    if (!values || !generateEnabled) return
    setGenerating(true)
    onError(null)
    try {
      onProjectChange(await generatePhysicalHeight(project.id, values))
    } catch (caught) {
      const failure = failedProcessingResult({
        error: caught,
        fallback: "Physical height could not be saved.",
        project,
        draft,
      })
      onError(failure.message)
    } finally {
      setGenerating(false)
    }
  }

  return (
    <section className="inspector-section physical-height" aria-labelledby="physical-height-heading">
      <h2 id="physical-height-heading">Physical Height</h2>
      <ProvenanceStatus title="Physical Height" status={status} stage="physical" />
      {saved && status !== "current" && status !== "missing" ? (
        <p className="field-note">The preview is the saved historical output.</p>
      ) : null}
      {pending ? (
        <p className="status-notice">Pending changes. The preview does not include these unsaved thicknesses.</p>
      ) : null}
      <p className="field-note">{stockNote}</p>
      {problem === "order" ? <p className="status-notice">{PHYSICAL_ORDER_ERROR}</p> : null}
      {problem === "incomplete" ? <p className="status-notice">{PHYSICAL_INCOMPLETE_ERROR}</p> : null}
      <label>
        Stock Thickness ({suffix})
        <input
          value={draft.stock}
          inputMode="decimal"
          onChange={(event) => setDraft({ ...draft, stock: event.target.value, usingExample: saved == null })}
        />
      </label>
      <label>
        Base Thickness ({suffix})
        <input
          value={draft.base}
          inputMode="decimal"
          onChange={(event) => setDraft({ ...draft, base: event.target.value, usingExample: saved == null })}
        />
      </label>
      <label>
        Maximum Surface Height ({suffix})
        <input
          value={draft.peak}
          inputMode="decimal"
          onChange={(event) => setDraft({ ...draft, peak: event.target.value, usingExample: saved == null })}
        />
      </label>
      <div className="card-actions">
        <button
          type="button"
          onClick={() => void generate()}
          disabled={generating || !values || !generateEnabled}
        >
          {physicalActionLabel(saved)}
        </button>
      </div>
      {!generateEnabled ? (
        <p className="field-note">
          {recoveryGuidance("physical", reliefStatus) ??
            (reliefStatus === "stale"
              ? statusExplanation("stale")
              : "Apply Composition before generating physical height.")}
        </p>
      ) : null}
      {saved ? (
        <dl className="physical-stats">
          <div>
            <dt>Height range</dt>
            <dd>
              {formatLength(saved.height_min_mm, unit)} – {formatLength(saved.height_max_mm, unit)}
            </dd>
          </div>
          <div>
            <dt>Maximum cut depth</dt>
            <dd>{formatLength(saved.maximum_cut_depth_mm, unit)}</dd>
          </div>
          <div>
            <dt>Relief amplitude</dt>
            <dd>{formatLength(saved.relief_amplitude_mm, unit)}</dd>
          </div>
        </dl>
      ) : null}
    </section>
  )
}
