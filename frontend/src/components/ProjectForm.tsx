import { useEffect, useRef, useState, type FormEvent } from "react"
import { Link } from "react-router"
import type { Project, ProjectCreateRequest } from "../types/project.ts"
import { useUnits } from "../units/UnitsContext.tsx"
import {
  ARTWORK_GUIDANCE,
  DISPLAY_UNIT_NOTE,
  STOCK_GUIDANCE,
  claimProjectSubmit,
  defaultStockThicknessField,
  geometryAfterUnitChange,
  projectGeometryRequest,
  releaseProjectSubmit,
  stockThicknessNote,
  type GeometryField,
  type GeometryFields,
} from "../utils/projectFormLayout.ts"
import { formatLengthInput, formatThicknessInput, unitLabel, type DisplayUnit } from "../utils/units.ts"

type FieldKey = keyof GeometryFields

type FieldState = GeometryField

type Fields = GeometryFields

const FIELD_KEYS: FieldKey[] = [
  "stockWidth",
  "stockHeight",
  "stockThickness",
  "artworkWidth",
  "artworkHeight",
  "centerX",
  "centerY",
]

type ProjectFormProps = {
  project?: Project
  submitLabel: string
  pending: boolean
  disabled?: boolean
  error: string | null
  cancelHref?: string
  cancelLabel?: string
  onSubmit: (request: ProjectCreateRequest) => void
}

function blankField(): FieldState {
  return { text: "", mm: null, edited: false }
}

function emptyFields(): Fields {
  return {
    stockWidth: blankField(),
    stockHeight: blankField(),
    stockThickness: blankField(),
    artworkWidth: blankField(),
    artworkHeight: blankField(),
    centerX: blankField(),
    centerY: blankField(),
  }
}

function newProjectFields(unit: DisplayUnit): Fields {
  return {
    ...emptyFields(),
    stockThickness: defaultStockThicknessField(unit, false),
  }
}

function fieldsFromProject(project: Project, unit: DisplayUnit): Fields {
  const millimeters: Record<Exclude<FieldKey, "stockThickness">, number> = {
    stockWidth: project.stock.width_mm,
    stockHeight: project.stock.height_mm,
    artworkWidth: project.artwork.width_mm,
    artworkHeight: project.artwork.height_mm,
    centerX: project.artwork.center_x_mm,
    centerY: project.artwork.center_y_mm,
  }
  const fields = emptyFields()
  for (const key of FIELD_KEYS) {
    if (key === "stockThickness") continue
    fields[key] = {
      text: formatLengthInput(millimeters[key], unit),
      mm: millimeters[key],
      edited: false,
    }
  }
  const savedThickness = project.stock.thickness_mm
  fields.stockThickness =
    savedThickness == null
      ? defaultStockThicknessField(unit, true)
      : {
          text: formatThicknessInput(savedThickness, unit),
          mm: savedThickness,
          edited: false,
        }
  return fields
}

export function ProjectForm({
  project,
  submitLabel,
  pending,
  disabled = false,
  error,
  cancelHref,
  cancelLabel,
  onSubmit,
}: ProjectFormProps) {
  const { unit } = useUnits()
  const previousUnit = useRef(unit)
  const submitLock = useRef({ held: false })
  const [name, setName] = useState(project?.name ?? "")
  const [fields, setFields] = useState<Fields>(() =>
    project ? fieldsFromProject(project, unit) : newProjectFields(unit),
  )
  const [localError, setLocalError] = useState<string | null>(null)

  useEffect(() => {
    setName(project?.name ?? "")
    setFields(project ? fieldsFromProject(project, previousUnit.current) : newProjectFields(previousUnit.current))
  }, [project])

  useEffect(() => {
    if (previousUnit.current === unit) return
    const from = previousUnit.current
    previousUnit.current = unit
    setFields((current) => geometryAfterUnitChange(current, from, unit))
  }, [unit])

  useEffect(() => {
    if (!pending) releaseProjectSubmit(submitLock.current)
  }, [pending])

  function updateField(key: FieldKey, text: string) {
    setFields((current) => ({
      ...current,
      [key]: { ...current[key], text, edited: true, omitIfUnedited: false },
    }))
  }

  function onFormSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (pending || disabled || submitLock.current.held) return
    const result = projectGeometryRequest(name, fields, unit)
    if (!result.ok) {
      setLocalError(result.error)
      return
    }
    if (!claimProjectSubmit(submitLock.current)) return
    setLocalError(null)
    onSubmit(result.request)
  }

  const suffix = unitLabel(unit)

  return (
    <form className="project-form" onSubmit={onFormSubmit}>
      <p className="field-note">
        Sizes are {suffix === "in" ? "inches" : "millimeters"}. {DISPLAY_UNIT_NOTE}
      </p>
      <fieldset className="form-section">
        <legend>Project</legend>
        <label className="field-wide">
          Project Name
          <input value={name} onChange={(event) => setName(event.target.value)} required />
        </label>
      </fieldset>
      <fieldset className="form-section">
        <legend>Stock dimensions</legend>
        <p className="field-note">{STOCK_GUIDANCE}</p>
        <LengthField
          label="Stock Width"
          suffix={suffix}
          value={fields.stockWidth.text}
          onChange={(text) => updateField("stockWidth", text)}
        />
        <LengthField
          label="Stock Height"
          suffix={suffix}
          value={fields.stockHeight.text}
          onChange={(text) => updateField("stockHeight", text)}
        />
        <p className="field-note" id="stock-thickness-note">
          {stockThicknessNote({
            creating: project == null,
            savedThicknessMm: project?.stock.thickness_mm ?? null,
            edited: fields.stockThickness.edited,
          })}
        </p>
        <LengthField
          label="Stock Thickness"
          suffix={suffix}
          value={fields.stockThickness.text}
          onChange={(text) => updateField("stockThickness", text)}
          describedBy="stock-thickness-note"
          textEntry
        />
      </fieldset>
      <fieldset className="form-section">
        <legend>Artwork size and placement</legend>
        <p className="field-note">{ARTWORK_GUIDANCE}</p>
        <LengthField
          label="Artwork Width"
          suffix={suffix}
          value={fields.artworkWidth.text}
          onChange={(text) => updateField("artworkWidth", text)}
        />
        <LengthField
          label="Artwork Height"
          suffix={suffix}
          value={fields.artworkHeight.text}
          onChange={(text) => updateField("artworkHeight", text)}
        />
        <LengthField
          label="Artwork Center X"
          suffix={suffix}
          value={fields.centerX.text}
          onChange={(text) => updateField("centerX", text)}
          positive={false}
        />
        <LengthField
          label="Artwork Center Y"
          suffix={suffix}
          value={fields.centerY.text}
          onChange={(text) => updateField("centerY", text)}
          positive={false}
        />
      </fieldset>
      <div className="form-actions">
        <button type="submit" disabled={pending || disabled}>
          {pending ? "Saving…" : submitLabel}
        </button>
        {cancelHref && cancelLabel ? (
          <Link className="button-link button-secondary" to={cancelHref}>
            {cancelLabel}
          </Link>
        ) : null}
      </div>
      {localError || error ? <p className="status-notice">{localError ?? error}</p> : null}
    </form>
  )
}

function LengthField({
  label,
  suffix,
  value,
  onChange,
  positive = true,
  describedBy,
  textEntry = false,
}: {
  label: string
  suffix: string
  value: string
  onChange: (text: string) => void
  positive?: boolean
  describedBy?: string
  textEntry?: boolean
}) {
  return (
    <label className={describedBy ? "field-wide" : undefined}>
      {label} ({suffix})
      <input
        type={textEntry ? "text" : "number"}
        inputMode="decimal"
        min={positive && !textEntry ? "0" : undefined}
        step={textEntry ? undefined : "any"}
        value={value}
        aria-describedby={describedBy}
        onChange={(event) => onChange(event.target.value)}
        required
      />
    </label>
  )
}
