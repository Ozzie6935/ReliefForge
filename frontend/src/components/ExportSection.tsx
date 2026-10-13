import { Fragment, useEffect, useRef, useState } from "react"
import { ApiError } from "../api/client.ts"
import { getExportReadiness, postStlExport } from "../api/projects.ts"
import type { Project } from "../types/project.ts"
import { ProvenanceStatus } from "./ProvenanceStatus.tsx"
import { WorkspaceInspector } from "./WorkspaceInspector.tsx"
import { useUnits } from "../units/UnitsContext.tsx"
import {
  ESTIMATE_NOTE,
  MISSING_PHYSICAL_MESSAGE,
  PLAN_NOT_MESH_NOTE,
  REDUCED_QUALITY_NOTE,
  STL_UNIT_NOTE,
  SUGGESTION_NOTE,
  Z_CONVENTION_NOTE,
  afterExportFailure,
  browserDownloadHost,
  decideExportClick,
  downloadAllowed,
  exportDownloadNote,
  geometryFacts,
  issueMessages,
  parseGridInput,
  placementExplanation,
  planValidationNotes,
  reducedGridError,
  resolutionDraftDirty,
  saveBlobDownload,
  selectionKey,
  shouldApplyReadiness,
  showsMissingPhysical,
  suggestReducedGrid,
  type ExportMode,
  type ExportReadiness,
  type ExportSelection,
} from "../utils/stlExport.ts"

function aborted(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError"
}

export function ExportSection({
  project,
  onOpenRelief,
}: {
  project: Project
  onOpenRelief: () => void
}) {
  const { unit } = useUnits()
  const [committed, setCommitted] = useState<ExportSelection>({ mode: "native" })
  const [draftMode, setDraftMode] = useState<ExportMode>("native")
  const [heightText, setHeightText] = useState("")
  const [widthText, setWidthText] = useState("")
  const [gridTouched, setGridTouched] = useState(false)
  const [gridError, setGridError] = useState<string | null>(null)
  const [report, setReport] = useState<ExportReadiness | null>(null)
  const [phase, setPhase] = useState<"loading" | "ready" | "error">("loading")
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [commitToken, setCommitToken] = useState(0)
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)
  const [exportNote, setExportNote] = useState<string | null>(null)
  const requestRef = useRef(0)
  const exportingRef = useRef(false)
  const committedRef = useRef(committed)
  const downloadAbort = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  committedRef.current = committed
  const committedKey = selectionKey(committed)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      downloadAbort.current?.abort()
    }
  }, [])

  useEffect(() => {
    setCommitted({ mode: "native" })
    setDraftMode("native")
    setHeightText("")
    setWidthText("")
    setGridTouched(false)
    setGridError(null)
    setExportError(null)
    setExportNote(null)
  }, [project.id])

  useEffect(() => {
    const controller = new AbortController()
    const requestId = requestRef.current + 1
    requestRef.current = requestId
    setPhase("loading")
    setErrorMessage(null)
    setReport(null)
    getExportReadiness(project.id, committedRef.current, controller.signal)
      .then((loaded) => {
        if (!shouldApplyReadiness(requestId, requestRef.current)) return
        setReport(loaded)
        setPhase("ready")
      })
      .catch((error: unknown) => {
        if (aborted(error) || !shouldApplyReadiness(requestId, requestRef.current)) return
        setErrorMessage(error instanceof ApiError ? error.message : "Unable to reach the backend.")
        setPhase("error")
      })
    return () => controller.abort()
  }, [project.id, committedKey, commitToken, refresh])

  useEffect(() => {
    if (gridTouched || report?.source_grid == null) return
    const suggestion = suggestReducedGrid({
      heightPx: report.source_grid.height_px,
      widthPx: report.source_grid.width_px,
    })
    if (!suggestion) return
    setHeightText(String(suggestion.heightPx))
    setWidthText(String(suggestion.widthPx))
  }, [gridTouched, report])

  const draftDirty = resolutionDraftDirty(draftMode, committed, heightText, widthText)
  const allowed = downloadAllowed({
    report,
    projectId: project.id,
    selection: committed,
    draftDirty,
    exporting,
    phase,
  })
  const missing = showsMissingPhysical(report)
  const facts = report ? geometryFacts(report, unit) : []
  const notes = report ? planValidationNotes(report) : []

  function chooseNative() {
    setDraftMode("native")
    setGridError(null)
    setExportNote(null)
    setCommitted({ mode: "native" })
    setCommitToken((value) => value + 1)
  }

  function chooseReduced() {
    setDraftMode("reduced")
    setExportNote(null)
  }

  function commitReduced() {
    const heightPx = parseGridInput(heightText)
    const widthPx = parseGridInput(widthText)
    if (heightPx == null || widthPx == null) {
      setGridError("Reduced export requires whole-number width and height of at least 2.")
      return
    }
    const source = report?.source_grid
      ? { heightPx: report.source_grid.height_px, widthPx: report.source_grid.width_px }
      : null
    const problem = reducedGridError(source, heightPx, widthPx)
    if (problem) {
      setGridError(problem)
      return
    }
    setGridError(null)
    setExportNote(null)
    setCommitted({ mode: "reduced", heightPx, widthPx })
    setCommitToken((value) => value + 1)
  }

  async function onDownload() {
    if (decideExportClick(exportingRef.current, allowed) === "ignored") return
    exportingRef.current = true
    setExporting(true)
    setExportError(null)
    setExportNote(null)
    const controller = new AbortController()
    downloadAbort.current = controller
    try {
      const result = await postStlExport(project.id, committedRef.current, controller.signal)
      if (!mounted.current) return
      saveBlobDownload(result.body, result.filename, browserDownloadHost())
      setExportNote(`The STL file was received and the download started. ${exportDownloadNote(result)}`)
    } catch (error: unknown) {
      if (aborted(error) || !mounted.current) return
      setExportError(error instanceof ApiError ? error.message : "Unable to reach the backend.")
      setRefresh((token) => afterExportFailure(token).refresh)
    } finally {
      exportingRef.current = false
      if (mounted.current) setExporting(false)
    }
  }

  return (
    <section className="export-workspace is-split" aria-labelledby="export-heading">
      <h2 id="export-heading" className="visually-hidden">
        Export
      </h2>
      <div className="export-main">
        {phase === "loading" ? <p className="field-note">Checking export readiness.</p> : null}
        {phase === "error" && errorMessage ? (
          <p className="status-notice" role="alert">
            {errorMessage}
          </p>
        ) : null}
        {phase === "error" ? (
          <button type="button" className="button-secondary" onClick={() => setRefresh((value) => value + 1)}>
            Retry
          </button>
        ) : null}
        {missing ? <p className="status-notice">{MISSING_PHYSICAL_MESSAGE}</p> : null}
        {missing ? (
          <button type="button" className="button-secondary" onClick={onOpenRelief}>
            Open Relief
          </button>
        ) : null}
        <fieldset className="export-resolution">
          <legend>Resolution</legend>
          <label>
            <input
              type="radio"
              name="export-resolution"
              value="native"
              checked={draftMode === "native"}
              onChange={chooseNative}
            />
            Native resolution
          </label>
          <p className="export-copy">Uses every saved physical-height sample.</p>
          <label>
            <input
              type="radio"
              name="export-resolution"
              value="reduced"
              checked={draftMode === "reduced"}
              onChange={chooseReduced}
            />
            Reduced resolution
          </label>
          {draftMode === "reduced" ? (
            <>
              <p className="export-copy">{REDUCED_QUALITY_NOTE}</p>
              <p className="field-note">{SUGGESTION_NOTE}</p>
              <div className="export-grid-fields">
                <label htmlFor="export-width-px">
                  Width (px)
                  <input
                    id="export-width-px"
                    inputMode="numeric"
                    value={widthText}
                    aria-invalid={gridError ? true : undefined}
                    aria-describedby={gridError ? "export-grid-error" : undefined}
                    onChange={(event) => {
                      setGridTouched(true)
                      setWidthText(event.target.value)
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault()
                        commitReduced()
                      }
                    }}
                  />
                </label>
                <label htmlFor="export-height-px">
                  Height (px)
                  <input
                    id="export-height-px"
                    inputMode="numeric"
                    value={heightText}
                    aria-invalid={gridError ? true : undefined}
                    aria-describedby={gridError ? "export-grid-error" : undefined}
                    onChange={(event) => {
                      setGridTouched(true)
                      setHeightText(event.target.value)
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault()
                        commitReduced()
                      }
                    }}
                  />
                </label>
              </div>
              {gridError ? (
                <p id="export-grid-error" className="status-notice" role="alert">
                  {gridError}
                </p>
              ) : null}
              <button type="button" className="button-secondary" onClick={commitReduced}>
                Check readiness
              </button>
            </>
          ) : null}
        </fieldset>
        {draftDirty && report ? (
          <p className="field-note">
            These figures match the last checked resolution. Check readiness to use a different grid.
          </p>
        ) : null}
        {report ? (
          <>
            <dl className="export-facts">
              {facts.map((fact) => (
                <Fragment key={fact.term}>
                  <dt>{fact.term}</dt>
                  <dd>{fact.detail}</dd>
                </Fragment>
              ))}
            </dl>
            <p className="export-copy">{placementExplanation(report, unit)}</p>
            <p className="export-copy">{STL_UNIT_NOTE}</p>
            <p className="export-copy">{Z_CONVENTION_NOTE}</p>
            <p className="field-note">{ESTIMATE_NOTE}</p>
          </>
        ) : null}
      </div>
      <WorkspaceInspector
        className="export-panel"
        label="Export readiness"
        footer={
          <div className="card-actions">
            <button type="button" onClick={() => void onDownload()} disabled={!allowed}>
              {exporting ? "Preparing STL…" : "Download STL"}
            </button>
            {exportError ? (
              <p className="status-notice" role="alert">
                {exportError}
              </p>
            ) : null}
            {exportNote ? (
              <p className="field-note" role="status">
                {exportNote}
              </p>
            ) : null}
          </div>
        }
      >
        <h2>Export</h2>
        {notes.map((note) => (
          <p key={note} className="export-copy">
            {note}
          </p>
        ))}
        {report ? (
          <ProvenanceStatus title="Physical height" status={report.provenance_status} />
        ) : (
          <p className="field-note">Provenance is shown after readiness is received.</p>
        )}
        {report ? <p className="export-copy">{report.coordinate_convention}</p> : null}
        {report ? <p className="export-copy">{report.resolution_policy}</p> : null}
        {report ? <p className="export-copy">{report.topology_note}</p> : null}
        <h3 className="region-group-heading">Blocking errors</h3>
        {report && report.blocking_errors.length > 0 ? (
          <ul className="export-list">
            {issueMessages(report.blocking_errors).map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        ) : (
          <p className="field-note">{report ? "None reported." : "Waiting for readiness."}</p>
        )}
        <h3 className="region-group-heading">Warnings</h3>
        {report && report.warnings.length > 0 ? (
          <ul className="export-list">
            {issueMessages(report.warnings).map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        ) : (
          <p className="field-note">{report ? "None reported." : "Waiting for readiness."}</p>
        )}
        <h3 className="region-group-heading">Corrective actions</h3>
        {report && report.corrective_actions.length > 0 ? (
          <ul className="export-list">
            {report.corrective_actions.map((action) => (
              <li key={action}>{action}</li>
            ))}
          </ul>
        ) : (
          <p className="field-note">{report ? "None reported." : "Waiting for readiness."}</p>
        )}
        <p className="field-note">{PLAN_NOT_MESH_NOTE}</p>
      </WorkspaceInspector>
    </section>
  )
}
