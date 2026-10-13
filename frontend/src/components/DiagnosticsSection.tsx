import { useEffect, useRef, useState } from "react"
import { ApiError } from "../api/client.ts"
import {
  getPhysicalDiagnostics,
  getPhysicalPreviewUrl,
  getPhysicalProfile,
} from "../api/projects.ts"
import type {
  PhysicalHeightDiagnostics,
  PhysicalHeightProfile,
  ProfileMode,
  ProfileSample,
} from "../types/project.ts"
import type { Project } from "../types/project.ts"
import { ProvenanceStatus } from "./ProvenanceStatus.tsx"
import { WorkspaceInspector } from "./WorkspaceInspector.tsx"
import {
  MISSING_PHYSICAL_MESSAGE,
  chartPoint,
  containedImageRect,
  defaultProfileSelection,
  diagnosticsArtifactStatus,
  diagnosticsTrustCopy,
  diagnosticsView,
  displayPointToPixel,
  formatMillimeters,
  formatPixel,
  histogramBars,
  histogramInputValid,
  nearestProfileSample,
  nudgeProfileSelection,
  pixelToDisplay,
  profileChartBounds,
  profilePathCommands,
  profileRequestAllowed,
  profileSelectionFromPointer,
  profileSelectionKey,
  selectionForMode,
  shouldApplyProfileResult,
  shouldFetchProfileOnPointer,
  showDiagnosticsFigures,
  type ProfileSelection,
} from "../utils/diagnosticsWorkspace.ts"

const CHART_PLOT = { left: 52, top: 16, width: 560, height: 150 }
const CHART_VIEW = { width: 640, height: 210 }
const HISTOGRAM_PLOT = { left: 36, top: 12, width: 250, height: 88 }
const HISTOGRAM_VIEW = { width: 300, height: 132 }

export function DiagnosticsSection({
  project,
  onOpenRelief,
}: {
  project: Project
  onOpenRelief: () => void
}) {
  const physical = project.relief?.physical ?? null
  const hasRecord = physical != null
  const [phase, setPhase] = useState<"idle" | "loading" | "ready" | "error">("idle")
  const [errorStatus, setErrorStatus] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [diagnostics, setDiagnostics] = useState<PhysicalHeightDiagnostics | null>(null)
  const [retry, setRetry] = useState(0)
  const [selection, setSelection] = useState<ProfileSelection>(() =>
    defaultProfileSelection(physical?.width_px ?? 1, physical?.height_px ?? 1),
  )
  const [committedKey, setCommittedKey] = useState<string | null>(null)
  const [profile, setProfile] = useState<PhysicalHeightProfile | null>(null)
  const [profileKey, setProfileKey] = useState<string | null>(null)
  const [profileLoading, setProfileLoading] = useState(false)
  const [profileError, setProfileError] = useState<string | null>(null)
  const [natural, setNatural] = useState({ width: 0, height: 0 })
  const [box, setBox] = useState({ width: 0, height: 0 })
  const [hover, setHover] = useState<ProfileSample | null>(null)
  const fitRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<"start" | "end" | "line" | null>(null)
  const selectionRef = useRef(selection)
  const requestRef = useRef(0)
  const primedRef = useRef<string | null>(null)
  const nudgedRef = useRef(false)
  selectionRef.current = selection

  const view = diagnosticsView({
    hasRecord,
    phase: hasRecord ? phase : "idle",
    errorStatus,
    errorMessage,
  })
  const status = diagnostics ? diagnosticsArtifactStatus(diagnostics.status) : null
  const figuresReady = showDiagnosticsFigures(view) && status !== "missing" && diagnostics != null
  const allowProfile = profileRequestAllowed(view) && status !== "missing"
  const widthPx = diagnostics?.width_px ?? physical?.width_px ?? 1
  const heightPx = diagnostics?.height_px ?? physical?.height_px ?? 1

  useEffect(() => {
    if (!hasRecord) return
    let cancelled = false
    setPhase("loading")
    setDiagnostics(null)
    setErrorMessage(null)
    setErrorStatus(null)
    setProfile(null)
    setProfileKey(null)
    setProfileError(null)
    setCommittedKey(null)
    getPhysicalDiagnostics(project.id)
      .then((loaded) => {
        if (cancelled) return
        setDiagnostics(loaded)
        setPhase("ready")
      })
      .catch((error: unknown) => {
        if (cancelled) return
        if (error instanceof ApiError) {
          setErrorStatus(error.status)
          setErrorMessage(error.message)
        } else {
          setErrorStatus(null)
          setErrorMessage("Unable to reach the backend.")
        }
        setPhase("error")
      })
    return () => {
      cancelled = true
    }
  }, [hasRecord, physical?.generated_at, project.id, retry])

  useEffect(() => {
    if (!figuresReady || !diagnostics) return
    const token = `${project.id}:${physical?.generated_at ?? ""}:${retry}:${diagnostics.width_px}x${diagnostics.height_px}`
    if (primedRef.current === token) return
    primedRef.current = token
    const next = defaultProfileSelection(diagnostics.width_px, diagnostics.height_px)
    selectionRef.current = next
    setSelection(next)
    setCommittedKey(profileSelectionKey(next))
  }, [diagnostics, figuresReady, physical?.generated_at, project.id, retry])

  useEffect(() => {
    if (!allowProfile || committedKey == null) return
    const requestId = requestRef.current + 1
    requestRef.current = requestId
    setProfileLoading(true)
    setProfileError(null)
    const requested = selectionRef.current
    getPhysicalProfile(project.id, requested)
      .then((loaded) => {
        if (!shouldApplyProfileResult(requestId, requestRef.current)) return
        setProfile(loaded)
        setProfileKey(profileSelectionKey(requested))
        setProfileLoading(false)
      })
      .catch((error: unknown) => {
        if (!shouldApplyProfileResult(requestId, requestRef.current)) return
        setProfileError(
          error instanceof ApiError ? error.message : "Unable to reach the backend.",
        )
        setProfileLoading(false)
      })
  }, [allowProfile, committedKey, project.id])

  useEffect(() => {
    const node = fitRef.current
    if (!node || !figuresReady) return
    const measure = () => setBox({ width: node.clientWidth, height: node.clientHeight })
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [figuresReady])

  if (!hasRecord || (view.kind === "ready" && status === "missing")) {
    return (
      <section className="diagnostics-workspace" aria-labelledby="diagnostics-empty-heading">
        <h2 id="diagnostics-empty-heading" className="visually-hidden">
          Diagnostics
        </h2>
        <p className="status-notice">{MISSING_PHYSICAL_MESSAGE}</p>
        <button type="button" className="button-secondary" onClick={onOpenRelief}>
          Open Relief
        </button>
      </section>
    )
  }

  const trust = status ? diagnosticsTrustCopy(status) : null
  const selectionKey = profileSelectionKey(selection)
  const profileIsPrevious = profile != null && (profileLoading || profileKey !== selectionKey)
  const rect = containedImageRect(box.width, box.height, natural.width, natural.height)
  const start = pixelToDisplay({
    x_px: selection.start_x_px,
    y_px: selection.start_y_px,
    boxWidth: box.width,
    boxHeight: box.height,
    imageWidth: natural.width,
    imageHeight: natural.height,
    arrayWidth: widthPx,
    arrayHeight: heightPx,
  })
  const end = pixelToDisplay({
    x_px: selection.end_x_px,
    y_px: selection.end_y_px,
    boxWidth: box.width,
    boxHeight: box.height,
    imageWidth: natural.width,
    imageHeight: natural.height,
    arrayWidth: widthPx,
    arrayHeight: heightPx,
  })

  function updateSelection(next: ProfileSelection, commit: boolean) {
    selectionRef.current = next
    setSelection(next)
    if (commit) setCommittedKey(profileSelectionKey(next))
  }

  function pointerPoint(event: React.PointerEvent) {
    const node = fitRef.current
    if (!node) return null
    const bounds = node.getBoundingClientRect()
    return displayPointToPixel({
      pointerX: event.clientX - bounds.left,
      pointerY: event.clientY - bounds.top,
      boxWidth: bounds.width,
      boxHeight: bounds.height,
      imageWidth: natural.width,
      imageHeight: natural.height,
      arrayWidth: widthPx,
      arrayHeight: heightPx,
    })
  }

  function onPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (!figuresReady) return
    const target = event.target instanceof Element ? event.target : null
    const named = target?.getAttribute("data-handle")
    const handle = named === "start" || named === "end" ? named : "line"
    const point = pointerPoint(event)
    if (!point && handle === "line") return
    dragRef.current = handle
    event.currentTarget.setPointerCapture(event.pointerId)
    if (!point) return
    updateSelection(
      profileSelectionFromPointer({
        selection: selectionRef.current,
        point,
        handle,
        widthPx,
        heightPx,
      }),
      shouldFetchProfileOnPointer("move"),
    )
  }

  function onPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!dragRef.current) return
    const point = pointerPoint(event)
    updateSelection(
      profileSelectionFromPointer({
        selection: selectionRef.current,
        point,
        handle: dragRef.current,
        widthPx,
        heightPx,
      }),
      shouldFetchProfileOnPointer("move"),
    )
  }

  function onPointerUp() {
    if (!dragRef.current) return
    dragRef.current = null
    if (shouldFetchProfileOnPointer("up")) {
      setCommittedKey(profileSelectionKey(selectionRef.current))
    }
  }

  function onPreviewKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const next = nudgeProfileSelection(selectionRef.current, event.key, widthPx, heightPx)
    if (!next) return
    event.preventDefault()
    nudgedRef.current = true
    updateSelection(next, false)
  }

  function onPreviewKeyUp() {
    if (!nudgedRef.current) return
    nudgedRef.current = false
    setCommittedKey(profileSelectionKey(selectionRef.current))
  }

  return (
    <section className="diagnostics-workspace is-split" aria-labelledby="diagnostics-heading">
      <div className="diagnostics-main">
        <h2 id="diagnostics-heading" className="visually-hidden">
          Diagnostics
        </h2>
        {view.kind === "loading" ? <p className="field-note">Checking physical height.</p> : null}
        {view.kind === "network" || view.kind === "invalid" || view.kind === "failed" || view.kind === "not-found" ? (
          <div className="diagnostics-message">
            <p className="status-notice" role="alert">
              {view.message}
            </p>
            <button type="button" className="button-secondary" onClick={() => setRetry((value) => value + 1)}>
              Retry
            </button>
          </div>
        ) : null}
        {figuresReady && physical ? (
          <>
            {trust ? <p className="status-notice">{trust}</p> : null}
            <figure className="diagnostics-preview">
              <div
                className="diagnostics-fit"
                ref={fitRef}
                tabIndex={0}
                role="application"
                aria-label="Cross-section line. Arrow keys move a horizontal or vertical line. Number fields in the inspector set the endpoints."
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
                onKeyDown={onPreviewKeyDown}
                onKeyUp={onPreviewKeyUp}
              >
                <img
                  className="diagnostics-fit-image"
                  src={getPhysicalPreviewUrl(project.id, physical.generated_at)}
                  alt="Saved physical height preview"
                  draggable={false}
                  onLoad={(event) => {
                    setNatural({
                      width: event.currentTarget.naturalWidth,
                      height: event.currentTarget.naturalHeight,
                    })
                  }}
                />
                {rect && start && end ? (
                  <svg
                    className="diagnostics-overlay"
                    viewBox={`0 0 ${box.width} ${box.height}`}
                    aria-hidden="true"
                  >
                    <line
                      x1={start.x}
                      y1={start.y}
                      x2={end.x}
                      y2={end.y}
                      className="diagnostics-line-halo"
                    />
                    <line
                      data-handle="line"
                      x1={start.x}
                      y1={start.y}
                      x2={end.x}
                      y2={end.y}
                      className="diagnostics-line"
                    />
                    <circle data-handle="start" cx={start.x} cy={start.y} r={7} className="diagnostics-handle" />
                    <circle data-handle="end" cx={end.x} cy={end.y} r={7} className="diagnostics-handle" />
                  </svg>
                ) : null}
              </div>
              <figcaption>
                Line {formatPixel(selection.start_x_px)}, {formatPixel(selection.start_y_px)} to{" "}
                {formatPixel(selection.end_x_px)}, {formatPixel(selection.end_y_px)}
                {profile && profileKey === selectionKey
                  ? ` · ${formatMillimeters(profile.summary.length_mm)}`
                  : ""}
              </figcaption>
            </figure>
            <ProfileChart
              profile={profile}
              previous={profileIsPrevious}
              loading={profileLoading}
              error={profileError}
              hover={hover}
              onHover={setHover}
            />
          </>
        ) : null}
      </div>
      <WorkspaceInspector
        className="diagnostics-panel"
        label="Diagnostics"
        footer={
          figuresReady ? (
            <ProfileControls
              selection={selection}
              widthPx={widthPx}
              heightPx={heightPx}
              onMode={(mode) => {
                updateSelection(
                  selectionForMode(mode, widthPx, heightPx, selectionRef.current),
                  true,
                )
              }}
              onChange={(next) => updateSelection(next, true)}
            />
          ) : (
            <p className="field-note">Profile controls appear after diagnostics load.</p>
          )
        }
      >
        {status ? <ProvenanceStatus title="Physical height" status={status} stage="physical" /> : null}
        {view.kind === "loading" ? <p className="field-note">Checking physical height.</p> : null}
        {diagnostics && figuresReady ? (
          <DiagnosticsDetails diagnostics={diagnostics} profile={profile} previous={profileIsPrevious} />
        ) : null}
        {view.kind !== "loading" && view.kind !== "ready" ? (
          <button type="button" className="button-secondary" onClick={() => setRetry((value) => value + 1)}>
            Retry
          </button>
        ) : null}
      </WorkspaceInspector>
    </section>
  )
}

function ProfileControls({
  selection,
  widthPx,
  heightPx,
  onMode,
  onChange,
}: {
  selection: ProfileSelection
  widthPx: number
  heightPx: number
  onMode: (mode: ProfileMode) => void
  onChange: (selection: ProfileSelection) => void
}) {
  return (
    <div className="diagnostics-controls">
      <div className="diagnostics-modes" role="radiogroup" aria-label="Profile line mode">
        {(
          [
            ["horizontal", "Horizontal"],
            ["vertical", "Vertical"],
            ["line", "Custom line"],
          ] as const
        ).map(([mode, label]) => (
          <label key={mode}>
            <input
              type="radio"
              name="profile-mode"
              value={mode}
              checked={selection.mode === mode}
              onChange={() => onMode(mode)}
            />
            {label}
          </label>
        ))}
      </div>
      {selection.mode === "horizontal" ? (
        <PixelField
          label="Y pixel"
          value={selection.y_px}
          max={heightPx - 1}
          onChange={(value) =>
            onChange(selectionForMode("horizontal", widthPx, heightPx, { ...selection, y_px: value }))
          }
        />
      ) : null}
      {selection.mode === "vertical" ? (
        <PixelField
          label="X pixel"
          value={selection.x_px}
          max={widthPx - 1}
          onChange={(value) =>
            onChange(selectionForMode("vertical", widthPx, heightPx, { ...selection, x_px: value }))
          }
        />
      ) : null}
      {selection.mode === "line" ? (
        <div className="diagnostics-endpoints">
          <PixelField
            label="Start X"
            value={selection.start_x_px}
            max={widthPx - 1}
            onChange={(value) => onChange({ ...selection, start_x_px: value })}
          />
          <PixelField
            label="Start Y"
            value={selection.start_y_px}
            max={heightPx - 1}
            onChange={(value) => onChange({ ...selection, start_y_px: value })}
          />
          <PixelField
            label="End X"
            value={selection.end_x_px}
            max={widthPx - 1}
            onChange={(value) => onChange({ ...selection, end_x_px: value })}
          />
          <PixelField
            label="End Y"
            value={selection.end_y_px}
            max={heightPx - 1}
            onChange={(value) => onChange({ ...selection, end_y_px: value })}
          />
        </div>
      ) : null}
    </div>
  )
}

function PixelField({
  label,
  value,
  max,
  onChange,
}: {
  label: string
  value: number
  max: number
  onChange: (value: number) => void
}) {
  const id = `profile-${label.toLowerCase().replace(/\s+/g, "-")}`
  return (
    <label htmlFor={id}>
      {label}
      <input
        id={id}
        type="number"
        min={0}
        max={max}
        step="any"
        value={Number.isFinite(value) ? value : 0}
        onChange={(event) => {
          const next = Number(event.target.value)
          if (!Number.isFinite(next)) return
          onChange(Math.min(max, Math.max(0, next)))
        }}
      />
    </label>
  )
}

function ProfileChart({
  profile,
  previous,
  loading,
  error,
  hover,
  onHover,
}: {
  profile: PhysicalHeightProfile | null
  previous: boolean
  loading: boolean
  error: string | null
  hover: ProfileSample | null
  onHover: (sample: ProfileSample | null) => void
}) {
  const bounds = profile ? profileChartBounds(profile.samples) : null
  const paths = profile && bounds ? profilePathCommands(profile.samples, CHART_PLOT) : []
  const summary = profile?.summary
  return (
    <section className="diagnostics-chart" aria-labelledby="profile-chart-heading">
      <h3 id="profile-chart-heading">Cross section</h3>
      {previous ? <p className="field-note">Previous profile</p> : null}
      {loading && !profile ? <p className="field-note">Checking profile.</p> : null}
      {error ? <p className="status-notice">{error}</p> : null}
      {profile && bounds ? (
        <svg
          className="diagnostics-chart-svg"
          viewBox={`0 0 ${CHART_VIEW.width} ${CHART_VIEW.height}`}
          role="img"
          aria-label={chartLabel(profile, previous)}
          onPointerMove={(event) => {
            const rect = event.currentTarget.getBoundingClientRect()
            const x = ((event.clientX - rect.left) / rect.width) * CHART_VIEW.width
            const span = bounds.maxDistance - bounds.minDistance || 1
            const distance = bounds.minDistance + ((x - CHART_PLOT.left) / CHART_PLOT.width) * span
            onHover(nearestProfileSample(profile.samples, distance))
          }}
          onPointerLeave={() => onHover(null)}
        >
          <line
            x1={CHART_PLOT.left}
            y1={CHART_PLOT.top + CHART_PLOT.height}
            x2={CHART_PLOT.left + CHART_PLOT.width}
            y2={CHART_PLOT.top + CHART_PLOT.height}
            className="diagnostics-axis"
          />
          <line
            x1={CHART_PLOT.left}
            y1={CHART_PLOT.top}
            x2={CHART_PLOT.left}
            y2={CHART_PLOT.top + CHART_PLOT.height}
            className="diagnostics-axis"
          />
          {paths.map((commands, index) => (
            <path key={index} d={commands} className="diagnostics-profile-path" />
          ))}
          {profilePolylineDots(profile.samples, bounds).map((dot) => (
            <circle key={`${dot.x}-${dot.y}`} cx={dot.x} cy={dot.y} r={2.5} className="diagnostics-profile-path" />
          ))}
          <text x={CHART_PLOT.left + CHART_PLOT.width / 2} y={204} textAnchor="middle" className="diagnostics-axis-label">
            Distance (mm)
          </text>
          <text x={16} y={CHART_PLOT.top + CHART_PLOT.height / 2} textAnchor="middle" className="diagnostics-axis-label" transform={`rotate(-90 16 ${CHART_PLOT.top + CHART_PLOT.height / 2})`}>
            Z (mm)
          </text>
        </svg>
      ) : null}
      <p className="field-note" aria-live="polite">
        {hover
          ? `${formatMillimeters(hover.distance_mm)} along the line, Z ${formatMillimeters(hover.height_mm)}`
          : "Move along the chart to read distance and Z."}
      </p>
      {summary ? (
        <dl className="diagnostics-facts">
          <div>
            <dt>Profile length</dt>
            <dd>{formatMillimeters(summary.length_mm)}</dd>
          </div>
          <div>
            <dt>Minimum height</dt>
            <dd>{formatMillimeters(summary.minimum_mm)}</dd>
          </div>
          <div>
            <dt>Maximum height</dt>
            <dd>{formatMillimeters(summary.maximum_mm)}</dd>
          </div>
          <div>
            <dt>Height range</dt>
            <dd>{formatMillimeters(summary.range_mm)}</dd>
          </div>
          <div>
            <dt>Maximum adjacent slope</dt>
            <dd>
              {summary.maximum_abs_slope_mm_per_mm == null
                ? "—"
                : `${formatMillimeters(summary.maximum_abs_slope_mm_per_mm).replace(" mm", "")} mm/mm`}
            </dd>
          </div>
          <div>
            <dt>Valid samples</dt>
            <dd>{summary.valid_count}</dd>
          </div>
          <div>
            <dt>Invalid samples</dt>
            <dd>{summary.invalid_count}</dd>
          </div>
        </dl>
      ) : null}
      {profile?.slope_note ? <p className="field-note">{profile.slope_note}</p> : null}
    </section>
  )
}

function profilePolylineDots(
  samples: ProfileSample[],
  bounds: NonNullable<ReturnType<typeof profileChartBounds>>,
): Array<{ x: number; y: number }> {
  const dots: Array<{ x: number; y: number }> = []
  samples.forEach((sample, index) => {
    if (sample.height_mm == null || !Number.isFinite(sample.height_mm)) return
    const previous = samples[index - 1]
    const next = samples[index + 1]
    const isolated =
      (previous == null || previous.height_mm == null) && (next == null || next.height_mm == null)
    if (!isolated && samples.length > 1) return
    dots.push(chartPoint(sample.distance_mm, sample.height_mm, bounds, CHART_PLOT))
  })
  return dots
}

function chartLabel(profile: PhysicalHeightProfile, previous: boolean): string {
  const prefix = previous ? "Previous profile. " : "Profile. "
  return `${prefix}Length ${formatMillimeters(profile.summary.length_mm)}. Z from ${formatMillimeters(profile.summary.minimum_mm)} to ${formatMillimeters(profile.summary.maximum_mm)}.`
}

function DiagnosticsDetails({
  diagnostics,
  profile,
  previous,
}: {
  diagnostics: PhysicalHeightDiagnostics
  profile: PhysicalHeightProfile | null
  previous: boolean
}) {
  const metrics = diagnostics.metrics
  const bars =
    diagnostics.histogram && histogramInputValid(diagnostics.histogram)
      ? histogramBars(diagnostics.histogram.edges_mm, diagnostics.histogram.counts, HISTOGRAM_PLOT)
      : null
  return (
    <>
      {diagnostics.validation.errors.map((error) => (
        <p key={error} className="status-notice">
          {error}
        </p>
      ))}
      {diagnostics.validation.warnings.map((warning) => (
        <p key={warning} className="field-note">
          {warning}
        </p>
      ))}
      {profile && !previous
        ? profile.warnings.map((warning) => (
            <p key={warning} className="field-note">
              {warning}
            </p>
          ))
        : null}
      <section className="inspector-section" aria-labelledby="observed-height-heading">
        <h2 id="observed-height-heading">Observed height</h2>
        {metrics ? (
          <dl className="diagnostics-facts">
            <Fact label="Minimum" value={formatMillimeters(metrics.minimum_mm)} />
            <Fact label="Maximum" value={formatMillimeters(metrics.maximum_mm)} />
            <Fact label="Mean" value={formatMillimeters(metrics.mean_mm)} />
            <Fact label="Median" value={formatMillimeters(metrics.median_mm)} />
            <Fact label="Standard deviation" value={formatMillimeters(metrics.standard_deviation_mm)} />
            <Fact label="P5" value={formatMillimeters(metrics.percentiles_mm.p5)} />
            <Fact label="P50" value={formatMillimeters(metrics.percentiles_mm.p50)} />
            <Fact label="P95" value={formatMillimeters(metrics.percentiles_mm.p95)} />
            <Fact label="Observed range" value={formatMillimeters(metrics.range_mm)} />
            <Fact label="Non-finite" value={String(metrics.non_finite_count)} />
            <Fact label="Below zero" value={String(metrics.below_zero_count)} />
            <Fact label="Above stock" value={String(metrics.above_stock_count)} />
          </dl>
        ) : (
          <p className="field-note">Observed height statistics are not available.</p>
        )}
      </section>
      <section className="inspector-section" aria-labelledby="recorded-mapping-heading">
        <h2 id="recorded-mapping-heading">Recorded mapping</h2>
        <dl className="diagnostics-facts">
          <Fact label="Stock thickness" value={formatMillimeters(diagnostics.mapping.stock_thickness_mm)} />
          <Fact label="Base thickness" value={formatMillimeters(diagnostics.mapping.base_thickness_mm)} />
          <Fact
            label="Maximum surface height"
            value={formatMillimeters(diagnostics.mapping.maximum_surface_height_mm)}
          />
          <Fact
            label="Maximum theoretical cut depth"
            value={formatMillimeters(diagnostics.mapping.maximum_theoretical_cut_depth_mm)}
          />
        </dl>
      </section>
      <section className="inspector-section" aria-labelledby="histogram-heading">
        <h2 id="histogram-heading">Histogram</h2>
        {bars ? (
          <svg
            className="diagnostics-histogram"
            viewBox={`0 0 ${HISTOGRAM_VIEW.width} ${HISTOGRAM_VIEW.height}`}
            role="img"
            aria-label="Physical height histogram in millimeters"
          >
            {bars.map((bar, index) => (
              <rect
                key={index}
                x={bar.x}
                y={bar.y}
                width={Math.max(bar.width, 0.5)}
                height={bar.height}
                className="diagnostics-bar"
              >
                <title>
                  {formatMillimeters(bar.startMm)} to {formatMillimeters(bar.endMm)}: {bar.count}
                </title>
              </rect>
            ))}
            <text x={HISTOGRAM_PLOT.left + HISTOGRAM_PLOT.width / 2} y={126} textAnchor="middle" className="diagnostics-axis-label">
              Height (mm)
            </text>
            <text x={12} y={56} className="diagnostics-axis-label" transform="rotate(-90 12 56)">
              Count
            </text>
          </svg>
        ) : (
          <p className="field-note">Histogram is not available.</p>
        )}
      </section>
    </>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}
