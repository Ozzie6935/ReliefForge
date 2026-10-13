import { Fragment, useEffect, useRef, useState, type DragEvent, type KeyboardEvent, type MouseEvent, type PointerEvent } from "react"
import {
  applyMagicWand,
  applyMaskStroke,
  applyReliefComposition,
  createReliefRegion,
  deleteReliefRegion,
  discardRegionMaskEdit,
  discardSegmentationProposal,
  getCurrentSegmentationProposal,
  getReliefMaskUrl,
  getPhysicalPreviewUrl,
  getReliefPreviewUrl,
  getSegmentationProposalUrl,
  getSourceImageUrl,
  predictSegmentation,
  resetMaskProposal,
  restoreRefinementSession,
  saveRegionMaskEdit,
  startRefinementSession,
  startRegionMaskEdit,
  updateProposalDraft,
  type MagicWandPayload,
  type MaskStrokePayload,
} from "../api/projects.ts"
import type { Project, ReliefRegion, ReliefRole, SegmentationProposal } from "../types/project.ts"
import {
  clampClientToDisplayedImage,
  clientOffsetFromPixel,
  displayScale,
  imagePixelFromClient,
  type ImageFrame,
  type ImagePixel,
} from "../utils/imageCoordinates.ts"
import {
  MAX_ZOOM_PERCENT,
  MIN_NAVIGATION_DRAG_PX,
  clampViewScale,
  clampViewportPan,
  contentPlacement,
  panDelta,
  viewAfterResize,
  zoomAreaView,
  type SourceRect,
} from "../utils/viewport.ts"
import {
  DISCARD_UNSAVED_CHANGES,
  acceptedRegionInspectorActions,
  closeShouldConfirmDiscard,
  editSessionDirty,
  editorKind,
  generateMaskEnabled,
  helpTopics,
  initialEditorTool,
  keepEditing,
  maskEditorNote,
  resetMaskAction,
  resetMaskPrompt,
  toolAfterZoom,
} from "../utils/editorSession.ts"
import { imagePointerAction, promptsAfterMaskGeneration, type EditorTool } from "../utils/editorTools.ts"
import {
  appendCommand,
  historyEntries,
  moveCursor,
  sessionStart,
  type HistoryCommand,
  type SessionHistory,
} from "../utils/history.ts"
import { editorActionBar, historyRegion, inspectorMode } from "../utils/inspectorLayout.ts"
import { PhysicalHeightPanel } from "./PhysicalHeightPanel.tsx"
import { physicalDisplayStatus } from "../utils/physicalHeight.ts"
import {
  artifactStatus,
  canApplyComposition,
  canGeneratePhysical,
  compositionPreviewCaption,
  failedProcessingResult,
  physicalPreviewCaption,
  regionSettingsPending,
} from "../utils/provenanceStatus.ts"
import { ProvenanceStatus } from "./ProvenanceStatus.tsx"
import { useUnits } from "../units/UnitsContext.tsx"
import {
  backgroundFromRelief,
  backgroundSettingsPending,
  backgroundSettingsProblem,
  type BackgroundSettings,
} from "../utils/backgroundSuppression.ts"
import { DETAIL_SCALE_INITIAL_MM, detailSamplingGuidance, parseDetailScaleMm } from "../utils/detailScale.ts"
import { formatLength, formatLengthInput, unitLabel } from "../utils/units.ts"
import { WAND_TOLERANCE_DEFAULT, WAND_TOLERANCE_MAX, WAND_TOLERANCE_MIN } from "../utils/magicWand.ts"
import {
  ACCEPTED_REGIONS_HEADING,
  DRAFT_REGIONS_HEADING,
  NO_ACCEPTED_REGION_MESSAGE,
  NO_ACCEPTED_REGIONS_MESSAGE,
  NO_COMPOSED_RELIEF_MESSAGE,
  NO_DRAFT_REGIONS_MESSAGE,
  NO_PHYSICAL_HEIGHT_MESSAGE,
  RELIEF_VIEWS,
  acceptedMaskDisplayUrl,
  addRegionAvailability,
  continueExistingDraft,
  defaultReliefView,
  draftRegionList,
  initialSelectedRegionId,
  reliefRegionList,
  selectAcceptedRegion,
  selectedRegionIdAfterChange,
  switchReliefView,
  viewSwitchRequests,
  workspaceCanvas,
  type ReliefView,
} from "../utils/reliefWorkspace.ts"

const ROLES: Array<{ value: ReliefRole; label: string }> = [
  { value: "primary", label: "Primary" },
  { value: "secondary", label: "Secondary" },
  { value: "supporting", label: "Supporting" },
  { value: "background", label: "Background" },
  { value: "custom", label: "Custom" },
]

type PromptBox = {
  x0: number
  y0: number
  x1: number
  y1: number
}

type RegionDraft = {
  id: string
  name: string
  role: ReliefRole
  elevation: number
  depth_range: number
  feather: number
  order: number
  detail_amount: number
  detail_scale_mm: number | null
}

type ReliefSectionProps = {
  project: Project
  onProjectChange: (project: Project) => void
}

function draftsFrom(project: Project): RegionDraft[] {
  return (project.relief?.regions ?? [])
    .slice()
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id))
    .map((region) => ({
      id: region.id,
      name: region.name,
      role: region.role,
      elevation: region.elevation,
      depth_range: region.depth_range,
      feather: region.feather,
      order: region.order,
      detail_amount: region.detail_amount ?? 0,
      detail_scale_mm: region.detail_scale_mm ?? null,
    }))
}

function roleLabel(role: ReliefRole): string {
  return ROLES.find((item) => item.value === role)?.label ?? role
}

function frameFrom(image: HTMLImageElement): ImageFrame {
  const rect = image.getBoundingClientRect()
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height,
    naturalWidth: image.naturalWidth,
    naturalHeight: image.naturalHeight,
  }
}

function formatSigned(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`
}

function boxFromPoints(start: ImagePixel, end: ImagePixel): PromptBox {
  return {
    x0: Math.min(start.x, end.x),
    y0: Math.min(start.y, end.y),
    x1: Math.max(start.x, end.x),
    y1: Math.max(start.y, end.y),
  }
}

function boxHasArea(box: PromptBox | null): box is PromptBox {
  return box !== null && box.x1 - box.x0 >= 1 && box.y1 - box.y0 >= 1
}

const MIN_BOX_DISPLAY_PX = 5

function brushLimits(width: number, height: number): { min: number; max: number; initial: number } {
  const shortSide = Math.min(width, height)
  const max = Math.max(4, Math.round(shortSide * 0.08))
  const min = 1
  return {
    min,
    max,
    initial: Math.min(max, Math.max(min, Math.round(shortSide * 0.012))),
  }
}

function sourceDistance(start: ImagePixel, end: ImagePixel): number {
  return Math.hypot(end.x - start.x, end.y - start.y)
}

type DisplayPoint = {
  x: number
  y: number
}

type DisplayRect = {
  left: number
  top: number
  width: number
  height: number
}

type BoxDrag = {
  pointerId: number
  startClient: DisplayPoint
  previousBox: PromptBox | null
}

type ZoomDrag = {
  pointerId: number
  startClient: DisplayPoint
}

type BrushDrag = {
  pointerId: number
  points: ImagePixel[]
}

type PanDrag = {
  pointerId: number
  originX: number
  originY: number
  panX: number
  panY: number
}

type ViewState = {
  scale: number
  panX: number
  panY: number
}

type IconKind =
  | "area"
  | "fit"
  | "actual"
  | "in"
  | "out"
  | "hand"
  | "positive"
  | "negative"
  | "box"
  | "brush-add"
  | "brush-erase"
  | "wand-add"
  | "wand-subtract"
  | "info"

function ToolbarIcon({ kind }: { kind: IconKind }) {
  if (kind === "area") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3" y="3" width="11" height="9" fill="none" stroke="currentColor" strokeWidth="1.6" strokeDasharray="2.2 1.6" />
        <circle cx="15.2" cy="15.2" r="3.4" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M17.7 17.7 L21 21" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    )
  }
  if (kind === "fit") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 9 V4 H9 M15 4 H20 V9 M20 15 V20 H15 M9 20 H4 V15" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  }
  if (kind === "actual") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <text x="12" y="16" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700">
          1:1
        </text>
      </svg>
    )
  }
  if (kind === "in") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="10.5" cy="10.5" r="5.2" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M14.4 14.4 L20 20 M10.5 8 V13 M8 10.5 H13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    )
  }
  if (kind === "out") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="10.5" cy="10.5" r="5.2" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M14.4 14.4 L20 20 M8 10.5 H13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    )
  }
  if (kind === "hand") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M9 11.2V6.4a1.25 1.25 0 0 1 2.5 0V11M11.5 10.4V4.8a1.25 1.25 0 0 1 2.5 0V11M14 10.8V6.6a1.25 1.25 0 0 1 2.5 0V12.4M16.5 12.6V9a1.25 1.25 0 0 1 2.5 0V15c0 3-1.8 4.8-5 4.8h-2.2C8.2 19.8 6.6 18 6.6 15.2V10.6a1.25 1.25 0 0 1 2.5 0"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    )
  }
  if (kind === "positive" || kind === "negative") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="7.2" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path
          d={kind === "positive" ? "M12 8 V16 M8 12 H16" : "M8 12 H16"}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    )
  }
  if (kind === "box") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="5" y="6" width="14" height="12" fill="none" stroke="currentColor" strokeWidth="1.6" />
      </svg>
    )
  }
  if (kind === "brush-add" || kind === "brush-erase") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="9" cy="13" r="4.2" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path
          d={kind === "brush-add" ? "M16 6 V12 M13 9 H19" : "M13 9 H19"}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    )
  }
  if (kind === "wand-add" || kind === "wand-subtract") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M14 13 L20 19" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <path d="M8 3.5 L9.1 6.4 L12 7.5 L9.1 8.6 L8 11.5 L6.9 8.6 L4 7.5 L6.9 6.4 Z" fill="currentColor" />
        <path
          d={kind === "wand-add" ? "M16 4 V10 M13 7 H19" : "M13 7 H19"}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 10.5 V16 M12 8 V8.2" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

function ToolButton({
  label,
  title,
  kind,
  pressed,
  tone,
  disabled = false,
  onClick,
}: {
  label: string
  title: string
  kind: IconKind
  pressed?: boolean
  tone?: "add" | "subtract"
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={title}
      aria-pressed={pressed}
      className={tone ? `tone-${tone}` : undefined}
      disabled={disabled}
      onClick={onClick}
    >
      <ToolbarIcon kind={kind} />
    </button>
  )
}

export function ReliefSection({ project, onProjectChange }: ReliefSectionProps) {
  const { unit } = useUnits()
  const imageRef = useRef<HTMLImageElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const boxDrag = useRef<BoxDrag | null>(null)
  const zoomDrag = useRef<ZoomDrag | null>(null)
  const brushDrag = useRef<BrushDrag | null>(null)
  const panDrag = useRef<PanDrag | null>(null)
  const previousTool = useRef<EditorTool>("positive")
  const draftSave = useRef<number | null>(null)
  const sessionProposal = useRef<string | null>(null)
  const historyRef = useRef<SessionHistory>(sessionStart())
  const viewRef = useRef<ViewState>({ scale: 1, panX: 0, panY: 0 })
  const brushTouched = useRef(false)
  const [drafts, setDrafts] = useState<RegionDraft[]>(() => draftsFrom(project))
  const [background, setBackground] = useState<BackgroundSettings>(() => backgroundFromRelief(project.relief))
  const [scaleText, setScaleText] = useState<Record<string, string>>({})
  const [scaleUnit, setScaleUnit] = useState(unit)
  if (scaleUnit !== unit) {
    setScaleUnit(unit)
    setScaleText({})
  }
  const [adding, setAdding] = useState(false)
  const [editRegionId, setEditRegionId] = useState<string | null>(null)
  const [editBaseline, setEditBaseline] = useState<{ name: string; role: ReliefRole } | null>(null)
  const [confirmDiscard, setConfirmDiscard] = useState(false)
  const [name, setName] = useState("")
  const [role, setRole] = useState<ReliefRole>("primary")
  const [editorTool, setEditorTool] = useState<EditorTool>("positive")
  const [positivePoints, setPositivePoints] = useState<ImagePixel[]>([])
  const [negativePoints, setNegativePoints] = useState<ImagePixel[]>([])
  const [box, setBox] = useState<PromptBox | null>(null)
  const [draftRect, setDraftRect] = useState<DisplayRect | null>(null)
  const [zoomRect, setZoomRect] = useState<DisplayRect | null>(null)
  const [proposalId, setProposalId] = useState<string | null>(null)
  const [sessionReadyId, setSessionReadyId] = useState<string | null>(null)
  const [confirmReset, setConfirmReset] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [panning, setPanning] = useState(false)
  const [maskUrl, setMaskUrl] = useState<string | null>(null)
  const [maskRevision, setMaskRevision] = useState(0)
  const [history, setHistory] = useState<SessionHistory>(() => sessionStart())
  const [brushRadius, setBrushRadius] = useState(12)
  const [wandTolerance, setWandTolerance] = useState(WAND_TOLERANCE_DEFAULT)
  const [showHelp, setShowHelp] = useState(false)
  const [stageBox, setStageBox] = useState<{
    width: number
    height: number
    offsetX: number
    offsetY: number
  } | null>(null)
  const [wandSeed, setWandSeed] = useState<ImagePixel | null>(null)
  const [overlayOpacity, setOverlayOpacity] = useState(45)
  const [showMask, setShowMask] = useState(true)
  const [viewScale, setViewScale] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [pendingStroke, setPendingStroke] = useState<ImagePixel[] | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [workspaceView, setWorkspaceView] = useState<ReliefView>(() =>
    defaultReliefView(Boolean(project.relief?.generated_at)),
  )
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(() =>
    initialSelectedRegionId(project.relief?.regions ?? []),
  )
  const [draftProposal, setDraftProposal] = useState<SegmentationProposal | null | undefined>(undefined)
  const [layoutTick, setLayoutTick] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  historyRef.current = history

  const savedSignature = (project.relief?.regions ?? [])
    .map((region) => `${region.id}:${region.updated_at}:${region.order}`)
    .join("|")
  const savedGeneratedAt = project.relief?.generated_at ?? ""

  useEffect(() => {
    setDrafts(draftsFrom(project))
    setBackground(backgroundFromRelief(project.relief))
    setScaleText({})
  }, [savedSignature, savedGeneratedAt, project])

  useEffect(() => {
    setSelectedRegionId((current) => selectedRegionIdAfterChange(current, project.relief?.regions ?? []))
  }, [savedSignature, project.relief?.regions])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    const observer = new ResizeObserver(() => setLayoutTick((value) => value + 1))
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [adding, workspaceView, project.source?.filename])

  useEffect(() => {
    if (!proposalId) {
      setMaskUrl(null)
      return
    }
    const controller = new AbortController()
    let objectUrl: string | null = null
    fetch(getSegmentationProposalUrl(project.id, proposalId, maskRevision), { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load the proposed mask.")
        return response.blob()
      })
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob)
        setMaskUrl(objectUrl)
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return
        setError(caught instanceof Error ? caught.message : "Unable to load the proposed mask.")
      })
    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [proposalId, project.id, maskRevision])

  useEffect(() => {
    if (adding) return
    let stop = false
    getCurrentSegmentationProposal(project.id)
      .then((proposal) => {
        if (!stop) setDraftProposal(proposal)
      })
      .catch((caught: unknown) => {
        if (stop) return
        setDraftProposal(null)
        setError(caught instanceof Error ? caught.message : "Unable to load the draft region.")
      })
    return () => {
      stop = true
    }
  }, [adding, project.id])

  useEffect(() => {
    if (!adding || !proposalId || sessionProposal.current === proposalId) return
    const id = proposalId
    sessionProposal.current = id
    void startRefinementSession(project.id, id)
      .then(() => {
        setSessionReadyId((current) => (sessionProposal.current === id ? id : current))
      })
      .catch((caught: unknown) => {
        if (sessionProposal.current === id) sessionProposal.current = null
        setError(caught instanceof Error ? caught.message : "Unable to start mask editing.")
      })
  }, [adding, proposalId, project.id])

  useEffect(() => {
    const image = imageRef.current
    if (!adding || !image || image.naturalWidth === 0 || brushTouched.current) return
    setBrushRadius(brushLimits(image.naturalWidth, image.naturalHeight).initial)
  }, [adding, layoutTick])

  useEffect(() => {
    setStageBox(null)
  }, [adding, workspaceView])

  useEffect(() => {
    const viewport = viewportRef.current
    const image = imageRef.current
    if (!viewport || !image || image.naturalWidth === 0) return
    const placed = contentPlacement(
      viewport.clientWidth,
      viewport.clientHeight,
      image.naturalWidth,
      image.naturalHeight,
    )
    if (!placed) return
    const width = image.naturalWidth * placed.scale
    const height = image.naturalHeight * placed.scale
    setStageBox((current) =>
      current &&
      Math.abs(current.width - width) < 0.5 &&
      Math.abs(current.height - height) < 0.5 &&
      Math.abs(current.offsetX - placed.offsetX) < 0.5 &&
      Math.abs(current.offsetY - placed.offsetY) < 0.5
        ? current
        : { width, height, offsetX: placed.offsetX, offsetY: placed.offsetY },
    )
  }, [adding, workspaceView, layoutTick])

  useEffect(() => {
    if (!adding || !stageBox) return
    const viewport = viewportRef.current
    if (!viewport) return
    const current = viewRef.current
    const next = viewAfterResize({
      scale: current.scale,
      panX: current.panX,
      panY: current.panY,
      layoutWidth: stageBox.width,
      layoutHeight: stageBox.height,
      viewportWidth: viewport.clientWidth,
      viewportHeight: viewport.clientHeight,
    })
    setView(next.scale, next.panX, next.panY)
  }, [adding, stageBox])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!adding || !viewport) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const image = imageRef.current
      if (!image || image.offsetWidth === 0 || image.naturalWidth === 0) return
      const currentPercent = (viewRef.current.scale * image.offsetWidth * 100) / image.naturalWidth
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15
      zoomAt(event.clientX, event.clientY, currentPercent * factor)
    }
    viewport.addEventListener("wheel", onWheel, { passive: false })
    return () => viewport.removeEventListener("wheel", onWheel)
  }, [adding])

  const conditioned = project.depth?.conditioning
  const relief = project.relief
  const conditionedStatus = artifactStatus(conditioned, conditioned?.status)
  const reliefStatus = artifactStatus(relief, relief?.status)
  const canGenerate = positivePoints.length > 0 || boxHasArea(box)
  const sessionKind = editorKind(editRegionId)
  const maskActions = editorActionBar(sessionKind)
  const [editMaskLabel, deleteRegionLabel] = acceptedRegionInspectorActions()
  const generatedAt = relief?.generated_at ?? null
  function scaleProblem(draft: RegionDraft): string | null {
    if (draft.detail_scale_mm == null) return null
    const text = scaleText[draft.id]
    if (text === undefined) return null
    if (parseDetailScaleMm(text, unit) == null) {
      return "Enter a feature width from 0.25 mm to 1.00 mm."
    }
    return null
  }
  const settingsPending =
    regionSettingsPending(drafts, relief?.regions ?? []) ||
    backgroundSettingsPending(background, backgroundFromRelief(relief)) ||
    drafts.some((draft) => scaleProblem(draft) != null)
  const physicalStatus = physicalDisplayStatus(relief?.physical)
  const canvas = workspaceCanvas({
    editing: adding,
    view: workspaceView,
    selectedRegionId,
    generatedAt,
    physicalGeneratedAt: relief?.physical?.generated_at ?? null,
    physicalStatus,
  })
  const selectedSaved = relief?.regions.find((region) => region.id === selectedRegionId) ?? null

  function resetPrompts() {
    boxDrag.current = null
    zoomDrag.current = null
    brushDrag.current = null
    panDrag.current = null
    setDraftRect(null)
    setZoomRect(null)
    setPendingStroke(null)
    setCursor(null)
    setPositivePoints([])
    setNegativePoints([])
    setBox(null)
    setProposalId(null)
    setHistory(sessionStart())
    setWandSeed(null)
    setShowHelp(false)
    setEditorTool("positive")
    previousTool.current = "positive"
    if (draftSave.current !== null) {
      window.clearTimeout(draftSave.current)
      draftSave.current = null
    }
  }

  function setView(scale: number, panX: number, panY: number) {
    const image = imageRef.current
    const viewport = viewportRef.current
    const next =
      image && viewport && image.offsetWidth > 0
        ? clampViewportPan({
            panX,
            panY,
            scale,
            layoutWidth: image.offsetWidth,
            layoutHeight: image.offsetHeight,
            viewportWidth: viewport.clientWidth,
            viewportHeight: viewport.clientHeight,
          })
        : { panX, panY }
    viewRef.current = { scale, panX: next.panX, panY: next.panY }
    setViewScale(scale)
    setPan({ x: next.panX, y: next.panY })
  }

  function panBy(direction: "up" | "down" | "left" | "right") {
    const image = imageRef.current
    const viewport = viewportRef.current
    if (!image || !viewport) return
    const step = panDelta(direction, viewport.clientWidth, viewport.clientHeight)
    const current = viewRef.current
    setView(current.scale, current.panX + step.x, current.panY + step.y)
  }

  function rememberTool(tool: EditorTool) {
    if (tool !== "zoom-area") previousTool.current = tool
    setEditorTool(tool)
  }

  function finishZoomArea() {
    const next = toolAfterZoom(sessionKind, previousTool.current)
    previousTool.current = next as EditorTool
    setEditorTool(next as EditorTool)
  }

  function saveDraft(nextName: string, nextRole: ReliefRole) {
    if (!proposalId) return
    if (draftSave.current !== null) window.clearTimeout(draftSave.current)
    draftSave.current = window.setTimeout(() => {
      draftSave.current = null
      void updateProposalDraft(project.id, proposalId, { name: nextName.trim(), role: nextRole }).catch(
        (caught: unknown) => {
          setError(caught instanceof Error ? caught.message : "Unable to save the region name.")
        },
      )
    }, 300)
  }

  function zoomAt(clientX: number, clientY: number, percent: number) {
    const image = imageRef.current
    const viewport = viewportRef.current
    if (!image || !viewport || image.offsetWidth === 0 || image.naturalWidth === 0) return
    const contentScale = image.offsetWidth / image.naturalWidth
    const requestedScale = (percent / 100) / contentScale
    const nextScale = clampViewScale(requestedScale, contentScale)
    const bounds = viewport.getBoundingClientRect()
    const pointerX = clientX - bounds.left - viewport.clientLeft
    const pointerY = clientY - bounds.top - viewport.clientTop
    const current = viewRef.current
    const stageX = (pointerX - current.panX) / current.scale
    const stageY = (pointerY - current.panY) / current.scale
    setView(nextScale, pointerX - stageX * nextScale, pointerY - stageY * nextScale)
  }

  function fitView() {
    setView(1, 0, 0)
  }

  function zoomToActualSize() {
    const image = imageRef.current
    const viewport = viewportRef.current
    if (!image || !viewport || image.offsetWidth === 0 || image.naturalWidth === 0) return
    const nextScale = image.naturalWidth / image.offsetWidth
    setView(
      nextScale,
      (viewport.clientWidth - image.offsetWidth * nextScale) / 2,
      (viewport.clientHeight - image.offsetHeight * nextScale) / 2,
    )
  }

  function zoomBy(factor: number) {
    const image = imageRef.current
    const viewport = viewportRef.current
    if (!image || !viewport || image.offsetWidth === 0 || image.naturalWidth === 0) return
    const bounds = viewport.getBoundingClientRect()
    const currentPercent = (viewRef.current.scale * image.offsetWidth * 100) / image.naturalWidth
    zoomAt(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2, currentPercent * factor)
  }

  function layoutFrame(image: HTMLImageElement): ImageFrame {
    return {
      left: 0,
      top: 0,
      width: stageBox?.width || image.offsetWidth,
      height: stageBox?.height || image.offsetHeight,
      naturalWidth: image.naturalWidth,
      naturalHeight: image.naturalHeight,
    }
  }

  function localFromClient(image: HTMLImageElement, clientX: number, clientY: number): DisplayPoint | null {
    const clientFrame = frameFrom(image)
    const clamped = clampClientToDisplayedImage(clientX, clientY, clientFrame)
    if (!clamped) return null
    const pixel = imagePixelFromClient(clamped.x, clamped.y, clientFrame)
    if (!pixel) return null
    return clientOffsetFromPixel(pixel, layoutFrame(image))
  }

  function displayRect(image: HTMLImageElement, start: DisplayPoint, end: DisplayPoint): DisplayRect | null {
    const origin = localFromClient(image, start.x, start.y)
    const current = localFromClient(image, end.x, end.y)
    if (!origin || !current) return null
    return {
      left: Math.min(origin.x, current.x),
      top: Math.min(origin.y, current.y),
      width: Math.abs(current.x - origin.x),
      height: Math.abs(current.y - origin.y),
    }
  }

  function releaseBoxPointer(event: PointerEvent<HTMLImageElement>) {
    try {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }
    } catch {
      // The pointer was already released.
    }
  }

  function pixelFromPointer(event: { clientX: number; clientY: number }): ImagePixel | null {
    const image = imageRef.current
    if (!image) return null
    return imagePixelFromClient(event.clientX, event.clientY, frameFrom(image))
  }

  function clampedPixel(event: { clientX: number; clientY: number }): ImagePixel | null {
    const image = imageRef.current
    if (!image) return null
    const frame = frameFrom(image)
    const clamped = clampClientToDisplayedImage(event.clientX, event.clientY, frame)
    if (!clamped) return null
    return imagePixelFromClient(clamped.x, clamped.y, frame)
  }

  function beginAdd() {
    if (!addRegionAvailability(draftProposal).enabled) return
    setEditRegionId(null)
    setEditBaseline(null)
    setConfirmDiscard(false)
    setView(1, 0, 0)
    setAdding(true)
    setShowMask(true)
    setName("")
    setRole("primary")
    setError(null)
    setPendingDelete(null)
    resetPrompts()
  }

  function continueDraft(proposal: SegmentationProposal) {
    const resumed = continueExistingDraft(proposal.proposal_id)
    if (
      resumed.createsProposal ||
      resumed.callsSam ||
      resumed.replacesWorkingMask ||
      resumed.replacesAiMask ||
      resumed.acceptsRegion ||
      resumed.changesAcceptedRegions ||
      resumed.recomputesComposition
    ) {
      return
    }
    setEditRegionId(null)
    setEditBaseline(null)
    setConfirmDiscard(false)
    setView(1, 0, 0)
    setAdding(true)
    setShowMask(true)
    setName(proposal.name ?? "")
    setRole(proposal.role ?? "primary")
    setError(null)
    setPendingDelete(null)
    resetPrompts()
    setProposalId(resumed.proposalId)
  }

  function closeEditor() {
    sessionProposal.current = null
    setSessionReadyId(null)
    setPanning(false)
    setConfirmReset(false)
    setConfirmDelete(false)
    setConfirmDiscard(false)
    setShowHelp(false)
    setEditRegionId(null)
    setEditBaseline(null)
    setAdding(false)
    resetPrompts()
    setError(null)
  }

  async function beginEdit(region: ReliefRegion) {
    setBusy(true)
    setError(null)
    setPendingDelete(null)
    try {
      const proposal = await startRegionMaskEdit(project.id, region.id)
      setView(1, 0, 0)
      setShowMask(true)
      const tool = initialEditorTool("accepted-region")
      setEditRegionId(region.id)
      setEditBaseline({ name: region.name, role: region.role })
      setName(region.name)
      setRole(region.role)
      setAdding(true)
      resetPrompts()
      setEditorTool(tool)
      previousTool.current = tool
      setProposalId(proposal.proposal_id)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to open the saved mask.")
    } finally {
      setBusy(false)
    }
  }

  function requestClose() {
    const dirty = editBaseline
      ? editSessionDirty(editBaseline, { name, role, historyCursor: history.cursor })
      : false
    if (closeShouldConfirmDiscard(sessionKind, dirty)) {
      setConfirmDiscard(true)
      return
    }
    if (editRegionId) {
      void discardEdit()
      return
    }
    closeEditor()
  }

  function cancelDiscard() {
    const session = keepEditing({ name, role, historyCursor: history.cursor })
    setName(session.name)
    setRole(session.role)
    setConfirmDiscard(false)
  }

  async function discardEdit() {
    const regionId = editRegionId
    const currentProposal = proposalId
    setConfirmDiscard(false)
    if (!regionId || !currentProposal) {
      closeEditor()
      return
    }
    setBusy(true)
    setError(null)
    try {
      await discardRegionMaskEdit(project.id, regionId, currentProposal)
      closeEditor()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to discard the mask edit.")
    } finally {
      setBusy(false)
    }
  }

  async function deleteProposal() {
    const currentProposal = proposalId
    if (!currentProposal) return
    setBusy(true)
    setError(null)
    try {
      await discardSegmentationProposal(project.id, currentProposal)
      setConfirmDelete(false)
      closeEditor()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to delete the proposal.")
    } finally {
      setBusy(false)
    }
  }

  function onImageClick(event: MouseEvent<HTMLImageElement>) {
    if (!adding || busy) return
    const action = imagePointerAction(editorTool)
    if (sessionKind === "accepted-region" && (action === "sam-point" || action === "sam-box")) return
    if (action === "wand-add" || action === "wand-subtract") {
      if (!proposalId || sessionReadyId !== proposalId) return
      const pixel = pixelFromPointer(event)
      if (!pixel) return
      void commitWand(action === "wand-add" ? "add" : "subtract", pixel)
      return
    }
    if (action !== "sam-point") return
    const pixel = pixelFromPointer(event)
    if (!pixel) return
    setProposalId(null)
    if (editorTool === "positive") {
      setPositivePoints((current) => [...current, pixel])
      return
    }
    setNegativePoints((current) => [...current, pixel])
  }

  function onImageDragStart(event: DragEvent<HTMLImageElement>) {
    event.preventDefault()
  }

  function updateCursor(event: { clientX: number; clientY: number }) {
    const viewport = viewportRef.current
    const image = imageRef.current
    if (!viewport || !image || (editorTool !== "add" && editorTool !== "erase")) {
      setCursor(null)
      return
    }
    const inside = imagePixelFromClient(event.clientX, event.clientY, frameFrom(image))
    if (!inside) {
      setCursor(null)
      return
    }
    const bounds = viewport.getBoundingClientRect()
    setCursor({
      x: event.clientX - bounds.left - viewport.clientLeft,
      y: event.clientY - bounds.top - viewport.clientTop,
    })
  }

  function onBrushPointerDown(event: PointerEvent<HTMLImageElement>) {
    if (!proposalId || (editorTool !== "add" && editorTool !== "erase") || event.button !== 0) return
    const pixel = pixelFromPointer(event)
    if (!pixel) return
    event.preventDefault()
    brushDrag.current = { pointerId: event.pointerId, points: [pixel] }
    setPendingStroke([pixel])
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // Capture requires an active pointer. Moves that stay on the image still extend the stroke.
    }
  }

  function onBrushPointerMove(event: PointerEvent<HTMLImageElement>) {
    const drag = brushDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    event.preventDefault()
    const pixel = clampedPixel(event)
    if (!pixel) return
    const last = drag.points[drag.points.length - 1]
    if (last && sourceDistance(last, pixel) < 0.5) return
    drag.points.push(pixel)
    setPendingStroke([...drag.points])
  }

  function onBrushPointerUp(event: PointerEvent<HTMLImageElement>) {
    const drag = brushDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    brushDrag.current = null
    event.preventDefault()
    releaseBoxPointer(event)
    const points = drag.points
    setPendingStroke(null)
    if (editorTool !== "add" && editorTool !== "erase") return
    const stroke: MaskStrokePayload = {
      mode: editorTool,
      brush_radius_px: brushRadius,
      points,
    }
    void commitStroke(stroke)
  }

  function onBrushPointerCancel(event: PointerEvent<HTMLImageElement>) {
    const drag = brushDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    brushDrag.current = null
    releaseBoxPointer(event)
    setPendingStroke(null)
  }

  function onPanPointerDown(event: PointerEvent<HTMLImageElement>) {
    if (editorTool !== "pan" || event.button !== 0) return
    event.preventDefault()
    panDrag.current = {
      pointerId: event.pointerId,
      originX: event.clientX,
      originY: event.clientY,
      panX: viewRef.current.panX,
      panY: viewRef.current.panY,
    }
    setPanning(true)
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // Capture requires an active pointer.
    }
  }

  function onPanPointerMove(event: PointerEvent<HTMLImageElement>) {
    const drag = panDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    event.preventDefault()
    setView(
      viewRef.current.scale,
      drag.panX + event.clientX - drag.originX,
      drag.panY + event.clientY - drag.originY,
    )
  }

  function onPanPointerUp(event: PointerEvent<HTMLImageElement>) {
    const drag = panDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    panDrag.current = null
    setPanning(false)
    event.preventDefault()
    releaseBoxPointer(event)
  }

  function onZoomAreaPointerDown(event: PointerEvent<HTMLImageElement>) {
    if (editorTool !== "zoom-area" || event.button !== 0) return
    const image = imageRef.current
    if (!image) return
    const frame = frameFrom(image)
    const start = clampClientToDisplayedImage(event.clientX, event.clientY, frame)
    if (!start || !imagePixelFromClient(start.x, start.y, frame)) return
    event.preventDefault()
    zoomDrag.current = { pointerId: event.pointerId, startClient: start }
    setZoomRect(displayRect(image, start, start))
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // Capture requires an active pointer.
    }
  }

  function onZoomAreaPointerMove(event: PointerEvent<HTMLImageElement>) {
    const drag = zoomDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    const image = imageRef.current
    if (!image) return
    event.preventDefault()
    const current = clampClientToDisplayedImage(event.clientX, event.clientY, frameFrom(image))
    if (!current) return
    setZoomRect(displayRect(image, drag.startClient, current))
  }

  function sourceRectFromClients(image: HTMLImageElement, start: DisplayPoint, end: DisplayPoint): SourceRect | null {
    const frame = frameFrom(image)
    const startClient = clampClientToDisplayedImage(start.x, start.y, frame)
    const endClient = clampClientToDisplayedImage(end.x, end.y, frame)
    if (!startClient || !endClient) return null
    const startPixel = imagePixelFromClient(startClient.x, startClient.y, frame)
    const endPixel = imagePixelFromClient(endClient.x, endClient.y, frame)
    if (!startPixel || !endPixel) return null
    return {
      x0: Math.min(startPixel.x, endPixel.x),
      y0: Math.min(startPixel.y, endPixel.y),
      x1: Math.max(startPixel.x, endPixel.x),
      y1: Math.max(startPixel.y, endPixel.y),
    }
  }

  function onZoomAreaPointerUp(event: PointerEvent<HTMLImageElement>) {
    const drag = zoomDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    zoomDrag.current = null
    event.preventDefault()
    releaseBoxPointer(event)
    const image = imageRef.current
    const viewport = viewportRef.current
    const rect = image ? displayRect(image, drag.startClient, { x: event.clientX, y: event.clientY }) : null
    setZoomRect(null)
    if (image && viewport && rect) {
      const screenWidth = rect.width * viewRef.current.scale
      const screenHeight = rect.height * viewRef.current.scale
      const selection = sourceRectFromClients(image, drag.startClient, { x: event.clientX, y: event.clientY })
      if (selection && screenWidth >= MIN_NAVIGATION_DRAG_PX && screenHeight >= MIN_NAVIGATION_DRAG_PX) {
        const next = zoomAreaView({
          selection,
          displayWidth: screenWidth,
          displayHeight: screenHeight,
          naturalWidth: image.naturalWidth,
          naturalHeight: image.naturalHeight,
          layoutWidth: image.offsetWidth,
          layoutHeight: image.offsetHeight,
          viewportWidth: viewport.clientWidth,
          viewportHeight: viewport.clientHeight,
        })
        if (next) setView(next.scale, next.panX, next.panY)
      }
    }
    finishZoomArea()
  }

  function onEditorKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const target = event.target
    if (target instanceof HTMLElement) {
      const tag = target.tagName
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return
    }
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown" && event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return
    }
    event.preventDefault()
    if (event.key === "ArrowUp") panBy("up")
    if (event.key === "ArrowDown") panBy("down")
    if (event.key === "ArrowLeft") panBy("left")
    if (event.key === "ArrowRight") panBy("right")
  }

  function onImagePointerDown(event: PointerEvent<HTMLImageElement>) {
    viewportRef.current?.focus({ preventScroll: true })
    updateCursor(event)
    if (!adding || busy || event.button !== 0) return
    const action = imagePointerAction(editorTool)
    if (sessionKind === "accepted-region" && (action === "sam-point" || action === "sam-box")) return
    if (action === "add" || action === "erase") {
      if (sessionReadyId === proposalId) onBrushPointerDown(event)
      return
    }
    if (action === "pan") {
      onPanPointerDown(event)
      return
    }
    if (action === "zoom-area") {
      onZoomAreaPointerDown(event)
      return
    }
    if (action !== "sam-box") return
    const image = imageRef.current
    if (!image) return
    const frame = frameFrom(image)
    const start = clampClientToDisplayedImage(event.clientX, event.clientY, frame)
    if (!start || !imagePixelFromClient(start.x, start.y, frame)) return
    event.preventDefault()
    boxDrag.current = {
      pointerId: event.pointerId,
      startClient: start,
      previousBox: box,
    }
    setDraftRect(displayRect(image, start, start))
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // Capture requires an active pointer. Moves that stay on the image still update the rectangle.
    }
  }

  function onImagePointerMove(event: PointerEvent<HTMLImageElement>) {
    updateCursor(event)
    if (brushDrag.current?.pointerId === event.pointerId) {
      onBrushPointerMove(event)
      return
    }
    if (panDrag.current?.pointerId === event.pointerId) {
      onPanPointerMove(event)
      return
    }
    if (zoomDrag.current?.pointerId === event.pointerId) {
      onZoomAreaPointerMove(event)
      return
    }
    const drag = boxDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    const image = imageRef.current
    if (!image) return
    event.preventDefault()
    const current = clampClientToDisplayedImage(event.clientX, event.clientY, frameFrom(image))
    if (!current) return
    setDraftRect(displayRect(image, drag.startClient, current))
  }

  function onImagePointerUp(event: PointerEvent<HTMLImageElement>) {
    if (brushDrag.current?.pointerId === event.pointerId) {
      onBrushPointerUp(event)
      return
    }
    if (panDrag.current?.pointerId === event.pointerId) {
      onPanPointerUp(event)
      return
    }
    if (zoomDrag.current?.pointerId === event.pointerId) {
      onZoomAreaPointerUp(event)
      return
    }
    const drag = boxDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    boxDrag.current = null
    event.preventDefault()
    releaseBoxPointer(event)
    const image = imageRef.current
    if (!image) {
      setDraftRect(null)
      setBox(drag.previousBox)
      return
    }
    const frame = frameFrom(image)
    const end = clampClientToDisplayedImage(event.clientX, event.clientY, frame)
    const rect = end ? displayRect(image, drag.startClient, end) : null
    setDraftRect(null)
    const screenWidth = rect ? rect.width * viewRef.current.scale : 0
    const screenHeight = rect ? rect.height * viewRef.current.scale : 0
    if (!end || !rect || screenWidth < MIN_BOX_DISPLAY_PX || screenHeight < MIN_BOX_DISPLAY_PX) {
      setBox(drag.previousBox)
      return
    }
    const startPixel = imagePixelFromClient(drag.startClient.x, drag.startClient.y, frame)
    const endPixel = imagePixelFromClient(end.x, end.y, frame)
    if (!startPixel || !endPixel) {
      setBox(drag.previousBox)
      return
    }
    const next = boxFromPoints(startPixel, endPixel)
    if (!boxHasArea(next)) {
      setBox(drag.previousBox)
      return
    }
    setProposalId(null)
    setBox(next)
  }

  function onImagePointerCancel(event: PointerEvent<HTMLImageElement>) {
    if (brushDrag.current?.pointerId === event.pointerId) {
      onBrushPointerCancel(event)
      return
    }
    if (panDrag.current?.pointerId === event.pointerId) {
      panDrag.current = null
      setPanning(false)
      releaseBoxPointer(event)
      return
    }
    if (zoomDrag.current?.pointerId === event.pointerId) {
      zoomDrag.current = null
      releaseBoxPointer(event)
      setZoomRect(null)
      return
    }
    const drag = boxDrag.current
    if (!drag || event.pointerId !== drag.pointerId) return
    boxDrag.current = null
    releaseBoxPointer(event)
    setDraftRect(null)
    setBox(drag.previousBox)
  }

  async function generateMask() {
    if (!canGenerate || !generateMaskEnabled(sessionKind)) return
    setBusy(true)
    setError(null)
    const currentPrompts = { positive: positivePoints, negative: negativePoints, box }
    try {
      const proposal = await predictSegmentation(project.id, {
        positive_points: positivePoints,
        negative_points: negativePoints,
        box,
        name: name.trim(),
        role,
      })
      const cleared = promptsAfterMaskGeneration(true, currentPrompts)
      setPositivePoints(cleared.positive)
      setNegativePoints(cleared.negative)
      setBox(cleared.box)
      setDraftRect(null)
      setProposalId(proposal.proposal_id)
      setHistory(sessionStart())
      setMaskRevision((value) => value + 1)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Segmentation failed.")
    } finally {
      setBusy(false)
    }
  }

  async function commitStroke(stroke: MaskStrokePayload) {
    if (!proposalId || sessionReadyId !== proposalId) return
    setBusy(true)
    setError(null)
    try {
      await applyMaskStroke(project.id, proposalId, stroke)
      setHistory((current) =>
        appendCommand(current, {
          kind: "brush",
          mode: stroke.mode,
          brush_radius_px: stroke.brush_radius_px,
          points: stroke.points,
        }),
      )
      setMaskRevision((value) => value + 1)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to apply the brush stroke.")
    } finally {
      setBusy(false)
    }
  }

  async function commitWand(mode: MagicWandPayload["mode"], pixel: ImagePixel) {
    if (!proposalId || sessionReadyId !== proposalId) return
    const command: HistoryCommand = {
      kind: "wand",
      mode,
      x: pixel.x,
      y: pixel.y,
      tolerance: wandTolerance,
      contiguous: true,
    }
    setWandSeed(pixel)
    setBusy(true)
    setError(null)
    try {
      await applyMagicWand(project.id, proposalId, {
        mode: command.mode,
        x: command.x,
        y: command.y,
        tolerance: command.tolerance,
        contiguous: command.contiguous,
      })
      setHistory((current) => appendCommand(current, command))
      setMaskRevision((value) => value + 1)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to apply the Magic Wand.")
    } finally {
      setWandSeed(null)
      setBusy(false)
    }
  }

  async function replayHistory(index: number) {
    if (!proposalId || sessionReadyId !== proposalId) return
    const current = historyRef.current
    if (index === current.cursor) return
    const next = moveCursor(current, index)
    setBusy(true)
    setError(null)
    try {
      await restoreRefinementSession(project.id, proposalId)
      for (const command of next.commands.slice(0, next.cursor)) {
        if (command.kind === "brush") {
          await applyMaskStroke(project.id, proposalId, {
            mode: command.mode,
            brush_radius_px: command.brush_radius_px,
            points: command.points,
          })
        } else {
          await applyMagicWand(project.id, proposalId, {
            mode: command.mode,
            x: command.x,
            y: command.y,
            tolerance: command.tolerance,
            contiguous: command.contiguous,
          })
        }
      }
      setHistory(next)
      setMaskRevision((value) => value + 1)
    } catch (caught) {
      setMaskRevision((value) => value + 1)
      setError(caught instanceof Error ? caught.message : "Unable to restore that history state.")
    } finally {
      setBusy(false)
    }
  }

  async function resetToAiMask() {
    if (!proposalId) return
    setBusy(true)
    setError(null)
    try {
      await resetMaskProposal(project.id, proposalId)
      setHistory(sessionStart())
      setConfirmReset(false)
      setMaskRevision((value) => value + 1)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to reset the mask.")
    } finally {
      setBusy(false)
    }
  }

  async function acceptMask() {
    const trimmed = name.trim()
    if (!proposalId || !trimmed) {
      setError("Enter a region name before accepting the mask.")
      return
    }
    setBusy(true)
    setError(null)
    try {
      const updated = await createReliefRegion(project.id, {
        name: trimmed,
        role,
        proposal_id: proposalId,
      })
      onProjectChange(updated)
      setAdding(false)
      resetPrompts()
      setWorkspaceView(updated.relief?.generated_at ? "composed" : "source")
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to accept the mask.")
    } finally {
      setBusy(false)
    }
  }

  async function saveEditedMask() {
    const regionId = editRegionId
    const trimmed = name.trim()
    if (!regionId || !proposalId || !trimmed) {
      setError("Enter a region name before saving the mask.")
      return
    }
    setBusy(true)
    setError(null)
    try {
      const updated = await saveRegionMaskEdit(project.id, regionId, {
        proposal_id: proposalId,
        name: trimmed,
        role,
      })
      onProjectChange(updated)
      closeEditor()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save the mask.")
    } finally {
      setBusy(false)
    }
  }

  function updateDraft(id: string, patch: Partial<RegionDraft>) {
    setDrafts((current) => current.map((draft) => (draft.id === id ? { ...draft, ...patch } : draft)))
  }

  async function applyComposition() {
    if (!canApplyComposition(conditionedStatus)) return
    const problem = drafts.map((draft) => scaleProblem(draft)).find((item) => item != null)
    const backgroundProblem = backgroundSettingsProblem(background)
    if (problem || backgroundProblem) {
      setError(problem ?? backgroundProblem)
      return
    }
    setBusy(true)
    setError(null)
    try {
      const updated = await applyReliefComposition(
        project.id,
        drafts.map((draft) => ({
          ...draft,
          name: draft.name.trim(),
        })),
        background,
      )
      onProjectChange(updated)
    } catch (caught) {
      const failure = failedProcessingResult({
        error: caught,
        fallback: "Relief composition failed.",
        project,
        draft: drafts,
      })
      setError(failure.message)
    } finally {
      setBusy(false)
    }
  }

  function chooseView(next: ReliefView) {
    const requests = viewSwitchRequests()
    if (requests.composition || requests.segmentation || requests.depth) return
    setWorkspaceView(switchReliefView(next))
  }

  function chooseRegion(regionId: string) {
    const choice = selectAcceptedRegion(regionId)
    if (choice.createsProposal || choice.deletesProposal || choice.recomputesComposition || choice.writesMask) return
    setSelectedRegionId(choice.selectedId)
  }

  async function removeRegion(region: ReliefRegion) {
    setBusy(true)
    setError(null)
    try {
      const updated = await deleteReliefRegion(project.id, region.id)
      onProjectChange(updated)
      setPendingDelete(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to delete the region.")
    } finally {
      setBusy(false)
    }
  }

  if (!conditioned) {
    return (
      <section className="status-card relief-workspace" aria-labelledby="relief-heading">
        <h2 id="relief-heading">Relief</h2>
        <p>Conditioned depth has not been created. Open Depth and apply conditioning before composing a relief.</p>
      </section>
    )
  }

  const image = imageRef.current
  const frame = image && image.naturalWidth > 0 ? layoutFrame(image) : null
  const limits =
    image && image.naturalWidth > 0 ? brushLimits(image.naturalWidth, image.naturalHeight) : { min: 1, max: 80, initial: 12 }
  const brushScale = image && image.naturalWidth > 0 ? displayScale(frameFrom(image)) : null
  const brushDiameter = brushScale ? brushRadius * 2 * brushScale : 0
  void layoutTick

  return (
    <section className={`relief-workspace is-split${adding ? " is-editing" : ""}`}>
      <div className="relief-main" role="region" aria-labelledby="relief-heading">
      <h2 id="relief-heading" className="visually-hidden">
        Relief
      </h2>
      <p id="relief-summary" className="visually-hidden">
        Composed relief is dimensionless, from 0 at the back to 1 at the front. It is not a CNC height.
      </p>
      <ProvenanceStatus title="Relief Composition" status={reliefStatus} stage="relief" />
      {settingsPending ? (
        <p className="status-notice">
          Pending changes. This preview is the last saved composition.
        </p>
      ) : null}
      {error ? <p className="status-notice">{error}</p> : null}
      <div className="relief-preview">
          {canvas.mode === "editor" ? null : (
            <div className="view-switch" role="radiogroup" aria-label="Relief view">
              {RELIEF_VIEWS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  role="radio"
                  aria-checked={workspaceView === item.id}
                  onClick={() => chooseView(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          )}
          {canvas.mode === "editor" && project.source ? (
            <figure aria-label="Working canvas">
              <div className={adding ? "editor-shell" : "mask-stage-row"}>
              {adding ? (
                <div className="editor-toolbar" role="toolbar" aria-label="Mask editor tools">
                  <ToolButton label="Zoom Area" title="Zoom Area — drag a rectangle to zoom into that part of the image" kind="area" pressed={editorTool === "zoom-area"} onClick={() => rememberTool("zoom-area")} />
                  <ToolButton label="Fit" title="Fit — fit the image inside the canvas" kind="fit" onClick={fitView} />
                  <ToolButton label="100%" title="100% — show one source pixel per screen pixel" kind="actual" onClick={zoomToActualSize} />
                  <ToolButton
                    label="Zoom In"
                    title="Zoom In — zoom in toward the center of the canvas"
                    kind="in"
                    disabled={
                      image != null &&
                      image.naturalWidth > 0 &&
                      (viewScale * image.offsetWidth * 100) / image.naturalWidth >= MAX_ZOOM_PERCENT - 0.5
                    }
                    onClick={() => zoomBy(1.25)}
                  />
                  <ToolButton
                    label="Zoom Out"
                    title="Zoom Out — zoom out. The image cannot become smaller than Fit"
                    kind="out"
                    disabled={viewScale <= 1.001}
                    onClick={() => zoomBy(1 / 1.25)}
                  />
                  <ToolButton label="Pan" title="Pan — drag to move the image inside the canvas" kind="hand" pressed={editorTool === "pan"} onClick={() => rememberTool("pan")} />
                  {generateMaskEnabled(sessionKind) ? (
                    <>
                      <span className="toolbar-separator" role="separator" />
                      <ToolButton label="Positive Prompt" title="Positive Prompt — tell AI this belongs to the subject" kind="positive" tone="add" pressed={editorTool === "positive"} onClick={() => rememberTool("positive")} />
                      <ToolButton label="Negative Prompt" title="Negative Prompt — tell AI this does not belong to the subject" kind="negative" tone="subtract" pressed={editorTool === "negative"} onClick={() => rememberTool("negative")} />
                      <ToolButton label="Box Prompt" title="Box Prompt — drag a rectangle to tell AI where the subject is" kind="box" pressed={editorTool === "box"} onClick={() => rememberTool("box")} />
                    </>
                  ) : null}
                  <span className="toolbar-separator" role="separator" />
                  <ToolButton label="Add Brush" title="Add Brush — manually add pixels to the mask" kind="brush-add" tone="add" pressed={editorTool === "add"} disabled={busy || sessionReadyId !== proposalId} onClick={() => rememberTool("add")} />
                  <ToolButton label="Erase Brush" title="Erase Brush — manually remove pixels from the mask" kind="brush-erase" tone="subtract" pressed={editorTool === "erase"} disabled={busy || sessionReadyId !== proposalId} onClick={() => rememberTool("erase")} />
                  <ToolButton label="Magic Wand Add" title="Magic Wand Add — add a contiguous area with similar color" kind="wand-add" tone="add" pressed={editorTool === "wand-add"} disabled={busy || sessionReadyId !== proposalId} onClick={() => rememberTool("wand-add")} />
                  <ToolButton label="Magic Wand Subtract" title="Magic Wand Subtract — remove a contiguous area with similar color" kind="wand-subtract" tone="subtract" pressed={editorTool === "wand-subtract"} disabled={busy || sessionReadyId !== proposalId} onClick={() => rememberTool("wand-subtract")} />
                  <span className="toolbar-separator" role="separator" />
                  <div className="pan-pad" role="group" aria-label="Pan image">
                    <button type="button" className="pan-up" aria-label="Pan up" title="Pan up" onClick={() => panBy("up")}>
                      ↑
                    </button>
                    <button type="button" className="pan-left" aria-label="Pan left" title="Pan left" onClick={() => panBy("left")}>
                      ←
                    </button>
                    <span className="pan-center" aria-hidden="true" />
                    <button type="button" className="pan-right" aria-label="Pan right" title="Pan right" onClick={() => panBy("right")}>
                      →
                    </button>
                    <button type="button" className="pan-down" aria-label="Pan down" title="Pan down" onClick={() => panBy("down")}>
                      ↓
                    </button>
                  </div>
                  <span className="toolbar-separator" role="separator" />
                  <ToolButton label="Editor help" title="Editor help" kind="info" onClick={() => setShowHelp(true)} />
                </div>
              ) : null}
              <div
                className={`mask-viewport tool-${editorTool}${panning ? " is-panning" : ""}`}
                ref={viewportRef}
                tabIndex={adding ? 0 : undefined}
                onKeyDown={adding ? onEditorKeyDown : undefined}
                onPointerLeave={() => {
                  if (!brushDrag.current) setCursor(null)
                }}
              >
              <div
                className="relief-stage is-placed"
                style={{
                  width: stageBox ? `${stageBox.width}px` : "100%",
                  height: stageBox ? `${stageBox.height}px` : "auto",
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${viewScale})`,
                }}
              >
              <img
                ref={imageRef}
                src={getSourceImageUrl(project.id)}
                alt="Source for relief region selection"
                draggable={false}
                onDragStart={onImageDragStart}
                onClick={onImageClick}
                onPointerDown={onImagePointerDown}
                onPointerMove={onImagePointerMove}
                onPointerUp={onImagePointerUp}
                onPointerCancel={onImagePointerCancel}
                onLoad={() => setLayoutTick((value) => value + 1)}
              />
              {showMask && maskUrl ? (
                <div
                  className="mask-overlay"
                  role="img"
                  aria-label="Proposed segmentation mask"
                  style={{
                    opacity: overlayOpacity / 100,
                    maskImage: `url(${maskUrl})`,
                    WebkitMaskImage: `url(${maskUrl})`,
                  }}
                />
              ) : null}
              {frame
                ? positivePoints.map((pixel, index) => {
                    const point = clientOffsetFromPixel(pixel, frame)
                    if (!point) return null
                    return (
                      <span
                        key={`positive-${index}`}
                        className="prompt-marker prompt-positive"
                        style={{ left: point.x, top: point.y }}
                      />
                    )
                  })
                : null}
              {frame
                ? negativePoints.map((pixel, index) => {
                    const point = clientOffsetFromPixel(pixel, frame)
                    if (!point) return null
                    return (
                      <span
                        key={`negative-${index}`}
                        className="prompt-marker prompt-negative"
                        style={{ left: point.x, top: point.y }}
                      />
                    )
                  })
                : null}
              {frame && wandSeed
                ? (() => {
                    const point = clientOffsetFromPixel(wandSeed, frame)
                    if (!point) return null
                    return <span className="wand-seed" style={{ left: point.x, top: point.y }} />
                  })()
                : null}
              {zoomRect ? <span className="zoom-area-rect" style={zoomRect} /> : null}
              {draftRect ? (
                <span className="prompt-box" style={draftRect} />
              ) : frame && box ? (
                (() => {
                  const start = clientOffsetFromPixel({ x: box.x0, y: box.y0 }, frame)
                  const end = clientOffsetFromPixel({ x: box.x1, y: box.y1 }, frame)
                  if (!start || !end) return null
                  return (
                    <span
                      className="prompt-box"
                      style={{
                        left: Math.min(start.x, end.x),
                        top: Math.min(start.y, end.y),
                        width: Math.abs(end.x - start.x),
                        height: Math.abs(end.y - start.y),
                      }}
                    />
                  )
                })()
              ) : null}
              {frame && pendingStroke && pendingStroke.length > 0 ? (
                <svg className="pending-stroke" viewBox={`0 0 ${frame.width} ${frame.height}`} preserveAspectRatio="none">
                  <polyline
                    points={pendingStroke
                      .map((pixel) => {
                        const point = clientOffsetFromPixel(pixel, frame)
                        return point ? `${point.x},${point.y}` : ""
                      })
                      .filter((point) => point.length > 0)
                      .join(" ")}
                    fill="none"
                    stroke={editorTool === "erase" ? "#c0392b" : "#1f8a4c"}
                    strokeWidth={Math.max(1, brushRadius * 2 * (displayScale(frame) ?? 1))}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity={0.85}
                  />
                </svg>
              ) : null}
              </div>
              {cursor && brushDiameter > 0 ? (
                <span
                  className={editorTool === "erase" ? "brush-cursor erase" : "brush-cursor"}
                  style={{ left: cursor.x, top: cursor.y, width: brushDiameter, height: brushDiameter }}
                />
              ) : null}
              </div>
              </div>
            </figure>
          ) : null}
          {canvas.mode === "source" && project.source ? (
            <figure>
              <figcaption>Source</figcaption>
              <div className="mask-stage-row">
                <div className="mask-viewport is-readonly" ref={viewportRef}>
                  <div
                    className="relief-stage is-placed"
                    style={
                      stageBox
                        ? {
                            width: `${stageBox.width}px`,
                            height: `${stageBox.height}px`,
                            left: `${stageBox.offsetX}px`,
                            top: `${stageBox.offsetY}px`,
                          }
                        : { visibility: "hidden" }
                    }
                  >
                    <img
                      ref={imageRef}
                      src={getSourceImageUrl(project.id)}
                      alt="Source image"
                      draggable={false}
                      onLoad={() => setLayoutTick((value) => value + 1)}
                    />
                  </div>
                </div>
              </div>
            </figure>
          ) : null}
          {canvas.mode === "mask" && project.source && selectedSaved ? (
            <figure>
              <figcaption>Mask · {selectedSaved.name}</figcaption>
              <div className="mask-stage-row">
                <div className="mask-viewport is-readonly" ref={viewportRef}>
                  <div
                    className="relief-stage is-placed"
                    style={
                      stageBox
                        ? {
                            width: `${stageBox.width}px`,
                            height: `${stageBox.height}px`,
                            left: `${stageBox.offsetX}px`,
                            top: `${stageBox.offsetY}px`,
                          }
                        : { visibility: "hidden" }
                    }
                  >
                    <img
                      ref={imageRef}
                      src={getSourceImageUrl(project.id)}
                      alt={`Source image with the ${selectedSaved.name} mask`}
                      draggable={false}
                      onLoad={() => setLayoutTick((value) => value + 1)}
                    />
                    <div
                      className="mask-overlay"
                      role="img"
                      aria-label={`${selectedSaved.name} accepted region mask`}
                      style={{
                        opacity: 0.45,
                        maskImage: `url(${acceptedMaskDisplayUrl(getReliefMaskUrl(project.id, selectedSaved.id), selectedSaved.updated_at)})`,
                        WebkitMaskImage: `url(${acceptedMaskDisplayUrl(getReliefMaskUrl(project.id, selectedSaved.id), selectedSaved.updated_at)})`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </figure>
          ) : null}
          {canvas.mode === "mask-empty" ? <p className="field-note">{NO_ACCEPTED_REGION_MESSAGE}</p> : null}
          {canvas.mode === "composed" && relief?.generated_at ? (
            <figure>
              <figcaption>{compositionPreviewCaption()}</figcaption>
              <div className="mask-stage-row">
                <div className="mask-viewport is-readonly" ref={viewportRef}>
                  <div
                    className="relief-stage is-placed"
                    style={
                      stageBox
                        ? {
                            width: `${stageBox.width}px`,
                            height: `${stageBox.height}px`,
                            left: `${stageBox.offsetX}px`,
                            top: `${stageBox.offsetY}px`,
                          }
                        : { visibility: "hidden" }
                    }
                  >
                    <img
                      ref={imageRef}
                      src={getReliefPreviewUrl(project.id, relief.generated_at)}
                      alt="Composed dimensionless relief preview"
                      draggable={false}
                      onLoad={() => setLayoutTick((value) => value + 1)}
                    />
                  </div>
                </div>
              </div>
            </figure>
          ) : null}
          {canvas.mode === "composed-empty" ? <p className="field-note">{NO_COMPOSED_RELIEF_MESSAGE}</p> : null}
          {canvas.mode === "physical" && relief?.physical ? (
            <figure>
              <figcaption>{physicalPreviewCaption(canvas.status)}</figcaption>
              <div className="mask-stage-row">
                <div className="mask-viewport is-readonly" ref={viewportRef}>
                  <div
                    className="relief-stage is-placed"
                    style={
                      stageBox
                        ? {
                            width: `${stageBox.width}px`,
                            height: `${stageBox.height}px`,
                            left: `${stageBox.offsetX}px`,
                            top: `${stageBox.offsetY}px`,
                          }
                        : { visibility: "hidden" }
                    }
                  >
                    <img
                      ref={imageRef}
                      src={getPhysicalPreviewUrl(project.id, relief.physical.generated_at)}
                      alt="Physical finished-surface height preview"
                      draggable={false}
                      onLoad={() => setLayoutTick((value) => value + 1)}
                    />
                  </div>
                </div>
              </div>
            </figure>
          ) : null}
          {canvas.mode === "physical-empty" ? <p className="field-note">{NO_PHYSICAL_HEIGHT_MESSAGE}</p> : null}
        </div>
      </div>
        <div className={`relief-panel${inspectorMode(adding) === "editor" ? " is-editing" : ""}`}>
          <div className="inspector-body">
          {!adding ? (
            <>
              <PhysicalHeightPanel
                project={project}
                compositionBlocked={!canGeneratePhysical(reliefStatus)}
                onProjectChange={onProjectChange}
                onError={setError}
              />
              <h2>Uncovered Background</h2>
              <p className="field-note">
                These controls affect pixels outside every accepted region. Region role labels do not set the
                background. Depth range and elevation are dimensionless, not millimeters.
              </p>
              <label className="conditioning-slider">
                <span className="slider-heading">
                  Background Depth Range{" "}
                  <span className="slider-value">{background.background_depth_range.toFixed(2)}</span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.01}
                  aria-label="Background depth range"
                  value={background.background_depth_range}
                  onChange={(event) =>
                    setBackground((current) => ({
                      ...current,
                      background_depth_range: Number(event.target.value),
                    }))
                  }
                />
              </label>
              <p className="field-note">
                Depth range 1 preserves existing background depth variation. Depth range 0 flattens the uncovered
                background to its mean.
              </p>
              <label className="conditioning-slider">
                <span className="slider-heading">
                  Background Elevation{" "}
                  <span className="slider-value">{formatSigned(background.background_elevation)}</span>
                </span>
                <input
                  type="range"
                  min={-1}
                  max={1}
                  step={0.01}
                  aria-label="Background elevation"
                  value={background.background_elevation}
                  onChange={(event) =>
                    setBackground((current) => ({
                      ...current,
                      background_elevation: Number(event.target.value),
                    }))
                  }
                />
              </label>
              <p className="field-note">Negative elevation lowers the background.</p>
              <label className="conditioning-slider">
                <span className="slider-heading">
                  Background Feather <span className="slider-value">{Math.round(background.background_feather)}</span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  aria-label="Background feather"
                  value={background.background_feather}
                  onChange={(event) =>
                    setBackground((current) => ({
                      ...current,
                      background_feather: Number(event.target.value),
                    }))
                  }
                />
              </label>
              <p className="field-note">
                Feather softens the transition but can affect the subject boundary. A wrong mask is not repaired.
              </p>
              <h2>Relief Regions</h2>
              <div className="card-actions">
                <button
                  type="button"
                  onClick={beginAdd}
                  disabled={busy || !addRegionAvailability(draftProposal).enabled}
                >
                  Add Region
                </button>
              </div>
            </>
          ) : (
            <>
              <section className="inspector-section" aria-label="Region identity">
                <h2>Region</h2>
                <label>
                  Name
                  <input
                    value={name}
                    onChange={(event) => {
                      const next = event.target.value
                      setName(next)
                      if (proposalId) saveDraft(next, role)
                    }}
                    onBlur={() => {
                      if (proposalId) saveDraft(name, role)
                    }}
                    maxLength={80}
                  />
                </label>
                <label>
                  Role
                  <select
                    value={role}
                    onChange={(event) => {
                      const next = event.target.value as ReliefRole
                      setRole(next)
                      if (proposalId) saveDraft(name, next)
                    }}
                  >
                    {ROLES.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
              </section>
              <section className="inspector-section" aria-label="Tool settings" title={maskEditorNote(sessionKind)}>
                {generateMaskEnabled(sessionKind) ? (
                  <div className="card-actions">
                    <button type="button" onClick={() => void generateMask()} disabled={busy || !canGenerate}>
                      {proposalId ? "Update Mask" : "Generate Mask"}
                    </button>
                  </div>
                ) : null}
                {editorTool === "add" || editorTool === "erase" ? (
                  <label className="conditioning-slider">
                    <span className="slider-heading">
                      Brush Size <span className="slider-value">{Math.round(brushRadius)} px</span>
                    </span>
                    <input
                      type="range"
                      min={limits.min}
                      max={limits.max}
                      step={1}
                      value={Math.min(limits.max, Math.max(limits.min, brushRadius))}
                      aria-label="Brush Size"
                      disabled={!proposalId}
                      onChange={(event) => {
                        brushTouched.current = true
                        setBrushRadius(Number(event.target.value))
                      }}
                    />
                  </label>
                ) : null}
                {editorTool === "wand-add" || editorTool === "wand-subtract" ? (
                  <label className="conditioning-slider">
                    <span className="slider-heading">
                      Tolerance <span className="slider-value">{Math.round(wandTolerance)}</span>
                    </span>
                    <input
                      type="range"
                      min={WAND_TOLERANCE_MIN}
                      max={WAND_TOLERANCE_MAX}
                      step={1}
                      value={wandTolerance}
                      aria-label="Tolerance"
                      onChange={(event) => setWandTolerance(Number(event.target.value))}
                    />
                  </label>
                ) : null}
                <label className="conditioning-slider">
                  <span className="slider-heading">
                    Overlay <span className="slider-value">{overlayOpacity}%</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={overlayOpacity}
                    aria-label="Overlay"
                    disabled={!proposalId}
                    onChange={(event) => setOverlayOpacity(Number(event.target.value))}
                  />
                </label>
                <div className="card-actions">
                  <button
                    type="button"
                    className="button-secondary"
                    onClick={() => setShowMask((value) => !value)}
                    disabled={!proposalId}
                  >
                    {showMask ? "Source only" : "Show mask"}
                  </button>
                  <button
                    type="button"
                    className="button-secondary"
                    onClick={() => void replayHistory(history.cursor - 1)}
                    disabled={busy || history.cursor === 0 || sessionReadyId !== proposalId}
                  >
                    Undo
                  </button>
                  <button
                    type="button"
                    className="button-secondary"
                    onClick={() => void replayHistory(history.cursor + 1)}
                    disabled={busy || history.cursor >= history.commands.length || sessionReadyId !== proposalId}
                  >
                    Redo
                  </button>
                  <button
                    type="button"
                    className="button-secondary"
                    onClick={() => setConfirmReset(true)}
                    disabled={busy || sessionReadyId !== proposalId}
                  >
                    {resetMaskAction(sessionKind)}
                  </button>
                </div>
              </section>
              <section className="inspector-history" aria-label="History">
                <h3
                  className="editor-heading"
                  title="History replays this editing session from its starting mask. Reloading keeps the working mask and starts again at Session Start, with Undo disabled."
                >
                  History
                </h3>
                <ol className={historyRegion().container} aria-label="Session history" data-scroll="internal">
                  {historyEntries(history).map((entry) => (
                    <li key={entry.index}>
                      <button
                        type="button"
                        className={entry.current ? "is-current" : undefined}
                        aria-current={entry.current ? "true" : undefined}
                        disabled={busy || sessionReadyId !== proposalId}
                        onClick={() => void replayHistory(entry.index)}
                      >
                        {entry.label}
                      </button>
                    </li>
                  ))}
                </ol>
              </section>
            </>
          )}
          <div className={adding ? "inspector-nav" : undefined}>
          {!adding && addRegionAvailability(draftProposal).message ? (
            <p className="field-note">{addRegionAvailability(draftProposal).message}</p>
          ) : null}
          <h3 className="region-group-heading">{ACCEPTED_REGIONS_HEADING}</h3>
          {drafts.length === 0 ? <p className="field-note">{NO_ACCEPTED_REGIONS_MESSAGE}</p> : (
            <ul className="region-list" aria-label="Accepted Regions">
              {reliefRegionList(drafts, selectedRegionId, (role) => roleLabel(role as ReliefRole)).map((item) => (
                <li key={item.id}>
                  <button
                    type="button"
                    className={item.className}
                    aria-pressed={item.selected}
                    onClick={() => chooseRegion(item.id)}
                  >
                    <strong>{item.name}</strong>
                    <span>{item.summary}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          <h3 className="region-group-heading">{DRAFT_REGIONS_HEADING}</h3>
          {draftProposal ? (
            <ul className="region-list" aria-label="Draft Regions">
              {draftRegionList(
                [{ proposalId: draftProposal.proposal_id, name: draftProposal.name ?? "", role: draftProposal.role ?? "primary" }],
                (role) => roleLabel(role as ReliefRole),
              ).map((item) => (
                <li key={item.proposalId}>
                  <div
                    className={item.className}
                    role="group"
                    aria-label={`${item.name}, ${item.summary}`}
                    onClick={() => {
                      if (!adding && !busy) continueDraft(draftProposal)
                    }}
                  >
                    <div>
                      <strong>{item.name}</strong>
                      <span>{item.summary}</span>
                    </div>
                    <button
                      type="button"
                      className="button-secondary"
                      disabled={adding || busy}
                      onClick={(event) => {
                        event.stopPropagation()
                        continueDraft(draftProposal)
                      }}
                    >
                      {item.continueLabel}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : draftProposal === undefined ? null : (
            <p className="field-note">{NO_DRAFT_REGIONS_MESSAGE}</p>
          )}
          {drafts.filter((draft) => !adding && draft.id === selectedRegionId).map((draft) => {
            const saved = project.relief?.regions.find((region) => region.id === draft.id)
            const sampling = detailSamplingGuidance({
              detailScaleMm: draft.detail_scale_mm,
              artworkWidthMm: project.artwork.width_mm,
              artworkHeightMm: project.artwork.height_mm,
              widthPx: project.source?.width_px,
              heightPx: project.source?.height_px,
            })
            return (
              <article key={draft.id} className="region-card">
                <label>
                  Name
                  <input
                    value={draft.name}
                    maxLength={80}
                    onChange={(event) => updateDraft(draft.id, { name: event.target.value })}
                  />
                </label>
                <label>
                  Role
                  <select
                    value={draft.role}
                    onChange={(event) => updateDraft(draft.id, { role: event.target.value as ReliefRole })}
                  >
                    {ROLES.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="field-note">
                  {roleLabel(draft.role)} · order {draft.order}
                </p>
                <label className="conditioning-slider">
                  <span className="slider-heading">
                    Elevation <span className="slider-value">{formatSigned(draft.elevation)}</span>
                  </span>
                  <input
                    type="range"
                    min={-1}
                    max={1}
                    step={0.01}
                    value={draft.elevation}
                    onChange={(event) => updateDraft(draft.id, { elevation: Number(event.target.value) })}
                  />
                </label>
                <label className="conditioning-slider">
                  <span className="slider-heading">
                    Depth Range <span className="slider-value">{draft.depth_range.toFixed(2)}</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.01}
                    value={draft.depth_range}
                    onChange={(event) => updateDraft(draft.id, { depth_range: Number(event.target.value) })}
                  />
                </label>
                <label className="conditioning-slider">
                  <span className="slider-heading">
                    Feather <span className="slider-value">{Math.round(draft.feather)}</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={draft.feather}
                    onChange={(event) => updateDraft(draft.id, { feather: Number(event.target.value) })}
                  />
                </label>
                <label className="conditioning-slider">
                  <span className="slider-heading">
                    Fine Detail <span className="slider-value">{draft.detail_amount.toFixed(3)}</span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={0.05}
                    step={0.001}
                    aria-label="Fine Detail"
                    value={draft.detail_amount}
                    onChange={(event) =>
                      updateDraft(draft.id, {
                        detail_amount: Math.round(Number(event.target.value) * 1000) / 1000,
                      })
                    }
                  />
                  <span className="field-note">
                    Amplitude of photographic texture. It adds shallow surface variation and does not reconstruct
                    true geometry.
                  </span>
                </label>
                <label>
                  Detail scale
                  <select
                    aria-label="Detail scale mode"
                    value={draft.detail_scale_mm == null ? "pixels" : "physical"}
                    onChange={(event) => {
                      if (event.target.value === "pixels") {
                        setScaleText((current) => {
                          const next = { ...current }
                          delete next[draft.id]
                          return next
                        })
                        updateDraft(draft.id, { detail_scale_mm: null })
                        return
                      }
                      setScaleText((current) => {
                        const next = { ...current }
                        delete next[draft.id]
                        return next
                      })
                      updateDraft(draft.id, {
                        detail_scale_mm: draft.detail_scale_mm ?? DETAIL_SCALE_INITIAL_MM,
                      })
                    }}
                  >
                    <option value="pixels">2 pixels</option>
                    <option value="physical">Physical</option>
                  </select>
                </label>
                {draft.detail_scale_mm != null ? (
                  <label>
                    Feature width ({unitLabel(unit)})
                    <input
                      type="number"
                      inputMode="decimal"
                      step="any"
                      aria-label="Detail scale"
                      aria-invalid={scaleProblem(draft) != null}
                      value={scaleText[draft.id] ?? formatLengthInput(draft.detail_scale_mm, unit)}
                      onChange={(event) => {
                        const text = event.target.value
                        setScaleText((current) => ({ ...current, [draft.id]: text }))
                        const parsed = parseDetailScaleMm(text, unit)
                        if (parsed != null) updateDraft(draft.id, { detail_scale_mm: parsed })
                      }}
                    />
                  </label>
                ) : null}
                <p className="field-note">
                  {draft.detail_scale_mm == null
                    ? "Feature width is 2 pixels. Fine Detail is the amplitude, not this width."
                    : `Feature width ${formatLength(draft.detail_scale_mm, unit)}. Fine Detail is the amplitude, not this width.`}{" "}
                  This does not reconstruct true geometry.
                </p>
                {sampling.warning ? <p className="field-note">{sampling.warning}</p> : null}
                {scaleProblem(draft) ? <p className="field-note">{scaleProblem(draft)}</p> : null}
                {pendingDelete === draft.id && saved ? (
                  <div className="card-actions">
                    <button type="button" onClick={() => void removeRegion(saved)} disabled={busy}>
                      Delete {saved.name}
                    </button>
                    <button
                      type="button"
                      className="button-secondary"
                      onClick={() => setPendingDelete(null)}
                      disabled={busy}
                    >
                      Keep
                    </button>
                  </div>
                ) : (
                  <div className="card-actions">
                    <button
                      type="button"
                      className="button-secondary"
                      onClick={() => {
                        if (saved) void beginEdit(saved)
                      }}
                      disabled={busy || adding || !saved}
                    >
                      {editMaskLabel}
                    </button>
                    <button
                      type="button"
                      className="button-secondary"
                      onClick={() => setPendingDelete(draft.id)}
                      disabled={busy || adding}
                    >
                      {deleteRegionLabel}
                    </button>
                  </div>
                )}
              </article>
            )
          })}
          </div>
          </div>
          {adding ? (
            <div className="inspector-actions" role="group" aria-label="Mask actions">
              <div className="card-actions">
                <button
                  type="button"
                  onClick={() => void (editRegionId ? saveEditedMask() : acceptMask())}
                  disabled={busy || !proposalId}
                >
                  {maskActions.primary}
                </button>
                <button type="button" className="button-secondary" onClick={requestClose} disabled={busy}>
                  {maskActions.close}
                </button>
              </div>
              {maskActions.deleteProposal ? (
                <div className="card-actions inspector-danger">
                  <button
                    type="button"
                    className="button-danger"
                    onClick={() => setConfirmDelete(true)}
                    disabled={busy || !proposalId}
                  >
                    Delete Proposal
                  </button>
                </div>
              ) : null}
            </div>
          ) : drafts.length > 0 ? (
            <div className="inspector-actions">
              <div className="card-actions">
                <button
                  type="button"
                  onClick={() => void applyComposition()}
                  disabled={busy || !canApplyComposition(conditionedStatus)}
                >
                  Apply Composition
                </button>
              </div>
            </div>
          ) : null}
        </div>
      {showHelp ? (
        <div className="modal-backdrop">
          <div className="modal editor-help" role="dialog" aria-modal="true" aria-labelledby="editor-help-title">
            <h2 id="editor-help-title">Mask editor</h2>
            <dl>
              {helpTopics(sessionKind).map((entry) => (
                <Fragment key={entry.term}>
                  <dt>{entry.term}</dt>
                  <dd>{entry.detail}</dd>
                </Fragment>
              ))}
            </dl>
            <div className="modal-actions">
              <button type="button" onClick={() => setShowHelp(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {confirmReset ? (
        <div className="modal-backdrop">
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="reset-mask-title">
            <h2 id="reset-mask-title">{resetMaskAction(sessionKind)}</h2>
            <p>{resetMaskPrompt(sessionKind)}</p>
            <div className="modal-actions">
              <button type="button" className="button-secondary" onClick={() => setConfirmReset(false)} disabled={busy}>
                Cancel
              </button>
              <button type="button" onClick={() => void resetToAiMask()} disabled={busy}>
                {resetMaskAction(sessionKind)}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {confirmDiscard ? (
        <div className="modal-backdrop">
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="discard-edit-title">
            <h2 id="discard-edit-title">{DISCARD_UNSAVED_CHANGES}</h2>
            <p>{DISCARD_UNSAVED_CHANGES}</p>
            <div className="modal-actions">
              <button type="button" className="button-secondary" onClick={cancelDiscard} disabled={busy}>
                Keep Editing
              </button>
              <button type="button" onClick={() => void discardEdit()} disabled={busy}>
                Discard Changes
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {confirmDelete ? (
        <div className="modal-backdrop">
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="delete-proposal-title">
            <h2 id="delete-proposal-title">Delete Proposal</h2>
            <p>Delete this mask proposal and all manual refinement? This cannot be undone.</p>
            <div className="modal-actions">
              <button type="button" className="button-secondary" onClick={() => setConfirmDelete(false)} disabled={busy}>
                Cancel
              </button>
              <button type="button" className="button-danger" onClick={() => void deleteProposal()} disabled={busy}>
                Delete Proposal
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  )
}
