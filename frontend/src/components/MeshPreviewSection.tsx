import { Canvas, useThree } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import {
  Component,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { MeshStandardMaterial, BufferAttribute, BufferGeometry, FrontSide, LineBasicMaterial } from "three"
import type { Project } from "../types/project.ts"
import { ApiError } from "../api/client.ts"
import { getMeshPreview } from "../api/projects.ts"
import { useUnits } from "../units/UnitsContext.tsx"
import { WorkspaceInspector } from "./WorkspaceInspector.tsx"
import { ProvenanceStatus } from "./ProvenanceStatus.tsx"
import {
  DOWNSAMPLE_WARNING,
  DISPLAY_TRANSFORM,
  GPU_FAILURE_MESSAGE,
  MESH_QUALITIES,
  MISSING_PHYSICAL_MESSAGE,
  PREVIEW_ONLY_NOTICE,
  SHADING_NOTE,
  artworkLocalRect,
  boundsOfPositions,
  decodeRfmp,
  PREVIEW_LIGHT_LIMITS,
  defaultPreviewLighting,
  displayBoundaryCorners,
  displayVertexNormals,
  fitCamera,
  formatMeshDimension,
  formatMeshRange,
  formatPreviewLight,
  previewLightValue,
  stockBoundaryGuide,
  type PreviewLightChannel,
  type PreviewLighting,
  largePreviewWarning,
  meshPreviewFetchToken,
  meshPreviewGenerationChange,
  meshPreviewRequestIdentity,
  meshPreviewRequestKey,
  meshPreviewShouldFetch,
  meshPreviewView,
  meshProvenanceChain,
  meshQuality,
  previewAriaStatus,
  previewGeometryCaution,
  retainReliefGeometry,
  retainReliefMaterial,
  retryAllowed,
  shouldApplyMeshResult,
  toDisplayPositions,
  topologyExplanation,
  topologyLabel,
  vertexAttributePlan,
  webglStatus,
  type CameraPose,
  type CameraViewName,
  type DecodedMeshPreview,
  type HeldReliefGeometry,
  type HeldReliefMaterial,
  type MeshQualityId,
  type Vec3,
} from "../utils/meshPreview.ts"

const PLAN = vertexAttributePlan()

export default function MeshPreviewSection({
  project,
  onOpenRelief,
}: {
  project: Project
  onOpenRelief: () => void
}) {
  const { unit } = useUnits()
  const [quality, setQuality] = useState<MeshQualityId>("standard")
  const [viewName, setViewName] = useState<CameraViewName>("perspective")
  const [fitToken, setFitToken] = useState(0)
  const [shading, setShading] = useState<"smooth" | "flat">("smooth")
  const [wireframe, setWireframe] = useState(false)
  const [showStockBoundary, setShowStockBoundary] = useState(true)
  const [showArtworkBoundary, setShowArtworkBoundary] = useState(true)
  const [lighting, setLighting] = useState<PreviewLighting>(defaultPreviewLighting)
  const [webgl, setWebgl] = useState<"available" | "unavailable">(() => webglStatus())
  const [phase, setPhase] = useState<"idle" | "loading" | "ready" | "error">("idle")
  const [errorStatus, setErrorStatus] = useState<number | null>(null)
  const [errorKind, setErrorKind] = useState<"http" | "malformed" | "gpu" | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [decoded, setDecoded] = useState<DecodedMeshPreview | null>(null)
  const [retry, setRetry] = useState(0)
  const requestRef = useRef(0)
  const maxVertices = meshQuality(quality).maxVertices
  const identity = meshPreviewRequestIdentity({
    projectId: project.id,
    maxVertices,
    physical: project.relief?.physical ?? null,
  })
  const requestKey = meshPreviewRequestKey(identity)
  const fetchToken = meshPreviewFetchToken(requestKey, retry)
  const [appliedToken, setAppliedToken] = useState<string | null>(fetchToken)
  if (appliedToken !== fetchToken) {
    const change = meshPreviewGenerationChange(appliedToken, fetchToken)
    setAppliedToken(fetchToken)
    if (change.clearGeometry) {
      setDecoded(null)
      setPhase(change.phase)
      setErrorMessage(null)
      setErrorStatus(null)
      setErrorKind(null)
    }
  }
  const activeTokenRef = useRef(fetchToken)
  activeTokenRef.current = fetchToken
  const identityRef = useRef(identity)
  identityRef.current = identity
  const projectRef = useRef(project.id)
  projectRef.current = project.id
  const view = meshPreviewView({
    hasRecord: identity.available,
    webgl,
    phase: identity.available ? phase : "idle",
    errorStatus,
    errorKind,
    errorMessage,
  })

  useEffect(() => {
    setWebgl(webglStatus())
  }, [retry])

  useEffect(() => {
    if (fetchToken == null || !meshPreviewShouldFetch({ requestKey, webgl })) return
    const controller = new AbortController()
    const nextId = requestRef.current + 1
    requestRef.current = nextId
    const requested = identityRef.current
    const projectId = requested.projectId
    const requestedToken = fetchToken
    setPhase("loading")
    setDecoded(null)
    setErrorMessage(null)
    setErrorStatus(null)
    setErrorKind(null)
    const accept = (aborted: boolean) =>
      shouldApplyMeshResult({
        requestId: nextId,
        currentId: requestRef.current,
        projectId: projectRef.current,
        requestedProjectId: projectId,
        requestedKey: requestedToken,
        activeKey: activeTokenRef.current,
        aborted,
      })
    getMeshPreview(projectId, requested.maxVertices, controller.signal)
      .then((buffer) => {
        if (!accept(controller.signal.aborted)) return
        try {
          const mesh = decodeRfmp(buffer)
          if (mesh.exportReady !== false || PLAN.faceNormalBinding !== "rejected") {
            throw new Error("The preview was marked export-ready.")
          }
          setDecoded(mesh)
          setPhase("ready")
          setViewName("perspective")
          setFitToken((value) => value + 1)
        } catch (error: unknown) {
          setErrorKind("malformed")
          setErrorMessage(error instanceof Error ? error.message : "Malformed RFMP payload.")
          setPhase("error")
        }
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return
        if (!accept(controller.signal.aborted)) return
        if (error instanceof ApiError) {
          setErrorStatus(error.status)
          setErrorKind("http")
          setErrorMessage(error.message)
        } else {
          setErrorStatus(null)
          setErrorKind("http")
          setErrorMessage("Unable to reach the backend.")
        }
        setPhase("error")
      })
    return () => controller.abort()
  }, [fetchToken, webgl])

  const showCanvas = view.kind === "ready" && decoded != null
  const artworkRect = useMemo(
    () => (decoded ? artworkLocalRect(decoded.artworkWidthMm, decoded.artworkHeightMm) : null),
    [decoded],
  )
  const stockGuide = useMemo(
    () =>
      decoded
        ? stockBoundaryGuide({
            stockWidthMm: project.stock.width_mm,
            stockHeightMm: project.stock.height_mm,
            artworkWidthMm: project.artwork.width_mm,
            artworkHeightMm: project.artwork.height_mm,
            centerXmm: project.artwork.center_x_mm,
            centerYmm: project.artwork.center_y_mm,
            meshArtworkWidthMm: decoded.artworkWidthMm,
            meshArtworkHeightMm: decoded.artworkHeightMm,
          })
        : null,
    [decoded, project.artwork, project.stock],
  )
  const stockCorners = useMemo(
    () => (showStockBoundary && stockGuide?.available ? displayBoundaryCorners(stockGuide.bounds) : null),
    [showStockBoundary, stockGuide],
  )
  const artworkCorners = useMemo(
    () => (showArtworkBoundary && artworkRect ? displayBoundaryCorners(artworkRect) : null),
    [artworkRect, showArtworkBoundary],
  )
  const trust = decoded ? previewGeometryCaution(decoded.status, identity.physicalStatus) : null
  const heavy = decoded ? largePreviewWarning(decoded.triangleCount) : null
  const provenance = meshProvenanceChain(project)

  if (!identity.available) {
    return (
      <section className="mesh-workspace" aria-labelledby="mesh-empty-heading">
        <h2 id="mesh-empty-heading" className="visually-hidden">
          3D preview
        </h2>
        <p className="status-notice">{MISSING_PHYSICAL_MESSAGE}</p>
        <p className="field-note">{PREVIEW_ONLY_NOTICE}</p>
        <MeshProvenanceList items={provenance} />
        <button type="button" className="button-secondary" onClick={onOpenRelief}>
          Open Relief
        </button>
      </section>
    )
  }

  return (
    <section className="mesh-workspace is-split" aria-labelledby="mesh-heading">
      <h2 id="mesh-heading" className="visually-hidden">
        3D preview
      </h2>
      <div className="mesh-main">
        <p className="mesh-preview-notice">{PREVIEW_ONLY_NOTICE}</p>
        {view.kind === "loading" ? <p className="field-note">Loading 3D preview.</p> : null}
        {"message" in view ? <p className="status-notice">{view.message}</p> : null}
        {retryAllowed(view) ? (
          <button type="button" className="button-secondary" onClick={() => setRetry((value) => value + 1)}>
            Retry
          </button>
        ) : null}
        {showCanvas && decoded ? (
          <div
            className="mesh-viewport"
            aria-label={`${previewAriaStatus(decoded.status, identity.physicalStatus)} relief preview, ${decoded.vertexCount} vertices, ${decoded.triangleCount} triangles`}
          >
            <MeshErrorBoundary
              onError={() => {
                setErrorKind("gpu")
                setErrorMessage(GPU_FAILURE_MESSAGE)
                setPhase("error")
                setDecoded(null)
              }}
            >
              <MeshCanvas
                decoded={decoded}
                viewName={viewName}
                fitToken={fitToken}
                flat={shading === "flat"}
                wireframe={wireframe}
                stockCorners={stockCorners}
                artworkCorners={artworkCorners}
                lighting={lighting}
                onGpuFailure={() => {
                  setErrorKind("gpu")
                  setErrorMessage(GPU_FAILURE_MESSAGE)
                  setPhase("error")
                  setDecoded(null)
                }}
              />
            </MeshErrorBoundary>
          </div>
        ) : null}
      </div>
      <WorkspaceInspector
        className="mesh-panel"
        label="3D preview details"
        footer={
          <p className="field-note mesh-shading-note">{SHADING_NOTE}</p>
        }
      >
        <h2>3D</h2>
        <div className="mesh-lighting" role="group" aria-label="Lighting">
          <p className="slider-heading">Lighting</p>
          <LightSlider
            channel="ambient"
            label="Ambient Light"
            value={lighting.ambient}
            onChange={(value) => setLighting((current) => ({ ...current, ambient: value }))}
          />
          <LightSlider
            channel="key"
            label="Key Light"
            value={lighting.key}
            onChange={(value) => setLighting((current) => ({ ...current, key: value }))}
          />
          <LightSlider
            channel="fill"
            label="Fill Light"
            value={lighting.fill}
            onChange={(value) => setLighting((current) => ({ ...current, fill: value }))}
          />
          <button type="button" onClick={() => setLighting(defaultPreviewLighting())}>
            Reset Lighting
          </button>
        </div>
        <p className="mesh-preview-notice">{PREVIEW_ONLY_NOTICE}</p>
        <MeshProvenanceList items={provenance} />
        {trust ? <p className="field-note">{trust}</p> : null}
        <label className="mesh-quality">
          Quality
          <select
            value={quality}
            onChange={(event) => setQuality(event.target.value as MeshQualityId)}
          >
            {MESH_QUALITIES.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label} ({item.maxVertices.toLocaleString("en-US")} top vertices)
              </option>
            ))}
          </select>
        </label>
        <p className="field-note">Standard is the default. Maximum is not requested automatically.</p>
        {decoded?.downsamplingRequired ? <p className="status-notice">{DOWNSAMPLE_WARNING}</p> : null}
        {heavy ? <p className="field-note">{heavy}</p> : null}
        <dl className="mesh-stats">
          <div>
            <dt>Source</dt>
            <dd>{decoded ? `${decoded.sourceWidthPx} × ${decoded.sourceHeightPx} px` : "—"}</dd>
          </div>
          <div>
            <dt>Preview grid</dt>
            <dd>{decoded ? `${decoded.meshWidthPx} × ${decoded.meshHeightPx}` : "—"}</dd>
          </div>
          <div>
            <dt>Vertices</dt>
            <dd>{decoded ? decoded.vertexCount.toLocaleString("en-US") : "—"}</dd>
          </div>
          <div>
            <dt>Triangles</dt>
            <dd>{decoded ? decoded.triangleCount.toLocaleString("en-US") : "—"}</dd>
          </div>
          <div>
            <dt>Artwork width</dt>
            <dd>{decoded ? formatMeshDimension(decoded.artworkWidthMm, unit) : "—"}</dd>
          </div>
          <div>
            <dt>Artwork height</dt>
            <dd>{decoded ? formatMeshDimension(decoded.artworkHeightMm, unit) : "—"}</dd>
          </div>
          <div>
            <dt>Bounding box X</dt>
            <dd>{decoded ? formatMeshRange(decoded.boundingBox.min[0], decoded.boundingBox.max[0], unit) : "—"}</dd>
          </div>
          <div>
            <dt>Bounding box Y</dt>
            <dd>{decoded ? formatMeshRange(decoded.boundingBox.min[1], decoded.boundingBox.max[1], unit) : "—"}</dd>
          </div>
          <div>
            <dt>Z range</dt>
            <dd>{decoded ? formatMeshRange(decoded.boundingBox.min[2], decoded.boundingBox.max[2], unit) : "—"}</dd>
          </div>
          <div>
            <dt>Downsampling</dt>
            <dd>{decoded ? (decoded.downsamplingRequired ? "Yes" : "No") : "—"}</dd>
          </div>
          <div>
            <dt>Topology</dt>
            <dd>{decoded ? topologyLabel(decoded.topologyLevel) : "—"}</dd>
          </div>
          <div>
            <dt>Export-ready</dt>
            <dd>{decoded ? "false" : "—"}</dd>
          </div>
        </dl>
        {decoded ? <p className="field-note">{topologyExplanation(decoded.topologyLevel)}</p> : null}
        <p className="field-note">{DISPLAY_TRANSFORM}</p>
        <div className="mesh-controls" role="group" aria-label="Camera">
          <button type="button" onClick={() => setFitToken((value) => value + 1)}>
            Fit
          </button>
          <button
            type="button"
            onClick={() => {
              setViewName("perspective")
              setFitToken((value) => value + 1)
            }}
          >
            Reset
          </button>
          <button type="button" onClick={() => chooseView("perspective", setViewName, setFitToken)}>
            Perspective
          </button>
          <button type="button" onClick={() => chooseView("top", setViewName, setFitToken)}>
            Top
          </button>
          <button type="button" onClick={() => chooseView("front", setViewName, setFitToken)}>
            Front
          </button>
          <button type="button" onClick={() => chooseView("side", setViewName, setFitToken)}>
            Side
          </button>
          <button type="button" onClick={() => chooseView("isometric", setViewName, setFitToken)}>
            Isometric
          </button>
        </div>
        <label className="mesh-quality">
          Shading
          <select
            value={shading}
            onChange={(event) => setShading(event.target.value === "flat" ? "flat" : "smooth")}
          >
            <option value="smooth">Smooth</option>
            <option value="flat">Flat</option>
          </select>
        </label>
        <label className="mesh-wire">
          <input
            type="checkbox"
            checked={wireframe}
            onChange={(event) => setWireframe(event.target.checked)}
          />
          Wireframe
        </label>
        <label className="mesh-wire">
          <input
            type="checkbox"
            checked={showStockBoundary && stockGuide?.available === true}
            disabled={stockGuide?.available !== true}
            onChange={(event) => setShowStockBoundary(event.target.checked)}
          />
          Show Stock Boundary
        </label>
        <label className="mesh-wire">
          <input
            type="checkbox"
            checked={showArtworkBoundary && artworkRect != null}
            disabled={artworkRect == null}
            onChange={(event) => setShowArtworkBoundary(event.target.checked)}
          />
          Show Artwork Boundary
        </label>
        {stockGuide?.available ? (
          <p className="field-note">
            Stock {formatMeshDimension(project.stock.width_mm, unit)} ×{" "}
            {formatMeshDimension(project.stock.height_mm, unit)}. Margins left{" "}
            {formatMeshDimension(stockGuide.leftMm, unit)}, right {formatMeshDimension(stockGuide.rightMm, unit)},
            front {formatMeshDimension(stockGuide.frontMm, unit)}, back {formatMeshDimension(stockGuide.backMm, unit)}.
            Display-only. Not part of the exported solid.
          </p>
        ) : decoded ? (
          <p className="field-note">{stockGuide?.reason}</p>
        ) : null}
      </WorkspaceInspector>
    </section>
  )
}

function chooseView(
  next: CameraViewName,
  setViewName: (view: CameraViewName) => void,
  setFitToken: (update: (value: number) => number) => void,
) {
  setViewName(next)
  setFitToken((value) => value + 1)
}

function MeshCanvas({
  decoded,
  viewName,
  fitToken,
  flat,
  wireframe,
  stockCorners,
  artworkCorners,
  lighting,
  onGpuFailure,
}: {
  decoded: DecodedMeshPreview
  viewName: CameraViewName
  fitToken: number
  flat: boolean
  wireframe: boolean
  stockCorners: Vec3[] | null
  artworkCorners: Vec3[] | null
  lighting: PreviewLighting
  onGpuFailure: () => void
}) {
  const positions = useMemo(() => toDisplayPositions(decoded.positions), [decoded])
  const bounds = useMemo(() => boundsOfPositions(positions), [positions])
  if (!bounds) return null
  return (
    <Canvas
      frameloop="demand"
      camera={{ fov: 45, position: [1, 1, 1], near: 0.05, far: 1000 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.setClearColor("#e4e7ec")
      }}
    >
      <ambientLight intensity={lighting.ambient} />
      <directionalLight position={[0.6, 1.4, 0.8]} intensity={lighting.key} />
      <directionalLight position={[-0.8, 0.4, -0.4]} intensity={lighting.fill} />
      <ReliefMesh positions={positions} indices={decoded.indices} flat={flat} wireframe={wireframe} />
      {stockCorners ? <BoundaryLoop corners={stockCorners} color="#243044" /> : null}
      {artworkCorners ? <BoundaryLoop corners={artworkCorners} color="#c45c26" /> : null}
      <GuideRedraw
        signature={`${stockCorners?.length ?? 0}:${artworkCorners?.length ?? 0}:${lighting.ambient}:${lighting.key}:${lighting.fill}`}
      />
      <CameraRig bounds={bounds} viewName={viewName} fitToken={fitToken} />
      <OrbitControls makeDefault enableDamping={false} />
      <PreviewSession onGpuFailure={onGpuFailure} />
    </Canvas>
  )
}

function MeshProvenanceList({
  items,
}: {
  items: ReturnType<typeof meshProvenanceChain>
}) {
  return (
    <div className="mesh-provenance" role="group" aria-label="Processing provenance">
      {items.map((item) => (
        <ProvenanceStatus key={item.id} title={item.title} status={item.status} stage={item.stage} />
      ))}
    </div>
  )
}

function LightSlider({
  channel,
  label,
  value,
  onChange,
}: {
  channel: PreviewLightChannel
  label: string
  value: number
  onChange: (value: number) => void
}) {
  const limits = PREVIEW_LIGHT_LIMITS[channel]
  return (
    <label className="conditioning-slider">
      <span className="slider-heading">
        <span>{label}</span>
        <span className="slider-value">{formatPreviewLight(value)}</span>
      </span>
      <input
        type="range"
        min={limits.min}
        max={limits.max}
        step={limits.step}
        value={value}
        onChange={(event) => onChange(previewLightValue(channel, Number(event.target.value)))}
      />
    </label>
  )
}

function BoundaryLoop({ corners, color }: { corners: Vec3[]; color: string }) {
  const geometry = useMemo(() => {
    const next = new BufferGeometry()
    const positions = new Float32Array(corners.length * 3)
    corners.forEach((corner, index) => {
      positions[index * 3] = corner[0]
      positions[index * 3 + 1] = corner[1]
      positions[index * 3 + 2] = corner[2]
    })
    next.setAttribute("position", new BufferAttribute(positions, 3))
    return next
  }, [corners])
  const material = useMemo(
    () => new LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }),
    [color],
  )
  useEffect(() => {
    return () => {
      geometry.dispose()
      material.dispose()
    }
  }, [geometry, material])
  return <lineLoop geometry={geometry} material={material} renderOrder={2} />
}

function GuideRedraw({ signature }: { signature: string }) {
  const invalidate = useThree((state) => state.invalidate)
  useEffect(() => {
    invalidate()
  }, [invalidate, signature])
  return null
}

function ReliefMesh({
  positions,
  indices,
  flat,
  wireframe,
}: {
  positions: Float32Array
  indices: Uint32Array
  flat: boolean
  wireframe: boolean
}) {
  const geometryHold = useRef<HeldReliefGeometry<BufferGeometry> | null>(null)
  const materialHold = useRef<HeldReliefMaterial<MeshStandardMaterial> | null>(null)
  const geometry = useMemo(() => {
    const retained = retainReliefGeometry(geometryHold.current, positions, indices, () => {
      const next = new BufferGeometry()
      next.setAttribute("position", new BufferAttribute(positions, 3))
      next.setAttribute("normal", new BufferAttribute(displayVertexNormals(positions, indices), 3))
      next.setIndex(new BufferAttribute(indices, 1))
      return next
    })
    geometryHold.current = retained.hold
    return retained.hold.geometry
  }, [indices, positions])
  const material = useMemo(() => {
    const retained = retainReliefMaterial(
      materialHold.current,
      flat,
      wireframe,
      () =>
        new MeshStandardMaterial({
          color: "#b7bcc3",
          metalness: 0.04,
          roughness: 0.8,
          side: FrontSide,
        }),
      (current, nextFlat, nextWire) => {
        current.flatShading = nextFlat
        current.wireframe = nextWire
        current.needsUpdate = true
      },
    )
    materialHold.current = retained.hold
    return retained.hold.material
  }, [flat, wireframe])
  const invalidate = useThree((state) => state.invalidate)
  useLayoutEffect(() => {
    invalidate()
  }, [flat, invalidate, wireframe])
  useEffect(() => {
    return () => geometry.dispose()
  }, [geometry])
  useEffect(() => {
    return () => material.dispose()
  }, [material])
  return <mesh geometry={geometry} material={material} />
}

function PreviewSession({ onGpuFailure }: { onGpuFailure: () => void }) {
  const gl = useThree((state) => state.gl)
  const controls = useThree((state) => state.controls)
  const invalidate = useThree((state) => state.invalidate)
  const closed = useRef(false)
  useLayoutEffect(() => {
    closed.current = false
    return () => {
      closed.current = true
    }
  }, [])
  useEffect(() => {
    const canvas = gl.domElement
    const onLost = (event: Event) => {
      if (closed.current) return
      event.preventDefault()
      onGpuFailure()
    }
    canvas.addEventListener("webglcontextlost", onLost)
    return () => {
      closed.current = true
      canvas.removeEventListener("webglcontextlost", onLost)
    }
  }, [gl, onGpuFailure])
  useEffect(() => {
    const orbit = controls as {
      addEventListener?: (type: string, listener: () => void) => void
      removeEventListener?: (type: string, listener: () => void) => void
    } | null
    if (!orbit?.addEventListener || !orbit.removeEventListener) return
    const redraw = () => invalidate()
    orbit.addEventListener("change", redraw)
    return () => orbit.removeEventListener?.("change", redraw)
  }, [controls, invalidate])
  return null
}

function CameraRig({
  bounds,
  viewName,
  fitToken,
}: {
  bounds: { min: Vec3; max: Vec3 }
  viewName: CameraViewName
  fitToken: number
}) {
  const camera = useThree((state) => state.camera)
  const controls = useThree((state) => state.controls)
  const size = useThree((state) => state.size)
  const invalidate = useThree((state) => state.invalidate)
  useEffect(() => {
    const aspect = size.height > 0 ? size.width / size.height : 1
    const pose: CameraPose = fitCamera(bounds, aspect, viewName)
    const perspective = camera as { fov?: number; near: number; far: number; position: { set: (...v: number[]) => void }; up: { set: (...v: number[]) => void }; lookAt: (...v: number[]) => void; updateProjectionMatrix: () => void }
    perspective.fov = pose.fov
    perspective.near = pose.near
    perspective.far = pose.far
    perspective.position.set(...pose.position)
    perspective.up.set(...pose.up)
    perspective.lookAt(...pose.target)
    perspective.updateProjectionMatrix()
    const orbit = controls as { target?: { set: (...v: number[]) => void }; update?: () => void } | null
    orbit?.target?.set(...pose.target)
    orbit?.update?.()
    invalidate()
  }, [bounds, camera, controls, fitToken, invalidate, size.height, size.width, viewName])
  return null
}

class MeshErrorBoundary extends Component<{ children: ReactNode; onError: () => void }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true }
  }

  componentDidCatch(): void {
    this.props.onError()
  }

  render() {
    if (this.state.failed) return null
    return this.props.children
  }
}
