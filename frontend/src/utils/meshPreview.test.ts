import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import {
  DOWNSAMPLE_WARNING,
  FACE_NORMALS_ARE_VERTEX_ATTRIBUTES,
  MESH_QUALITIES,
  MISSING_PHYSICAL_MESSAGE,
  PREVIEW_ONLY_NOTICE,
  STALE_GEOMETRY_NOTE,
  UNKNOWN_GEOMETRY_NOTE,
  WEBGL_UNAVAILABLE_MESSAGE,
  BOUNDARY_GUIDE_OFFSET_MM,
  STOCK_BOUNDARY_MISMATCH,
  STOCK_BOUNDARY_UNAVAILABLE,
  artworkLocalRect,
  decodeRfmp,
  displayBoundaryCorners,
  displayHandedness,
  displayVertexNormals,
  fitCamera,
  stockBoundaryGuide,
  DEFAULT_PREVIEW_LIGHTING,
  PREVIEW_LIGHT_LIMITS,
  defaultPreviewLighting,
  formatMeshDimension,
  formatMeshRange,
  formatPreviewLight,
  previewLightValue,
  frontFaceToward,
  meshLayout,
  meshPreviewFetchToken,
  meshPreviewGenerationChange,
  meshPreviewPath,
  meshPreviewRequestIdentity,
  meshPreviewRequestKey,
  meshPreviewShouldFetch,
  meshPreviewView,
  meshProvenanceChain,
  meshQuality,
  previewAriaStatus,
  previewGeometryCaution,
  previewVerifiedCurrent,
  reliefShadingSample,
  retainReliefGeometry,
  retainReliefMaterial,
  webglStatus,
  provenanceLabel,
  provenanceNote,
  shouldApplyMeshResult,
  toDisplayPoint,
  toDisplayPositions,
  vertexAttributePlan,
} from "./meshPreview.ts"

const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")
const api = readFileSync(new URL("../api/projects.ts", import.meta.url), "utf8")
const client = readFileSync(new URL("../api/client.ts", import.meta.url), "utf8")
const workspace = readFileSync(new URL("../pages/WorkspacePage.tsx", import.meta.url), "utf8")
const section = readFileSync(new URL("../components/MeshPreviewSection.tsx", import.meta.url), "utf8")

function writeToken(view: DataView, offset: number, size: number, text: string) {
  for (let index = 0; index < size; index += 1) view.setUint8(offset + index, 0)
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index))
  }
}

function encodeFixture(options?: { status?: number; flags?: number; topology?: number }) {
  const meshWidth = 2
  const meshHeight = 2
  const vertexCount = 8
  const triangleCount = 12
  const positionsBytes = vertexCount * 12
  const indicesBytes = triangleCount * 12
  const normalsBytes = triangleCount * 12
  const payload = 256 + positionsBytes + indicesBytes + normalsBytes
  const buffer = new ArrayBuffer(payload)
  const view = new DataView(buffer)
  view.setUint8(0, 82)
  view.setUint8(1, 70)
  view.setUint8(2, 77)
  view.setUint8(3, 80)
  view.setUint32(4, 1, true)
  view.setUint32(8, 1, true)
  view.setUint32(12, 256, true)
  view.setUint32(16, vertexCount, true)
  view.setUint32(20, triangleCount, true)
  view.setUint32(24, 3, true)
  view.setUint32(28, 3, true)
  view.setUint32(32, 3, true)
  view.setUint32(36, options?.flags ?? 4, true)
  view.setUint32(40, options?.status ?? 2, true)
  view.setUint32(44, 2, true)
  view.setUint32(48, 2, true)
  view.setUint32(52, meshWidth, true)
  view.setUint32(56, meshHeight, true)
  view.setFloat32(60, 10, true)
  view.setFloat32(64, 20, true)
  view.setFloat32(68, 0, true)
  view.setFloat32(72, 0, true)
  view.setFloat32(76, 0, true)
  view.setFloat32(80, 10, true)
  view.setFloat32(84, 20, true)
  view.setFloat32(88, 5, true)
  view.setUint32(92, 256, true)
  view.setUint32(96, positionsBytes, true)
  view.setUint32(100, 256 + positionsBytes, true)
  view.setUint32(104, indicesBytes, true)
  view.setUint32(108, 256 + positionsBytes + indicesBytes, true)
  view.setUint32(112, normalsBytes, true)
  view.setUint32(116, payload, true)
  view.setUint32(120, options?.topology ?? 1, true)
  view.setUint32(124, 0, true)
  view.setUint32(128, 50000, true)
  view.setUint32(132, 1, true)
  writeToken(view, 136, 64, "artwork-xy-y-down-z-up")
  writeToken(view, 200, 32, "face")
  const corners = [
    [0, 0, 5],
    [10, 0, 5],
    [0, 20, 5],
    [10, 20, 5],
    [0, 0, 0],
    [10, 0, 0],
    [0, 20, 0],
    [10, 20, 0],
  ]
  corners.forEach((point, vertex) => {
    point.forEach((value, axis) => {
      view.setFloat32(256 + (vertex * 3 + axis) * 4, value, true)
    })
  })
  const indexOffset = 256 + positionsBytes
  for (let index = 0; index < triangleCount * 3; index += 1) {
    view.setUint32(indexOffset + index * 4, index % vertexCount, true)
  }
  const normalOffset = indexOffset + indicesBytes
  for (let index = 0; index < triangleCount; index += 1) {
    view.setFloat32(normalOffset + index * 12 + 8, 1, true)
  }
  return buffer
}

describe("RFMP decoder", () => {
  it("decodes a valid payload, including a byteOffset view", () => {
    const payload = encodeFixture()
    const decoded = decodeRfmp(payload)
    assert.equal(decoded.version, 1)
    assert.equal(decoded.vertexCount, 8)
    assert.equal(decoded.triangleCount, 12)
    assert.equal(decoded.status, "current")
    assert.equal(decoded.exportReady, false)
    assert.equal(decoded.normalKind, "face")
    assert.equal(decoded.positions[0], 0)
    assert.equal(decoded.positions[2], 5)
    const padded = new Uint8Array(payload.byteLength + 8)
    padded.set(new Uint8Array(payload), 8)
    const wrapped = new Uint8Array(padded.buffer, 8, payload.byteLength)
    const fromView = decodeRfmp(wrapped)
    assert.equal(fromView.positions[2], 5)
    assert.equal(fromView.indices.length, 36)
  })

  it("rejects invalid magic, version, and endianness", () => {
    const magic = encodeFixture()
    new DataView(magic).setUint8(0, 88)
    assert.throws(() => decodeRfmp(magic), /magic is not RFMP/)
    const version = encodeFixture()
    new DataView(version).setUint32(4, 2, true)
    assert.throws(() => decodeRfmp(version), /version is not 1/)
    const endian = encodeFixture()
    new DataView(endian).setUint32(8, 1, false)
    assert.throws(() => decodeRfmp(endian), /endian marker/)
  })

  it("rejects invalid offsets, counts, and truncated or trailing bytes", () => {
    const offsets = encodeFixture()
    new DataView(offsets).setUint32(92, 260, true)
    assert.throws(() => decodeRfmp(offsets), /offsets or counts/)
    const truncated = encodeFixture().slice(0, 400)
    assert.throws(() => decodeRfmp(truncated), /trailing or truncated|shorter than/)
    const trailing = new Uint8Array(encodeFixture().byteLength + 4)
    trailing.set(new Uint8Array(encodeFixture()))
    assert.throws(() => decodeRfmp(trailing.buffer), /trailing or truncated/)
  })

  it("rejects out-of-range indices and non-finite geometry", () => {
    const indices = encodeFixture()
    new DataView(indices).setUint32(256 + 96, 99, true)
    assert.throws(() => decodeRfmp(indices), /outside the vertex range/)
    const finite = encodeFixture()
    new DataView(finite).setFloat32(256, Number.NaN, true)
    assert.throws(() => decodeRfmp(finite), /non-finite/)
    const ready = encodeFixture()
    new DataView(ready).setUint32(124, 1, true)
    assert.throws(() => decodeRfmp(ready), /export_ready/)
  })

  it("decodes current, stale, and unknown provenance", () => {
    assert.equal(decodeRfmp(encodeFixture({ status: 2 })).status, "current")
    assert.equal(decodeRfmp(encodeFixture({ status: 3 })).status, "stale")
    assert.equal(decodeRfmp(encodeFixture({ status: 1 })).status, "unknown")
    const unknown = encodeFixture()
    new DataView(unknown).setUint32(40, 0, true)
    assert.throws(() => decodeRfmp(unknown), /provenance/)
    assert.equal(decodeRfmp(encodeFixture({ flags: 5 })).downsamplingRequired, true)
  })
})

describe("display transform and normals", () => {
  it("rotates axes without changing handedness or source data", () => {
    assert.equal(displayHandedness(), 1)
    assert.deepEqual(toDisplayPoint(4, 7, 2), [4, 2, -7])
    const source = Float32Array.from([1, 2, 3])
    const display = toDisplayPositions(source)
    assert.deepEqual(Array.from(source), [1, 2, 3])
    assert.deepEqual(Array.from(display), [1, 3, -2])
    const top = [
      toDisplayPoint(0, 0, 1),
      toDisplayPoint(1, 0, 1),
      toDisplayPoint(0, 1, 1),
    ]
    const e1 = [top[1][0] - top[0][0], top[1][1] - top[0][1], top[1][2] - top[0][2]]
    const e2 = [top[2][0] - top[0][0], top[2][1] - top[0][1], top[2][2] - top[0][2]]
    const normal: [number, number, number] = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ]
    assert.ok(Math.abs(normal[0]) < 1e-9)
    assert.ok(Math.abs(normal[1] - 1) < 1e-9)
    assert.ok(Math.abs(normal[2]) < 1e-9)
    const centroid: [number, number, number] = [1 / 3, 1, -1 / 3]
    assert.equal(frontFaceToward(normal, centroid, [0, 3, 0]), true)
    assert.equal(frontFaceToward(normal, centroid, [0, -2, 0]), false)
  })

  it("computes vertex normals and does not reuse face normals", () => {
    const plan = vertexAttributePlan()
    assert.equal(FACE_NORMALS_ARE_VERTEX_ATTRIBUTES, false)
    assert.equal(plan.faceNormalBinding, "rejected")
    assert.equal(plan.normal, "computed-display-vertex-normals")
    const positions = Float32Array.from([0, 1, 0, 1, 1, 0, 0, 1, -1, -1, 0, 0])
    const indices = Uint32Array.from([0, 1, 2, 0, 3, 2])
    const normals = displayVertexNormals(positions, indices)
    assert.ok(Math.abs(normals[0] - 1) < 1e-6)
    assert.ok(Math.abs(normals[1]) < 1e-6)
    assert.ok(Math.abs(normals[2]) < 1e-6)
    assert.notDeepEqual([normals[0], normals[1], normals[2]], [0, 1, 0])
  })
})

describe("camera fit and quality", () => {
  it("fits thin and thick reliefs without extreme clip planes", () => {
    for (const bounds of [
      { min: [0, 0, 0] as [number, number, number], max: [200, 0.2, 100] as [number, number, number] },
      { min: [0, 0, 0] as [number, number, number], max: [10, 80, 10] as [number, number, number] },
    ]) {
      const pose = fitCamera(bounds, 16 / 9, "perspective")
      const radius = 0.5 * Math.hypot(
        bounds.max[0] - bounds.min[0],
        bounds.max[1] - bounds.min[1],
        bounds.max[2] - bounds.min[2],
      )
      const distance = Math.hypot(
        pose.position[0] - pose.target[0],
        pose.position[1] - pose.target[1],
        pose.position[2] - pose.target[2],
      )
      assert.ok(pose.near >= 0.05)
      assert.ok(pose.far > pose.near)
      assert.ok(pose.far / pose.near <= 20_000)
      assert.ok(distance > radius)
    }
    const box = { min: [0, 0, -20] as [number, number, number], max: [10, 5, 0] as [number, number, number] }
    const top = fitCamera(box, 1, "top")
    const front = fitCamera(box, 1, "front")
    const side = fitCamera(box, 1, "side")
    const perspective = fitCamera(box, 1, "perspective")
    assert.ok(top.position[1] > top.target[1])
    assert.deepEqual(top.up, [0, 0, 1])
    assert.ok(front.position[2] < front.target[2])
    assert.ok(side.position[0] > side.target[0])
    assertClose(unitOffset(top), [0, 1, 0])
    assertClose(unitOffset(front), [0, 0, -1])
    assertClose(unitOffset(side), [1, 0, 0])
    assertClose(unitOffset(perspective), unit([1, 0.85, 1]))
    assert.deepEqual(perspective.up, [0, 1, 0])
  })

  it("fits an isometric diagonal without roll and keeps the relief inside the frustum", () => {
    const views = ["perspective", "top", "front", "side", "isometric"] as const
    assert.equal(views.includes("isometric"), true)
    const bounds = { min: [0, 0, -40] as [number, number, number], max: [80, 25, 0] as [number, number, number] }
    const center = bounds.min.map((value, index) => (value + bounds.max[index]) / 2)
    const first = fitCamera(bounds, 16 / 9, "isometric")
    const again = fitCamera(bounds, 16 / 9, "isometric")
    assert.deepEqual(again, first)
    assert.deepEqual(first.target, center)
    assert.deepEqual(first.up, [0, 1, 0])
    const direction = unitOffset(first)
    assert.ok(Math.abs(Math.abs(direction[0]) - Math.abs(direction[1])) < 1e-9)
    assert.ok(Math.abs(Math.abs(direction[1]) - Math.abs(direction[2])) < 1e-9)
    assertClose(direction, unit([1, 1, -1]))
    const frame = cameraFrame(first)
    assert.ok(Math.abs(frame.right[1]) < 1e-9)
    assert.ok(frame.up[1] > 0.5)
    assert.ok(first.position[1] > first.target[1])
    assert.ok(first.near >= 0.05)
    assert.ok(first.far > first.near)
    assert.ok(first.far / first.near <= 20_000)
    const perspective = fitCamera(bounds, 16 / 9, "perspective")
    assert.ok(Math.abs(distanceOf(first) - distanceOf(perspective)) < 1e-9)
    assert.equal(first.near, perspective.near)
    assert.equal(first.far, perspective.far)
    assert.equal(first.fov, perspective.fov)
    const boxes = [
      bounds,
      { min: [0, 0, 0] as [number, number, number], max: [300, 12, 180] as [number, number, number] },
      { min: [0, 0, 0] as [number, number, number], max: [20, 90, 15] as [number, number, number] },
      { min: [-10, 2, -80] as [number, number, number], max: [40, 30, 5] as [number, number, number] },
    ]
    for (const box of boxes) {
      for (const aspect of [16 / 9, 4 / 3, 1, 3 / 4, 9 / 16, 2.5]) {
        const pose = fitCamera(box, aspect, "isometric")
        for (const corner of corners(box)) assert.equal(insideFrustum(pose, corner, aspect), true)
      }
    }
    const reset = section.slice(section.lastIndexOf("setViewName(\"perspective\")", section.indexOf("Reset")), section.indexOf("Reset"))
    assert.match(reset, /setViewName\("perspective"\)/)
    assert.doesNotMatch(reset, /isometric/)
    assert.match(section, /useState<CameraViewName>\("perspective"\)/)
    assert.match(section, /chooseView\("isometric", setViewName, setFitToken\)/)
    assert.match(section, />\s*Isometric\s*</)
    const effect = section.slice(section.indexOf("if (!meshPreviewShouldFetch"), section.indexOf("}, [fetchToken, webgl]"))
    assert.doesNotMatch(effect, /viewName|isometric/)
  })

  it("formats inspector lengths from canonical millimeters without touching counts", () => {
    assert.equal(formatMeshDimension(25.4, "in"), "1 in")
    assert.equal(formatMeshDimension(279.4, "in"), "11 in")
    assert.equal(formatMeshDimension(355.6, "in"), "14 in")
    assert.equal(formatMeshDimension(35.56, "in"), "1.4 in")
    assert.equal(formatMeshDimension(25.4, "mm"), "25.4 mm")
    assert.equal(formatMeshDimension(279.4, "mm"), "279.4 mm")
    assert.equal(formatMeshDimension(355.6, "mm"), "355.6 mm")
    assert.equal(formatMeshDimension(35.56, "mm"), "35.56 mm")
    assert.equal(formatMeshDimension(new Float32Array([279.4])[0], "mm"), "279.4 mm")
    assert.equal(formatMeshDimension(new Float32Array([279.4])[0], "in"), "11 in")
    assert.equal(formatMeshDimension(new Float32Array([355.6])[0], "mm"), "355.6 mm")
    assert.equal(formatMeshDimension(new Float32Array([355.6])[0], "in"), "14 in")
    assert.equal(formatMeshDimension(new Float32Array([35.56])[0], "mm"), "35.56 mm")
    assert.equal(formatMeshDimension(new Float32Array([35.56])[0], "in"), "1.4 in")
    assert.equal(formatMeshDimension(new Float32Array([258.839])[0], "mm"), "258.839 mm")
    assert.equal(formatMeshDimension(new Float32Array([258.839])[0], "in"), "10.1905 in")
    assert.equal(formatMeshDimension(0, "in"), "0 in")
    assert.equal(formatMeshDimension(Number.NaN, "mm"), "—")
    assert.equal(formatMeshRange(0, 35.56, "in"), "0 in to 1.4 in")
    assert.equal(formatMeshRange(0, 35.56, "mm"), "0 mm to 35.56 mm")
    const canonical = 279.4
    let unit: "in" | "mm" = "mm"
    for (let step = 0; step < 8; step += 1) {
      const shown = formatMeshDimension(canonical, unit)
      assert.equal(shown, unit === "mm" ? "279.4 mm" : "11 in")
      unit = unit === "mm" ? "in" : "mm"
    }
    assert.equal(canonical, 279.4)
    assert.match(section, /useUnits\(\)/)
    assert.match(section, /formatMeshDimension\(decoded\.artworkWidthMm, unit\)/)
    assert.match(section, /formatMeshDimension\(decoded\.artworkHeightMm, unit\)/)
    assert.match(section, /formatMeshRange\(decoded\.boundingBox\.min\[2\], decoded\.boundingBox\.max\[2\], unit\)/)
    assert.match(section, /vertexCount\.toLocaleString/)
    assert.match(section, /triangleCount\.toLocaleString/)
    assert.match(section, /sourceWidthPx/)
    assert.match(section, /topologyLabel/)
    assert.match(section, /Export-ready/)
    assert.doesNotMatch(section, /formatMillimeters/)
    assert.match(section, /Artwork width/)
    assert.match(section, /\}, \[fetchToken, webgl\]\)/)
    assert.match(section, /useMemo\(\(\) => toDisplayPositions\(decoded\.positions\), \[decoded\]\)/)
    assert.match(section, /\[bounds, camera, controls, fitToken, invalidate, size\.height, size\.width, viewName\]/)
    assert.match(section, /chooseView\("isometric", setViewName, setFitToken\)/)
    assert.match(section, /setViewName\("perspective"\)/)
    assert.match(section, /useState<MeshQualityId>\("standard"\)/)
  })

  it("places stock and artwork guides without moving mesh vertices or the camera fit", () => {
    const wedding = stockBoundaryGuide({
      stockWidthMm: 304.8,
      stockHeightMm: 304.8,
      artworkWidthMm: 279.4,
      artworkHeightMm: 279.4,
      centerXmm: 152.4,
      centerYmm: 152.4,
      meshArtworkWidthMm: 279.4,
      meshArtworkHeightMm: 279.4,
    })
    assert.equal(wedding.available, true)
    if (!wedding.available) return
    assertClose([wedding.leftMm, wedding.rightMm, wedding.frontMm, wedding.backMm], [12.7, 12.7, 12.7, 12.7])
    assertClose(
      [wedding.bounds.xMin, wedding.bounds.xMax, wedding.bounds.yMin, wedding.bounds.yMax],
      [-12.7, 292.1, -12.7, 292.1],
    )
    const angie = stockBoundaryGuide({
      stockWidthMm: 381,
      stockHeightMm: 279.4,
      artworkWidthMm: 355.6,
      artworkHeightMm: 258.839,
      centerXmm: 190.5,
      centerYmm: 139.7,
      meshArtworkWidthMm: new Float32Array([355.6])[0],
      meshArtworkHeightMm: new Float32Array([258.839])[0],
    })
    assert.equal(angie.available, true)
    if (!angie.available) return
    assertClose([angie.leftMm, angie.rightMm, angie.frontMm, angie.backMm], [12.7, 12.7, 10.2805, 10.2805])
    assertClose(
      [angie.bounds.xMin, angie.bounds.xMax, angie.bounds.yMin, angie.bounds.yMax],
      [-12.7, 368.3, -10.2805, 269.1195],
    )
    const shifted = stockBoundaryGuide({
      stockWidthMm: 100,
      stockHeightMm: 80,
      artworkWidthMm: 40,
      artworkHeightMm: 20,
      centerXmm: 30,
      centerYmm: 50,
      meshArtworkWidthMm: 40,
      meshArtworkHeightMm: 20,
    })
    assert.equal(shifted.available, true)
    if (!shifted.available) return
    assertClose([shifted.leftMm, shifted.rightMm, shifted.frontMm, shifted.backMm], [10, 50, 40, 20])
    assert.equal(
      stockBoundaryGuide({
        stockWidthMm: 100,
        stockHeightMm: 80,
        artworkWidthMm: 40,
        artworkHeightMm: 20,
        centerXmm: null,
        centerYmm: 40,
        meshArtworkWidthMm: 40,
        meshArtworkHeightMm: 20,
      }).available,
      false,
    )
    assert.equal(
      stockBoundaryGuide({
        stockWidthMm: 100,
        stockHeightMm: 80,
        artworkWidthMm: 40,
        artworkHeightMm: 20,
        centerXmm: null,
        centerYmm: 40,
        meshArtworkWidthMm: 40,
        meshArtworkHeightMm: 20,
      }).reason,
      STOCK_BOUNDARY_UNAVAILABLE,
    )
    assert.equal(
      stockBoundaryGuide({
        stockWidthMm: 100,
        stockHeightMm: 80,
        artworkWidthMm: 40,
        artworkHeightMm: 20,
        centerXmm: 50,
        centerYmm: 40,
        meshArtworkWidthMm: 41,
        meshArtworkHeightMm: 20,
      }).reason,
      STOCK_BOUNDARY_MISMATCH,
    )
    const artwork = artworkLocalRect(279.4, 279.4)
    assert.ok(artwork)
    if (!artwork) return
    const displayed = displayBoundaryCorners(artwork)
    assert.equal(BOUNDARY_GUIDE_OFFSET_MM, 0.2)
    assertClose(displayed[0], [0, -0.2, -0])
    assertClose(displayed[1], [279.4, -0.2, -0])
    assertClose(displayed[2], [279.4, -0.2, -279.4])
    assertClose(displayed[3], [0, -0.2, -279.4])
    const stockDisplay = displayBoundaryCorners(wedding.bounds)
    assertClose(stockDisplay[0], [-12.7, -0.2, 12.7])
    assertClose(stockDisplay[2], [292.1, -0.2, -292.1])
    const positions = Float32Array.from([0, 0, 0, 10, 4, 2])
    const before = Array.from(positions)
    const display = toDisplayPositions(positions)
    stockBoundaryGuide({
      stockWidthMm: 20,
      stockHeightMm: 12,
      artworkWidthMm: 10,
      artworkHeightMm: 4,
      centerXmm: 10,
      centerYmm: 6,
      meshArtworkWidthMm: 10,
      meshArtworkHeightMm: 4,
    })
    assert.deepEqual(Array.from(positions), before)
    assert.deepEqual(Array.from(display), Array.from(toDisplayPositions(positions)))
    const relief = {
      min: [0, 0, -279.4] as [number, number, number],
      max: [279.4, 35.56, 0] as [number, number, number],
    }
    for (const aspect of [16 / 9, 1, 0.45]) {
      const pose = fitCamera(relief, aspect, "top")
      for (const corner of stockDisplay) assert.equal(insideFrustum(pose, corner, aspect), true)
    }
    const wide = stockBoundaryGuide({
      stockWidthMm: 400,
      stockHeightMm: 400,
      artworkWidthMm: 100,
      artworkHeightMm: 100,
      centerXmm: 200,
      centerYmm: 200,
      meshArtworkWidthMm: 100,
      meshArtworkHeightMm: 100,
    })
    assert.equal(wide.available, true)
    if (!wide.available) return
    const widePose = fitCamera(
      { min: [0, 0, -100] as [number, number, number], max: [100, 20, 0] as [number, number, number] },
      1,
      "top",
    )
    const wideInside = displayBoundaryCorners(wide.bounds).every((corner) => insideFrustum(widePose, corner, 1))
    assert.equal(wideInside, false)
    assert.match(section, /Show Stock Boundary/)
    assert.match(section, /Show Artwork Boundary/)
    assert.match(section, /useState\(true\)/)
    assert.match(section, /stockBoundaryGuide\(/)
    assert.match(section, /displayBoundaryCorners\(/)
    assert.match(section, /formatMeshDimension\(stockGuide\.leftMm, unit\)/)
    assert.match(section, /Not part of the exported solid/)
    assert.match(section, /\}, \[fetchToken, webgl\]\)/)
    assert.match(section, /useMemo\(\(\) => toDisplayPositions\(decoded\.positions\), \[decoded\]\)/)
    assert.match(section, /\[bounds, camera, controls, fitToken, invalidate, size\.height, size\.width, viewName\]/)
    const mesh = section.slice(section.indexOf("function ReliefMesh"))
    assert.match(mesh, /\[indices, positions\]/)
    assert.doesNotMatch(mesh.slice(0, mesh.indexOf("function CameraRig")), /stockBoundaryGuide|displayBoundaryCorners/)
  })

  it("keeps preview lighting at the exact defaults without touching geometry or the preview request", () => {
    assert.equal(DEFAULT_PREVIEW_LIGHTING.ambient, 0.58)
    assert.equal(DEFAULT_PREVIEW_LIGHTING.key, 1.15)
    assert.equal(DEFAULT_PREVIEW_LIGHTING.fill, 0.28)
    assert.equal(formatPreviewLight(0.58), "0.58")
    assert.equal(formatPreviewLight(1.15), "1.15")
    assert.equal(formatPreviewLight(0.28), "0.28")
    const initial = defaultPreviewLighting()
    assert.deepEqual(initial, { ambient: 0.58, key: 1.15, fill: 0.28 })
    assert.equal(previewLightValue("ambient", initial.ambient), 0.58)
    assert.equal(previewLightValue("key", initial.key), 1.15)
    assert.equal(previewLightValue("fill", initial.fill), 0.28)
    assert.equal(previewLightValue("ambient", -1), 0)
    assert.equal(previewLightValue("ambient", 2.4), 2)
    assert.equal(previewLightValue("key", 3.2), 3)
    assert.equal(previewLightValue("key", 1.15), 1.15)
    assert.equal(previewLightValue("fill", 2.01), 2)
    assert.equal(previewLightValue("fill", Number.NaN), 0.28)
    for (const channel of ["ambient", "key", "fill"] as const) {
      const limits = PREVIEW_LIGHT_LIMITS[channel]
      const value = DEFAULT_PREVIEW_LIGHTING[channel]
      assert.equal(limits.step, 0.01)
      assert.ok(value >= limits.min && value <= limits.max)
      assert.ok(Math.abs(value / limits.step - Math.round(value / limits.step)) < 1e-8)
    }
    const reset = defaultPreviewLighting()
    reset.ambient = previewLightValue("ambient", 1.2)
    reset.key = previewLightValue("key", 2)
    reset.fill = previewLightValue("fill", 0)
    assert.deepEqual(defaultPreviewLighting(), { ambient: 0.58, key: 1.15, fill: 0.28 })
    assert.match(section, /Ambient Light/)
    assert.match(section, /Key Light/)
    assert.match(section, /Fill Light/)
    assert.match(section, /Reset Lighting/)
    assert.match(section, /useState<PreviewLighting>\(defaultPreviewLighting\)/)
    assert.match(section, /setLighting\(defaultPreviewLighting\(\)\)/)
    assert.match(section, /step=\{limits\.step\}/)
    assert.match(section, /<ambientLight intensity=\{lighting\.ambient\} \/>/)
    assert.match(section, /position=\{\[0\.6, 1\.4, 0\.8\]\} intensity=\{lighting\.key\}/)
    assert.match(section, /position=\{\[-0\.8, 0\.4, -0\.4\]\} intensity=\{lighting\.fill\}/)
    assert.match(section, /frameloop="demand"/)
    assert.match(section, /\}, \[fetchToken, webgl\]\)/)
    assert.match(section, /useMemo\(\(\) => toDisplayPositions\(decoded\.positions\), \[decoded\]\)/)
    assert.match(section, /\[indices, positions\]/)
    assert.match(section, /\[bounds, camera, controls, fitToken, invalidate, size\.height, size\.width, viewName\]/)
    assert.doesNotMatch(section, /localStorage[\s\S]{0,80}lighting|lighting[\s\S]{0,40}project\.json/)
    assert.doesNotMatch(section.slice(section.indexOf("function ReliefMesh"), section.indexOf("function PreviewSession")), /lighting/)
  })

  it("builds quality URLs under the server cap and ignores stale results", () => {
    assert.equal(meshQuality("standard").maxVertices, 50_000)
    assert.notEqual(MESH_QUALITIES[0].id, "maximum")
    for (const item of MESH_QUALITIES) {
      assert.ok(item.maxVertices <= 250_000)
      assert.match(meshPreviewPath("proj 1", item.maxVertices), new RegExp(`max_vertices=${item.maxVertices}`))
    }
    assert.equal(
      shouldApplyMeshResult({
        requestId: 2,
        currentId: 2,
        projectId: "a",
        requestedProjectId: "a",
        requestedKey: "same",
        activeKey: "same",
      }),
      true,
    )
    assert.equal(
      shouldApplyMeshResult({
        requestId: 1,
        currentId: 2,
        projectId: "a",
        requestedProjectId: "a",
        requestedKey: "same",
        activeKey: "same",
      }),
      false,
    )
    assert.equal(
      shouldApplyMeshResult({
        requestId: 2,
        currentId: 2,
        projectId: "b",
        requestedProjectId: "a",
        requestedKey: "same",
        activeKey: "same",
      }),
      false,
    )
  })

  it("refreshes the preview from the physical generation and keeps provenance authoritative", () => {
    const physical = {
      generated_at: "2026-10-08T00:00:00.000Z",
      status: "current",
      source_relief_generated_at: "2026-10-08T00:00:00.000Z",
      source_relief_sha256: "relief-sha",
      content_sha256: "height-sha",
      settings_sha256: "settings-sha",
    }
    const first = meshPreviewRequestIdentity({ projectId: "wedding", maxVertices: 50_000, physical })
    const rerender = meshPreviewRequestIdentity({
      projectId: "wedding",
      maxVertices: 50_000,
      physical: { ...physical },
    })
    const firstKey = meshPreviewRequestKey(first)
    const rerenderKey = meshPreviewRequestKey(rerender)
    assert.equal(firstKey, rerenderKey)
    assert.equal(meshPreviewFetchToken(firstKey, 0), meshPreviewFetchToken(rerenderKey, 0))
    assert.equal(meshPreviewGenerationChange(firstKey, rerenderKey).changed, false)
    assert.equal(meshPreviewShouldFetch({ requestKey: firstKey, webgl: "available" }), true)

    const regenerated = meshPreviewRequestIdentity({
      projectId: "wedding",
      maxVertices: 50_000,
      physical: { ...physical, generated_at: "2026-10-12T00:00:00.000Z" },
    })
    const regeneratedKey = meshPreviewRequestKey(regenerated)
    assert.notEqual(regeneratedKey, firstKey)
    const regeneratedToken = meshPreviewFetchToken(regeneratedKey, 0)
    const firstToken = meshPreviewFetchToken(firstKey, 0)
    assert.equal(meshPreviewGenerationChange(firstToken, regeneratedToken).clearGeometry, true)
    assert.equal(meshPreviewGenerationChange(firstToken, regeneratedToken).phase, "loading")
    assert.equal(
      shouldApplyMeshResult({
        requestId: 4,
        currentId: 4,
        projectId: "wedding",
        requestedProjectId: "wedding",
        requestedKey: firstToken ?? "",
        activeKey: regeneratedToken,
      }),
      false,
    )
    assert.equal(
      shouldApplyMeshResult({
        requestId: 3,
        currentId: 4,
        projectId: "wedding",
        requestedProjectId: "wedding",
        requestedKey: firstToken ?? "",
        activeKey: regeneratedToken,
        aborted: false,
      }),
      false,
    )

    const removed = meshPreviewRequestIdentity({ projectId: "wedding", maxVertices: 50_000, physical: null })
    assert.equal(removed.available, false)
    assert.equal(removed.physicalStatus, "missing")
    assert.equal(meshPreviewRequestKey(removed), null)
    assert.equal(meshPreviewShouldFetch({ requestKey: null, webgl: "available" }), false)
    assert.equal(meshPreviewGenerationChange(firstToken, null).clearGeometry, true)
    assert.equal(meshPreviewGenerationChange(firstToken, null).phase, "idle")
    assert.equal(
      shouldApplyMeshResult({
        requestId: 5,
        currentId: 5,
        projectId: "wedding",
        requestedProjectId: "wedding",
        requestedKey: firstToken ?? "",
        activeKey: null,
      }),
      false,
    )

    const restored = meshPreviewRequestIdentity({ projectId: "wedding", maxVertices: 50_000, physical })
    const restoredKey = meshPreviewRequestKey(restored)
    assert.equal(meshPreviewShouldFetch({ requestKey: restoredKey, webgl: "available" }), true)
    assert.equal(meshPreviewGenerationChange(null, meshPreviewFetchToken(restoredKey, 0)).phase, "loading")

    const staleRecord = meshPreviewRequestIdentity({
      projectId: "wedding",
      maxVertices: 50_000,
      physical: { ...physical, status: "stale" },
    })
    assert.equal(meshPreviewRequestKey(staleRecord), firstKey)
    assert.equal(staleRecord.physicalStatus, "stale")
    assert.equal(previewGeometryCaution("current", "current"), null)
    assert.equal(previewVerifiedCurrent("current", "current"), true)
    assert.equal(previewGeometryCaution("current", "stale"), STALE_GEOMETRY_NOTE)
    assert.equal(previewVerifiedCurrent("current", "stale"), false)
    assert.equal(previewGeometryCaution("stale", "current"), STALE_GEOMETRY_NOTE)
    assert.equal(previewGeometryCaution("unknown", "current"), UNKNOWN_GEOMETRY_NOTE)
    assert.equal(previewGeometryCaution("current", "unknown"), UNKNOWN_GEOMETRY_NOTE)
    assert.equal(previewAriaStatus("current", "stale"), "Stale")
    assert.equal(previewAriaStatus("current", "current"), "Current")
    assert.equal(meshPreviewGenerationChange(firstToken, meshPreviewFetchToken(meshPreviewRequestKey(staleRecord), 0)).changed, false)

    const chain = meshProvenanceChain({
      source: { status: "current" },
      depth: { status: "current", conditioning: { status: "current" } },
      relief: { status: "current", physical },
    } as Parameters<typeof meshProvenanceChain>[0])
    assert.deepEqual(
      chain.map((item) => [item.title, item.status]),
      [
        ["Source", "current"],
        ["Raw depth", "current"],
        ["Conditioned depth", "current"],
        ["Composed relief", "current"],
        ["Physical height", "current"],
      ],
    )
    const unverified = meshProvenanceChain({
      source: { status: null },
      depth: { status: "unknown" },
      relief: { status: "stale", physical: { ...physical, status: null } },
    } as Parameters<typeof meshProvenanceChain>[0])
    assert.equal(unverified.find((item) => item.title === "Source")?.status, "unknown")
    assert.equal(unverified.find((item) => item.title === "Raw depth")?.status, "unknown")
    assert.equal(unverified.find((item) => item.title === "Composed relief")?.status, "stale")
    assert.equal(unverified.find((item) => item.title === "Physical height")?.status, "unknown")
    assert.notEqual(unverified.find((item) => item.title === "Physical height")?.status, "current")
    const missing = meshProvenanceChain({
      source: null,
      depth: null,
      relief: null,
    })
    assert.deepEqual(
      missing.map((item) => item.status),
      ["missing", "missing", "missing", "missing", "missing"],
    )
    assert.equal(
      shouldApplyMeshResult({
        requestId: 8,
        currentId: 8,
        projectId: "angie",
        requestedProjectId: "wedding",
        requestedKey: firstToken ?? "",
        activeKey: meshPreviewFetchToken(
          meshPreviewRequestKey(meshPreviewRequestIdentity({ projectId: "angie", maxVertices: 50_000, physical })),
          0,
        ),
      }),
      false,
    )
    const detailed = meshPreviewRequestKey(
      meshPreviewRequestIdentity({ projectId: "wedding", maxVertices: 150_000, physical }),
    )
    assert.notEqual(detailed, firstKey)
    assert.equal(meshPreviewFetchToken(firstKey, 0), meshPreviewFetchToken(firstKey, 0))
    assert.notEqual(meshPreviewFetchToken(firstKey, 1), meshPreviewFetchToken(firstKey, 0))
    assert.match(section, /meshPreviewRequestIdentity\(/)
    assert.match(section, /meshProvenanceChain\(project\)/)
    assert.match(section, /ProvenanceStatus/)
    assert.match(section, /previewGeometryCaution\(decoded\.status, identity\.physicalStatus\)/)
    assert.match(section, /exportReady !== false/)
    assert.match(section, /\}, \[fetchToken, webgl\]\)/)
    assert.doesNotMatch(section, /generatePhysicalHeight|apiPost/)
    const effect = section.slice(
      section.indexOf("if (!meshPreviewShouldFetch"),
      section.indexOf("}, [fetchToken, webgl]"),
    )
    assert.doesNotMatch(effect, /viewName|wireframe|lighting|unit|showStockBoundary|showArtworkBoundary/)
    assert.match(section, /MISSING_PHYSICAL_MESSAGE/)
    assert.match(section, /Open Relief/)
    assert.match(section, /PREVIEW_ONLY_NOTICE/)
    assert.match(section, /\{trust \? <p className="field-note">\{trust\}<\/p> : null\}/)
  })
})

describe("relief presentation resources", () => {
  it("keeps position and index geometry stable across shading and wireframe", () => {
    const positions = Float32Array.from([0, 1, 0, 1, 1, 0, 0, 1, -1, -1, 0, 0])
    const indices = Uint32Array.from([0, 1, 2, 0, 3, 2])
    const face: [number, number, number] = [0, 1, 0]
    const normals = displayVertexNormals(positions, indices)
    const vertex: [number, number, number] = [normals[0], normals[1], normals[2]]
    assert.ok(Math.abs(vertex[0] - face[0]) > 1e-6 || Math.abs(vertex[1] - face[1]) > 1e-6)
    assert.deepEqual(reliefShadingSample("smooth", vertex, face), vertex)
    assert.deepEqual(reliefShadingSample("flat", vertex, face), face)

    let geometryHold: { positions: Float32Array; indices: Uint32Array; geometry: { id: number } } | null = null
    let materialHold: {
      material: { id: number; flatShading: boolean; wireframe: boolean; needsUpdate: boolean }
      flat: boolean
      wireframe: boolean
    } | null = null
    let geometryCreates = 0
    let materialCreates = 0
    let geometryDisposals = 0
    let materialDisposals = 0
    let liveGeometry: { id: number } | null = null
    let liveMaterial: { id: number; flatShading: boolean; wireframe: boolean; needsUpdate: boolean } | null = null

    const step = (nextPositions: Float32Array, nextIndices: Uint32Array, flat: boolean, wireframe: boolean) => {
      const geometry = retainReliefGeometry(geometryHold, nextPositions, nextIndices, () => {
        geometryCreates += 1
        return { id: geometryCreates }
      })
      if (liveGeometry && liveGeometry !== geometry.hold.geometry) geometryDisposals += 1
      geometryHold = geometry.hold
      liveGeometry = geometry.hold.geometry
      const material = retainReliefMaterial(
        materialHold,
        flat,
        wireframe,
        () => {
          materialCreates += 1
          return { id: materialCreates, flatShading: false, wireframe: false, needsUpdate: false }
        },
        (current, nextFlat, nextWire) => {
          current.flatShading = nextFlat
          current.wireframe = nextWire
          current.needsUpdate = true
        },
      )
      if (liveMaterial && liveMaterial !== material.hold.material) materialDisposals += 1
      materialHold = material.hold
      liveMaterial = material.hold.material
    }

    step(positions, indices, false, false)
    const geometryId = liveGeometry
    const materialId = liveMaterial
    for (const flat of [true, false, true, false]) step(positions, indices, flat, false)
    assert.equal(liveMaterial?.flatShading, false)
    assert.equal(reliefShadingSample("flat", vertex, face), face)
    assert.equal(reliefShadingSample("smooth", vertex, face), vertex)
    for (const wireframe of [true, false, true, false, true]) step(positions, indices, false, wireframe)
    assert.equal(geometryCreates, 1)
    assert.equal(materialCreates, 1)
    assert.equal(geometryDisposals, 0)
    assert.equal(materialDisposals, 0)
    assert.equal(liveGeometry, geometryId)
    assert.equal(liveMaterial, materialId)
    assert.equal(liveGeometry?.id, 1)
    assert.equal(positions[0], 0)
    assert.equal(indices[0], 0)
    assert.equal(liveMaterial?.wireframe, true)
    assert.equal(liveMaterial?.needsUpdate, true)

    const replacement = new Float32Array(positions)
    step(replacement, indices, true, false)
    assert.equal(geometryCreates, 2)
    assert.equal(geometryDisposals, 1)
    assert.equal(materialCreates, 1)
    assert.equal(materialDisposals, 0)
    assert.notEqual(liveGeometry, geometryId)
    assert.equal(liveMaterial, materialId)
    assert.equal(liveMaterial?.flatShading, true)
    geometryDisposals += 1
    materialDisposals += 1
    assert.equal(geometryDisposals, geometryCreates)
    assert.equal(materialDisposals, materialCreates)

    const physical = {
      generated_at: "2026-10-08T00:00:00.000Z",
      status: "current",
      source_relief_generated_at: "2026-10-08T00:00:00.000Z",
      source_relief_sha256: "relief",
      content_sha256: "height",
      settings_sha256: "settings",
    }
    const identity = meshPreviewRequestIdentity({ projectId: "wedding", maxVertices: 50_000, physical })
    const key = meshPreviewRequestKey(identity)
    step(replacement, indices, false, true)
    assert.equal(meshPreviewRequestKey(identity), key)
    assert.equal(meshPreviewFetchToken(key, 0), meshPreviewFetchToken(meshPreviewRequestKey(identity), 0))
    assert.equal(geometryCreates, 2)
  })
})

describe("viewer states and layout", () => {
  it("labels missing, provenance, downsampling, and WebGL failure", () => {
    assert.deepEqual(
      meshPreviewView({
        hasRecord: false,
        webgl: "available",
        phase: "idle",
        errorStatus: null,
        errorKind: null,
        errorMessage: null,
      }),
      { kind: "absent" },
    )
    assert.equal(provenanceLabel("current"), "Current")
    assert.equal(provenanceLabel("stale"), "Stale")
    assert.equal(provenanceLabel("unknown"), "Unknown")
    assert.equal(provenanceNote("stale"), STALE_GEOMETRY_NOTE)
    assert.equal(provenanceNote("unknown"), UNKNOWN_GEOMETRY_NOTE)
    assert.equal(provenanceNote("current"), null)
    assert.equal(DOWNSAMPLE_WARNING, "Reduced-resolution previews can omit narrow peaks.")
    const webgl = meshPreviewView({
      hasRecord: true,
      webgl: "unavailable",
      phase: "idle",
      errorStatus: null,
      errorKind: null,
      errorMessage: null,
    })
    assert.equal(webgl.kind, "webgl")
    assert.equal(webgl.kind === "webgl" ? webgl.message : "", WEBGL_UNAVAILABLE_MESSAGE)
    assert.equal(webglStatus(() => false), "unavailable")
    assert.equal(MISSING_PHYSICAL_MESSAGE, "Physical height has not been generated.")
    assert.equal(PREVIEW_ONLY_NOTICE, "Preview only — not validated for STL export or machining.")
  })

  it("stacks the inspector at narrow widths and keeps 340–380px on desktop", () => {
    for (const width of [1920, 1366, 1100, 961]) {
      const layout = meshLayout(width)
      assert.equal(layout.stacked, false)
      assert.equal(layout.inspectorMinPx, 340)
      assert.equal(layout.inspectorMaxPx, 380)
    }
    for (const width of [960, 900, 800]) {
      assert.equal(meshLayout(width).stacked, true)
    }
    assert.match(css, /minmax\(340px, 380px\)/)
    assert.match(css, /max-width: 960px/)
    assert.match(css, /\.mesh-workspace\.is-split/)
    assert.match(css, /\.mesh-viewport/)
  })

  it("enables the 3D tab with a GET preview and computed normals", () => {
    assert.match(workspace, /MeshPreviewSection/)
    assert.match(workspace, />\s*3D\s*</)
    assert.doesNotMatch(workspace, /FUTURE_SECTIONS/)
    assert.match(workspace, /section === "export"/)
    assert.match(workspace, /lazy\(\(\) => import\("\.\.\/components\/MeshPreviewSection\.tsx"\)\)/)
    assert.match(section, /PREVIEW_ONLY_NOTICE/)
    assert.match(section, /MISSING_PHYSICAL_MESSAGE/)
    assert.match(section, /displayVertexNormals/)
    assert.match(section, /shouldApplyMeshResult/)
    assert.match(section, /Open Relief/)
    assert.doesNotMatch(section, /generatePhysicalHeight|apiPost/)
    const mesh = section.slice(section.indexOf("function ReliefMesh"))
    assert.match(mesh, /setAttribute\("normal", new BufferAttribute\(displayVertexNormals/)
    assert.doesNotMatch(mesh, /faceNormals/)
    const getter = api.slice(api.indexOf("export function getMeshPreview"), api.indexOf("export function getPhysicalProfile"))
    assert.match(getter, /apiGetArrayBuffer/)
    assert.match(getter, /max_vertices/)
    assert.doesNotMatch(getter, /apiPost/)
    assert.doesNotMatch(api.slice(api.indexOf("export function getPhysicalDiagnostics")), /apiPost/)
    const binary = client.slice(client.indexOf("export async function apiGetArrayBuffer"))
    assert.match(binary, /method: "GET"/)
  })
})

function unit(vector: number[]): [number, number, number] {
  const length = Math.hypot(vector[0], vector[1], vector[2])
  return [vector[0] / length, vector[1] / length, vector[2] / length]
}

function unitOffset(pose: { position: number[]; target: number[] }): [number, number, number] {
  return unit([
    pose.position[0] - pose.target[0],
    pose.position[1] - pose.target[1],
    pose.position[2] - pose.target[2],
  ])
}

function distanceOf(pose: { position: number[]; target: number[] }): number {
  return Math.hypot(
    pose.position[0] - pose.target[0],
    pose.position[1] - pose.target[1],
    pose.position[2] - pose.target[2],
  )
}

function assertClose(actual: number[], expected: number[]) {
  for (let index = 0; index < expected.length; index += 1) {
    assert.ok(Math.abs(actual[index] - expected[index]) < 1e-9, `${actual} vs ${expected}`)
  }
}

function cameraFrame(pose: { position: number[]; target: number[]; up: number[] }) {
  const backward = unitOffset(pose)
  const right = unit([
    pose.up[1] * backward[2] - pose.up[2] * backward[1],
    pose.up[2] * backward[0] - pose.up[0] * backward[2],
    pose.up[0] * backward[1] - pose.up[1] * backward[0],
  ])
  const up = [
    backward[1] * right[2] - backward[2] * right[1],
    backward[2] * right[0] - backward[0] * right[2],
    backward[0] * right[1] - backward[1] * right[0],
  ]
  return { right, up }
}

function corners(bounds: { min: number[]; max: number[] }): number[][] {
  const points = []
  for (const x of [bounds.min[0], bounds.max[0]]) {
    for (const y of [bounds.min[1], bounds.max[1]]) {
      for (const z of [bounds.min[2], bounds.max[2]]) points.push([x, y, z])
    }
  }
  return points
}

function insideFrustum(
  pose: { position: number[]; target: number[]; up: number[]; near: number; far: number; fov: number },
  point: number[],
  aspect: number,
): boolean {
  const frame = cameraFrame(pose)
  const offset = [
    point[0] - pose.position[0],
    point[1] - pose.position[1],
    point[2] - pose.position[2],
  ]
  const backward = unitOffset(pose)
  const depth = -(offset[0] * backward[0] + offset[1] * backward[1] + offset[2] * backward[2])
  const x = offset[0] * frame.right[0] + offset[1] * frame.right[1] + offset[2] * frame.right[2]
  const y = offset[0] * frame.up[0] + offset[1] * frame.up[1] + offset[2] * frame.up[2]
  const vertical = Math.tan(((pose.fov * Math.PI) / 180) / 2)
  const safeAspect = Number.isFinite(aspect) && aspect > 0.05 ? aspect : 1
  const horizontal = vertical * safeAspect
  if (!(depth > pose.near && depth < pose.far)) return false
  return Math.abs(x) <= depth * horizontal + 1e-6 && Math.abs(y) <= depth * vertical + 1e-6
}
