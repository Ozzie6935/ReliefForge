import assert from "node:assert/strict"
import { readdirSync, readFileSync, statSync } from "node:fs"
import { describe, it } from "node:test"

import {
  decodeRfmp,
  displayVertexNormals,
  frontFaceToward,
  toDisplayPoint,
  type DecodedMeshPreview,
  type Vec3,
} from "./meshPreview.ts"

const fixtureDir = new URL("../fixtures/synthetic-relief/", import.meta.url)
const workspace = readFileSync(new URL("../pages/WorkspacePage.tsx", import.meta.url), "utf8")
const section = readFileSync(new URL("../components/MeshPreviewSection.tsx", import.meta.url), "utf8")

function loadFixture(name: string): DecodedMeshPreview {
  const bytes = readFileSync(new URL(name, fixtureDir))
  return decodeRfmp(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
}

function regionPeak(mesh: DecodedMeshPreview, x0: number, x1: number, y0: number, y1: number): number {
  let peak = -Infinity
  const positions = mesh.positions
  for (let index = 0; index < positions.length; index += 3) {
    const x = positions[index]
    const y = positions[index + 1]
    const z = positions[index + 2]
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) peak = Math.max(peak, z)
  }
  assert.ok(Number.isFinite(peak), `No samples in ${x0},${y0} to ${x1},${y1}`)
  return peak
}

function triangleNormal(mesh: DecodedMeshPreview, triangle: number): Vec3 {
  const display = new Float32Array(9)
  for (let corner = 0; corner < 3; corner += 1) {
    const vertex = mesh.indices[triangle * 3 + corner] * 3
    const point = toDisplayPoint(
      mesh.positions[vertex],
      mesh.positions[vertex + 1],
      mesh.positions[vertex + 2],
    )
    display[corner * 3] = point[0]
    display[corner * 3 + 1] = point[1]
    display[corner * 3 + 2] = point[2]
  }
  const e1 = [display[3] - display[0], display[4] - display[1], display[5] - display[2]]
  const e2 = [display[6] - display[0], display[7] - display[1], display[8] - display[2]]
  return [
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0],
  ]
}

describe("synthetic RFMP fixture", () => {
  it("keeps the asymmetric relief upright in backend and display space", () => {
    const mesh = loadFixture("50000.rfmp")
    assert.equal(mesh.artworkWidthMm, 100)
    assert.equal(mesh.artworkHeightMm, 70)
    assert.equal(mesh.exportReady, false)
    assert.equal(mesh.status, "current")
    assert.equal(mesh.downsamplingRequired, false)
    assert.equal(mesh.maxVertices, 50_000)
    const upperLeft = regionPeak(mesh, 4, 36, 3, 20)
    const lowerRight = regionPeak(mesh, 74, 90, 50, 64)
    const middle = regionPeak(mesh, 45, 60, 30, 40)
    assert.ok(upperLeft > 17)
    assert.ok(lowerRight > 8 && lowerRight < 10)
    assert.ok(middle > 4 && middle < 6)
    assert.ok(upperLeft > lowerRight)
    let peak = 0
    for (let index = 0; index < mesh.positions.length; index += 3) {
      if (mesh.positions[index + 2] > mesh.positions[peak + 2]) peak = index
    }
    const displayed = toDisplayPoint(
      mesh.positions[peak],
      mesh.positions[peak + 1],
      mesh.positions[peak + 2],
    )
    assert.ok(mesh.positions[peak] < 40)
    assert.ok(mesh.positions[peak + 1] < 25)
    assert.ok(displayed[1] > 17)
    assert.ok(displayed[2] > -25)
  })

  it("gives top, bottom, and side faces outward display normals", () => {
    const mesh = loadFixture("50000.rfmp")
    let top = -1
    let bottom = -1
    let side = -1
    for (let triangle = 0; triangle < mesh.triangleCount; triangle += 1) {
      const zs = [0, 1, 2].map((corner) => mesh.positions[mesh.indices[triangle * 3 + corner] * 3 + 2])
      const xs = [0, 1, 2].map((corner) => mesh.positions[mesh.indices[triangle * 3 + corner] * 3])
      if (top < 0 && zs.every((value) => value > 17)) top = triangle
      if (bottom < 0 && zs.every((value) => value < 0.01)) bottom = triangle
      if (side < 0 && xs.every((value) => value < 0.01) && zs.some((value) => value < 0.01) && zs.some((value) => value > 1)) {
        side = triangle
      }
    }
    assert.ok(top >= 0 && bottom >= 0 && side >= 0)
    const topNormal = triangleNormal(mesh, top)
    const bottomNormal = triangleNormal(mesh, bottom)
    const sideNormal = triangleNormal(mesh, side)
    assert.ok(topNormal[1] > 0)
    assert.equal(frontFaceToward(topNormal, [10, 18, -8], [10, 40, -8]), true)
    assert.ok(bottomNormal[1] < 0)
    assert.ok(sideNormal[0] < 0)
    const vertexNormals = displayVertexNormals(mesh.positions, mesh.indices)
    assert.notEqual(vertexNormals.length, mesh.faceNormals.length)
  })

  it("returns a distinct resolution and provenance for each quality budget", () => {
    const fast = loadFixture("20000.rfmp")
    const standard = loadFixture("50000.rfmp")
    const detailed = loadFixture("150000.rfmp")
    const maximum = loadFixture("250000.rfmp")
    assert.deepEqual(
      [fast, standard, detailed, maximum].map((mesh) => mesh.maxVertices),
      [20_000, 50_000, 150_000, 250_000],
    )
    const grids = [fast, standard, detailed, maximum].map((mesh) => `${mesh.meshWidthPx}x${mesh.meshHeightPx}`)
    assert.equal(new Set(grids).size, 4)
    assert.equal(fast.downsamplingRequired, true)
    assert.equal(standard.downsamplingRequired, false)
    assert.equal(detailed.status, "unknown")
    assert.equal(maximum.status, "stale")
    assert.ok(regionPeak(fast, 4, 36, 3, 20) > regionPeak(fast, 74, 90, 50, 64))
    const thin = loadFixture("thin.rfmp")
    const tall = loadFixture("tall.rfmp")
    assert.ok(thin.boundingBox.max[2] < 2)
    assert.ok(tall.boundingBox.max[2] > 60)
    assert.equal(thin.artworkWidthMm, 100)
    assert.equal(tall.artworkHeightMm, 70)
  })
})

describe("viewer loading and replacement", () => {
  it("lazy-loads the Three.js section only from the 3D tab", () => {
    assert.match(workspace, /lazy\(\(\) => import\("\.\.\/components\/MeshPreviewSection\.tsx"\)\)/)
    assert.match(workspace, /<Suspense fallback=\{<p className="field-note">Loading 3D preview\.<\/p>\}>/)
    assert.doesNotMatch(workspace, /import \{ MeshPreviewSection \}/)
    assert.doesNotMatch(workspace, /from "three"|@react-three/)
    assert.match(section, /from "three"/)
    assert.match(section, /geometry\.dispose\(\)/)
    assert.match(section, /material\.dispose\(\)/)
    assert.match(section, /closed\.current = true/)
    assert.match(section, /setDecoded\(null\)/)
    const effect = section.slice(section.indexOf("if (!meshPreviewShouldFetch"), section.indexOf("}, [fetchToken, webgl]"))
    assert.doesNotMatch(effect, /viewName|wireframe|flat/)
    assert.match(section, /Fit/)
    assert.match(section, /Reset/)
    assert.match(section, /Perspective/)
    assert.match(section, /Top/)
    assert.match(section, /Front/)
    assert.match(section, /Side/)
    assert.match(section, /Wireframe/)
    assert.match(section, /value="flat"/)
  })

  it("keeps Three.js out of the main production chunk", () => {
    const assets = new URL("../../dist/assets/", import.meta.url)
    const files = readdirSync(assets).filter((name) => name.endsWith(".js"))
    const mainName = files.find((name) => name.startsWith("index-"))
    assert.ok(mainName)
    const marker = "WebGLRenderer"
    const main = readFileSync(new URL(mainName, assets), "utf8")
    assert.equal(main.includes(marker), false)
    assert.ok(statSync(new URL(mainName, assets)).size < 800_000)
    const lazy = files.filter((name) => {
      if (name === mainName) return false
      return readFileSync(new URL(name, assets), "utf8").includes(marker)
    })
    assert.equal(lazy.length, 1)
  })
})
