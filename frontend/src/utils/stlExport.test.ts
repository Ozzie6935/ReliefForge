import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import { ApiError } from "../api/client.ts"
import {
  afterExportFailure,
  decideExportClick,
  downloadAllowed,
  exportDownloadNote,
  exportLayout,
  exportReadinessPath,
  filenameFromContentDisposition,
  formatByteSize,
  geometryFacts,
  interpretStlResponse,
  stlLengthError,
  issueMessages,
  parseExportReadiness,
  parseGridInput,
  planValidationNotes,
  readinessMatches,
  reducedGridError,
  resolutionDraftDirty,
  saveBlobDownload,
  shouldApplyReadiness,
  showsMissingPhysical,
  stlExportBody,
  stlExportPath,
  suggestReducedGrid,
  type DownloadHost,
  type ExportReadiness,
} from "./stlExport.ts"

const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")
const workspace = readFileSync(new URL("../pages/WorkspacePage.tsx", import.meta.url), "utf8")
const section = readFileSync(new URL("../components/ExportSection.tsx", import.meta.url), "utf8")
const api = readFileSync(new URL("../api/projects.ts", import.meta.url), "utf8")

const PROJECT_ID = "d276243e-0258-4dc0-86d4-0a416c1ef1bf"

function report(overrides: Partial<ExportReadiness> = {}): ExportReadiness {
  return {
    project_id: PROJECT_ID,
    provenance_status: "current",
    export_target: "artwork-local",
    export_certificate: "none",
    coordinate_convention: "Artwork-local millimeters.",
    resolution_policy: "Native export keeps every saved physical-height sample.",
    topology_note: "Topology was not measured for this plan.",
    mode: "native",
    source_grid: { height_px: 8, width_px: 10 },
    planned_grid: { height_px: 8, width_px: 10 },
    spacing_mm: { x_mm: 1, y_mm: 1, uniform: true },
    artwork_width_mm: 90,
    artwork_height_mm: 70.5,
    stock_width_mm: 100,
    stock_height_mm: 80,
    stock_thickness_mm: 19,
    base_thickness_mm: 2,
    maximum_surface_height_mm: 8,
    placement: {
      recorded: true,
      applied: false,
      center_x_mm: 50,
      center_y_mm: 40,
      origin: "front-left-stock",
    },
    z_minimum_mm: 2,
    z_maximum_mm: 8,
    estimated_vertices: 160,
    estimated_triangles: 300,
    estimated_stl_bytes: 15084,
    estimated_peak_memory_bytes: 40000,
    topology_expected: "closed-solid",
    topology_validated: false,
    self_intersection_checked: false,
    zero_area_measured: false,
    blocking_errors: [],
    warnings: [],
    corrective_actions: [],
    ready_for_export: true,
    limits: { max_triangles: 2_000_000, max_stl_bytes: 104857600, max_peak_memory_bytes: 536870912 },
    ...overrides,
  }
}

test("readiness and STL requests use the export contract", () => {
  const nativePath = exportReadinessPath(PROJECT_ID, { mode: "native" })
  assert.equal(
    nativePath,
    `/api/projects/${PROJECT_ID}/relief/export/readiness?mode=native`,
  )
  assert.equal(nativePath.includes("height_px"), false)
  assert.equal(nativePath.includes("width_px"), false)
  const reducedPath = exportReadinessPath(PROJECT_ID, { mode: "reduced", heightPx: 4, widthPx: 6 })
  assert.equal(
    reducedPath,
    `/api/projects/${PROJECT_ID}/relief/export/readiness?mode=reduced&height_px=4&width_px=6`,
  )
  assert.equal(stlExportPath(PROJECT_ID), `/api/projects/${PROJECT_ID}/relief/export/stl`)
  assert.deepEqual(stlExportBody({ mode: "native" }), { mode: "native" })
  assert.deepEqual(Object.keys(stlExportBody({ mode: "native" })), ["mode"])
  assert.deepEqual(stlExportBody({ mode: "reduced", heightPx: 4, widthPx: 6 }), {
    mode: "reduced",
    height_px: 4,
    width_px: 6,
  })
  const readinessFn = api.slice(api.indexOf("export function getExportReadiness"))
  assert.match(readinessFn, /method: "GET"/)
  const postFn = api.slice(api.indexOf("export async function postStlExport"))
  assert.match(postFn, /method: "POST"/)
  assert.match(postFn, /JSON\.stringify\(stlExportBody\(selection\)\)/)
  assert.match(postFn, /interpretStlResponse/)
})

test("reduced grids stay explicit and native is not rewritten", () => {
  const source = { heightPx: 8, widthPx: 10 }
  assert.equal(parseGridInput("4"), 4)
  assert.equal(parseGridInput("2.5"), null)
  assert.equal(parseGridInput(""), null)
  assert.match(reducedGridError(source, 1, 4) ?? "", /at least 2/)
  assert.match(reducedGridError(source, 9, 4) ?? "", /cannot exceed/)
  assert.match(reducedGridError(source, 8, 10) ?? "", /smaller/)
  assert.match(reducedGridError(source, 12, 2) ?? "", /cannot exceed/)
  assert.equal(reducedGridError(source, 4, 6), null)
  assert.deepEqual(suggestReducedGrid(source), { heightPx: 4, widthPx: 5 })
  assert.equal(suggestReducedGrid({ heightPx: 2, widthPx: 2 }), null)
  assert.equal(stlExportBody({ mode: "native" }).mode, "native")
  assert.equal("height_px" in stlExportBody({ mode: "native" }), false)
})

test("readiness states keep backend messages and gate the download", () => {
  const missing = report({
    provenance_status: "missing",
    source_grid: null,
    planned_grid: null,
    stock_thickness_mm: null,
    base_thickness_mm: null,
    maximum_surface_height_mm: null,
    z_minimum_mm: null,
    z_maximum_mm: null,
    estimated_triangles: null,
    estimated_stl_bytes: null,
    estimated_peak_memory_bytes: null,
    ready_for_export: false,
    blocking_errors: [{ code: "physical_missing", message: "Physical height has not been generated." }],
    corrective_actions: ["Generate physical height from the current composed relief before planning an STL."],
  })
  assert.equal(parseExportReadiness(missing)?.provenance_status, "missing")
  assert.equal(showsMissingPhysical(missing), true)
  assert.equal(issueMessages(missing.blocking_errors)[0], "Physical height has not been generated.")

  const stale = report({
    provenance_status: "stale",
    ready_for_export: false,
    blocking_errors: [{ code: "provenance_stale", message: "Physical height is stale. Stale geometry is not export-ready." }],
  })
  const unknown = report({
    provenance_status: "unknown",
    ready_for_export: false,
    blocking_errors: [{ code: "provenance_unknown", message: "Physical-height provenance is unknown. Unknown geometry is not export-ready." }],
  })
  assert.equal(parseExportReadiness(stale)?.provenance_status, "stale")
  assert.equal(parseExportReadiness(unknown)?.provenance_status, "unknown")
  assert.equal(showsMissingPhysical(stale), false)

  const blocked = report({
    ready_for_export: false,
    blocking_errors: [
      { code: "triangle_limit", message: "Estimated triangle count 3000000 exceeds the export planning limit 2000000. Native resolution is not reduced automatically." },
      { code: "stl_byte_limit", message: "Estimated binary STL size 150000000 exceeds the export planning limit 104857600. Native resolution is not reduced automatically." },
      { code: "memory_limit", message: "Estimated peak memory 600000000 exceeds the export planning limit 536870912. Native resolution is not reduced automatically." },
    ],
  })
  assert.deepEqual(issueMessages(blocked.blocking_errors), blocked.blocking_errors.map((issue) => issue.message))
  assert.match(planValidationNotes(blocked)[0], /did not pass/)
  assert.match(planValidationNotes(report())[0], /passed/)
  assert.match(planValidationNotes(report()).join(" "), /not run for this plan/)
  assert.match(planValidationNotes(report()).join(" "), /not a machining certificate/)

  const ready = report()
  assert.equal(
    downloadAllowed({
      report: ready,
      projectId: PROJECT_ID,
      selection: { mode: "native" },
      draftDirty: false,
      exporting: false,
      phase: "ready",
    }),
    true,
  )
  assert.equal(
    downloadAllowed({
      report: blocked,
      projectId: PROJECT_ID,
      selection: { mode: "native" },
      draftDirty: false,
      exporting: false,
      phase: "ready",
    }),
    false,
  )
  assert.equal(
    downloadAllowed({
      report: ready,
      projectId: "other",
      selection: { mode: "native" },
      draftDirty: false,
      exporting: false,
      phase: "ready",
    }),
    false,
  )
  const reduced = report({ mode: "reduced", planned_grid: { height_px: 4, width_px: 6 } })
  assert.equal(readinessMatches(reduced, PROJECT_ID, { mode: "reduced", heightPx: 4, widthPx: 6 }), true)
  assert.equal(readinessMatches(reduced, PROJECT_ID, { mode: "reduced", heightPx: 3, widthPx: 6 }), false)
  assert.equal(readinessMatches(ready, PROJECT_ID, { mode: "reduced", heightPx: 4, widthPx: 6 }), false)
  assert.equal(
    downloadAllowed({
      report: ready,
      projectId: PROJECT_ID,
      selection: { mode: "native" },
      draftDirty: true,
      exporting: false,
      phase: "ready",
    }),
    false,
  )
  assert.equal(resolutionDraftDirty("reduced", { mode: "native" }, "4", "6"), true)
  assert.equal(resolutionDraftDirty("native", { mode: "native" }, "4", "6"), false)
  const facts = geometryFacts(ready, "mm")
  assert.equal(facts.find((fact) => fact.term === "Artwork width")?.detail, "90 mm")
  assert.equal(facts.find((fact) => fact.term === "Stock width")?.detail, "100 mm")
  assert.match(formatByteSize(15084), /15,084 bytes/)
  assert.equal(parseExportReadiness({ ready_for_export: true }), null)
})

test("stale readiness responses and duplicate downloads are ignored", () => {
  assert.equal(shouldApplyReadiness(1, 2), false)
  assert.equal(shouldApplyReadiness(2, 2), true)
  assert.equal(decideExportClick(true, true), "ignored")
  assert.equal(decideExportClick(false, false), "ignored")
  assert.equal(decideExportClick(false, true), "started")
  assert.deepEqual(afterExportFailure(4), { refresh: 5, downloadEnabled: false })
})

test("STL filenames are sanitized and the object URL is revoked", async () => {
  assert.equal(
    filenameFromContentDisposition('attachment; filename="wedding-relief-benchmark-relief.stl"'),
    "wedding-relief-benchmark-relief.stl",
  )
  assert.equal(filenameFromContentDisposition('attachment; filename="../../secret.stl"'), "relief-export.stl")
  assert.equal(filenameFromContentDisposition('attachment; filename="bad name.stl"'), "relief-export.stl")
  assert.equal(filenameFromContentDisposition(null), "relief-export.stl")

  const bytes = new Uint8Array(84)
  bytes[0] = 82
  const response = new Response(bytes, {
    status: 200,
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": 'attachment; filename="sample-relief.stl"',
      "X-Relief-Stl-Units": "mm",
      "X-Relief-Export-Certificate": "none",
      "X-Relief-Self-Intersection-Checked": "false",
      "X-Relief-Topology-Limitation": "Self-intersection was not fully checked.",
    },
  })
  const parsed = await interpretStlResponse(response)
  assert.equal(parsed.body.byteLength, 84)
  assert.equal(parsed.filename, "sample-relief.stl")
  assert.match(exportDownloadNote(parsed), /millimeters/)
  assert.match(exportDownloadNote(parsed), /not a machining certificate/)
  assert.match(exportDownloadNote(parsed), /not completed/)

  const rejected = new Response(
    JSON.stringify({ detail: "Physical height is stale. Stale geometry is not export-ready." }),
    { status: 409, headers: { "Content-Type": "application/json" } },
  )
  await assert.rejects(interpretStlResponse(rejected), (error: unknown) => {
    assert.ok(error instanceof ApiError)
    assert.equal(error.status, 409)
    assert.match(error.message, /stale/)
    return true
  })

  const incomplete = new Response(new Uint8Array(8), {
    status: 200,
    headers: { "Content-Type": "application/octet-stream" },
  })
  await assert.rejects(interpretStlResponse(incomplete), /incomplete STL/)

  const truncated = new Uint8Array(84)
  new DataView(truncated.buffer).setUint32(80, 1, true)
  assert.match(stlLengthError(truncated.buffer) ?? "", /does not match its triangle count/)
  const trailing = new Uint8Array(84 + 50 + 1)
  new DataView(trailing.buffer).setUint32(80, 1, true)
  assert.match(stlLengthError(trailing.buffer) ?? "", /does not match its triangle count/)
  const exact = new Uint8Array(84 + 50)
  new DataView(exact.buffer).setUint32(80, 1, true)
  assert.equal(stlLengthError(exact.buffer), null)
  await assert.rejects(
    interpretStlResponse(
      new Response(truncated, { status: 200, headers: { "Content-Type": "application/octet-stream" } }),
    ),
    /does not match its triangle count/,
  )

  const revoked: string[] = []
  const downloads: string[] = []
  let removed = false
  const host: DownloadHost = {
    createObjectURL: () => "blob:stl",
    revokeObjectURL: (url) => revoked.push(url),
    createElement: () => ({
      href: "",
      download: "",
      click() {
        downloads.push(this.download)
      },
      remove() {
        removed = true
      },
    }),
    appendChild: () => undefined,
  }
  const url = saveBlobDownload(bytes.buffer, "../../secret.stl", host)
  assert.equal(url, "blob:stl")
  assert.deepEqual(revoked, ["blob:stl"])
  assert.deepEqual(downloads, ["relief-export.stl"])
  assert.equal(removed, true)
  saveBlobDownload(bytes.buffer, "sample-relief.stl", host)
  assert.equal(downloads[1], "sample-relief.stl")
})

test("export workspace keeps the raised tab, split layout, and lazy 3D", () => {
  assert.match(workspace, /aria-current=\{section === "export" \? "page" : undefined\}/)
  assert.match(workspace, /<ExportSection/)
  assert.match(workspace, /lazy\(\(\) => import\("\.\.\/components\/MeshPreviewSection\.tsx"\)\)/)
  assert.doesNotMatch(section, /generatePhysicalHeight/)
  assert.match(section, /getExportReadiness/)
  assert.match(section, /postStlExport/)
  assert.match(section, /decideExportClick/)
  assert.match(section, /saveBlobDownload/)
  assert.match(section, /afterExportFailure/)
  assert.match(section, /shouldApplyReadiness/)
  assert.match(section, /Open Relief/)
  assert.match(section, /Download STL/)
  assert.match(section, /Check readiness/)
  assert.match(section, /suggestReducedGrid/)
  const body = section.slice(section.indexOf("export function ExportSection"))
  const download = body.slice(body.indexOf("async function onDownload"))
  assert.match(download, /postStlExport\(/)
  assert.doesNotMatch(body.slice(0, body.indexOf("async function onDownload")), /postStlExport\(/)
  assert.match(css, /\.export-workspace\.is-split/)
  assert.match(css, /\.export-panel/)
  assert.match(css, /minmax\(340px, 380px\)/)
  assert.match(css, /max-width: 960px/)
  assert.match(css, /\.export-main input:focus-visible/)
  for (const width of [1920, 1366, 1100, 961]) {
    const layout = exportLayout(width)
    assert.equal(layout.stacked, false)
    assert.equal(layout.inspectorMinPx, 340)
    assert.equal(layout.inspectorMaxPx, 380)
  }
  for (const width of [960, 800]) {
    assert.equal(exportLayout(width).stacked, true)
  }
})
