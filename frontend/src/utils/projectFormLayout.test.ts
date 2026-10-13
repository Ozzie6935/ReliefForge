import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import {
  ARTWORK_GUIDANCE,
  DEFAULT_STOCK_THICKNESS_MM,
  DISPLAY_UNIT_NOTE,
  PROJECT_FORM_SECTIONS,
  STOCK_GUIDANCE,
  claimProjectSubmit,
  defaultStockThicknessField,
  geometryAfterUnitChange,
  projectFormColumns,
  projectGeometryRequest,
  releaseProjectSubmit,
  stockThicknessNote,
  type GeometryField,
  type GeometryFields,
} from "./projectFormLayout.ts"

const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")

function field(text: string, mm: number | null = null, edited = false): GeometryField {
  return { text, mm, edited }
}

function fields(overrides: Partial<GeometryFields> = {}): GeometryFields {
  return {
    stockWidth: field("12", 304.8),
    stockHeight: field("12", 304.8),
    stockThickness: defaultStockThicknessField("in", false),
    artworkWidth: field("11", 279.4),
    artworkHeight: field("11", 279.4),
    centerX: field("6", 152.4),
    centerY: field("6", 152.4),
    ...overrides,
  }
}

describe("new project form", () => {
  it("groups the existing fields and explains stock, artwork, and the front-left origin", () => {
    assert.deepEqual(
      PROJECT_FORM_SECTIONS.map((section) => section.title),
      ["Project", "Stock dimensions", "Artwork size and placement"],
    )
    assert.deepEqual(PROJECT_FORM_SECTIONS[0]?.fields, ["Project Name"])
    assert.deepEqual(PROJECT_FORM_SECTIONS[1]?.fields, ["Stock Width", "Stock Height", "Stock Thickness"])
    assert.deepEqual(PROJECT_FORM_SECTIONS[2]?.fields, [
      "Artwork Width",
      "Artwork Height",
      "Artwork Center X",
      "Artwork Center Y",
    ])
    assert.match(STOCK_GUIDANCE, /physical material/)
    assert.match(ARTWORK_GUIDANCE, /relief area/)
    assert.match(ARTWORK_GUIDANCE, /front-left stock origin/)
    assert.match(DISPLAY_UNIT_NOTE, /does not change the stored geometry/)
    assert.equal(projectFormColumns(1920), 2)
    assert.equal(projectFormColumns(800), 2)
    assert.equal(projectFormColumns(640), 1)
    assert.match(css, /@media \(max-width:\s*640px\)[\s\S]*\.form-section\s*\{[^}]*grid-template-columns:\s*1fr/)
  })

  it("submits unedited canonical millimeters and converts only an edited display value", () => {
    const stored = fields({
      stockWidth: field("3.937", 100.0001, false),
    })
    const unchanged = projectGeometryRequest("Bench", stored, "in")
    assert.equal(unchanged.ok, true)
    if (unchanged.ok) assert.equal(unchanged.request.stock.width_mm, 100.0001)

    const edited = projectGeometryRequest(
      "  Bench  ",
      fields({ stockWidth: field("14", null, true) }),
      "in",
    )
    assert.equal(edited.ok, true)
    if (edited.ok) {
      assert.equal(edited.request.name, "Bench")
      assert.equal(edited.request.stock.width_mm, 355.6)
      assert.equal(edited.request.stock.thickness_mm, DEFAULT_STOCK_THICKNESS_MM)
    }

    const switched = geometryAfterUnitChange(stored, "in", "mm")
    assert.equal(switched.stockWidth.mm, 100.0001)
    assert.equal(switched.stockWidth.edited, false)
    assert.equal(switched.stockWidth.text, "100.0001")
  })

  it("rejects a blank name, a non-positive size, and keeps a zero center", () => {
    assert.equal(projectGeometryRequest("   ", fields(), "in").ok, false)
    const zeroStock = projectGeometryRequest("Bench", fields({ stockHeight: field("0", null, true) }), "mm")
    assert.equal(zeroStock.ok, false)
    if (!zeroStock.ok) assert.match(zeroStock.error, /positive mm dimensions/)
    const center = projectGeometryRequest("Bench", fields({ centerX: field("0", null, true) }), "mm")
    assert.equal(center.ok, true)
    if (center.ok) assert.equal(center.request.artwork.center_x_mm, 0)
  })

  it("allows one submission to start and blocks the next until it is released", () => {
    const lock = { held: false }
    assert.equal(claimProjectSubmit(lock), true)
    assert.equal(claimProjectSubmit(lock), false)
    releaseProjectSubmit(lock)
    assert.equal(claimProjectSubmit(lock), true)
  })

  it("prefills 1.50 in and 38.10 mm and keeps that thickness across unit changes", () => {
    const inches = defaultStockThicknessField("in", false)
    const millimeters = defaultStockThicknessField("mm", false)
    assert.equal(inches.text, "1.50")
    assert.equal(inches.mm, 38.1)
    assert.equal(millimeters.text, "38.10")
    assert.equal(millimeters.mm, 38.1)

    let current = fields({ stockThickness: inches })
    let unit: "in" | "mm" = "in"
    for (let step = 0; step < 8; step += 1) {
      const next = unit === "in" ? "mm" : "in"
      current = geometryAfterUnitChange(current, unit, next)
      unit = next
      assert.equal(current.stockThickness.mm, 38.1)
      assert.equal(current.stockThickness.text, next === "mm" ? "38.10" : "1.50")
    }
    const submitted = projectGeometryRequest("Bench", current, unit)
    assert.equal(submitted.ok, true)
    if (submitted.ok) assert.equal(submitted.request.stock.thickness_mm, 38.1)
  })

  it("stores a custom thickness and leaves an unsaved legacy project unset", () => {
    const custom = projectGeometryRequest(
      "Bench",
      fields({ stockThickness: field("0.75", null, true) }),
      "in",
    )
    assert.equal(custom.ok, true)
    if (custom.ok) assert.equal(custom.request.stock.thickness_mm, 19.05)

    const twoInches = projectGeometryRequest(
      "Bench",
      fields({ stockThickness: field("2.00", null, true) }),
      "in",
    )
    assert.equal(twoInches.ok, true)
    if (twoInches.ok) assert.equal(twoInches.request.stock.thickness_mm, 50.8)

    const legacy = projectGeometryRequest(
      "Bench",
      fields({ stockThickness: defaultStockThicknessField("in", true) }),
      "in",
    )
    assert.equal(legacy.ok, true)
    if (legacy.ok) assert.equal(legacy.request.stock.thickness_mm, null)

    assert.match(stockThicknessNote({ creating: true, savedThicknessMm: null, edited: false }), /Application default/)
    assert.match(
      stockThicknessNote({ creating: false, savedThicknessMm: 50.8, edited: false }),
      /Project stock thickness/,
    )
    assert.match(
      stockThicknessNote({ creating: false, savedThicknessMm: null, edited: false }),
      /no saved stock thickness/,
    )
  })
})
