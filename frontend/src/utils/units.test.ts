import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { formatThicknessInput, inchesToMillimeters, toMillimeters } from "./units.ts"

describe("stock thickness units", () => {
  it("converts the named inch thicknesses to exact millimeters", () => {
    assert.equal(inchesToMillimeters(1.5), 38.1)
    assert.equal(inchesToMillimeters(0.75), 19.05)
    assert.equal(inchesToMillimeters(2), 50.8)
    assert.equal(toMillimeters(38.1, "mm"), 38.1)
    assert.equal(toMillimeters(19.05, "mm"), 19.05)
    assert.equal(toMillimeters(50.8, "mm"), 50.8)
  })

  it("shows 1.50 in and 38.10 mm and does not drift across repeated toggles", () => {
    let millimeters = 38.1
    let unit: "in" | "mm" = "in"
    let text = formatThicknessInput(millimeters, unit)
    assert.equal(text, "1.50")
    for (let step = 0; step < 6; step += 1) {
      const parsed = unit === "in" ? inchesToMillimeters(Number(text)) : toMillimeters(Number(text), "mm")
      millimeters = text === formatThicknessInput(millimeters, unit) ? millimeters : parsed
      unit = unit === "in" ? "mm" : "in"
      text = formatThicknessInput(millimeters, unit)
    }
    assert.equal(millimeters, 38.1)
    assert.equal(formatThicknessInput(19.05, "in"), "0.75")
    assert.equal(formatThicknessInput(19.05, "mm"), "19.05")
    assert.equal(formatThicknessInput(50.8, "in"), "2.00")
    assert.equal(formatThicknessInput(50.8, "mm"), "50.80")
    assert.equal(formatThicknessInput(38.1, "mm"), "38.10")
  })
})
