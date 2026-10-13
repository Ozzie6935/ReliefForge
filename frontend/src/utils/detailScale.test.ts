import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { formatLengthInput } from "./units.ts"
import {
  DETAIL_SAMPLING_WARNING,
  detailSamplingGuidance,
  parseDetailScaleMm,
  type DetailSamplingInput,
} from "./detailScale.ts"

const WEDDING: DetailSamplingInput = {
  detailScaleMm: null,
  artworkWidthMm: 279.4,
  artworkHeightMm: 279.4,
  widthPx: 1013,
  heightPx: 1023,
}

const ANGIE: DetailSamplingInput = {
  detailScaleMm: null,
  artworkWidthMm: 355.6,
  artworkHeightMm: 258.839,
  widthPx: 2940,
  heightPx: 2140,
}

describe("physical detail scale input", () => {
  it("stores millimeters and accepts the inch display of the endpoints", () => {
    assert.equal(parseDetailScaleMm("0.50", "mm"), 0.5)
    assert.equal(parseDetailScaleMm("0.25", "mm"), 0.25)
    assert.equal(parseDetailScaleMm("1", "mm"), 1)
    assert.equal(parseDetailScaleMm(formatLengthInput(0.5, "in"), "in"), 0.5)
    assert.equal(parseDetailScaleMm(formatLengthInput(0.25, "in"), "in"), 0.25)
    assert.equal(parseDetailScaleMm(formatLengthInput(1, "in"), "in"), 1)
    assert.equal(parseDetailScaleMm("0.02", "in"), 0.508)
  })

  it("rejects empty, non-finite, and out-of-range values without clamping", () => {
    assert.equal(parseDetailScaleMm("", "mm"), null)
    assert.equal(parseDetailScaleMm("abc", "mm"), null)
    assert.equal(parseDetailScaleMm("0", "mm"), null)
    assert.equal(parseDetailScaleMm("-1", "mm"), null)
    assert.equal(parseDetailScaleMm("0.24", "mm"), null)
    assert.equal(parseDetailScaleMm("1.01", "mm"), null)
    assert.equal(parseDetailScaleMm("1", "in"), null)
    assert.equal(parseDetailScaleMm("NaN", "mm"), null)
    assert.equal(parseDetailScaleMm("Infinity", "mm"), null)
  })
})

describe("physical detail sampling guidance", () => {
  it("warns for Wedding dimensions at 0.25 mm", () => {
    const guidance = detailSamplingGuidance({ ...WEDDING, detailScaleMm: 0.25 })
    assert.equal(guidance.warning, DETAIL_SAMPLING_WARNING)
    assert.ok(guidance.sigmaXPx != null && guidance.sigmaXPx < 1)
    assert.ok(guidance.sigmaYPx != null && guidance.sigmaYPx < 1)
    assert.ok(Math.abs(guidance.sigmaXPx - (0.25 * 1012) / 279.4) < 1e-12)
    assert.ok(Math.abs(guidance.sigmaYPx - (0.25 * 1022) / 279.4) < 1e-12)
  })

  it("does not warn for Wedding dimensions at 0.50 mm", () => {
    const guidance = detailSamplingGuidance({ ...WEDDING, detailScaleMm: 0.5 })
    assert.equal(guidance.warning, null)
    assert.ok(guidance.sigmaXPx != null && guidance.sigmaXPx >= 1)
    assert.ok(guidance.sigmaYPx != null && guidance.sigmaYPx >= 1)
  })

  it("does not warn for Angie dimensions at 0.25 mm", () => {
    const guidance = detailSamplingGuidance({ ...ANGIE, detailScaleMm: 0.25 })
    assert.equal(guidance.warning, null)
    assert.ok(guidance.sigmaXPx != null && guidance.sigmaXPx >= 1)
    assert.ok(guidance.sigmaYPx != null && guidance.sigmaYPx >= 1)
    assert.ok(Math.abs(guidance.sigmaXPx - (0.25 * 2939) / 355.6) < 1e-12)
    assert.ok(Math.abs(guidance.sigmaYPx - (0.25 * 2139) / 258.839) < 1e-12)
  })

  it("omits a result when dimensions are missing", () => {
    for (const input of [
      { ...WEDDING, detailScaleMm: 0.25, artworkWidthMm: null },
      { ...WEDDING, detailScaleMm: 0.25, artworkHeightMm: undefined },
      { ...WEDDING, detailScaleMm: 0.25, widthPx: null },
      { ...WEDDING, detailScaleMm: 0.25, heightPx: undefined },
      { detailScaleMm: 0.25 },
    ]) {
      assert.deepEqual(detailSamplingGuidance(input), {
        warning: null,
        sigmaXPx: null,
        sigmaYPx: null,
      })
    }
  })

  it("handles zero, negative, non-finite, and one-pixel dimensions without a fabricated pitch", () => {
    const unsafe = [
      { ...WEDDING, detailScaleMm: 0.25, artworkWidthMm: 0 },
      { ...WEDDING, detailScaleMm: 0.25, artworkHeightMm: -279.4 },
      { ...WEDDING, detailScaleMm: 0.25, artworkWidthMm: Number.NaN },
      { ...WEDDING, detailScaleMm: 0.25, artworkHeightMm: Number.POSITIVE_INFINITY },
      { ...WEDDING, detailScaleMm: 0.25, widthPx: 1, heightPx: 1023 },
      { ...WEDDING, detailScaleMm: 0.25, widthPx: 1013, heightPx: 1 },
      { ...WEDDING, detailScaleMm: 0.25, widthPx: 0, heightPx: 0 },
      { ...WEDDING, detailScaleMm: 0.25, widthPx: 1.5, heightPx: 1023 },
      { ...WEDDING, detailScaleMm: 0 },
      { ...WEDDING, detailScaleMm: -0.25 },
      { ...WEDDING, detailScaleMm: Number.NaN },
      { ...WEDDING, detailScaleMm: Number.POSITIVE_INFINITY },
    ]
    for (const input of unsafe) {
      assert.deepEqual(detailSamplingGuidance(input), {
        warning: null,
        sigmaXPx: null,
        sigmaYPx: null,
      })
    }
  })

  it("evaluates both axes of a non-square pitch", () => {
    const wide = detailSamplingGuidance({
      detailScaleMm: 0.75,
      artworkWidthMm: 100,
      artworkHeightMm: 100,
      widthPx: 101,
      heightPx: 201,
    })
    assert.equal(wide.sigmaXPx, 0.75)
    assert.equal(wide.sigmaYPx, 1.5)
    assert.equal(wide.warning, DETAIL_SAMPLING_WARNING)

    const tall = detailSamplingGuidance({
      detailScaleMm: 0.75,
      artworkWidthMm: 100,
      artworkHeightMm: 100,
      widthPx: 201,
      heightPx: 101,
    })
    assert.equal(tall.sigmaXPx, 1.5)
    assert.equal(tall.sigmaYPx, 0.75)
    assert.equal(tall.warning, DETAIL_SAMPLING_WARNING)

    const resolved = detailSamplingGuidance({
      detailScaleMm: 1,
      artworkWidthMm: 100,
      artworkHeightMm: 100,
      widthPx: 101,
      heightPx: 201,
    })
    assert.equal(resolved.sigmaXPx, 1)
    assert.equal(resolved.sigmaYPx, 2)
    assert.equal(resolved.warning, null)
  })

  it("never warns in legacy 2-pixel mode", () => {
    const guidance = detailSamplingGuidance({ ...WEDDING, detailScaleMm: null })
    assert.deepEqual(guidance, { warning: null, sigmaXPx: null, sigmaYPx: null })
  })

  it("uses millimeters after either display unit is parsed", () => {
    const fromMillimeters = parseDetailScaleMm("0.25", "mm")
    const fromInches = parseDetailScaleMm(formatLengthInput(0.25, "in"), "in")
    assert.equal(fromMillimeters, 0.25)
    assert.equal(fromInches, 0.25)
    assert.deepEqual(
      detailSamplingGuidance({ ...WEDDING, detailScaleMm: fromMillimeters }),
      detailSamplingGuidance({ ...WEDDING, detailScaleMm: fromInches }),
    )
    assert.notEqual(
      detailSamplingGuidance({ ...ANGIE, detailScaleMm: fromMillimeters }).warning,
      detailSamplingGuidance({ ...WEDDING, detailScaleMm: fromInches }).warning,
    )
  })

  it("does not issue a processing request when the draft scale changes", () => {
    const original = globalThis.fetch
    let calls = 0
    globalThis.fetch = (() => {
      calls += 1
      return Promise.reject(new Error("processing request"))
    }) as typeof fetch
    try {
      for (const detailScaleMm of [null, 0.25, 0.5, 1]) {
        detailSamplingGuidance({ ...WEDDING, detailScaleMm })
        detailSamplingGuidance({ ...ANGIE, detailScaleMm })
      }
      assert.equal(calls, 0)
    } finally {
      globalThis.fetch = original
    }
  })
})
