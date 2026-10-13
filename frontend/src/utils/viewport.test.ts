import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { imagePixelFromClient } from "./imageCoordinates.ts"
import {
  clampViewScale,
  clampViewportPan,
  contentPlacement,
  MIN_VIEW_SCALE,
  panDelta,
  sourcePixelScale,
  viewAfterResize,
  zoomAreaView,
} from "./viewport.ts"

describe("zoom area", () => {
  it("fits and centers a source rectangle", () => {
    const view = zoomAreaView({
      selection: { x0: 100, y0: 100, x1: 300, y1: 200 },
      displayWidth: 80,
      displayHeight: 40,
      naturalWidth: 800,
      naturalHeight: 600,
      layoutWidth: 800,
      layoutHeight: 600,
      viewportWidth: 800,
      viewportHeight: 600,
    })
    assert.ok(view)
    assert.equal(view.percent, 400)
    assert.equal(view.scale, 4)
    assert.equal(view.panX, -400)
    assert.equal(view.panY, -300)
    const top = view.panY + 100 * view.scale
    const bottom = view.panY + 200 * view.scale
    assert.equal((top + bottom) / 2, 300)
  })

  it("clamps zoom to 800 percent", () => {
    const view = zoomAreaView({
      selection: { x0: 390, y0: 290, x1: 410, y1: 310 },
      displayWidth: 40,
      displayHeight: 40,
      naturalWidth: 800,
      naturalHeight: 600,
      layoutWidth: 800,
      layoutHeight: 600,
      viewportWidth: 800,
      viewportHeight: 600,
    })
    assert.ok(view)
    assert.equal(view.percent, 800)
    const centerX = view.panX + 400 * view.scale
    const centerY = view.panY + 300 * view.scale
    assert.equal(centerX, 400)
    assert.equal(centerY, 300)
  })

  it("does not zoom out smaller than the editor box", () => {
    const view = zoomAreaView({
      selection: { x0: 0, y0: 0, x1: 400, y1: 300 },
      displayWidth: 200,
      displayHeight: 150,
      naturalWidth: 800,
      naturalHeight: 600,
      layoutWidth: 800,
      layoutHeight: 600,
      viewportWidth: 200,
      viewportHeight: 150,
    })
    assert.ok(view)
    assert.equal(view.scale, 1)
    assert.equal(clampViewScale(0.4, 0.8), 1)
    assert.equal(clampViewScale(1.2, 0.8), 1.2)
  })

  it("ignores a drag smaller than 5 CSS pixels", () => {
    const view = zoomAreaView({
      selection: { x0: 0, y0: 0, x1: 40, y1: 40 },
      displayWidth: 4,
      displayHeight: 20,
      naturalWidth: 800,
      naturalHeight: 600,
      layoutWidth: 800,
      layoutHeight: 600,
      viewportWidth: 800,
      viewportHeight: 600,
    })
    assert.equal(view, null)
  })
})

describe("fit", () => {
  it("uses both viewport width and height", () => {
    const placed = contentPlacement(800, 300, 1600, 900)
    assert.ok(placed)
    assert.equal(placed.scale, 300 / 900)
    assert.ok(Math.abs(placed.offsetX - (800 - 1600 * placed.scale) / 2) < 1e-6)
    assert.equal(placed.offsetY, 0)
  })

  it("height-constrains a landscape image in a wide short viewport", () => {
    const placed = contentPlacement(2000, 400, 1600, 900)
    assert.ok(placed)
    assert.equal(placed.scale, 400 / 900)
    assert.ok(placed.scale < 2000 / 1600)
    assert.ok(placed.offsetX > 0)
    assert.equal(placed.offsetY, 0)
  })

  it("width-constrains a landscape image in a tall narrow viewport", () => {
    const placed = contentPlacement(500, 800, 1600, 900)
    assert.ok(placed)
    assert.equal(placed.scale, 500 / 1600)
    assert.ok(placed.scale < 800 / 900)
    assert.equal(placed.offsetX, 0)
    assert.ok(placed.offsetY > 0)
  })

  it("keeps 100 percent at one source pixel per CSS pixel", () => {
    const placed = contentPlacement(400, 600, 800, 600)
    assert.ok(placed)
    const fittedWidth = 800 * placed.scale
    assert.equal(fittedWidth, 400)
    assert.equal(sourcePixelScale(fittedWidth, 800), 2)
    assert.ok(sourcePixelScale(fittedWidth, 800) > MIN_VIEW_SCALE)
  })

  it("does not zoom out below Fit", () => {
    assert.equal(clampViewScale(0.25, 0.5), 1)
  })

  it("recomputes Fit when the viewport changes and clamps a zoomed pan", () => {
    const fitted = viewAfterResize({
      scale: 1,
      panX: 40,
      panY: 10,
      layoutWidth: 400,
      layoutHeight: 225,
      viewportWidth: 800,
      viewportHeight: 300,
    })
    assert.equal(fitted.scale, 1)
    assert.equal(fitted.panX, (800 - 400) / 2)
    assert.equal(fitted.panY, (300 - 225) / 2)
    const zoomed = viewAfterResize({
      scale: 2,
      panX: 5000,
      panY: -4000,
      layoutWidth: 400,
      layoutHeight: 225,
      viewportWidth: 500,
      viewportHeight: 280,
    })
    assert.equal(zoomed.scale, 2)
    assert.ok(zoomed.panX < 5000)
    assert.ok(zoomed.panY > -4000)
  })

  it("keeps the same source pixel after the fitted frame changes", () => {
    function frame(viewportWidth: number, viewportHeight: number) {
      const placed = contentPlacement(viewportWidth, viewportHeight, 800, 600)
      assert.ok(placed)
      return {
        left: 20 + placed.offsetX,
        top: 40 + placed.offsetY,
        width: 800 * placed.scale,
        height: 600 * placed.scale,
        naturalWidth: 800,
        naturalHeight: 600,
      }
    }
    const wide = frame(1000, 400)
    const tall = frame(500, 900)
    const widePixel = imagePixelFromClient(wide.left + wide.width / 2, wide.top + wide.height / 2, wide)
    const tallPixel = imagePixelFromClient(tall.left + tall.width / 2, tall.top + tall.height / 2, tall)
    assert.ok(widePixel && tallPixel)
    assert.equal(Math.round(widePixel.x), 400)
    assert.equal(Math.round(widePixel.y), 300)
    assert.equal(Math.round(tallPixel.x), 400)
    assert.equal(Math.round(tallPixel.y), 300)
  })
})

describe("pan", () => {
  it("moves the view toward the requested side", () => {
    assert.deepEqual(panDelta("up", 200, 100), { x: 0, y: 10 })
    assert.deepEqual(panDelta("down", 200, 100), { x: 0, y: -10 })
    assert.deepEqual(panDelta("left", 200, 100), { x: 20, y: 0 })
    assert.deepEqual(panDelta("right", 200, 100), { x: -20, y: 0 })
  })

  it("centers an image that is smaller than the viewport", () => {
    const pan = clampViewportPan({
      panX: 40,
      panY: -20,
      scale: 1,
      layoutWidth: 400,
      layoutHeight: 300,
      viewportWidth: 800,
      viewportHeight: 600,
    })
    assert.deepEqual(pan, { panX: 200, panY: 150 })
  })

  it("does not pan an image that already fills the viewport", () => {
    const step = panDelta("right", 800, 600)
    const pan = clampViewportPan({
      panX: step.x,
      panY: step.y,
      scale: 1,
      layoutWidth: 800,
      layoutHeight: 600,
      viewportWidth: 800,
      viewportHeight: 600,
    })
    assert.deepEqual(pan, { panX: 0, panY: 0 })
  })

  it("keeps part of a zoomed image inside the viewport", () => {
    const pan = clampViewportPan({
      panX: 5000,
      panY: -5000,
      scale: 4,
      layoutWidth: 800,
      layoutHeight: 600,
      viewportWidth: 800,
      viewportHeight: 600,
    })
    const visualWidth = 800 * 4
    const visualHeight = 600 * 4
    assert.ok(pan.panX < 800)
    assert.ok(pan.panX + visualWidth > 0)
    assert.ok(pan.panY < 600)
    assert.ok(pan.panY + visualHeight > 0)
  })
})
