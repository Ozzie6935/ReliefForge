import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { clientOffsetFromPixel, elementLayoutFrame, imagePixelFromClient } from "./imageCoordinates.ts"
import { contentPlacement } from "./viewport.ts"

const SOURCE_WIDTH = 1600
const SOURCE_HEIGHT = 900
const BORDER = 2

/** The editor stage sits on the viewport padding edge. translate(pan) scale(view) uses origin 0,0. */
function projectClick(input: {
  viewportX: number
  viewportY: number
  viewportWidth: number
  viewportHeight: number
  viewScale: number
  panX: number
  panY: number
  clientX: number
  clientY: number
}) {
  const placed = contentPlacement(input.viewportWidth, input.viewportHeight, SOURCE_WIDTH, SOURCE_HEIGHT)
  assert.ok(placed)
  const layoutWidth = SOURCE_WIDTH * placed.scale
  const layoutHeight = SOURCE_HEIGHT * placed.scale
  const stageOriginX = input.viewportX + BORDER
  const stageOriginY = input.viewportY + BORDER
  const visual = {
    left: stageOriginX + input.panX,
    top: stageOriginY + input.panY,
    width: layoutWidth * input.viewScale,
    height: layoutHeight * input.viewScale,
    naturalWidth: SOURCE_WIDTH,
    naturalHeight: SOURCE_HEIGHT,
  }
  const source = imagePixelFromClient(input.clientX, input.clientY, visual)
  assert.ok(source)
  const local = clientOffsetFromPixel(
    source,
    elementLayoutFrame({
      offsetWidth: layoutWidth,
      offsetHeight: layoutHeight,
      naturalWidth: SOURCE_WIDTH,
      naturalHeight: SOURCE_HEIGHT,
    }),
  )
  assert.ok(local)
  return {
    source,
    screenX: stageOriginX + input.panX + local.x * input.viewScale,
    screenY: stageOriginY + input.panY + local.y * input.viewScale,
    layoutWidth,
    contentScale: placed.scale,
  }
}

function assertRoundTrip(
  label: string,
  input: {
    viewportWidth: number
    viewportHeight: number
    viewScale: number
    panX: number
    panY: number
    clientX: number
    clientY: number
  },
) {
  const projected = projectClick({
    viewportX: 40,
    viewportY: 70,
    ...input,
  })
  const error = Math.hypot(projected.screenX - input.clientX, projected.screenY - input.clientY)
  assert.ok(error <= 0.05, `${label} round-trip error ${error}`)
  return projected
}

describe("editor source coordinate round trip", () => {
  it("keeps an 800% click on screen when the stage width is fractional", () => {
    const naturalWidth = 1013
    const naturalHeight = 1023
    const layoutWidth = 459.453
    const layoutHeight = 464.219
    const viewScale = 8 / (layoutWidth / naturalWidth)
    const stageOriginX = 120
    const stageOriginY = 80
    const panX = -3662
    const panY = -3862.77
    const visualWidth = layoutWidth * viewScale
    const visualHeight = layoutHeight * viewScale
    const clientX = stageOriginX + panX + visualWidth * 0.82
    const clientY = stageOriginY + panY + visualHeight * 0.67
    const visual = {
      left: stageOriginX + panX,
      top: stageOriginY + panY,
      width: visualWidth,
      height: visualHeight,
      naturalWidth,
      naturalHeight,
    }
    const source = imagePixelFromClient(clientX, clientY, visual)
    assert.ok(source)
    const rounded = clientOffsetFromPixel(
      source,
      elementLayoutFrame({
        offsetWidth: Math.round(layoutWidth),
        offsetHeight: Math.round(layoutHeight),
        naturalWidth,
        naturalHeight,
      }),
    )
    const fractional = clientOffsetFromPixel(source, {
      left: 0,
      top: 0,
      width: layoutWidth,
      height: layoutHeight,
      naturalWidth,
      naturalHeight,
    })
    assert.ok(rounded && fractional)
    const roundedError = Math.hypot(
      stageOriginX + panX + rounded.x * viewScale - clientX,
      stageOriginY + panY + rounded.y * viewScale - clientY,
    )
    const fractionalError = Math.hypot(
      stageOriginX + panX + fractional.x * viewScale - clientX,
      stageOriginY + panY + fractional.y * viewScale - clientY,
    )
    assert.ok(roundedError > 1)
    assert.ok(fractionalError <= 0.05)
  })

  const wide = { viewportWidth: 2000, viewportHeight: 400 }
  const tall = { viewportWidth: 500, viewportHeight: 800 }

  function fitted(viewportWidth: number, viewportHeight: number) {
    const placed = contentPlacement(viewportWidth, viewportHeight, SOURCE_WIDTH, SOURCE_HEIGHT)
    assert.ok(placed)
    return {
      layoutWidth: SOURCE_WIDTH * placed.scale,
      layoutHeight: SOURCE_HEIGHT * placed.scale,
      contentScale: placed.scale,
    }
  }

  function clickIn(viewportWidth: number, viewportHeight: number, viewScale: number, panX: number, panY: number) {
    const { layoutWidth, layoutHeight } = fitted(viewportWidth, viewportHeight)
    return {
      viewportWidth,
      viewportHeight,
      viewScale,
      panX,
      panY,
      clientX: 40 + BORDER + panX + layoutWidth * viewScale * 0.37,
      clientY: 70 + BORDER + panY + layoutHeight * viewScale * 0.62,
    }
  }

  it("centers a wide Fit viewport on the horizontal slack", () => {
    const { layoutWidth } = fitted(wide.viewportWidth, wide.viewportHeight)
    const panX = (wide.viewportWidth - layoutWidth) / 2
    const projected = assertRoundTrip("Fit wide", clickIn(wide.viewportWidth, wide.viewportHeight, 1, panX, 0))
    assert.ok(panX > 1)
    assert.ok(projected.source.x > 0 && projected.source.x < SOURCE_WIDTH)
  })

  it("centers a tall Fit viewport on the vertical slack", () => {
    const { layoutHeight } = fitted(tall.viewportWidth, tall.viewportHeight)
    const panY = (tall.viewportHeight - layoutHeight) / 2
    assertRoundTrip("Fit tall", clickIn(tall.viewportWidth, tall.viewportHeight, 1, 0, panY))
    assert.ok(panY > 1)
  })

  for (const percent of [100, 125, 200, 400, 800]) {
    it(`round-trips ${percent}% with pan in a centered Fit viewport`, () => {
      const { layoutWidth, contentScale } = fitted(wide.viewportWidth, wide.viewportHeight)
      const viewScale = percent / 100 / contentScale
      const panX = (wide.viewportWidth - layoutWidth) / 2 - 80
      const panY = -36
      assertRoundTrip(`${percent}%`, clickIn(wide.viewportWidth, wide.viewportHeight, viewScale, panX, panY))
    })
  }
})
