import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  imagePointerAction,
  maskOperation,
  promptsAfterMaskGeneration,
  type EditorTool,
} from "./editorTools.ts"
import { appendCommand, historyEntries, moveCursor, sessionStart, type HistoryCommand } from "./history.ts"

const tools: EditorTool[] = [
  "positive",
  "negative",
  "box",
  "add",
  "erase",
  "wand-add",
  "wand-subtract",
  "pan",
  "zoom-area",
]

describe("image pointer action", () => {
  it("resolves every tool to exactly one image action", () => {
    const actions = tools.map((tool) => imagePointerAction(tool))
    assert.deepEqual(actions, [
      "sam-point",
      "sam-point",
      "sam-box",
      "add",
      "erase",
      "wand-add",
      "wand-subtract",
      "pan",
      "zoom-area",
    ])
    assert.equal(new Set(actions).size, actions.length - 1)
  })

  it("does not treat pan or zoom area as a mask operation", () => {
    assert.equal(maskOperation("pan"), null)
    assert.equal(maskOperation("zoom-area"), null)
    assert.equal(maskOperation("positive"), null)
    assert.equal(maskOperation("wand-add"), "wand-add")
    assert.equal(maskOperation("wand-subtract"), "wand-subtract")
    assert.equal(maskOperation("add"), "add")
    assert.equal(maskOperation("erase"), "erase")
  })
})

describe("sam prompts after generation", () => {
  const prompts = {
    positive: [{ x: 1, y: 2 }],
    negative: [{ x: 3, y: 4 }],
    box: { x0: 0, y0: 0, x1: 5, y1: 6 },
  }

  it("clears prompts after a successful mask update", () => {
    assert.deepEqual(promptsAfterMaskGeneration(true, prompts), {
      positive: [],
      negative: [],
      box: null,
    })
  })

  it("keeps prompts when mask generation fails", () => {
    assert.equal(promptsAfterMaskGeneration(false, prompts), prompts)
  })
})

describe("session history", () => {
  const brush: HistoryCommand = {
    kind: "brush",
    mode: "add",
    brush_radius_px: 4,
    points: [{ x: 1, y: 1 }],
  }
  const wand: HistoryCommand = {
    kind: "wand",
    mode: "subtract",
    x: 2,
    y: 3,
    tolerance: 18,
    contiguous: true,
  }

  it("moves the cursor without dropping later commands", () => {
    const history = appendCommand(appendCommand(sessionStart(), brush), wand)
    assert.deepEqual(
      historyEntries(history).map((entry) => entry.label),
      ["Session Start", "Add Brush", "Magic Wand Subtract"],
    )
    assert.equal(historyEntries(history)[2]?.current, true)
    const earlier = moveCursor(history, 1)
    assert.equal(earlier.cursor, 1)
    assert.equal(earlier.commands.length, 2)
    assert.equal(historyEntries(earlier)[1]?.current, true)
    const forward = moveCursor(earlier, 2)
    assert.equal(forward.cursor, 2)
    assert.equal(forward.commands.length, 2)
  })

  it("drops the forward branch when a new operation is made", () => {
    const history = appendCommand(appendCommand(sessionStart(), brush), wand)
    const branched = appendCommand(moveCursor(history, 0), {
      kind: "brush",
      mode: "erase",
      brush_radius_px: 2,
      points: [{ x: 4, y: 4 }],
    })
    assert.equal(branched.cursor, 1)
    assert.deepEqual(
      historyEntries(branched).map((entry) => entry.label),
      ["Session Start", "Erase Brush"],
    )
  })
})
