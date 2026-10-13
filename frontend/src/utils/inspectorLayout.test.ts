import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import { appendCommand, historyEntries, sessionStart, type HistoryCommand } from "./history.ts"
import {
  BROWSE_INSPECTOR_SECTIONS,
  EDITOR_INSPECTOR_SECTIONS,
  editorActionBar,
  historyRegion,
  inspectorFrame,
  inspectorMode,
  inspectorSizing,
  projectHeadingChrome,
  reliefVisibleChrome,
  showsViewSwitch,
} from "./inspectorLayout.ts"

const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")

function wand(index: number): HistoryCommand {
  return { kind: "wand", mode: index % 2 === 0 ? "add" : "subtract", x: index, y: index, tolerance: 12, contiguous: true }
}

describe("inspector layout", () => {
  it("organizes a draft editor into identity, tool, history, navigation, and actions", () => {
    assert.equal(inspectorMode(true), "editor")
    assert.equal(showsViewSwitch("editor"), false)
    assert.deepEqual(EDITOR_INSPECTOR_SECTIONS, ["identity", "tool", "history", "navigation", "actions"])
    const actions = editorActionBar("new-region")
    assert.equal(actions.primary, "Accept Mask")
    assert.equal(actions.close, "Close")
    assert.equal(actions.deleteProposal, true)
  })

  it("uses Save Mask and Close for an accepted region and hides Delete Proposal", () => {
    const actions = editorActionBar("accepted-region")
    assert.equal(actions.primary, "Save Mask")
    assert.equal(actions.close, "Close")
    assert.equal(actions.deleteProposal, false)
  })

  it("keeps every history entry and scrolls that list internally", () => {
    let history = sessionStart()
    for (let index = 0; index < 12; index += 1) history = appendCommand(history, wand(index))
    history = appendCommand(history, {
      kind: "brush",
      mode: "erase",
      brush_radius_px: 8,
      points: [{ x: 1, y: 2 }],
    })
    const entries = historyEntries(history)
    assert.equal(entries.length, 14)
    assert.equal(entries[0]?.label, "Session Start")
    assert.equal(entries[13]?.label, "Erase Brush")
    assert.equal(entries[13]?.current, true)
    const region = historyRegion()
    assert.equal(region.container, "history-list")
    assert.equal(region.scrollsInternally, true)
    assert.equal(region.preservesEntries, true)
    assert.match(css, /\.history-list\s*\{[^}]*overflow:\s*auto/)
    assert.match(css, /\.inspector-history\s*\{[^}]*min-height:\s*5\.5rem/)
    assert.match(css, /\.inspector-actions\s*\{/)
  })

  it("keeps browse mode on separate accepted and draft groups with Apply Composition", () => {
    assert.equal(inspectorMode(false), "browse")
    assert.equal(showsViewSwitch("browse"), true)
    assert.deepEqual(BROWSE_INSPECTOR_SECTIONS, ["accepted", "drafts", "controls", "apply"])
    assert.equal(BROWSE_INSPECTOR_SECTIONS.includes("accepted"), true)
    assert.equal(BROWSE_INSPECTOR_SECTIONS.includes("drafts"), true)
    assert.equal(EDITOR_INSPECTOR_SECTIONS.includes("apply"), false)
  })

  it("sizes the desktop inspector between 340 and 380 pixels and stacks narrow screens", () => {
    for (const width of [1920, 1366, 1100]) {
      const sizing = inspectorSizing(width)
      assert.equal(sizing.stacked, false)
      if (sizing.stacked) continue
      assert.equal(sizing.width.min >= 340 && sizing.width.max <= 380, true)
    }
    for (const width of [960, 800, 700]) {
      const sizing = inspectorSizing(width)
      assert.equal(sizing.stacked, true)
      assert.equal(sizing.width, "fluid")
    }
    assert.match(css, /grid-template-columns:\s*minmax\(0,\s*1fr\)\s*minmax\(340px,\s*380px\)/)
    assert.match(css, /@media \(max-width:\s*960px\)/)
  })

  it("anchors one full-height inspector under the navbar for browse and both editors", () => {
    assert.deepEqual(inspectorFrame("browse"), inspectorFrame("editor"))
    assert.equal(inspectorFrame("browse").belowNavbar, true)
    assert.equal(inspectorFrame("editor").architecture, "full-height")
    assert.equal(inspectorFrame("editor").pinnedPrimaryActions, true)
    assert.match(css, /\.relief-panel\s*\{[^}]*grid-row:\s*1\s*\/\s*-1/)
    assert.match(css, /grid-template-areas:\s*"heading inspector"\s*"tabs inspector"\s*"canvas inspector"/)
  })

  it("places the project kind on the title line and keeps Relief labels accessible only", () => {
    const heading = projectHeadingChrome()
    assert.equal(heading.label, "Project workspace")
    assert.equal(heading.presentation, "pill")
    assert.equal(heading.sameLineAsTitle, true)
    assert.equal(heading.longNames, "truncate")
    assert.deepEqual(reliefVisibleChrome(), {
      reliefHeading: "accessible-only",
      workingCanvas: "accessible-only",
    })
    assert.match(css, /\.project-heading\s*\{[^}]*display:\s*flex/)
    assert.match(css, /\.project-kind\s*\{/)
    assert.match(css, /\.project-heading h1\s*\{[^}]*text-overflow:\s*ellipsis/)
  })
})
