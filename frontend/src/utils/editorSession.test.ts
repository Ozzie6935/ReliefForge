import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { historyEntries, sessionStart } from "./history.ts"
import {
  DISCARD_UNSAVED_CHANGES,
  acceptedRegionInspectorActions,
  closeShouldConfirmDiscard,
  compositionNeedsApply,
  discardPrompt,
  editOpenRequests,
  editSessionDirty,
  editorKind,
  generateMaskEnabled,
  helpTopics,
  keepEditing,
  primaryMaskAction,
  resetMaskAction,
  visibleSamLabels,
} from "./editorSession.ts"

describe("accepted region editor", () => {
  it("offers Edit Mask on an accepted region", () => {
    assert.deepEqual(acceptedRegionInspectorActions(), ["Edit Mask", "Delete"])
  })

  it("keeps edit mode distinct from a new region", () => {
    assert.equal(editorKind("92c35f7a-0814-458b-a456-9621ed32cccc"), "accepted-region")
    assert.equal(editorKind(null), "new-region")
  })

  it("labels the primary action Save Mask while editing and Accept Mask for a new region", () => {
    assert.equal(primaryMaskAction("accepted-region"), "Save Mask")
    assert.equal(primaryMaskAction("new-region"), "Accept Mask")
  })

  it("labels reset from the saved mask while editing and from the AI mask for a new proposal", () => {
    assert.equal(resetMaskAction("accepted-region"), "Reset to Saved Mask")
    assert.equal(resetMaskAction("new-region"), "Reset to AI Mask")
  })

  it("asks before closing a dirty edit and lets a clean edit close", () => {
    const baseline = { name: "Bride", role: "primary" }
    const clean = { ...baseline, historyCursor: 0 }
    assert.equal(editSessionDirty(baseline, clean), false)
    assert.equal(discardPrompt("accepted-region", editSessionDirty(baseline, clean)), null)
    assert.equal(closeShouldConfirmDiscard("new-region", true), false)
  })

  it("treats name, role, and mask edits as unsaved changes", () => {
    const baseline = { name: "Bride", role: "primary" }
    assert.equal(editSessionDirty(baseline, { name: "Bride Renamed", role: "primary", historyCursor: 0 }), true)
    assert.equal(editSessionDirty(baseline, { name: "Bride", role: "secondary", historyCursor: 0 }), true)
    assert.equal(editSessionDirty(baseline, { name: "Bride", role: "primary", historyCursor: 1 }), true)
    assert.equal(editSessionDirty(baseline, { name: "Bride Renamed", role: "primary", historyCursor: 2 }), true)
    assert.equal(discardPrompt("accepted-region", true), DISCARD_UNSAVED_CHANGES)
    assert.equal(DISCARD_UNSAVED_CHANGES.includes("mask"), false)
  })

  it("keeps the current edit when the user chooses Keep Editing", () => {
    const session = { name: "Bride Renamed", role: "secondary", historyCursor: 1 }
    assert.equal(keepEditing(session), session)
    assert.deepEqual(keepEditing(session), { name: "Bride Renamed", role: "secondary", historyCursor: 1 })
  })

  it("hides SAM controls while editing an accepted region and keeps them for a new proposal", () => {
    assert.deepEqual(visibleSamLabels("accepted-region"), [])
    assert.deepEqual(visibleSamLabels("new-region"), [
      "Positive Prompt",
      "Negative Prompt",
      "Box Prompt",
      "Generate Mask",
      "Update Mask",
    ])
    assert.equal(generateMaskEnabled("new-region"), true)
    const acceptedHelp = helpTopics("accepted-region").map((entry) => entry.term)
    const newHelp = helpTopics("new-region").map((entry) => entry.term)
    assert.equal(acceptedHelp.includes("Positive Prompt"), false)
    assert.equal(acceptedHelp.includes("Negative Prompt"), false)
    assert.equal(acceptedHelp.includes("Box Prompt"), false)
    assert.equal(acceptedHelp.includes("Reset to Saved Mask"), true)
    assert.equal(acceptedHelp.includes("Save Mask"), true)
    assert.equal(newHelp.includes("Positive Prompt"), true)
    assert.equal(newHelp.includes("Negative Prompt"), true)
    assert.equal(newHelp.includes("Box Prompt"), true)
  })

  it("opens from the accepted mask and starts history at Session Start without generating a mask", () => {
    assert.deepEqual(editOpenRequests(), ["mask-edit"])
    assert.equal(editOpenRequests().includes("segmentation/predict"), false)
    assert.equal(generateMaskEnabled("accepted-region"), false)
    assert.equal(generateMaskEnabled("new-region"), true)
    const entries = historyEntries(sessionStart())
    assert.equal(entries[0]?.label, "Session Start")
    assert.equal(entries[0]?.current, true)
  })

  it("marks composition stale after a mask save until Apply Composition runs", () => {
    assert.equal(compositionNeedsApply("2026-10-07T06:00:00.000Z", "2026-10-07T05:00:00.000Z"), true)
    assert.equal(compositionNeedsApply("2026-10-07T05:00:00.000Z", "2026-10-07T06:00:00.000Z"), false)
  })
})
