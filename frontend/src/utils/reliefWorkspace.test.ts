import assert from "node:assert/strict"
import { describe, it } from "node:test"

import { editOpenRequests } from "./editorSession.ts"
import {
  ACCEPTED_REGIONS_HEADING,
  DRAFT_REGIONS_HEADING,
  NO_ACCEPTED_REGION_MESSAGE,
  NO_ACCEPTED_REGIONS_MESSAGE,
  NO_COMPOSED_RELIEF_MESSAGE,
  NO_DRAFT_REGIONS_MESSAGE,
  STALE_MASK_NOTICE,
  acceptedMaskDisplayUrl,
  addRegionAvailability,
  continueExistingDraft,
  defaultReliefView,
  displayedMaskRegionId,
  draftIsAccepted,
  draftRegionList,
  initialSelectedRegionId,
  maskViewIsReadOnly,
  regionsAfterDraftAccepted,
  reliefRegionList,
  selectAcceptedRegion,
  showStaleCompositionNotice,
  switchReliefView,
  viewSwitchRequests,
  workspaceCanvas,
  workspaceEditorKind,
} from "./reliefWorkspace.ts"

const bride = {
  id: "92c35f7a-0814-458b-a456-9621ed32cccc",
  name: "Bride",
  role: "primary",
  order: 0,
  updated_at: "2026-10-07T05:00:00.000000Z",
}
const groom = {
  id: "groom-region",
  name: "Groom",
  role: "primary",
  order: 1,
  updated_at: "2026-10-07T05:10:00.000000Z",
}
const groomProposal = "cc37c0b6-6ccb-47f5-9bec-46e8528ffba2"
const generatedAt = "2026-10-07T05:33:11.507050Z"

function roleLabel(role: string): string {
  return role === "primary" ? "Primary" : role
}

describe("relief workspace", () => {
  it("lists accepted regions and selects Bride with a visible selected state", () => {
    const selected = initialSelectedRegionId([groom, bride])
    assert.equal(selected, bride.id)
    const list = reliefRegionList([bride, groom], selected, roleLabel)
    assert.deepEqual(
      list.map((item) => item.name),
      ["Bride", "Groom"],
    )
    assert.equal(list[0]?.summary, "Primary · order 0")
    assert.equal(list[1]?.summary, "Primary · order 1")
    const choice = selectAcceptedRegion(bride.id)
    assert.equal(choice.selectedId, bride.id)
    const selectedEntry = reliefRegionList([bride, groom], choice.selectedId, roleLabel)[0]
    assert.equal(selectedEntry?.selected, true)
    assert.equal(selectedEntry?.className, "region-list-item is-selected")
  })

  it("shows Bride's accepted mask as a read-only overlay and switches when another region is selected", () => {
    const brideView = workspaceCanvas({
      editing: false,
      view: "mask",
      selectedRegionId: bride.id,
      generatedAt,
    })
    assert.equal(displayedMaskRegionId(brideView), bride.id)
    assert.equal(maskViewIsReadOnly(brideView), true)
    assert.equal(brideView.mode, "mask")
    if (brideView.mode === "mask") {
      assert.equal(brideView.image, "source")
      assert.equal(brideView.overlay, "positive-mask")
      assert.equal(brideView.readOnly, true)
    }
    const url = acceptedMaskDisplayUrl(`/relief/regions/${bride.id}/mask`, bride.updated_at)
    assert.match(url, new RegExp(`/regions/${bride.id}/mask`))
    const groomChoice = selectAcceptedRegion(groom.id)
    const groomView = workspaceCanvas({
      editing: false,
      view: "mask",
      selectedRegionId: groomChoice.selectedId,
      generatedAt,
    })
    assert.equal(displayedMaskRegionId(groomView), groom.id)
    assert.notEqual(displayedMaskRegionId(brideView), displayedMaskRegionId(groomView))
  })

  it("shows the source image in Source view", () => {
    const canvas = workspaceCanvas({
      editing: false,
      view: "source",
      selectedRegionId: bride.id,
      generatedAt,
    })
    assert.equal(canvas.mode, "source")
    if (canvas.mode === "source") assert.equal(canvas.image, "source")
  })

  it("shows the persisted composition in Composed Relief view", () => {
    assert.equal(defaultReliefView(true), "composed")
    const canvas = workspaceCanvas({
      editing: false,
      view: switchReliefView("composed"),
      selectedRegionId: bride.id,
      generatedAt,
    })
    assert.equal(canvas.mode, "composed")
    if (canvas.mode === "composed") {
      assert.equal(canvas.image, "relief-preview")
      assert.equal(canvas.generatedAt, generatedAt)
    }
  })

  it("switches views without composition, SAM, or depth requests", () => {
    const requests = viewSwitchRequests()
    assert.equal(requests.composition, false)
    assert.equal(requests.segmentation, false)
    assert.equal(requests.depth, false)
    assert.equal(switchReliefView("mask"), "mask")
    assert.equal(switchReliefView("source"), "source")
    assert.equal(switchReliefView("composed"), "composed")
    assert.equal(switchReliefView("physical"), "physical")
  })

  it("lists Bride as accepted and Groom as a draft that Continue resumes in place", () => {
    assert.equal(ACCEPTED_REGIONS_HEADING, "Accepted Regions")
    assert.equal(DRAFT_REGIONS_HEADING, "Draft Regions")
    const accepted = reliefRegionList([bride], bride.id, roleLabel)
    assert.equal(accepted[0]?.name, "Bride")
    assert.equal(accepted[0]?.summary, "Primary · order 0")
    assert.equal(accepted[0]?.selected, true)
    const drafts = draftRegionList(
      [{ proposalId: groomProposal, name: "Groom", role: "primary" }],
      roleLabel,
    )
    assert.equal(drafts[0]?.name, "Groom")
    assert.equal(drafts[0]?.summary, "Primary · Draft")
    assert.equal(drafts[0]?.className, "region-list-item is-draft")
    assert.equal(drafts[0]?.continueLabel, "Continue")
    assert.equal(draftIsAccepted(groomProposal, [bride.id]), false)
    const resumed = continueExistingDraft(groomProposal)
    assert.equal(resumed.proposalId, groomProposal)
    assert.equal(resumed.createsProposal, false)
    assert.equal(resumed.callsSam, false)
    assert.equal(resumed.replacesWorkingMask, false)
    assert.equal(resumed.replacesAiMask, false)
    assert.equal(resumed.acceptsRegion, false)
    assert.equal(resumed.changesAcceptedRegions, false)
    assert.equal(resumed.recomputesComposition, false)
    const choice = selectAcceptedRegion(bride.id)
    assert.equal(choice.selectedId, bride.id)
    assert.equal(choice.writesMask, false)
  })

  it("moves an accepted draft out of Draft Regions without duplicating it", () => {
    const next = regionsAfterDraftAccepted([bride], {
      id: "groom-region",
      name: "Groom",
      role: "primary",
    })
    assert.deepEqual(
      next.accepted.map((region) => [region.name, region.order]),
      [
        ["Bride", 0],
        ["Groom", 1],
      ],
    )
    assert.deepEqual(next.drafts, [])
    assert.equal(new Set(next.accepted.map((region) => region.name)).size, next.accepted.length)
  })

  it("blocks Add Region while a draft exists and shows an empty draft list otherwise", () => {
    const blocked = addRegionAvailability({ name: "Groom" })
    assert.equal(blocked.enabled, false)
    assert.equal(blocked.message, "Finish or delete the Groom draft before adding another region.")
    const open = addRegionAvailability(null)
    assert.equal(open.enabled, true)
    assert.equal(open.message, null)
    assert.equal(NO_DRAFT_REGIONS_MESSAGE, "No draft regions.")
    assert.equal(NO_ACCEPTED_REGIONS_MESSAGE, "No accepted regions.")
    const angie = reliefRegionList(
      [
        { id: "angie", name: "Angie", role: "primary", order: 0 },
        { id: "capricorn", name: "Capricorn", role: "secondary", order: 1 },
      ],
      "angie",
      (role) => (role === "secondary" ? "Secondary" : "Primary"),
    )
    assert.deepEqual(
      angie.map((item) => item.summary),
      ["Primary · order 0", "Secondary · order 1"],
    )
    assert.equal(draftRegionList([], roleLabel).length, 0)
  })

  it("keeps Edit Mask on the accepted-region editor", () => {
    assert.equal(workspaceEditorKind(bride.id), "accepted-region")
    assert.deepEqual(editOpenRequests(), ["mask-edit"])
  })

  it("leaves an outstanding proposal untouched when an accepted region is selected", () => {
    const choice = selectAcceptedRegion(bride.id)
    assert.equal(choice.selectedId, bride.id)
    assert.equal(choice.createsProposal, false)
    assert.equal(choice.deletesProposal, false)
    assert.equal(choice.recomputesComposition, false)
    assert.equal(choice.writesMask, false)
    assert.equal(groomProposal, "cc37c0b6-6ccb-47f5-9bec-46e8528ffba2")
  })

  it("keeps timestamp comparison out of provenance status", () => {
    assert.equal(
      showStaleCompositionNotice(
        [{ updated_at: "2026-10-07T06:00:00.000000Z" }],
        generatedAt,
      ),
      true,
    )
    assert.equal(STALE_MASK_NOTICE, "A region mask was saved. Apply Composition to update the relief.")
    assert.equal(showStaleCompositionNotice([{ updated_at: "2026-10-07T05:00:00.000000Z" }], generatedAt), false)
    const legacy = workspaceCanvas({
      editing: false,
      view: "composed",
      selectedRegionId: bride.id,
      generatedAt,
    })
    assert.equal(legacy.mode, "composed")
  })

  it("uses a safe empty state when no accepted region or composed relief exists", () => {
    assert.equal(defaultReliefView(false), "source")
    assert.equal(initialSelectedRegionId([]), null)
    const mask = workspaceCanvas({
      editing: false,
      view: "mask",
      selectedRegionId: null,
      generatedAt: null,
    })
    assert.equal(mask.mode, "mask-empty")
    if (mask.mode === "mask-empty") assert.equal(mask.message, NO_ACCEPTED_REGION_MESSAGE)
    const composed = workspaceCanvas({
      editing: false,
      view: "composed",
      selectedRegionId: null,
      generatedAt: null,
    })
    assert.equal(composed.mode, "composed-empty")
    if (composed.mode === "composed-empty") assert.equal(composed.message, NO_COMPOSED_RELIEF_MESSAGE)
  })
})
