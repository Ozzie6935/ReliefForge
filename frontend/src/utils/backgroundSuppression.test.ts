import assert from "node:assert/strict"
import { describe, it } from "node:test"

import {
  BACKGROUND_IDENTITY,
  backgroundFromRelief,
  backgroundSettingsPending,
  backgroundSettingsProblem,
} from "./backgroundSuppression.ts"

describe("uncovered background settings", () => {
  it("treats a missing record as the identity defaults", () => {
    assert.deepEqual(backgroundFromRelief(null), BACKGROUND_IDENTITY)
    assert.deepEqual(backgroundFromRelief(undefined), BACKGROUND_IDENTITY)
  })

  it("marks only a changed draft as pending and does not treat identity as a request", () => {
    const original = globalThis.fetch
    let calls = 0
    globalThis.fetch = (() => {
      calls += 1
      return Promise.reject(new Error("processing request"))
    }) as typeof fetch
    try {
      assert.equal(backgroundSettingsPending(BACKGROUND_IDENTITY, BACKGROUND_IDENTITY), false)
      assert.equal(
        backgroundSettingsPending({ ...BACKGROUND_IDENTITY, background_depth_range: 0 }, BACKGROUND_IDENTITY),
        true,
      )
      assert.equal(
        backgroundSettingsPending(
          { ...BACKGROUND_IDENTITY, background_elevation: -0.2 },
          { ...BACKGROUND_IDENTITY, background_elevation: -0.2 },
        ),
        false,
      )
      assert.equal(calls, 0)
    } finally {
      globalThis.fetch = original
    }
  })

  it("rejects non-finite and out-of-range values without clamping", () => {
    assert.equal(backgroundSettingsProblem(BACKGROUND_IDENTITY), null)
    assert.equal(backgroundSettingsProblem({ ...BACKGROUND_IDENTITY, background_depth_range: 2 }), null)
    assert.equal(backgroundSettingsProblem({ ...BACKGROUND_IDENTITY, background_elevation: -1 }), null)
    assert.equal(backgroundSettingsProblem({ ...BACKGROUND_IDENTITY, background_feather: 100 }), null)
    assert.ok(backgroundSettingsProblem({ ...BACKGROUND_IDENTITY, background_depth_range: 2.01 }))
    assert.ok(backgroundSettingsProblem({ ...BACKGROUND_IDENTITY, background_elevation: Number.NaN }))
    assert.ok(backgroundSettingsProblem({ ...BACKGROUND_IDENTITY, background_feather: Number.POSITIVE_INFINITY }))
    assert.ok(backgroundSettingsProblem({ ...BACKGROUND_IDENTITY, background_depth_range: -0.01 }))
  })
})
