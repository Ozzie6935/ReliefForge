import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import type { HealthResponse } from "../types/health.ts"
import {
  SETTINGS_STOCK_NOTE,
  SETTINGS_UNIT_NOTE,
  SETTINGS_UNIT_OPTIONS,
  SETTINGS_UNIT_STORAGE_KEY,
  settingsApiVersion,
  settingsAvailableLabel,
  settingsConnectionLabel,
  settingsDeviceLabel,
  settingsInstalledLabel,
} from "./settingsPresentation.ts"
import { formatLength } from "./units.ts"

const health: HealthResponse = {
  status: "ok",
  application: "Relief Forge",
  api_version: "0.1.0",
  compute: {
    preferred_device: "mps",
    torch_installed: true,
    mps_available: true,
  },
}

describe("settings presentation", () => {
  it("keeps inches and millimeters as the display preference and stores geometry in millimeters", () => {
    assert.deepEqual(
      SETTINGS_UNIT_OPTIONS.map((option) => option.label),
      ["Inches", "Millimeters"],
    )
    assert.deepEqual(
      SETTINGS_UNIT_OPTIONS.map((option) => option.value),
      ["in", "mm"],
    )
    assert.equal(SETTINGS_UNIT_STORAGE_KEY, "cnc-relief-studio.units")
    assert.match(SETTINGS_UNIT_NOTE, /stored internally in millimeters/)
    assert.match(SETTINGS_UNIT_NOTE, /does not change the saved project/)
    assert.match(SETTINGS_STOCK_NOTE, /1\.50 in \(38\.10 mm\)/)
    assert.match(SETTINGS_STOCK_NOTE, /own thickness/)
    assert.equal(formatLength(304.8, "in"), "12 in")
    assert.equal(formatLength(304.8, "mm"), "304.8 mm")
  })

  it("shows health fields from the API and leaves them blank when the backend is disconnected", () => {
    assert.equal(settingsConnectionLabel("connected"), "Connected")
    assert.equal(settingsConnectionLabel("disconnected"), "Disconnected")
    assert.equal(settingsApiVersion(health, "connected"), "0.1.0")
    assert.equal(settingsDeviceLabel(health, "connected"), "MPS")
    assert.equal(settingsInstalledLabel(health.compute.torch_installed, "connected"), "Installed")
    assert.equal(settingsAvailableLabel(health.compute.mps_available, "connected"), "Available")
    assert.equal(settingsDeviceLabel({ ...health, compute: { ...health.compute, preferred_device: "cpu" } }, "connected"), "CPU")

    assert.equal(settingsApiVersion(null, "disconnected"), "—")
    assert.equal(settingsDeviceLabel(null, "disconnected"), "—")
    assert.equal(settingsInstalledLabel(undefined, "disconnected"), "—")
    assert.equal(settingsAvailableLabel(undefined, "disconnected"), "—")
    assert.equal(settingsConnectionLabel("checking"), "Checking")
  })

  it("keeps the settings page centered and gives unit controls a visible focus ring", () => {
    const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")
    assert.match(css, /\.settings-page\s*\{[^}]*width:\s*min\(880px,\s*100%\)/)
    assert.match(css, /\.settings-page button:focus-visible[\s\S]*outline:\s*2px solid #1d6fbf/)
  })

  it("does not describe PyTorch or MPS availability as a loaded model", () => {
    const lines = [
      settingsDeviceLabel(health, "connected"),
      settingsInstalledLabel(true, "connected"),
      settingsAvailableLabel(true, "connected"),
    ]
    for (const line of lines) {
      assert.doesNotMatch(line, /model/i)
      assert.doesNotMatch(line, /loaded/i)
    }
  })
})
