import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"

import type { Project } from "../types/project.ts"
import {
  duplicateNavigationTarget,
  duplicateRequestBody,
  projectDeleteConfirmed,
  projectLibraryEntries,
  suggestedDuplicateName,
} from "./projectLibrary.ts"

function project(overrides: Partial<Project> = {}): Project {
  return {
    schema_version: 1,
    id: "project",
    name: "Sample",
    slug: "sample",
    units: "mm",
    stock: { width_mm: 304.8, height_mm: 203.2 },
    artwork: { width_mm: 279.4, height_mm: 177.8, center_x_mm: 152.4, center_y_mm: 101.6 },
    source: null,
    updated_at: "2026-10-08T13:00:00.000Z",
    ...overrides,
  }
}

describe("project library", () => {
  it("formats stock, artwork, source, and modified date without reordering the list", () => {
    const newer = project({
      id: "newer",
      name: "Newer",
      source: {
        filename: "original.jpg",
        original_filename: "photo.jpg",
        media_type: "image/jpeg",
        width_px: 1013,
        height_px: 1023,
      },
    })
    const older = project({ id: "older", name: "Older", updated_at: null, source: null })
    const entries = projectLibraryEntries([newer, older], "in")
    assert.deepEqual(
      entries.map((entry) => entry.id),
      ["newer", "older"],
    )
    assert.equal(entries[0]?.stock, "12 in × 8 in")
    assert.equal(entries[0]?.artwork, "11 in × 7 in")
    assert.equal(entries[0]?.source, "Available · 1013 × 1023 px")
    assert.equal(typeof entries[0]?.modified, "string")
    assert.notEqual(entries[0]?.modified, "None")
    assert.equal(entries[1]?.source, "No source image")
    assert.equal(entries[1]?.modified, "None")
  })

  it("formats the same geometry in millimeters", () => {
    const [entry] = projectLibraryEntries([project()], "mm")
    assert.equal(entry?.stock, "304.8 mm × 203.2 mm")
    assert.equal(entry?.artwork, "279.4 mm × 177.8 mm")
  })

  it("gives library and delete actions a visible keyboard focus ring", () => {
    const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")
    assert.match(css, /\.projects-library a:focus-visible[\s\S]*outline:\s*2px solid #1d6fbf/)
    assert.match(css, /\.modal button:focus-visible[\s\S]*outline:\s*2px solid #1d6fbf/)
  })

  it("suggests a copy name and opens the duplicate only after an id exists", () => {
    const page = readFileSync(new URL("../pages/ProjectsPage.tsx", import.meta.url), "utf8")
    const dialog = readFileSync(new URL("../components/DuplicateProjectDialog.tsx", import.meta.url), "utf8")
    assert.equal(suggestedDuplicateName("Wedding Relief Benchmark"), "Wedding Relief Benchmark copy")
    assert.deepEqual(duplicateRequestBody("  Angie Dance copy  "), { name: "Angie Dance copy" })
    assert.equal(duplicateRequestBody("   "), null)
    assert.equal(duplicateNavigationTarget("abc"), "/projects/abc")
    assert.equal(duplicateNavigationTarget(""), null)
    assert.match(page, /aria-label=\{`Duplicate \$\{entry\.name\}`\}/)
    const duplicateHandler = page.slice(page.indexOf("async function confirmDuplicate"), page.indexOf("async function confirmDelete"))
    assert.match(duplicateHandler, /duplicateProject\(pendingDuplicate\.id, name\)/)
    assert.match(duplicateHandler, /navigate\(target\)/)
    assert.doesNotMatch(duplicateHandler, /setProjects/)
    assert.match(dialog, /Duplicating…/)
    assert.match(dialog, /disabled=\{body === null \|\| pending\}/)
    assert.match(page, /useUnits\(\)/)
    assert.doesNotMatch(page, /setUnit\(/)
  })

  it("requires the exact project name before delete is confirmed", () => {
    assert.equal(projectDeleteConfirmed("Wedding Relief Benchmark", "Wedding Relief Benchmark"), true)
    assert.equal(projectDeleteConfirmed("wedding relief benchmark", "Wedding Relief Benchmark"), false)
    assert.equal(projectDeleteConfirmed("Wedding Relief Benchmark ", "Wedding Relief Benchmark"), false)
    assert.equal(projectDeleteConfirmed("", "Wedding Relief Benchmark"), false)
  })
})