import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")
const provenance = readFileSync(new URL("../components/ProvenanceStatus.tsx", import.meta.url), "utf8")
const diagnostics = readFileSync(new URL("../components/DiagnosticsSection.tsx", import.meta.url), "utf8")
const workspace = readFileSync(new URL("../pages/WorkspacePage.tsx", import.meta.url), "utf8")
const shell = readFileSync(new URL("../components/AppShell.tsx", import.meta.url), "utf8")

test("provenance stays labeled for every status and keeps recovery guidance", () => {
  assert.match(provenance, /statusLabel\(status\)/)
  assert.match(provenance, /statusExplanation\(status\)/)
  assert.match(provenance, /provenance-guidance/)
  assert.match(provenance, /role="status"/)
  assert.match(css, /\.provenance-status\.is-current/)
  assert.match(css, /\.provenance-status\.is-stale/)
  assert.match(css, /\.provenance-status\.is-unknown/)
  assert.match(css, /\.provenance-status\.is-missing/)
  assert.match(css, /border-left:\s*3px solid #0d7a3f/)
  assert.match(css, /border-left:\s*3px dashed #243044/)
  assert.match(css, /border-left:\s*3px dotted #5c6775/)
  assert.doesNotMatch(provenance, /status === "unknown" \? "Current"/)
})

test("artwork frames keep contain sizing and diagnostics mapping markup", () => {
  assert.match(css, /\.depth-fit-image\s*\{[^}]*object-fit:\s*contain/)
  assert.match(css, /\.diagnostics-fit-image\s*\{[^}]*width:\s*100%/)
  assert.match(css, /\.diagnostics-fit-image\s*\{[^}]*height:\s*100%/)
  assert.match(css, /\.diagnostics-fit-image\s*\{[^}]*object-fit:\s*contain/)
  assert.match(css, /\.diagnostics-fit-image\s*\{[^}]*object-position:\s*center/)
  assert.match(css, /\.depth-fit\s*\{[^}]*background:\s*#e4e7ec/)
  const viewport = css.slice(css.lastIndexOf(".mask-viewport {"))
  assert.match(viewport, /background:\s*#e4e7ec/)
  assert.match(viewport, /border:\s*1px solid #c5ced8/)
  assert.doesNotMatch(css, /\.diagnostics-fit\s*\{[^}]*padding:/)
  assert.match(diagnostics, /displayPointToPixel/)
  assert.match(diagnostics, /className="diagnostics-fit"/)
  assert.match(diagnostics, /className="diagnostics-fit-image"/)
})

test("workspace chrome keeps the inspector contract and branded navigation", () => {
  assert.match(css, /minmax\(340px,\s*380px\)/)
  assert.match(css, /grid-template-areas:\s*"heading inspector"\s*"tabs inspector"\s*"canvas inspector"/)
  assert.match(css, /\.inspector-body\s*\{[^}]*overflow-y:\s*auto/)
  assert.match(workspace, /aria-label="Workspace"/)
  assert.match(workspace, /aria-current=\{section === "export" \? "page" : undefined\}/)
  assert.doesNotMatch(workspace, /Not yet available/)
  assert.match(shell, /src="\/favicon\.svg"/)
  assert.match(shell, /CNC Relief Studio/)
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)/)
  assert.match(css, /\.workspace-nav button:focus-visible/)
})
