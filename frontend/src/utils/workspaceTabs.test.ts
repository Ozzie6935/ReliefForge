import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

const shell = readFileSync(new URL("../components/AppShell.tsx", import.meta.url), "utf8")
const workspace = readFileSync(new URL("../pages/WorkspacePage.tsx", import.meta.url), "utf8")
const relief = readFileSync(new URL("../components/ReliefSection.tsx", import.meta.url), "utf8")
const css = readFileSync(new URL("../index.css", import.meta.url), "utf8")
const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8")
const views = readFileSync(new URL("./reliefWorkspace.ts", import.meta.url), "utf8")

test("navbar reuses the browser favicon as the brand mark", () => {
  assert.match(html, /rel="icon" type="image\/svg\+xml" href="\/favicon\.svg"/)
  assert.match(shell, /className="brand-mark"/)
  assert.match(shell, /src="\/favicon\.svg"/)
  assert.match(shell, /alt=""/)
  assert.match(shell, /Relief Forge/)
  assert.match(shell, /CNC Relief Studio/)
  assert.match(shell, /to="\/projects"/)
  assert.match(shell, />\s*Projects\s*</)
  assert.match(shell, />\s*New Project\s*</)
  assert.match(shell, />\s*Settings\s*</)
  assert.match(css, /\.brand-mark[\s\S]*height:\s*32px/)
  assert.match(css, /\.brand-mark[\s\S]*aspect-ratio:\s*48\s*\/\s*46/)
  assert.match(css, /\.brand-mark[\s\S]*object-fit:\s*contain/)
})

test("primary workspace tabs keep order, active semantics, and an enabled Export", () => {
  const labels = [...workspace.matchAll(/>(\s*)(Overview|Source|Depth|Relief|Diagnostics|3D|Export)(\s*)</g)].map(
    (match) => match[2],
  )
  assert.deepEqual(labels, ["Overview", "Source", "Depth", "Relief", "Diagnostics", "3D", "Export"])
  assert.match(workspace, /aria-label="Workspace"/)
  assert.match(workspace, /aria-current=\{section === "overview" \? "page" : undefined\}/)
  assert.match(workspace, /aria-current=\{section === "mesh" \? "page" : undefined\}/)
  assert.match(workspace, /aria-current=\{section === "export" \? "page" : undefined\}/)
  assert.match(workspace, /setSection\("export"\)/)
  assert.match(workspace, /ExportSection/)
  assert.doesNotMatch(workspace, /FUTURE_SECTIONS/)
  assert.doesNotMatch(workspace, /Not yet available/)
  assert.match(workspace, /lazy\(\(\) => import\("\.\.\/components\/MeshPreviewSection\.tsx"\)\)/)
})

test("relief secondary tabs share the compact view switch", () => {
  assert.match(views, /label: "Source"/)
  assert.match(views, /label: "Mask"/)
  assert.match(views, /label: "Composed Relief"/)
  assert.match(views, /label: "Physical Height"/)
  assert.match(relief, /className="view-switch" role="radiogroup" aria-label="Relief view"/)
  assert.match(relief, /role="radio"/)
  assert.match(relief, /aria-checked=\{workspaceView === item\.id\}/)
})

test("raised tabs join the workspace surface and stay on one scrolling row", () => {
  assert.match(css, /\.workspace-nav button \{[\s\S]*min-height:\s*40px/)
  assert.match(css, /\.workspace-nav button \{[\s\S]*border-radius:\s*10px 10px 0 0/)
  assert.match(css, /\.view-switch button \{[\s\S]*min-height:\s*32px/)
  assert.match(css, /\.view-switch button \{[\s\S]*border-radius:\s*8px 8px 0 0/)
  assert.match(css, /\.workspace-nav button\.active[\s\S]*background:\s*#ffffff/)
  assert.match(css, /\.relief-main,[\s\S]*\.mesh-main\s*\{[\s\S]*background:\s*#ffffff/)
  assert.match(css, /\.workspace-nav button\.active[\s\S]*margin-bottom:\s*0/)
  assert.match(css, /box-shadow:\s*inset 0 -1px 0 #c5ced8/)
  assert.match(css, /\.workspace-nav button:disabled[\s\S]*opacity:\s*1/)
  assert.match(css, /\.workspace-nav[\s\S]*min-width:\s*0/)
  assert.match(css, /\.workspace-nav,\s*\.view-switch\s*\{[\s\S]*background:\s*transparent/)
  assert.match(relief, /aria-checked=\{workspaceView === item\.id\}/)
})

test("tab styles stay scoped, focusable, and scroll instead of shrinking", () => {
  assert.match(css, /\.workspace-nav button\.active/)
  assert.match(css, /\.view-switch button\[aria-checked="true"\]/)
  assert.match(css, /\.workspace-nav button:disabled/)
  assert.match(css, /\.workspace-nav button:focus-visible/)
  assert.match(css, /\.view-switch button:focus-visible/)
  assert.match(css, /\.workspace-nav[\s\S]*overflow-x:\s*auto/)
  assert.match(css, /\.workspace-nav[\s\S]*flex-wrap:\s*nowrap/)
  assert.match(css, /\.view-switch button[\s\S]*font-size:\s*0\.8rem/)
  assert.match(css, /\.workspace-nav button \{[\s\S]*font-size:\s*0\.86rem/)
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)/)
  assert.doesNotMatch(css, /button:hover,\s*\n\s*\.workspace-nav/)
})
