export type BrushCommand = {
  kind: "brush"
  mode: "add" | "erase"
  brush_radius_px: number
  points: Array<{ x: number; y: number }>
}

export type WandCommand = {
  kind: "wand"
  mode: "add" | "subtract"
  x: number
  y: number
  tolerance: number
  contiguous: true
}

export type HistoryCommand = BrushCommand | WandCommand

export type SessionHistory = {
  commands: HistoryCommand[]
  cursor: number
}

export type HistoryEntry = {
  label: string
  index: number
  current: boolean
}

export function sessionStart(): SessionHistory {
  return { commands: [], cursor: 0 }
}

export function appendCommand(history: SessionHistory, command: HistoryCommand): SessionHistory {
  const commands = history.commands.slice(0, history.cursor)
  commands.push(command)
  return { commands, cursor: commands.length }
}

export function moveCursor(history: SessionHistory, cursor: number): SessionHistory {
  const next = Math.min(history.commands.length, Math.max(0, Math.trunc(cursor)))
  return { commands: history.commands, cursor: next }
}

export function historyEntries(history: SessionHistory): HistoryEntry[] {
  return [
    { label: "Session Start", index: 0, current: history.cursor === 0 },
    ...history.commands.map((command, index) => ({
      label: commandLabel(command),
      index: index + 1,
      current: history.cursor === index + 1,
    })),
  ]
}

function commandLabel(command: HistoryCommand): string {
  if (command.kind === "brush") return command.mode === "add" ? "Add Brush" : "Erase Brush"
  return command.mode === "add" ? "Magic Wand Add" : "Magic Wand Subtract"
}
