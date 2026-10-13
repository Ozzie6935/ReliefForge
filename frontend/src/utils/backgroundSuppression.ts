import type { ReliefRecord } from "../types/project.ts"

export interface BackgroundSettings {
  background_depth_range: number
  background_elevation: number
  background_feather: number
}

export const BACKGROUND_IDENTITY: BackgroundSettings = {
  background_depth_range: 1,
  background_elevation: 0,
  background_feather: 0,
}

function finiteInRange(value: number, low: number, high: number): boolean {
  return Number.isFinite(value) && value >= low && value <= high
}

export function backgroundFromRelief(relief: ReliefRecord | null | undefined): BackgroundSettings {
  return {
    background_depth_range: relief?.background_depth_range ?? BACKGROUND_IDENTITY.background_depth_range,
    background_elevation: relief?.background_elevation ?? BACKGROUND_IDENTITY.background_elevation,
    background_feather: relief?.background_feather ?? BACKGROUND_IDENTITY.background_feather,
  }
}

export function backgroundSettingsPending(draft: BackgroundSettings, saved: BackgroundSettings): boolean {
  return (
    draft.background_depth_range !== saved.background_depth_range ||
    draft.background_elevation !== saved.background_elevation ||
    draft.background_feather !== saved.background_feather
  )
}

export function backgroundSettingsProblem(draft: BackgroundSettings): string | null {
  if (
    finiteInRange(draft.background_depth_range, 0, 2) &&
    finiteInRange(draft.background_elevation, -1, 1) &&
    finiteInRange(draft.background_feather, 0, 100)
  ) {
    return null
  }
  return "Background depth range must be from 0 to 2, elevation from -1 to 1, and feather from 0 to 100."
}
