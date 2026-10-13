export interface Stock {
  width_mm: number
  height_mm: number
  thickness_mm?: number | null
}

export interface Artwork {
  width_mm: number
  height_mm: number
  center_x_mm: number
  center_y_mm: number
}

export type ArtifactStatus = "missing" | "unknown" | "current" | "stale"

export interface SourceImage {
  filename: string
  original_filename: string
  media_type: string
  width_px: number
  height_px: number
  content_sha256?: string | null
  status?: ArtifactStatus | null
}

export interface DepthConditioningParameters {
  far_percentile: number
  near_percentile: number
  gamma: number
  contrast: number
  smoothing: number
  edge_preservation: number
  invert: boolean
}

export interface ConditioningRecord {
  parameters: DepthConditioningParameters
  width_px: number
  height_px: number
  dtype: "float32"
  range_min: number
  range_max: number
  conditioned_filename: "conditioned_depth.npy"
  preview_filename: "preview.png"
  generated_at: string
  content_sha256?: string | null
  raw_sha256?: string | null
  settings_sha256?: string | null
  status?: ArtifactStatus | null
}

export interface DepthArtifact {
  model_id: string
  model_version: string
  device: "mps" | "cpu"
  input_size: number
  width_px: number
  height_px: number
  raw_filename: "raw_depth.npy"
  preview_filename: "preview.png"
  generated_at: string
  conditioning?: ConditioningRecord | null
  content_sha256?: string | null
  source_sha256?: string | null
  settings_sha256?: string | null
  status?: ArtifactStatus | null
}

export interface HistogramSeries {
  bins: number
  counts: number[]
  minimum: number
  maximum: number
}

export interface DepthHistogram {
  raw: HistogramSeries
  conditioned: HistogramSeries | null
}

export type ReliefRole = "primary" | "secondary" | "supporting" | "background" | "custom"

export interface ReliefRegion {
  id: string
  name: string
  role: ReliefRole
  mask_filename: string
  elevation: number
  depth_range: number
  feather: number
  order: number
  detail_amount?: number
  detail_scale_mm?: number | null
  created_at: string
  updated_at: string
}

export interface PhysicalHeightRecord {
  stock_thickness_mm: number
  base_thickness_mm: number
  maximum_surface_height_mm: number
  generated_at: string
  height_path: "depth/composition/physical/physical_height_mm.npy"
  preview_path: "depth/composition/physical/preview.png"
  stats_path: "depth/composition/physical/stats.json"
  source_relief_sha256: string
  source_relief_generated_at: string
  width_px: number
  height_px: number
  dtype: "float32"
  height_min_mm: number
  height_max_mm: number
  maximum_cut_depth_mm: number
  relief_amplitude_mm: number
  content_sha256?: string | null
  settings_sha256?: string | null
  status?: ArtifactStatus | null
}

export interface ReliefRecord {
  regions: ReliefRegion[]
  conditioned_generated_at?: string | null
  width_px?: number | null
  height_px?: number | null
  dtype?: "float32" | null
  range_min?: number | null
  range_max?: number | null
  relief_filename?: "relief_depth.npy" | null
  preview_filename?: "preview.png" | null
  generated_at?: string | null
  physical?: PhysicalHeightRecord | null
  content_sha256?: string | null
  conditioned_sha256?: string | null
  settings_sha256?: string | null
  settings_version?: 2 | 3 | 4 | null
  background_depth_range?: number
  background_elevation?: number
  background_feather?: number
  status?: ArtifactStatus | null
}

export interface SegmentationProposal {
  proposal_id: string
  width_px: number
  height_px: number
  model_id: string
  name?: string
  role?: ReliefRole | null
  origin_type?: "new_region" | "accepted_region"
  region_id?: string | null
}

export interface Project {
  schema_version: 1
  id: string
  name: string
  slug: string
  units: "mm"
  stock: Stock
  artwork: Artwork
  source: SourceImage | null
  depth?: DepthArtifact | null
  relief?: ReliefRecord | null
  created_at?: string | null
  updated_at?: string | null
}

export interface ProjectCreateRequest {
  name: string
  stock: Stock
  artwork: Artwork
}

export interface HeightPercentiles {
  p1: number
  p5: number
  p25: number
  p50: number
  p75: number
  p95: number
  p99: number
}

export interface PhysicalHeightMetrics {
  width_px: number
  height_px: number
  sample_count: number
  minimum_mm: number
  maximum_mm: number
  mean_mm: number
  median_mm: number
  standard_deviation_mm: number
  percentiles_mm: HeightPercentiles
  range_mm: number
  non_finite_count: number
  below_zero_count: number
  above_stock_count: number
}

export interface PhysicalHeightHistogram {
  bins: number
  edges_mm: number[]
  counts: number[]
  sample_count: number
}

export interface RecordedPhysicalMapping {
  stock_thickness_mm: number
  base_thickness_mm: number
  maximum_surface_height_mm: number
  maximum_theoretical_cut_depth_mm: number
}

export interface PhysicalHeightValidation {
  ok: boolean
  errors: string[]
  warnings: string[]
}

export interface PhysicalHeightDiagnostics {
  project_id: string
  status: ArtifactStatus
  width_px: number
  height_px: number
  metrics: PhysicalHeightMetrics | null
  histogram: PhysicalHeightHistogram | null
  validation: PhysicalHeightValidation
  mapping: RecordedPhysicalMapping
}

export interface ProfileSample {
  x_px: number
  y_px: number
  distance_mm: number
  height_mm: number | null
}

export interface ProfileLine {
  start_x_px: number
  start_y_px: number
  end_x_px: number
  end_y_px: number
}

export interface ProfileSummary {
  length_mm: number
  sample_count: number
  valid_count: number
  invalid_count: number
  minimum_mm: number | null
  maximum_mm: number | null
  range_mm: number | null
  mean_mm: number | null
  maximum_adjacent_delta_mm: number | null
  maximum_abs_slope_mm_per_mm: number | null
}

export type ProfileMode = "horizontal" | "vertical" | "line"

export interface PhysicalHeightProfile {
  project_id: string
  status: ArtifactStatus
  mode: ProfileMode
  coordinate_convention: string
  slope_note: string
  units: "mm"
  artwork_width_mm: number
  artwork_height_mm: number
  width_px: number
  height_px: number
  line: ProfileLine
  summary: ProfileSummary
  samples: ProfileSample[]
  warnings: string[]
}
