export type PreferredDevice = "mps" | "cpu"

export interface ComputeInfo {
  preferred_device: PreferredDevice
  torch_installed: boolean
  mps_available: boolean
}

export interface HealthResponse {
  status: "ok"
  application: string
  api_version: string
  compute: ComputeInfo
}
