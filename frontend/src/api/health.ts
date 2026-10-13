import { apiGet } from "./client.ts"
import type { HealthResponse } from "../types/health.ts"

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/api/health")
}
