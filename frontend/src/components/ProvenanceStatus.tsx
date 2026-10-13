import type { ArtifactStatus } from "../types/project.ts"
import { recoveryGuidance, statusExplanation, statusLabel } from "../utils/provenanceStatus.ts"

export function ProvenanceStatus({
  title,
  status,
  stage,
}: {
  title: string
  status: ArtifactStatus
  stage?: "raw" | "conditioned" | "relief" | "physical"
}) {
  const guidance = stage ? recoveryGuidance(stage, status) : null
  return (
    <div className={`provenance-status is-${status}`} role="status">
      <span className="provenance-label">{title}</span>
      <span className="provenance-state">{statusLabel(status)}</span>
      <span className="provenance-detail">{statusExplanation(status)}</span>
      {guidance ? <span className="provenance-guidance">{guidance}</span> : null}
    </div>
  )
}
