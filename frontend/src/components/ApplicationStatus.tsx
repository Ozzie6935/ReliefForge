import type { HealthResponse } from "../types/health.ts"
import {
  settingsApiVersion,
  settingsAvailableLabel,
  settingsConnectionLabel,
  settingsDeviceLabel,
  settingsInstalledLabel,
  type SettingsConnection,
} from "../utils/settingsPresentation.ts"

export type BackendConnection = SettingsConnection

type ApplicationStatusProps = {
  connection: BackendConnection
  health: HealthResponse | null
  errorMessage: string | null
}

function connectionClass(connection: BackendConnection): string {
  if (connection === "connected") return "value-ok"
  if (connection === "disconnected") return "value-bad"
  return "value-wait"
}

export function ApplicationStatus({
  connection,
  health,
  errorMessage,
}: ApplicationStatusProps) {
  return (
    <section className="status-card" aria-labelledby="system-heading">
      <h2 id="system-heading" className="section-title">
        System Information
      </h2>
      <dl>
        <div className="status-row">
          <dt>Backend</dt>
          <dd className={connectionClass(connection)}>{settingsConnectionLabel(connection)}</dd>
        </div>
        <div className="status-row">
          <dt>API Version</dt>
          <dd>{settingsApiVersion(health, connection)}</dd>
        </div>
        <div className="status-row">
          <dt>Compute Device</dt>
          <dd>{settingsDeviceLabel(health, connection)}</dd>
        </div>
        <div className="status-row">
          <dt>PyTorch</dt>
          <dd>{settingsInstalledLabel(health?.compute.torch_installed, connection)}</dd>
        </div>
        <div className="status-row">
          <dt>Apple MPS</dt>
          <dd>{settingsAvailableLabel(health?.compute.mps_available, connection)}</dd>
        </div>
      </dl>
      {connection === "disconnected" && errorMessage ? (
        <p className="status-notice">{errorMessage}</p>
      ) : null}
    </section>
  )
}
