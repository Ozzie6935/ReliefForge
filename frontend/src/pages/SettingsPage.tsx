import { ApplicationStatus } from "../components/ApplicationStatus.tsx"
import { useHealth } from "../health/HealthContext.tsx"
import { useUnits } from "../units/UnitsContext.tsx"
import { SETTINGS_STOCK_NOTE, SETTINGS_UNIT_NOTE, SETTINGS_UNIT_OPTIONS } from "../utils/settingsPresentation.ts"
import type { DisplayUnit } from "../utils/units.ts"

export function SettingsPage() {
  const { unit, setUnit } = useUnits()
  const { connection, health, errorMessage } = useHealth()

  return (
    <section className="settings-page" aria-labelledby="settings-heading">
      <h1 id="settings-heading">Settings</h1>
      <p className="subtitle">Application preferences for this browser.</p>
      <section className="status-card" aria-labelledby="units-heading">
        <h2 id="units-heading" className="section-title">
          Display Units
        </h2>
        <p className="field-note">{SETTINGS_UNIT_NOTE}</p>
        <div className="unit-choice" role="radiogroup" aria-label="Display Units">
          {SETTINGS_UNIT_OPTIONS.map((option) => (
            <UnitOption
              key={option.value}
              label={option.label}
              value={option.value}
              current={unit}
              onSelect={setUnit}
            />
          ))}
        </div>
      </section>
      <section className="status-card" aria-labelledby="stock-default-heading">
        <h2 id="stock-default-heading" className="section-title">
          Stock Thickness
        </h2>
        <p className="field-note">{SETTINGS_STOCK_NOTE}</p>
      </section>
      <ApplicationStatus connection={connection} health={health} errorMessage={errorMessage} />
    </section>
  )
}

function UnitOption({
  label,
  value,
  current,
  onSelect,
}: {
  label: string
  value: DisplayUnit
  current: DisplayUnit
  onSelect: (unit: DisplayUnit) => void
}) {
  const selected = current === value
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      className={selected ? "unit-option selected" : "unit-option"}
      onClick={() => onSelect(value)}
    >
      {label}
    </button>
  )
}
