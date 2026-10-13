import { createContext, useContext, useMemo, useState, type ReactNode } from "react"
import {
  readUnitPreference,
  writeUnitPreference,
  type DisplayUnit,
} from "../utils/units.ts"

type UnitsContextValue = {
  unit: DisplayUnit
  setUnit: (unit: DisplayUnit) => void
}

const UnitsContext = createContext<UnitsContextValue | null>(null)

export function UnitsProvider({ children }: { children: ReactNode }) {
  const [unit, setUnitState] = useState<DisplayUnit>(readUnitPreference)

  const value = useMemo<UnitsContextValue>(
    () => ({
      unit,
      setUnit(next) {
        writeUnitPreference(next)
        setUnitState(next)
      },
    }),
    [unit],
  )

  return <UnitsContext.Provider value={value}>{children}</UnitsContext.Provider>
}

export function useUnits(): UnitsContextValue {
  const value = useContext(UnitsContext)
  if (!value) {
    throw new Error("useUnits must be used inside UnitsProvider.")
  }
  return value
}
