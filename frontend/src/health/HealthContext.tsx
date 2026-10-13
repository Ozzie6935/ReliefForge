import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import { getHealth } from "../api/health.ts"
import type { BackendConnection } from "../components/ApplicationStatus.tsx"
import type { HealthResponse } from "../types/health.ts"

type HealthContextValue = {
  connection: BackendConnection
  health: HealthResponse | null
  errorMessage: string | null
}

const HealthContext = createContext<HealthContextValue | null>(null)

export function HealthProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<BackendConnection>("checking")
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    getHealth()
      .then((data) => {
        if (cancelled) return
        setHealth(data)
        setErrorMessage(null)
        setConnection("connected")
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setHealth(null)
        setConnection("disconnected")
        setErrorMessage(error instanceof Error ? error.message : "Backend unavailable.")
      })

    return () => {
      cancelled = true
    }
  }, [])

  const value = useMemo(
    () => ({ connection, health, errorMessage }),
    [connection, health, errorMessage],
  )

  return <HealthContext.Provider value={value}>{children}</HealthContext.Provider>
}

export function useHealth(): HealthContextValue {
  const value = useContext(HealthContext)
  if (!value) {
    throw new Error("useHealth must be used inside HealthProvider.")
  }
  return value
}
