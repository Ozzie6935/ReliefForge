import type { ReactNode } from "react"
import { NavLink, useLocation } from "react-router"
import { useHealth } from "../health/HealthContext.tsx"

function projectsLinkActive(pathname: string): boolean {
  if (pathname === "/projects") return true
  return pathname.startsWith("/projects/") && !pathname.startsWith("/projects/new")
}

export function AppShell({ children }: { children: ReactNode }) {
  const { connection } = useHealth()
  const location = useLocation()
  const connectionLabel =
    connection === "connected"
      ? "Connected"
      : connection === "disconnected"
        ? "Disconnected"
        : "Checking"

  return (
    <div className="shell">
      <header className="navbar">
        <NavLink className="brand" to="/projects">
          <img
            className="brand-mark"
            src="/favicon.svg"
            alt=""
            width={33}
            height={32}
          />
          <span className="brand-copy">
            <strong>Relief Forge</strong>
            <span>CNC Relief Studio</span>
          </span>
        </NavLink>
        <nav className="nav-links" aria-label="Application">
          <NavLink
            className={() =>
              projectsLinkActive(location.pathname) ? "nav-link active" : "nav-link"
            }
            to="/projects"
          >
            Projects
          </NavLink>
          <NavLink
            className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            to="/projects/new"
          >
            New Project
          </NavLink>
          <NavLink
            className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            to="/settings"
          >
            Settings
          </NavLink>
        </nav>
        <NavLink className={`backend-pill ${connection}`} to="/settings">
          Backend
          <span className="status-dot" aria-hidden="true" />
          {connectionLabel}
        </NavLink>
      </header>
      <main className="app-shell">
        <div className="panel">{children}</div>
      </main>
    </div>
  )
}
