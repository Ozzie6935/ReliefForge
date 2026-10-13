import { BrowserRouter, Navigate, Route, Routes } from "react-router"
import { AppShell } from "./components/AppShell.tsx"
import { HealthProvider } from "./health/HealthContext.tsx"
import { CreateProjectPage } from "./pages/CreateProjectPage.tsx"
import { EditProjectPage } from "./pages/EditProjectPage.tsx"
import { ProjectsPage } from "./pages/ProjectsPage.tsx"
import { SettingsPage } from "./pages/SettingsPage.tsx"
import { WorkspacePage } from "./pages/WorkspacePage.tsx"
import { UnitsProvider } from "./units/UnitsContext.tsx"

export default function App() {
  return (
    <BrowserRouter>
      <HealthProvider>
        <UnitsProvider>
          <AppShell>
            <Routes>
              <Route path="/" element={<Navigate to="/projects" replace />} />
              <Route path="/projects" element={<ProjectsPage />} />
              <Route path="/projects/new" element={<CreateProjectPage />} />
              <Route path="/projects/:projectId/edit" element={<EditProjectPage />} />
              <Route path="/projects/:projectId" element={<WorkspacePage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </AppShell>
        </UnitsProvider>
      </HealthProvider>
    </BrowserRouter>
  )
}
