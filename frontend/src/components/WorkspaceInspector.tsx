import type { ReactNode } from "react"

/** Full-height inspector: scrolling body and a pinned footer. */
export function WorkspaceInspector({
  className,
  label,
  children,
  footer,
}: {
  className: string
  label: string
  children: ReactNode
  footer: ReactNode
}) {
  return (
    <aside className={className} aria-label={label}>
      <div className="inspector-body">{children}</div>
      <div className="inspector-actions">{footer}</div>
    </aside>
  )
}
