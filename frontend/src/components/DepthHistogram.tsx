import type { HistogramSeries } from "../types/project.ts"

type DepthHistogramChartProps = {
  title: string
  note: string
  series: HistogramSeries | null
  empty: string
}

export function DepthHistogramChart({ title, note, series, empty }: DepthHistogramChartProps) {
  const width = 320
  const height = 88
  const counts = series?.counts ?? []
  const peak = Math.max(1, ...counts)
  const bar = counts.length > 0 ? width / counts.length : width

  return (
    <figure className="depth-histogram">
      <figcaption>
        {title}
        <span>{note}</span>
      </figcaption>
      {series ? (
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
          {counts.map((count, index) => {
            const barHeight = (count / peak) * (height - 2)
            return (
              <rect
                key={index}
                x={index * bar}
                y={height - barHeight}
                width={Math.max(bar, 0.5)}
                height={barHeight}
                fill="#5c6775"
              />
            )
          })}
        </svg>
      ) : (
        <p className="field-note">{empty}</p>
      )}
    </figure>
  )
}
