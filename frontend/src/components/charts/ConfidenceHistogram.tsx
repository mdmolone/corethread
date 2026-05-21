import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from "recharts"
import { useMemo } from "react"
import type { components } from "@/api/types"
import {
  chartTooltipContentStyle,
  chartTooltipItemStyle,
  chartTooltipLabelStyle,
} from "./tooltipStyles"

type TraceEvent = components["schemas"]["TraceEvent"]

const BIN_COUNT = 10 // D-23: fixed 10 bins of width 0.1 over [0, 1]

interface Props {
  traces: TraceEvent[] // from useTraceStore (client-side bin source)
  configuredThreshold: number
  sliderValue: number
}

function bin(score: number): number {
  // Maps a score in [0, 1] to a bin 0..9 (clamping any rare 1.0 case to bin 9).
  const idx = Math.min(BIN_COUNT - 1, Math.floor(score * BIN_COUNT))
  return idx
}

function binLabel(thresholdValue: number): string {
  const i = Math.min(BIN_COUNT - 1, Math.floor(thresholdValue * BIN_COUNT))
  return `${(i * 0.1).toFixed(1)}-${((i + 1) * 0.1).toFixed(1)}`
}

export function ConfidenceHistogram({
  traces,
  configuredThreshold,
  sliderValue,
}: Props) {
  const data = useMemo(() => {
    const counts = Array.from({ length: BIN_COUNT }, () => 0)
    for (const t of traces) {
      const i = bin(t.confidence_score)
      counts[i] = (counts[i] ?? 0) + 1
    }
    return counts.map((c, i) => ({
      bin: `${(i * 0.1).toFixed(1)}-${((i + 1) * 0.1).toFixed(1)}`,
      bin_center: i * 0.1 + 0.05,
      count: c,
    }))
  }, [traces])

  return (
    <div className="h-64 w-full" data-testid="confidence-histogram">
      <ResponsiveContainer>
        <BarChart data={data}>
          <XAxis dataKey="bin" />
          <YAxis allowDecimals={false} />
          <Tooltip
            cursor={false}
            contentStyle={chartTooltipContentStyle}
            itemStyle={chartTooltipItemStyle}
            labelStyle={chartTooltipLabelStyle}
            wrapperStyle={{ outline: "none" }}
          />
          <Bar dataKey="count" fill="hsl(var(--primary))" />
          <ReferenceLine
            x={binLabel(configuredThreshold)}
            stroke="hsl(var(--ring))"
            strokeDasharray="3 3"
            label={{
              value: `cfg ${configuredThreshold.toFixed(2)}`,
              position: "top",
            }}
          />
          <ReferenceLine
            x={binLabel(sliderValue)}
            stroke="hsl(var(--destructive))"
            strokeDasharray="3 3"
            label={{
              value: `slider ${sliderValue.toFixed(2)}`,
              position: "insideTop",
            }}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// Exported for tests
export const _testing = { bin, BIN_COUNT }
