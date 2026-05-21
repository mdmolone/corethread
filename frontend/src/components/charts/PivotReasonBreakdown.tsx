import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { useMemo } from "react"
import {
  chartTooltipContentStyle,
  chartTooltipItemStyle,
  chartTooltipLabelStyle,
} from "./tooltipStyles"

interface Props {
  pivot_reasons: { [key: string]: number } | undefined // undefined when snapshot is empty
}

const ORDERED_BUCKETS = [
  "none",
  "low_score",
  "local_truncated",
  "local_error",
  "judge_error",
] as const

export function PivotReasonBreakdown({ pivot_reasons }: Props) {
  // Phase 8 D-07: pivot_reasons is ALWAYS the 4-bucket dict (none, low_score,
  // local_error, judge_error) — quiet windows have all-zero values, NOT a
  // smaller dict. The chart's x-axis is stable across the lifecycle.
  const data = useMemo(() => {
    return ORDERED_BUCKETS.map((b) => ({
      reason: b,
      count: pivot_reasons?.[b] ?? 0,
    }))
  }, [pivot_reasons])

  return (
    <div className="h-64 w-full" data-testid="pivot-reason-breakdown">
      <ResponsiveContainer>
        <BarChart data={data}>
          <XAxis dataKey="reason" />
          <YAxis allowDecimals={false} />
          <Tooltip
            cursor={false}
            contentStyle={chartTooltipContentStyle}
            itemStyle={chartTooltipItemStyle}
            labelStyle={chartTooltipLabelStyle}
            wrapperStyle={{ outline: "none" }}
          />
          <Bar dataKey="count" fill="hsl(var(--primary))" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
