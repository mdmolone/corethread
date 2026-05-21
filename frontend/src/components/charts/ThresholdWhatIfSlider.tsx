import { useMemo } from "react"
import { Card, CardContent } from "@/components/ui/card"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

interface Props {
  traces: TraceEvent[] // time-windowed buffer slice from useTraceStore
  configuredThreshold: number
  sliderValue: number
  onSliderChange: (v: number) => void
}

/**
 * D-22: slider range [0.0, 1.0] step 0.01; default = configured threshold.
 *
 * Pure-client predicate (D-19 STA-05 lock):
 *   Would-have-pivoted at threshold X iff:
 *     pivot_reason !== "none"  // already pivoted for non-low-score reason; pivots regardless
 *     OR confidence_score < X   // would have pivoted at the slider's threshold
 *
 * Note: the original D-22 predicate `confidence_score < threshold || pivot_reason !== "none"`
 * is what we implement verbatim — the OR is union semantics not exclusion.
 */
export function ThresholdWhatIfSlider({
  traces,
  configuredThreshold,
  sliderValue,
  onSliderChange,
}: Props) {
  const { wouldPivot, total, pct } = useMemo(() => {
    const total = traces.length
    if (total === 0) return { wouldPivot: 0, total: 0, pct: 0 }
    const wouldPivot = traces.filter(
      (t) => t.confidence_score < sliderValue || t.pivot_reason !== "none",
    ).length
    return { wouldPivot, total, pct: (wouldPivot / total) * 100 }
  }, [traces, sliderValue])

  return (
    <Card data-testid="threshold-what-if-slider">
      <CardContent className="space-y-3 p-4">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground text-sm">
            Threshold what-if (DF-2 / STA-05)
          </span>
          <span className="text-muted-foreground text-xs">
            configured: {configuredThreshold.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={sliderValue}
          onChange={(e) => onSliderChange(parseFloat(e.target.value))}
          className="w-full"
          data-testid="threshold-slider"
          aria-label="threshold what-if slider"
        />
        <div className="text-sm tabular-nums" data-testid="threshold-readout">
          At threshold {sliderValue.toFixed(2)}, {wouldPivot}/{total} would have
          pivoted ({pct.toFixed(1)}%)
        </div>
      </CardContent>
    </Card>
  )
}
