import { useState, useMemo, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { StatHero } from "@/components/StatHero"
import { ConfidenceHistogram } from "@/components/charts/ConfidenceHistogram"
import { PivotReasonBreakdown } from "@/components/charts/PivotReasonBreakdown"
import { ThresholdWhatIfSlider } from "@/components/charts/ThresholdWhatIfSlider"
import { CostSavedHero } from "@/components/charts/CostSavedHero"
import { usePolledStats, type StatsWindow } from "@/hooks/usePolledStats"
import { useTraceStore } from "@/store/traceStore"
import { useConfigStore } from "@/store/configStore"

const WINDOWS: { value: StatsWindow; label: string; seconds: number | null }[] =
  [
    { value: "1h", label: "1h", seconds: 60 * 60 },
    { value: "24h", label: "24h", seconds: 24 * 60 * 60 },
    { value: "7d", label: "7d", seconds: 7 * 24 * 60 * 60 },
    { value: "all", label: "all", seconds: null },
  ]

export function StatsView() {
  const [window, setWindow] = useState<StatsWindow>("1h")
  const { snapshot, status } = usePolledStats(window)

  const traces = useTraceStore((s) => s.traces)
  const configuredThreshold = useConfigStore(
    (s) => s.config?.routing.threshold ?? 0.7,
  )

  // Initialize slider to the configured threshold (D-22 — read once on mount).
  const [sliderValue, setSliderValue] = useState<number>(configuredThreshold)
  // If config wasn't ready at first render, sync once when it arrives.
  useEffect(() => {
    setSliderValue(configuredThreshold)
  }, [configuredThreshold])

  // D-19 NOTE: client-derived charts (ThresholdWhatIfSlider + CostSavedHero)
  // would ideally read a TIME-WINDOWED slice of the buffer per `now -
  // window_seconds <= trace.timestamp`. TraceEvent has NO `timestamp` field
  // (Phase 9 TRC-09 lock — only the 15 documented fields). For v1.1 we use
  // Option A: treat all client-derived computations as buffer-wide ("all"
  // semantics) regardless of window selector value. Phase 12+ adds
  // `_client_received_at: Date.now()` in traceStore.addTrace and switches to
  // true window-filtering. The `seconds` field on WINDOWS above is reserved
  // for the v1.2 implementation.
  const windowedTraces = useMemo(() => traces, [traces])

  // DEF-10-01 #3 / Phase 11 D-02: extract a discriminated-union narrowed local
  // so `populated.pivot_reasons` access type-checks. StatsSnapshotEmpty has
  // `window_size: 0` (literal); StatsSnapshotPopulated has `window_size: number`
  // (which includes 0), so `window_size === 0` is NOT a clean discriminator.
  // The `pivot_reasons` key only appears on StatsSnapshotPopulated, so the
  // `in`-operator type guard narrows correctly. Extract once at the top of the
  // render scope and pass the narrowed local to consumers — narrowing at the
  // access site (in JSX prop expressions) doesn't propagate reliably.
  const populated = snapshot && "pivot_reasons" in snapshot ? snapshot : null
  // populated is now StatsSnapshotPopulated | null

  return (
    <div className="space-y-4" data-testid="stats-view">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground text-sm">Window:</span>
        {WINDOWS.map((w) => (
          <Button
            key={w.value}
            size="sm"
            variant={window === w.value ? "default" : "outline"}
            onClick={() => setWindow(w.value)}
            data-testid={`window-${w.value}`}
          >
            {w.label}
          </Button>
        ))}
        <Badge variant="outline" className="ml-auto" data-testid="poll-status">
          poll: {status}
        </Badge>
      </div>

      {snapshot ? (
        <StatHero snapshot={snapshot} />
      ) : (
        <div className="text-muted-foreground text-sm">Loading…</div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div data-testid="chart-df1-confidence-histogram">
          <h3 className="mb-2 text-sm font-medium">
            Confidence histogram (DF-1)
          </h3>
          <ConfidenceHistogram
            traces={windowedTraces}
            configuredThreshold={configuredThreshold}
            sliderValue={sliderValue}
          />
        </div>
        <div data-testid="chart-df3-pivot-reason-breakdown">
          <h3 className="mb-2 text-sm font-medium">Pivot reasons (DF-3)</h3>
          <PivotReasonBreakdown pivot_reasons={populated?.pivot_reasons} />
        </div>
        <div data-testid="chart-df2-threshold-what-if">
          <ThresholdWhatIfSlider
            traces={windowedTraces}
            configuredThreshold={configuredThreshold}
            sliderValue={sliderValue}
            onSliderChange={setSliderValue}
          />
        </div>
        <div data-testid="chart-df4-cost-saved">
          <CostSavedHero traces={windowedTraces} />
        </div>
      </div>
    </div>
  )
}
