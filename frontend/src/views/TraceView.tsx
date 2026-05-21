import { useMemo, useRef, useEffect } from "react"
import type { UIEvent } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { TraceRow } from "@/components/TraceRow"
import { FilterToolbar } from "@/components/FilterToolbar"
import { LiveRouteGraph } from "@/components/LiveRouteGraph"
import type { UseHealthPollResult } from "@/hooks/useHealthPoll"
import { useTraceStore, type Filters } from "@/store/traceStore"
import { latencyBand, confidenceBand, rowTotalLatencyMs } from "@/lib/bands"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

function applyFilters(traces: TraceEvent[], filters: Filters): TraceEvent[] {
  return traces.filter((t) => {
    if (filters.pivoted === "yes" && !t.pivoted) return false
    if (filters.pivoted === "no" && t.pivoted) return false
    if (
      filters.pivot_reason !== "any" &&
      t.pivot_reason !== filters.pivot_reason
    )
      return false
    if (
      filters.model !== "any" &&
      t.selected_local_model !== filters.model &&
      t.frontier_model !== filters.model
    ) {
      return false
    }
    if (
      filters.latency_band !== "any" &&
      latencyBand(rowTotalLatencyMs(t)) !== filters.latency_band
    )
      return false
    if (
      filters.confidence_band !== "any" &&
      confidenceBand(t.confidence_score) !== filters.confidence_band
    )
      return false
    return true
  })
}

export function TraceView({
  healthResult,
}: {
  healthResult: UseHealthPollResult
}) {
  const traces = useTraceStore((s) => s.traces)
  const filters = useTraceStore((s) => s.filters)
  const autoTail = useTraceStore((s) => s.autoTail)
  const setAutoTail = useTraceStore((s) => s.setAutoTail)

  const filtered = useMemo(
    () => applyFilters(traces, filters),
    [traces, filters],
  )
  // Newest-last in buffer → reverse for newest-first display.
  const display = useMemo(() => [...filtered].reverse(), [filtered])

  const scrollRef = useRef<HTMLDivElement>(null)
  const newSinceFreezeRef = useRef(0)

  // D-18: auto-tail behavior — when user scrolls up, freeze; resume chip lets them re-engage.
  useEffect(() => {
    if (autoTail && scrollRef.current) {
      scrollRef.current.scrollTop = 0 // newest-first → top is the live edge
      newSinceFreezeRef.current = 0
    } else {
      newSinceFreezeRef.current += 1
    }
  }, [traces.length, autoTail])

  const handleScroll = (e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    // Newest-first list: at-top means at-live-edge; if user scrolled past 4px, freeze.
    if (el.scrollTop > 4 && autoTail) setAutoTail(false)
  }

  return (
    <div className="flex flex-col gap-2" data-testid="trace-view">
      <LiveRouteGraph healthResult={healthResult} />
      <FilterToolbar />
      {!autoTail && newSinceFreezeRef.current > 0 && (
        <Button
          variant="secondary"
          size="sm"
          className="self-center"
          onClick={() => {
            setAutoTail(true)
            scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" })
          }}
          data-testid="resume-tail"
        >
          {newSinceFreezeRef.current} new traces — click to resume
        </Button>
      )}
      <ScrollArea className="h-[70vh] rounded-md border">
        <div ref={scrollRef} onScroll={handleScroll} data-testid="trace-list">
          {display.length === 0 ? (
            <div className="text-muted-foreground p-6 text-center text-sm">
              <Badge variant="outline">no traces yet</Badge>
              <p className="mt-2">
                Send a request to <code>/v1/chat/completions</code> on the
                backend to see live traces here.
              </p>
            </div>
          ) : (
            display.map((t) => <TraceRow key={t.request_id} trace={t} />)
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
