import { create } from "zustand"
import type { components } from "@/api/types"

// Phase 9 wire-shape alias — never re-declare TraceEvent (D-32).
export type TraceEvent = components["schemas"]["TraceEvent"]

// D-16: ring buffer cap mirrors Phase 8 backend deque maxlen=1000.
export const BUFFER_CAP = 1000

// D-13: five-dimension filter selections — in-memory only (D-17 — no URL / localStorage).
export type LatencyBand = "lt200" | "200_500" | "500_1000" | "gt1000" | "any"
export type ConfidenceBand = "lt03" | "03_07" | "gt07" | "any"
export type PivotedFilter = "yes" | "no" | "any"
export type PivotReasonFilter =
  | "none"
  | "low_score"
  | "local_truncated"
  | "local_error"
  | "judge_error"
  | "any"

export interface Filters {
  pivoted: PivotedFilter
  model: string | "any"
  pivot_reason: PivotReasonFilter
  latency_band: LatencyBand
  confidence_band: ConfidenceBand
}

const DEFAULT_FILTERS: Filters = {
  pivoted: "any",
  model: "any",
  pivot_reason: "any",
  latency_band: "any",
  confidence_band: "any",
}

interface TraceState {
  traces: TraceEvent[]
  expandedRequestId: string | null
  filters: Filters
  autoTail: boolean
}

interface TraceActions {
  addTrace: (t: TraceEvent) => void
  clearTraces: () => void
  setExpanded: (request_id: string | null) => void
  setFilter: <K extends keyof Filters>(key: K, value: Filters[K]) => void
  resetFilters: () => void
  setAutoTail: (enabled: boolean) => void
}

export const useTraceStore = create<TraceState & TraceActions>()((set) => ({
  traces: [],
  expandedRequestId: null,
  filters: { ...DEFAULT_FILTERS },
  autoTail: true,

  addTrace: (t) =>
    set((s) => {
      // Pitfall 26 dedupe — drop a TraceEvent whose request_id is already in the buffer.
      // Phase 9 D-11 (unknown Last-Event-ID falls through to full replay) means a reconnect
      // can replay events the client already has; dedup by request_id keeps the buffer clean.
      if (s.traces.some((x) => x.request_id === t.request_id)) return s
      const next =
        s.traces.length >= BUFFER_CAP
          ? [...s.traces.slice(1), t] // drop-oldest at cap (D-16)
          : [...s.traces, t]
      return { traces: next }
    }),

  clearTraces: () => set({ traces: [], expandedRequestId: null }),

  setExpanded: (request_id) =>
    set((s) => ({
      // D-11: single-row-expanded-at-a-time — clicking expanded row collapses it.
      expandedRequestId: s.expandedRequestId === request_id ? null : request_id,
    })),

  setFilter: (key, value) =>
    set((s) => ({ filters: { ...s.filters, [key]: value } })),

  resetFilters: () => set({ filters: { ...DEFAULT_FILTERS } }),

  setAutoTail: (enabled) => set({ autoTail: enabled }),
}))
