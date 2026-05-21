import { describe, it, expect, beforeEach } from "vitest"
import { useTraceStore, BUFFER_CAP, type TraceEvent } from "./traceStore"

const makeTrace = (
  request_id: string,
  overrides: Partial<TraceEvent> = {},
): TraceEvent => ({
  request_id,
  selected_local_model: "test-model",
  judge_model: "judge-model",
  frontier_model: null,
  confidence_score: 0.9,
  pivoted: false,
  local_latency_ms: 100,
  judge_latency_ms: 50,
  frontier_latency_ms: null,
  input_tokens: 10,
  output_tokens: 20,
  frontier_cost_est: null,
  judge_parse_failed: false,
  pivot_reason: "none",
  local_error_class: null,
  ...overrides,
})

describe("traceStore (Phase 10 / Plan 10-02)", () => {
  beforeEach(() => {
    useTraceStore.setState({
      traces: [],
      expandedRequestId: null,
      autoTail: true,
      filters: {
        pivoted: "any",
        model: "any",
        pivot_reason: "any",
        latency_band: "any",
        confidence_band: "any",
      },
    })
  })

  it("addTrace appends to the buffer", () => {
    useTraceStore.getState().addTrace(makeTrace("req-1"))
    useTraceStore.getState().addTrace(makeTrace("req-2"))
    expect(useTraceStore.getState().traces.map((t) => t.request_id)).toEqual([
      "req-1",
      "req-2",
    ])
  })

  it("addTrace caps buffer at BUFFER_CAP (1000) and drops oldest (D-16)", () => {
    expect(BUFFER_CAP).toBe(1000)
    for (let i = 0; i < BUFFER_CAP + 5; i++) {
      useTraceStore.getState().addTrace(makeTrace(`req-${i}`))
    }
    const buf = useTraceStore.getState().traces
    expect(buf.length).toBe(BUFFER_CAP)
    // oldest 5 should have been evicted; buffer starts at req-5
    expect(buf[0]?.request_id).toBe("req-5")
    expect(buf[buf.length - 1]?.request_id).toBe(`req-${BUFFER_CAP + 4}`)
  })

  it("addTrace dedupes by request_id (Pitfall 26 — Last-Event-ID replay overlap)", () => {
    useTraceStore.getState().addTrace(makeTrace("req-1"))
    useTraceStore.getState().addTrace(makeTrace("req-1")) // duplicate
    expect(useTraceStore.getState().traces.length).toBe(1)
  })

  it("setExpanded toggles single-row-at-a-time (D-11)", () => {
    const s = useTraceStore.getState()
    s.setExpanded("req-A")
    expect(useTraceStore.getState().expandedRequestId).toBe("req-A")
    s.setExpanded("req-B") // different row -> switches expansion
    expect(useTraceStore.getState().expandedRequestId).toBe("req-B")
    s.setExpanded("req-B") // same row -> collapses
    expect(useTraceStore.getState().expandedRequestId).toBeNull()
  })

  it("setFilter updates a single dimension without affecting others", () => {
    useTraceStore.getState().setFilter("pivoted", "yes")
    useTraceStore.getState().setFilter("pivot_reason", "low_score")
    expect(useTraceStore.getState().filters.pivoted).toBe("yes")
    expect(useTraceStore.getState().filters.pivot_reason).toBe("low_score")
    expect(useTraceStore.getState().filters.model).toBe("any") // untouched
  })

  it("resetFilters returns all 5 dimensions to 'any'", () => {
    useTraceStore.getState().setFilter("pivoted", "yes")
    useTraceStore.getState().setFilter("model", "gpt-oss:20b")
    useTraceStore.getState().resetFilters()
    expect(useTraceStore.getState().filters).toEqual({
      pivoted: "any",
      model: "any",
      pivot_reason: "any",
      latency_band: "any",
      confidence_band: "any",
    })
  })

  it("clearTraces empties the buffer and collapses any expanded row", () => {
    useTraceStore.getState().addTrace(makeTrace("req-1"))
    useTraceStore.getState().setExpanded("req-1")
    useTraceStore.getState().clearTraces()
    expect(useTraceStore.getState().traces.length).toBe(0)
    expect(useTraceStore.getState().expandedRequestId).toBeNull()
  })

  it("setAutoTail toggles the auto-tail flag (D-18)", () => {
    expect(useTraceStore.getState().autoTail).toBe(true)
    useTraceStore.getState().setAutoTail(false)
    expect(useTraceStore.getState().autoTail).toBe(false)
  })
})
