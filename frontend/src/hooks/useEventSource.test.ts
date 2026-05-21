import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { renderHook, act } from "@testing-library/react"
import { useTraceStore } from "@/store/traceStore"

// ---------------------------------------------------------------------------
// FakeEventSource: replaces the `eventsource` npm polyfill at the test boundary.
// Captures the wrapped-fetch headers at construction so we can assert that
// Last-Event-ID is sent on (re)connect (D-07 binding contract).
//
// Rule-3 deviation per the plan's "Anticipated Rule 3 deviations": vi.mock is
// hoisted to top-of-file by Vitest, so we use vi.hoisted() to define the
// FakeEventSource class BEFORE the hoisted vi.mock factory references it.
// ---------------------------------------------------------------------------

const { FakeEventSource } = vi.hoisted(() => {
  type Listener = (ev: unknown) => void

  class FakeEventSource {
    static instances: FakeEventSource[] = []
    static lastProbeHeaders: Record<string, string> = {}
    url: string
    capturedHeaders: Record<string, string>
    onopen: Listener | null = null
    onmessage: Listener | null = null
    onerror: Listener | null = null
    listeners: Record<string, Listener[]> = {}
    closed = false

    constructor(
      url: string | URL,
      init?: {
        fetch?: (
          url: string | URL,
          init?: { headers?: Record<string, string> },
        ) => Promise<unknown>
      },
    ) {
      this.url = String(url)
      this.capturedHeaders = {}
      // The hook's polyfill init wraps fetch and merges Last-Event-ID into
      // headers. Probe by invoking the wrapped fetch with an empty init; the
      // wrapper will call our global fetch mock with the merged headers.
      if (init?.fetch) {
        try {
          void init.fetch(url, { headers: {} })
          this.capturedHeaders = { ...FakeEventSource.lastProbeHeaders }
        } catch {
          // ignore
        }
      }
      FakeEventSource.instances.push(this)
    }

    close() {
      this.closed = true
    }

    addEventListener(type: string, listener: Listener) {
      this.listeners[type] = [...(this.listeners[type] ?? []), listener]
    }

    removeEventListener(type: string, listener: Listener) {
      this.listeners[type] = (this.listeners[type] ?? []).filter(
        (existing) => existing !== listener,
      )
    }

    // Test helper to fire a message with a Last-Event-ID.
    emit(data: string, id: string) {
      const ev = { data, lastEventId: id }
      this.onmessage?.(ev)
      for (const listener of this.listeners.message ?? []) listener(ev)
    }

    // Test helper for backend named SSE events: `event: trace`.
    emitTrace(data: string, id: string) {
      const ev = { data, lastEventId: id }
      for (const listener of this.listeners.trace ?? []) listener(ev)
    }

    triggerError() {
      this.onerror?.({})
    }
  }

  return { FakeEventSource }
})

vi.mock("eventsource", () => ({
  // Match `import { EventSource } from "eventsource"`.
  EventSource: FakeEventSource,
}))

// Import AFTER vi.mock is registered so the hook's `EventSourcePolyfill` is
// the FakeEventSource. `import` statements are hoisted by ESM but vi.mock is
// hoisted higher by Vitest.
import { useEventSource, _testing } from "./useEventSource"

describe("useEventSource — backoff math (D-06)", () => {
  it("exp backoff: 500 * 2^N capped at 30_000ms with ±10% jitter", () => {
    // For each retry count, the unjittered base is min(30000, 500 * 2^N).
    // Jittered range is [0.9 * base, 1.1 * base].
    const cases = [
      { retries: 0, base: 500 },
      { retries: 1, base: 1000 },
      { retries: 2, base: 2000 },
      { retries: 3, base: 4000 },
      { retries: 4, base: 8000 },
      { retries: 5, base: 16000 },
      { retries: 6, base: 30000 }, // capped (32000 > 30000)
      { retries: 10, base: 30000 }, // still capped
    ]
    for (const { retries, base } of cases) {
      const samples = Array.from({ length: 200 }, () =>
        _testing.backoffDelayMs(retries),
      )
      const min = Math.min(...samples)
      const max = Math.max(...samples)
      // ±10% bounds with comfortable margin (jitter is 0.9..1.1 * base by design).
      expect(min).toBeGreaterThanOrEqual(base * 0.9)
      expect(max).toBeLessThanOrEqual(base * 1.1)
    }
  })

  it("retry count 0 produces 500ms ± 10% (the first reconnect)", () => {
    // Force jitter to its midpoint with a deterministic Math.random.
    const spy = vi.spyOn(Math, "random").mockReturnValue(0.5)
    expect(_testing.backoffDelayMs(0)).toBeCloseTo(500 * 1.0, 5) // 0.9 + 0.5*0.2 = 1.0
    spy.mockRestore()
  })
})

// ---------------------------------------------------------------------------
// Reconnect + Last-Event-ID test (D-07 binding contract)
// ---------------------------------------------------------------------------

describe("useEventSource — Last-Event-ID reconnect (D-07 / TRC-05)", () => {
  beforeEach(() => {
    FakeEventSource.instances = []
    FakeEventSource.lastProbeHeaders = {}
    useTraceStore.setState({
      traces: [],
      expandedRequestId: null,
      autoTail: true,
    })

    // Global fetch mock that records the headers passed (the hook's wrapped
    // fetch merges Last-Event-ID into init.headers; we capture by reading them
    // off the second argument).
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: unknown, init?: RequestInit) => {
        const h: Record<string, string> = {}
        const incoming = (init?.headers ?? {}) as Record<string, string>
        for (const [k, v] of Object.entries(incoming)) h[k] = v
        FakeEventSource.lastProbeHeaders = h
        return new Response("", { status: 200 })
      }),
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("first connect sends NO Last-Event-ID header", () => {
    renderHook(() => useEventSource("/v1/traces/stream"))
    // The first FakeEventSource was constructed; its capturedHeaders should
    // not contain Last-Event-ID (no events seen yet).
    expect(FakeEventSource.instances.length).toBeGreaterThanOrEqual(1)
    const first = FakeEventSource.instances[0]
    if (!first) throw new Error("FakeEventSource instance not created")
    expect(first.capturedHeaders["Last-Event-ID"]).toBeUndefined()
  })

  it("stores backend named trace events", () => {
    renderHook(() => useEventSource("/v1/traces/stream"))
    expect(FakeEventSource.instances.length).toBe(1)
    const first = FakeEventSource.instances[0]
    if (!first) throw new Error("FakeEventSource instance not created")

    const t = {
      request_id: "req-trace",
      selected_local_model: "m",
      judge_model: "j",
      frontier_model: null,
      confidence_score: 0.9,
      pivoted: false,
      local_latency_ms: 10,
      judge_latency_ms: 5,
      frontier_latency_ms: null,
      input_tokens: 1,
      output_tokens: 2,
      frontier_cost_est: null,
      judge_parse_failed: false,
      pivot_reason: "none" as const,
      local_error_class: null,
    }

    act(() => first.emitTrace(JSON.stringify(t), "req-trace"))

    expect(useTraceStore.getState().traces[0]?.request_id).toBe("req-trace")
  })

  it("publishes 5 events then reconnects on error with Last-Event-ID = 5th event id", async () => {
    renderHook(() => useEventSource("/v1/traces/stream"))
    expect(FakeEventSource.instances.length).toBe(1)
    const first = FakeEventSource.instances[0]
    if (!first) throw new Error("FakeEventSource instance not created")

    // Emit 5 events, each with an id; useEventSource writes lastEventIdRef on each.
    for (let i = 1; i <= 5; i++) {
      const t = {
        request_id: `req-${i}`,
        selected_local_model: "m",
        judge_model: "j",
        frontier_model: null,
        confidence_score: 0.9,
        pivoted: false,
        local_latency_ms: 10,
        judge_latency_ms: 5,
        frontier_latency_ms: null,
        input_tokens: 1,
        output_tokens: 2,
        frontier_cost_est: null,
        judge_parse_failed: false,
        pivot_reason: "none" as const,
        local_error_class: null,
      }
      act(() => first.emit(JSON.stringify(t), `req-${i}`))
    }

    expect(useTraceStore.getState().traces.length).toBe(5)

    // Trigger reconnect via onerror; backoff fires after a delay — we use fake timers.
    vi.useFakeTimers()
    act(() => first.triggerError())
    // Advance timers past the maximum first-retry jitter window (500 * 1.1 = 550ms).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    vi.useRealTimers()

    // A new FakeEventSource instance must have been constructed.
    expect(FakeEventSource.instances.length).toBe(2)
    const second = FakeEventSource.instances[1]
    if (!second) throw new Error("Reconnect FakeEventSource not created")
    // The reconnect must carry Last-Event-ID = 'req-5'.
    expect(second.capturedHeaders["Last-Event-ID"]).toBe("req-5")
  })
})
