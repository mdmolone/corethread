import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { renderHook, act } from "@testing-library/react"
import { usePolledStats, type StatsWindow } from "./usePolledStats"

const FAKE_EMPTY = {
  window_started_at: 1_700_000_000,
  window_size: 0 as const,
  max_window_size: 1000,
  note: "Stats reset on CoreThread restart (in-memory).",
}

const FAKE_POPULATED = {
  window_started_at: 1_700_000_000,
  window_size: 12,
  max_window_size: 1000,
  pivot_rate: 0.25,
  pivot_reasons: {
    none: 9,
    low_score: 3,
    local_truncated: 0,
    local_error: 0,
    judge_error: 0,
  },
  tokens_in_total: 1500,
  tokens_out_total: 4500,
  local_latency_ms: { p50: 100, p95: 200, max: 250, min: 50 },
  judge_latency_ms: { p50: 30, p95: 60, max: 80, min: 15 },
  frontier_latency_ms: null,
  confidence_score: { p50: 0.85, p95: 0.95, max: 0.99, min: 0.42 },
  note: "Stats reset on CoreThread restart (in-memory).",
}

describe("usePolledStats (Phase 10 / Plan 10-04 / D-19 / STA-02)", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    // Ensure document.hidden defaults to false in jsdom for tests below.
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => false,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("calls /v1/stats?window={window} on mount", async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => FAKE_POPULATED,
    } as Response)
    vi.stubGlobal("fetch", fetchSpy)
    renderHook(() => usePolledStats("24h"))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(fetchSpy).toHaveBeenCalled()
    const call = fetchSpy.mock.calls[0]
    if (!call) throw new Error("expected fetch to be called")
    expect(call[0]).toBe("/v1/stats?window=24h")
  })

  it("polls again every 2s (D-19 lock)", async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => FAKE_EMPTY,
    } as Response)
    vi.stubGlobal("fetch", fetchSpy)
    renderHook(() => usePolledStats("1h"))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(fetchSpy).toHaveBeenCalledTimes(3)
  })

  it("changing window triggers an immediate refetch with the new ?window= value", async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => FAKE_EMPTY,
    } as Response)
    vi.stubGlobal("fetch", fetchSpy)
    const { rerender } = renderHook(
      ({ w }: { w: StatsWindow }) => usePolledStats(w),
      {
        initialProps: { w: "1h" as StatsWindow },
      },
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    const firstCall = fetchSpy.mock.calls[0]
    if (!firstCall) throw new Error("expected first fetch call")
    expect(firstCall[0]).toBe("/v1/stats?window=1h")
    rerender({ w: "24h" as StatsWindow })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(fetchSpy).toHaveBeenLastCalledWith(
      "/v1/stats?window=24h",
      expect.anything(),
    )
  })

  it("flips status to ready and exposes the snapshot when fetch resolves", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => FAKE_POPULATED,
      } as Response),
    )
    const { result } = renderHook(() => usePolledStats("all"))
    // Advance several microtask cycles so fetch + .json() + setState all flush.
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
    }
    expect(result.current.status).toBe("ready")
    expect(result.current.snapshot).toEqual(FAKE_POPULATED)
  })

  it("flips status to 'paused' and skips fetch while document.hidden is true", async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => FAKE_EMPTY,
    } as Response)
    vi.stubGlobal("fetch", fetchSpy)
    const { result } = renderHook(() => usePolledStats("1h"))
    // Initial mount triggers one fetch (visible state).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    const callsBeforeHide = fetchSpy.mock.calls.length
    // Now flip to hidden and dispatch visibilitychange.
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => true,
    })
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"))
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(result.current.status).toBe("paused")
    // No new fetch should have fired while hidden.
    expect(fetchSpy.mock.calls.length).toBe(callsBeforeHide)
    // Reset for other tests.
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => false,
    })
  })

  it("flips to 'error' on non-200 with an HTTP_NNN errorClass", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({}),
      } as Response),
    )
    const { result } = renderHook(() => usePolledStats("1h"))
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
    }
    expect(result.current.status).toBe("error")
    expect(result.current.errorClass).toBe("HTTP_503")
  })
})
