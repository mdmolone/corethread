import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { renderHook, act } from "@testing-library/react"
import { useHealthPoll } from "./useHealthPoll"

const HEALTHY = {
  status: "ok" as const,
  version: "1.1.0",
  providers: {
    local: { kind: "ollama", state: "ready" as const, last_error: null },
    frontier: { kind: "openai", state: "ready" as const, last_error: null },
  },
}

const DEGRADED = {
  status: "degraded" as const,
  version: "1.1.0",
  providers: {
    local: { kind: "ollama", state: "warming" as const, last_error: null },
    frontier: {
      kind: "openai",
      state: "unhealthy" as const,
      last_error: "AuthenticationError",
    },
  },
}

describe("useHealthPoll (Phase 10 / Plan 10-05 / D-26 / HEA-01)", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("calls /health on mount and exposes the response", async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => HEALTHY,
    } as Response)
    vi.stubGlobal("fetch", fetchSpy)
    const { result } = renderHook(() => useHealthPoll())
    // Flush microtasks to allow the first fetch to resolve under fake timers.
    // (waitFor uses real-time polling that does not advance fake timers — see
    // 10-04 SUMMARY deviation #1 for the established pattern.)
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
    }
    expect(fetchSpy).toHaveBeenCalledWith("/health", expect.anything())
    expect(result.current.status).toBe("ready")
    expect(result.current.health).toEqual(HEALTHY)
  })

  it("polls every 5s (D-26 lock)", async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => HEALTHY,
    } as Response)
    vi.stubGlobal("fetch", fetchSpy)
    renderHook(() => useHealthPoll())
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
    }
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(fetchSpy).toHaveBeenCalledTimes(3)
  })

  it("exposes degraded providers (warming + unhealthy + last_error)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => DEGRADED,
      } as Response),
    )
    const { result } = renderHook(() => useHealthPoll())
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
    }
    expect(result.current.status).toBe("ready")
    expect(result.current.health?.providers.local.state).toBe("warming")
    expect(result.current.health?.providers.frontier.state).toBe("unhealthy")
    expect(result.current.health?.providers.frontier.last_error).toBe(
      "AuthenticationError",
    )
  })

  it("flips to 'paused' on document.hidden", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => HEALTHY,
      } as Response),
    )
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => true,
    })
    const { result } = renderHook(() => useHealthPoll())
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"))
      await vi.advanceTimersByTimeAsync(5000)
    })
    expect(result.current.status).toBe("paused")
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => false,
    })
  })
})
