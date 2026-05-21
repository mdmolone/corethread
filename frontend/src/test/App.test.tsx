import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen } from "@testing-library/react"

// Mock useEventSource so the App test doesn't open a real SSE connection.
// Vitest hoists vi.mock above all imports in the same file — the App import
// below sees the mocked module.
vi.mock("@/hooks/useEventSource", () => ({
  useEventSource: () => ({ status: "connecting" }),
}))

vi.mock("@/hooks/useRouteActivitySource", () => ({
  useRouteActivitySource: () => ({ status: "connecting" }),
}))

// Mock usePolledStats so StatsView (mounted lazily but imported at top) does
// not trigger a real /v1/stats fetch under the App-shell smoke test. App.test
// is a SHELL smoke test, not a deep StatsView test (those live in 10-04 unit
// tests for StatHero / charts / hook).
vi.mock("@/hooks/usePolledStats", () => ({
  usePolledStats: () => ({ snapshot: null, status: "idle", errorClass: null }),
}))

// Mock useHealthPoll so TopBar (mounted in the header) does not trigger a real
// /health fetch under the App-shell smoke test. Phase 10 / Plan 10-05 wire-up.
vi.mock("@/hooks/useHealthPoll", () => ({
  useHealthPoll: () => ({ health: null, status: "idle", errorClass: null }),
}))

import { App } from "@/App"

beforeEach(() => {
  // Mock global fetch so configStore.fetchConfig doesn't hit a real /v1/config.
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({}),
    } as Response),
  )
})

describe("App shell (Phase 10 / Plan 10-01 + 10-03 wiring)", () => {
  it("renders the 3-tab bar", () => {
    render(<App />)
    expect(screen.getByRole("tab", { name: /config/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /traces/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /stats/i })).toBeInTheDocument()
  })

  it("defaults to the Traces tab (D-domain item 6) — TraceView renders", () => {
    render(<App />)
    expect(screen.getByTestId("trace-view")).toBeInTheDocument()
  })

  it("registers EventSource polyfill in jsdom (Pitfall 28 / SC#6)", () => {
    expect(typeof globalThis.EventSource).toBe("function")
  })

  it("renders SSE status badge with the mocked 'connecting' state", () => {
    render(<App />)
    expect(screen.getByTestId("sse-status")).toHaveTextContent(
      "SSE: connecting",
    )
    expect(screen.getByTestId("route-status")).toHaveTextContent(
      "Route: connecting",
    )
  })
})
