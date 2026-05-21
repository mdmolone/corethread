import { describe, it, expect } from "vitest"
import { render, screen } from "@testing-library/react"
import { StatHero } from "./StatHero"

const EMPTY = {
  window_started_at: 0,
  window_size: 0 as const,
  max_window_size: 1000,
  note: "Stats reset on CoreThread restart (in-memory).",
}

const POP = {
  window_started_at: 1_700_000_000,
  window_size: 100,
  max_window_size: 1000,
  pivot_rate: 0.42,
  pivot_reasons: {
    none: 58,
    low_score: 30,
    local_truncated: 0,
    local_error: 7,
    judge_error: 5,
  },
  tokens_in_total: 1234,
  tokens_out_total: 5678,
  local_latency_ms: { p50: 100, p95: 200, max: 250, min: 50 },
  judge_latency_ms: { p50: 30, p95: 60, max: 80, min: 15 },
  frontier_latency_ms: { p50: 600, p95: 900, max: 1200, min: 400 },
  confidence_score: { p50: 0.85, p95: 0.95, max: 0.99, min: 0.42 },
  note: "Stats reset on CoreThread restart (in-memory).",
}

describe("StatHero (Phase 10 / Plan 10-04 / STA-01)", () => {
  it("renders an empty layout when window_size === 0 (D-15 narrowing)", () => {
    render(<StatHero snapshot={EMPTY} />)
    expect(screen.getByTestId("stat-hero-empty")).toBeInTheDocument()
  })

  it("renders populated counters: total, pivot_rate %, p50/p95 per stage, confidence_p50", () => {
    render(<StatHero snapshot={POP} />)
    expect(screen.getByTestId("stat-hero-total")).toHaveTextContent("100")
    expect(screen.getByTestId("stat-hero-pivot_rate")).toHaveTextContent(
      "42.0%",
    )
    expect(screen.getByTestId("stat-hero-local_p50")).toHaveTextContent("100ms")
    expect(screen.getByTestId("stat-hero-local_p95")).toHaveTextContent("200ms")
    expect(screen.getByTestId("stat-hero-judge_p50")).toHaveTextContent("30ms")
    expect(screen.getByTestId("stat-hero-judge_p95")).toHaveTextContent("60ms")
    expect(screen.getByTestId("stat-hero-frontier_p50")).toHaveTextContent(
      "600ms",
    )
    expect(screen.getByTestId("stat-hero-confidence_p50")).toHaveTextContent(
      "0.85",
    )
  })
})
