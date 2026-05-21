import { describe, it, expect } from "vitest"
import { render, screen } from "@testing-library/react"
import { CostSavedHero } from "./CostSavedHero"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

function makeTrace(
  request_id: string,
  input: number,
  output: number,
  pivoted: boolean,
): TraceEvent {
  return {
    request_id,
    selected_local_model: "m",
    judge_model: "j",
    frontier_model: pivoted ? "gpt-4o" : null,
    confidence_score: 0.9,
    pivoted,
    local_latency_ms: 1,
    judge_latency_ms: 1,
    frontier_latency_ms: pivoted ? 1 : null,
    input_tokens: input,
    output_tokens: output,
    frontier_cost_est: pivoted ? 0.001 : null,
    judge_parse_failed: false,
    pivot_reason: pivoted ? "low_score" : "none",
    local_error_class: null,
  }
}

describe("CostSavedHero (Phase 10 / Plan 10-04 / D-20 / STA-06)", () => {
  it("$0.00 + 0 requests when buffer is empty", () => {
    render(<CostSavedHero traces={[]} />)
    expect(screen.getByTestId("cost-saved-amount")).toHaveTextContent("$0.00")
    expect(screen.getByTestId("cost-saved-count")).toHaveTextContent(
      "0 requests handled locally",
    )
  })

  it("counts only !pivoted traces (D-20 — pivoted requests cost the same in all-frontier)", () => {
    // 1 non-pivoted trace: 1M input + 1M output → $2.50 + $10 = $12.50
    // 1 pivoted trace (ignored)
    render(
      <CostSavedHero
        traces={[
          makeTrace("a", 1_000_000, 1_000_000, false),
          makeTrace("b", 1_000_000, 1_000_000, true),
        ]}
      />,
    )
    expect(screen.getByTestId("cost-saved-amount")).toHaveTextContent("$12.50")
    expect(screen.getByTestId("cost-saved-count")).toHaveTextContent(
      "1 requests handled locally",
    )
  })

  it("realistic small request: 50 input + 100 output (non-pivoted) → ~$0.00", () => {
    // 50 * 2.50/1M + 100 * 10/1M = 0.000125 + 0.001 = 0.001125 → rounds to $0.00 in USD (2 decimals)
    render(<CostSavedHero traces={[makeTrace("a", 50, 100, false)]} />)
    expect(screen.getByTestId("cost-saved-amount")).toHaveTextContent("$0.00")
    expect(screen.getByTestId("cost-saved-count")).toHaveTextContent(
      "1 requests handled locally",
    )
  })
})
