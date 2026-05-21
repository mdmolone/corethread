import { describe, it, expect } from "vitest"
import { narratePivot } from "./narration"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

const baseTrace: TraceEvent = {
  request_id: "req-test",
  selected_local_model: "qwen2.5:7b",
  judge_model: "qwen2.5:1.5b",
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
}

describe("narratePivot (Phase 10 / Plan 10-03 / D-14 / TRC-08)", () => {
  it("returns empty string when pivoted=false (no narration)", () => {
    expect(narratePivot(baseTrace, 0.7)).toBe("")
  })

  it("low_score: 'Confidence X.XX < Y.YY' (TraceEvent.confidence vs threshold)", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "low_score",
      confidence_score: 0.42,
      frontier_model: "gpt-4o",
      frontier_latency_ms: 800,
    }
    expect(narratePivot(t, 0.7)).toBe("Confidence 0.42 < 0.70")
  })

  it("local_error: 'Local provider unreachable (<error_class>)'", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "local_error",
      local_error_class: "ProviderUnavailable",
      frontier_model: "gpt-4o",
      frontier_latency_ms: 700,
    }
    expect(narratePivot(t, 0.7)).toBe(
      "Local provider unreachable (ProviderUnavailable)",
    )
  })

  it("local_truncated: 'Local response hit token limit'", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "local_truncated",
      confidence_score: 0,
      frontier_model: "gpt-4o",
      frontier_latency_ms: 900,
    }
    expect(narratePivot(t, 0.7)).toBe("Local response hit token limit")
  })

  it("local_error with null local_error_class falls back to 'unknown'", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "local_error",
      local_error_class: null,
      frontier_model: "gpt-4o",
    }
    expect(narratePivot(t, 0.7)).toBe("Local provider unreachable (unknown)")
  })

  it("judge_error: 'Judge JSON parse failed'", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "judge_error",
      judge_parse_failed: true,
    }
    expect(narratePivot(t, 0.7)).toBe("Judge JSON parse failed")
  })

  it("pivot_reason='none' with pivoted=true returns empty string (defensive)", () => {
    const t: TraceEvent = { ...baseTrace, pivoted: true, pivot_reason: "none" }
    expect(narratePivot(t, 0.7)).toBe("")
  })

  it("threshold formatting: always 2 decimal places (toFixed(2))", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "low_score",
      confidence_score: 0.5,
    }
    expect(narratePivot(t, 0.7)).toBe("Confidence 0.50 < 0.70")
    expect(narratePivot(t, 0.123456)).toBe("Confidence 0.50 < 0.12")
  })
})
