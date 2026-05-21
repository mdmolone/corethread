import { describe, it, expect } from "vitest"
import { classifyTrace } from "./classifyTrace"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

const baseTrace: TraceEvent = {
  request_id: "req",
  selected_local_model: "m",
  judge_model: "j",
  frontier_model: null,
  confidence_score: 0.9,
  pivoted: false,
  local_latency_ms: 1,
  judge_latency_ms: 1,
  frontier_latency_ms: null,
  input_tokens: 1,
  output_tokens: 1,
  frontier_cost_est: null,
  judge_parse_failed: false,
  pivot_reason: "none",
  local_error_class: null,
}

describe("classifyTrace (Phase 10 / Plan 10-03 / D-12 + D-15)", () => {
  it("accepted when pivoted=false and pivot_reason=none", () => {
    expect(classifyTrace(baseTrace)).toBe("accepted")
  })

  it("pivoted when pivoted=true and pivot_reason=low_score", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "low_score",
    }
    expect(classifyTrace(t)).toBe("pivoted")
  })

  it("pivoted when pivoted=true and pivot_reason=local_truncated", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "local_truncated",
    }
    expect(classifyTrace(t)).toBe("pivoted")
  })

  it("errored when pivot_reason=local_error (overrides pivoted=true)", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "local_error",
    }
    expect(classifyTrace(t)).toBe("errored")
  })

  it("errored when pivot_reason=judge_error (overrides pivoted=true)", () => {
    const t: TraceEvent = {
      ...baseTrace,
      pivoted: true,
      pivot_reason: "judge_error",
    }
    expect(classifyTrace(t)).toBe("errored")
  })

  it("errored when judge_parse_failed=true even if pivot_reason=none", () => {
    const t: TraceEvent = { ...baseTrace, judge_parse_failed: true }
    expect(classifyTrace(t)).toBe("errored")
  })
})
