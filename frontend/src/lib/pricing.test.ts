import { describe, it, expect } from "vitest"
import { GPT4O_PRICING, estimateCost } from "./pricing"

describe("pricing (Phase 10 / Plan 10-04 / D-20 / D-21 / STA-06 / Pitfall 24)", () => {
  it("GPT4O_PRICING matches the 2026-05-05 verified rates", () => {
    expect(GPT4O_PRICING.input_per_1m_tokens).toBe(2.5)
    expect(GPT4O_PRICING.output_per_1m_tokens).toBe(10)
  })

  it("GPT4O_PRICING values are exactly the lock", () => {
    expect(GPT4O_PRICING).toEqual({
      input_per_1m_tokens: 2.5,
      output_per_1m_tokens: 10,
    })
  })

  it("estimateCost: 1M input tokens = $2.50", () => {
    expect(estimateCost(1_000_000, 0)).toBeCloseTo(2.5, 5)
  })

  it("estimateCost: 1M output tokens = $10.00", () => {
    expect(estimateCost(0, 1_000_000)).toBeCloseTo(10.0, 5)
  })

  it("estimateCost: 500k input + 100k output = $1.25 + $1.00 = $2.25", () => {
    expect(estimateCost(500_000, 100_000)).toBeCloseTo(2.25, 5)
  })

  it("estimateCost: 0 tokens = $0", () => {
    expect(estimateCost(0, 0)).toBe(0)
  })

  it("estimateCost: realistic small request (50 input + 100 output) ~ $0.001125", () => {
    // 50 * 2.50 / 1M + 100 * 10 / 1M = 0.000125 + 0.001 = 0.001125
    expect(estimateCost(50, 100)).toBeCloseTo(0.001125, 7)
  })
})
