import { describe, it, expect } from "vitest"
import { _testing } from "./ConfidenceHistogram"

describe("ConfidenceHistogram bin math (Phase 10 / Plan 10-04 / D-23 / STA-03)", () => {
  const { bin, BIN_COUNT } = _testing

  it("BIN_COUNT is 10 (D-23 / planner discretion default)", () => {
    expect(BIN_COUNT).toBe(10)
  })

  it("bin(0.0) -> 0; bin(0.05) -> 0; bin(0.099) -> 0", () => {
    expect(bin(0)).toBe(0)
    expect(bin(0.05)).toBe(0)
    expect(bin(0.099)).toBe(0)
  })

  it("bin(0.5) -> 5; bin(0.55) -> 5; bin(0.59) -> 5", () => {
    expect(bin(0.5)).toBe(5)
    expect(bin(0.55)).toBe(5)
    expect(bin(0.59)).toBe(5)
  })

  it("bin(1.0) -> 9 (clamped); bin(0.999) -> 9", () => {
    expect(bin(1.0)).toBe(9)
    expect(bin(0.999)).toBe(9)
  })
})
