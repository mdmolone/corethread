import { describe, it, expect } from "vitest"
import { render, screen } from "@testing-library/react"
import { HealthDot } from "./HealthDot"

describe("HealthDot (Phase 10 / Plan 10-05 / D-26 / HEA-01)", () => {
  it("ready -> bg-green-500", () => {
    render(<HealthDot label="local" state="ready" last_error={null} />)
    const dot = screen.getByTestId("health-dot-local")
    expect(dot.className).toMatch(/bg-green-500/)
    expect(dot.dataset.state).toBe("ready")
  })

  it("warming -> bg-amber-500", () => {
    render(<HealthDot label="local" state="warming" last_error={null} />)
    expect(screen.getByTestId("health-dot-local").className).toMatch(
      /bg-amber-500/,
    )
  })

  it("unhealthy -> bg-red-500", () => {
    render(
      <HealthDot
        label="frontier"
        state="unhealthy"
        last_error="AuthenticationError"
      />,
    )
    expect(screen.getByTestId("health-dot-frontier").className).toMatch(
      /bg-red-500/,
    )
  })

  it("loading -> bg-gray-400 (initial state before first poll resolves)", () => {
    render(<HealthDot label="local" state="loading" last_error={null} />)
    expect(screen.getByTestId("health-dot-local").className).toMatch(
      /bg-gray-400/,
    )
  })

  it("aria-label matches the dot state for screen readers", () => {
    render(
      <HealthDot
        label="frontier"
        state="unhealthy"
        last_error="AuthenticationError"
      />,
    )
    const dot = screen.getByTestId("health-dot-frontier")
    expect(dot.getAttribute("aria-label")).toBe("frontier: unhealthy")
  })
})
