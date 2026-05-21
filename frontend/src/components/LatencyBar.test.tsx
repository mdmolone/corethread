import { describe, it, expect } from "vitest"
import { render, screen } from "@testing-library/react"
import { LatencyBar } from "./LatencyBar"

describe("LatencyBar (Phase 10 / Plan 10-03 / D-10 / TRC-03)", () => {
  it("renders 3 segments when frontier_latency_ms is non-null", () => {
    render(
      <LatencyBar
        local_latency_ms={400}
        judge_latency_ms={100}
        frontier_latency_ms={500}
      />,
    )
    expect(screen.getByTestId("latency-segment-local")).toBeInTheDocument()
    expect(screen.getByTestId("latency-segment-judge")).toBeInTheDocument()
    expect(screen.getByTestId("latency-segment-frontier")).toBeInTheDocument()
  })

  it("renders only 2 segments when frontier_latency_ms is null (un-pivoted row)", () => {
    render(
      <LatencyBar
        local_latency_ms={400}
        judge_latency_ms={100}
        frontier_latency_ms={null}
      />,
    )
    expect(screen.getByTestId("latency-segment-local")).toBeInTheDocument()
    expect(screen.getByTestId("latency-segment-judge")).toBeInTheDocument()
    expect(
      screen.queryByTestId("latency-segment-frontier"),
    ).not.toBeInTheDocument()
  })

  it("uses the configured time scale instead of stretching each row to 100%", () => {
    render(
      <LatencyBar
        local_latency_ms={400}
        judge_latency_ms={100}
        frontier_latency_ms={500}
        scaleMs={2000}
      />,
    )
    const local = screen.getByTestId("latency-segment-local") as HTMLDivElement
    const judge = screen.getByTestId("latency-segment-judge") as HTMLDivElement
    const frontier = screen.getByTestId(
      "latency-segment-frontier",
    ) as HTMLDivElement
    // The row totals 1000ms, but the track is 2000ms, so only half fills.
    expect(local.style.width).toBe("20%")
    expect(judge.style.width).toBe("5%")
    expect(frontier.style.width).toBe("25%")
  })

  it("renders a readable total and scale label", () => {
    render(
      <LatencyBar
        local_latency_ms={1500}
        judge_latency_ms={500}
        frontier_latency_ms={null}
        scaleMs={120_000}
      />,
    )
    expect(screen.getByText("total 2.00s")).toBeInTheDocument()
    expect(screen.getByText("scale 2.0m")).toBeInTheDocument()
  })

  it("marks rows that exceed the fixed scale", () => {
    render(
      <LatencyBar
        local_latency_ms={40_000}
        judge_latency_ms={25_000}
        frontier_latency_ms={null}
        scaleMs={60_000}
      />,
    )
    expect(screen.getByTestId("latency-over-scale")).toBeInTheDocument()
    expect(screen.getByTestId("latency-over-scale-label")).toHaveTextContent(
      "over scale by 5.00s",
    )
  })

  it("returns null when total latency is 0 (defensive - should never happen in prod)", () => {
    const { container } = render(
      <LatencyBar
        local_latency_ms={0}
        judge_latency_ms={0}
        frontier_latency_ms={null}
      />,
    )
    expect(container.firstChild).toBeNull()
  })
})
