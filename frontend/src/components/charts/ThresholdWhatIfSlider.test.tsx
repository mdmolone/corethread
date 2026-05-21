import { describe, it, expect } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"
import { ThresholdWhatIfSlider } from "./ThresholdWhatIfSlider"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

function trace(
  request_id: string,
  conf: number,
  pivoted = false,
  reason: TraceEvent["pivot_reason"] = "none",
): TraceEvent {
  return {
    request_id,
    selected_local_model: "m",
    judge_model: "j",
    frontier_model: pivoted ? "gpt-4o" : null,
    confidence_score: conf,
    pivoted,
    local_latency_ms: 1,
    judge_latency_ms: 1,
    frontier_latency_ms: pivoted ? 1 : null,
    input_tokens: 1,
    output_tokens: 1,
    frontier_cost_est: pivoted ? 0.001 : null,
    judge_parse_failed: false,
    pivot_reason: reason,
    local_error_class: null,
  }
}

describe("ThresholdWhatIfSlider (Phase 10 / Plan 10-04 / D-22 / STA-05)", () => {
  it("renders the slider with min=0 max=1 step=0.01 (D-22 lock)", () => {
    render(
      <ThresholdWhatIfSlider
        traces={[]}
        configuredThreshold={0.7}
        sliderValue={0.7}
        onSliderChange={() => {}}
      />,
    )
    const slider = screen.getByTestId("threshold-slider") as HTMLInputElement
    expect(slider.min).toBe("0")
    expect(slider.max).toBe("1")
    expect(slider.step).toBe("0.01")
  })

  it("readout 'At threshold 0.70, 0/0 would have pivoted (0.0%)' for empty buffer", () => {
    render(
      <ThresholdWhatIfSlider
        traces={[]}
        configuredThreshold={0.7}
        sliderValue={0.7}
        onSliderChange={() => {}}
      />,
    )
    expect(screen.getByTestId("threshold-readout")).toHaveTextContent(
      "At threshold 0.70, 0/0 would have pivoted (0.0%)",
    )
  })

  it("predicate over a synthetic 100-trace buffer: confidence < threshold OR non-none pivot_reason", () => {
    // Build 100 traces: 50 with confidence 0.5 (would pivot at 0.7), 50 with confidence 0.9 (would not).
    // Sprinkle 5 of the high-confidence ones with pivot_reason='judge_error' (would-pivot regardless).
    const traces: TraceEvent[] = []
    for (let i = 0; i < 50; i++) traces.push(trace(`a-${i}`, 0.5))
    for (let i = 0; i < 50; i++) {
      traces.push(trace(`b-${i}`, 0.9, i < 5, i < 5 ? "judge_error" : "none"))
    }
    render(
      <ThresholdWhatIfSlider
        traces={traces}
        configuredThreshold={0.7}
        sliderValue={0.7}
        onSliderChange={() => {}}
      />,
    )
    // 50 (low-conf) + 5 (judge_error) = 55 of 100 → 55.0%
    expect(screen.getByTestId("threshold-readout")).toHaveTextContent(
      "At threshold 0.70, 55/100 would have pivoted (55.0%)",
    )
  })

  it("dragging the slider invokes onSliderChange with the new value", () => {
    let captured = -1
    render(
      <ThresholdWhatIfSlider
        traces={[]}
        configuredThreshold={0.7}
        sliderValue={0.7}
        onSliderChange={(v) => (captured = v)}
      />,
    )
    fireEvent.change(screen.getByTestId("threshold-slider"), {
      target: { value: "0.42" },
    })
    expect(captured).toBe(0.42)
  })
})
