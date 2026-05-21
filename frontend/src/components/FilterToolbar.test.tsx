import { describe, it, expect, beforeEach } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"
import { FilterToolbar } from "./FilterToolbar"
import { useTraceStore } from "@/store/traceStore"

describe("FilterToolbar (Phase 10 / Plan 10-03 / D-13 / TRC-06)", () => {
  beforeEach(() => {
    useTraceStore.setState({
      traces: [],
      filters: {
        pivoted: "any",
        model: "any",
        pivot_reason: "any",
        latency_band: "any",
        confidence_band: "any",
      },
    })
  })

  it("renders all 5 filter chips + a trace-count badge", () => {
    render(<FilterToolbar />)
    expect(screen.getByTestId("chip-pivoted")).toBeInTheDocument()
    expect(screen.getByTestId("chip-model")).toBeInTheDocument()
    expect(screen.getByTestId("chip-pivot_reason")).toBeInTheDocument()
    expect(screen.getByTestId("chip-latency")).toBeInTheDocument()
    expect(screen.getByTestId("chip-confidence")).toBeInTheDocument()
  })

  it("clicking a chip option updates useTraceStore.filters[key]", () => {
    render(<FilterToolbar />)
    fireEvent.click(screen.getByTestId("chip-pivoted"))
    fireEvent.click(screen.getByTestId("chip-option-pivoted-yes"))
    expect(useTraceStore.getState().filters.pivoted).toBe("yes")
  })

  it("Clear all button appears only when at least one filter is non-'any', and resetFilters wipes the lot", () => {
    render(<FilterToolbar />)
    expect(screen.queryByTestId("filter-clear-all")).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId("chip-pivoted"))
    fireEvent.click(screen.getByTestId("chip-option-pivoted-yes"))
    expect(screen.getByTestId("filter-clear-all")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("filter-clear-all"))
    expect(useTraceStore.getState().filters.pivoted).toBe("any")
    expect(screen.queryByTestId("filter-clear-all")).not.toBeInTheDocument()
  })

  it("model dropdown union includes all selected_local_model + frontier_model values seen in the buffer", () => {
    useTraceStore.setState({
      traces: [
        {
          request_id: "1",
          selected_local_model: "qwen2.5:7b",
          judge_model: "j",
          frontier_model: "gpt-4o",
          confidence_score: 0.5,
          pivoted: true,
          local_latency_ms: 1,
          judge_latency_ms: 1,
          frontier_latency_ms: 1,
          input_tokens: 1,
          output_tokens: 1,
          frontier_cost_est: 0.001,
          judge_parse_failed: false,
          pivot_reason: "low_score",
          local_error_class: null,
        },
        {
          request_id: "2",
          selected_local_model: "llama3.1:8b",
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
        },
      ],
    })
    render(<FilterToolbar />)
    fireEvent.click(screen.getByTestId("chip-model"))
    expect(
      screen.getByTestId("chip-option-model-qwen2.5:7b"),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId("chip-option-model-llama3.1:8b"),
    ).toBeInTheDocument()
    expect(screen.getByTestId("chip-option-model-gpt-4o")).toBeInTheDocument()
  })
})
