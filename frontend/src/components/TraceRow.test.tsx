import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"
import { TraceRow } from "./TraceRow"
import { useTraceStore } from "@/store/traceStore"
import {
  useConfigStore,
  type ConfigView as StoreConfigView,
} from "@/store/configStore"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

const acceptedTrace: TraceEvent = {
  request_id: "req-acc",
  selected_local_model: "qwen2.5:7b",
  judge_model: "qwen2.5:1.5b",
  frontier_model: null,
  confidence_score: 0.95,
  pivoted: false,
  local_latency_ms: 100,
  judge_latency_ms: 50,
  frontier_latency_ms: null,
  input_tokens: 12,
  output_tokens: 34,
  frontier_cost_est: null,
  judge_parse_failed: false,
  pivot_reason: "none",
  local_error_class: null,
}

const erroredTrace: TraceEvent = {
  ...acceptedTrace,
  request_id: "req-err",
  pivoted: true,
  pivot_reason: "judge_error",
  judge_parse_failed: true,
  frontier_model: "gpt-4o",
  frontier_latency_ms: 700,
  frontier_cost_est: 0.0123,
}

const lowScoreTrace: TraceEvent = {
  ...acceptedTrace,
  request_id: "req-low",
  pivoted: true,
  pivot_reason: "low_score",
  confidence_score: 0.42,
  frontier_model: "gpt-4o",
  frontier_latency_ms: 800,
  frontier_cost_est: 0.0089,
}

// Synthetic ConfigView for tests — mirrors the Phase 9 wire shape; threshold=0.7
// is the only field TraceRow reads (via narratePivot's threshold arg).
const TEST_CONFIG: StoreConfigView = {
  local: {
    kind: "ollama",
    base_url: "http://localhost:11434",
    model: "qwen2.5:7b",
    num_ctx_default: 8192,
    num_ctx_overrides: {},
  },
  judge: {
    model: "qwen2.5:1.5b",
    prompt: "Judge the local answer and return the expected JSON shape.",
  },
  frontier: {
    model: "gpt-4o",
    api_key_env: "${OPENAI_API_KEY}",
    max_tokens: 512,
  },
  routing: { threshold: 0.7, constraint_prompt: "Be concise." },
  model_profiles: {
    "local-default": {
      provider: "ollama",
      base_url: "http://localhost:11434",
      model: "qwen2.5:7b",
      num_ctx_default: 8192,
      num_ctx_overrides: {},
    },
    "judge-default": {
      provider: "ollama",
      base_url: "http://localhost:11434",
      model: "qwen2.5:1.5b",
      num_ctx_default: 8192,
      num_ctx_overrides: {},
    },
    "frontier-default": {
      provider: "openai",
      base_url: "https://api.openai.com/v1",
      model: "gpt-4o",
      api_key_env: "${OPENAI_API_KEY}",
      max_tokens: 512,
      num_ctx_overrides: {},
    },
  },
  role_profiles: {
    local: "local-default",
    judge: "judge-default",
    frontier: "frontier-default",
  },
  ui: { theme: "system" },
  privacy: { capture_transcripts: false, transcript_max: 25 },
  controls: {
    requests_per_minute: null,
    daily_request_quota: null,
    daily_token_quota: null,
    daily_cost_quota_usd: null,
    pricing: {},
    audit_enabled: false,
    audit_path: "outputs/audit.jsonl",
    audit_include_request_body: false,
  },
}

describe("TraceRow (Phase 10 / Plan 10-03 / TRC-02 + TRC-04 + TRC-07 + TRC-08)", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise(() => {})),
    )
    useTraceStore.setState({ expandedRequestId: null, traces: [] })
    useConfigStore.setState({
      config: TEST_CONFIG,
      status: "ready",
      errorMessage: null,
      restartRequired: false,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("TRC-02: accepted row has blue border-l class", () => {
    render(<TraceRow trace={acceptedTrace} />)
    const row = screen.getByTestId("trace-row")
    expect(row.className).toMatch(/border-l-blue-500/)
    expect(row.dataset.classification).toBe("accepted")
  })

  it("TRC-02: errored row has red border-l class", () => {
    render(<TraceRow trace={erroredTrace} />)
    const row = screen.getByTestId("trace-row")
    expect(row.className).toMatch(/border-l-red-500/)
    expect(row.dataset.classification).toBe("errored")
  })

  it("TRC-02: pivoted (low_score) row has amber border-l class", () => {
    render(<TraceRow trace={lowScoreTrace} />)
    const row = screen.getByTestId("trace-row")
    expect(row.className).toMatch(/border-l-amber-500/)
    expect(row.dataset.classification).toBe("pivoted")
  })

  it("TRC-08: pivoted row shows narration string from narration.ts", () => {
    render(<TraceRow trace={lowScoreTrace} />)
    expect(screen.getByTestId("pivot-narration")).toHaveTextContent(
      "Confidence 0.42 < 0.70",
    )
  })

  it("TRC-08: accepted row does NOT show narration", () => {
    render(<TraceRow trace={acceptedTrace} />)
    expect(screen.queryByTestId("pivot-narration")).not.toBeInTheDocument()
  })

  it("TRC-04: clicking the row expands and shows all 15 RequestTrace fields", () => {
    render(<TraceRow trace={lowScoreTrace} />)
    fireEvent.click(screen.getByTestId("trace-row"))
    const grid = screen.getByTestId("trace-expanded-grid")
    expect(grid).toBeInTheDocument()
    // Assert every one of the 15 field labels appears.
    const fields = [
      "request_id:",
      "selected_local_model:",
      "judge_model:",
      "frontier_model:",
      "confidence_score:",
      "pivoted:",
      "local_latency_ms:",
      "judge_latency_ms:",
      "frontier_latency_ms:",
      "input_tokens:",
      "output_tokens:",
      "frontier_cost_est:",
      "judge_parse_failed:",
      "pivot_reason:",
      "local_error_class:",
    ]
    for (const f of fields) {
      expect(grid.textContent).toContain(f)
    }
  })

  it("TRC-04 / D-11: clicking again collapses (single-row-expanded-at-a-time)", () => {
    render(<TraceRow trace={lowScoreTrace} />)
    const row = screen.getByTestId("trace-row")
    fireEvent.click(row)
    expect(screen.getByTestId("trace-expanded-grid")).toBeInTheDocument()
    fireEvent.click(row)
    expect(screen.queryByTestId("trace-expanded-grid")).not.toBeInTheDocument()
  })
})
