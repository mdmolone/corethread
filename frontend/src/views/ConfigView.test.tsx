import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigView } from "./ConfigView"
import {
  useConfigStore,
  type ConfigView as ConfigViewType,
} from "@/store/configStore"

const SYNTHETIC_CONFIG: ConfigViewType = {
  local: {
    kind: "ollama",
    base_url: "http://localhost:11434",
    model: "qwen2.5:7b",
    num_ctx_default: 8192,
    num_ctx_overrides: { "qwen2.5:14b": 16384 },
  },
  judge: {
    model: "qwen2.5:1.5b",
    prompt: "Judge the local answer and return the expected JSON shape.",
  },
  frontier: {
    model: "gpt-4o",
    api_key_env: "${OPENAI_API_KEY}", // D-25 binding literal
    max_tokens: 512,
  },
  routing: {
    threshold: 0.7,
    constraint_prompt:
      "Respond concisely. No code blocks unless explicitly requested.",
  },
  model_profiles: {
    "local-default": {
      provider: "ollama",
      base_url: "http://localhost:11434",
      model: "qwen2.5:7b",
      num_ctx_default: 8192,
      num_ctx_overrides: { "qwen2.5:14b": 16384 },
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
    pricing: {
      "gpt-4o": {
        input_per_1m_tokens: 2.5,
        output_per_1m_tokens: 10,
      },
    },
    audit_enabled: false,
    audit_path: "outputs/audit.jsonl",
    audit_include_request_body: false,
  },
}

describe("ConfigView (Phase 10 / Plan 10-05 / D-24 / D-25 / CFG-01 + CFG-02 + CFG-03 rendering)", () => {
  beforeEach(() => {
    useConfigStore.setState({
      config: SYNTHETIC_CONFIG,
      status: "ready",
      errorMessage: null,
      restartRequired: false,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("renders config cards including privacy controls", () => {
    render(<ConfigView />)
    expect(screen.getByTestId("config-card-local")).toBeInTheDocument()
    expect(screen.getByTestId("config-card-judge")).toBeInTheDocument()
    expect(screen.getByTestId("config-card-frontier")).toBeInTheDocument()
    expect(screen.getByTestId("config-card-routing")).toBeInTheDocument()
    expect(screen.getByTestId("config-card-privacy")).toBeInTheDocument()
    expect(screen.getByTestId("config-card-controls")).toBeInTheDocument()
  })

  it("does not blank the page for older config payloads without judge.prompt", () => {
    useConfigStore.setState({
      config: {
        ...SYNTHETIC_CONFIG,
        judge: { model: "qwen2.5:1.5b" },
      } as ConfigViewType,
      status: "ready",
      errorMessage: null,
      restartRequired: false,
    })

    render(<ConfigView />)

    expect(screen.getByTestId("config-card-judge")).toBeInTheDocument()
    expect(screen.getByTestId("judge-prompt-block")).toBeInTheDocument()
  })

  it("CFG-02 binding: renders ${OPENAI_API_KEY} env-var name and NO `sk-` substring anywhere in document.body.textContent (D-25)", () => {
    const { container } = render(<ConfigView />)
    // 1. The literal env-var name appears.
    expect(screen.getByTestId("frontier-api-key-pill")).toHaveTextContent(
      "${OPENAI_API_KEY}",
    )
    // 2. No `sk-` substring anywhere in the rendered output (D-25 literal-absence pin).
    expect(document.body.textContent ?? "").not.toContain("sk-")
    expect(container.textContent ?? "").not.toContain("sk-")
  })

  it("CFG-03 binding: renders max_tokens with the cost-lever annotation badge (D-24 / DF-11)", () => {
    render(<ConfigView />)
    expect(screen.getByTestId("max-tokens-annotation")).toHaveTextContent(
      /caps every frontier response/i,
    )
    // The numeric value 512 is rendered next to it.
    expect(screen.getByTestId("config-card-frontier").textContent).toContain(
      "512",
    )
  })

  it("CFG-03 binding: renders constraint_prompt as a pre block with character count + annotation (D-24 / DF-10)", () => {
    render(<ConfigView />)
    expect(
      screen.getByTestId("constraint-prompt-annotation"),
    ).toHaveTextContent(/prepends to every frontier call/i)
    expect(screen.getByTestId("constraint-prompt-block")).toHaveTextContent(
      "Respond concisely. No code blocks unless explicitly requested.",
    )
    expect(screen.getByTestId("config-card-routing").textContent).toContain(
      `${SYNTHETIC_CONFIG.routing.constraint_prompt.length} chars`,
    )
  })

  it("CFG-01 carry-forward: every config.yaml field is rendered grouped by section", () => {
    render(<ConfigView />)
    // Local card fields
    const local = screen.getByTestId("config-card-local").textContent ?? ""
    expect(local).toContain("ollama")
    expect(local).toContain("http://localhost:11434")
    expect(local).toContain("qwen2.5:7b")
    expect(local).toContain("8192")
    // num_ctx_overrides JSON
    expect(local).toContain("16384")
    // Judge card
    expect(screen.getByTestId("config-card-judge").textContent).toContain(
      "qwen2.5:1.5b",
    )
    expect(screen.getByTestId("config-card-judge").textContent).toContain(
      "expected JSON shape",
    )
    // Frontier card
    expect(screen.getByTestId("config-card-frontier").textContent).toContain(
      "gpt-4o",
    )
    // Routing card threshold
    expect(screen.getByTestId("config-card-routing").textContent).toContain(
      "0.70",
    )
  })

  it("renders a loading placeholder when configStore is loading", () => {
    useConfigStore.setState({
      config: null,
      status: "loading",
      errorMessage: null,
      restartRequired: false,
    })
    render(<ConfigView />)
    expect(screen.getByTestId("config-loading")).toBeInTheDocument()
  })

  it("renders a destructive error card when configStore is error", () => {
    useConfigStore.setState({
      config: null,
      status: "error",
      errorMessage: "TypeError",
      restartRequired: false,
    })
    render(<ConfigView />)
    expect(screen.getByTestId("config-error")).toBeInTheDocument()
    expect(screen.getByTestId("config-error").textContent).toContain(
      "TypeError",
    )
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
  })

  it("renames a profile id and keeps role assignments pointed at it", async () => {
    const user = userEvent.setup()
    render(<ConfigView />)

    const input = screen.getByLabelText(
      "Profile id for local-default",
    ) as HTMLInputElement
    await user.clear(input)
    await user.type(input, "local-fast")
    await user.tab()

    await waitFor(() => {
      expect(
        screen.getByTestId("config-profile-local-fast"),
      ).toBeInTheDocument()
    })
    expect(
      screen.queryByTestId("config-profile-local-default"),
    ).not.toBeInTheDocument()
    expect(screen.getByLabelText("local")).toHaveValue("local-fast")
  })

  it("tests a profile connection and exposes returned models as a selector", async () => {
    const user = userEvent.setup()
    const fetchSpy = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          ok: true,
          provider: "ollama",
          models: ["qwen2.5:14b", "qwen2.5:7b"],
          selected_model_available: true,
          error_class: null,
          message: "Connected; found 2 models.",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    })
    vi.stubGlobal("fetch", fetchSpy)

    render(<ConfigView />)
    await user.click(
      screen.getByRole("button", {
        name: "Test connection for local-default",
      }),
    )

    expect(
      await screen.findByTestId("probe-message-local-default"),
    ).toHaveTextContent("Connected; found 2 models.")
    const selector = (await screen.findByLabelText(
      "Available models for local-default",
    )) as HTMLSelectElement
    expect(selector).toHaveValue("qwen2.5:7b")

    await user.selectOptions(selector, "qwen2.5:14b")
    expect(selector).toHaveValue("qwen2.5:14b")
    expect(fetchSpy).toHaveBeenCalledWith(
      "/v1/config/probe",
      expect.objectContaining({ method: "POST" }),
    )
  })

  it("shows restart control after a config save requires restart", async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.fn(() => false)
    const fetchSpy = vi.fn()
    vi.stubGlobal("confirm", confirmSpy)
    vi.stubGlobal("fetch", fetchSpy)
    useConfigStore.setState({ restartRequired: true })

    render(<ConfigView />)
    await user.click(screen.getByRole("button", { name: "Restart" }))

    expect(screen.getByText("Restart required")).toBeInTheDocument()
    expect(confirmSpy).toHaveBeenCalledOnce()
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("edits request controls and pricing JSON", async () => {
    const user = userEvent.setup()
    render(<ConfigView />)

    const rpm = screen.getByLabelText("requests_per_minute")
    await user.type(rpm, "12")

    const auditEnabled = screen.getByLabelText("audit_enabled")
    await user.click(auditEnabled)

    const pricing = screen.getByLabelText("pricing")
    const pricingValue = JSON.stringify({
      "openai/gpt-4o-mini": {
        input_per_1m_tokens: 0.15,
        output_per_1m_tokens: 0.6,
      },
    })
    fireEvent.change(pricing, { target: { value: pricingValue } })

    expect(rpm).toHaveValue(12)
    expect(auditEnabled).toBeChecked()
    expect(pricing).toHaveValue(pricingValue)
  })
})
