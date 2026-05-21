import { describe, it, expect, beforeEach, vi, afterEach } from "vitest"
import { useConfigStore, type ConfigView } from "./configStore"

const FAKE_CONFIG: ConfigView = {
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
    api_key_env: "${OPENAI_API_KEY}", // CFG-02 — never the resolved value
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

describe("configStore (Phase 10 / Plan 10-02)", () => {
  beforeEach(() => {
    useConfigStore.getState()._resetForTest()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("starts in idle status with no config", () => {
    expect(useConfigStore.getState().config).toBeNull()
    expect(useConfigStore.getState().status).toBe("idle")
  })

  it("fetchConfig populates config and flips status to ready on 200 (idempotent)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => FAKE_CONFIG,
      } as Response),
    )
    await useConfigStore.getState().fetchConfig()
    expect(useConfigStore.getState().status).toBe("ready")
    expect(useConfigStore.getState().config).toEqual(FAKE_CONFIG)
    expect(useConfigStore.getState().config?.routing.threshold).toBe(0.7)
  })

  it("normalizes older config payloads that do not include judge.prompt", async () => {
    const legacyConfig = {
      ...FAKE_CONFIG,
      judge: { model: "qwen2.5:1.5b" },
    } as ConfigView
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => legacyConfig,
      } as Response),
    )

    await useConfigStore.getState().fetchConfig()

    expect(useConfigStore.getState().config?.judge.prompt).toBe("")
  })

  it("saveConfig PUTs config and records restart-required header", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ "X-CoreThread-Restart-Required": "true" }),
        json: async () => FAKE_CONFIG,
      } as Response),
    )
    await useConfigStore.getState().saveConfig(FAKE_CONFIG)
    expect(useConfigStore.getState().status).toBe("ready")
    expect(useConfigStore.getState().restartRequired).toBe(true)
    expect(fetch).toHaveBeenCalledWith(
      "/v1/config",
      expect.objectContaining({ method: "PUT" }),
    )
  })

  it("fetchConfig is a no-op when status is already 'ready' (idempotent)", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => FAKE_CONFIG,
    } as Response)
    vi.stubGlobal("fetch", fetchMock)
    await useConfigStore.getState().fetchConfig() // 1st call
    await useConfigStore.getState().fetchConfig() // 2nd call — must NOT re-fetch
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("fetchConfig flips status to 'error' on non-200", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({}),
      } as Response),
    )
    await useConfigStore.getState().fetchConfig()
    expect(useConfigStore.getState().status).toBe("error")
    expect(useConfigStore.getState().config).toBeNull()
  })

  it("fetchConfig flips to 'error' with class-name-only message on network throw (Pitfall #12 ethos)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    )
    await useConfigStore.getState().fetchConfig()
    expect(useConfigStore.getState().status).toBe("error")
    expect(useConfigStore.getState().errorMessage).toBe("TypeError")
    // The error STRING ("Failed to fetch") must NOT appear in the stored message.
    expect(useConfigStore.getState().errorMessage).not.toContain(
      "Failed to fetch",
    )
  })
})
