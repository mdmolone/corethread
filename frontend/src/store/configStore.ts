import { create } from "zustand"
import type { components } from "@/api/types"

export type ProviderKind = "ollama" | "lmstudio" | "openai" | "openrouter"
export type ThemeMode = "system" | "light" | "dark"

export interface ModelProfileView {
  provider: ProviderKind
  model: string
  base_url: string
  api_key_env?: string | null
  temperature?: number | null
  top_p?: number | null
  max_tokens?: number | null
  timeout_s?: number | null
  num_ctx_default?: number | null
  num_ctx_overrides: Record<string, number>
}

export interface RoleProfilesView {
  local: string
  judge: string
  frontier: string
}

export interface UiView {
  theme: ThemeMode
}

export interface PrivacyView {
  capture_transcripts: boolean
  transcript_max: number
}

export interface ModelPricingView {
  input_per_1m_tokens: number
  output_per_1m_tokens: number
}

export interface ControlsView {
  requests_per_minute?: number | null
  daily_request_quota?: number | null
  daily_token_quota?: number | null
  daily_cost_quota_usd?: number | null
  pricing: Record<string, ModelPricingView>
  audit_enabled: boolean
  audit_path: string
  audit_include_request_body: boolean
}

export type ConfigView = components["schemas"]["ConfigView"] & {
  model_profiles: Record<string, ModelProfileView>
  role_profiles: RoleProfilesView
  ui: UiView
  privacy: PrivacyView
  controls: ControlsView
}

const FALLBACK_JUDGE_PROMPT = ""

function normalizeConfig(config: ConfigView): ConfigView {
  return {
    ...config,
    judge: {
      ...config.judge,
      prompt: config.judge.prompt ?? FALLBACK_JUDGE_PROMPT,
    },
  }
}

type ConfigStatus = "idle" | "loading" | "ready" | "saving" | "error"

interface ConfigState {
  config: ConfigView | null
  status: ConfigStatus
  errorMessage: string | null
  restartRequired: boolean
}

interface ConfigActions {
  // Fetch /v1/config once. Idempotent — second call while ready is a no-op.
  // D-19 dependency: Trace narration reads `routing.threshold` at call site;
  // Stats threshold-what-if defaults to `routing.threshold` on mount.
  fetchConfig: () => Promise<void>
  saveConfig: (config: ConfigView) => Promise<void>
  // Test helper — resets to idle so a fresh fetch can be exercised in unit tests.
  _resetForTest: () => void
}

export const useConfigStore = create<ConfigState & ConfigActions>()(
  (set, get) => ({
    config: null,
    status: "idle",
    errorMessage: null,
    restartRequired: false,

    fetchConfig: async () => {
      if (get().status === "loading" || get().status === "ready") return
      set({ status: "loading", errorMessage: null })
      try {
        const res = await fetch("/v1/config")
        if (!res.ok) {
          set({
            status: "error",
            errorMessage: `Config fetch returned ${res.status}`,
          })
          return
        }
        const cfg = normalizeConfig((await res.json()) as ConfigView)
        set({ config: cfg, status: "ready", restartRequired: false })
      } catch (err) {
        // Pitfall #12 ethos: log class name only, never the error string content.
        const cls = err instanceof Error ? err.constructor.name : "UnknownError"
        console.warn(`[configStore] fetch /v1/config failed: ${cls}`)
        set({ status: "error", errorMessage: cls })
      }
    },

    saveConfig: async (config) => {
      set({ status: "saving", errorMessage: null })
      try {
        const res = await fetch("/v1/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
        })
        if (!res.ok) {
          set({
            status: "error",
            errorMessage: `Config save returned ${res.status}`,
          })
          return
        }
        const cfg = normalizeConfig((await res.json()) as ConfigView)
        set({
          config: cfg,
          status: "ready",
          restartRequired:
            res.headers.get("X-CoreThread-Restart-Required") === "true",
        })
      } catch (err) {
        const cls = err instanceof Error ? err.constructor.name : "UnknownError"
        console.warn(`[configStore] save /v1/config failed: ${cls}`)
        set({ status: "error", errorMessage: cls })
      }
    },

    _resetForTest: () =>
      set({
        config: null,
        status: "idle",
        errorMessage: null,
        restartRequired: false,
      }),
  }),
)
