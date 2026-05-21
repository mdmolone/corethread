import { useEffect, useMemo, useState } from "react"
import type React from "react"
import { Plus, Power, RefreshCw, RotateCcw, Save, Trash2 } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  useConfigStore,
  type ConfigView as ConfigViewType,
  type ModelProfileView,
  type ProviderKind,
  type ThemeMode,
} from "@/store/configStore"

interface FieldProps {
  label: string
  value: React.ReactNode
  mono?: boolean
  dim?: boolean
}

type ProbeStatus = "testing" | "ok" | "error"
type RestartStatus = "idle" | "restarting" | "ready" | "error"

interface ProfileProbeResult {
  ok: boolean
  provider: ProviderKind
  models: string[]
  selected_model_available: boolean | null
  error_class: string | null
  message: string
}

interface ProfileProbeState {
  status: ProbeStatus
  message: string
  models: string[]
  selectedModelAvailable: boolean | null
  errorClass: string | null
}

function Field({ label, value, mono = false, dim = false }: FieldProps) {
  return (
    <div className="grid grid-cols-[120px_1fr] gap-2 py-1 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={`${mono ? "font-mono" : ""} ${dim ? "text-muted-foreground" : ""}`}
      >
        {value}
      </span>
    </div>
  )
}

const PROVIDERS: ProviderKind[] = ["lmstudio", "ollama", "openai", "openrouter"]
const THEMES: ThemeMode[] = ["system", "light", "dark"]

const controlClass =
  "h-9 w-full rounded-md border bg-background px-2 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring"
const labelClass = "space-y-1 text-xs font-medium text-muted-foreground"

function cloneConfig(config: ConfigViewType): ConfigViewType {
  return JSON.parse(JSON.stringify(config)) as ConfigViewType
}

function defaultBaseUrl(
  provider: ProviderKind,
  config: ConfigViewType,
): string {
  if (provider === "openrouter") return "https://openrouter.ai/api/v1"
  if (provider === "openai") return "https://api.openai.com/v1"
  if (provider === "ollama") return "http://localhost:11434"
  return config.local.base_url || "http://localhost:1234/v1"
}

function defaultApiKeyEnv(provider: ProviderKind): string | null {
  if (provider === "openrouter") return "OPENROUTER_API_KEY"
  if (provider === "openai") return "OPENAI_API_KEY"
  return null
}

function optionalNumber(value: string): number | null {
  if (value.trim() === "") return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function numberValue(value: number | null | undefined): string {
  return value == null ? "" : String(value)
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function syncLegacyBlocks(config: ConfigViewType): ConfigViewType {
  const next = cloneConfig(config)
  const local = next.model_profiles[next.role_profiles.local]
  const judge = next.model_profiles[next.role_profiles.judge]
  const frontier = next.model_profiles[next.role_profiles.frontier]

  if (local && (local.provider === "ollama" || local.provider === "lmstudio")) {
    next.local = {
      kind: local.provider,
      base_url: local.base_url,
      model: local.model,
      num_ctx_default: local.num_ctx_default ?? next.local.num_ctx_default,
      num_ctx_overrides: local.num_ctx_overrides,
    }
  }
  if (judge) {
    next.judge = {
      ...next.judge,
      model: judge.model,
      prompt: next.judge.prompt ?? "",
    }
  }
  if (
    frontier &&
    (frontier.provider === "openai" || frontier.provider === "openrouter")
  ) {
    next.frontier = {
      model: frontier.model,
      api_key_env:
        frontier.api_key_env ?? defaultApiKeyEnv(frontier.provider) ?? "",
      max_tokens: frontier.max_tokens ?? next.frontier.max_tokens,
    }
  }
  return next
}

export function ConfigView() {
  const config = useConfigStore((s) => s.config)
  const status = useConfigStore((s) => s.status)
  const errorMessage = useConfigStore((s) => s.errorMessage)
  const fetchConfig = useConfigStore((s) => s.fetchConfig)
  const saveConfig = useConfigStore((s) => s.saveConfig)
  const restartRequired = useConfigStore((s) => s.restartRequired)
  const [draft, setDraft] = useState<ConfigViewType | null>(null)
  const [overridesText, setOverridesText] = useState<Record<string, string>>({})
  const [profileIdText, setProfileIdText] = useState<Record<string, string>>({})
  const [pricingText, setPricingText] = useState<string>("{}")
  const [probeResults, setProbeResults] = useState<
    Record<string, ProfileProbeState>
  >({})
  const [parseError, setParseError] = useState<string | null>(null)
  const [restartStatus, setRestartStatus] = useState<RestartStatus>("idle")
  const [restartMessage, setRestartMessage] = useState<string | null>(null)

  useEffect(() => {
    if (!config) return
    setDraft(cloneConfig(config))
    setOverridesText(
      Object.fromEntries(
        Object.entries(config.model_profiles).map(([profileId, profile]) => [
          profileId,
          JSON.stringify(profile.num_ctx_overrides ?? {}),
        ]),
      ),
    )
    setProfileIdText(
      Object.fromEntries(
        Object.keys(config.model_profiles).map((profileId) => [
          profileId,
          profileId,
        ]),
      ),
    )
    setPricingText(JSON.stringify(config.controls.pricing ?? {}))
    setProbeResults({})
    setParseError(null)
  }, [config])

  const profileIds = useMemo(
    () => Object.keys(draft?.model_profiles ?? {}),
    [draft],
  )

  function updateDraft(mutator: (next: ConfigViewType) => void) {
    setDraft((current) => {
      if (!current) return current
      const next = cloneConfig(current)
      mutator(next)
      return syncLegacyBlocks(next)
    })
  }

  function updateProfile(
    profileId: string,
    patch: Partial<ModelProfileView>,
    options: { clearProbe?: boolean } = {},
  ) {
    updateDraft((next) => {
      const current = next.model_profiles[profileId]
      if (!current) return
      next.model_profiles[profileId] = {
        ...current,
        ...patch,
      }
    })
    if (options.clearProbe ?? true) {
      setProbeResults((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([id]) => id !== profileId),
        ),
      )
    }
  }

  function commitProfileRename(profileId: string) {
    if (!draft) return
    const nextProfileId = (profileIdText[profileId] ?? profileId).trim()
    if (nextProfileId === profileId) {
      setProfileIdText((current) => ({ ...current, [profileId]: profileId }))
      return
    }
    if (!nextProfileId) {
      setParseError("Profile id cannot be blank.")
      return
    }
    if (!/^[A-Za-z0-9._-]+$/.test(nextProfileId)) {
      setParseError(
        "Profile id can use letters, numbers, dots, underscores, and dashes.",
      )
      return
    }
    if (draft.model_profiles[nextProfileId]) {
      setParseError(`Profile id "${nextProfileId}" already exists.`)
      return
    }

    updateDraft((next) => {
      next.model_profiles = Object.fromEntries(
        Object.entries(next.model_profiles).map(([id, profile]) =>
          id === profileId ? [nextProfileId, profile] : [id, profile],
        ),
      )
      ;(["local", "judge", "frontier"] as const).forEach((role) => {
        if (next.role_profiles[role] === profileId) {
          next.role_profiles[role] = nextProfileId
        }
      })
    })
    setOverridesText((current) => {
      const renamed = Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== profileId),
      )
      return { ...renamed, [nextProfileId]: current[profileId] ?? "{}" }
    })
    setProfileIdText((current) => {
      const renamed = Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== profileId),
      )
      return { ...renamed, [nextProfileId]: nextProfileId }
    })
    setProbeResults((current) => {
      const renamed = Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== profileId),
      )
      return current[profileId]
        ? { ...renamed, [nextProfileId]: current[profileId] }
        : renamed
    })
    setParseError(null)
  }

  function addProfile() {
    if (!draft) return
    let index = profileIds.length + 1
    let profileId = `profile-${index}`
    while (draft.model_profiles[profileId]) {
      index += 1
      profileId = `profile-${index}`
    }
    updateDraft((next) => {
      next.model_profiles[profileId] = {
        provider: "lmstudio",
        model: "",
        base_url: defaultBaseUrl("lmstudio", next),
        api_key_env: null,
        temperature: null,
        top_p: null,
        max_tokens: null,
        timeout_s: null,
        num_ctx_default: next.local.num_ctx_default,
        num_ctx_overrides: {},
      }
    })
    setOverridesText((current) => ({ ...current, [profileId]: "{}" }))
    setProfileIdText((current) => ({ ...current, [profileId]: profileId }))
  }

  function removeProfile(profileId: string) {
    updateDraft((next) => {
      next.model_profiles = Object.fromEntries(
        Object.entries(next.model_profiles).filter(([id]) => id !== profileId),
      )
    })
    setOverridesText((current) => {
      return Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== profileId),
      )
    })
    setProfileIdText((current) => {
      return Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== profileId),
      )
    })
    setProbeResults((current) => {
      return Object.fromEntries(
        Object.entries(current).filter(([id]) => id !== profileId),
      )
    })
  }

  async function probeProfile(profileId: string) {
    const profile = editorConfig.model_profiles[profileId]
    if (!profile) return
    setProbeResults((current) => ({
      ...current,
      [profileId]: {
        status: "testing",
        message: "Testing",
        models: current[profileId]?.models ?? [],
        selectedModelAvailable: null,
        errorClass: null,
      },
    }))
    try {
      const res = await fetch("/v1/config/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(profile),
      })
      if (!res.ok) {
        setProbeResults((current) => ({
          ...current,
          [profileId]: {
            status: "error",
            message: `Probe returned ${res.status}`,
            models: [],
            selectedModelAvailable: null,
            errorClass: "HTTPError",
          },
        }))
        return
      }
      const result = (await res.json()) as ProfileProbeResult
      setProbeResults((current) => ({
        ...current,
        [profileId]: {
          status: result.ok ? "ok" : "error",
          message: result.message,
          models: result.models,
          selectedModelAvailable: result.selected_model_available,
          errorClass: result.error_class,
        },
      }))
      const firstModel = result.models[0]
      if (
        result.ok &&
        !profile.model &&
        result.models.length === 1 &&
        firstModel
      ) {
        updateProfile(profileId, { model: firstModel }, { clearProbe: false })
      }
    } catch (err) {
      const cls = err instanceof Error ? err.constructor.name : "UnknownError"
      setProbeResults((current) => ({
        ...current,
        [profileId]: {
          status: "error",
          message: cls,
          models: [],
          selectedModelAvailable: null,
          errorClass: cls,
        },
      }))
    }
  }

  async function waitForRestart() {
    await sleep(1200)
    for (let attempt = 0; attempt < 90; attempt += 1) {
      try {
        const res = await fetch("/health", { cache: "no-store" })
        if (res.ok) {
          setRestartStatus("ready")
          setRestartMessage("Restarted. Reloading")
          window.location.reload()
          return
        }
      } catch {
        // The service is expected to disappear briefly while restarting.
      }
      await sleep(1000)
    }
    setRestartStatus("error")
    setRestartMessage(
      "Restart requested, but CoreThread did not come back yet.",
    )
  }

  async function restartCoreThread() {
    if (
      !window.confirm(
        "Restart CoreThread now? Active requests may be interrupted briefly.",
      )
    ) {
      return
    }
    setRestartStatus("restarting")
    setRestartMessage("Restart requested")
    try {
      const res = await fetch("/v1/admin/restart", { method: "POST" })
      if (!res.ok) {
        setRestartStatus("error")
        setRestartMessage(`Restart returned ${res.status}`)
        return
      }
      setRestartMessage("Waiting for CoreThread to come back")
      void waitForRestart()
    } catch (err) {
      const cls = err instanceof Error ? err.constructor.name : "UnknownError"
      setRestartStatus("error")
      setRestartMessage(cls)
    }
  }

  function handleOverridesChange(profileId: string, value: string) {
    setOverridesText((current) => ({ ...current, [profileId]: value }))
    try {
      const parsed = value.trim() === "" ? {} : JSON.parse(value)
      if (
        parsed === null ||
        Array.isArray(parsed) ||
        typeof parsed !== "object" ||
        Object.values(parsed).some((entry) => typeof entry !== "number")
      ) {
        throw new Error("InvalidOverrides")
      }
      setParseError(null)
      updateProfile(profileId, {
        num_ctx_overrides: parsed as Record<string, number>,
      })
    } catch {
      setParseError('num_ctx_overrides must be JSON like {"model": 16384}.')
    }
  }

  function handlePricingChange(value: string) {
    setPricingText(value)
    try {
      const parsed = value.trim() === "" ? {} : JSON.parse(value)
      if (
        parsed === null ||
        Array.isArray(parsed) ||
        typeof parsed !== "object" ||
        Object.values(parsed).some((entry) => {
          if (
            entry === null ||
            Array.isArray(entry) ||
            typeof entry !== "object"
          ) {
            return true
          }
          const rate = entry as Record<string, unknown>
          return (
            typeof rate.input_per_1m_tokens !== "number" ||
            typeof rate.output_per_1m_tokens !== "number"
          )
        })
      ) {
        throw new Error("InvalidPricing")
      }
      setParseError(null)
      updateDraft((next) => {
        next.controls.pricing = parsed as ConfigViewType["controls"]["pricing"]
      })
    } catch {
      setParseError(
        'pricing must be JSON like {"gpt-4o":{"input_per_1m_tokens":2.5,"output_per_1m_tokens":10}}.',
      )
    }
  }

  function handleSave(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!draft || parseError) return
    void saveConfig(syncLegacyBlocks(draft))
  }

  if (status === "loading" || status === "idle") {
    return (
      <div
        className="text-muted-foreground text-sm"
        data-testid="config-loading"
      >
        Loading config…
      </div>
    )
  }
  if (status === "error" || !config) {
    return (
      <Card>
        <CardContent className="p-4">
          <Badge variant="destructive">Config fetch failed</Badge>
          <p
            className="text-muted-foreground mt-2 text-sm"
            data-testid="config-error"
          >
            Unable to read /v1/config ({errorMessage ?? "unknown error"}).
          </p>
          <Button
            type="button"
            size="sm"
            className="mt-3"
            onClick={() => void fetchConfig()}
          >
            Retry
          </Button>
        </CardContent>
      </Card>
    )
  }

  const editorConfig = draft ?? config
  const editorProfileIds = Object.keys(editorConfig.model_profiles)
  const assignedProfileIds = new Set(Object.values(editorConfig.role_profiles))

  // Type-narrowed below this point.
  return (
    <div className="space-y-4" data-testid="config-view">
      <form onSubmit={handleSave}>
        <Card data-testid="config-editor">
          <CardHeader className="flex flex-row items-center justify-between gap-3">
            <CardTitle>Config editor</CardTitle>
            <div className="flex items-center gap-2">
              {restartRequired ? (
                <Badge variant="secondary">Restart required</Badge>
              ) : null}
              {restartRequired ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={restartStatus === "restarting"}
                  onClick={() => void restartCoreThread()}
                >
                  <Power className="mr-2 h-4 w-4" />
                  {restartStatus === "restarting" ? "Restarting" : "Restart"}
                </Button>
              ) : null}
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  setDraft(cloneConfig(config))
                  setProfileIdText(
                    Object.fromEntries(
                      Object.keys(config.model_profiles).map((profileId) => [
                        profileId,
                        profileId,
                      ]),
                    ),
                  )
                  setPricingText(JSON.stringify(config.controls.pricing ?? {}))
                  setProbeResults({})
                  setParseError(null)
                }}
              >
                <RotateCcw className="mr-2 h-4 w-4" />
                Reset
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addProfile}
              >
                <Plus className="mr-2 h-4 w-4" />
                Add profile
              </Button>
              <Button
                type="submit"
                size="sm"
                disabled={status === "saving" || !!parseError}
              >
                <Save className="mr-2 h-4 w-4" />
                {status === "saving" ? "Saving" : "Save"}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {parseError ? (
              <p
                className="text-destructive text-sm"
                data-testid="config-parse-error"
              >
                {parseError}
              </p>
            ) : null}
            {restartMessage ? (
              <p
                className={`text-sm ${
                  restartStatus === "error"
                    ? "text-destructive"
                    : "text-muted-foreground"
                }`}
                data-testid="restart-message"
              >
                {restartMessage}
              </p>
            ) : null}
            <div className="grid gap-3 md:grid-cols-6">
              {(["local", "judge", "frontier"] as const).map((role) => (
                <label key={role} className={labelClass}>
                  <span>{role}</span>
                  <select
                    className={controlClass}
                    value={editorConfig.role_profiles[role]}
                    onChange={(event) =>
                      updateDraft((next) => {
                        next.role_profiles[role] = event.target.value
                      })
                    }
                  >
                    {editorProfileIds.map((profileId) => (
                      <option key={profileId} value={profileId}>
                        {profileId}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
              <label className={labelClass}>
                <span>theme</span>
                <select
                  className={controlClass}
                  value={editorConfig.ui.theme}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.ui.theme = event.target.value as ThemeMode
                    })
                  }
                >
                  {THEMES.map((theme) => (
                    <option key={theme} value={theme}>
                      {theme}
                    </option>
                  ))}
                </select>
              </label>
              <label className={labelClass}>
                <span>capture_transcripts</span>
                <span className="border-input bg-background text-foreground flex h-9 items-center gap-2 rounded-md border px-2 text-sm">
                  <input
                    type="checkbox"
                    checked={editorConfig.privacy.capture_transcripts}
                    onChange={(event) =>
                      updateDraft((next) => {
                        next.privacy.capture_transcripts = event.target.checked
                      })
                    }
                  />
                  <span>
                    {editorConfig.privacy.capture_transcripts ? "on" : "off"}
                  </span>
                </span>
              </label>
              <label className={labelClass}>
                <span>transcript_max</span>
                <input
                  className={controlClass}
                  type="number"
                  min="1"
                  step="1"
                  value={editorConfig.privacy.transcript_max}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.privacy.transcript_max =
                        optionalNumber(event.target.value) ?? 1
                    })
                  }
                />
              </label>
            </div>
            <div className="grid gap-3 md:grid-cols-[180px_1fr]">
              <label className={labelClass}>
                <span>threshold</span>
                <input
                  className={controlClass}
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={editorConfig.routing.threshold}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.routing.threshold =
                        optionalNumber(event.target.value) ?? 0
                    })
                  }
                />
              </label>
              <label className={`${labelClass} md:col-span-2`}>
                <span>judge_prompt</span>
                <textarea
                  className="bg-background text-foreground focus:ring-ring min-h-40 w-full rounded-md border px-2 py-2 font-mono text-xs outline-none focus:ring-2"
                  value={editorConfig.judge.prompt ?? ""}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.judge.prompt = event.target.value
                    })
                  }
                />
              </label>
              <label className={labelClass}>
                <span>constraint_prompt</span>
                <textarea
                  className="bg-background text-foreground focus:ring-ring min-h-20 w-full rounded-md border px-2 py-2 text-sm outline-none focus:ring-2"
                  value={editorConfig.routing.constraint_prompt}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.routing.constraint_prompt = event.target.value
                    })
                  }
                />
              </label>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <label className={labelClass}>
                <span>requests_per_minute</span>
                <input
                  className={controlClass}
                  type="number"
                  min="1"
                  step="1"
                  value={numberValue(editorConfig.controls.requests_per_minute)}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.controls.requests_per_minute = optionalNumber(
                        event.target.value,
                      )
                    })
                  }
                />
              </label>
              <label className={labelClass}>
                <span>daily_request_quota</span>
                <input
                  className={controlClass}
                  type="number"
                  min="1"
                  step="1"
                  value={numberValue(editorConfig.controls.daily_request_quota)}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.controls.daily_request_quota = optionalNumber(
                        event.target.value,
                      )
                    })
                  }
                />
              </label>
              <label className={labelClass}>
                <span>daily_token_quota</span>
                <input
                  className={controlClass}
                  type="number"
                  min="1"
                  step="1"
                  value={numberValue(editorConfig.controls.daily_token_quota)}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.controls.daily_token_quota = optionalNumber(
                        event.target.value,
                      )
                    })
                  }
                />
              </label>
              <label className={labelClass}>
                <span>daily_cost_quota_usd</span>
                <input
                  className={controlClass}
                  type="number"
                  min="0"
                  step="0.01"
                  value={numberValue(
                    editorConfig.controls.daily_cost_quota_usd,
                  )}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.controls.daily_cost_quota_usd = optionalNumber(
                        event.target.value,
                      )
                    })
                  }
                />
              </label>
              <label className={labelClass}>
                <span>audit_enabled</span>
                <span className="border-input bg-background text-foreground flex h-9 items-center gap-2 rounded-md border px-2 text-sm">
                  <input
                    type="checkbox"
                    aria-label="audit_enabled"
                    checked={editorConfig.controls.audit_enabled}
                    onChange={(event) =>
                      updateDraft((next) => {
                        next.controls.audit_enabled = event.target.checked
                      })
                    }
                  />
                  <span>
                    {editorConfig.controls.audit_enabled ? "on" : "off"}
                  </span>
                </span>
              </label>
              <label className={labelClass}>
                <span>audit_bodies</span>
                <span className="border-input bg-background text-foreground flex h-9 items-center gap-2 rounded-md border px-2 text-sm">
                  <input
                    type="checkbox"
                    aria-label="audit_include_request_body"
                    checked={editorConfig.controls.audit_include_request_body}
                    onChange={(event) =>
                      updateDraft((next) => {
                        next.controls.audit_include_request_body =
                          event.target.checked
                      })
                    }
                  />
                  <span>
                    {editorConfig.controls.audit_include_request_body
                      ? "on"
                      : "off"}
                  </span>
                </span>
              </label>
              <label className={`${labelClass} md:col-span-2`}>
                <span>audit_path</span>
                <input
                  className={controlClass}
                  value={editorConfig.controls.audit_path}
                  onChange={(event) =>
                    updateDraft((next) => {
                      next.controls.audit_path = event.target.value
                    })
                  }
                />
              </label>
              <label className={`${labelClass} md:col-span-4`}>
                <span>pricing</span>
                <textarea
                  className="bg-background text-foreground focus:ring-ring min-h-20 w-full rounded-md border px-2 py-2 font-mono text-sm outline-none focus:ring-2"
                  value={pricingText}
                  onChange={(event) => handlePricingChange(event.target.value)}
                />
              </label>
            </div>
            <div className="grid gap-3">
              {editorProfileIds.map((profileId) => {
                const profile = editorConfig.model_profiles[profileId]
                if (!profile) return null
                const cloudProfile =
                  profile.provider === "openai" ||
                  profile.provider === "openrouter"
                const probe = probeResults[profileId]
                const modelOptions = Array.from(
                  new Set([
                    ...(profile.model ? [profile.model] : []),
                    ...(probe?.models ?? []),
                  ]),
                )
                return (
                  <Card
                    key={profileId}
                    data-testid={`config-profile-${profileId}`}
                  >
                    <CardHeader className="flex flex-row items-end justify-between gap-3 py-3">
                      <label className={`${labelClass} min-w-0 flex-1`}>
                        <span>profile_id</span>
                        <input
                          className={controlClass}
                          aria-label={`Profile id for ${profileId}`}
                          value={profileIdText[profileId] ?? profileId}
                          onChange={(event) =>
                            setProfileIdText((current) => ({
                              ...current,
                              [profileId]: event.target.value,
                            }))
                          }
                          onBlur={() => commitProfileRename(profileId)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              event.preventDefault()
                              commitProfileRename(profileId)
                            }
                          }}
                        />
                      </label>
                      <div className="flex items-center gap-2">
                        {probe ? (
                          <Badge
                            variant={
                              probe.status === "ok" ? "secondary" : "outline"
                            }
                            data-testid={`probe-status-${profileId}`}
                          >
                            {probe.status === "testing"
                              ? "testing"
                              : probe.status === "ok"
                                ? "connected"
                                : "failed"}
                          </Badge>
                        ) : null}
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          disabled={probe?.status === "testing"}
                          onClick={() => void probeProfile(profileId)}
                          aria-label={`Test connection for ${profileId}`}
                        >
                          <RefreshCw
                            className={`mr-2 h-4 w-4 ${
                              probe?.status === "testing" ? "animate-spin" : ""
                            }`}
                          />
                          Test
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          disabled={assignedProfileIds.has(profileId)}
                          onClick={() => removeProfile(profileId)}
                          aria-label={`Remove ${profileId}`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent className="grid gap-3 md:grid-cols-4">
                      {probe ? (
                        <div
                          className={`text-sm ${
                            probe.status === "error"
                              ? "text-destructive"
                              : "text-muted-foreground"
                          } md:col-span-4`}
                          data-testid={`probe-message-${profileId}`}
                        >
                          {probe.message}
                          {probe.errorClass ? ` (${probe.errorClass})` : ""}
                        </div>
                      ) : null}
                      <label className={labelClass}>
                        <span>provider</span>
                        <select
                          className={controlClass}
                          value={profile.provider}
                          onChange={(event) => {
                            const provider = event.target.value as ProviderKind
                            updateProfile(profileId, {
                              provider,
                              base_url: defaultBaseUrl(provider, editorConfig),
                              api_key_env: defaultApiKeyEnv(provider),
                            })
                          }}
                        >
                          {PROVIDERS.map((provider) => (
                            <option key={provider} value={provider}>
                              {provider}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className={labelClass}>
                        <span>model</span>
                        {modelOptions.length > 0 && probe?.models.length ? (
                          <select
                            className={controlClass}
                            aria-label={`Available models for ${profileId}`}
                            value={profile.model}
                            onChange={(event) =>
                              updateProfile(
                                profileId,
                                {
                                  model: event.target.value,
                                },
                                { clearProbe: false },
                              )
                            }
                          >
                            {!profile.model ? (
                              <option value="">Select model</option>
                            ) : null}
                            {modelOptions.map((model) => (
                              <option key={model} value={model}>
                                {model}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            className={controlClass}
                            value={profile.model}
                            onChange={(event) =>
                              updateProfile(profileId, {
                                model: event.target.value,
                              })
                            }
                          />
                        )}
                      </label>
                      <label className={labelClass}>
                        <span>base_url</span>
                        <input
                          className={controlClass}
                          value={profile.base_url}
                          onChange={(event) =>
                            updateProfile(profileId, {
                              base_url: event.target.value,
                            })
                          }
                        />
                      </label>
                      <label className={labelClass}>
                        <span>api_key_env</span>
                        <input
                          className={controlClass}
                          disabled={!cloudProfile}
                          value={profile.api_key_env ?? ""}
                          onChange={(event) =>
                            updateProfile(profileId, {
                              api_key_env: event.target.value || null,
                            })
                          }
                        />
                      </label>
                      <label className={labelClass}>
                        <span>temperature</span>
                        <input
                          className={controlClass}
                          type="number"
                          min="0"
                          max="2"
                          step="0.05"
                          value={numberValue(profile.temperature)}
                          onChange={(event) =>
                            updateProfile(profileId, {
                              temperature: optionalNumber(event.target.value),
                            })
                          }
                        />
                      </label>
                      <label className={labelClass}>
                        <span>top_p</span>
                        <input
                          className={controlClass}
                          type="number"
                          min="0"
                          max="1"
                          step="0.05"
                          value={numberValue(profile.top_p)}
                          onChange={(event) =>
                            updateProfile(profileId, {
                              top_p: optionalNumber(event.target.value),
                            })
                          }
                        />
                      </label>
                      <label className={labelClass}>
                        <span>max_tokens</span>
                        <input
                          className={controlClass}
                          type="number"
                          min="1"
                          step="1"
                          value={numberValue(profile.max_tokens)}
                          onChange={(event) =>
                            updateProfile(profileId, {
                              max_tokens: optionalNumber(event.target.value),
                            })
                          }
                        />
                      </label>
                      <label className={labelClass}>
                        <span>timeout_s</span>
                        <input
                          className={controlClass}
                          type="number"
                          min="1"
                          step="1"
                          value={numberValue(profile.timeout_s)}
                          onChange={(event) =>
                            updateProfile(profileId, {
                              timeout_s: optionalNumber(event.target.value),
                            })
                          }
                        />
                      </label>
                      <label className={labelClass}>
                        <span>num_ctx_default</span>
                        <input
                          className={controlClass}
                          type="number"
                          min="1"
                          step="1"
                          value={numberValue(profile.num_ctx_default)}
                          onChange={(event) =>
                            updateProfile(profileId, {
                              num_ctx_default: optionalNumber(
                                event.target.value,
                              ),
                            })
                          }
                        />
                      </label>
                      <label className={`${labelClass} md:col-span-3`}>
                        <span>num_ctx_overrides</span>
                        <input
                          className={controlClass}
                          value={overridesText[profileId] ?? "{}"}
                          onChange={(event) =>
                            handleOverridesChange(profileId, event.target.value)
                          }
                        />
                      </label>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </CardContent>
        </Card>
      </form>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* D-24 LOCAL card */}
        <Card data-testid="config-card-local">
          <CardHeader>
            <CardTitle>Local provider</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <Field
              label="kind"
              value={<Badge variant="outline">{config.local.kind}</Badge>}
            />
            <Field label="url" value={config.local.base_url} mono />
            <Field label="model" value={config.local.model} mono />
            <Field
              label="num_ctx_default"
              value={config.local.num_ctx_default}
            />
            <Field
              label="num_ctx_overrides"
              value={
                Object.keys(config.local.num_ctx_overrides).length === 0 ? (
                  <span className="text-muted-foreground">(none)</span>
                ) : (
                  <code className="text-xs">
                    {JSON.stringify(config.local.num_ctx_overrides)}
                  </code>
                )
              }
              mono
            />
          </CardContent>
        </Card>

        {/* D-24 JUDGE card */}
        <Card data-testid="config-card-judge">
          <CardHeader>
            <CardTitle>Judge</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Field label="model" value={config.judge.model} mono />
            <div>
              <div className="text-muted-foreground mb-1 flex items-center gap-2 text-sm">
                <span>prompt</span>
                <Badge variant="secondary" className="text-[10px]">
                  sent to the judge before every grade
                </Badge>
                <span className="ml-auto text-[10px]">
                  {(config.judge.prompt ?? "").length} chars
                </span>
              </div>
              <ScrollArea
                className="bg-muted h-36 rounded-md border p-2"
                data-testid="judge-prompt-block"
              >
                <pre className="text-xs whitespace-pre-wrap">
                  {config.judge.prompt ?? ""}
                </pre>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>

        {/* D-24 FRONTIER card — CFG-02 + CFG-03 binding render */}
        <Card data-testid="config-card-frontier">
          <CardHeader>
            <CardTitle>Frontier</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <Field label="model" value={config.frontier.model} mono />
            {/* CFG-02 rendering pin (D-24 / D-25): the env-var NAME ONLY, never the resolved value. */}
            <Field
              label="api_key_env"
              value={
                <code
                  className="bg-muted rounded px-1.5 py-0.5 text-xs"
                  data-testid="frontier-api-key-pill"
                >
                  {config.frontier.api_key_env}
                </code>
              }
            />
            {/* CFG-03 cost-lever (D-24 + DF-11) */}
            <Field
              label="max_tokens"
              value={
                <span className="flex items-center gap-2">
                  <span className="font-mono">
                    {config.frontier.max_tokens}
                  </span>
                  <Badge
                    variant="secondary"
                    className="text-[10px]"
                    data-testid="max-tokens-annotation"
                  >
                    this caps every frontier response, regardless of
                    caller&apos;s max_tokens
                  </Badge>
                </span>
              }
            />
          </CardContent>
        </Card>

        {/* D-24 ROUTING card — CFG-03 constraint_prompt + threshold */}
        <Card data-testid="config-card-routing">
          <CardHeader>
            <CardTitle>Routing</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Field
              label="threshold"
              value={
                <span className="font-mono">
                  {config.routing.threshold.toFixed(2)}
                </span>
              }
            />
            <div>
              <div className="text-muted-foreground mb-1 flex items-center gap-2 text-sm">
                <span>constraint_prompt</span>
                <Badge
                  variant="secondary"
                  className="text-[10px]"
                  data-testid="constraint-prompt-annotation"
                >
                  this prompt prepends to every frontier call
                </Badge>
                <span className="ml-auto text-[10px]">
                  {config.routing.constraint_prompt.length} chars
                </span>
              </div>
              <ScrollArea
                className="bg-muted h-32 rounded-md border p-2"
                data-testid="constraint-prompt-block"
              >
                <pre className="text-xs whitespace-pre-wrap">
                  {config.routing.constraint_prompt}
                </pre>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>

        <Card data-testid="config-card-privacy">
          <CardHeader>
            <CardTitle>Privacy</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <Field
              label="transcripts"
              value={
                <Badge
                  variant={
                    config.privacy.capture_transcripts ? "secondary" : "outline"
                  }
                >
                  {config.privacy.capture_transcripts ? "enabled" : "disabled"}
                </Badge>
              }
            />
            <Field
              label="transcript_max"
              value={config.privacy.transcript_max}
            />
          </CardContent>
        </Card>

        <Card data-testid="config-card-controls">
          <CardHeader>
            <CardTitle>Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <Field
              label="rate"
              value={
                config.controls.requests_per_minute == null
                  ? "unlimited"
                  : `${config.controls.requests_per_minute}/min`
              }
            />
            <Field
              label="requests"
              value={config.controls.daily_request_quota ?? "unlimited"}
            />
            <Field
              label="tokens"
              value={config.controls.daily_token_quota ?? "unlimited"}
            />
            <Field
              label="cost"
              value={
                config.controls.daily_cost_quota_usd == null
                  ? "unlimited"
                  : `$${config.controls.daily_cost_quota_usd.toFixed(2)}/day`
              }
            />
            <Field
              label="audit"
              value={
                <Badge
                  variant={
                    config.controls.audit_enabled ? "secondary" : "outline"
                  }
                >
                  {config.controls.audit_enabled ? "enabled" : "disabled"}
                </Badge>
              }
            />
            <Field label="audit_path" value={config.controls.audit_path} mono />
            <Field
              label="pricing"
              value={`${Object.keys(config.controls.pricing).length} model rate${
                Object.keys(config.controls.pricing).length === 1 ? "" : "s"
              }`}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
