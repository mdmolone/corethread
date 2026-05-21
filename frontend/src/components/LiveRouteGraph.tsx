import { useEffect, useMemo, useState } from "react"
import type { LucideIcon } from "lucide-react"
import { Activity, BrainCircuit, Cpu, Radio, Sparkles } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { HealthDot } from "@/components/HealthDot"
import type { ProviderState, UseHealthPollResult } from "@/hooks/useHealthPoll"
import {
  useRouteActivityStore,
  type RouteActivityEvent,
} from "@/store/routeActivityStore"
import { cn } from "@/lib/utils"

type RouteStage = RouteActivityEvent["stage"]
type NodeState = "idle" | "active" | "done" | "error" | "skipped"
type LinkState = "idle" | "active" | "done" | "skipped"

const ACTIVE_STAGES = new Set<RouteStage>([
  "received",
  "local_started",
  "judge_started",
  "frontier_started",
])

const STAGE_LABEL: Record<RouteStage, string> = {
  received: "Incoming",
  local_started: "Local evaluating",
  local_completed: "Local complete",
  local_error: "Local error",
  judge_started: "Judge reviewing",
  judge_completed: "Judge complete",
  judge_error: "Judge error",
  frontier_started: "Frontier active",
  frontier_completed: "Frontier complete",
  frontier_error: "Frontier error",
  completed: "Complete",
  error: "Blocked",
}

const NODE_STYLE: Record<
  NodeState,
  { ring: string; dot: string; text: string; status: string }
> = {
  idle: {
    ring: "border-border bg-background text-muted-foreground opacity-70",
    dot: "bg-muted",
    text: "text-muted-foreground",
    status: "text-muted-foreground",
  },
  active: {
    ring: "border-cyan-300 bg-cyan-500/15 text-cyan-200 shadow-[0_0_34px_rgba(34,211,238,0.55)] ring-2 ring-cyan-300/60",
    dot: "bg-cyan-400",
    text: "text-foreground",
    status: "bg-cyan-400 text-black",
  },
  done: {
    ring: "border-emerald-400 bg-emerald-500/10 text-emerald-300",
    dot: "bg-emerald-400",
    text: "text-foreground",
    status: "bg-emerald-500/15 text-emerald-300",
  },
  error: {
    ring: "border-red-400 bg-red-500/10 text-red-300",
    dot: "bg-red-400",
    text: "text-foreground",
    status: "bg-red-500/15 text-red-300",
  },
  skipped: {
    ring: "border-border bg-muted/40 text-muted-foreground",
    dot: "bg-muted-foreground/50",
    text: "text-muted-foreground",
    status: "bg-muted text-muted-foreground",
  },
}

const LINK_STYLE: Record<LinkState, string> = {
  idle: "bg-border",
  active:
    "h-1 rounded-full bg-cyan-300 shadow-[0_0_22px_rgba(34,211,238,0.65)] after:absolute after:inset-y-0 after:left-0 after:w-1/2 after:animate-pulse after:rounded-full after:bg-white/60",
  done: "bg-emerald-400/75",
  skipped: "bg-border opacity-40",
}

function providerState(
  healthResult: UseHealthPollResult,
  role: "local" | "frontier",
): ProviderState | "loading" {
  if (!healthResult.health || healthResult.status === "loading") {
    return "loading"
  }
  return healthResult.health.providers[role].state
}

function modelLabel(value: string | null | undefined): string {
  if (!value) return "not set"
  return value.length > 28 ? `${value.slice(0, 25)}...` : value
}

function elapsedLabel(startMs: number | null, nowMs: number): string | null {
  if (startMs === null) return null
  const seconds = Math.max(0, Math.floor((nowMs - startMs) / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${seconds % 60}s`
}

function RouteNode({
  icon: Icon,
  label,
  model,
  state,
  statusLabel,
  testId,
}: {
  icon: LucideIcon
  label: string
  model: string
  state: NodeState
  statusLabel: string | null
  testId: string
}) {
  const style = NODE_STYLE[state]
  return (
    <div
      className="min-w-0 text-center"
      data-testid={testId}
      data-state={state}
    >
      <div className="relative mx-auto flex h-16 w-16 items-center justify-center">
        {state === "active" ? (
          <span className="absolute h-14 w-14 animate-ping rounded-full border border-cyan-300/60" />
        ) : null}
        <div
          className={cn(
            "relative flex h-12 w-12 items-center justify-center rounded-full border transition-colors",
            state === "active" ? "scale-110" : "",
            style.ring,
          )}
        >
          <Icon
            className={cn("h-5 w-5", state === "active" ? "animate-pulse" : "")}
            aria-hidden="true"
          />
        </div>
      </div>
      <div className={cn("text-xs font-medium", style.text)}>{label}</div>
      <div className="text-muted-foreground mt-0.5 truncate text-[10px]">
        {model}
      </div>
      {statusLabel ? (
        <div
          className={cn(
            "mx-auto mt-2 w-fit rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-normal uppercase",
            style.status,
          )}
          data-testid={`${testId}-status`}
        >
          {statusLabel}
        </div>
      ) : null}
      <span
        className={cn("mx-auto mt-2 block h-1.5 w-1.5 rounded-full", style.dot)}
      />
    </div>
  )
}

function RouteLink({ state, testId }: { state: LinkState; testId: string }) {
  return (
    <div
      className="relative mt-7 h-px min-w-8 overflow-hidden rounded-full"
      data-testid={testId}
      data-state={state}
    >
      <div
        className={cn("absolute inset-0 transition-colors", LINK_STYLE[state])}
      />
    </div>
  )
}

export function LiveRouteGraph({
  healthResult,
}: {
  healthResult: UseHealthPollResult
}) {
  const [nowMs, setNowMs] = useState(() => Date.now())
  const events = useRouteActivityStore((state) => state.events)
  const latest = events.length > 0 ? events[events.length - 1] : null
  const requestEvents = latest
    ? events.filter((event) => event.request_id === latest.request_id)
    : []
  const stages = new Set(requestEvents.map((event) => event.stage))
  const latestStage = latest?.stage

  const hasLocalStart = stages.has("local_started")
  const hasLocalDone = stages.has("local_completed")
  const hasLocalError = stages.has("local_error")
  const hasJudgeStart = stages.has("judge_started")
  const hasJudgeDone = stages.has("judge_completed")
  const hasJudgeError = stages.has("judge_error")
  const hasFrontierStart = stages.has("frontier_started")
  const hasFrontierDone = stages.has("frontier_completed")
  const hasFrontierError = stages.has("frontier_error")
  const isActiveStage = latestStage ? ACTIVE_STAGES.has(latestStage) : false

  useEffect(() => {
    if (!isActiveStage) {
      setNowMs(Date.now())
      return
    }
    setNowMs(Date.now())
    const intervalId = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(intervalId)
  }, [isActiveStage, latestStage, latest?.request_id])

  const ingressState: NodeState = latest
    ? latestStage === "received"
      ? "active"
      : "done"
    : "idle"
  const localState: NodeState =
    latestStage === "local_started"
      ? "active"
      : hasLocalError
        ? "error"
        : hasLocalDone
          ? "done"
          : "idle"
  const judgeState: NodeState =
    latestStage === "judge_started"
      ? "active"
      : hasJudgeError
        ? "error"
        : hasJudgeDone
          ? "done"
          : hasLocalError
            ? "skipped"
            : "idle"
  const frontierState: NodeState =
    latestStage === "frontier_started"
      ? "active"
      : hasFrontierError
        ? "error"
        : hasFrontierDone
          ? "done"
          : latestStage === "completed" && latest?.pivoted === false
            ? "skipped"
            : "idle"

  const incomingLink: LinkState = hasLocalStart
    ? "done"
    : latestStage === "received"
      ? "active"
      : "idle"
  const localJudgeLink: LinkState = hasJudgeStart
    ? "done"
    : hasLocalDone
      ? "active"
      : hasLocalError
        ? "skipped"
        : "idle"
  const judgeFrontierLink: LinkState = hasFrontierStart
    ? "done"
    : hasJudgeDone && latest?.pivoted
      ? "active"
      : latestStage === "completed" && latest?.pivoted === false
        ? "skipped"
        : "idle"

  const confidence =
    latest?.confidence_score == null
      ? null
      : `conf ${latest.confidence_score.toFixed(2)}`
  const localHealth = providerState(healthResult, "local")
  const frontierHealth = providerState(healthResult, "frontier")
  const activeElapsed = isActiveStage
    ? elapsedLabel(latest?.ts_ms ?? null, nowMs)
    : null
  const nowLabel = latest
    ? isActiveStage
      ? `Now: ${STAGE_LABEL[latest.stage]}${activeElapsed ? ` · ${activeElapsed}` : ""}`
      : `Last: ${STAGE_LABEL[latest.stage]}`
    : "No route activity yet"

  const statusByState = useMemo(
    () =>
      ({
        idle: null,
        active: `working${activeElapsed ? ` ${activeElapsed}` : ""}`,
        done: "done",
        error: "error",
        skipped: "skipped",
      }) satisfies Record<NodeState, string | null>,
    [activeElapsed],
  )

  return (
    <section
      className="bg-card/55 rounded-md border p-4"
      data-testid="live-route-graph"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <Activity className="h-4 w-4 text-cyan-400" aria-hidden="true" />
          <span className="text-sm font-semibold">Live route</span>
          <Badge variant="outline" data-testid="live-route-stage">
            {latest ? STAGE_LABEL[latest.stage] : "Idle"}
          </Badge>
          {latest?.error_class ? (
            <Badge variant="secondary">{latest.error_class}</Badge>
          ) : null}
          {confidence ? (
            <span className="text-muted-foreground text-xs">{confidence}</span>
          ) : null}
        </div>
        <div className="text-muted-foreground flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1">
            <HealthDot
              label="route-local"
              state={localHealth}
              last_error={
                healthResult.health?.providers.local.last_error ?? null
              }
            />
            local
          </span>
          <span className="flex items-center gap-1">
            <HealthDot
              label="route-frontier"
              state={frontierHealth}
              last_error={
                healthResult.health?.providers.frontier.last_error ?? null
              }
            />
            frontier
          </span>
          {latest ? (
            <span className="font-mono text-[10px]">
              {latest.request_id.slice(0, 12)}
            </span>
          ) : null}
        </div>
      </div>
      <div
        className={cn(
          "mt-4 rounded-md border px-3 py-2 text-sm font-semibold",
          isActiveStage
            ? "border-cyan-400/60 bg-cyan-500/10 text-cyan-200 shadow-[0_0_18px_rgba(34,211,238,0.18)]"
            : "border-border bg-muted/25 text-muted-foreground",
        )}
        data-testid="live-route-now"
      >
        {nowLabel}
      </div>

      <div className="mt-5 grid grid-cols-[minmax(64px,auto)_minmax(28px,1fr)_minmax(64px,auto)_minmax(28px,1fr)_minmax(64px,auto)_minmax(28px,1fr)_minmax(64px,auto)] items-start gap-2">
        <RouteNode
          icon={Radio}
          label="Incoming"
          model="client"
          state={ingressState}
          statusLabel={statusByState[ingressState]}
          testId="route-node-incoming"
        />
        <RouteLink state={incomingLink} testId="route-link-incoming-local" />
        <RouteNode
          icon={Cpu}
          label="Local"
          model={modelLabel(latest?.selected_local_model)}
          state={localState}
          statusLabel={statusByState[localState]}
          testId="route-node-local"
        />
        <RouteLink state={localJudgeLink} testId="route-link-local-judge" />
        <RouteNode
          icon={BrainCircuit}
          label="Judge"
          model={modelLabel(latest?.judge_model)}
          state={judgeState}
          statusLabel={statusByState[judgeState]}
          testId="route-node-judge"
        />
        <RouteLink
          state={judgeFrontierLink}
          testId="route-link-judge-frontier"
        />
        <RouteNode
          icon={Sparkles}
          label="Frontier"
          model={modelLabel(latest?.frontier_model)}
          state={frontierState}
          statusLabel={statusByState[frontierState]}
          testId="route-node-frontier"
        />
      </div>
    </section>
  )
}
