import type { UseHealthPollResult } from "@/hooks/useHealthPoll"
import { HealthDot } from "@/components/HealthDot"

export function TopBar({
  healthResult,
}: {
  healthResult: UseHealthPollResult
}) {
  const { health, status } = healthResult

  if (status === "loading" || status === "idle" || !health) {
    return (
      <div
        className="text-muted-foreground flex items-center gap-2 text-xs"
        data-testid="top-bar"
      >
        <HealthDot label="local" state="loading" last_error={null} />
        <HealthDot label="frontier" state="loading" last_error={null} />
      </div>
    )
  }

  if (status === "paused") {
    return (
      <div
        className="text-muted-foreground flex items-center gap-2 text-xs"
        data-testid="top-bar"
      >
        <span className="text-[10px]">tab hidden — poll paused</span>
      </div>
    )
  }

  return (
    <div
      className="text-muted-foreground flex items-center gap-3 text-xs"
      data-testid="top-bar"
    >
      <span className="flex items-center gap-1">
        <HealthDot
          label="local"
          state={health.providers.local.state}
          last_error={health.providers.local.last_error}
        />
        local ({health.providers.local.kind})
      </span>
      <span className="flex items-center gap-1">
        <HealthDot
          label="frontier"
          state={health.providers.frontier.state}
          last_error={health.providers.frontier.last_error}
        />
        frontier ({health.providers.frontier.kind})
      </span>
      <span className="text-[10px]">v{health.version}</span>
    </div>
  )
}
