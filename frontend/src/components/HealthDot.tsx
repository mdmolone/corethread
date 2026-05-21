import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip"
import type { ProviderState } from "@/hooks/useHealthPoll"

interface HealthDotProps {
  label: string // "local" | "frontier"
  state: ProviderState | "loading" // "loading" while the first poll is in flight
  last_error: string | null
}

const STATE_COLOR: Record<ProviderState | "loading", string> = {
  // D-26 verbatim mapping:
  ready: "bg-green-500",
  warming: "bg-amber-500",
  unhealthy: "bg-red-500",
  loading: "bg-gray-400",
}

export function HealthDot({ label, state, last_error }: HealthDotProps) {
  const color = STATE_COLOR[state]
  const dot = (
    <span
      className={`inline-block h-2.5 w-2.5 rounded-full ${color}`}
      data-testid={`health-dot-${label}`}
      data-state={state}
      aria-label={`${label}: ${state}`}
    />
  )

  if (state === "unhealthy" && last_error) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{dot}</TooltipTrigger>
          <TooltipContent>
            {label}: {last_error}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  return dot
}
