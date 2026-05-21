import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip"

interface LatencyBarProps {
  local_latency_ms: number
  judge_latency_ms: number
  frontier_latency_ms: number | null
  scaleMs?: number
}

/**
 * D-10: 3-segment stacked horizontal bar via CSS flexbox. The full track is a
 * fixed wall-clock scale, so rows can be compared by actual elapsed time rather
 * than being normalized to each row's own total.
 *
 * Dimensions: ~15 LOC of JSX. Recharts BarChart-per-row was rejected (D-10).
 */
const DEFAULT_SCALE_MS = 120_000

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 10_000) return `${(ms / 1000).toFixed(2)}s`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

export function LatencyBar({
  local_latency_ms,
  judge_latency_ms,
  frontier_latency_ms,
  scaleMs = DEFAULT_SCALE_MS,
}: LatencyBarProps) {
  const f = frontier_latency_ms ?? 0
  const total = local_latency_ms + judge_latency_ms + f
  if (total === 0) return null
  const scale = Math.max(scaleMs, 1)
  const segmentStyle = (segment: number) => {
    const basis = `${(Math.max(segment, 0) / scale) * 100}%`
    return { flex: `0 0 ${basis}`, width: basis }
  }
  const totalLabel = `${formatDuration(total)} / ${formatDuration(scale)}`
  const overScale = total > scale

  return (
    <TooltipProvider>
      <div className="space-y-1">
        <div className="text-muted-foreground flex items-center justify-between text-[11px] tabular-nums">
          <span>total {formatDuration(total)}</span>
          <span>scale {formatDuration(scale)}</span>
        </div>
        <div
          className="bg-muted/40 relative h-2 w-full overflow-hidden rounded-sm"
          data-testid="latency-bar"
          aria-label={`Latency timeline: ${totalLabel}`}
        >
          <div className="flex h-full w-full">
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className="bg-blue-500"
                  style={segmentStyle(local_latency_ms)}
                  data-testid="latency-segment-local"
                />
              </TooltipTrigger>
              <TooltipContent>local: {local_latency_ms}ms</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className="bg-amber-500"
                  style={segmentStyle(judge_latency_ms)}
                  data-testid="latency-segment-judge"
                />
              </TooltipTrigger>
              <TooltipContent>judge: {judge_latency_ms}ms</TooltipContent>
            </Tooltip>
            {f > 0 && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="bg-purple-500"
                    style={segmentStyle(f)}
                    data-testid="latency-segment-frontier"
                  />
                </TooltipTrigger>
                <TooltipContent>frontier: {f}ms</TooltipContent>
              </Tooltip>
            )}
          </div>
          {overScale && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className="absolute inset-y-0 right-0 w-1 bg-red-500"
                  data-testid="latency-over-scale"
                />
              </TooltipTrigger>
              <TooltipContent>
                total exceeded scale: {formatDuration(total)}
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        {overScale && (
          <div
            className="text-destructive text-[11px] tabular-nums"
            data-testid="latency-over-scale-label"
          >
            over scale by {formatDuration(total - scale)}
          </div>
        )}
      </div>
    </TooltipProvider>
  )
}
