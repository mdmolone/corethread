import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { LatencyBar } from "./LatencyBar"
import { classifyTrace, type TraceClass } from "@/lib/classifyTrace"
import { narratePivot } from "@/lib/narration"
import { useTraceStore } from "@/store/traceStore"
import { useConfigStore } from "@/store/configStore"
import { formatUsd } from "@/lib/format"
import { TraceTranscript } from "./TraceTranscript"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

interface TraceRowProps {
  trace: TraceEvent
}

const STRIPE_COLOR: Record<TraceClass, string> = {
  // D-12: Tailwind border-l-4 with light+dark color pairs.
  accepted: "border-l-blue-500 dark:border-l-blue-400",
  pivoted: "border-l-amber-500 dark:border-l-amber-400",
  errored: "border-l-red-500 dark:border-l-red-400",
}

export function TraceRow({ trace }: TraceRowProps) {
  const expandedId = useTraceStore((s) => s.expandedRequestId)
  const setExpanded = useTraceStore((s) => s.setExpanded)
  const threshold = useConfigStore((s) => s.config?.routing.threshold ?? 0.7)
  const expanded = expandedId === trace.request_id
  const cls = classifyTrace(trace)
  const narration = narratePivot(trace, threshold)

  // D-15: errored rows render inside a destructive Card. The expanded grid
  // still renders below the friendly card surface (per CONTEXT D-15).
  const isErrored = cls === "errored"

  return (
    <div
      className={`border-l-4 ${STRIPE_COLOR[cls]} ${isErrored ? "bg-destructive/5" : ""} cursor-pointer border-b p-3`}
      onClick={() => setExpanded(trace.request_id)}
      data-testid="trace-row"
      data-classification={cls}
      data-request-id={trace.request_id}
    >
      <div className="flex items-center gap-3">
        <code className="text-muted-foreground text-xs">
          {trace.request_id}
        </code>
        {isErrored ? (
          <Badge variant="destructive">{trace.pivot_reason}</Badge>
        ) : (
          <Badge variant={cls === "pivoted" ? "secondary" : "outline"}>
            {cls === "pivoted" ? "pivoted" : "accepted"}
          </Badge>
        )}
        <span className="truncate text-sm">
          {trace.selected_local_model}
          {trace.frontier_model ? ` -> ${trace.frontier_model}` : ""}
        </span>
        <span className="ml-auto text-xs tabular-nums">
          conf {trace.confidence_score.toFixed(2)}
        </span>
      </div>

      <div className="mt-2">
        <LatencyBar
          local_latency_ms={trace.local_latency_ms}
          judge_latency_ms={trace.judge_latency_ms}
          frontier_latency_ms={trace.frontier_latency_ms}
        />
      </div>

      {narration && (
        <div
          className="text-muted-foreground mt-1 text-xs"
          data-testid="pivot-narration"
        >
          {narration}
        </div>
      )}

      {expanded && (
        <Card className="mt-3" data-testid="trace-expanded">
          <CardContent className="pt-4">
            <div
              className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs"
              data-testid="trace-expanded-grid"
            >
              <div>
                <span className="text-muted-foreground">request_id:</span>{" "}
                {trace.request_id}
              </div>
              <div>
                <span className="text-muted-foreground">
                  selected_local_model:
                </span>{" "}
                {trace.selected_local_model}
              </div>
              <div>
                <span className="text-muted-foreground">judge_model:</span>{" "}
                {trace.judge_model}
              </div>
              <div>
                <span className="text-muted-foreground">frontier_model:</span>{" "}
                {trace.frontier_model ?? "—"}
              </div>
              <div>
                <span className="text-muted-foreground">confidence_score:</span>{" "}
                {trace.confidence_score.toFixed(4)}
              </div>
              <div>
                <span className="text-muted-foreground">pivoted:</span>{" "}
                {String(trace.pivoted)}
              </div>
              <div>
                <span className="text-muted-foreground">local_latency_ms:</span>{" "}
                {trace.local_latency_ms}
              </div>
              <div>
                <span className="text-muted-foreground">judge_latency_ms:</span>{" "}
                {trace.judge_latency_ms}
              </div>
              <div>
                <span className="text-muted-foreground">
                  frontier_latency_ms:
                </span>{" "}
                {trace.frontier_latency_ms ?? "—"}
              </div>
              <div>
                <span className="text-muted-foreground">input_tokens:</span>{" "}
                {trace.input_tokens}
              </div>
              <div>
                <span className="text-muted-foreground">output_tokens:</span>{" "}
                {trace.output_tokens}
              </div>
              <div>
                <span className="text-muted-foreground">
                  frontier_cost_est:
                </span>{" "}
                {trace.frontier_cost_est !== null
                  ? formatUsd(trace.frontier_cost_est)
                  : "—"}
              </div>
              <div>
                <span className="text-muted-foreground">
                  judge_parse_failed:
                </span>{" "}
                {String(trace.judge_parse_failed)}
              </div>
              <div>
                <span className="text-muted-foreground">pivot_reason:</span>{" "}
                {trace.pivot_reason}
              </div>
              <div>
                <span className="text-muted-foreground">
                  local_error_class:
                </span>{" "}
                {trace.local_error_class ?? "—"}
              </div>
            </div>
            <TraceTranscript requestId={trace.request_id} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
