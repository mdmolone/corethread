import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

/**
 * D-14: pure mapping from a TraceEvent to a display string for the inline
 * pivot-reason narration on Trace rows (TRC-08).
 *
 * Threshold is passed in (read from useConfigStore at call site). Keeping
 * narration.ts pure means it stays unit-testable in isolation; backend-side
 * narration was rejected (would mutate Phase 9 TRC-09-locked TraceEvent).
 *
 * Mapping table (D-14 verbatim):
 *   - pivoted=false        -> "" (empty — no narration shown)
 *   - low_score            -> "Confidence X.XX < Y.YY" (X = trace, Y = threshold)
 *   - local_truncated      -> "Local response hit token limit"
 *   - local_error          -> "Local provider unreachable (local_error_class)"
 *   - judge_error          -> "Judge JSON parse failed"
 *   - none                 -> "" (defensive — pivoted=true with reason=none should not happen)
 *   - default (future)     -> "Pivoted (<reason>)"
 */
export function narratePivot(t: TraceEvent, threshold: number): string {
  if (!t.pivoted) return ""
  switch (t.pivot_reason) {
    case "low_score":
      return `Confidence ${t.confidence_score.toFixed(2)} < ${threshold.toFixed(2)}`
    case "local_truncated":
      return "Local response hit token limit"
    case "local_error":
      return `Local provider unreachable (${t.local_error_class ?? "unknown"})`
    case "judge_error":
      return `Judge JSON parse failed`
    case "none":
      return "" // defensive — pivoted=true with reason=none should not happen
    default:
      // Forward-compat: a 5th pivot_reason added by Phase 12 backend would
      // surface as a generic "Pivoted (...)" pill until narration.ts is updated.
      return `Pivoted (${t.pivot_reason as string})`
  }
}
