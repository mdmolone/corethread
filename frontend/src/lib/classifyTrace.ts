import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

export type TraceClass = "accepted" | "pivoted" | "errored"

/**
 * D-12 + D-15: classification predicate reused by:
 *   - TraceRow color stripe (border-l-4) for TRC-02
 *   - Error-card render branch (shadcn Card variant=destructive) for TRC-07
 *
 * Predicate (D-12):
 *   errored  := pivot_reason in {"local_error", "judge_error"} OR judge_parse_failed === true
 *   pivoted  := pivoted === true AND pivot_reason in {"none", "low_score", "local_truncated"}
 *              (note: "none" with pivoted=true is defensive — should not happen, but
 *               do not double-classify as errored)
 *   accepted := pivoted === false AND local_error_class === null
 *
 * Order matters: errored takes priority over pivoted (a judge_error trace IS
 * pivoted=true with pivot_reason="judge_error", and we want it to render as
 * a destructive card, not an amber pivot row).
 */
export function classifyTrace(t: TraceEvent): TraceClass {
  if (
    t.judge_parse_failed === true ||
    t.pivot_reason === "local_error" ||
    t.pivot_reason === "judge_error"
  ) {
    return "errored"
  }
  if (t.pivoted) return "pivoted"
  return "accepted"
}
