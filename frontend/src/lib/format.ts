import { formatDistanceToNowStrict } from "date-fns"

/**
 * Trace events do not carry a timestamp field (TRC-09 lock — only the 15
 * documented fields). For display, callers can pass `Date.now()` at the time
 * the row was added to the store; downstream a future field can replace this
 * pattern with a TraceEvent.received_at.
 */
export function relativeTime(ts_ms: number): string {
  return formatDistanceToNowStrict(new Date(ts_ms), { addSuffix: true })
}

/**
 * Compact integer formatter — "1234" -> "1.23k", "1234567" -> "1.23M".
 * Used by token-count and request-count surfaces.
 */
export function compactInt(n: number): string {
  if (n < 1000) return n.toString()
  if (n < 1_000_000) return `${(n / 1000).toFixed(2).replace(/\.?0+$/, "")}k`
  return `${(n / 1_000_000).toFixed(2).replace(/\.?0+$/, "")}M`
}

/**
 * USD with 2 decimal places, used by cost-saved counter (10-04) AND the
 * frontier_cost_est field on Trace rows (TRC-04 expanded view).
 */
export function formatUsd(n: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(n)
}
