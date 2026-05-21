import type { LatencyBand, ConfidenceBand } from "@/store/traceStore"

/**
 * D-13: latency-band classifier — map an absolute ms value to one of 4 bands.
 * Bands defined per CONTEXT.md D-13: <200 / 200-500 / 500-1000 / >1000.
 */
export function latencyBand(latency_ms: number): Exclude<LatencyBand, "any"> {
  if (latency_ms < 200) return "lt200"
  if (latency_ms < 500) return "200_500"
  if (latency_ms < 1000) return "500_1000"
  return "gt1000"
}

/**
 * D-13: confidence-band classifier — 3 bands per CONTEXT.md: <0.3 / 0.3-0.7 / >0.7.
 */
export function confidenceBand(score: number): Exclude<ConfidenceBand, "any"> {
  if (score < 0.3) return "lt03"
  if (score < 0.7) return "03_07"
  return "gt07"
}

/**
 * Use the per-row "wall-clock" latency for band classification: local + judge
 * + frontier (treating null frontier as 0 since unpivoted rows have no frontier).
 */
export function rowTotalLatencyMs(row: {
  local_latency_ms: number
  judge_latency_ms: number
  frontier_latency_ms: number | null
}): number {
  return (
    row.local_latency_ms + row.judge_latency_ms + (row.frontier_latency_ms ?? 0)
  )
}
