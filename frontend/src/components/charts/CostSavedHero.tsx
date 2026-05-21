import { useMemo } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { estimateCost, GPT4O_PRICING } from "@/lib/pricing"
import { formatUsd } from "@/lib/format"
import type { components } from "@/api/types"

type TraceEvent = components["schemas"]["TraceEvent"]

interface Props {
  traces: TraceEvent[] // time-windowed slice (StatsView filters before passing)
}

export function CostSavedHero({ traces }: Props) {
  const { saved, count } = useMemo(() => {
    let saved = 0
    let count = 0
    for (const t of traces) {
      if (!t.pivoted) {
        saved += estimateCost(t.input_tokens, t.output_tokens)
        count += 1
      }
    }
    return { saved, count }
  }, [traces])

  return (
    <Card data-testid="cost-saved-hero">
      <CardContent className="space-y-1 p-6 text-center">
        <div className="text-muted-foreground text-xs">
          Saved vs all-frontier (DF-4 / STA-06)
        </div>
        <div
          className="text-4xl font-bold tabular-nums"
          data-testid="cost-saved-amount"
        >
          {formatUsd(saved)}
        </div>
        <div
          className="text-muted-foreground text-sm"
          data-testid="cost-saved-count"
        >
          {count} requests handled locally
        </div>
        <div className="text-muted-foreground text-[10px]">
          @ GPT-4o ${GPT4O_PRICING.input_per_1m_tokens.toFixed(2)}/1M in · $
          {GPT4O_PRICING.output_per_1m_tokens.toFixed(2)}/1M out
        </div>
      </CardContent>
    </Card>
  )
}
