import { Card, CardContent } from "@/components/ui/card"
import { compactInt } from "@/lib/format"
import type { components } from "@/api/types"

type StatsSnapshotPopulated = components["schemas"]["StatsSnapshotPopulated"]
type StatsSnapshotEmpty = components["schemas"]["StatsSnapshotEmpty"]

interface StatHeroProps {
  snapshot: StatsSnapshotEmpty | StatsSnapshotPopulated
}

function HeroCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-muted-foreground text-xs">{label}</div>
        <div
          className="text-2xl font-semibold tabular-nums"
          data-testid={`stat-hero-${label}`}
        >
          {value}
        </div>
      </CardContent>
    </Card>
  )
}

function isEmpty(s: StatHeroProps["snapshot"]): s is StatsSnapshotEmpty {
  return s.window_size === 0
}

export function StatHero({ snapshot }: StatHeroProps) {
  if (isEmpty(snapshot)) {
    return (
      <div
        className="grid grid-cols-2 gap-4 md:grid-cols-4"
        data-testid="stat-hero-empty"
      >
        <HeroCard label="total" value="0" />
        <HeroCard label="pivot_rate" value="—" />
        <HeroCard label="local_p50" value="—" />
        <HeroCard label="confidence_p50" value="—" />
      </div>
    )
  }
  // STA-01 — populated counters from snapshot.window_size, snapshot.pivot_rate,
  // snapshot.local_latency_ms.p50 / .p95, snapshot.judge_latency_ms.p50 / .p95,
  // snapshot.frontier_latency_ms.p50 (when not null), snapshot.confidence_score.p50.
  const s: StatsSnapshotPopulated = snapshot
  return (
    <div
      className="grid grid-cols-2 gap-4 md:grid-cols-4"
      data-testid="stat-hero"
    >
      <HeroCard label="total" value={compactInt(s.window_size)} />
      <HeroCard
        label="pivot_rate"
        value={`${(s.pivot_rate * 100).toFixed(1)}%`}
      />
      <HeroCard
        label="local_p50"
        value={s.local_latency_ms ? `${s.local_latency_ms.p50}ms` : "—"}
      />
      <HeroCard
        label="local_p95"
        value={s.local_latency_ms ? `${s.local_latency_ms.p95}ms` : "—"}
      />
      <HeroCard
        label="judge_p50"
        value={s.judge_latency_ms ? `${s.judge_latency_ms.p50}ms` : "—"}
      />
      <HeroCard
        label="judge_p95"
        value={s.judge_latency_ms ? `${s.judge_latency_ms.p95}ms` : "—"}
      />
      <HeroCard
        label="frontier_p50"
        value={s.frontier_latency_ms ? `${s.frontier_latency_ms.p50}ms` : "—"}
      />
      <HeroCard
        label="confidence_p50"
        value={s.confidence_score ? s.confidence_score.p50.toFixed(2) : "—"}
      />
    </div>
  )
}
