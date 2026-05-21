import { useState, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import {
  useTraceStore,
  type PivotedFilter,
  type PivotReasonFilter,
  type LatencyBand,
  type ConfidenceBand,
  type Filters,
} from "@/store/traceStore"

const PIVOTED_OPTIONS: { value: PivotedFilter; label: string }[] = [
  { value: "any", label: "any" },
  { value: "yes", label: "yes" },
  { value: "no", label: "no" },
]
const PIVOT_REASON_OPTIONS: { value: PivotReasonFilter; label: string }[] = [
  { value: "any", label: "any" },
  { value: "none", label: "none" },
  { value: "low_score", label: "low_score" },
  { value: "local_truncated", label: "local_truncated" },
  { value: "local_error", label: "local_error" },
  { value: "judge_error", label: "judge_error" },
]
const LATENCY_OPTIONS: { value: LatencyBand; label: string }[] = [
  { value: "any", label: "any" },
  { value: "lt200", label: "<200ms" },
  { value: "200_500", label: "200-500ms" },
  { value: "500_1000", label: "500-1000ms" },
  { value: "gt1000", label: ">1000ms" },
]
const CONFIDENCE_OPTIONS: { value: ConfidenceBand; label: string }[] = [
  { value: "any", label: "any" },
  { value: "lt03", label: "<0.3" },
  { value: "03_07", label: "0.3-0.7" },
  { value: "gt07", label: ">0.7" },
]

interface ChipProps<V extends string> {
  label: string
  value: V
  options: { value: V; label: string }[]
  onChange: (v: V) => void
}

function Chip<V extends string>({
  label,
  value,
  options,
  onChange,
}: ChipProps<V>) {
  const [open, setOpen] = useState(false)
  const active = value !== "any"
  return (
    <div className="relative inline-block">
      <Button
        size="sm"
        variant={active ? "default" : "outline"}
        onClick={() => setOpen((v) => !v)}
        data-testid={`chip-${label}`}
      >
        {label}: {options.find((o) => o.value === value)?.label ?? value}
      </Button>
      {open && (
        <Card
          className="absolute z-50 mt-1 min-w-[10rem]"
          data-testid={`chip-popover-${label}`}
        >
          <CardContent className="flex flex-col gap-1 p-2">
            {options.map((o) => (
              <Button
                key={o.value}
                variant={o.value === value ? "default" : "ghost"}
                size="sm"
                onClick={() => {
                  onChange(o.value)
                  setOpen(false)
                }}
                data-testid={`chip-option-${label}-${o.value}`}
              >
                {o.label}
              </Button>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export function FilterToolbar() {
  const filters = useTraceStore((s) => s.filters)
  const setFilter = useTraceStore((s) => s.setFilter)
  const resetFilters = useTraceStore((s) => s.resetFilters)
  const traces = useTraceStore((s) => s.traces)

  // Derive the model-options union from the live trace stream:
  // selected_local_model + frontier_model (filtered for null), per D-13.
  const modelOptions = useMemo(() => {
    const set = new Set<string>()
    for (const t of traces) {
      set.add(t.selected_local_model)
      if (t.frontier_model) set.add(t.frontier_model)
    }
    return [
      { value: "any" as const, label: "any" },
      ...Array.from(set).map((m) => ({ value: m, label: m })),
    ]
  }, [traces])

  const hasActiveFilter = (Object.keys(filters) as (keyof Filters)[]).some(
    (k) => filters[k] !== "any",
  )

  return (
    <div
      className="bg-background sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b p-2"
      data-testid="filter-toolbar"
    >
      <Chip
        label="pivoted"
        value={filters.pivoted}
        options={PIVOTED_OPTIONS}
        onChange={(v) => setFilter("pivoted", v)}
      />
      <Chip
        label="model"
        value={filters.model}
        options={modelOptions}
        onChange={(v) => setFilter("model", v)}
      />
      <Chip
        label="pivot_reason"
        value={filters.pivot_reason}
        options={PIVOT_REASON_OPTIONS}
        onChange={(v) => setFilter("pivot_reason", v)}
      />
      <Chip
        label="latency"
        value={filters.latency_band}
        options={LATENCY_OPTIONS}
        onChange={(v) => setFilter("latency_band", v)}
      />
      <Chip
        label="confidence"
        value={filters.confidence_band}
        options={CONFIDENCE_OPTIONS}
        onChange={(v) => setFilter("confidence_band", v)}
      />
      {hasActiveFilter && (
        <>
          <span className="ml-2">·</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetFilters}
            data-testid="filter-clear-all"
          >
            Clear all
          </Button>
        </>
      )}
      <Badge variant="outline" className="ml-auto">
        {traces.length} traces
      </Badge>
    </div>
  )
}
