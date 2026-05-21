import type { CSSProperties } from "react"

export const chartTooltipContentStyle: CSSProperties = {
  background: "hsl(var(--popover))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 6,
  color: "hsl(var(--popover-foreground))",
  boxShadow: "0 10px 30px hsl(0 0% 0% / 0.35)",
}

export const chartTooltipLabelStyle: CSSProperties = {
  color: "hsl(var(--popover-foreground))",
  fontWeight: 600,
}

export const chartTooltipItemStyle: CSSProperties = {
  color: "hsl(var(--popover-foreground))",
}
