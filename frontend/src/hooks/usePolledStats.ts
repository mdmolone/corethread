import { useEffect, useState, useRef } from "react"
import type { components } from "@/api/types"

type StatsSnapshot =
  | components["schemas"]["StatsSnapshotEmpty"]
  | components["schemas"]["StatsSnapshotPopulated"]

export type StatsWindow = "1h" | "24h" | "7d" | "all"

export type PolledStatsStatus =
  | "idle"
  | "loading"
  | "ready"
  | "error"
  | "paused"

export interface PolledStatsResult {
  snapshot: StatsSnapshot | null
  status: PolledStatsStatus
  errorClass: string | null
}

const POLL_MS = 2000

/**
 * D-19: poll /v1/stats?window=${window} every 2s.
 *   - Uses AbortController to cancel in-flight requests on unmount / window change.
 *   - Pauses polling when document.hidden is true (visibilitychange listener).
 *   - Resumes immediately on visibility return + at the next interval tick.
 *
 * Returns the latest snapshot + a status badge. The hook does NOT cache across
 * window changes — switching window resets snapshot to null until first poll
 * resolves (this is the right UX: visibly tells the user the window changed).
 */
export function usePolledStats(window: StatsWindow): PolledStatsResult {
  const [snapshot, setSnapshot] = useState<StatsSnapshot | null>(null)
  const [status, setStatus] = useState<PolledStatsStatus>("idle")
  const [errorClass, setErrorClass] = useState<string | null>(null)
  const isHiddenRef = useRef<boolean>(
    typeof document !== "undefined" && document.hidden,
  )

  useEffect(() => {
    let cancelled = false
    let intervalHandle: ReturnType<typeof setInterval> | null = null
    let abortCtl: AbortController | null = null

    const fetchOnce = async () => {
      if (cancelled) return
      if (isHiddenRef.current) {
        setStatus("paused")
        return
      }
      abortCtl?.abort()
      abortCtl = new AbortController()
      try {
        setStatus((s) => (s === "ready" ? s : "loading"))
        const res = await fetch(
          `/v1/stats?window=${encodeURIComponent(window)}`,
          { signal: abortCtl.signal },
        )
        if (!res.ok) {
          if (cancelled) return
          setStatus("error")
          setErrorClass(`HTTP_${res.status}`)
          return
        }
        const json = (await res.json()) as StatsSnapshot
        if (cancelled) return
        setSnapshot(json)
        setStatus("ready")
        setErrorClass(null)
      } catch (err) {
        if (cancelled) return
        // Pitfall #12 ethos: log class name only, never the error string content.
        const cls = err instanceof Error ? err.constructor.name : "UnknownError"
        // AbortError on window-change / unmount is expected; ignore.
        if (cls === "AbortError") return
        setStatus("error")
        setErrorClass(cls)
        console.warn(`[usePolledStats] fetch failed: ${cls}`)
      }
    }

    const onVisibilityChange = () => {
      isHiddenRef.current = document.hidden
      if (!document.hidden) {
        // Refetch immediately on resume.
        void fetchOnce()
      } else {
        setStatus("paused")
      }
    }

    setSnapshot(null) // reset on window change so old window's data doesn't linger
    void fetchOnce()
    intervalHandle = setInterval(fetchOnce, POLL_MS)
    document.addEventListener("visibilitychange", onVisibilityChange)

    return () => {
      cancelled = true
      if (intervalHandle) clearInterval(intervalHandle)
      abortCtl?.abort()
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }, [window])

  return { snapshot, status, errorClass }
}
