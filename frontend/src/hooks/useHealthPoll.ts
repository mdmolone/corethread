import { useEffect, useState, useRef } from "react"

// /health response shape (hand-typed — main.py's /health endpoint has no
// Pydantic response model so types.ts cannot generate this).
export type ProviderState = "ready" | "warming" | "unhealthy"
export type HealthStatus = "ok" | "degraded"
export type PollStatus = "idle" | "loading" | "ready" | "error" | "paused"

export interface ProviderHealth {
  kind: string
  state: ProviderState
  last_error: string | null
}

export interface HealthResponse {
  status: HealthStatus
  version: string
  providers: {
    local: ProviderHealth
    frontier: ProviderHealth
  }
}

export interface UseHealthPollResult {
  health: HealthResponse | null
  status: PollStatus
  errorClass: string | null
}

const POLL_MS = 5000

/**
 * D-26: poll /health every 5s.
 *   - AbortController cancels in-flight on unmount.
 *   - visibilitychange listener pauses polling when document.hidden is true.
 *   - Resumes immediately on visibility return + at next interval.
 *
 * Mounted ONCE at App level (TopBar). Single-instance policy — flipping tabs
 * does NOT spin up a second poll loop (mirror of D-09 for the SSE hook).
 */
export function useHealthPoll(): UseHealthPollResult {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [status, setStatus] = useState<PollStatus>("idle")
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
        const res = await fetch("/health", { signal: abortCtl.signal })
        if (!res.ok) {
          if (cancelled) return
          setStatus("error")
          setErrorClass(`HTTP_${res.status}`)
          return
        }
        const json = (await res.json()) as HealthResponse
        if (cancelled) return
        setHealth(json)
        setStatus("ready")
        setErrorClass(null)
      } catch (err) {
        if (cancelled) return
        const cls = err instanceof Error ? err.constructor.name : "UnknownError"
        if (cls === "AbortError") return
        setStatus("error")
        setErrorClass(cls)
        // Pitfall #12 ethos: log class name only, never the error string content.
        console.warn(`[useHealthPoll] fetch failed: ${cls}`)
      }
    }

    const onVisibilityChange = () => {
      isHiddenRef.current = document.hidden
      if (!document.hidden) {
        void fetchOnce()
      } else {
        setStatus("paused")
      }
    }

    void fetchOnce()
    intervalHandle = setInterval(fetchOnce, POLL_MS)
    document.addEventListener("visibilitychange", onVisibilityChange)

    return () => {
      cancelled = true
      if (intervalHandle) clearInterval(intervalHandle)
      abortCtl?.abort()
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }, [])

  return { health, status, errorClass }
}
