import { useEffect, useRef, useState } from "react"
import { useTraceStore, type TraceEvent } from "@/store/traceStore"

// D-05: the `eventsource` npm polyfill ships in BOTH prod and tests for
// behavior parity. Using it in prod also unlocks Last-Event-ID header on
// every reconnect — native browser EventSource cannot set custom headers
// across a manual close+reopen.
//
// The polyfill is registered on globalThis.EventSource by:
//   - frontend/src/test/setup.ts (Vitest jsdom)
//
// We import the EventSource class directly here so this hook does not depend
// on a specific globalThis registration order.
import { EventSource as EventSourcePolyfill } from "eventsource"

export type SSEStatus = "connecting" | "open" | "closed" | "error"

interface UseEventSourceResult {
  status: SSEStatus
}

// D-06 backoff math — VERBATIM from CONTEXT.md:
//   const base = Math.min(30_000, 500 * 2 ** retries.current++)
//   const delay = base * (0.9 + Math.random() * 0.2)
// Returns a delay in ms. Caller increments retries.current.
function backoffDelayMs(retries: number): number {
  const base = Math.min(30_000, 500 * 2 ** retries)
  return base * (0.9 + Math.random() * 0.2)
}

// Default parser: data is a JSON-encoded TraceEvent payload from /v1/traces/stream.
const defaultParse = (data: string): TraceEvent =>
  JSON.parse(data) as TraceEvent

/**
 * D-08 side-effect hook: connects to an SSE endpoint, parses incoming events,
 * and pushes them into useTraceStore via getState().addTrace(). Does NOT return
 * the event stream — components subscribe via useTraceStore.
 *
 * D-09 lock: mount this hook ONCE at App level. Multiple instantiations would
 * open multiple connections and burn the browser's 6-conn HTTP/1.1 cap (Pitfall 22).
 *
 * Returns: { status: 'connecting' | 'open' | 'closed' | 'error' } for a small
 * connection indicator in the App header.
 *
 * D-07 Last-Event-ID: the hook tracks the last received event id in a useRef
 * (writes happen in onmessage). On reconnect, the ref's CURRENT value is read
 * inside the connect() closure, so we always send the freshest id — not a
 * stale capture from an old render.
 */
export function useEventSource(
  url: string,
  parse: (data: string) => TraceEvent = defaultParse,
): UseEventSourceResult {
  const [status, setStatus] = useState<SSEStatus>("connecting")
  const retriesRef = useRef(0)
  const lastEventIdRef = useRef<string | null>(null)

  useEffect(() => {
    let es: EventSourcePolyfill | null = null
    let timer: ReturnType<typeof setTimeout> | null = null
    let cancelled = false

    const connect = () => {
      if (cancelled) return
      setStatus("connecting")

      // D-07: read live ref value at (re)connect — captures freshest Last-Event-ID.
      const headers: Record<string, string> = {}
      if (lastEventIdRef.current !== null) {
        headers["Last-Event-ID"] = lastEventIdRef.current
      }

      // eventsource@4: 2nd-arg eventSourceInitDict accepts a `fetch` override
      // (FetchLike). We wrap globalThis.fetch to merge our Last-Event-ID header
      // into init.headers on every (re)connection request.
      es = new EventSourcePolyfill(url, {
        fetch: (input, init) =>
          fetch(input, {
            ...init,
            headers: { ...(init?.headers ?? {}), ...headers },
          }),
      })

      es.onopen = () => {
        retriesRef.current = 0 // D-06: reset on successful open
        setStatus("open")
      }

      const handleMessageEvent = (ev: Event) => {
        const message = ev as MessageEvent<string>
        // D-07: write-through update of Last-Event-ID on every message.
        if (message.lastEventId) {
          lastEventIdRef.current = message.lastEventId
        }
        try {
          const t = parse(message.data)
          // D-08: side-effect into Zustand — does NOT setState a returned value.
          useTraceStore.getState().addTrace(t)
        } catch (err) {
          // Pitfall #12 ethos: class name only, never the error string.
          const cls =
            err instanceof Error ? err.constructor.name : "UnknownError"
          console.warn(`[useEventSource] parse failed: ${cls}`)
        }
      }

      es.onmessage = handleMessageEvent as (ev: MessageEvent<string>) => void
      es.addEventListener("trace", handleMessageEvent)

      es.onerror = () => {
        setStatus("error")
        es?.close()
        if (cancelled) return
        // D-06: exp backoff with ±10% jitter, capped 30s.
        const delay = backoffDelayMs(retriesRef.current++)
        timer = setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
      es?.close()
      setStatus("closed")
    }
  }, [url, parse])

  return { status }
}

// Exported for unit tests — allows assertion against the verbatim D-06 math.
export const _testing = { backoffDelayMs }
