import { useEffect, useRef, useState } from "react"
import { EventSource as EventSourcePolyfill } from "eventsource"
import {
  useRouteActivityStore,
  type RouteActivityEvent,
} from "@/store/routeActivityStore"
import type { SSEStatus } from "@/hooks/useEventSource"

interface UseRouteActivitySourceResult {
  status: SSEStatus
}

function backoffDelayMs(retries: number): number {
  const base = Math.min(30_000, 500 * 2 ** retries)
  return base * (0.9 + Math.random() * 0.2)
}

const defaultParse = (data: string): RouteActivityEvent =>
  JSON.parse(data) as RouteActivityEvent

export function useRouteActivitySource(
  url: string,
  parse: (data: string) => RouteActivityEvent = defaultParse,
): UseRouteActivitySourceResult {
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

      const headers: Record<string, string> = {}
      if (lastEventIdRef.current !== null) {
        headers["Last-Event-ID"] = lastEventIdRef.current
      }

      es = new EventSourcePolyfill(url, {
        fetch: (input, init) =>
          fetch(input, {
            ...init,
            headers: { ...(init?.headers ?? {}), ...headers },
          }),
      })

      es.onopen = () => {
        retriesRef.current = 0
        setStatus("open")
      }

      const handleRouteEvent = (ev: Event) => {
        const message = ev as MessageEvent<string>
        if (message.lastEventId) {
          lastEventIdRef.current = message.lastEventId
        }
        try {
          useRouteActivityStore.getState().addRouteEvent(parse(message.data))
        } catch (err) {
          const cls =
            err instanceof Error ? err.constructor.name : "UnknownError"
          console.warn(`[useRouteActivitySource] parse failed: ${cls}`)
        }
      }

      es.addEventListener("route", handleRouteEvent)

      es.onerror = () => {
        setStatus("error")
        es?.close()
        if (cancelled) return
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

export const _testing = { backoffDelayMs }
