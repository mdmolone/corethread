import { create } from "zustand"
import type { components } from "@/api/types"

export type RouteActivityEvent = components["schemas"]["RouteActivityEventView"]

export const ROUTE_ACTIVITY_BUFFER_CAP = 200

interface RouteActivityState {
  events: RouteActivityEvent[]
  activeRequestId: string | null
}

interface RouteActivityActions {
  addRouteEvent: (event: RouteActivityEvent) => void
  clearRouteEvents: () => void
}

function eventKey(event: RouteActivityEvent): string {
  return `${event.request_id}:${event.ts_ms}:${event.stage}`
}

export const useRouteActivityStore = create<
  RouteActivityState & RouteActivityActions
>()((set) => ({
  events: [],
  activeRequestId: null,

  addRouteEvent: (event) =>
    set((state) => {
      const key = eventKey(event)
      if (state.events.some((existing) => eventKey(existing) === key)) {
        return state
      }
      const next =
        state.events.length >= ROUTE_ACTIVITY_BUFFER_CAP
          ? [...state.events.slice(1), event]
          : [...state.events, event]
      return { events: next, activeRequestId: event.request_id }
    }),

  clearRouteEvents: () => set({ events: [], activeRequestId: null }),
}))
