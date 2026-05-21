import { describe, it, expect, beforeEach } from "vitest"
import {
  ROUTE_ACTIVITY_BUFFER_CAP,
  useRouteActivityStore,
  type RouteActivityEvent,
} from "./routeActivityStore"

const makeEvent = (
  request_id: string,
  stage: RouteActivityEvent["stage"] = "received",
  overrides: Partial<RouteActivityEvent> = {},
): RouteActivityEvent => ({
  request_id,
  stage,
  selected_local_model: "local-model",
  judge_model: "judge-model",
  frontier_model: "frontier-model",
  pivoted: null,
  confidence_score: null,
  error_class: null,
  ts_ms: 1,
  ...overrides,
})

describe("routeActivityStore", () => {
  beforeEach(() => {
    useRouteActivityStore.setState({ events: [], activeRequestId: null })
  })

  it("stores events and updates the active request", () => {
    useRouteActivityStore.getState().addRouteEvent(makeEvent("req-1"))
    useRouteActivityStore
      .getState()
      .addRouteEvent(makeEvent("req-2", "local_started", { ts_ms: 2 }))

    expect(useRouteActivityStore.getState().events).toHaveLength(2)
    expect(useRouteActivityStore.getState().activeRequestId).toBe("req-2")
  })

  it("dedupes replayed route events by request/stage/timestamp", () => {
    const event = makeEvent("req-1", "received", { ts_ms: 42 })
    useRouteActivityStore.getState().addRouteEvent(event)
    useRouteActivityStore.getState().addRouteEvent(event)

    expect(useRouteActivityStore.getState().events).toHaveLength(1)
  })

  it("caps the buffer and drops oldest events", () => {
    for (let i = 0; i < ROUTE_ACTIVITY_BUFFER_CAP + 3; i++) {
      useRouteActivityStore
        .getState()
        .addRouteEvent(makeEvent(`req-${i}`, "received", { ts_ms: i }))
    }

    const events = useRouteActivityStore.getState().events
    expect(events).toHaveLength(ROUTE_ACTIVITY_BUFFER_CAP)
    expect(events[0]?.request_id).toBe("req-3")
  })
})
