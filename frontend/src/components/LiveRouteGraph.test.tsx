import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { LiveRouteGraph } from "@/components/LiveRouteGraph"
import type { UseHealthPollResult } from "@/hooks/useHealthPoll"
import {
  useRouteActivityStore,
  type RouteActivityEvent,
} from "@/store/routeActivityStore"

const healthResult: UseHealthPollResult = {
  status: "ready",
  errorClass: null,
  health: {
    status: "ok",
    version: "1.0.0",
    providers: {
      local: { kind: "lmstudio", state: "ready", last_error: null },
      frontier: { kind: "openai", state: "ready", last_error: null },
    },
  },
}

const makeEvent = (
  stage: RouteActivityEvent["stage"],
  overrides: Partial<RouteActivityEvent> = {},
): RouteActivityEvent => ({
  request_id: "req-live",
  stage,
  selected_local_model: "local-model",
  judge_model: "judge-model",
  frontier_model: "frontier-model",
  pivoted: null,
  confidence_score: null,
  error_class: null,
  ts_ms: 100,
  ...overrides,
})

describe("LiveRouteGraph", () => {
  beforeEach(() => {
    useRouteActivityStore.setState({ events: [], activeRequestId: null })
  })

  it("renders an idle route before any request activity", () => {
    render(<LiveRouteGraph healthResult={healthResult} />)

    expect(screen.getByTestId("live-route-graph")).toBeInTheDocument()
    expect(screen.getByTestId("live-route-stage")).toHaveTextContent("Idle")
    expect(screen.getByTestId("live-route-now")).toHaveTextContent(
      "No route activity yet",
    )
    expect(screen.getByTestId("route-node-incoming")).toHaveAttribute(
      "data-state",
      "idle",
    )
  })

  it("shows judge activity and model labels for the latest request", () => {
    useRouteActivityStore
      .getState()
      .addRouteEvent(makeEvent("received", { ts_ms: 1 }))
    useRouteActivityStore
      .getState()
      .addRouteEvent(makeEvent("local_completed", { ts_ms: 2, pivoted: false }))
    useRouteActivityStore
      .getState()
      .addRouteEvent(makeEvent("judge_started", { ts_ms: 3 }))

    render(<LiveRouteGraph healthResult={healthResult} />)

    expect(screen.getByTestId("live-route-stage")).toHaveTextContent(
      "Judge reviewing",
    )
    expect(screen.getByTestId("route-node-local")).toHaveAttribute(
      "data-state",
      "done",
    )
    expect(screen.getByTestId("route-node-judge")).toHaveAttribute(
      "data-state",
      "active",
    )
    expect(screen.getByTestId("live-route-now")).toHaveTextContent(
      "Now: Judge reviewing",
    )
    expect(screen.getByTestId("route-node-judge-status")).toHaveTextContent(
      "working",
    )
    expect(screen.getByText("local-model")).toBeInTheDocument()
  })

  it("marks frontier active on a pivot", () => {
    for (const event of [
      makeEvent("received", { ts_ms: 1 }),
      makeEvent("local_completed", { ts_ms: 2, pivoted: false }),
      makeEvent("judge_completed", {
        ts_ms: 3,
        pivoted: true,
        confidence_score: 0.42,
      }),
      makeEvent("frontier_started", {
        ts_ms: 4,
        pivoted: true,
        confidence_score: 0.42,
      }),
    ]) {
      useRouteActivityStore.getState().addRouteEvent(event)
    }

    render(<LiveRouteGraph healthResult={healthResult} />)

    expect(screen.getByTestId("route-node-frontier")).toHaveAttribute(
      "data-state",
      "active",
    )
    expect(screen.getByTestId("route-node-frontier-status")).toHaveTextContent(
      "working",
    )
    expect(screen.getByText("conf 0.42")).toBeInTheDocument()
  })
})
