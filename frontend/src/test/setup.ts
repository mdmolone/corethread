// Phase 10 Vitest setup. Pitfall 28: jsdom does NOT ship EventSource, so
// the eventsource npm polyfill is registered globally before any test runs.
// D-05 lock: the SAME polyfill ships in production via useEventSource (10-02).
import "@testing-library/jest-dom/vitest"
import { EventSource as EventSourcePolyfill } from "eventsource"

// Register the polyfill on globalThis so tests that grep for `EventSource`
// pass without per-file imports. Casting is required because `eventsource@4`
// exports a class that satisfies the EventSource interface but TS doesn't
// auto-widen `globalThis.EventSource` from the lib.dom.d.ts definition.
if (typeof globalThis.EventSource === "undefined") {
  ;(globalThis as { EventSource: typeof EventSource }).EventSource =
    EventSourcePolyfill as unknown as typeof EventSource
}
