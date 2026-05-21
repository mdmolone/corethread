import { useEffect, useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { useEventSource } from "@/hooks/useEventSource"
import { useHealthPoll } from "@/hooks/useHealthPoll"
import { useRouteActivitySource } from "@/hooks/useRouteActivitySource"
import { useConfigStore } from "@/store/configStore"
import { TraceView } from "@/views/TraceView"
import { StatsView } from "@/views/StatsView"
import { ConfigView } from "@/views/ConfigView"
import { TopBar } from "@/components/TopBar"

type ViewName = "config" | "traces" | "stats"

export function App() {
  const [view, setView] = useState<ViewName>("traces")

  // D-09: single useEventSource instance at App level — connection persists
  // across view switches; flipping to Stats does NOT close the SSE stream
  // (Pitfall 22 defense). Hook calls useTraceStore.getState().addTrace(t)
  // imperatively per D-08 — components subscribe via useTraceStore.
  const { status: sseStatus } = useEventSource("/v1/traces/stream")
  const { status: routeStatus } = useRouteActivitySource("/v1/route/stream")
  const healthResult = useHealthPoll()

  // Fetch /v1/config once on app mount; configStore is idempotent on ready.
  const fetchConfig = useConfigStore((s) => s.fetchConfig)
  const configStatus = useConfigStore((s) => s.status)
  const themeMode = useConfigStore((s) => s.config?.ui?.theme ?? "system")
  useEffect(() => {
    void fetchConfig()
  }, [fetchConfig])

  useEffect(() => {
    if (configStatus !== "error") return
    const retryId = window.setTimeout(() => {
      void fetchConfig()
    }, 3000)
    return () => window.clearTimeout(retryId)
  }, [configStatus, fetchConfig])

  useEffect(() => {
    const root = document.documentElement
    if (themeMode === "system") {
      root.removeAttribute("data-theme")
      return
    }
    root.setAttribute("data-theme", themeMode)
  }, [themeMode])

  return (
    <div className="bg-background text-foreground min-h-screen">
      <Tabs value={view} onValueChange={(v) => setView(v as ViewName)}>
        <header className="border-b p-4">
          <div className="mx-auto flex max-w-6xl items-center justify-between">
            <h1 className="text-xl font-semibold">CoreThread</h1>
            <div className="flex items-center gap-3">
              <TopBar healthResult={healthResult} />
              <Badge
                variant={sseStatus === "open" ? "outline" : "secondary"}
                data-testid="sse-status"
              >
                SSE: {sseStatus}
              </Badge>
              <Badge
                variant={routeStatus === "open" ? "outline" : "secondary"}
                data-testid="route-status"
              >
                Route: {routeStatus}
              </Badge>
              <Badge
                variant={configStatus === "ready" ? "outline" : "secondary"}
                data-testid="config-status"
              >
                Config: {configStatus}
              </Badge>
              <TabsList>
                <TabsTrigger value="config">Config</TabsTrigger>
                <TabsTrigger value="traces">Traces</TabsTrigger>
                <TabsTrigger value="stats">Stats</TabsTrigger>
              </TabsList>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-6xl p-4">
          <TabsContent value="config">
            <ConfigView />
          </TabsContent>
          <TabsContent value="traces">
            <TraceView healthResult={healthResult} />
          </TabsContent>
          <TabsContent value="stats">
            <StatsView />
          </TabsContent>
        </main>
      </Tabs>
    </div>
  )
}
