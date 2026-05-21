import { useEffect, useState } from "react"

interface TranscriptMessage {
  role?: string
  content?: string | unknown[] | null
}

interface TraceTranscriptPayload {
  request_id: string
  request_messages: TranscriptMessage[]
  local_response: string | null
  judge_response: string | null
  frontier_response: string | null
  final_response: string | null
  final_response_source: "local" | "frontier" | "none"
  errors: Record<string, string>
}

interface Props {
  requestId: string
}

function displayContent(value: string | unknown[] | null | undefined): string {
  if (typeof value === "string") return value
  if (value == null) return "None"
  return JSON.stringify(value, null, 2)
}

function TextBlock({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="space-y-1">
      <div className="text-muted-foreground text-[11px] font-medium tracking-normal uppercase">
        {label}
      </div>
      <pre className="bg-muted/40 max-h-44 overflow-auto rounded border p-2 text-xs whitespace-pre-wrap">
        {value || "None"}
      </pre>
    </div>
  )
}

export function TraceTranscript({ requestId }: Props) {
  const [payload, setPayload] = useState<TraceTranscriptPayload | null>(null)
  const [status, setStatus] = useState<
    "loading" | "ready" | "missing" | "disabled"
  >("loading")

  useEffect(() => {
    const controller = new AbortController()
    setStatus("loading")
    setPayload(null)

    fetch(`/v1/traces/${encodeURIComponent(requestId)}/transcript`, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as {
            detail?: string
          } | null
          throw new Error(body?.detail ?? "missing")
        }
        return response.json() as Promise<TraceTranscriptPayload>
      })
      .then((body) => {
        setPayload(body)
        setStatus("ready")
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return
        const message = err instanceof Error ? err.message : ""
        setStatus(
          message.includes("Transcript capture is disabled")
            ? "disabled"
            : "missing",
        )
      })

    return () => controller.abort()
  }, [requestId])

  if (status === "loading") {
    return (
      <div className="text-muted-foreground text-xs">Loading transcript...</div>
    )
  }

  if (status === "disabled") {
    return (
      <div className="text-muted-foreground text-xs">
        Transcript capture is disabled. Enable privacy.capture_transcripts and
        restart CoreThread to inspect prompt and response bodies.
      </div>
    )
  }

  if (status === "missing" || payload === null) {
    return (
      <div className="text-muted-foreground text-xs">
        Transcript evicted or unavailable.
      </div>
    )
  }

  return (
    <div className="mt-4 space-y-3" data-testid="trace-transcript">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div className="space-y-2">
          <div className="text-muted-foreground text-[11px] font-medium tracking-normal uppercase">
            Request messages
          </div>
          {payload.request_messages.map((message, index) => (
            <div
              key={`${message.role ?? "message"}-${index}`}
              className="space-y-1"
            >
              <div className="text-muted-foreground text-[11px]">
                {message.role ?? "message"}
              </div>
              <pre className="bg-muted/40 max-h-44 overflow-auto rounded border p-2 text-xs whitespace-pre-wrap">
                {displayContent(message.content)}
              </pre>
            </div>
          ))}
        </div>

        <div className="space-y-3">
          <TextBlock label="Local response" value={payload.local_response} />
          <TextBlock label="Judge response" value={payload.judge_response} />
          <TextBlock
            label="Frontier response"
            value={payload.frontier_response}
          />
          <TextBlock
            label={`Final response (${payload.final_response_source})`}
            value={payload.final_response}
          />
        </div>
      </div>

      {Object.keys(payload.errors).length > 0 && (
        <TextBlock
          label="Provider errors"
          value={JSON.stringify(payload.errors)}
        />
      )}
    </div>
  )
}
