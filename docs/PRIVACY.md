# Privacy

CoreThread has two observability modes:

- Body-free traces and stats, enabled by default.
- Prompt/response transcript capture, disabled by default.
- Body-free audit logging, disabled by default.

## Default Behavior

With the default config:

```yaml
privacy:
  capture_transcripts: false
  transcript_max: 25
```

CoreThread does not instrument provider calls for prompt or response body capture.
Trace SSE events and stats snapshots do not include prompt text, message bodies,
request bodies, or response bodies.

Transcript endpoints return a disabled message instead of silently exposing data.

Request-control audit logs are also disabled by default. When enabled with
`controls.audit_enabled: true`, audit events record metadata such as request id,
model, pivot status, token usage, and estimated cost. They do not include prompt
or response bodies unless `controls.audit_include_request_body: true` is set.

## Opt-In Transcript Capture

Enable transcript capture only for local debugging:

```yaml
privacy:
  capture_transcripts: true
  transcript_max: 25
```

When enabled, CoreThread keeps the last `transcript_max` transcripts in memory.
Each transcript can include:

- User and system messages sent to CoreThread.
- The local model response.
- The judge response.
- The frontier response, when a pivot occurs.
- The final response returned to the client.

This data is not persisted by CoreThread, but it is sensitive while the service is
running and visible to anyone who can reach the local UI/API.

## Recommended Public Defaults

For public examples and shared configs:

- Keep `capture_transcripts: false`.
- Keep `config.yaml` out of git.
- Do not publish `outputs/`, logs, screenshots, or run artifacts.
- Run `uv run python scripts/public_audit.py` before creating a clean public tree.
