# CoreThread

CoreThread is a local-first, OpenAI-compatible LLM router. Point an OpenAI-style
client at `http://localhost:8000/v1`, let a local model answer first, have a judge
model score the result, and pivot to a frontier model only when the local answer
does not clear your quality threshold.

The narrow goal: keep normal requests local while preserving the `/v1/chat/completions`
surface that existing OpenAI clients already understand.

## What It Does

- Serves `POST /v1/chat/completions` and `GET /v1/models`.
- Supports non-streaming and streaming chat-completion responses.
- Routes through configurable `local`, `judge`, and `frontier` model profiles.
- Works with Ollama, LM Studio, OpenAI, and OpenRouter-compatible profiles.
- Provides a local UI for traces, stats, and config editing.
- Keeps prompt/response transcript capture disabled by default.
- Offers opt-in local guardrails: rate limits, daily quotas, cost estimates, and audit logs.

## Quickstart

Prerequisites:

- Python 3.12+
- `uv`
- Node 20.19+ and `pnpm`
- A local Ollama or LM Studio server
- A frontier API key if your config can pivot to OpenAI or OpenRouter

```bash
git clone https://github.com/mdmolone/corethread.git
cd corethread
uv sync
pnpm --dir frontend install --frozen-lockfile
cp config.yaml.example config.yaml
```

Set your key using the environment variable named by `frontier.api_key_env`:

```bash
export OPENAI_API_KEY="<your-openai-api-key>"
```

On Windows PowerShell, you can use the helper script:

```powershell
.\scripts\set-openai-api-key.ps1 -Name OPENAI_API_KEY
```

Start the service:

```bash
pnpm --dir frontend build
uv run corethread
```

Open the UI at `http://localhost:8000`.

Send a first request with the official OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="corethread-local-client",
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Say OK."}],
)

print(response.choices[0].message.content)
```

Or with raw HTTP:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Say OK."}]
  }'
```

## UI

CoreThread ships a local single-page UI for operating the router.

![CoreThread trace view showing live route status, deep route timing, and recent request traces](docs/assets/screenshots/corethread-traces.png)

![CoreThread stats view showing request counts, latency percentiles, confidence histogram, and pivot reasons](docs/assets/screenshots/corethread-stats.png)

![CoreThread config view showing role profile selection, routing settings, and privacy controls](docs/assets/screenshots/corethread-config.png)

## Configuration

Start with `config.yaml.example` or one of the examples in `examples/`:

- `examples/config.ollama-openai.yaml`
- `examples/config.lmstudio-openai.yaml`
- `examples/config.lmstudio-openai-judge.yaml`
- `examples/config.openrouter.yaml`

The current public config shape uses reusable `model_profiles`, active
`role_profiles`, `routing`, `ui`, and `privacy` blocks. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

Local-only/no-cloud operation is possible only when your routing threshold and
model choices keep requests on the local path. If a request can pivot to a cloud
frontier, CoreThread requires the configured frontier environment variable at
startup so the fallback path is ready.

## Compatibility

| Client or Feature | Status |
| --- | --- |
| Raw HTTP `/v1/chat/completions` | Supported |
| Raw SSE `stream=true` | Supported |
| Official OpenAI Python SDK | Supported, including streaming |
| Unknown extra OpenAI request fields | Accepted and preserved where possible |
| Official OpenAI JS SDK | Not a release target yet |
| Embeddings, images, audio, assistants | Not implemented |

## Privacy And Security

CoreThread is designed for a single-user local service. It does not implement
authentication, tenancy, or public internet hardening. Keep it bound to localhost
unless you know exactly what is on the network path.

Prompt/response transcript capture is opt-in:

```yaml
privacy:
  capture_transcripts: false
  transcript_max: 25
```

When disabled, trace and stats surfaces remain body-free and transcript endpoints
return a clear disabled message. When enabled, CoreThread keeps the last N prompt
and response bodies in memory for local debugging. See [docs/PRIVACY.md](docs/PRIVACY.md)
and [SECURITY.md](SECURITY.md).

Rate limits, quotas, billing estimates, and audit logs are configured under
`controls:`. They are global in-process guardrails for a local router, not a
multi-tenant billing system.

## Development

Run the backend and frontend checks:

```bash
uv run ruff check
uv run ruff format --check .
uv run mypy .
uv run pytest -q
pnpm --dir frontend lint
pnpm --dir frontend test
pnpm --dir frontend build
```

Run the public-readiness audit on a clean public export:

```bash
uv run python scripts/public_audit.py --history
```

The existing private development history contains internal planning files. Do not
make that history public directly. Use the clean-history release process in
[docs/PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md).

## License

MIT. See [LICENSE](LICENSE).
