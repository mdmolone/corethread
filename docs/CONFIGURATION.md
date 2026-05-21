# Configuration

CoreThread reads a YAML config file at startup. By default it uses `config.yaml`
in the current working directory. Override that with `CORETHREAD_CONFIG_PATH`.

The modern public shape is profile-based:

```yaml
model_profiles:
  local-ollama:
    provider: ollama
    base_url: http://localhost:11434
    model: llama3.1:8b

  judge-ollama:
    provider: ollama
    base_url: http://localhost:11434
    model: qwen2.5:7b
    temperature: 0

  frontier-openai:
    provider: openai
    base_url: https://api.openai.com/v1
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    max_tokens: 512

role_profiles:
  local: local-ollama
  judge: judge-ollama
  frontier: frontier-openai
```

The legacy `local`, `judge`, and `frontier` blocks are still present for backward
compatibility and are mirrored by the example configs.

The `judge.prompt` field is also configurable. Keep the required JSON output
shape and `[answered_core_q=..., no_disclaimers=..., no_contradictions=...]`
reasoning prefix unless you also change the parser/scorer.

```yaml
judge:
  model: qwen2.5:7b
  prompt: |
    You are a strict quality judge...

    Rubric (three booleans, embed as a structured prefix in `reasoning`):
      answered_core_q: did ANSWER address QUESTION's core ask? (true/false)
        Mark false if ANSWER violates exact wording, only/no-extra-text
        constraints, requested format, requested length, requested number
        of items, or requested language.
        Markdown fences count as extra text when JSON-only or no extra
        text was requested.
        Add concrete negative examples for exact-output, JSON-only,
        and item-count failures if your judge model is too permissive.
```

## Model Profile Fields

| Field | Providers | Notes |
| --- | --- | --- |
| `provider` | all | `ollama`, `lmstudio`, `openai`, or `openrouter` |
| `base_url` | all | Include `/v1` for LM Studio and OpenAI-compatible APIs |
| `model` | all | Must match the upstream model id |
| `api_key_env` | cloud | Name of the environment variable holding the key |
| `temperature` | all | Optional request default |
| `top_p` | all | Optional request default |
| `max_tokens` | cloud | Optional cap, useful for frontier cost control |
| `timeout_s` | all | Optional per-profile timeout |
| `num_ctx_default` | Ollama/LM Studio | Local context default |
| `num_ctx_overrides` | Ollama/LM Studio | Per-model local context overrides |

## Roles

`role_profiles` chooses which configured profile is used for each CoreThread role:

- `local`: model that answers first.
- `judge`: model that scores the local answer.
- `frontier`: fallback model used on low confidence or local failure.

The judge can be local, OpenAI, or OpenRouter. A cheap cloud judge is useful when
you want stronger scoring than a small local model without paying frontier prices
for every answer.

## Routing

```yaml
routing:
  threshold: 0.7
  constraint_prompt: "Be concise."
```

If the judge returns `pass=false` or a confidence score below `threshold`,
CoreThread pivots to the frontier profile.

## UI

```yaml
ui:
  theme: system
```

`theme` can be `system`, `light`, or `dark`.

## Privacy

```yaml
privacy:
  capture_transcripts: false
  transcript_max: 25
```

Transcript capture is disabled by default. See [PRIVACY.md](PRIVACY.md).

## Controls

```yaml
controls:
  requests_per_minute: null
  daily_request_quota: null
  daily_token_quota: null
  daily_cost_quota_usd: null
  pricing:
    "*":
      input_per_1m_tokens: 0
      output_per_1m_tokens: 0
  audit_enabled: false
  audit_path: outputs/audit.jsonl
  audit_include_request_body: false
```

`null` limits are disabled. When set, controls are global to the local
CoreThread process: they are guardrails for a personal router, not per-user
multi-tenant policy.

`pricing` is used for estimated cost tracking and `daily_cost_quota_usd`.
Use exact model ids as keys, or `*` as a fallback rate. Keep rates in config so
operators can update them when their provider pricing changes.

Audit logs are append-only JSONL metadata by default. They do not include prompt
or response bodies unless `audit_include_request_body` is explicitly set to
`true`.

## Example Configs

- `config.yaml.example`: default Ollama + OpenAI setup.
- `examples/config.ollama-openai.yaml`: Ollama local, Ollama judge, OpenAI frontier.
- `examples/config.lmstudio-openai.yaml`: LM Studio local, LM Studio judge, OpenAI frontier.
- `examples/config.lmstudio-openai-judge.yaml`: LM Studio local, cheap OpenAI judge, OpenAI frontier.
- `examples/config.openrouter.yaml`: Ollama local, Ollama judge, OpenRouter frontier.
