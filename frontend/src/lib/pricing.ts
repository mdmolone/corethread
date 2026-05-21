// Phase 10 / Plan 10-04 / D-21
// Last verified: 2026-05-05 against https://openai.com/api/pricing
//
// Pitfall 24 defense: this is a MODULE-LEVEL CONST baked into the SPA bundle.
// Do NOT migrate to a VITE_GPT4O_INPUT_PRICE env var — VITE_ prefixed vars are
// dumped into the prod bundle visible to anyone with DevTools, which primes
// devs to put OTHER secrets in VITE_. The price table is public information,
// but the pattern matters.
//
// Phase 11 PKG-04 README quickstart will document: "GPT-4o price table is
// baked into the SPA bundle — update frontend/src/lib/pricing.ts when OpenAI
// changes prices and rebuild."
//
// Backend-served /v1/pricing endpoint rejected (out of scope; ARC-01 limits
// Phase 10's backend touches to CI workflow YAML only).
//
// Canonical published rates (USD per 1M tokens):
//   input_per_1m_tokens: 2.50
//   output_per_1m_tokens: 10.00
// (Prettier collapses numeric literals to 2.5 / 10.0 — the canonical $2.50 /
// $10.00 form is preserved here for grep.)

export const GPT4O_PRICING = {
  input_per_1m_tokens: 2.5,
  output_per_1m_tokens: 10.0,
} as const

/**
 * D-20 cost math: per-non-pivoted-trace estimated GPT-4o cost-avoided.
 *
 * For each trace t in window where !t.pivoted:
 *   counterfactual_cost
 *     = (t.input_tokens / 1_000_000) * GPT4O_PRICING.input_per_1m_tokens
 *     + (t.output_tokens / 1_000_000) * GPT4O_PRICING.output_per_1m_tokens
 *
 * Pure function — no side effects, no external state. Unit-testable in isolation.
 */
export function estimateCost(
  input_tokens: number,
  output_tokens: number,
): number {
  return (
    (input_tokens * GPT4O_PRICING.input_per_1m_tokens +
      output_tokens * GPT4O_PRICING.output_per_1m_tokens) /
    1_000_000
  )
}
