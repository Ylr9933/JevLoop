/**
 * Minimal OpenAI-compatible client for the experiment arms.
 *
 * Two shapes on purpose: `/chat/completions` (most providers) and
 * `/responses` (gateways that only speak the newer wire API). The arm
 * picks nothing — the runner probes the endpoint once and passes the
 * result down, so the same accounting covers both.
 *
 * No dependency, and no key ever written inside this repo: the token comes
 * from `LLM_TOKEN`, or from a file outside the repo via `LLM_TOKEN_FILE`
 * (either a toml with `..._token = "..."` lines or plain `KEY=VALUE`).
 * Anything logged is masked by `maskToken` first.
 *
 * @module JevLoop/llm-client
 */

import { readFileSync } from 'node:fs'

/** Everything one model call must report back — becomes one `UsageRow`. */
export interface LlmCallResult {
  text: string
  succeeded: boolean
  latency_ms: number
  ttft_ms: number | null
  input_tokens_uncached: number
  input_tokens_cached: number
  output_tokens_visible: number
  output_tokens_reasoning: number
  usd: number
  retries: number
  retry_ms: number
  /** Last error verbatim when `succeeded` is false — never a paraphrase. */
  error: string | null
}

export interface LlmClientConfig {
  baseUrl: string
  apiKey: string
  model: string
  /** Which wire the endpoint speaks; `probeWire` decides, nobody guesses. */
  wire: 'chat' | 'responses'
  temperature: number
  topP: number
  maxTokens: number
  timeoutMs: number
  /** Retries for 429 / 5xx / transport errors, with linear backoff. */
  maxRetries: number
  /** Priced per 1M tokens; only feeds the `usd` column, kept in one place. */
  pricePerMTokensIn: number
  pricePerMTokensOut: number
}

/** Strip a bearer token to the only form allowed into cmd.txt / meta / stderr. */
export function maskToken(token: string): string {
  if (token.length <= 8) return '***'
  return `${token.slice(0, 4)}***${token.slice(-3)}`
}

/**
 * Extract the token from a file without printing it. Accepts both a toml
 * fragment (`experimental_bearer_token = "..."`, `api_key = "..."`) and
 * dotenv lines (`SOME_KEY=...`). Returns '' when nothing resolves —
 * the caller must refuse to run, never run keyless.
 */
export function readTokenFile(path: string): string {
  let raw: string
  try {
    raw = readFileSync(path, 'utf8')
  } catch {
    return ''
  }
  const toml = raw.match(/^\s*(?:experimental_bearer_token|api_key|bearer_token)\s*=\s*"?([^"\n]+)"?/m)
  if (toml) return toml[1]!.trim()
  const dotenv = raw.match(/^\s*(?:LLM_TOKEN|API_KEY|DEEPSEEK_API_KEY)\s*=\s*([^\s\n]+)/m)
  return dotenv ? dotenv[1]!.trim() : ''
}

/** Uniform result assembler so both wires report identical accounting. */
function assembled(t0: number, out: any, cfg: LlmClientConfig, retries: number, retryMs: number): LlmCallResult {
  const promptTokens = out?.usage?.prompt_tokens ?? out?.usage?.input_tokens ?? 0
  const completionTokens = out?.usage?.completion_tokens ?? out?.usage?.output_tokens ?? 0
  const reasoningTokens = out?.usage?.completion_tokens_details?.reasoning_tokens ?? 0
  const cached = out?.usage?.prompt_tokens_details?.cached_tokens ?? 0
  const visible = Math.max(0, completionTokens - reasoningTokens)
  return {
    text: out.__text ?? '',
    succeeded: true,
    latency_ms: performance.now() - t0,
    ttft_ms: null,
    input_tokens_uncached: promptTokens - cached,
    input_tokens_cached: cached,
    output_tokens_visible: visible,
    output_tokens_reasoning: reasoningTokens,
    usd: (promptTokens / 1e6) * cfg.pricePerMTokensIn + (completionTokens / 1e6) * cfg.pricePerMTokensOut,
    retries,
    retry_ms: retryMs,
    error: null,
  }
}

/**
 * One call, either wire, with retries on 429 / 5xx / transport errors and
 * a recorded reason on every other failure. Non-2xx that is not retriable
 * breaks the loop immediately — retrying a 401 would only burn quota.
 */
export async function callLlm(cfg: LlmClientConfig, system: string, user: string): Promise<LlmCallResult> {
  const url = cfg.wire === 'chat' ? `${cfg.baseUrl.replace(/\/+$/, '')}/chat/completions` : `${cfg.baseUrl.replace(/\/+$/, '')}/responses`
  const body =
    cfg.wire === 'chat'
      ? {
          model: cfg.model,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user },
          ],
          temperature: cfg.temperature,
          top_p: cfg.topP,
          max_tokens: cfg.maxTokens,
          stream: false,
        }
      : {
          model: cfg.model,
          instructions: system,
          input: [{ role: 'user', content: user }],
          temperature: cfg.temperature,
          top_p: cfg.topP,
          max_output_tokens: cfg.maxTokens,
          stream: false,
        }
  let retries = 0
  let retryMs = 0
  let lastError: string | null = null
  let carriedText = ''
  for (let attempt = 0; attempt <= cfg.maxRetries; attempt++) {
    if (attempt > 0) {
      const backoff = 2000 * attempt
      await new Promise((r) => setTimeout(r, backoff))
      retryMs += backoff
      retries++
    }
    const t0 = performance.now()
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), cfg.timeoutMs)
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${cfg.apiKey}` },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      })
      if (!res.ok) {
        const errBody = (await res.text()).slice(0, 300)
        lastError = `HTTP ${res.status}: ${errBody}`
        if (res.status === 429 || res.status >= 500) continue
        break
      }
      const out: any = await res.json()
      if (cfg.wire === 'chat') {
        const msg = out?.choices?.[0]?.message ?? {}
        carriedText = typeof msg?.content === 'string' ? msg.content : String(msg?.content ?? '')
      } else {
        const parts = (out?.output ?? []).filter((o: any) => o?.type === 'message')
        carriedText = parts
          .flatMap((p: any) => (p?.content ?? []).filter((c: any) => c?.type === 'output_text').map((c: any) => c?.text ?? ''))
          .join('')
      }
      return assembled(t0, { ...out, __text: carriedText }, cfg, retries, retryMs)
    } catch (err) {
      lastError = `transport: ${(err as Error).message}`
      continue
    } finally {
      clearTimeout(timer)
    }
  }
  return {
    text: '',
    succeeded: false,
    latency_ms: 0,
    ttft_ms: null,
    input_tokens_uncached: 0,
    input_tokens_cached: 0,
    output_tokens_visible: 0,
    output_tokens_reasoning: 0,
    usd: 0,
    retries,
    retry_ms: retryMs,
    error: lastError,
  }
}

/**
 * Probe which wire the endpoint actually serves: one 8-token request via
 * chat/completions, and fall back to /responses only on failure. The docs
 * of a gateway are a claim; this is a measurement.
 */
export async function probeWire(cfg: LlmClientConfig): Promise<{ wire: 'chat' | 'responses'; probe: LlmCallResult }> {
  const viaChat = await callLlm({ ...cfg, wire: 'chat', maxTokens: 8, maxRetries: 0 }, 'You are a connectivity probe.', 'Reply with the single word: pong')
  if (viaChat.succeeded) return { wire: 'chat', probe: viaChat }
  const viaResponses = await callLlm({ ...cfg, wire: 'responses', maxTokens: 8, maxRetries: 1 }, 'You are a connectivity probe.', 'Reply with the single word: pong')
  if (viaResponses.succeeded) return { wire: 'responses', probe: viaResponses }
  return { wire: 'chat', probe: viaChat }
}
