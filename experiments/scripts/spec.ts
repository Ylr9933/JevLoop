/**
 * Result schema — the contract every arm answers to.
 *
 * This file is the freeze README §2 asked for: one shape for every run of
 * every arm on every dataset, so seven people can produce rows that join.
 * A result missing a field is rejected, not patched — see `assertResult`.
 *
 * Field-for-field this is `experiments/README.md` §4 plus the per-call
 * and per-decision records `PROTOCOL.md` §3.3/§3.4 store in the run dir.
 * The task-level `ResultRecord` is the summary; `trace.jsonl` (one row per
 * decision) and `usage.json` (one row per call) carry the evidence behind it.
 *
 * @module JevLoop/spec
 */

/** How we identify the model that answered. `version` is the provider's exact id string. */
export interface ModelRef {
  id: string
  version: string
  provider: string
}

/** One benchmark task, as the runner hands it to an arm. */
export interface Task {
  id: string
  question: string
  /** Every acceptable surface form of the gold answer. Empty = not scorable offline. */
  gold: readonly string[]
  /** Dataset-specific slot (e.g. the level for HotpotQA, the facts for StrategyQA). */
  meta?: Record<string, unknown>
}

/** One row of `trace.jsonl` — one decision, its confidence and its ground truth. */
export interface TraceRow {
  run_id: string
  task_id: string
  step: number
  /** Decision node id, e.g. `pickTool`. Empty for arms with no decisions. */
  node: string
  answer: string
  confidence: number | null
  correct: boolean | null
  /** How many questions shared the batch this decision was answered in. */
  batch: number
  latency_ms: number
}

/** One row of `usage.json` — one model call. */
export interface UsageRow {
  task_id: string
  call_index: number
  succeeded: boolean
  latency_ms: number
  ttft_ms: number | null
  input_tokens_cached: number
  input_tokens_uncached: number
  output_tokens_visible: number
  output_tokens_reasoning: number
  usd: number
}

/** Time attribution for one task, split the way README §4 demands. */
export interface TimeBreakdown {
  wall_ms: number
  model_ms: { handshake: number; ttft: number | null; after_ttft: number }
  decision_ms: { handshake: number; compute: number }
  tool_ms: number
  framework_ms: number
  retry_ms: number
  round_trips: number
}

/**
 * One task under one arm — the unit every table in `result/` is built from.
 * Field list is README §4 exactly; `assertResult` refuses anything missing.
 */
export interface ResultRecord {
  run_id: string
  dataset: string
  task_id: string
  arm: string
  seed: number
  model: ModelRef
  thinking_budget: string
  temperature: number
  top_p: number
  max_tokens: number
  prompt_hash: string
  dataset_version: string
  commit: string
  region: string
  cold_start: boolean
  success: boolean
  steps: number
  first_divergence_step: number | null
  escalated: boolean
  gate_false_reject: boolean | null
  gate_false_deny: boolean | null
  failure_class: string | null
  llm_calls: number
  decision_requests: number
  questions_per_request: number
  tool_calls: number
  tool_calls_necessary: number
  tool_calls_exploratory: number
  input_tokens_cached: number
  input_tokens_uncached: number
  output_tokens_reasoning: number
  output_tokens_visible: number
  usd: number
  time: TimeBreakdown
  /** Mean top-answer confidence over this task's decisions, when the arm has any. */
  confidence: number | null
  correct: boolean | null
  answer: string
  gold: string
}

/** `PROTOCOL.md` §3.2 — the metadata that makes a run reproducible or marks it as noise. */
export interface RunMeta {
  run_id: string
  started_at: string
  finished_at: string | null
  duration_ms: number | null
  git_commit: string
  /** True when the worktree had uncommitted changes — such numbers are void. */
  dirty: boolean
  dataset: string
  dataset_version: string
  dataset_split: string
  dump_version: string | null
  task_count: number
  sample_seed: number
  arm: string
  generator: ModelRef
  decider: ModelRef | null
  thinking_budget: string
  temperature: number
  top_p: number
  max_tokens: number
  prompt_hash: string
  decision_md_hash: string | null
  region: string
  cold_start: boolean
  max_steps: number | null
  timeout_s: number | null
  retry_policy: string
}

/** Every top-level field of `ResultRecord`, checked by `assertResult`. */
export const RESULT_FIELDS = [
  'run_id', 'dataset', 'task_id', 'arm', 'seed', 'model', 'thinking_budget', 'temperature',
  'top_p', 'max_tokens', 'prompt_hash', 'dataset_version', 'commit', 'region', 'cold_start',
  'success', 'steps', 'first_divergence_step', 'escalated', 'gate_false_reject', 'gate_false_deny',
  'failure_class', 'llm_calls', 'decision_requests', 'questions_per_request', 'tool_calls',
  'tool_calls_necessary', 'tool_calls_exploratory', 'input_tokens_cached', 'input_tokens_uncached',
  'output_tokens_reasoning', 'output_tokens_visible', 'usd', 'time', 'confidence', 'correct',
  'answer', 'gold',
] as const

/** Every top-level field of `RunMeta`, checked by `assertMeta`. */
export const META_FIELDS = [
  'run_id', 'started_at', 'finished_at', 'duration_ms', 'git_commit', 'dirty', 'dataset',
  'dataset_version', 'dataset_split', 'dump_version', 'task_count', 'sample_seed', 'arm',
  'generator', 'decider', 'thinking_budget', 'temperature', 'top_p', 'max_tokens',
  'prompt_hash', 'decision_md_hash', 'region', 'cold_start', 'max_steps', 'timeout_s',
  'retry_policy',
] as const

function missingField(obj: Record<string, unknown>, fields: readonly string[]): string[] {
  return fields.filter((f) => obj[f] === undefined)
}

/**
 * Reject a result row that is missing schema fields. Returns the violations
 * as strings (caller decides to throw); empty array = conforming.
 * `null` is a legal value — only `undefined` counts as missing.
 */
export function resultViolations(r: Record<string, unknown>): string[] {
  const missing = missingField(r, RESULT_FIELDS)
  if (jsonTimebreakdownIssues(r['time'] as Record<string, unknown> | undefined)) {
    missing.push('time.{wall_ms,model_ms,decision_ms,tool_ms,framework_ms,retry_ms,round_trips}')
  }
  return missing
}

/**
 * Same contract for `RunMeta`. An unstarted run has `finished_at: null` —
 * that is a legal value, only `undefined` is rejected.
 */
export function metaViolations(m: Record<string, unknown>): string[] {
  return missingField(m, META_FIELDS)
}

function jsonTimebreakdownIssues(time: Record<string, unknown> | undefined): boolean {
  if (!time) return true
  const need = ['wall_ms', 'model_ms', 'decision_ms', 'tool_ms', 'framework_ms', 'retry_ms', 'round_trips']
  return need.some((k) => time[k] === undefined)
}

/**
 * 32-bit deterministic PRNG — the same seed picks the same subset everywhere.
 * Sampling must be reproducible or a rerun is a different run.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Deterministically pick `count` tasks out of `all`. Same seed, same subset —
 * that is what makes arm-vs-arm comparisons paired.
 */
export function sampleTasks(all: readonly Task[], count: number, seed: number): Task[] {
  const rand = mulberry32(seed)
  const idx = all.map((_, i) => i)
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    const t = idx[i]!
    idx[i] = idx[j]!
    idx[j] = t
  }
  return idx.slice(0, count).sort((a, b) => a - b).map((i) => all[i]!)
}

/**
 * Short stable hash of the prompt a run used. Two runs with different prompts
 * must differ here — otherwise the table silently compares different things.
 */
export function promptHash(system: string, user: string): string {
  const input = `system:\n${system}\nuser:\n${user}`
  let h1 = 0x811c9dc5
  for (let i = 0; i < input.length; i++) {
    h1 ^= input.charCodeAt(i)
    h1 = Math.imul(h1, 0x01000193)
  }
  return (h1 >>> 0).toString(16).padStart(8, '0')
}
