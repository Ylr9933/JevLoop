/**
 * JevLoop arm — the kernel's own loop, with a real Jev decider and a real
 * generator. This is the arm the paper's Table 2 row is built from: the
 * judgement calls (needsTool / canDeliver / ...) are answered by Jev, and
 * the one expensive step (writing the answer) by the generator.
 *
 * It reuses the kernel verbatim — `runAgent` from `src/agent.ts`, the
 * `HttpProvider` from `src/provider-http.ts`, the `HttpGenerator` from
 * `src/llm.ts` — so the arm measures the real loop, not a re-implementation.
 * The runner owns run-level accounting; this module is a pure "task in,
 * outcome out" transform, like `direct.ts`.
 *
 * `confidence` is null in the row for now: the meter records an `answers`
 * summary per decision, not a numeric confidence, and RQ2 (per-decision
 * calibration) is a later phase that reads `trace.jsonl`. What this arm
 * does record honestly: how many model calls happened (1 = the gate
 * passed first try, 2 = it rejected once and revised), how many decision
 * batches ran, and whether any decision escalated.
 *
 * @module JevLoop/jevloop-baseline
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Decider, Meter, runAgent } from '../../src/index.ts'
import { HttpProvider } from '../../src/provider-http.ts'
import { HttpGenerator } from '../../src/llm.ts'

import type { LlmClientConfig } from '../scripts/llm-client.ts'
import type { Task } from '../scripts/spec.ts'
import type { Benchmark } from '../benchmark/task.ts'

/** Jev decision backend configuration (separate from the generator). */
export interface JevConfig {
  baseUrl: string
  apiKey: string
  model: string
  timeoutMs: number
}

export interface JevLoopOutcome {
  answer: string
  halt: string
  steps: number
  /** Wall clock for the whole runAgent call, end to end. */
  wallMs: number
  decisionCount: number
  decisionBatches: number
  decisionMs: number
  modelCalls: number
  modelMs: number
  inputTokens: number
  outputTokens: number
  escalated: number
  degraded: number
}

/** One temp cwd per run, shared across tasks (needsTool should be false on a tool-less task, so no tool touches it). */
let sharedCwd: string | null = null
async function cwdForRun(): Promise<string> {
  if (sharedCwd) return sharedCwd
  sharedCwd = await mkdtemp(join(tmpdir(), 'jevloop-arm-'))
  return sharedCwd
}

/** Drop the shared cwd; the runner calls this in its finally block. */
export async function disposeJevLoopArm(): Promise<void> {
  if (sharedCwd) {
    await rm(sharedCwd, { recursive: true, force: true })
    sharedCwd = null
  }
}

/**
 * Run the kernel loop on one task. The generator is the same DeepSeek
 * endpoint the direct arm uses (paired comparison); the decider is Jev.
 * `maxSteps` is kept low because a tool-less task should finish in one or
 * two steps — a run that hits the cap is itself a finding.
 */
export async function runJevLoop(
  genCfg: LlmClientConfig,
  jevCfg: JevConfig,
  _bench: Benchmark,
  task: Task,
  maxSteps = 8,
): Promise<JevLoopOutcome> {
  const cwd = await cwdForRun()
  const provider = new HttpProvider({
    baseUrl: jevCfg.baseUrl,
    apiKey: jevCfg.apiKey,
    defaultModel: jevCfg.model,
    name: 'jev',
    timeoutMs: jevCfg.timeoutMs,
  })

  const t0 = performance.now()
  let result
  let meter = new Meter()
  // The kernel's HttpGenerator does not retry on 429/transport (it throws);
  // for a 200-task sweep that is too fragile against gateway rate limits, so
  // the whole runAgent call is retried here with linear backoff. The meter
  // and decider are rebuilt per attempt so a partial run (decisions recorded
  // before the generator 429'd) is not double-counted into the next attempt.
  // Only retriable classes (rate-limit / transport) retry; a 400/401 would
  // loop forever otherwise, so non-retriable errors break out immediately.
  const maxRetries = 4
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    meter = new Meter()
    const decider = new Decider({ provider, meter })
    const generator = new HttpGenerator({
      baseUrl: genCfg.baseUrl,
      apiKey: genCfg.apiKey,
      model: genCfg.model,
      name: `http(${genCfg.model})`,
      timeoutMs: genCfg.timeoutMs,
    })
    try {
      result = await runAgent({
        task: task.question,
        cwd,
        decider,
        generator,
        maxSteps,
        onAskHuman: async () => false,
      })
      break
    } catch (err) {
      const msg = (err as Error).message
      const retriable = /429|RATE_LIMIT|RateLimit|concurrent|ETIMEDOUT|ECONNRESET|fetch failed|UND_ERR/.test(msg)
      if (!retriable || attempt === maxRetries) {
        return {
          answer: '',
          halt: `error: ${msg.slice(0, 200)}`,
          steps: 0,
          wallMs: performance.now() - t0,
          decisionCount: meter.decisions.length,
          decisionBatches: new Set(meter.decisions.map((d) => d.batch)).size,
          decisionMs: meter.stats.decisionMs,
          modelCalls: meter.modelCalls.length,
          modelMs: meter.stats.modelMs,
          inputTokens: meter.stats.inputTokens ?? 0,
          outputTokens: meter.stats.outputTokens ?? 0,
          escalated: meter.stats.escalated,
          degraded: meter.stats.degraded,
        }
      }
      await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)))
    }
  }
  const wallMs = performance.now() - t0
  const stats = meter.stats
  return {
    answer: result!.answer,
    halt: result!.halt,
    steps: result!.steps,
    wallMs,
    decisionCount: meter.decisions.length,
    decisionBatches: new Set(meter.decisions.map((d) => d.batch)).size,
    decisionMs: stats.decisionMs,
    modelCalls: meter.modelCalls.length,
    modelMs: stats.modelMs,
    inputTokens: stats.inputTokens ?? 0,
    outputTokens: stats.outputTokens ?? 0,
    escalated: stats.escalated,
    degraded: stats.degraded,
  }
}
