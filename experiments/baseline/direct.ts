/**
 * Direct arm — the no-loop floor. One model call answers the question,
 * no tools, no retrieval, no planning.
 *
 * This is the arm every other arm must beat before its machinery earns
 * a place in the paper: if direct already scores high on a dataset, that
 * dataset does not pressure the loop and the appendix gets its downhill
 * row (PLAN §2, Table 1). It is also the arm used to decide which
 * datasets survive the first cut — a so-easy-that-direct-solves-it
 * dataset is a benchmark of nothing.
 *
 * @module JevLoop/direct
 */

import type { LlmCallResult, LlmClientConfig } from '../scripts/llm-client.ts'
import { callLlm } from '../scripts/llm-client.ts'
import type { Task } from '../scripts/spec.ts'
import type { Benchmark } from '../benchmark/task.ts'

/** What one task cost the arm: the call's accounting plus nothing else. */
export interface DirectOutcome {
  answer: string
  call: LlmCallResult
}

/**
 * Answer one task in one shot. Scoring is not here — the benchmark owns
 * `score`, and the runner owns run-level accounting, so the arm stays a
 * pure "prompt in, text out" transformation.
 */
export async function runDirect(cfg: LlmClientConfig, bench: Benchmark, task: Task): Promise<DirectOutcome> {
  const { system, user } = bench.prompts.build(task)
  const call = await callLlm(cfg, system, user)
  return { answer: call.text, call }
}
