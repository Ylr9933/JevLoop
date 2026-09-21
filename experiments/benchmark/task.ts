/**
 * The contract every benchmark loader implements.
 *
 * README §3 fixed the file names; this file fixes the shape behind them.
 * A benchmark owns three things and nothing else:
 *
 *   · `load`     — files under `experiments/dataset/<name>/` into `Task[]`
 *   · `prompt`   — how the question is asked in the **direct** arm;
 *                  arms that plan de-statify this themselves, off `Task`
 *   · `score`    — answer + gold into one boolean, offline, no model
 *
 * Scoring stays offline on purpose: a model-judged scorer inside the
 * benchmark would make every table depend on the scorer's provider and
 * version, and README §2 already fixes the scorer as a shared seam for
 * the arms that need one.
 *
 * @module JevLoop/task
 */

import type { Task } from '../scripts/spec.ts'

export type { Task }

/** What a benchmark module must export for the runner to drive it. */
export interface Benchmark {
  /** Directory name under `experiments/dataset/`. */
  readonly name: string
  /** The split we actually run, e.g. `test` / `dev` / `train`. */
  readonly split: string
  /** Version string for `meta.json`, sourced from the download, not memory. */
  readonly version: string
  /** Read and parse the dataset directory into tasks, in dataset order. */
  load(dir: string): Promise<Task[]>
  /** The prompts of the direct arm. Kept here so Dataset-specific phrasing. */
  readonly prompts: {
    /** Prompt used for every task in this dataset. */
    build(task: Task): { system: string; user: string }
  }
  /** Exact-match scoring against gold; never touches a model. */
  score(task: Task, answer: string): boolean
}

/**
 * Lowercase, strip articles/punct/whitespace — the SQuAD normalization
 * family. Shared so EM columns mean the same thing across datasets.
 */
export function normalizeAnswer(s: string): string {
  return s
    .toLowerCase()
    .replace(/\b(a|an|the)\b/g, ' ')
    .replace(/[^a-z0-9 ]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Normalized exact match against any of the gold forms. */
export function emScore(task: Task, answer: string): boolean {
  const norm = normalizeAnswer(answer)
  if (!norm) return false
  return task.gold.some((g) => norm === normalizeAnswer(g)) || false
}

/**
 * Pull the first `NUMBER` out of `answer` (digits, commas, decimals),
 * normalized for comparison. Math datasets score on the number, not the
 * sentence around it.
 */
export function extractNumber(answer: string): string | null {
  const m = answer.replace(/,/g, '').match(/-?\d+(?:\.\d+)?/g)
  if (!m || m.length === 0) return null
  return m[m.length - 1]!.replace(/\.0+$/, '')
}
