/**
 * HotpotQA loader — the flagship row of the ReWOO comparison, so the
 * direct arm on this dataset is the floor the tool arms must beat
 * (PLAN §Benchmark: 5x tokens / +4% acc lives on these questions).
 *
 * Source: the official `hotpot_dev_distractor_v1.json`, downloaded by
 * `scripts/fetch.ts` into `experiments/dataset/hotpotqa/`. The `answer`
 * field is the gold surface form; scoring is normalized exact match,
 * the metric the original paper reports.
 *
 * @module JevLoop/hotpotqa
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { Benchmark, Task } from './task.ts'
import { emScore } from './task.ts'

const SYSTEM =
  'Answer the multi-hop question with only the short factual answer — an entity, a person, ' +
  'a date, or a short phrase. No explanation, no full sentence.'

export const hotpotqa: Benchmark = {
  name: 'hotpotqa',
  split: 'dev_distractor',
  version: 'see experiments/dataset/hotpotqa/SOURCES.json (official file preferred over a rows export)',
  async load(dir: string): Promise<Task[]> {
    try {
      const raw = JSON.parse(readFileSync(join(dir, 'hotpot_dev_distractor_v1.json'), 'utf8')) as Array<{
        _id: string
        question: string
        answer: string
        level?: string
        type?: string
      }>
      return raw.map((item) => ({
        id: `hotpotqa_${item._id}`,
        question: item.question.trim(),
        gold: [item.answer.trim()],
        meta: { level: item.level ?? null, type: item.type ?? null, source: 'official_dev_json' },
      }))
    } catch {
      const exported = JSON.parse(readFileSync(join(dir, 'hotpot_rows.json'), 'utf8')) as {
        rows: Array<{ id: string; question: string; answer: string; level?: string; type?: string }>
      }
      return exported.rows.map((item) => ({
        id: `hotpotqa_${item.id}`,
        question: String(item.question).trim(),
        gold: [String(item.answer).trim()],
        meta: { level: item.level ?? null, type: item.type ?? null, source: 'datasets_server_rows' },
      }))
    }
  },
  prompts: {
    build(task: Task): { system: string; user: string } {
      return { system: SYSTEM, user: task.question }
    },
  },
  score(task: Task, answer: string): boolean {
    const stripped = answer.split('\n')[0] ?? answer
    return emScore(task, stripped.length > 0 ? stripped : answer)
  },
}
