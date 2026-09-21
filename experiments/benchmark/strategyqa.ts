/**
 * StrategyQA loader — binary yes/no questions about implicit multi-step
 * reasoning, and the dataset that gives the `noul` primitive its
 * calibration labels (PLAN §Benchmark).
 *
 * Source: `strategyqa_train.json` from the official repository (the
 * ReWOO subset also draws from the train file), downloaded by
 * `scripts/fetch.ts` into `experiments/dataset/strategyqa/`.
 *
 * @module JevLoop/strategyqa
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { Benchmark, Task } from './task.ts'

const SYSTEM =
  'Answer the question with only the single word Yes or No. Do not explain.'

/** A free-form answer still counts when it leads with the verdict word. */
function yesNo(answer: string): string | null {
  const m = answer.toLowerCase().match(/\b(yes|no)\b/)
  return m ? m[1]! : null
}

export const strategyqa: Benchmark = {
  name: 'strategyqa',
  split: 'train',
  version: 'recorded-at-fetch',
  async load(dir: string): Promise<Task[]> {
    const raw = JSON.parse(readFileSync(join(dir, 'strategyqa_train.json'), 'utf8')) as Array<{
      question: string
      answer: boolean
      facts?: string[]
      id?: string
    }>
    return raw
      .filter((item) => typeof item.question === 'string')
      .map((item, i) => ({
        id: `strategyqa_${item.id ?? String(i).padStart(4, '0')}`,
        question: item.question.trim(),
        gold: [item.answer ? 'yes' : 'no'],
        meta: { facts: item.facts ?? null },
      }))
  },
  prompts: {
    build(task: Task): { system: string; user: string } {
      return { system: SYSTEM, user: task.question }
    },
  },
  score(task: Task, answer: string): boolean {
    const verdict = yesNo(answer)
    return verdict !== null && verdict === task.gold[0]?.toLowerCase()
  },
}
