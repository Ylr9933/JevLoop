/**
 * StrategyQA loader — binary yes/no questions about implicit multi-step
 * reasoning, and the dataset that gives the `noul` primitive its
 * calibration labels (PLAN §Benchmark).
 *
 * Two input shapes, tried in order: the official `strategyqa_train.json`
 * (array of `{question, answer: boolean, facts}`) or a datasets-server
 * rows export (array where `answer` may be boolean or "true"/"false"
 * strings). Provenance of whichever served is in SOURCES.json — numbers
 * cite it, not this comment.
 *
 * @module JevLoop/strategyqa
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { Benchmark, Task } from './task.ts'

const SYSTEM =
  'Answer the question with only the single word Yes or No. Do not explain.'

/** Both known shapes of this dataset: official JSON and rows exports. */
function parseOfficial(items: unknown): Task[] {
  const arr = Array.isArray(items) ? items : []
  return arr
    .map((item, i): Task | null => {
      const o = item as { question?: unknown; answer?: unknown; facts?: unknown; id?: unknown }
      if (typeof o.question !== 'string' || o.question.trim() === '') return null
      let yes: boolean | null = null
      if (typeof o.answer === 'boolean') yes = o.answer
      else if (typeof o.answer === 'string') {
        const a = o.answer.trim().toLowerCase()
        if (a === 'true' || a === 'yes') yes = true
        else if (a === 'false' || a === 'no') yes = false
      }
      if (yes === null) return null
      const idx = typeof o.id === 'string' && o.id !== '' ? o.id : String(i).padStart(4, '0')
      return {
        id: `strategyqa_${idx}`,
        question: o.question.trim(),
        gold: [yes ? 'yes' : 'no'],
        meta: { facts: Array.isArray(o.facts) ? (o.facts as string[]) : null },
      }
    })
    .filter((t): t is Task => t !== null)
}

export const strategyqa: Benchmark = {
  name: 'strategyqa',
  split: 'train',
  version: 'see experiments/dataset/strategyqa/SOURCES.json',
  async load(dir: string): Promise<Task[]> {
    try {
      return parseOfficial(JSON.parse(readFileSync(join(dir, 'strategyqa_train.json'), 'utf8')) as unknown)
    } catch {
      const exported = JSON.parse(readFileSync(join(dir, 'strategyqa_rows.json'), 'utf8')) as { rows: unknown[] }
      return parseOfficial(exported.rows)
    }
  },
  prompts: {
    build(task: Task): { system: string; user: string } {
      return { system: SYSTEM, user: task.question }
    },
  },
  score(task: Task, answer: string): boolean {
    const verdict = answer.toLowerCase().match(/\b(yes|no)\b/)
    return verdict !== null && verdict[1] === task.gold[0]?.toLowerCase()
  },
}
