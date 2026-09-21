/**
 * GSM8K loader — grade-school math, the expected "direct is already high"
 * row that keeps the appendix table honest (PLAN §Benchmark).
 *
 * Source: the raw `test.jsonl` from the official repository, one
 * `{"question", "answer"}` per line, gold after `####`. Downloaded by
 * `scripts/fetch.ts` into `experiments/dataset/gsm8k/`.
 *
 * @module JevLoop/gsm8k
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { Benchmark, Task } from './task.ts'
import { extractNumber } from './task.ts'

/** Gold answers sit behind `####` on the last reasoning line. */
function goldOf(answer: string): string {
  const idx = answer.lastIndexOf('####')
  return idx === -1 ? answer.trim() : answer.slice(idx + 4).trim()
}

const SYSTEM =
  'Solve the math problem. Think briefly, then end your reply with the final numeric result ' +
  'on its own last line in the exact form: "The answer is N".'

export const gsm8k: Benchmark = {
  name: 'gsm8k',
  split: 'test',
  version: 'recorded-at-fetch',
  async load(dir: string): Promise<Task[]> {
    const raw = readFileSync(join(dir, 'test.jsonl'), 'utf8')
    return raw
      .split('\n')
      .filter((l) => l.trim() !== '')
      .map((line, i) => {
        const item = JSON.parse(line) as { question: string; answer: string }
        return { id: `gsm8k_${String(i).padStart(4, '0')}`, question: item.question.trim(), gold: [goldOf(item.answer)] }
      })
  },
  prompts: {
    build(task: Task): { system: string; user: string } {
      return { system: SYSTEM, user: task.question }
    },
  },
  score(task: Task, answer: string): boolean {
    const m = answer.match(/The answer is\s*(-?[0-9][0-9,]*)/i)
    const produced = m ? m[1]!.replace(/,/g, '') : extractNumber(answer)
    if (produced === null) return false
    const want = extractNumber(task.gold[0] ?? '')
    return want !== null && produced.replace(/\.0+$/, '') === want.replace(/\.0+$/, '')
  },
}
