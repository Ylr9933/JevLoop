/**
 * One-shot probe: does the Jev API key in `study/config/.env` (var
 * `jev_api_key`) actually answer a `noul` question through the kernel's
 * own `HttpProvider`? No guessing the wire — this is the same path the
 * jevloop arm will take. Prints only: ok/fail, latency, the returned
 * probability, and a masked key. Never prints the raw key.
 *
 *   node --experimental-strip-types experiments/scripts/probe-jev.ts
 *
 * @module JevLoop/probe-jev
 */

import { readFileSync } from 'node:fs'
import { HttpProvider } from '../../src/provider-http.ts'

function loadKey(): string {
  const raw = readFileSync(process.env.JEV_KEY_FILE ?? '/Users/ylr9933/study/config/.env', 'utf8')
  const m = raw.match(/^\s*jev_api_key\s*=\s*"?"?([^"\n]+)"?"?/m)
  if (!m) throw new Error('no jev_api_key in the file')
  return m[1]!.trim()
}

const key = loadKey()
const masked = key.length <= 8 ? '***' : `${key.slice(0, 4)}***${key.slice(-3)}`

const provider = new HttpProvider({
  baseUrl: process.env.JEV_BASE_URL ?? 'https://api.typesafe.ai',
  apiKey: key,
  name: 'jev-probe',
  defaultModel: process.env.JEV_MODEL ?? 'jev-latest',
  timeoutMs: 30_000,
})

const state = { task: 'What is 2 + 2?', context: 'A simple arithmetic question.' }
const questions = {
  is_math: {
    type: 'noul' as const,
    instructions: 'Is the state an arithmetic question?',
    criteria: { true: 'It asks to compute a numeric result', false: 'It is not about computation' },
  },
}

const t0 = performance.now()
try {
  const res = await provider.decide({ state, questions, timeoutMs: 30_000 })
  const dt = Math.round(performance.now() - t0)
  const a = res.answers['is_math']
  console.log(`jev probe: ok in ${dt}ms (provider ${res.provider}, model ${res.model ?? '?'}, degraded=${res.degraded})`)
  console.log(`  is_math.noul = ${a?.type === 'noul' ? a.noul : 'NOT-NOUL: ' + JSON.stringify(a)}`)
  console.log(`  key = ${masked}`)
} catch (err) {
  console.log(`jev probe: FAILED in ${Math.round(performance.now() - t0)}ms`)
  console.log(`  ${(err as Error).message}`)
  console.log(`  key = ${masked}`)
}
