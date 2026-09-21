/**
 * Unit tests for the experiment harness's pure functions.
 * No network, no datasets, no API key — the properties that make runs
 * reproducible are exactly the ones that must hold under test.
 *
 *   node --experimental-strip-types --test tests/experiments.test.ts
 *
 * @module JevLoop/experiments.test
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { META_FIELDS, RESULT_FIELDS, metaViolations, mulberry32, promptHash, resultViolations, sampleTasks } from '../experiments/scripts/spec.ts'
import { extractNumber, normalizeAnswer } from '../experiments/benchmark/task.ts'
import { gsm8k } from '../experiments/benchmark/gsm8k.ts'
import { strategyqa } from '../experiments/benchmark/strategyqa.ts'
import { maskToken, readTokenFile } from '../experiments/scripts/llm-client.ts'

const tasks = (n: number) => Array.from({ length: n }, (_, i) => ({ id: `t${i}`, question: `q${i}`, gold: [`${i}`] }))

test('sampleTasks is deterministic: same seed, same subset, same order', () => {
  const a = sampleTasks(tasks(100), 10, 7).map((t) => t.id)
  const b = sampleTasks(tasks(100), 10, 7).map((t) => t.id)
  assert.deepEqual(a, b)
  assert.equal(new Set(a).size, 10, 'no duplicates in the sample')
})

test('sampleTasks seeds diverge, and count over the pool returns everything', () => {
  const a = sampleTasks(tasks(100), 10, 1).map((t) => t.id)
  const b = sampleTasks(tasks(100), 10, 2).map((t) => t.id)
  assert.notDeepEqual(a, b)
  const all = sampleTasks(tasks(5), 50, 3)
  assert.equal(all.length, 5)
})

test('sampleTasks keeps dataset order for paired comparisons', () => {
  const sampled = sampleTasks(tasks(50), 10, 9).map((t) => Number(t.id.slice(1)))
  const sorted = [...sampled].sort((x, y) => x - y)
  assert.deepEqual(sampled, sorted)
})

test('mulberry32 is deterministic per seed', () => {
  const a = mulberry32(42)
  const b = mulberry32(42)
  const c = mulberry32(43)
  const seq = (r: () => number) => Array.from({ length: 5 }, () => r())
  assert.deepEqual(seq(a), seq(b))
  assert.notDeepEqual(seq(a), seq(c))
})

test('promptHash is stable and distinguishes prompts', () => {
  assert.equal(promptHash('s', 'u'), promptHash('s', 'u'))
  assert.notEqual(promptHash('s', 'u'), promptHash('s', 'u2'))
  assert.notEqual(promptHash('s', 'u'), promptHash('s2', 'u'))
})

test('resultViolations rejects rows with missing fields and accepts complete ones', () => {
  const row: Record<string, unknown> = {}
  for (const f of RESULT_FIELDS) row[f] = null
  row['time'] = { wall_ms: 1, model_ms: { handshake: 0, ttft: null, after_ttft: 0 }, decision_ms: { handshake: 0, compute: 0 }, tool_ms: 0, framework_ms: 0, retry_ms: 0, round_trips: 1 }
  assert.deepEqual(resultViolations(row), [])
  delete row['usd']
  assert.ok(resultViolations(row).includes('usd'))
  delete row['time']
  assert.ok(resultViolations(row).includes('time'))
})

test('metaViolations mirrors RESULT_FIELDS contract', () => {
  const meta: Record<string, unknown> = {}
  for (const f of META_FIELDS) meta[f] = null
  assert.deepEqual(metaViolations(meta), [])
  delete meta['git_commit']
  assert.deepEqual(metaViolations(meta), ['git_commit'])
})

test('normalizeAnswer strips case, articles and punctuation like SQuAD', () => {
  assert.equal(normalizeAnswer('The Eiffel Tower'), 'eiffel tower')
  assert.equal(normalizeAnswer('eiffel tower!'), 'eiffel tower')
  assert.equal(normalizeAnswer('A  B   C'), 'b c')
})

test('extractNumber takes the last number and drops comma separators', () => {
  assert.equal(extractNumber('so it is 1,234.5 units'), '1234.5')
  assert.equal(extractNumber('minus -3 here'), '-3')
  assert.equal(extractNumber('no digits'), null)
})

test('gsm8k.score accepts the stated format and format-independent numbers', () => {
  const task = { id: 'x', question: 'q', gold: ['72'] }
  assert.equal(gsm8k.score(task, 'blah blah\n\nThe answer is 72'), true)
  assert.equal(gsm8k.score(task, 'The answer is 72.0'), true)
  assert.equal(gsm8k.score(task, '7+2 make 71, actually 72.'), true)
  assert.equal(gsm8k.score(task, 'The answer is 73'), false)
  assert.equal(gsm8k.score(task, 'I cannot compute'), false)
})

test('strategyqa.score matches the leading yes/no verdict only', () => {
  const yesTask = { id: 'x', question: 'q', gold: ['yes'] }
  assert.equal(strategyqa.score(yesTask, 'Yes.'), true)
  assert.equal(strategyqa.score(yesTask, 'The answer is yes'), true)
  assert.equal(strategyqa.score(yesTask, 'No'), false)
  assert.equal(strategyqa.score(yesTask, 'maybe'), false)
})

test('maskToken never emits the middle of a token', () => {
  assert.equal(maskToken('short'), '***')
  const masked = maskToken('abcdefghijklmnop')
  assert.ok(masked.startsWith('abcd') && masked.endsWith('nop'))
  assert.ok(!masked.includes('efghijklm'))
})

test('readTokenFile reads toml bearer tokens and dotenv keys, and nothing else', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'jevexp-'))
  try {
    const toml = join(dir, 'config.toml')
    await writeFile(toml, 'model_provider = "ZAI"\nexperimental_bearer_token = "secret-secret-secret"\n')
    assert.equal(readTokenFile(toml), 'secret-secret-secret')
    const dotenv = join(dir, 'keys.env')
    await writeFile(dotenv, '# comment\nDEEPSEEK_API_KEY=abc123def456\n')
    assert.equal(readTokenFile(dotenv), 'abc123def456')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})
