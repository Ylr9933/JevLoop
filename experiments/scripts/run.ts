/**
 * The one entry every arm goes through: `run.ts --dataset gsm8k --arm direct`.
 *
 * Implements PROTOCOL.md mechanically, because that is the only way it
 * holds under time pressure:
 *
 *   · run dir exists BEFORE the first call runs        (§5 ③, 先建后跑)
 *   · cmd.txt carries the full command with masked keys (§3.1)
 *   · meta.json records commit + dirty + prompt hashes  (§3.2)
 *   · trace.jsonl exists even with zero rows — the direct arm decides
 *     nothing, and an absent file could be read as "forgot to record"  (§3.3)
 *   · usage rows are flushed per task, so a kill leaves a factual      (§3.4 / §8)
 *     stop point instead of a guess
 *   · exit.json is written on success AND on interruption              (§3.5)
 *
 * `result/` is not written here — `summarize.ts` generates it from the
 * run dir, one direction only (§6).
 *
 * @module JevLoop/run
 */

import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { execSync } from 'node:child_process'
import { join } from 'node:path'

import type { LlmClientConfig, LlmCallResult } from './llm-client.ts'
import { maskToken, probeWire, readTokenFile } from './llm-client.ts'
import type { ResultRecord, RunMeta, Task } from './spec.ts'
import { metaViolations, promptHash, resultViolations, sampleTasks } from './spec.ts'
import type { Benchmark } from '../benchmark/task.ts'
import { gsm8k } from '../benchmark/gsm8k.ts'
import { hotpotqa } from '../benchmark/hotpotqa.ts'
import { strategyqa } from '../benchmark/strategyqa.ts'
import { runDirect } from '../baseline/direct.ts'
import { disposeJevLoopArm, runJevLoop, type JevConfig } from '../baseline/jevloop.ts'
import { readJevKey } from './jev-key.ts'

/** Registry instead of dynamic import — names must stay the README §3 spelling. */
const BENCHMARKS: Record<string, Benchmark> = { gsm8k, hotpotqa, strategyqa }
const ARMS = ['direct', 'jevloop'] as const
type Arm = (typeof ARMS)[number]

interface Cli {
  dataset: string
  arm: string
  seed: number
  limit: number
  model: string
  baseUrl: string
  token?: string
  tokenFile?: string
  temperature: number
  topP: number
  maxTokens: number
  timeoutS: number
  maxRetries: number
  region: string
  priceIn: number
  priceOut: number
  jevKeyFile?: string
  jevBaseUrl: string
  jevModel: string
  maxSteps: number
}

function parseArgs(argv: string[]): Cli {
  const out: Record<string, string> = {}
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!
    if (!a.startsWith('--')) throw new Error(`unexpected argument: ${a}`)
    const key = a.slice(2)
    const val = argv[i + 1]
    if (val === undefined || val.startsWith('--')) throw new Error(`--${key} needs a value`)
    out[key] = val
    i++
  }
  const need = (k: string): string => {
    const v = out[k]
    if (!v) throw new Error(`missing required --${k}`)
    return v
  }
  const num = (k: string, dflt: number): number => (out[k] === undefined ? dflt : Number(out[k]))
  return {
    dataset: need('dataset'),
    arm: need('arm'),
    seed: num('seed', 0),
    limit: num('limit', 200),
    model: need('model'),
    baseUrl: process.env.LLM_BASE_URL ?? need('base-url'),
    token: process.env.LLM_TOKEN,
    tokenFile: out['token-file'] ?? process.env.LLM_TOKEN_FILE,
    temperature: num('temperature', 0),
    topP: num('top-p', 1),
    maxTokens: num('max-tokens', 1024),
    timeoutS: num('timeout-s', 120),
    maxRetries: num('max-retries', 3),
    region: out['region'] ?? 'unknown',
    priceIn: num('price-in', 0),
    priceOut: num('price-out', 0),
    jevKeyFile: out['jev-key-file'],
    jevBaseUrl: out['jev-base-url'] ?? 'https://api.typesafe.ai',
    jevModel: out['jev-model'] ?? 'jev-latest',
    maxSteps: num('max-steps', 8),
  }
}

function gitInfo(): { commit: string; dirty: boolean } {
  const inRepo = (cmd: string): string => {
    try {
      return execSync(cmd, { cwd: new URL('..', import.meta.url).pathname, encoding: 'utf8' }).trim()
    } catch {
      return ''
    }
  }
  const commit = inRepo('git rev-parse HEAD')
  const porcelain = inRepo('git status --porcelain --untracked-files=no')
  return { commit, dirty: porcelain.length > 0 }
}

function utcStamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-')
}

/** One noul question to Jev, via the kernel's own HttpProvider — the same path the arm takes. */
async function jevPing(jev: JevConfig): Promise<{ ok: boolean; model?: string; ms: number; reason?: string }> {
  const { HttpProvider } = await import('../../src/provider-http.ts')
  const provider = new HttpProvider({ baseUrl: jev.baseUrl, apiKey: jev.apiKey, defaultModel: jev.model, name: 'jev-probe', timeoutMs: jev.timeoutMs })
  const t0 = performance.now()
  try {
    const res = await provider.decide({
      state: { task: 'connectivity probe', context: 'ping' },
      questions: { up: { type: 'noul', instructions: 'Is this a connectivity probe?', criteria: { true: 'It is a ping', false: 'It is not a ping' } } },
      timeoutMs: jev.timeoutMs,
    })
    return { ok: !res.degraded, model: res.model, ms: Math.round(performance.now() - t0) }
  } catch (err) {
    return { ok: false, ms: Math.round(performance.now() - t0), reason: (err as Error).message.slice(0, 200) }
  }
}

class Tee {
  #out: (line: string) => void
  #err: (line: string) => void
  constructor(dir: string) {
    this.#out = (l) => {
      process.stdout.write(`${l}\n`)
      appendFileSync(join(dir, 'stdout.txt'), `${l}\n`)
    }
    this.#err = (l) => {
      process.stderr.write(`${l}\n`)
      appendFileSync(join(dir, 'stderr.txt'), `${l}\n`)
    }
  }
  out(l: string): void {
    this.#out(l)
  }
  err(l: string): void {
    this.#err(l)
  }
}

async function main(): Promise<number> {
  const cli = parseArgs(process.argv.slice(2))
  const bench = BENCHMARKS[cli.dataset]
  if (!bench) throw new Error(`unknown dataset '${cli.dataset}' (known: ${Object.keys(BENCHMARKS).join(', ')})`)
  if (!ARMS.includes(cli.arm as Arm)) throw new Error(`unknown arm '${cli.arm}' (known: ${ARMS.join(', ')})`)

  const apiKey = cli.token ?? (cli.tokenFile ? readTokenFile(cli.tokenFile) : '')
  if (!apiKey) throw new Error('no token: pass --token-file or set LLM_TOKEN (refusing to run keyless)')

  const runId = `${utcStamp()}Z-seed${cli.seed}`
  const runDir = join(new URL('..', import.meta.url).pathname, 'log', cli.dataset, cli.arm, runId)
  mkdirSync(runDir, { recursive: true })
  const tee = new Tee(runDir)

  const maskedEnv = [`LLM_BASE_URL=${cli.baseUrl}`, `LLM_TOKEN=${maskToken(apiKey)}`, cli.tokenFile ? `LLM_TOKEN_FILE=${cli.tokenFile}` : ''].filter(Boolean)
  writeFileSync(join(runDir, 'cmd.txt'), `node --experimental-strip-types experiments/scripts/run.ts ${process.argv.slice(2).join(' ')}\n# env\n${maskedEnv.join('\n')}\n`)
  writeFileSync(join(runDir, 'trace.jsonl'), '')

  const startedAt = new Date()
  const git = gitInfo()
  const all: Task[] = await bench.load(join(new URL('..', import.meta.url).pathname, 'dataset', cli.dataset))
  const tasks = sampleTasks(all, Math.min(cli.limit, all.length), cli.seed)
  const firstPrompt = bench.prompts.build(tasks[0] as Task)

  const meta: RunMeta = {
    run_id: runId,
    started_at: startedAt.toISOString(),
    finished_at: null,
    duration_ms: null,
    git_commit: git.commit,
    dirty: git.dirty,
    dataset: cli.dataset,
    dataset_version: bench.version,
    dataset_split: bench.split,
    dump_version: null,
    task_count: tasks.length,
    sample_seed: cli.seed,
    arm: cli.arm,
    generator: { id: cli.model, version: cli.model, provider: cli.baseUrl },
    decider: null,
    thinking_budget: 'off',
    temperature: cli.temperature,
    top_p: cli.topP,
    max_tokens: cli.maxTokens,
    prompt_hash: promptHash(firstPrompt.system, firstPrompt.user),
    decision_md_hash: null,
    region: cli.region,
    cold_start: true,
    max_steps: null,
    timeout_s: cli.timeoutS,
    retry_policy: `linear backoff 2s*n, max ${cli.maxRetries} retries on 429/5xx/transport`,
  }
  const metaProblems = metaViolations(meta as unknown as Record<string, unknown>)
  if (metaProblems.length > 0) throw new Error(`meta.json missing fields: ${metaProblems.join(', ')}`)
  writeFileSync(join(runDir, 'meta.json'), JSON.stringify(meta, null, 2))

  const cfg: LlmClientConfig = {
    baseUrl: cli.baseUrl,
    apiKey,
    model: cli.model,
    wire: 'chat',
    temperature: cli.temperature,
    topP: cli.topP,
    maxTokens: cli.maxTokens,
    timeoutMs: cli.timeoutS * 1000,
    maxRetries: cli.maxRetries,
    pricePerMTokensIn: cli.priceIn,
    pricePerMTokensOut: cli.priceOut,
  }

  tee.out(`probe ${cli.baseUrl} model=${cli.model} (cold TLS counted here)`)
  const probed = await probeWire(cfg)
  if (!probed.probe.succeeded) {
    writeFileSync(join(runDir, 'exit.json'), JSON.stringify({ exit_code: 2, completed: false, interrupted_at: { task: null, step: null }, reason: `probe failed: ${probed.probe.error}` }, null, 2))
    tee.err(`probe failed, aborting: ${probed.probe.error}`)
    return 2
  }
  cfg.wire = probed.wire
  tee.out(`wire = ${probed.wire}`)

  // ── Jev decider (jevloop arm only) ──────────────────────────────
  // The generator probe above covers the writer; the decider is a separate
  // endpoint, so for the jevloop arm we resolve its key, set meta.decider,
  // and probe it once — a 200-task run against a dead key is a waste.
  let jevCfg: JevConfig | null = null
  if (cli.arm === 'jevloop') {
    if (!cli.jevKeyFile) throw new Error('jevloop arm needs --jev-key-file (a file with jev_api_key=...)')
    const jevKey = readJevKey(cli.jevKeyFile)
    if (!jevKey) throw new Error(`no jev_api_key in ${cli.jevKeyFile} (refusing to run keyless)`)
    jevCfg = { baseUrl: cli.jevBaseUrl, apiKey: jevKey, model: cli.jevModel, timeoutMs: cli.timeoutS * 1000 }
    meta.decider = { id: cli.jevModel, version: cli.jevModel, provider: cli.jevBaseUrl }
    meta.max_steps = cli.maxSteps
    writeFileSync(join(runDir, 'meta.json'), JSON.stringify(meta, null, 2))
    tee.out(`probe ${cli.jevBaseUrl} model=${cli.jevModel} (decider)`)
    const jevProbe = await jevPing(jevCfg)
    if (!jevProbe.ok) {
      writeFileSync(join(runDir, 'exit.json'), JSON.stringify({ exit_code: 2, completed: false, interrupted_at: { task: null, step: null }, reason: `jev probe failed: ${jevProbe.reason}` }, null, 2))
      tee.err(`jev probe failed, aborting: ${jevProbe.reason}`)
      return 2
    }
    tee.out(`jev ok (${jevProbe.model ?? '?'}, ${jevProbe.ms}ms)`)
  }

  const usageRows: (LlmCallResult & { task_id: string; call_index: number })[] = []
  const records: ResultRecord[] = []
  const halted = { stopped: null as { task: string; step: number | null; reason: string } | null, lastTaskId: null as string | null }
  const stopped = (): { task: string; step: number | null; reason: string } | null => halted.stopped

  process.on('SIGINT', () => {
    halted.stopped = { task: halted.lastTaskId ?? 'unknown', step: null, reason: 'SIGINT' }
  })

  for (let i = 0; i < tasks.length; i++) {
    if (stopped()) break
    const task = tasks[i]!
    halted.lastTaskId = task.id
    const t0 = performance.now()
    let record: ResultRecord
    let statusLine: string

    if (cli.arm === 'direct') {
      const outcome = await runDirect(cfg, bench, task)
      const wallMs = performance.now() - t0
      const success = outcome.call.succeeded && bench.score(task, outcome.answer)
      usageRows.push({ ...outcome.call, task_id: task.id, call_index: 0 })
      record = {
        run_id: runId,
        dataset: cli.dataset,
        task_id: task.id,
        arm: cli.arm,
        seed: cli.seed,
        model: meta.generator,
        thinking_budget: 'off',
        temperature: cli.temperature,
        top_p: cli.topP,
        max_tokens: cli.maxTokens,
        prompt_hash: meta.prompt_hash,
        dataset_version: bench.version,
        commit: git.commit,
        region: cli.region,
        cold_start: false,
        success,
        steps: 1,
        first_divergence_step: success ? null : 1,
        escalated: false,
        gate_false_reject: null,
        gate_false_deny: null,
        failure_class: !outcome.call.succeeded ? 'generator_unavailable' : success ? null : 'wrong_answer',
        llm_calls: 1,
        decision_requests: 0,
        questions_per_request: 0,
        tool_calls: 0,
        tool_calls_necessary: 0,
        tool_calls_exploratory: 0,
        input_tokens_cached: outcome.call.input_tokens_cached,
        input_tokens_uncached: outcome.call.input_tokens_uncached,
        output_tokens_reasoning: outcome.call.output_tokens_reasoning,
        output_tokens_visible: outcome.call.output_tokens_visible,
        usd: outcome.call.usd,
        time: {
          wall_ms: wallMs,
          model_ms: { handshake: 0, ttft: null, after_ttft: outcome.call.succeeded ? outcome.call.latency_ms : 0 },
          decision_ms: { handshake: 0, compute: 0 },
          tool_ms: 0,
          framework_ms: Math.max(0, wallMs - (outcome.call.latency_ms + outcome.call.retry_ms)),
          retry_ms: outcome.call.retry_ms,
          round_trips: outcome.call.succeeded ? 1 : 0,
        },
        confidence: null,
        correct: success,
        answer: outcome.answer.slice(0, 2000),
        gold: task.gold[0] ?? '',
      }
      statusLine = `${success ? 'ok  ' : 'MISS'} ${Math.round(wallMs)}ms retry=${outcome.call.retries}${outcome.call.succeeded ? '' : ` err=${outcome.call.error?.slice(0, 120)}`}`
    } else {
      // jevloop arm — the kernel loop with Jev as decider, DeepSeek as generator.
      const outcome = await runJevLoop(cfg, jevCfg!, bench, task, cli.maxSteps)
      const wallMs = outcome.wallMs
      // Success halts: the loop finished on its own terms (answered directly,
      // ran tools then done, or isDone). Failure halts: max_steps, unclear
      // input/tool, unknown tool, denied authorisation, a failed step. The
      // `+revise` suffix means the delivery gate asked for another draft —
      // that is a normal outcome, not a failure.
      const baseHalt = outcome.halt.split('+')[0] ?? outcome.halt
      const badHalts = new Set(['max_steps', 'tool_unclear', 'unknown_tool', 'input_unclear', 'denied', 'step_failed'])
      const clean = !badHalts.has(baseHalt) && !baseHalt.startsWith('error')
      const success = clean && outcome.answer.length > 0 && bench.score(task, outcome.answer)
      record = {
        run_id: runId,
        dataset: cli.dataset,
        task_id: task.id,
        arm: cli.arm,
        seed: cli.seed,
        model: meta.generator,
        thinking_budget: 'off',
        temperature: cli.temperature,
        top_p: cli.topP,
        max_tokens: cli.maxTokens,
        prompt_hash: meta.prompt_hash,
        dataset_version: bench.version,
        commit: git.commit,
        region: cli.region,
        cold_start: false,
        success,
        steps: outcome.steps,
        first_divergence_step: success ? null : 1,
        escalated: outcome.escalated > 0,
        // gate fired a revision when the model was called twice; whether that
        // reject was *false* needs the first draft, which the meter does not
        // keep — recorded as unknown (null) until RQ2 reads trace.jsonl.
        gate_false_reject: outcome.modelCalls > 1 ? null : false,
        gate_false_deny: null,
        failure_class: !clean ? `halt:${outcome.halt}` : success ? null : 'wrong_answer',
        llm_calls: outcome.modelCalls,
        decision_requests: outcome.decisionBatches,
        questions_per_request: outcome.decisionBatches > 0 ? Math.round(outcome.decisionCount / outcome.decisionBatches) : 0,
        tool_calls: 0,
        tool_calls_necessary: 0,
        tool_calls_exploratory: 0,
        input_tokens_cached: 0,
        input_tokens_uncached: outcome.inputTokens,
        output_tokens_reasoning: 0,
        output_tokens_visible: outcome.outputTokens,
        usd: 0,
        time: {
          wall_ms: wallMs,
          model_ms: { handshake: 0, ttft: null, after_ttft: outcome.modelMs },
          decision_ms: { handshake: 0, compute: outcome.decisionMs },
          tool_ms: 0,
          framework_ms: Math.max(0, wallMs - outcome.modelMs - outcome.decisionMs),
          retry_ms: 0,
          round_trips: outcome.modelCalls + outcome.decisionBatches,
        },
        confidence: null,
        correct: success,
        answer: outcome.answer.slice(0, 2000),
        gold: task.gold[0] ?? '',
      }
      statusLine = `${success ? 'ok  ' : 'MISS'} ${Math.round(wallMs)}ms m=${outcome.modelCalls} d=${outcome.decisionBatches} halt=${outcome.halt}`
    }

    const problems = resultViolations(record as unknown as Record<string, unknown>)
    if (problems.length > 0) throw new Error(`result row missing fields: ${problems.join(', ')}`)
    records.push(record)
    appendFileSync(join(runDir, 'results.jsonl'), `${JSON.stringify(record)}\n`)
    writeFileSync(join(runDir, 'usage.json'), JSON.stringify(usageRows, null, 1))
    tee.out(`${String(i + 1).padStart(String(tasks.length).length, ' ')}/${tasks.length} ${task.id} ${statusLine}`)
  }

  if (cli.arm === 'jevloop') await disposeJevLoopArm()

  const finishedAt = new Date()
  const halt = stopped()
  const completed = records.length === tasks.length
  meta.finished_at = finishedAt.toISOString()
  meta.duration_ms = finishedAt.getTime() - startedAt.getTime()
  if (halt) meta.task_count = records.length
  writeFileSync(join(runDir, 'meta.json'), JSON.stringify(meta, null, 2))
  writeFileSync(
    join(runDir, 'exit.json'),
    JSON.stringify(
      {
        exit_code: 0,
        completed,
        interrupted_at: halt ? { task: halt.task, step: halt.step } : null,
        reason: halt ? `ran ${records.length}/${tasks.length}, stopped at task ${halt.task}, reason: ${halt.reason}` : null,
      },
      null,
      2,
    ),
  )
  if (records.length > 0) {
    const acc = records.filter((r) => r.success).length / records.length
    const toks = records.reduce((s, r) => s + r.input_tokens_uncached + r.output_tokens_visible, 0)
    tee.out(`summary ${cli.dataset}/${cli.arm}: ${records.length} tasks, acc ${(acc * 100).toFixed(1)}%, ${toks} tokens, ${Math.round((finishedAt.getTime() - startedAt.getTime()) / 1000)}s`)
  }
  return 0
}

main().then(
  (code) => process.exit(code),
  (err) => {
    process.stderr.write(`run.ts failed: ${(err as Error).stack}\n`)
    process.exit(1)
  },
)
