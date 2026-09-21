/**
 * Dataset fetcher — download into `experiments/dataset/<name>/` and write
 * a SOURCES.json next to the data: the URL that actually served the bytes,
 * fetch time, byte size, SHA-256, record count. Those fields are what
 * MANIFEST.md §2 demands ("版本写进这里才能用") — a dataset without a
 * provenance record is unusable for numbers, and memory is not a source.
 *
 * Two transport modes, per file:
 *
 *   · `urls`    — plain HTTP GET of a raw file, first URL that answers wins
 *                 (raw GitHub -> jsdelivr CDN -> other hosts). No parquet,
 *                 no datasets library.
 *   · `apiRows` — the HF datasets-server rows endpoint (`/rows?dataset=
 *                 ...&config=...&split=...&offset=...&length=100`), which
 *                 returns plain JSON. Used when no raw-file host is
 *                 reachable; `length` is capped at 100 per page by the
 *                 server, so the range actually fetched is recorded —
 *                 a partial fetch must say it is partial.
 *
 * @module JevLoop/fetch
 */

import { createHash } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

interface FileSource {
  path: string
  urls: string[]
}

interface ApiRowsSource {
  path: string
  dataset: string
  config: string
  split: string
  /** How many records to page through (server caps a page at 100). */
  count: number
}

interface Source {
  name: string
  files: FileSource[]
  apiRows?: ApiRowsSource[]
}

const SOURCES: Source[] = [
  {
    name: 'gsm8k',
    files: [
      {
        path: 'test.jsonl',
        urls: [
          'https://cdn.jsdelivr.net/gh/openai/grade-school-math@master/grade_school_math/data/test.jsonl',
          'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl',
          'https://fastly.jsdelivr.net/gh/openai/grade-school-math@master/grade_school_math/data/test.jsonl',
        ],
      },
    ],
  },
  {
    name: 'strategyqa',
    files: [],
    apiRows: [
      {
        path: 'strategyqa_rows.json',
        dataset: 'ChilleD/StrategyQA',
        config: 'train',
        split: 'train',
        count: 300,
      },
    ],
  },
  {
    name: 'hotpotqa',
    files: [
      {
        path: 'hotpot_dev_distractor_v1.json',
        urls: [
          'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
          'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
        ],
      },
    ],
    apiRows: [
      {
        path: 'hotpot_rows.json',
        dataset: 'hotpotqa/hotpot_qa',
        config: 'distractor',
        split: 'validation',
        count: 300,
      },
    ],
  },
]

const DATASET_ROOT = join(new URL('..', import.meta.url).pathname, 'dataset')

function sha256(bytes: Buffer | string): string {
  return createHash('sha256').update(bytes).digest('hex')
}

/**
 * Plain-file download, trying every URL in order. Returns the buffer,
 * its SHA-256, the URL that served it, and every URL that failed.
 */
async function download(urls: string[]): Promise<{ bytes: Buffer; sha256: string; from: string; failures: string[] }> {
  const failures: string[] = []
  for (const url of urls) {
    try {
      const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(300_000) })
      if (!res.ok) {
        failures.push(`${url} -> HTTP ${res.status}`)
        continue
      }
      const bytes = Buffer.from(await res.arrayBuffer())
      if (bytes.length === 0) {
        failures.push(`${url} -> empty body`)
        continue
      }
      return { bytes, sha256: sha256(bytes), from: url, failures }
    } catch (err) {
      failures.push(`${url} -> ${(err as Error).message}`)
    }
  }
  throw new Error(`all sources failed: ${failures.join('; ')}`)
}

/**
 * Page through the datasets-server rows endpoint and keep the `row`
 * objects. The server answers at most 100 rows per page; each page is a
 * separate request, so a run is reproducible from (dataset, config,
 * split, offset range) recorded in SOURCES.json.
 */
async function downloadApiRows(spec: ApiRowsSource): Promise<{ rows: unknown[]; pages: string[]; total?: number }> {
  const rows: unknown[] = []
  const pages: string[] = []
  let total: number | undefined
  for (let offset = 0; offset < spec.count; offset += 100) {
    const length = Math.min(100, spec.count - offset)
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(spec.dataset)}&config=${encodeURIComponent(spec.config)}&split=${encodeURIComponent(spec.split)}&offset=${offset}&length=${length}`
    const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(120_000) })
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`)
    const body = (await res.json()) as { rows?: Array<{ row: unknown }>; num_rows_total?: number }
    if (!Array.isArray(body.rows)) throw new Error(`${url} -> no rows array`)
    if (body.num_rows_total !== undefined) total = body.num_rows_total
    if (body.rows.length === 0) break
    for (const r of body.rows) rows.push(r.row)
    pages.push(url)
    if (body.rows.length < length) break
  }
  return { rows, pages, total }
}

function countRecords(name: string, file: string, bytes: Buffer): number {
  if (file.endsWith('.jsonl')) {
    return bytes.toString('utf8').split('\n').filter((l) => l.trim() !== '').length
  }
  const parsed = JSON.parse(bytes.toString('utf8')) as unknown
  if (Array.isArray(parsed)) return parsed.length
  if (parsed && typeof parsed === 'object' && Array.isArray((parsed as { rows: unknown[] }).rows)) {
    return (parsed as { rows: unknown[] }).rows.length
  }
  throw new Error(`${name}/${file}: unexpected content shape, refusing to record`)
}

async function main(): Promise<void> {
  const only = process.argv.slice(2).filter((a) => !a.startsWith('--'))
  let failures = 0
  for (const source of SOURCES) {
    if (only.length > 0 && !only.includes(source.name)) continue
    const dir = join(DATASET_ROOT, source.name)
    mkdirSync(dir, { recursive: true })
    const provenance: Record<string, unknown> = { dataset: source.name, fetched_at: new Date().toISOString() }
    let wroteAny = false
    try {
      let rawSucceeded = false
      for (const f of source.files) {
        process.stdout.write(`fetching ${source.name}/${f.path} ... `)
        try {
          const { bytes, sha256: hash, from, failures: tried } = await download(f.urls)
          writeFileSync(join(dir, f.path), bytes)
          provenance[f.path] = { url: from, sha256: hash, bytes: bytes.length, records: countRecords(source.name, f.path, bytes) }
          if (tried.length > 0) provenance['failed_sources'] = tried
          process.stdout.write(`ok from ${from} (${bytes.length} bytes)\n`)
          wroteAny = true
          rawSucceeded = true
        } catch (err) {
          process.stdout.write(`FAILED: ${(err as Error).message}\n`)
        }
      }
      for (const spec of source.apiRows ?? []) {
        if (rawSucceeded) continue
        process.stdout.write(`fetching ${source.name}/${spec.path} via datasets-server ... `)
        try {
          const { rows, pages, total } = await downloadApiRows(spec)
          if (rows.length === 0) throw new Error('no rows returned')
          const payload = JSON.stringify({ source: 'datasets-server.huggingface.co rows api', dataset: spec.dataset, config: spec.config, split: spec.split, rows }, null, 1)
          writeFileSync(join(dir, spec.path), payload)
          provenance[spec.path] = {
            url: pages[0],
            url_last_page: pages[pages.length - 1],
            records_fetched: rows.length,
            split_total: total ?? null,
            partial: rows.length < (total ?? rows.length),
            sha256: sha256(payload),
          }
          process.stdout.write(`ok (${rows.length} rows, split total ${total ?? 'unknown'})\n`)
          wroteAny = true
        } catch (err) {
          process.stdout.write(`FAILED: ${(err as Error).message}\n`)
        }
      }
      if (wroteAny) writeFileSync(join(dir, 'SOURCES.json'), JSON.stringify(provenance, null, 2))
      else failures++
    } catch (err) {
      failures++
      process.stdout.write(`FAILED: ${(err as Error).message}\n`)
    }
  }
  process.exit(failures > 0 ? 1 : 0)
}

main()
