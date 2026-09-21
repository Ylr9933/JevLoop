/**
 * Dataset fetcher — download into `experiments/dataset/<name>/` and write
 * a SOURCES.json next to the data: URL, fetch time, byte size, SHA-256,
 * record count. Those fields are what MANIFEST.md §2 demands ("版本写进
 * 这里才能用") — a dataset without a provenance record is unusable for
 * numbers, and memory is not a source.
 *
 * All downloads are plain HTTP GET of raw files — no parquet, no datasets
 * library, no build step. If a URL is unreachable, the script fails loudly
 * for that dataset and continues with the rest.
 *
 * @module JevLoop/fetch
 */

import { createHash } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

interface Source {
  name: string
  files: Array<{ path: string; url: string }>
}

const SOURCES: Source[] = [
  {
    name: 'gsm8k',
    files: [
      {
        path: 'test.jsonl',
        url: 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl',
      },
    ],
  },
  {
    name: 'strategyqa',
    files: [
      { path: 'strategyqa_train.json', url: 'https://raw.githubusercontent.com/allenai/strategyqa/main/strategyqa_train.json' },
    ],
  },
  {
    name: 'hotpotqa',
    files: [
      { path: 'hotpot_dev_distractor_v1.json', url: 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json' },
    ],
  },
]

const DATASET_ROOT = join(new URL('..', import.meta.url).pathname, 'dataset')

/**
 * Download one file. Streams to memory (largest file is ~45MB) and returns
 * the buffer plus its SHA-256, so callers can record provenance without a
 * second read.
 */
async function download(url: string): Promise<{ bytes: Buffer; sha256: string }> {
  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`)
  const bytes = Buffer.from(await res.arrayBuffer())
  return { bytes, sha256: createHash('sha256').update(bytes).digest('hex') }
}

function countRecords(name: string, file: string, bytes: Buffer): number {
  if (file.endsWith('.jsonl')) {
    return bytes.toString('utf8').split('\n').filter((l) => l.trim() !== '').length
  }
  const parsed = JSON.parse(bytes.toString('utf8')) as unknown
  if (Array.isArray(parsed)) return parsed.length
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
    try {
      for (const f of source.files) {
        process.stdout.write(`fetching ${source.name}/${f.path} from ${f.url} ... `)
        const { bytes, sha256 } = await download(f.url)
        writeFileSync(join(dir, f.path), bytes)
        const records = countRecords(source.name, f.path, bytes)
        provenance[f.path] = { url: f.url, sha256, bytes: bytes.length, records }
        process.stdout.write(`ok (${bytes.length} bytes, ${records} records)\n`)
      }
      writeFileSync(join(dir, 'SOURCES.json'), JSON.stringify(provenance, null, 2))
    } catch (err) {
      failures++
      process.stdout.write(`FAILED: ${(err as Error).message}\n`)
    }
  }
  process.exit(failures > 0 && only.length === 0 ? 1 : 0)
}

main()
