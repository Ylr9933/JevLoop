/**
 * Read the Jev API key from the env file (var `jev_api_key`), stripping
 * surrounding quotes. Sibling to `readTokenFile` but for the Jev-decider
 * key, which lives in a different file than the generator token. The key
 * is never logged; callers mask it via `maskToken` before writing anything.
 *
 * @module JevLoop/jev-key
 */

import { readFileSync } from 'node:fs'

/** Return '' when the file or the var is missing — the runner refuses to run keyless. */
export function readJevKey(path: string): string {
  let raw: string
  try {
    raw = readFileSync(path, 'utf8')
  } catch {
    return ''
  }
  const m = raw.match(/^\s*jev_api_key\s*=\s*"?"?([^"\n]+?)"?"?\s*$/m)
  return m ? m[1]!.trim() : ''
}
