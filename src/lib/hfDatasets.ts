/**
 * Hugging Face datasets-server client — pull real dataset rows into the tab.
 * Read-only: rows download in; nothing is sent out. Same endpoints power the
 * starter-set fetch script, so this is the single implementation.
 */

import type { DatasetRow } from './dataset';

const API = 'https://datasets-server.huggingface.co';
const PAGE = 100; // datasets-server max rows per request

export class HfError extends Error {}

/** 'owner/name' from a bare repo id or a huggingface.co dataset URL. */
export function parseDatasetRef(input: string): string | null {
  let s = input.trim();
  if (!s) return null;
  if (/^https?:\/\//i.test(s) && !/^https?:\/\/(www\.)?huggingface\.co/i.test(s)) return null;
  s = s.replace(/^(https?:\/\/)?(www\.)?huggingface\.co\/?/i, '');
  s = s.replace(/^\/+|\/+$/g, '');
  if (s.startsWith('datasets/')) s = s.slice('datasets/'.length);
  const id = s.split('/').filter(Boolean).slice(0, 2).join('/');
  return /^[\w.-]+\/[\w.-]+$/.test(id) ? id : null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function getJson(url: string): Promise<Record<string, unknown>> {
  let r: Response | null = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      r = await fetch(url);
    } catch {
      throw new HfError('Network error reaching Hugging Face.');
    }
    if (r.ok) return r.json() as Promise<Record<string, unknown>>;
    if (r.status === 401 || r.status === 403)
      throw new HfError('Dataset is gated or private — it needs to be public.');
    if (r.status === 404)
      throw new HfError('Not found — check the dataset id (owner/name).');
    if (r.status !== 429 && r.status < 500) break; // real client error — don't retry
    await sleep(1500 * 2 ** attempt + Math.random() * 400); // backoff on 429/5xx
  }
  const body = (r ? ((await r.json().catch(() => null)) as { cause_message?: string; error?: string } | null) : null);
  throw new HfError(
    body?.cause_message || body?.error || `HF returned ${r?.status ?? 'no response'} — try again in a moment.`,
  );
}

export interface HfSplit {
  config: string;
  split: string;
}

/** List available config:split pairs for a dataset. */
export async function fetchSplits(id: string): Promise<HfSplit[]> {
  const j = await getJson(`${API}/splits?dataset=${encodeURIComponent(id)}`);
  const splits = (j.splits as { config: string; split: string }[] | undefined) ?? [];
  if (splits.length === 0)
    throw new HfError('Dataset has no viewable splits (script-based or empty).');
  return splits;
}

export interface HfRows {
  columns: string[];
  rows: DatasetRow[];
  /** total rows in the split server-side */
  total: number;
}

interface Feature {
  feature_idx?: number;
  name: string;
  type?: { _type?: string; names?: string[] };
}

/** Download up to `maxRows` rows of a split; ClassLabel ints → label names. */
export async function fetchRows(
  id: string,
  config: string,
  split: string,
  maxRows = 512,
  onProgress?: (got: number) => void,
  delayMs = 0,
): Promise<HfRows> {
  const rows: DatasetRow[] = [];
  let columns: string[] = [];
  const labelMaps: Record<string, string[]> = {};
  let total = 0;

  for (let offset = 0; rows.length < maxRows; offset += PAGE) {
    const want = Math.min(PAGE, maxRows - rows.length);
    const j = await getJson(
      `${API}/rows?dataset=${encodeURIComponent(id)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=${offset}&length=${want}`,
    );
    const page = (j.rows as { row: Record<string, unknown> }[] | undefined) ?? [];
    if (offset === 0) {
      const feats = ((j.features as Feature[] | undefined) ?? []).slice();
      feats.sort((a, b) => (a.feature_idx ?? 0) - (b.feature_idx ?? 0));
      columns = feats.map((f) => f.name);
      for (const f of feats)
        if (f.type?._type === 'ClassLabel' && Array.isArray(f.type.names))
          labelMaps[f.name] = f.type.names;
    }
    total = (j.num_rows_total as number | undefined) ?? total;
    for (const { row } of page) {
      const d: DatasetRow = {};
      for (const c of columns) {
        let v = row[c];
        if (typeof v === 'number' && labelMaps[c]) v = labelMaps[c][v] ?? String(v);
        d[c] = v == null ? '' : typeof v === 'string' ? v : JSON.stringify(v);
      }
      rows.push(d);
    }
    onProgress?.(rows.length);
    if (page.length < want || rows.length >= total) break;
    if (delayMs > 0) await sleep(delayMs);
  }
  if (rows.length === 0) throw new HfError('Split returned no rows.');
  return { columns, rows, total };
}
