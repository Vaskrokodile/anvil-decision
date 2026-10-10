/**
 * JevBench public-item runner logic — pure functions, no React, testable in Node.
 * Items come from JevBench (MIT) via scripts/fetch-jevbench.mts. Only PUBLIC items
 * are runnable here; the official score also uses sealed items, which are never
 * published, so a run here is an unofficial estimate.
 */
import { jevDecide, JevError, type JevAnswer, type JevQuestion } from '../lib/jevApi.ts';

export type Tier = 'easy' | 'original' | 'hard';
export const TIERS: Tier[] = ['easy', 'original', 'hard'];

export interface BenchItem {
  id: string;
  family: string;
  expected: string | number;
  labels: string[];
  state: string;
  question: {
    type: 'choice' | 'noul' | 'score';
    instructions: string;
    criteria: Record<string, string | null> | Record<string, string> | string[];
  };
}

export interface ItemResult {
  id: string;
  tier: Tier;
  correct: boolean;
  /** probability the model put on the gold answer */
  pGold: number;
  /** confidence in its own top answer, 0..1 */
  conf: number;
  predicted: string | null;
  expected: string;
  ms: number;
  error?: string;
}

/** Map a JevBench item onto the Jev API question shape. */
export function toJevQuestion(item: BenchItem): JevQuestion {
  const q = item.question;
  if (q.type === 'score') return { type: 'score', instructions: q.instructions, criteria: q.criteria as string[] };
  if (q.type === 'noul') return { type: 'noul', instructions: q.instructions, criteria: q.criteria as { true?: string; false?: string } };
  return { type: 'choice', instructions: q.instructions, criteria: q.criteria as Record<string, string | null> };
}

/** Score one Jev answer against the gold label. */
export function judge(
  item: BenchItem,
  a: JevAnswer | undefined,
): { predicted: string | null; correct: boolean; pGold: number; conf: number } {
  const gold = String(item.expected);
  if (!a) return { predicted: null, correct: false, pGold: 0, conf: 0 };
  if (item.question.type === 'noul') {
    const p = a.noul ?? 0.5;
    const predicted = p >= 0.5 ? 'yes' : 'no';
    return {
      predicted,
      correct: predicted === gold,
      pGold: gold === 'yes' ? p : 1 - p,
      conf: Math.abs(p - 0.5) * 2,
    };
  }
  const probs = a.probabilities ?? {};
  const entries = Object.entries(probs);
  if (entries.length === 0) return { predicted: null, correct: false, pGold: 0, conf: 0 };
  // score items: probability keys may be level indices or legend text; map to labels
  const [topKey, topP] = entries.reduce((best, cur) => (cur[1] > best[1] ? cur : best));
  let predicted = topKey;
  if (item.question.type === 'score' && !item.labels.includes(topKey)) {
    const idx = (item.question.criteria as string[]).indexOf(topKey);
    if (idx >= 0) predicted = String(idx);
  }
  const pGold = probs[gold] ?? probs[String(Number(gold))] ?? 0;
  return {
    predicted,
    correct: predicted === gold,
    pGold,
    conf: a.confidence ?? topP,
  };
}

/** Chance level for the item: 0.5 for yes/no, 1/K otherwise. */
export function chanceOf(item: BenchItem): number {
  return item.question.type === 'noul' ? 0.5 : 1 / item.labels.length;
}

export interface RunSummary {
  n: number;
  errors: number;
  accuracy: number;
  /** JevBench-style chance-corrected intelligence, 0..100, macro-averaged over tiers */
  intelligence: number;
  perTier: Record<Tier, { n: number; accuracy: number; intelligence: number }>;
  /** expected calibration error over 10 confidence bins, 0..1 (lower is better) */
  ece: number;
  meanPGold: number;
  p50Ms: number;
}

export function summarize(results: ItemResult[], items: Map<string, BenchItem>): RunSummary {
  const ok = results.filter((r) => !r.error);
  const correct = ok.filter((r) => r.correct).length;

  const perTier = {} as RunSummary['perTier'];
  for (const t of TIERS) {
    const rs = ok.filter((r) => r.tier === t);
    if (rs.length === 0) {
      perTier[t] = { n: 0, accuracy: 0, intelligence: 0 };
      continue;
    }
    const acc = rs.filter((r) => r.correct).length / rs.length;
    const chance = chanceOf(items.get(rs[0].id) as BenchItem);
    perTier[t] = {
      n: rs.length,
      accuracy: acc,
      intelligence: Math.max(0, ((acc - chance) / (1 - chance)) * 100),
    };
  }
  const tiersUsed = TIERS.filter((t) => perTier[t].n > 0);
  const intelligence = tiersUsed.length
    ? tiersUsed.reduce((s, t) => s + perTier[t].intelligence, 0) / tiersUsed.length
    : 0;

  // ECE over 10 equal-width confidence bins
  const bins = Array.from({ length: 10 }, () => ({ n: 0, conf: 0, hit: 0 }));
  for (const r of ok) {
    const b = Math.min(9, Math.floor(r.conf * 10));
    bins[b].n++;
    bins[b].conf += r.conf;
    bins[b].hit += r.correct ? 1 : 0;
  }
  let ece = 0;
  for (const b of bins) if (b.n) ece += (b.n / Math.max(1, ok.length)) * Math.abs(b.hit / b.n - b.conf / b.n);

  const ms = ok.map((r) => r.ms).sort((a, b) => a - b);
  return {
    n: results.length,
    errors: results.length - ok.length,
    accuracy: ok.length ? correct / ok.length : 0,
    intelligence,
    perTier,
    ece,
    meanPGold: ok.length ? ok.reduce((s, r) => s + r.pGold, 0) / ok.length : 0,
    p50Ms: ms.length ? ms[ms.length >> 1] : 0,
  };
}

/** Deterministic sample of n items (seeded), so reruns compare like with like. */
export function sampleItems<T>(items: T[], n: number, seed = 1337): T[] {
  if (n >= items.length) return items.slice();
  let s = seed >>> 0;
  const rnd = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const a = items.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

export async function loadTier(tier: Tier): Promise<BenchItem[]> {
  const r = await fetch(`${import.meta.env?.BASE_URL ?? '/'}jevbench/items/${tier}.jsonl`);
  if (!r.ok) throw new Error(`could not load ${tier} items (${r.status})`);
  return (await r.text())
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l) as BenchItem);
}

/**
 * Run items through the Jev API with bounded concurrency. An invalid key (401)
 * aborts the whole run; other per-item failures are recorded and counted.
 */
export async function runItems(
  apiKey: string,
  items: BenchItem[],
  tier: Tier,
  onItem: (r: ItemResult, done: number) => void,
  signal: AbortSignal,
  concurrency = 4,
): Promise<ItemResult[]> {
  const out: ItemResult[] = [];
  let next = 0;
  let fatal: unknown = null;

  const worker = async () => {
    while (next < items.length && !fatal && !signal.aborted) {
      const item = items[next++];
      try {
        const { answers, ms } = await jevDecide(apiKey, item.state, { q: toJevQuestion(item) }, { signal });
        const j = judge(item, answers.q);
        const r: ItemResult = { id: item.id, tier, correct: j.correct, pGold: j.pGold, conf: j.conf, predicted: j.predicted, expected: String(item.expected), ms };
        out.push(r);
        onItem(r, out.length);
      } catch (e) {
        if (e instanceof JevError && e.status === 401) {
          fatal = e;
          return;
        }
        if (signal.aborted) return;
        const r: ItemResult = { id: item.id, tier, correct: false, pGold: 0, conf: 0, predicted: null, expected: String(item.expected), ms: 0, error: (e as Error).message };
        out.push(r);
        onItem(r, out.length);
      }
    }
  };

  await Promise.all(Array.from({ length: concurrency }, worker));
  if (fatal) throw fatal;
  return out;
}
