import type { Answer } from './jev';

/**
 * Held-out evaluation for a trained artifact vs an inference baseline on the
 * SAME rows. Closures keep this UI-agnostic — the caller wires inferFt /
 * inferHead / DecideSession / OpenJev / DecoderSession behind two functions.
 */

export interface PerClass {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EngineMetrics {
  acc: number;
  macroF1: number;
  /** ms per row, mean */
  ms: number;
  /** rows where the engine produced a usable prediction */
  answered: number;
}

export interface EvalReport {
  labels: string[];
  rows: number;
  trained: EngineMetrics & { confusion: number[][]; perClass: PerClass[] };
  baseline: (EngineMetrics & { name: string }) | null;
  /** fraction of rows where trained argmax == baseline argmax */
  agreement: number | null;
}

export function metricsFromPreds(
  preds: (number | null)[],
  gold: number[],
  labels: string[],
  ms: number[],
): EngineMetrics & { confusion: number[][]; perClass: PerClass[] } {
  const c = labels.length;
  const confusion = Array.from({ length: c }, () => new Array<number>(c).fill(0));
  let answered = 0,
    hit = 0;
  for (let i = 0; i < gold.length; i++) {
    const p = preds[i];
    if (p == null) continue;
    answered++;
    confusion[gold[i]][p]++;
    if (p === gold[i]) hit++;
  }
  const perClass: PerClass[] = labels.map((label, j) => {
    let tp = 0,
      fp = 0,
      fn = 0,
      support = 0;
    for (let i = 0; i < gold.length; i++) {
      const p = preds[i];
      if (gold[i] === j) {
        support++;
        if (p === j) tp++;
        else if (p != null) fn++;
      } else if (p === j) fp++;
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = support > 0 ? tp / support : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { label, precision, recall, f1, support };
  });
  const macroF1 = perClass.reduce((s, p) => s + p.f1, 0) / Math.max(1, c);
  return {
    acc: answered > 0 ? hit / answered : 0,
    macroF1,
    ms: ms.length > 0 ? ms.reduce((a, b) => a + b, 0) / ms.length : 0,
    answered,
    confusion,
    perClass,
  };
}

function argmaxRecord(probs: Record<string, number>, labels: string[]): number {
  let best = 0;
  for (let i = 1; i < labels.length; i++) {
    if ((probs[labels[i]] ?? 0) > (probs[labels[best]] ?? 0)) best = i;
  }
  return best;
}

/** Pull a label-index prediction out of a JEV-shaped choice answer. */
export function answerToIndex(a: Answer, labels: string[]): number | null {
  if (a.type !== 'choice') return null;
  const idx = labels.indexOf(a.choice);
  if (idx >= 0) return idx;
  // generated answers can drift from the exact option string — fuzzy rescue
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const target = norm(a.choice);
  const fuzzy = labels.findIndex((l) => norm(l) === target);
  return fuzzy >= 0 ? fuzzy : null;
}

export async function evaluateRows(args: {
  texts: string[];
  gold: number[];
  labels: string[];
  /** trained classifier → per-label probabilities */
  trained: (text: string) => Promise<Record<string, number>>;
  /** baseline → JEV-shaped choice answer; name shown in the report */
  baseline?: { name: string; run: (text: string) => Promise<Answer> };
  onRow?: (done: number, total: number) => void;
}): Promise<EvalReport> {
  const { texts, gold, labels, trained, baseline, onRow } = args;
  const total = texts.length * (baseline ? 2 : 1);
  let done = 0;

  const trainedPreds: (number | null)[] = [];
  const trainedMs: number[] = [];
  for (const text of texts) {
    const t0 = performance.now();
    try {
      trainedPreds.push(argmaxRecord(await trained(text), labels));
    } catch {
      trainedPreds.push(null);
    }
    trainedMs.push(performance.now() - t0);
    onRow?.(++done, total);
  }

  let baselineMetrics: EvalReport['baseline'] = null;
  let agreement: number | null = null;
  if (baseline) {
    const basePreds: (number | null)[] = [];
    const baseMs: number[] = [];
    for (const text of texts) {
      const t0 = performance.now();
      try {
        basePreds.push(answerToIndex(await baseline.run(text), labels));
      } catch {
        basePreds.push(null);
      }
      baseMs.push(performance.now() - t0);
      onRow?.(++done, total);
    }
    const m = metricsFromPreds(basePreds, gold, labels, baseMs);
    baselineMetrics = {
      name: baseline.name,
      acc: m.acc,
      macroF1: m.macroF1,
      ms: m.ms,
      answered: m.answered,
    };
    let agree = 0,
      both = 0;
    for (let i = 0; i < texts.length; i++) {
      if (trainedPreds[i] != null && basePreds[i] != null) {
        both++;
        if (trainedPreds[i] === basePreds[i]) agree++;
      }
    }
    agreement = both > 0 ? agree / both : null;
  }

  return {
    labels,
    rows: texts.length,
    trained: metricsFromPreds(trainedPreds, gold, labels, trainedMs),
    baseline: baselineMetrics,
    agreement,
  };
}

/** Held-out rows of one trained artifact, saved at train time for later comparison. */
export interface EvalSet {
  kind: 'ft' | 'head';
  createdAt: string;
  tasks: { name: string; labels: string[] }[];
  /** `gold[k]` is the label index for `tasks[k]`, or -1 when that task did not hold this row out */
  rows: { text: string; gold: number[] }[];
}

export interface TaskComparison {
  name: string;
  labels: string[];
  n: number;
  trained: EngineMetrics & { confusion: number[][]; perClass: PerClass[] };
  base: EngineMetrics & { confusion: number[][]; perClass: PerClass[] };
}

export interface ComparisonReport {
  base: string;
  rows: number;
  tasks: TaskComparison[];
  /** fraction of rows where trained and base agree on the primary task */
  agreement: number | null;
  /** rows where either engine missed the primary task, for inspection */
  misses: {
    text: string;
    gold: (string | null)[];
    trained: (string | null)[];
    base: (string | null)[];
  }[];
}

/**
 * Scores the trained artifact and a base model on the SAME held-out rows.
 * `trained` and `base` return one label index (or null) per task; the base
 * is asked every task in a single JEV-shaped call by the caller.
 */
export async function compareSet(args: {
  set: EvalSet;
  limit: number;
  trained: (text: string) => Promise<(number | null)[]>;
  base: { name: string; run: (text: string) => Promise<(number | null)[]> };
  onRow?: (done: number, total: number) => void;
}): Promise<ComparisonReport> {
  const { set, limit, trained, base, onRow } = args;
  const rows = set.rows.slice(0, limit);
  const total = rows.length * 2;
  let done = 0;

  const tPreds: (number | null)[][] = [];
  const tMs: number[] = [];
  for (const row of rows) {
    const t0 = performance.now();
    try {
      tPreds.push(await trained(row.text));
    } catch {
      tPreds.push(set.tasks.map(() => null));
    }
    tMs.push(performance.now() - t0);
    onRow?.(++done, total);
  }

  const bPreds: (number | null)[][] = [];
  const bMs: number[] = [];
  for (const row of rows) {
    const t0 = performance.now();
    try {
      bPreds.push(await base.run(row.text));
    } catch {
      bPreds.push(set.tasks.map(() => null));
    }
    bMs.push(performance.now() - t0);
    onRow?.(++done, total);
  }

  const tasks: TaskComparison[] = set.tasks.map((task, k) => {
    const idx = rows.flatMap((r, i) => (r.gold[k] >= 0 ? [i] : []));
    const gold = idx.map((i) => rows[i].gold[k]);
    return {
      name: task.name,
      labels: task.labels,
      n: idx.length,
      trained: metricsFromPreds(idx.map((i) => tPreds[i][k]), gold, task.labels, idx.map((i) => tMs[i])),
      base: metricsFromPreds(idx.map((i) => bPreds[i][k]), gold, task.labels, idx.map((i) => bMs[i])),
    };
  });

  let agree = 0,
    both = 0;
  rows.forEach((_, i) => {
    const t = tPreds[i][0],
      b = bPreds[i][0];
    if (t != null && b != null) {
      both++;
      if (t === b) agree++;
    }
  });

  const labelOf = (k: number, idx: number | null) => (idx != null && idx >= 0 ? set.tasks[k].labels[idx] ?? null : null);
  const misses = rows
    .flatMap((row, i) => (row.gold[0] >= 0 && (tPreds[i][0] !== row.gold[0] || bPreds[i][0] !== row.gold[0]) ? [i] : []))
    .slice(0, 8)
    .map((i) => ({
      text: rows[i].text,
      gold: set.tasks.map((_, k) => labelOf(k, rows[i].gold[k])),
      trained: set.tasks.map((_, k) => labelOf(k, tPreds[i][k])),
      base: set.tasks.map((_, k) => labelOf(k, bPreds[i][k])),
    }));

  return {
    base: base.name,
    rows: rows.length,
    tasks,
    agreement: both > 0 ? agree / both : null,
    misses,
  };
}
