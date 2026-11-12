/**
 * Collects GLiNER2.5 Decide's per-label probabilities for JEV validation
 * questions (in-distribution) and JevBench items (out-of-distribution), so
 * the two can be compared directly. Output: E:/tmp/decide-probs.json
 *
 *   node scripts/jev-decide-probs.mts
 * Env: PER_FILE (validation questions per dataset, default 40), OUT.
 */
import { readdirSync } from 'node:fs';
import { loadDecide, DECIDE_REPO } from '../src/lib/gliner2.ts';
import type { Answer, Question } from '../src/lib/jev.ts';
import { benchItems, decideQuestion, answerLabel, questionsFrom, type Ex } from './jev-bench-compare.mts';

const PER_FILE = Number(process.env.PER_FILE ?? 40);
const OUT = process.env.OUT ?? 'E:/tmp/decide-probs.json';

interface Row {
  set: 'val' | 'bench';
  grp: string;
  labels: string[];
  gold: string;
  pred: string | null;
  probs: Record<string, number>;
}

function probsOf(a: Answer | undefined, labels: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  if (!a) return out;
  if (a.type === 'noul') {
    out.no = 1 - a.probability;
    out.yes = a.probability;
    return out;
  }
  for (const l of labels) out[l] = a.probabilities[l] ?? 0;
  return out;
}

function toDecide(ex: Ex): Question {
  const q = ex.q;
  if (q.kind === 'noul') return { type: 'noul', instructions: q.instructions };
  if (q.kind === 'score') return { type: 'score', instructions: q.instructions, options: q.labels };
  return { type: 'choice', instructions: q.instructions, options: q.labels, descriptions: q.desc };
}

const decide = await loadDecide(DECIDE_REPO, { device: 'cpu', dtype: 'q4' });
const rows: Row[] = [];
const t0 = Date.now();

const files = readdirSync('public/jev-datasets').filter((f) => f.endsWith('.jsonl') && (f.startsWith('jev-') || f.startsWith('general-')));
const valEx: Ex[] = [];
for (const f of files) {
  valEx.push(...questionsFrom(`public/jev-datasets/${f}`, 'val', f.startsWith('jev-') ? 'jev' : 'general').slice(0, PER_FILE));
}
console.log(`val questions ${valEx.length}`);
for (const [i, ex] of valEx.entries()) {
  const [a] = await decide.decide(ex.state, [toDecide(ex)]).catch(() => [undefined]);
  const probs = probsOf(a, ex.q.labels);
  const pred = a ? answerLabel(a) : null;
  rows.push({ set: 'val', grp: ex.grp, labels: ex.q.labels, gold: ex.gold, pred, probs });
  if (i % 50 === 0) console.log(`val ${i}/${valEx.length} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}

const items = benchItems();
for (const [i, it] of items.entries()) {
  const [a] = await decide.decide(it.state, [decideQuestion(it)]).catch(() => [undefined]);
  rows.push({ set: 'bench', grp: it.family, labels: it.labels, gold: it.expected, pred: answerLabel(a), probs: probsOf(a, it.labels) });
  if (i % 50 === 0) console.log(`bench ${i}/${items.length} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}

const acc = (set: string) => {
  const r = rows.filter((x) => x.set === set);
  return `${set}: ${((r.filter((x) => x.pred === x.gold).length / r.length) * 100).toFixed(1)}% (n=${r.length})`;
};
console.log(acc('val'), '|', acc('bench'));
await writeOut(OUT, rows);

async function writeOut(path: string, data: unknown) {
  const { writeFileSync } = await import('node:fs');
  writeFileSync(path, JSON.stringify(data));
  console.log('wrote', path);
}
