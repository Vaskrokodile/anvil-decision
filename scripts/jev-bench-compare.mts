/**
 * Before / after benchmark: does training on the JEV-format datasets improve
 * the anvil classifier on JevBench (the tasks JEV models are designed for)?
 *
 * Model: frozen MiniLM-L6 embeddings of a (question, candidate label, state)
 * pair → linear head → yes/no. Scoring every candidate label lets one head
 * answer questions with arbitrary label sets, which the fixed-label trainer
 * cannot. Conditions:
 *   untrained  — random head (before training)
 *   general    — trained on general-* datasets only
 *   jev        — trained on jev-* datasets only
 *   all        — both
 *   decide     — GLiNER2.5 Decide, zero-shot, same JEV questions (reference)
 *
 * Train rows come from each dataset's `train` split; in-distribution accuracy
 * is reported on its `val` split; JevBench public items are never trained on.
 *
 * Env: CAP (records per dataset), EPOCHS, LR, DECIDE=0 to skip the Decide
 * baseline, OUT (json output), CACHE (embedding cache dir, reused if present).
 *
 *   node scripts/jev-bench-compare.mts
 */
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { embedTexts, EMBED_DIM } from '../src/lib/embed.ts';
import { CpuHead } from '../src/lib/head.ts';
import { loadDecide, DECIDE_REPO } from '../src/lib/gliner2.ts';
import type { Answer, Question } from '../src/lib/jev.ts';

const CAP = Number(process.env.CAP ?? 300);
const EPOCHS = Number(process.env.EPOCHS ?? 5);
const LR = Number(process.env.LR ?? 1e-2);
const BATCH = 32;
const STATE_CHARS = 1000;
const OUT = process.env.OUT ?? 'E:/tmp/bench-compare.json';
const CACHE = process.env.CACHE ?? 'E:/tmp/bench-emb';

type Kind = 'choice' | 'score' | 'noul';
interface Q2 {
  kind: Kind;
  instructions: string;
  labels: string[];
  desc: Record<string, string>;
}
export interface Ex {
  q: Q2;
  state: string;
  gold: string;
  grp: string;
}
export interface BenchItem extends Q2 {
  id: string;
  family: string;
  state: string;
  expected: string;
}

const readJsonl = (path: string): Record<string, unknown>[] =>
  readFileSync(path, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Record<string, unknown>);

/** `state` is a string or a structured object in the JEV format. */
export function stateText(s: unknown): string {
  return typeof s === 'string' ? s : JSON.stringify(s);
}

function levelName(criterion: string): string {
  return criterion.slice(0, Math.max(0, criterion.indexOf(':'))).trim() || criterion.trim();
}

function descriptions(kind: Kind, criteria: unknown): Record<string, string> {
  if (kind === 'choice') return (criteria as Record<string, string>) ?? {};
  if (kind === 'score') {
    const out: Record<string, string> = {};
    for (const s of (criteria as string[]) ?? []) out[levelName(s)] = s.slice(s.indexOf(':') + 1).trim();
    return out;
  }
  const c = (criteria as { true?: string; false?: string }) ?? {};
  return { yes: c.true ?? '', no: c.false ?? '' };
}

export function pairText(q: Q2, label: string, state: string): string {
  const d = q.desc[label];
  return `${q.instructions} Option: ${label}${d ? ` (${d})` : ''}. Text: ${state.slice(0, STATE_CHARS)}`;
}

/** JEV dataset records → one gold-labelled question per record question. */
export function questionsFrom(file: string, split: string, grp: string): Ex[] {
  const out: Ex[] = [];
  for (const r of readJsonl(file)
    .filter((x) => x.split === split)
    .slice(0, CAP)) {
    const qs = r.questions as Record<string, { type: Kind; instructions: string; criteria: unknown }>;
    const as = r.answers as Record<string, Record<string, unknown>>;
    for (const [qid, q] of Object.entries(qs)) {
      const a = as[qid];
      let labels: string[];
      let gold: string | undefined;
      if (q.type === 'choice') {
        labels = Object.keys(q.criteria as object);
        gold = a.choice as string;
      } else if (q.type === 'noul') {
        labels = ['no', 'yes'];
        gold = (a.noul as number) >= 0.5 ? 'yes' : 'no';
      } else {
        labels = (q.criteria as string[]).map(levelName);
        const best = Object.entries(a.probabilities as Record<string, number>).sort((x, y) => y[1] - x[1])[0];
        gold = labels[Number(best[0])];
      }
      if (!gold || !labels.includes(gold)) continue;
      out.push({
        q: { kind: q.type, instructions: q.instructions, labels, desc: descriptions(q.type, q.criteria) },
        state: stateText(r.state),
        gold,
        grp,
      });
    }
  }
  return out;
}

export function benchItems(): BenchItem[] {
  const items: BenchItem[] = [];
  for (const tier of ['easy', 'original', 'hard']) {
    for (const r of readJsonl(`public/jevbench/items/${tier}.jsonl`)) {
      const q = r.question as { type: Kind; instructions: string; criteria: unknown };
      const labels = (r.labels as string[]).map(String);
      let expected = String(r.expected);
      if (!labels.includes(expected) && /^\d+$/.test(expected)) expected = labels[Number(expected)];
      items.push({
        id: r.id as string,
        family: r.family as string,
        kind: q.type,
        instructions: q.instructions,
        labels,
        desc: descriptions(q.type, q.criteria),
        state: stateText(r.state),
        expected,
      });
    }
  }
  return items;
}

/** Decide-ready question for a bench item (same shapes as the API question). */
export function decideQuestion(it: BenchItem): Question {
  if (it.kind === 'noul') return { type: 'noul', instructions: it.instructions };
  if (it.kind === 'score') return { type: 'score', instructions: it.instructions, options: it.labels };
  return { type: 'choice', instructions: it.instructions, options: it.labels, descriptions: it.desc };
}

export function answerLabel(a: Answer | undefined): string | null {
  if (!a) return null;
  if (a.type === 'noul') return a.answer ? 'yes' : 'no';
  if (a.type === 'score') return a.level;
  return a.choice;
}

/** Embeddings cached on disk so reruns skip the slow MiniLM pass. */
async function cachedEmbed(name: string, texts: string[]): Promise<Float32Array> {
  const bin = `${CACHE}-${name}.f32`;
  const meta = `${CACHE}-${name}.json`;
  if (existsSync(bin) && existsSync(meta)) {
    const { n } = JSON.parse(readFileSync(meta, 'utf8')) as { n: number };
    if (n === texts.length) {
      const buf = readFileSync(bin);
      return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    }
  }
  const E = await embedTexts(texts, (d, t) => {
    if (d % 3200 === 0 || d === t) console.log(`  ${name} embed ${d}/${t}`);
  });
  writeFileSync(bin, Buffer.from(E.buffer, E.byteOffset, E.byteLength));
  writeFileSync(meta, JSON.stringify({ n: texts.length }));
  return E;
}

function wilson(k: number, n: number): [number, number] {
  if (n === 0) return [0, 0];
  const z = 1.96,
    p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [c - h, c + h];
}

/** Linear head over pair embeddings: logit(yes) − logit(no) per row. */
function pairMargins(E: Float32Array, n: number, w: Float32Array, b: Float32Array): Float32Array {
  const m = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    let s = b[1] - b[0];
    for (let d = 0; d < EMBED_DIM; d++) s += E[r * EMBED_DIM + d] * (w[d * 2 + 1] - w[d * 2]);
    m[r] = s;
  }
  return m;
}

async function trainHead(X: Float32Array, Y: Uint32Array, n: number, seed: number) {
  const head = new CpuHead();
  head.init({ x: X, y: Y, n, d: EMBED_DIM, c: 2, lr: LR, batch: BATCH, epochs: EPOCHS, seed, backend: 'cpu', valFrac: 0 });
  const order = Array.from({ length: n }, (_, i) => i);
  let s = seed;
  const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let ep = 0; ep < EPOCHS && n > 0; ep++) {
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(rnd() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    for (let off = 0; off < n; off += BATCH) await head.step(Uint32Array.from(order.slice(off, off + BATCH)));
  }
  return head.weights();
}

/** Pick the highest-margin row per owner; returns the chosen label per owner. */
function argmaxPerOwner(owners: { owner: number; label: string }[], margins: Float32Array, n: number): (string | null)[] {
  const best = new Array<{ label: string; s: number } | undefined>(n);
  owners.forEach((row, r) => {
    const cur = best[row.owner];
    if (!cur || margins[r] > cur.s) best[row.owner] = { label: row.label, s: margins[r] };
  });
  return Array.from({ length: n }, (_, i) => best[i]?.label ?? null);
}

function summarize(name: string, items: BenchItem[], pred: (string | null)[]) {
  const per: Record<string, { n: number; k: number }> = {};
  let n = 0,
    k = 0;
  items.forEach((it, i) => {
    const ok = pred[i] === it.expected ? 1 : 0;
    n++;
    k += ok;
    for (const key of [it.family, it.kind]) {
      per[key] ??= { n: 0, k: 0 };
      per[key].n++;
      per[key].k += ok;
    }
  });
  const [lo, hi] = wilson(k, n);
  return { name, n, acc: k / n, lo, hi, per };
}

async function main() {
  const files = readdirSync('public/jev-datasets').filter((f) => f.endsWith('.jsonl'));
  const groupOf = (f: string) => (f.startsWith('jev-') ? 'jev' : f.startsWith('general-') ? 'general' : null);
  const train: Ex[] = [];
  const val: Ex[] = [];
  for (const f of files) {
    const grp = groupOf(f);
    if (!grp) continue;
    train.push(...questionsFrom(`public/jev-datasets/${f}`, 'train', grp));
    val.push(...questionsFrom(`public/jev-datasets/${f}`, 'val', grp));
  }
  const items = benchItems();
  console.log(
    `train questions ${train.length} (jev ${train.filter((e) => e.grp === 'jev').length}, general ${train.filter((e) => e.grp === 'general').length}); val ${val.length}; bench items ${items.length}`,
  );

  type Row = { owner: number; label: string; y: number; grp: string };
  const trainRows: Row[] = [];
  const trainTexts: string[] = [];
  train.forEach((e, owner) => {
    for (const label of e.q.labels) {
      trainRows.push({ owner, label, y: label === e.gold ? 1 : 0, grp: e.grp });
      trainTexts.push(pairText(e.q, label, e.state));
    }
  });
  const valRows: Row[] = [];
  const valTexts: string[] = [];
  val.forEach((e, owner) => {
    for (const label of e.q.labels) {
      valRows.push({ owner, label, y: label === e.gold ? 1 : 0, grp: e.grp });
      valTexts.push(pairText(e.q, label, e.state));
    }
  });
  const benchRows: { owner: number; label: string }[] = [];
  const benchTexts: string[] = [];
  items.forEach((it, owner) => {
    for (const label of it.labels) {
      benchRows.push({ owner, label });
      benchTexts.push(pairText(it, label, it.state));
    }
  });

  const t0 = Date.now();
  const Etrain = await cachedEmbed('train', trainTexts);
  const Eval = await cachedEmbed('val', valTexts);
  const Ebench = await cachedEmbed('bench', benchTexts);
  console.log(`embeddings ready in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

  const conditions = [
    { name: 'untrained', pick: () => false },
    { name: 'general', pick: (g: string) => g === 'general' },
    { name: 'jev', pick: (g: string) => g === 'jev' },
    { name: 'all', pick: () => true },
  ];
  const results: ReturnType<typeof summarize>[] = [];
  const inDist: { name: string; acc: number; n: number }[] = [];
  for (const c of conditions) {
    const rows = trainRows.map((r, i) => ({ ...r, i })).filter((r) => c.pick(r.grp));
    const X = new Float32Array(rows.length * EMBED_DIM);
    const Y = new Uint32Array(rows.length);
    rows.forEach((r, k) => {
      X.set(Etrain.subarray(r.i * EMBED_DIM, (r.i + 1) * EMBED_DIM), k * EMBED_DIM);
      Y[k] = r.y;
    });
    const { w, b } = await trainHead(X, Y, rows.length, 1337);

    const vm = pairMargins(Eval, valRows.length, w, b);
    const vpred = argmaxPerOwner(valRows, vm, val.length);
    const vhit = val.filter((e, i) => vpred[i] === e.gold).length;
    inDist.push({ name: c.name, acc: vhit / val.length, n: val.length });

    const bm = pairMargins(Ebench, benchRows.length, w, b);
    results.push(summarize(`${c.name} (${rows.length} pairs)`, items, argmaxPerOwner(benchRows, bm, items.length)));
    console.log(`${c.name}: trained on ${rows.length} pairs`);
  }

  if (process.env.DECIDE !== '0') {
    const decide = await loadDecide(DECIDE_REPO, { device: 'cpu', dtype: 'q4' });
    const pred: (string | null)[] = [];
    const t1 = Date.now();
    for (const [i, it] of items.entries()) {
      try {
        const [a] = await decide.decide(it.state, [decideQuestion(it)]);
        pred.push(answerLabel(a));
      } catch {
        pred.push(null);
      }
      if (i % 50 === 0) console.log(`  decide ${i}/${items.length} (${((Date.now() - t1) / 1000).toFixed(0)}s)`);
    }
    results.push(summarize(`GLiNER2.5 Decide zero-shot (${decide.device} ${decide.dtype})`, items, pred));
  }

  console.log('\nIN-DISTRIBUTION (jev/general val split, held out from training)');
  for (const d of inDist) console.log(`  ${d.name.padEnd(10)} ${(d.acc * 100).toFixed(1)}%  (n=${d.n})`);
  console.log('\nJEVBENCH (231 public items)');
  for (const r of results) {
    console.log(`  ${r.name}\n    overall ${(r.acc * 100).toFixed(1)}%  95% CI [${(r.lo * 100).toFixed(1)}, ${(r.hi * 100).toFixed(1)}]`);
    console.log('    ' + Object.entries(r.per).map(([k, v]) => `${k} ${((v.k / v.n) * 100).toFixed(0)}%(${v.n})`).join(' · '));
  }
  writeFileSync(OUT, JSON.stringify({ inDist, results }, null, 1));
  console.log(`\nwrote ${OUT}`);
}

if (import.meta.main) await main();
