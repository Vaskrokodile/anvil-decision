// Validate Jev-format training records: request shape, answer shape, and that
// every answer is consistent with its question. Exits 1 on any violation.
// Run: node scripts/datasets/validate-jev.mts [file.jsonl ...]   (default: all of public/jev-datasets)
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'public', 'jev-datasets');
const files = process.argv.length > 2 ? process.argv.slice(2) : readdirSync(DIR).filter((f) => f.endsWith('.jsonl')).map((f) => join(DIR, f));

let bad = 0;
const fail = (file: string, id: string, msg: string) => {
  bad++;
  if (bad <= 20) console.log(`FAIL ${file} ${id}: ${msg}`);
};
const near1 = (x: number) => Math.abs(x - 1) < 1e-3;

for (const file of files) {
  const lines = readFileSync(file, 'utf8').trim().split('\n');
  const ids = new Set<string>();
  const classes = { train: new Set<string>(), val: new Set<string>() };
  let n = 0;
  for (const line of lines) {
    const r = JSON.parse(line) as Record<string, any>;
    n++;
    const id = String(r.id);
    if (ids.has(id)) fail(file, id, 'duplicate id');
    ids.add(id);
    if (r.split !== 'train' && r.split !== 'val') fail(file, id, `bad split ${r.split}`);
    if (typeof r.state !== 'string' || !r.state.trim()) fail(file, id, 'empty state');
    const qids = Object.keys(r.questions ?? {});
    const aids = Object.keys(r.answers ?? {});
    if (qids.length === 0) fail(file, id, 'no questions');
    if (qids.join() !== aids.join()) fail(file, id, `answers keys ${aids} ≠ questions keys ${qids}`);
    for (const qid of qids) {
      const q = r.questions[qid];
      const a = r.answers[qid];
      if (!a) continue;
      if (q.type !== a.type) fail(file, id, `type mismatch ${q.type}/${a.type}`);
      if (typeof q.instructions !== 'string' || !q.instructions) fail(file, id, 'missing instructions');
      if (q.type === 'choice') {
        const keys = Object.keys(q.criteria ?? {});
        if (keys.length < 2) fail(file, id, 'choice with <2 options');
        if (!keys.includes(a.choice)) fail(file, id, `choice ${a.choice} not in criteria`);
        const pk = Object.keys(a.probabilities ?? {});
        if (pk.sort().join() !== [...keys].sort().join()) fail(file, id, 'probabilities keys ≠ criteria');
        const sum = Object.values(a.probabilities ?? {}).reduce((s: number, x: any) => s + Number(x), 0) as number;
        if (!near1(sum)) fail(file, id, `probabilities sum ${sum}`);
        if (typeof a.confidence !== 'number' || a.confidence < 0 || a.confidence > 1) fail(file, id, 'confidence out of range');
        if (r.split === 'train' || r.split === 'val') classes[r.split as 'train' | 'val'].add(a.choice);
      } else if (q.type === 'score') {
        const levels = q.criteria;
        if (!Array.isArray(levels) || levels.length < 2 || levels.length > 10) fail(file, id, 'score levels 2–10');
        if (a.score < 0 || a.score > levels.length - 1) fail(file, id, `score ${a.score} out of range`);
        const sum = Object.values(a.probabilities ?? {}).reduce((s: number, x: any) => s + Number(x), 0) as number;
        if (!near1(sum)) fail(file, id, `probabilities sum ${sum}`);
        if (Object.keys(a.probabilities ?? {}).length !== levels.length) fail(file, id, 'probabilities ≠ levels');
        const exp = Object.values(a.probabilities as Record<string, number>).reduce((s, p, i) => s + i * p, 0);
        if (Math.abs(exp - a.score) > 1e-3) fail(file, id, `score ${a.score} ≠ Σ i·p ${exp.toFixed(4)}`);
        if (typeof a.confidence !== 'number' || a.confidence < 0 || a.confidence > 1) fail(file, id, 'confidence out of range');
        classes[r.split as 'train' | 'val'].add(String(Math.round(a.score)));
      } else if (q.type === 'noul') {
        if (!q.criteria || !('true' in q.criteria) || !('false' in q.criteria)) fail(file, id, 'noul criteria needs true/false');
        if (typeof a.noul !== 'number' || a.noul < 0 || a.noul > 1) fail(file, id, 'noul out of [0,1]');
        classes[r.split as 'train' | 'val'].add(a.noul >= 0.5 ? 'yes' : 'no');
      } else fail(file, id, `unknown question type ${q.type}`);
    }
  }
  const name = file.split(/[\\/]/).pop();
  const missing = [...classes.train].filter((c) => !classes.val.has(c));
  console.log(`${name}: ${n} records · train classes ${classes.train.size} · val classes ${classes.val.size}${missing.length ? ' · val missing ' + missing.join(',') : ''}`);
}
console.log(bad ? `RESULT: ${bad} violation(s)` : 'RESULT: all records valid');
process.exit(bad ? 1 : 0);
