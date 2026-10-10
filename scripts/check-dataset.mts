// Check a {text,label} .jsonl dataset file: parses, counts dupes/classes,
// prints health stats and warnings. Exit 1 on structural problems.
// Run: node scripts/check-dataset.mts public/datasets/x.jsonl
import { readFileSync } from 'node:fs';
import { parseJsonl, analyzeDataset, dedupeDataset } from '../src/lib/dataset.ts';

const path = process.argv[2];
if (!path) {
  console.error('usage: node scripts/check-dataset.mts <file.jsonl>');
  process.exit(2);
}

const { columns, rows } = parseJsonl(readFileSync(path, 'utf8'));
const d = { name: path, sector: '', columns, rows, source: 'import' as const, createdAt: '' };
const h = analyzeDataset(d);

let fail = false;
const expect = ['text', 'label'];
if (JSON.stringify(columns) !== JSON.stringify(expect)) {
  console.log(`FAIL columns ${JSON.stringify(columns)} — expected ${JSON.stringify(expect)}`);
  fail = true;
}
const empties = rows.filter((r) => !(r.text ?? '').trim() || !(r.label ?? '').trim()).length;
if (empties > 0) {
  console.log(`FAIL ${empties} rows with empty text/label`);
  fail = true;
}
if (h.dupes > 0) {
  console.log(`FAIL ${h.dupes} duplicate rows (dedupe removes ${d.rows.length - dedupeDataset(d).rows.length})`);
  fail = true;
}

const lens = rows.map((r) => r.text.length).sort((a, b) => a - b);
const med = lens[Math.floor(lens.length / 2)];
const counts = h.classCounts.map(([, c]) => c);
const skew = counts.length ? Math.max(...counts) / Math.min(...counts) : 0;

console.log(`rows ${h.rows} · classes ${h.classCounts.length} · skew ${skew.toFixed(2)} · len min/med/max ${lens[0]}/${med}/${lens[lens.length - 1]}`);
console.log('class counts:');
for (const [l, c] of h.classCounts) console.log(`  ${String(c).padStart(4)}  ${l}`);
for (const w of h.warnings) console.log(`warn: ${w}`);
console.log(fail ? 'RESULT: FAIL' : 'RESULT: PASS');
process.exit(fail ? 1 : 0);
