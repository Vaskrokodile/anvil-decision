// Finalize a hand-authored dataset from its sources. Reproducible: same inputs,
// same output. Sources live in datasets/sources/<name>/:
//   gold.jsonl          author-written rows {id,text,label}
//   blind.jsonl         blind relabel by a second agent {id,label,confidence}
//   decisions.jsonl     (optional) adjudication of disputed rows {id,label,sure}
//   exclude.json        (optional) array of ids dropped for safety/quality reasons
//
// Rule: a row is kept with label L when
//   - it is in decisions.jsonl with sure=true (label L from the adjudicator), or
//   - it is not disputed (blind label == gold label), and it is not excluded.
// Everything else is dropped. Classes are then downsampled to the smallest class
// (seeded, deterministic) so the published file is balanced.
//
// Run: node scripts/datasets/finalize-hand.mts <name>
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mulberry } from '../../src/lib/split.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const name = process.argv[2];
if (!name) {
  console.error('usage: node scripts/datasets/finalize-hand.mts <name>');
  process.exit(2);
}
const src = join(ROOT, 'datasets', 'sources', name);
const readText = (f: string) => readFileSync(join(src, f), 'utf8').replace(/^\uFEFF/, '');
const readJsonl = (f: string) => readText(f).trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

const gold = readJsonl('gold.jsonl') as { id: string; text: string; label: string }[];
const blind = new Map((readJsonl('blind.jsonl') as { id: string; label: string }[]).map((r) => [r.id, r.label]));
const decisions = existsSync(join(src, 'decisions.jsonl'))
  ? new Map((readJsonl('decisions.jsonl') as { id: string; label: string; sure: boolean }[]).map((r) => [r.id, r]))
  : new Map<string, { id: string; label: string; sure: boolean }>();
const excluded = new Set<string>(
  existsSync(join(src, 'exclude.json')) ? (JSON.parse(readText('exclude.json')) as string[]) : [],
);

const kept: { id: string; text: string; label: string; via: string }[] = [];
const drops: Record<string, number> = { excluded: 0, disputed: 0, unsure: 0 };
for (const g of gold) {
  if (excluded.has(g.id)) {
    drops.excluded++;
    continue;
  }
  const d = decisions.get(g.id);
  if (d) {
    if (!d.sure) {
      drops.unsure++;
      continue;
    }
    kept.push({ id: g.id, text: g.text, label: d.label, via: 'adjudicated' });
    continue;
  }
  if (blind.get(g.id) === g.label) kept.push({ id: g.id, text: g.text, label: g.label, via: 'agreed' });
  else drops.disputed++;
}

// balance: cap every class at 1.15 x the smallest class, deterministically
const byLabel = new Map<string, typeof kept>();
for (const r of kept) byLabel.set(r.label, [...(byLabel.get(r.label) ?? []), r]);
const minN = Math.min(...[...byLabel.values()].map((a) => a.length));
const cap = Math.floor(minN * 1.15);
const rng = mulberry(1337);
const balanced: { text: string; label: string }[] = [];
for (const [label, rows] of [...byLabel.entries()].sort()) {
  const a = rows.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  for (const r of a.slice(0, cap)) balanced.push({ text: r.text, label });
}
for (let i = balanced.length - 1; i > 0; i--) {
  const j = Math.floor(rng() * (i + 1));
  [balanced[i], balanced[j]] = [balanced[j], balanced[i]];
}

const out = join(ROOT, 'public', 'datasets', `${name}.jsonl`);
writeFileSync(out, balanced.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf8');

const viaCount = kept.reduce((m, r) => ((m[r.via] = (m[r.via] ?? 0) + 1), m), {} as Record<string, number>);
console.log(`${name}: gold ${gold.length} → kept ${kept.length} (${JSON.stringify(viaCount)}) → balanced ${balanced.length}`);
console.log(`dropped: ${JSON.stringify(drops)}`);
console.log(`classes: ${[...byLabel.entries()].map(([l, a]) => `${l}=${a.length}`).join(' ')} → ${minN} each`);
console.log(`wrote ${out}`);
