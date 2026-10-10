// Shortcut-cue audit for {text,label} .jsonl datasets. Flags any token (or
// digit-shape like "$9,999") that predicts one label too reliably: a model can
// score well by memorising the cue instead of learning the task.
// Run: node scripts/leak-check.mts public/datasets/x.jsonl [...more files]
// Report-only: topical words ("iban" in a financial row) are legitimate cues;
// review the list for template artifacts (fixed phrasings, digit shapes, footers).
import { readFileSync } from 'node:fs';
import { parseJsonl } from '../src/lib/dataset.ts';

const MIN_SUPPORT = 8;
const MAX_SHARE = 0.85;

/** word tokens + digit-shape tokens ("$5,000" -> "shape:$9,999") */
function features(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of text.matchAll(/\$?\d[\d,.:/-]*\d|\$?\d|[\p{L}][\p{L}'-]*/gu)) {
    const t = m[0];
    if (/\d/.test(t)) {
      out.add(`shape:${t.replace(/\d/g, '9')}`);
      out.add(`num:${t.toLowerCase()}`);
    } else out.add(t.toLowerCase());
  }
  return out;
}

let flagged = 0;
for (const path of process.argv.slice(2)) {
  const { rows } = parseJsonl(readFileSync(path, 'utf8'));
  const labels = [...new Set(rows.map((r) => r.label))];
  const perFeat = new Map<string, Map<string, number>>();
  for (const r of rows)
    for (const f of features(r.text)) {
      const m = perFeat.get(f) ?? new Map<string, number>();
      m.set(r.label, (m.get(r.label) ?? 0) + 1);
      perFeat.set(f, m);
    }
  const cues: { f: string; label: string; n: number; share: number }[] = [];
  for (const [f, m] of perFeat) {
    const n = [...m.values()].reduce((a, b) => a + b, 0);
    if (n < MIN_SUPPORT) continue;
    const [label, top] = [...m.entries()].sort((a, b) => b[1] - a[1])[0];
    const share = top / n;
    if (share >= MAX_SHARE) cues.push({ f, label, n, share });
  }
  cues.sort((a, b) => b.n - a.n || b.share - a.share);
  flagged += cues.length;
  console.log(`${path}: ${rows.length} rows, ${labels.length} labels, ${cues.length} shortcut cue(s)`);
  for (const c of cues.slice(0, 25))
    console.log(`  ${(c.share * 100).toFixed(0).padStart(3)}% ${c.label.padEnd(14)} n=${String(c.n).padStart(3)}  ${c.f}`);
}
console.log(`RESULT: ${flagged} cue(s) listed for review`);
