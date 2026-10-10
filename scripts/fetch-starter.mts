// Regenerate the bundled starter datasets from real Hugging Face data.
// Pulls subsets via datasets-server and writes public/datasets/*.jsonl +
// index.json (the manifest the studio reads at runtime).
// Run: node scripts/fetch-starter.mts
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fetchRows } from '../src/lib/hfDatasets.ts';
import type { DatasetRow } from '../src/lib/dataset.ts';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'datasets');

function mulberry(seed: number) {
  let s = seed >>> 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Target {
  file: string;
  name: string;
  desc: string;
  sector: string;
  dataset: string;
  config: string;
  split: string;
  license: string;
  /** how many raw rows to pull before class-balanced sampling */
  pool: number;
  perClass: number;
  /** build {text,label} from a raw row; return null to skip */
  toRow: (r: DatasetRow) => { text: string; label: string } | null;
  keepClasses?: string[];
  maxClasses?: number;
}

const clean = (s: string) => s.replace(/\s+/g, ' ').trim();

const TARGETS: Target[] = [
  {
    file: 'support-routing.jsonl',
    name: 'Support ticket routing',
    desc: 'Real customer tickets → route to the right support queue.',
    sector: 'decision',
    dataset: 'Tobi-Bueck/customer-support-tickets',
    config: 'default',
    split: 'train',
    license: 'CDLA-Sharing-1.0',
    pool: 2200,
    perClass: 40,
    maxClasses: 8,
    toRow: (r) => {
      if (r.language !== 'en') return null;
      const text = clean(`${r.subject}: ${r.body}`).slice(0, 600);
      const label = clean(r.queue ?? '');
      return text.length > 40 && label ? { text, label } : null;
    },
  },
  {
    file: 'banking77.jsonl',
    name: 'Banking77 · intent subset',
    desc: 'Banking customer queries → intent. 8 of the original 77 intents.',
    sector: 'decision',
    dataset: 'mteb/banking77',
    config: 'default',
    split: 'train',
    license: 'CC-BY-4.0',
    pool: 2500, // rows are label-sorted upstream; pool covers ~19 classes, take the biggest 8
    perClass: 35,
    maxClasses: 8,
    toRow: (r) => {
      const text = clean(r.text ?? '');
      const label = clean(r.label_text ?? '');
      return text && label ? { text, label } : null;
    },
  },
  {
    file: 'emotion.jsonl',
    name: 'Emotion classification',
    desc: 'Short texts → one of six emotions. Small, clean, fast to train on.',
    sector: 'labelling',
    dataset: 'dair-ai/emotion',
    config: 'split',
    split: 'train',
    license: 'see dataset card',
    pool: 1600,
    perClass: 50,
    toRow: (r) => {
      const text = clean(r.text ?? '');
      const label = clean(r.label ?? '');
      return text && label ? { text, label } : null;
    },
  },
  {
    file: 'agnews.jsonl',
    name: 'AG News topics',
    desc: 'News blurbs → world / sports / business / sci-tech taxonomy tagging.',
    sector: 'labelling',
    dataset: 'fancyzhx/ag_news',
    config: 'default',
    split: 'train',
    license: 'research use',
    pool: 1000,
    perClass: 65,
    toRow: (r) => {
      const text = clean(r.text ?? '').slice(0, 600);
      const label = clean(r.label ?? '');
      return text && label ? { text, label } : null;
    },
  },
  {
    file: 'sst2.jsonl',
    name: 'SST-2 sentiment',
    desc: 'The classic binary benchmark — positive vs. negative reviews.',
    sector: 'general',
    dataset: 'stanfordnlp/sst2',
    config: 'default',
    split: 'train',
    license: 'research use',
    pool: 800,
    perClass: 100,
    toRow: (r) => {
      const text = clean(r.text ?? r.sentence ?? '');
      const label = clean(r.label ?? '');
      return text && label ? { text, label } : null;
    },
  },
];

const rng = mulberry(1337);
mkdirSync(OUT, { recursive: true });
const manifest: Record<string, unknown>[] = [];
const force = process.argv.includes('--force');
// resumable: existing files keep their manifest entry (read back the counts)
const prevManifest: Record<string, Record<string, unknown>> = {};
try {
  const prev = JSON.parse(readFileSync(join(OUT, 'index.json'), 'utf8'));
  for (const d of prev.datasets ?? []) prevManifest[d.file] = d;
} catch {}

for (const t of TARGETS) {
  if (!force && existsSync(join(OUT, t.file))) {
    // file written by a previous (possibly crashed) run — reuse it
    let prev = prevManifest[t.file];
    if (!prev) {
      const lines = readFileSync(join(OUT, t.file), 'utf8').trim().split('\n');
      const labels = [...new Set(lines.map((l) => JSON.parse(l).label as string))];
      prev = {
        file: t.file,
        name: t.name,
        desc: t.desc,
        sector: t.sector,
        dataset: t.dataset,
        license: t.license,
        rows: lines.length,
        classes: labels.length,
        labels,
      };
    }
    manifest.push(prev);
    console.log(`${t.dataset} … cached (${prev.rows} rows) — skipped`);
    continue;
  }
  process.stdout.write(`${t.dataset} … `);
  const { rows } = await fetchRows(t.dataset, t.config, t.split, t.pool, undefined, 900);
  const byClass = new Map<string, { text: string; label: string }[]>();
  for (const r of rows) {
    const out = t.toRow(r);
    if (!out) continue;
    if (t.keepClasses && !t.keepClasses.includes(out.label)) continue;
    const arr = byClass.get(out.label) ?? [];
    arr.push(out);
    byClass.set(out.label, arr);
  }
  // keepClasses order wins; otherwise largest classes first, capped at maxClasses
  const classes = t.keepClasses
    ? t.keepClasses.filter((c) => byClass.has(c))
    : [...byClass.entries()]
        .sort((a, b) => b[1].length - a[1].length)
        .slice(0, t.maxClasses ?? 64)
        .map(([c]) => c);
  const picked: { text: string; label: string }[] = [];
  for (const c of classes) {
    const pool = byClass.get(c)!;
    for (let k = pool.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [pool[k], pool[j]] = [pool[j], pool[k]];
    }
    picked.push(...pool.slice(0, t.perClass));
  }
  for (let k = picked.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [picked[k], picked[j]] = [picked[j], picked[k]];
  }
  const jsonl = picked.map((r) => JSON.stringify(r)).join('\n') + '\n';
  writeFileSync(join(OUT, t.file), jsonl, 'utf8');
  manifest.push({
    file: t.file,
    name: t.name,
    desc: t.desc,
    sector: t.sector,
    dataset: t.dataset,
    license: t.license,
    rows: picked.length,
    classes: classes.length,
    labels: classes,
  });
  console.log(`${picked.length} rows · ${classes.length} classes → ${t.file}`);
}

// preserve hand-authored datasets not produced by this script but present on disk
const targetFiles = new Set(TARGETS.map((t) => t.file));
for (const [file, prev] of Object.entries(prevManifest)) {
  if (targetFiles.has(file) || manifest.some((m) => m.file === file)) continue;
  if (!existsSync(join(OUT, file))) continue;
  manifest.push(prev);
  console.log(`${file} … preserved (not managed by this script)`);
}

writeFileSync(
  join(OUT, 'index.json'),
  JSON.stringify({ generatedAt: new Date().toISOString(), datasets: manifest }, null, 2) + '\n',
);
console.log('manifest → index.json');
