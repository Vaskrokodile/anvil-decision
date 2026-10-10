// Pull larger real-text samples for the general-purpose Jev sets into
// public/datasets/general-*.jsonl ({text,label}). Shuffled with a fixed seed.
// Run: node scripts/fetch-general.mts
import { existsSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fetchRows } from '../src/lib/hfDatasets.ts';
import { mulberry } from '../src/lib/split.ts';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'datasets');
const N = 2000;
const clean = (s: string) => s.replace(/\s+/g, ' ').trim();

const SOURCES: {
  file: string;
  dataset: string;
  config: string;
  split: string;
  pick: (r: Record<string, string>) => { text: string; label: string } | null;
}[] = [
  {
    file: 'general-topic.jsonl',
    dataset: 'fancyzhx/ag_news',
    config: 'default',
    split: 'train',
    pick: (r) => {
      const text = clean(r.text ?? '').slice(0, 700);
      return text && r.label ? { text, label: clean(r.label) } : null;
    },
  },
  {
    file: 'general-emotion.jsonl',
    dataset: 'dair-ai/emotion',
    config: 'split',
    split: 'train',
    pick: (r) => {
      const text = clean(r.text ?? '');
      return text && r.label ? { text, label: clean(r.label) } : null;
    },
  },
  {
    file: 'general-sentiment.jsonl',
    dataset: 'stanfordnlp/sst2',
    config: 'default',
    split: 'train',
    pick: (r) => {
      const text = clean(r.sentence ?? r.text ?? '');
      return text && r.label ? { text, label: clean(r.label) } : null;
    },
  },
];

for (const s of SOURCES) {
  if (existsSync(join(OUT, s.file))) {
    console.log(`${s.file} exists — skipped`);
    continue;
  }
  process.stdout.write(`${s.dataset} … `);
  let rows: Record<string, string>[] = [];
  for (let attempt = 0; ; attempt++) {
    try {
      ({ rows } = await fetchRows(s.dataset, s.config, s.split, N, undefined, 2500));
      break;
    } catch (e) {
      if (attempt >= 4) throw e;
      console.log(`rate-limited, cooling down 90s…`);
      await new Promise((r) => setTimeout(r, 90000));
    }
  }
  const picked = rows.map(s.pick).filter((x): x is { text: string; label: string } => x !== null);
  const seen = new Set<string>();
  const uniq = picked.filter((r) => (seen.has(r.text) ? false : (seen.add(r.text), true)));
  const rng = mulberry(1337);
  for (let k = uniq.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [uniq[k], uniq[j]] = [uniq[j], uniq[k]];
  }
  writeFileSync(join(OUT, s.file), uniq.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf8');
  const classes = new Set(uniq.map((r) => r.label)).size;
  console.log(`${uniq.length} rows · ${classes} classes → ${s.file}`);
}
