// Mirror JevBench (MIT, github.com/fstandhartinger/jevbench) into public/jevbench/:
//   leaderboard.json  — published aggregate results (official board, v1.4.2.2)
//   items/<tier>.jsonl — PUBLIC items only (easy, original, hard). Sealed/held-out
//                        items are never published, so they cannot be run here.
// Run: node scripts/fetch-jevbench.mts
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'public', 'jevbench');
const RAW = 'https://raw.githubusercontent.com/fstandhartinger/jevbench/HEAD';
const RESULTS = 'results/v1.4.2.2/jevbench-v1.4.2.2-results.json';
const TIERS = ['easy', 'original', 'hard'] as const;

async function text(path: string): Promise<string> {
  const r = await fetch(`${RAW}/${path}`);
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
  return r.text();
}

mkdirSync(join(OUT, 'items'), { recursive: true });

const raw = JSON.parse(await text(RESULTS)) as {
  revision: string;
  generated_utc: string;
  scoring: { jevbench_score: string };
  systems: Record<string, unknown>[];
};

const systems = raw.systems.map((s) => {
  const cost = s.cost as { kind?: string; usd_per_1000?: number } | undefined;
  const speed = s.speed as { p50_s_adjusted?: number; p95_s_adjusted?: number } | undefined;
  const cal = s.calibration as { score?: number } | undefined;
  return {
    key: s.key,
    display: s.display,
    author: s.author,
    repo: s.repo,
    class: s.class,
    licence: s.licence,
    open: s.open,
    endpoint_kind: s.endpoint_kind,
    api_flag: s.api_flag ?? false,
    partial: s.partial ?? false,
    ranked: s.ranked ?? false,
    listing: s.listing,
    not_ranked_because: s.not_ranked_because ?? null,
    rank: s.rank ?? null,
    jevbench_score: s.jevbench_score ?? null,
    axes: {
      intelligence: (s.axes as Record<string, number>)?.intelligence ?? null,
      calibration: (s.axes as Record<string, number>)?.calibration ?? cal?.score ?? null,
      speed: (s.axes as Record<string, number>)?.speed ?? null,
      cost: (s.axes as Record<string, number>)?.cost ?? null,
    },
    tiers: s.tiers ?? null,
    p50_s_adjusted: speed?.p50_s_adjusted ?? null,
    p95_s_adjusted: speed?.p95_s_adjusted ?? null,
    usd_per_1000: cost?.usd_per_1000 ?? null,
    cost_kind: cost?.kind ?? null,
    public_accuracy: s.public_accuracy ?? null,
    sealed_accuracy: s.sealed_accuracy ?? null,
    public_minus_sealed_gap_pp: s.public_minus_sealed_gap_pp ?? null,
  };
});

writeFileSync(
  join(OUT, 'leaderboard.json'),
  JSON.stringify(
    {
      source: 'https://github.com/fstandhartinger/jevbench',
      site: 'https://benchmarkheaven.com/jev-models',
      license: 'MIT (JevBench); each system keeps its own licence',
      revision: raw.revision,
      generatedUtc: raw.generated_utc,
      scoring: raw.scoring.jevbench_score,
      systems,
    },
    null,
    1,
  ) + '\n',
  'utf8',
);
console.log(`leaderboard: ${systems.length} systems, revision ${raw.revision}`);

for (const t of TIERS) {
  writeFileSync(join(OUT, 'items', `${t}.jsonl`), await text(`datasets/public/${t}.jsonl`), 'utf8');
  console.log(`items/${t}.jsonl written`);
}
