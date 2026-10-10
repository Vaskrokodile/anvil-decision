// Teacher-verify a {text,label} dataset with the OpenJEV decision API.
// Each row is asked one `choice` question over the dataset's label set; we keep
// the row when Jev's argmax agrees with the gold label. Results are cached in
// .ref/teacher-cache/ so interrupted runs resume without re-spending calls.
//
//   node scripts/jev-teacher.mts <data.jsonl> <teacher.json> [--write] [--min-conf=0.5]
//
// teacher.json: { "instructions": "...", "criteria": { "<label>": "<description>" } }
// Key: OPENJEV_API_KEY env var, or OPENJEV_API_KEY= line in .env.local (gitignored).
import { readFileSync, writeFileSync, existsSync, mkdirSync, appendFileSync } from 'node:fs';
import { basename, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseJsonl } from '../src/lib/dataset.ts';
import { jevDecide, choiceProb, type JevAnswer } from '../src/lib/jevApi.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const [dataPath, teacherPath] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const flags = Object.fromEntries(
  process.argv.filter((a) => a.startsWith('--')).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v ?? 'true'];
  }),
);
if (!dataPath || !teacherPath) {
  console.error('usage: node scripts/jev-teacher.mts <data.jsonl> <teacher.json> [--write] [--min-conf=0.5]');
  process.exit(2);
}

function loadKey(): string {
  if (process.env.OPENJEV_API_KEY) return process.env.OPENJEV_API_KEY;
  const envFile = join(ROOT, '.env.local');
  if (existsSync(envFile)) {
    const line = readFileSync(envFile, 'utf8').split('\n').find((l) => l.startsWith('OPENJEV_API_KEY='));
    if (line) return line.slice('OPENJEV_API_KEY='.length).trim();
  }
  throw new Error('No OPENJEV_API_KEY (env var or .env.local).');
}

const key = loadKey();
const teacher = JSON.parse(readFileSync(teacherPath, 'utf8')) as {
  instructions: string;
  criteria: Record<string, string>;
};
const labels = Object.keys(teacher.criteria);
const { rows } = parseJsonl(readFileSync(dataPath, 'utf8'));

const cacheDir = join(ROOT, '.ref', 'teacher-cache');
mkdirSync(cacheDir, { recursive: true });
const cachePath = join(cacheDir, `${basename(dataPath, '.jsonl')}.${basename(teacherPath, '.json')}.jsonl`);
const cache = new Map<string, JevAnswer>();
if (existsSync(cachePath))
  for (const line of readFileSync(cachePath, 'utf8').split('\n').filter(Boolean)) {
    const c = JSON.parse(line) as { k: string; a: JevAnswer };
    cache.set(c.k, c.a);
  }

const questions = {
  team: { type: 'choice' as const, instructions: teacher.instructions, criteria: teacher.criteria },
};

const CONC = 4;
let next = 0;
let done = 0;
let failed = 0;
const started = Date.now();

async function worker() {
  while (next < rows.length) {
    const i = next++;
    const text = rows[i].text;
    if (cache.has(text)) {
      done++;
      continue;
    }
    try {
      const { answers } = await jevDecide(key, text, questions);
      const a = answers.team;
      cache.set(text, a);
      appendFileSync(cachePath, JSON.stringify({ k: text, a }) + '\n');
    } catch (e) {
      failed++;
      if (failed <= 5) console.error(`row ${i}: ${(e as Error).message}`);
      if ((e as { status?: number }).status === 401) throw e; // no point continuing
    }
    done++;
    if (done % 100 === 0) {
      const rate = done / ((Date.now() - started) / 1000);
      console.log(`${done}/${rows.length}  ${rate.toFixed(1)} rows/s  failed=${failed}`);
    }
  }
}

try {
  await Promise.all(Array.from({ length: CONC }, worker));
} catch (e) {
  console.error(`stopped: ${(e as Error).message}`);
  process.exit(1);
}

// ---- agreement report ----
const minConf = Number(flags['min-conf'] ?? 0.5);
let agree = 0;
let scored = 0;
let goldProbSum = 0;
const perClass = new Map<string, { n: number; hit: number }>();
const verified: { text: string; label: string }[] = [];
for (const r of rows) {
  const a = cache.get(r.text);
  if (!a) continue;
  scored++;
  const pred = a.choice ?? labels[0];
  const pGold = choiceProb(a, r.label);
  goldProbSum += pGold;
  const pc = perClass.get(r.label) ?? { n: 0, hit: 0 };
  pc.n++;
  if (pred === r.label) {
    agree++;
    pc.hit++;
    if ((a.confidence ?? 0) >= minConf) verified.push({ text: r.text, label: r.label });
  }
  perClass.set(r.label, pc);
}

console.log(`\n${basename(dataPath)} vs ${basename(teacherPath)}`);
console.log(`scored ${scored}/${rows.length}  agreement ${((agree / Math.max(1, scored)) * 100).toFixed(1)}%  mean p(gold) ${(goldProbSum / Math.max(1, scored)).toFixed(3)}`);
for (const [l, c] of [...perClass.entries()].sort()) console.log(`  ${l.padEnd(24)} ${((c.hit / c.n) * 100).toFixed(1).padStart(5)}%  n=${c.n}`);
console.log(`verified (agree & conf>=${minConf}): ${verified.length}`);

if (flags.write) {
  const outPath = dataPath.replace(/\.jsonl$/, '.verified.jsonl');
  writeFileSync(outPath, verified.map((v) => JSON.stringify(v)).join('\n') + '\n', 'utf8');
  console.log(`wrote ${outPath}`);
}
