import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowSquareOut, Play, SpinnerGap, Stop } from '@phosphor-icons/react';
import { Tag } from '../components/ui';
import {
  TIERS,
  loadTier,
  runItems,
  sampleItems,
  summarize,
  type BenchItem,
  type ItemResult,
  type RunSummary,
  type Tier,
} from './jevbench';

interface LbSystem {
  key: string;
  display: string;
  author: string | null;
  repo: string | null;
  class: string;
  licence: string | null;
  open: string;
  endpoint_kind: string | null;
  api_flag: boolean;
  partial: boolean;
  ranked: boolean;
  not_ranked_because: string | null;
  rank: number | null;
  jevbench_score: number | null;
  axes: { intelligence: number | null; calibration: number | null; speed: number | null; cost: number | null };
  p50_s_adjusted: number | null;
  usd_per_1000: number | null;
  cost_kind: string | null;
  public_accuracy: number | null;
  sealed_accuracy: number | null;
  public_minus_sealed_gap_pp: number | null;
}

interface Leaderboard {
  source: string;
  site: string;
  license: string;
  revision: string;
  generatedUtc: string;
  systems: LbSystem[];
}

type SortKey = 'score' | 'intelligence' | 'calibration' | 'speed' | 'cost' | 'public' | 'sealed';
const SORTS: { key: SortKey; label: string; get: (s: LbSystem) => number | null }[] = [
  { key: 'score', label: 'JevBench score', get: (s) => s.jevbench_score },
  { key: 'intelligence', label: 'intelligence', get: (s) => s.axes.intelligence },
  { key: 'calibration', label: 'calibration', get: (s) => s.axes.calibration },
  { key: 'speed', label: 'speed', get: (s) => s.axes.speed },
  { key: 'cost', label: 'cost', get: (s) => s.axes.cost },
  { key: 'public', label: 'public accuracy', get: (s) => s.public_accuracy },
  { key: 'sealed', label: 'sealed accuracy', get: (s) => s.sealed_accuracy },
];

const fmt = (v: number | null, d = 1) => (v == null ? '—' : v.toFixed(d));
const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

function Bar({ v, tone = 'blue' }: { v: number | null; tone?: 'blue' | 'dim' }) {
  const w = v == null ? 0 : Math.max(0, Math.min(100, v));
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-line">
      <div className={`h-full rounded-full ${tone === 'blue' ? 'bg-blue' : 'bg-dim'}`} style={{ width: `${w}%` }} />
    </div>
  );
}

function Notice() {
  return (
    <div className="mb-6 rounded-2xl border border-line bg-paper p-4 text-[13px] leading-relaxed text-dim">
      <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">how to read this</p>
      <p>
        Official scores are <a className="text-fg underline underline-offset-2" href="https://benchmarkheaven.com/jev-models">JevBench</a> v1.4.2.2
        (Benchmark Heaven, MIT). Their score also uses held-out <em>sealed</em> items that are never published, so
        the board shows their published aggregates. The in-browser runner can only use the <em>public</em> items,
        which are not sealed and may be trained on. Runner results are unofficial and labelled as such.
      </p>
    </div>
  );
}

function Leaderboard() {
  const [data, setData] = useState<Leaderboard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState('');
  const [cls, setCls] = useState('all');
  const [rankedOnly, setRankedOnly] = useState(true);
  const [sort, setSort] = useState<SortKey>('score');
  const [sel, setSel] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}jevbench/leaderboard.json`)
      .then((r) => (r.ok ? (r.json() as Promise<Leaderboard>) : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e: Error) => setErr(e.message));
  }, []);

  const classes = useMemo(() => (data ? [...new Set(data.systems.map((s) => s.class))].sort() : []), [data]);

  const rows = useMemo(() => {
    if (!data) return [];
    const sortDef = SORTS.find((s) => s.key === sort)!;
    const needle = q.trim().toLowerCase();
    return data.systems
      .filter((s) => (!rankedOnly || s.ranked))
      .filter((s) => cls === 'all' || s.class === cls)
      .filter((s) => !needle || s.display.toLowerCase().includes(needle) || (s.author ?? '').toLowerCase().includes(needle))
      .sort((a, b) => {
        const va = sortDef.get(a);
        const vb = sortDef.get(b);
        if (va == null && vb == null) return 0;
        if (va == null) return 1;
        if (vb == null) return -1;
        return vb - va;
      });
  }, [data, q, cls, rankedOnly, sort]);

  if (err) return <p className="font-mono text-[12px] text-fg">Could not load leaderboard ({err}).</p>;
  if (!data) return <p className="font-mono text-[11px] uppercase tracking-widest text-dim">loading board…</p>;

  const detail = data.systems.find((s) => s.key === sel) ?? null;

  return (
    <div>
      <Notice />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="search system or author"
          className="min-w-[14rem] flex-1 rounded-lg border border-line bg-card px-3 py-2 font-mono text-[12px] text-fg placeholder:text-faint focus:border-faint focus:outline-none"
        />
        <select value={cls} onChange={(e) => setCls(e.target.value)} className="rounded-lg border border-line bg-card px-2.5 py-2 font-mono text-[11px] text-fg">
          <option value="all">all classes</option>
          {classes.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className="rounded-lg border border-line bg-card px-2.5 py-2 font-mono text-[11px] text-fg">
          {SORTS.map((s) => (
            <option key={s.key} value={s.key}>sort: {s.label}</option>
          ))}
        </select>
        <label className="flex cursor-pointer items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-dim">
          <input type="checkbox" checked={rankedOnly} onChange={(e) => setRankedOnly(e.target.checked)} className="accent-blue" />
          ranked only
        </label>
        <span className="ml-auto font-mono text-[10px] uppercase tracking-widest text-faint">
          {rows.length} systems · rev {data.revision}
        </span>
      </div>

      {detail && <Detail s={detail} onClose={() => setSel(null)} />}

      <div className="overflow-x-auto rounded-2xl border border-line bg-card">
        <table className="w-full min-w-[860px] border-collapse text-left">
          <thead>
            <tr className="border-b border-line-soft font-mono text-[10px] font-normal uppercase tracking-[0.14em] text-faint">
              <th className="px-3 py-2.5">#</th>
              <th className="px-3 py-2.5">system</th>
              <th className="px-3 py-2.5 text-right">score</th>
              <th className="px-3 py-2.5 text-right">intel</th>
              <th className="px-3 py-2.5 text-right">calib</th>
              <th className="px-3 py-2.5 text-right">speed</th>
              <th className="px-3 py-2.5 text-right">cost</th>
              <th className="px-3 py-2.5 text-right">public acc</th>
              <th className="px-3 py-2.5 text-right">sealed acc</th>
              <th className="px-3 py-2.5 text-right">$/1k dec.</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr
                key={s.key}
                onClick={() => setSel(s.key === sel ? null : s.key)}
                className={`cursor-pointer border-b border-line-soft font-mono text-[11.5px] last:border-0 hover:bg-paper ${s.key === sel ? 'bg-paper' : ''}`}
              >
                <td className="px-3 py-2.5 text-faint">{s.rank ?? '—'}</td>
                <td className="px-3 py-2.5">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="font-sans text-[13px] text-fg">{s.display}</span>
                    {s.api_flag && <Tag tone="line">API</Tag>}
                    {s.partial && <Tag tone="faint">partial</Tag>}
                    {!s.ranked && <Tag tone="faint">unranked</Tag>}
                  </div>
                  <div className="text-[10.5px] text-faint">{s.author ?? '—'} · {s.class}</div>
                </td>
                <td className="px-3 py-2.5 text-right text-fg">{fmt(s.jevbench_score)}</td>
                <td className="px-3 py-2.5 text-right text-dim">{fmt(s.axes.intelligence)}</td>
                <td className="px-3 py-2.5 text-right text-dim">{fmt(s.axes.calibration)}</td>
                <td className="px-3 py-2.5 text-right text-dim">{fmt(s.axes.speed)}</td>
                <td className="px-3 py-2.5 text-right text-dim">{fmt(s.axes.cost)}</td>
                <td className="px-3 py-2.5 text-right text-dim">{pct(s.public_accuracy)}</td>
                <td className="px-3 py-2.5 text-right text-dim">{pct(s.sealed_accuracy)}</td>
                <td className="px-3 py-2.5 text-right text-faint">{s.usd_per_1000 == null ? '—' : `$${s.usd_per_1000.toFixed(3)}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 font-mono text-[10px] leading-relaxed text-faint">
        Score = equal-weight harmonic mean of intelligence, calibration, speed and cost (JevBench v1.4). Cost is per 1,000
        decisions, mostly estimated from model price cards. Click a row for details.
      </p>
    </div>
  );
}

function Detail({ s, onClose }: { s: LbSystem; onClose: () => void }) {
  const axes: [string, number | null][] = [
    ['intelligence', s.axes.intelligence],
    ['calibration', s.axes.calibration],
    ['speed', s.axes.speed],
    ['cost', s.axes.cost],
  ];
  return (
    <div className="mb-4 rounded-2xl border border-line bg-card p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-faint">[ system ]</p>
          <h3 className="font-display text-2xl text-fg">{s.display}</h3>
          <p className="mt-1 font-mono text-[11px] text-dim">
            {s.author ?? 'unknown author'} · {s.class} · {s.licence ?? 'licence unknown'} · endpoint {s.endpoint_kind ?? '—'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {s.repo && (
            <a href={s.repo} target="_blank" rel="noreferrer" className="flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-dim hover:border-fg hover:text-fg">
              repo <ArrowSquareOut size={11} />
            </a>
          )}
          <button onClick={onClose} className="rounded-full border border-line px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-dim hover:text-fg">
            close
          </button>
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-2.5">
          {axes.map(([k, v]) => (
            <div key={k} className="grid grid-cols-[6rem_1fr_3rem] items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-widest text-faint">{k}</span>
              <Bar v={v} />
              <span className="text-right font-mono text-[11px] text-dim">{fmt(v)}</span>
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-2 font-mono text-[11px] text-dim">
          <p>public accuracy <span className="text-fg">{pct(s.public_accuracy)}</span></p>
          <p>sealed accuracy <span className="text-fg">{pct(s.sealed_accuracy)}</span></p>
          <p>
            public − sealed gap{' '}
            <span className="text-fg">{s.public_minus_sealed_gap_pp == null ? '—' : `${s.public_minus_sealed_gap_pp.toFixed(1)} pp`}</span>
          </p>
          <p>p50 latency (adjusted) <span className="text-fg">{s.p50_s_adjusted == null ? '—' : `${s.p50_s_adjusted.toFixed(2)} s`}</span></p>
          <p>cost ({s.cost_kind ?? 'n/a'}) <span className="text-fg">{s.usd_per_1000 == null ? '—' : `$${s.usd_per_1000.toFixed(3)} / 1k`}</span></p>
          {s.not_ranked_because && <p className="text-fg">not ranked: {s.not_ranked_because}</p>}
          {s.api_flag && <p>Rows flagged API sent item text to an operator endpoint.</p>}
        </div>
      </div>
    </div>
  );
}

function RunTab() {
  const [key, setKey] = useState('');
  const [tier, setTier] = useState<Tier>('easy');
  const [n, setN] = useState(24);
  const [loaded, setLoaded] = useState<{ tier: Tier; items: BenchItem[] } | null>(null);
  const items = loaded && loaded.tier === tier ? loaded.items : null;
  const [phase, setPhase] = useState<'idle' | 'running' | 'done'>('idle');
  const [live, setLive] = useState<ItemResult[]>([]);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const itemMap = useRef<Map<string, BenchItem>>(new Map());

  useEffect(() => {
    let active = true;
    loadTier(tier)
      .then((xs) => active && setLoaded({ tier, items: xs }))
      .catch((e: Error) => active && setErr(e.message));
    return () => {
      active = false;
    };
  }, [tier]);

  const start = async () => {
    if (!items || !key.trim()) return;
    setErr(null);
    setLive([]);
    setSummary(null);
    const sample = sampleItems(items, n);
    setTotal(sample.length);
    itemMap.current = new Map(sample.map((i) => [i.id, i]));
    const ac = new AbortController();
    abortRef.current = ac;
    setPhase('running');
    try {
      const res = await runItems(key.trim(), sample, tier, (r) => setLive((prev) => [...prev, r]), ac.signal);
      setSummary(summarize(res, itemMap.current));
      setPhase('done');
    } catch (e) {
      setErr((e as Error).message);
      setPhase('idle');
    } finally {
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();
  const correct = live.filter((r) => !r.error && r.correct).length;
  const okSoFar = live.filter((r) => !r.error).length;

  return (
    <div className="grid gap-3 lg:grid-cols-5">
      <div className="rounded-2xl border border-line bg-card p-5 lg:col-span-2">
        <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">[ api key ]</p>
        <input
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder="OpenJEV API key"
          autoComplete="off"
          spellCheck={false}
          className="mb-2 w-full rounded-lg border border-line bg-paper px-3 py-2.5 font-mono text-[12px] text-fg placeholder:text-faint focus:border-faint focus:outline-none"
        />
        <p className="mb-6 font-mono text-[10px] leading-relaxed text-faint">
          Held in memory for this tab only — never saved, never logged. It is sent only to api.openjev.sh. Get a key from
          openjev.sh; the first run checks it.
        </p>

        <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">[ run ]</p>
        <div className="mb-3 flex flex-col gap-2">
          {TIERS.map((t) => (
            <label key={t} className={`flex cursor-pointer items-center gap-3 rounded-xl border px-3.5 py-2.5 transition-colors ${tier === t ? 'border-blue bg-blue-dim/50' : 'border-line hover:border-faint'}`}>
              <input type="radio" name="tier" className="accent-blue" checked={tier === t} onChange={() => setTier(t)} disabled={phase === 'running'} />
              <span className="font-mono text-[11px] uppercase tracking-widest text-fg">{t}</span>
              <span className="ml-auto font-mono text-[10px] text-faint">
                {t === 'hard' ? 'long states · ~8.6k chars' : t === 'original' ? 'mixed families' : 'short, simple'}
              </span>
            </label>
          ))}
        </div>
        <div className="mb-6 flex items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-widest text-faint">items</span>
          {[12, 24, 48, 999].map((v) => (
            <button key={v} onClick={() => setN(v)} disabled={phase === 'running'} className={`rounded-full border px-3 py-1 font-mono text-[11px] ${n === v ? 'border-blue bg-blue-dim text-blue' : 'border-line text-dim'}`}>
              {v === 999 ? 'all' : v}
            </button>
          ))}
        </div>

        {phase === 'running' ? (
          <button onClick={stop} className="flex w-full items-center justify-center gap-2 rounded-full border border-fg px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-fg hover:bg-fg hover:text-card">
            <Stop size={13} weight="fill" /> stop
          </button>
        ) : (
          <button
            onClick={start}
            disabled={!items || !key.trim()}
            className="flex w-full items-center justify-center gap-2 rounded-full bg-fg px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Play size={13} weight="fill" /> run {tier} {n === 999 ? `(${items?.length ?? '…'})` : `(${Math.min(n, items?.length ?? n)})`}
          </button>
        )}
        {err && <p className="mt-3 rounded-xl border border-dashed border-fg/40 bg-paper px-4 py-3 font-mono text-[11px] text-fg">{err}</p>}
      </div>

      <div className="rounded-2xl border border-line bg-card p-5 lg:col-span-3">
        <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">[ results · unofficial ]</p>
        {phase === 'idle' && !summary && (
          <p className="font-mono text-[12px] leading-relaxed text-dim">
            Pick a tier and run the public JevBench items against the Jev decision API. Each item is one request; the
            score is chance-corrected intelligence plus calibration (ECE) on those items only.
          </p>
        )}

        {phase === 'running' && (
          <div>
            <div className="mb-2 flex items-center justify-between font-mono text-[11px] text-dim">
              <span className="flex items-center gap-2"><SpinnerGap size={13} className="animate-spin text-blue" /> {live.length}/{total}</span>
              <span>{okSoFar ? `${((correct / okSoFar) * 100).toFixed(0)}% so far` : '—'}</span>
            </div>
            <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-line">
              <div className="h-full rounded-full bg-blue transition-all" style={{ width: `${total ? (live.length / total) * 100 : 0}%` }} />
            </div>
          </div>
        )}

        {summary && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ['accuracy', pct(summary.accuracy)],
                ['intelligence', fmt(summary.intelligence)],
                ['ECE', summary.ece.toFixed(3)],
                ['p50 latency', `${(summary.p50Ms / 1000).toFixed(2)} s`],
              ].map(([k, v]) => (
                <div key={k} className="rounded-xl border border-line bg-paper p-3">
                  <p className="font-mono text-[9.5px] uppercase tracking-widest text-faint">{k}</p>
                  <p className="font-display text-2xl text-fg">{v}</p>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-1.5">
              <Tag tone="blue">n={summary.n}</Tag>
              {summary.errors > 0 && <Tag tone="line">{summary.errors} errors</Tag>}
              <Tag>mean p(gold) {summary.meanPGold.toFixed(3)}</Tag>
              <Tag tone="faint">public items only · not official</Tag>
            </div>
            <table className="w-full border-collapse text-left font-mono text-[11px]">
              <thead>
                <tr className="border-b border-line-soft text-[10px] uppercase tracking-widest text-faint">
                  <th className="py-2 font-normal">tier</th>
                  <th className="py-2 text-right font-normal">n</th>
                  <th className="py-2 text-right font-normal">accuracy</th>
                  <th className="py-2 text-right font-normal">intelligence</th>
                </tr>
              </thead>
              <tbody>
                {TIERS.filter((t) => summary.perTier[t].n > 0).map((t) => (
                  <tr key={t} className="border-b border-line-soft last:border-0 text-dim">
                    <td className="py-2 text-fg">{t}</td>
                    <td className="py-2 text-right">{summary.perTier[t].n}</td>
                    <td className="py-2 text-right">{pct(summary.perTier[t].accuracy)}</td>
                    <td className="py-2 text-right">{fmt(summary.perTier[t].intelligence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="font-mono text-[10px] leading-relaxed text-faint">
              Intelligence = (accuracy − chance) / (1 − chance), clipped at 0, macro-averaged over tiers — the JevBench
              chance-corrected form. ECE = expected calibration error over 10 confidence bins. This does not reproduce the
              official score (no sealed items, no speed/cost axes).
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function BenchPage() {
  const [tab, setTab] = useState<'board' | 'run'>('board');
  return (
    <div className="mx-auto max-w-[1200px] px-4 pb-16 pt-28 sm:px-6">
      <p className="mb-3 font-mono text-[10.5px] uppercase tracking-[0.28em] text-faint">[ 06 — benchmarks ]</p>
      <h1 className="font-display text-[clamp(2.2rem,4.4vw,3.4rem)] leading-[1.04] tracking-[-0.015em] text-fg">
        Decision benchmarks.
      </h1>
      <p className="mb-8 mt-3 max-w-[60ch] text-[15px] leading-relaxed text-dim">
        Compare decision models on JevBench, and run its public items against the Jev API yourself.
      </p>

      <div className="mb-6 flex gap-2">
        {(
          [
            ['board', 'JevBench leaderboard'],
            ['run', 'run on JevBench'],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            onClick={() => setTab(k)}
            className={`rounded-full border px-4 py-2 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors ${tab === k ? 'border-fg bg-fg text-card' : 'border-line text-dim hover:text-fg'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'board' ? <Leaderboard /> : <RunTab />}
    </div>
  );
}
