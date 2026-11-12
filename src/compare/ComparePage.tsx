import { useEffect, useState } from 'react';
import { ArrowRight, Play } from '@phosphor-icons/react';
import { Tag } from '../components/ui';
import {
  answerToIndex,
  compareSet,
  type ComparisonReport,
  type EvalSet,
  type TaskComparison,
} from '../lib/evaluate';
import { RUNNABLE, isDecideOnnx, loadJev, type Answer, type Question } from '../lib/jev';
import { loadDecide } from '../lib/gliner2';
import { releaseSessions } from '../lib/sessions';
import { getTrainedHead, inferHead, type TrainedHead } from '../lib/trainedHead';
import { getFtArtifact, inferFtTasks, type FtArtifact } from '../lib/xformer/ft';
import { getWorkspaceItem } from '../lib/workspace';

/** Held-out rows scored per run — enough for stable accuracy, fast in a tab. */
const LIMIT = 48;

type Artifact = FtArtifact | TrainedHead;

function argmax(probs: Record<string, number>, labels: string[]): number | null {
  if (labels.length === 0) return null;
  let best = 0;
  labels.forEach((l, i) => {
    if ((probs[l] ?? 0) > (probs[labels[best]] ?? 0)) best = i;
  });
  return best;
}

/** One JEV-shaped choice question per task — the base answers them all in one call. */
function questionsFor(tasks: EvalSet['tasks']): Question[] {
  return tasks.map((t) => ({
    type: 'choice',
    instructions: `${t.name}: which label fits this text best?`,
    options: t.labels,
  }));
}

async function loadArtifact(kind: EvalSet['kind']): Promise<Artifact | null> {
  if (kind === 'ft') return getFtArtifact() ?? (await getWorkspaceItem('ftArtifact'));
  return getTrainedHead() ?? (await getWorkspaceItem('trainedHead'));
}

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

function delta(a: number, b: number): string {
  const d = (a - b) * 100;
  return `${d >= 0 ? '+' : ''}${d.toFixed(1)} pts`;
}

function Metric({ label, acc, sub, strong }: { label: string; acc: number; sub: string; strong?: boolean }) {
  return (
    <div className="min-w-0">
      <p className="mb-2 truncate font-mono text-[10px] uppercase tracking-[0.18em] text-faint">{label}</p>
      <p className={`font-display leading-none ${strong ? 'text-6xl text-fg' : 'text-5xl text-dim'}`}>
        {(acc * 100).toFixed(1)}
        <span className="text-2xl text-faint">%</span>
      </p>
      <p className="mt-2 truncate font-mono text-[10px] text-dim">{sub}</p>
    </div>
  );
}

function TaskTable({ tasks }: { tasks: TaskComparison[] }) {
  return (
    <table className="mt-5 w-full border-t border-line-soft font-mono text-[10.5px]">
      <thead>
        <tr className="text-left text-faint">
          <th className="py-1.5 pr-2 font-normal">task</th>
          <th className="py-1.5 pr-2 text-right font-normal">n</th>
          <th className="py-1.5 pr-2 text-right font-normal">trained</th>
          <th className="py-1.5 pr-2 text-right font-normal">base</th>
          <th className="py-1.5 text-right font-normal">Δ</th>
        </tr>
      </thead>
      <tbody>
        {tasks.map((t) => (
          <tr key={t.name} className="border-t border-line-soft text-dim">
            <td className="max-w-[12rem] truncate py-1.5 pr-2 text-fg">{t.name}</td>
            <td className="py-1.5 pr-2 text-right">{t.n}</td>
            <td className="py-1.5 pr-2 text-right text-fg">{pct(t.trained.acc)}</td>
            <td className="py-1.5 pr-2 text-right">{pct(t.base.acc)}</td>
            <td className="py-1.5 text-right">{delta(t.trained.acc, t.base.acc)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Confusion({ task }: { task: TaskComparison }) {
  const { confusion } = task.trained;
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse font-mono text-[10px]">
        <thead>
          <tr>
            <th className="p-1" />
            {task.labels.map((l) => (
              <th key={l} className="max-w-[5rem] truncate p-1 font-normal text-faint">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {confusion.map((row, i) => {
            const max = Math.max(1, ...row);
            return (
              <tr key={i}>
                <th className="max-w-[7rem] truncate p-1 text-left font-normal text-faint">{task.labels[i]}</th>
                {row.map((n, j) => (
                  <td
                    key={j}
                    className={`h-8 w-10 border border-line-soft p-1 text-center ${i === j ? 'text-fg' : 'text-dim'}`}
                    style={
                      n > 0
                        ? {
                            backgroundColor: `color-mix(in srgb, var(--color-ink) ${Math.round(
                              (n / max) * (i === j ? 85 : 45),
                            )}%, transparent)`,
                            color: n / max > 0.5 ? 'var(--color-card)' : undefined,
                          }
                        : undefined
                    }
                  >
                    {n || ''}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function ComparePage() {
  const [set, setSet] = useState<EvalSet | null | undefined>(undefined);
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [baseId, setBaseId] = useState(RUNNABLE[0].id);
  const [phase, setPhase] = useState<'idle' | 'loading' | 'running' | 'done'>('idle');
  const [pct01, setPct01] = useState(0);
  const [weightsPct, setWeightsPct] = useState<number | null>(null);
  const [report, setReport] = useState<ComparisonReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getWorkspaceItem('evalSet')
      .then(async (s) => {
        if (!live) return;
        setSet(s);
        if (s) {
          const a = await loadArtifact(s.kind);
          if (live) setArtifact(a);
        }
      })
      .catch(() => {
        if (live) setSet(null);
      });
    return () => {
      live = false;
    };
  }, []);

  const run = async () => {
    if (!set || !artifact) return;
    const s0 = set;
    const a0 = artifact;
    setPhase('loading');
    setPct01(0);
    setError(null);
    setReport(null);
    try {
      await releaseSessions(isDecideOnnx(baseId) ? 'decide' : 'jev');
      let decide: (text: string, qs: Question[]) => Promise<Answer[]>;
      let name: string;
      if (isDecideOnnx(baseId)) {
        const sess = await loadDecide(baseId, { onProgress: (p) => setPct01(p.progress) });
        decide = (t, q) => sess.decide(t, q);
        name = `GLiNER2.5 Decide · ${sess.device} ${sess.dtype}`;
      } else {
        const j = await loadJev(baseId, (p) => setPct01(p.progress));
        decide = (t, q) => j.decide(t, q);
        name = `${RUNNABLE.find((r) => r.id === baseId)?.label ?? baseId} · ${j.runtime.device} ${j.runtime.dtype}`;
      }
      const qs = questionsFor(s0.tasks);

      setPhase('running');
      const rep = await compareSet({
        set: s0,
        limit: LIMIT,
        trained: async (text) => {
          if (s0.kind === 'ft') {
            const res = await inferFtTasks(text, a0 as FtArtifact, {
              preferWgsl: true,
              onWeights: setWeightsPct,
            });
            return res.map((r) => argmax(r.probs, r.labels));
          }
          const probs = await inferHead(text, a0 as TrainedHead);
          return [argmax(probs, s0.tasks[0].labels)];
        },
        base: {
          name,
          run: async (text) =>
            (await decide(text, qs)).map((a, k) => answerToIndex(a, s0.tasks[k].labels)),
        },
        onRow: (d, t) => setPct01(d / t),
      });
      setReport(rep);
      setPhase('done');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('idle');
    }
  };

  const busy = phase === 'loading' || phase === 'running';
  const primary = report?.tasks[0];

  return (
    <main className="mx-auto max-w-[1200px] px-6 py-10 sm:py-14">
      <header className="mb-10">
        <p className="mb-4 font-mono text-[10px] uppercase tracking-[0.25em] text-faint">
          [ compare · trained vs base ]
        </p>
        <h1 className="max-w-[24ch] font-display text-[clamp(2.2rem,4.6vw,3.6rem)] leading-[1.03] tracking-[-0.015em] text-fg">
          Did tuning <span className="serif-i">beat the base?</span>
        </h1>
        <p className="mt-5 max-w-[62ch] text-[15px] leading-relaxed text-dim">
          Both engines answer the same held-out rows from your last training run. The base gets every task as
          one JEV-shaped decision call. Your tuned encoder answers each task from its own head.
        </p>
      </header>

      {set === undefined && <p className="font-mono text-[11px] uppercase tracking-widest text-faint">loading…</p>}

      {set === null && (
        <div className="rounded-2xl border border-line bg-card p-8">
          <p className="font-display text-2xl text-fg">Nothing to compare yet</p>
          <p className="mt-2 max-w-[54ch] text-[14px] text-dim">
            Train a model in the studio first. Its held-out rows and tuned weights are saved in this browser, and this
            page scores them against the base.
          </p>
          <a
            href="#/studio"
            className="mt-6 inline-flex items-center gap-2 rounded-full bg-fg px-5 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85"
          >
            Open the studio
            <ArrowRight size={13} />
          </a>
        </div>
      )}

      {set && (
        <div className="grid gap-4 lg:grid-cols-5">
          <div className="flex flex-col gap-4 lg:col-span-2">
            <div className="rounded-2xl border border-line bg-card p-5">
              <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">[ trained ]</p>
              {artifact ? (
                <>
                  <p className="font-display text-2xl text-fg">
                    {set.kind === 'ft'
                      ? `${(artifact as FtArtifact).mode === 'lora' ? `LoRA r=${(artifact as FtArtifact).rank}` : 'Full fine-tune'}`
                      : 'Linear head'}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Tag>
                      {set.kind === 'ft' && (artifact as FtArtifact).arch === 'deberta'
                        ? 'DeBERTa-v3-small'
                        : 'MiniLM-L6'}
                    </Tag>
                    <Tag tone="ink">{set.rows.length} held-out rows</Tag>
                    <Tag>{artifact.backend}</Tag>
                  </div>
                  <ul className="mt-4 flex flex-col gap-2 border-t border-line-soft pt-4">
                    {set.tasks.map((t, k) => (
                      <li key={t.name} className="flex items-center justify-between gap-3 font-mono text-[11px]">
                        <span className="truncate text-fg">{t.name}</span>
                        <span className="shrink-0 text-faint">
                          {set.rows.filter((r) => r.gold[k] >= 0).length} rows · {t.labels.length} labels
                        </span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-4 font-mono text-[10px] text-faint">
                    trained {new Date(set.createdAt).toLocaleString()} · split 80/20, seed 1337
                  </p>
                </>
              ) : (
                <p className="text-[13px] text-dim">
                  The held-out rows are saved, but the tuned weights are gone. Train again in the studio.
                </p>
              )}
            </div>

            <div className="rounded-2xl border border-line bg-card p-5">
              <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">[ base ]</p>
              <div className="flex flex-col gap-2">
                {RUNNABLE.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    disabled={busy}
                    onClick={() => setBaseId(r.id)}
                    className={`rounded-xl border px-4 py-2.5 text-left transition-colors disabled:opacity-50 ${
                      baseId === r.id ? 'border-fg bg-fg text-card' : 'border-line text-dim hover:border-fg hover:text-fg'
                    }`}
                  >
                    <span className="block font-mono text-[11px] uppercase tracking-[0.12em]">{r.label}</span>
                    <span className={`mt-0.5 block font-mono text-[10px] ${baseId === r.id ? 'text-card/70' : 'text-faint'}`}>
                      {r.note}
                    </span>
                  </button>
                ))}
              </div>
              <p className="mt-3 text-[12px] leading-relaxed text-faint">
                The base is not fine-tuned. It answers every task in one JEV-shaped call.
              </p>
              <button
                type="button"
                onClick={run}
                disabled={busy || !artifact}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-full bg-fg px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Play size={13} weight="fill" />
                {report ? 'Run again' : 'Run comparison'} · {Math.min(LIMIT, set.rows.length)} rows
              </button>
              {error && (
                <p className="mt-3 rounded-xl border border-dashed border-fg/40 px-4 py-3 font-mono text-[11px] text-fg">
                  {error}
                </p>
              )}
            </div>
          </div>

          <div className="lg:col-span-3">
            {busy && (
              <div className="rounded-2xl border border-line bg-card p-6">
                <div className="mb-3 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.18em] text-dim">
                  <span className="flex items-center gap-2">
                    <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ink" />
                    {phase === 'loading'
                      ? `loading base · ${Math.round(pct01 * 100)}%`
                      : `scoring rows · ${Math.round(pct01 * 100)}%`}
                  </span>
                  {weightsPct != null && weightsPct < 1 && phase === 'running' && (
                    <span className="text-faint">encoder weights {Math.round(weightsPct * 100)}%</span>
                  )}
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-line">
                  <div
                    className="h-full rounded-full bg-ink transition-[width] duration-300 ease-out"
                    style={{ width: `${Math.round(pct01 * 1000) / 10}%` }}
                  />
                </div>
                <p className="mt-4 font-mono text-[10px] text-faint">
                  {phase === 'loading'
                    ? 'The base loads once and stays cached for later runs.'
                    : 'Trained first, then the base, on the same rows.'}
                </p>
              </div>
            )}

            {!busy && !report && (
              <div className="flex h-full min-h-[260px] items-center justify-center rounded-2xl border border-dashed border-line p-8 text-center">
                <p className="max-w-[40ch] font-display text-2xl leading-snug text-dim">
                  Run the comparison to see accuracy, per-class results, and where the two disagree.
                </p>
              </div>
            )}

            {!busy && report && primary && (
              <div className="flex flex-col gap-4">
                <div className="rounded-2xl border border-line bg-card p-6">
                  <p className="mb-6 font-display text-3xl leading-tight text-fg">
                    {primary.trained.acc >= primary.base.acc ? 'Tuned model' : 'Base model'}{' '}
                    <span className="serif-i">leads</span> on {primary.name} by{' '}
                    {(Math.abs(primary.trained.acc - primary.base.acc) * 100).toFixed(1)} pts
                  </p>
                  <div className="mb-6 grid grid-cols-2 gap-6">
                    <Metric
                      label="trained"
                      acc={primary.trained.acc}
                      strong
                      sub={`f1 ${primary.trained.macroF1.toFixed(2)} · ${primary.trained.ms.toFixed(0)} ms/row`}
                    />
                    <Metric
                      label={report.base}
                      acc={primary.base.acc}
                      sub={`f1 ${primary.base.macroF1.toFixed(2)} · ${primary.base.ms.toFixed(0)} ms/row`}
                    />
                  </div>
                  <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-widest">
                    <Tag tone={primary.trained.acc >= primary.base.acc ? 'ink' : 'faint'}>
                      {delta(primary.trained.acc, primary.base.acc)}
                    </Tag>
                    <span className="text-faint">
                      on {primary.name} · {primary.n} rows
                      {report.agreement != null && ` · agree on ${Math.round(report.agreement * 100)}%`}
                    </span>
                  </div>
                  {report.tasks.length > 1 && <TaskTable tasks={report.tasks} />}
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-2xl border border-line bg-card p-5">
                    <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                      per class · {primary.name}
                    </p>
                    <table className="w-full font-mono text-[10.5px]">
                      <thead>
                        <tr className="text-left text-faint">
                          <th className="py-1.5 pr-2 font-normal">label</th>
                          <th className="py-1.5 pr-2 text-right font-normal">n</th>
                          <th className="py-1.5 pr-2 text-right font-normal">trained f1</th>
                          <th className="py-1.5 text-right font-normal">base f1</th>
                        </tr>
                      </thead>
                      <tbody>
                        {primary.trained.perClass.map((pc, i) => (
                          <tr key={pc.label} className="border-t border-line-soft text-dim">
                            <td className="max-w-[9rem] truncate py-1.5 pr-2 text-fg">{pc.label}</td>
                            <td className="py-1.5 pr-2 text-right">{pc.support}</td>
                            <td className="py-1.5 pr-2 text-right text-fg">{pc.f1.toFixed(2)}</td>
                            <td className="py-1.5 text-right">{primary.base.perClass[i]?.f1.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="rounded-2xl border border-line bg-card p-5">
                    <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                      confusion · trained · rows true, cols predicted
                    </p>
                    <Confusion task={primary} />
                  </div>
                </div>

                {report.misses.length > 0 && (
                  <div className="rounded-2xl border border-line bg-card p-5">
                    <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                      where they miss · first {report.misses.length}
                    </p>
                    <ul className="flex flex-col divide-y divide-line-soft">
                      {report.misses.map((m, i) => (
                        <li key={i} className="py-3">
                          <p className="line-clamp-2 text-[13px] leading-relaxed text-dim">{m.text}</p>
                          <div className="mt-2 flex flex-wrap gap-1.5 font-mono text-[10px]">
                            {report.tasks.map((t, k) => (
                              <span key={t.name} className="flex flex-wrap items-center gap-1.5 text-faint">
                                <span className="text-dim">{t.name}:</span>
                                <Tag tone="solid">gold {m.gold[k] ?? '—'}</Tag>
                                <Tag tone={m.trained[k] === m.gold[k] ? 'ink' : 'faint'}>
                                  trained {m.trained[k] ?? '—'}
                                </Tag>
                                <Tag tone={m.base[k] === m.gold[k] ? 'ink' : 'faint'}>base {m.base[k] ?? '—'}</Tag>
                              </span>
                            ))}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
