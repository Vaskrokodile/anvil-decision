import { useMemo, useState, type PointerEvent as ReactPointerEvent } from 'react';

/** One live training sample from the worker — a batch (ft) or an epoch (head). */
export interface TrainPoint {
  /** fractional epoch, 0 → epochs */
  epoch: number;
  loss: number;
  acc: number;
  /** performance.now() when the sample arrived */
  t: number;
}

export type TrainStage = 'embed' | 'fetch' | 'tok' | 'train' | 'done';

const PLAN: Record<'head' | 'ft', { id: TrainStage; label: string; weight: number }[]> = {
  head: [
    { id: 'embed', label: 'embed', weight: 0.15 },
    { id: 'train', label: 'train', weight: 0.85 },
  ],
  ft: [
    { id: 'fetch', label: 'weights', weight: 0.3 },
    { id: 'tok', label: 'tokenize', weight: 0.05 },
    { id: 'train', label: 'train', weight: 0.65 },
  ],
};

const PHASE_TEXT: Record<TrainStage, string> = {
  embed: 'embedding rows',
  fetch: 'fetching encoder weights',
  tok: 'tokenizing',
  train: 'training · live',
  done: 'complete',
};

function fmtDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}

const W = 640,
  H = 200,
  L = 46,
  R = 40,
  T = 12,
  B = 24;

/** Loss over training, with a smoothed trend, accuracy, and a hover readout. */
function LossChart({ points, epochs }: { points: TrainPoint[]; epochs: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const smooth = useMemo(() => {
    const out: number[] = [];
    points.forEach((p, i) => out.push(i === 0 ? p.loss : out[i - 1] * 0.88 + p.loss * 0.12));
    return out;
  }, [points]);

  const iw = W - L - R,
    ih = H - T - B;
  const span = Math.max(1, epochs);
  const x = (e: number) => L + (Math.min(span, Math.max(0, e)) / span) * iw;
  const lo = Math.min(...points.map((p) => p.loss));
  const hi = Math.max(...points.map((p) => p.loss));
  const pad = Math.max(0.02, (hi - lo) * 0.15);
  const yLo = Math.max(0, lo - pad),
    yHi = hi + pad;
  const yLoss = (v: number) => T + (1 - Math.min(1, Math.max(0, (v - yLo) / (yHi - yLo)))) * ih;
  const yAcc = (a: number) => T + (1 - Math.min(1, Math.max(0, a))) * ih;
  const grid = [0, 0.25, 0.5, 0.75, 1];
  const tickEvery = Math.max(1, Math.ceil(epochs / 8));
  const ticks = Array.from({ length: epochs + 1 }, (_, e) => e).filter(
    (e) => e % tickEvery === 0 || e === epochs,
  );

  if (points.length < 2) {
    return (
      <div
        style={{ aspectRatio: `${W} / ${H}` }}
        className="flex w-full items-center justify-center rounded-xl border border-dashed border-line font-mono text-[10px] uppercase tracking-[0.15em] text-faint"
      >
        waiting for the first batches…
      </div>
    );
  }

  const last = points[points.length - 1];
  const h = hover != null ? points[hover] : null;
  const hx = h ? x(h.epoch) : 0;
  const onMove = (ev: ReactPointerEvent<SVGSVGElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const e = ((((ev.clientX - rect.left) / rect.width) * W - L) / iw) * span;
    let best = 0,
      bestD = Infinity;
    points.forEach((p, i) => {
      const d = Math.abs(p.epoch - e);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    setHover(best);
  };

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full touch-none select-none"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={`Loss over ${epochs} epochs, latest ${last.loss.toFixed(4)}`}
      >
        {grid.map((f) => (
          <g key={f}>
            <line
              x1={L}
              x2={W - R}
              y1={T + (1 - f) * ih}
              y2={T + (1 - f) * ih}
              stroke="var(--color-line)"
              strokeWidth="1"
            />
            <text x={L - 8} y={T + (1 - f) * ih + 3} textAnchor="end" className="font-mono" fontSize="9" fill="var(--color-faint)">
              {(yLo + (yHi - yLo) * f).toFixed(2)}
            </text>
            <text x={W - R + 8} y={T + (1 - f) * ih + 3} className="font-mono" fontSize="9" fill="var(--color-faint)">
              {Math.round(f * 100)}%
            </text>
          </g>
        ))}
        {ticks.map((e) => (
          <text key={e} x={x(e)} y={H - 6} textAnchor="middle" className="font-mono" fontSize="9" fill="var(--color-faint)">
            {e}
          </text>
        ))}
        <polygon
          points={`${x(points[0].epoch)},${T + ih} ${points.map((p, i) => `${x(p.epoch)},${yLoss(smooth[i])}`).join(' ')} ${x(last.epoch)},${T + ih}`}
          fill="var(--color-ink)"
          fillOpacity="0.06"
        />
        <polyline
          points={points.map((p) => `${x(p.epoch)},${yLoss(p.loss)}`).join(' ')}
          fill="none"
          stroke="var(--color-faint)"
          strokeWidth="1"
          strokeOpacity="0.5"
          strokeLinejoin="round"
        />
        <polyline
          points={points.map((p) => `${x(p.epoch)},${yAcc(p.acc)}`).join(' ')}
          fill="none"
          stroke="var(--color-dim)"
          strokeWidth="1"
          strokeDasharray="3 3"
          strokeLinejoin="round"
        />
        <polyline
          points={points.map((p, i) => `${x(p.epoch)},${yLoss(smooth[i])}`).join(' ')}
          fill="none"
          stroke="var(--color-ink)"
          strokeWidth="1.75"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {h && (
          <g>
            <line x1={hx} x2={hx} y1={T} y2={T + ih} stroke="var(--color-faint)" strokeDasharray="2 3" />
            <circle cx={hx} cy={yLoss(smooth[hover!])} r="3.5" fill="var(--color-ink)" />
            <circle cx={hx} cy={yAcc(h.acc)} r="2.5" fill="var(--color-card)" stroke="var(--color-dim)" />
          </g>
        )}
        <circle cx={x(last.epoch)} cy={yLoss(smooth[smooth.length - 1])} r="2.5" fill="var(--color-ink)">
          <animate attributeName="r" values="2.5;5;2.5" dur="1.6s" repeatCount="indefinite" />
        </circle>
      </svg>
      {h && hover != null && (
        <div
          className={`pointer-events-none absolute top-0 whitespace-nowrap rounded-lg border border-line bg-card px-2.5 py-1.5 font-mono text-[10px] leading-relaxed text-fg shadow-sm ${
            hx / W > 0.6 ? '-translate-x-[105%]' : 'translate-x-[5%]'
          }`}
          style={{ left: `${(hx / W) * 100}%` }}
        >
          <div className="text-faint">epoch {h.epoch.toFixed(2)}</div>
          <div>loss {h.loss.toFixed(4)} <span className="text-faint">· trend {smooth[hover].toFixed(4)}</span></div>
          <div className="text-dim">acc {(h.acc * 100).toFixed(1)}%</div>
        </div>
      )}
    </div>
  );
}

export default function TrainingMonitor({
  mode,
  phase,
  embedPct,
  fetchPct,
  points,
  epochs,
  steps,
  backend,
}: {
  mode: 'head' | 'ft';
  phase: TrainStage;
  embedPct: number;
  fetchPct: number;
  points: TrainPoint[];
  epochs: number;
  /** planned optimizer steps (ft); null for the head trainer */
  steps: number | null;
  backend: string | null;
}) {
  const plan = PLAN[mode];
  const last = points[points.length - 1];
  const lastEpoch = last?.epoch ?? 0;
  const trainFrac = Math.min(1, lastEpoch / Math.max(1, epochs));
  const current = phase === 'done' ? plan.length : plan.findIndex((s) => s.id === phase);

  const stageFrac = (i: number): number => {
    if (i < current) return 1;
    if (i > current) return 0;
    switch (plan[i].id) {
      case 'embed':
        return embedPct;
      case 'fetch':
        return fetchPct;
      case 'tok':
        return 0.5;
      default:
        return phase === 'done' ? 1 : trainFrac;
    }
  };
  const overall = plan.reduce((s, st, i) => s + st.weight * stageFrac(i), 0);

  const eta = useMemo(() => {
    if (phase !== 'train' || points.length < 3) return null;
    const first = points[0];
    const rate = (lastEpoch - first.epoch) / Math.max(1, last.t - first.t);
    return rate > 0 ? (epochs - lastEpoch) / rate : null;
  }, [phase, points, lastEpoch, epochs, last]);

  const stepsPerSec =
    steps && points.length >= 2 ? (points.length - 1) / Math.max(1, (last.t - points[0].t) / 1000) : null;

  const stats: [string, string][] = [
    ['epoch', `${Math.min(lastEpoch, epochs).toFixed(2)} / ${epochs}`],
    ...(steps ? [['step', `${points.length} / ${steps}`] as [string, string]] : []),
    ...(stepsPerSec != null ? [['speed', `${stepsPerSec.toFixed(2)} steps/s`] as [string, string]] : []),
    [mode === 'ft' ? 'batch loss' : 'loss', last ? last.loss.toFixed(4) : '—'],
    ['eta', eta != null ? fmtDuration(eta) : '—'],
  ];

  return (
    <div className="rounded-2xl border border-line bg-paper p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {plan.map((s, i) => {
            const f = stageFrac(i);
            const active = i === current && phase !== 'done';
            return (
              <span
                key={s.id}
                className={`rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] ${
                  f >= 1
                    ? 'border-ink/60 bg-ink-dim text-ink'
                    : active
                      ? 'border-fg text-fg'
                      : 'border-line text-faint'
                }`}
              >
                {s.label}
                {active && ` ${Math.round(f * 100)}%`}
              </span>
            );
          })}
        </div>
        {backend && <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-faint">{backend}</span>}
      </div>

      <div className="mb-2 flex items-end justify-between gap-3">
        <span className="font-display text-4xl leading-none text-fg">
          {Math.round(overall * 100)}
          <span className="text-xl text-dim">%</span>
        </span>
        <span className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.15em] text-dim">
          {phase !== 'done' && <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ink" />}
          {PHASE_TEXT[phase]}
        </span>
      </div>
      <div className="mb-5 h-1.5 w-full overflow-hidden rounded-full bg-line">
        <div
          className="h-full rounded-full bg-ink transition-[width] duration-300 ease-out"
          style={{ width: `${Math.round(overall * 1000) / 10}%` }}
        />
      </div>

      <dl className={`mb-5 grid grid-cols-2 gap-x-4 gap-y-3 ${stats.length >= 5 ? 'sm:grid-cols-5' : 'sm:grid-cols-4'}`}>
        {stats.map(([k, v]) => (
          <div key={k}>
            <dt className="mb-1 font-mono text-[9.5px] uppercase tracking-[0.2em] text-faint">{k}</dt>
            <dd className="truncate font-mono text-[12px] text-fg">{v}</dd>
          </div>
        ))}
      </dl>

      <LossChart points={points} epochs={epochs} />

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-[10px] uppercase tracking-[0.14em] text-faint">
        <span className="flex items-center gap-2">
          <span className="inline-block h-px w-5 bg-ink" /> smoothed loss
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-px w-5 bg-faint opacity-60" /> {mode === 'ft' ? 'batch loss' : 'epoch loss'}
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block w-5 border-t border-dashed border-dim" /> accuracy
        </span>
      </div>
    </div>
  );
}
