import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { CheckCircle, Download, Play, SpinnerGap } from '@phosphor-icons/react';
import type { GpuReport } from '../lib/gpu';
import { fmtBytes } from '../lib/gpu';
import type { Dataset } from '../lib/dataset';
import type { ModelEntry, TrainMode } from '../data/models';
import { Tag } from '../components/ui';
import Step from './Step';
import type { TrainConfig } from './Studio';
import { embedTexts, EMBED_DIM, EMBED_MODEL } from '../lib/embed';
import type { TrainMsg, TrainRequest } from '../lib/train.worker';
import { writeSafetensors, downloadBytes } from '../lib/safetensors';
import { setTrainedHead, type TrainedHead } from '../lib/trainedHead';
import type { FtMsg, FtRequest } from '../lib/xformer/ft.worker';
import { setFtArtifact, type FtArtifact } from '../lib/xformer/ft';
import { MINILM_L6, paramShape } from '../lib/xformer/spec';

/** Rough resident-memory estimate in MB for a training config. */
function estimateMB(model: ModelEntry, cfg: TrainConfig): number {
  const p = model.paramsM;
  const weightsFP32 = p * 4;
  const weightsFP16 = p * 2;
  const act = Math.min(2048, (cfg.batch * cfg.seqLen * 8) / 1024);
  switch (cfg.mode) {
    case 'full':
      return weightsFP32 * 4 + act;
    case 'lora':
      return weightsFP16 + p * 0.02 * 16 + act;
    case 'head':
      return weightsFP16 + act * 0.5;
  }
}

const LR_OPTIONS = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2];

const MODE_LABELS: Record<TrainMode, string> = {
  head: 'Head — linear probe over frozen MiniLM embeddings',
  lora: 'LoRA — adapters on the MiniLM encoder (own WGSL forward/backward)',
  full: 'Full fine-tune — every MiniLM encoder parameter moves',
};

const MAX_ROWS = 512;

/** Live-loss sparkline over epoch history. */
function LossCurve({ history }: { history: { loss: number; acc: number }[] }) {
  if (history.length < 2) return null;
  const w = 240,
    h = 56,
    pad = 4;
  const max = Math.max(...history.map((p) => p.loss));
  const min = Math.min(...history.map((p) => p.loss));
  const span = Math.max(1e-9, max - min);
  const pts = history
    .map(
      (p, i) =>
        `${pad + (i / (history.length - 1)) * (w - 2 * pad)},${pad + (1 - (p.loss - min) / span) * (h - 2 * pad)}`,
    )
    .join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-14 w-full">
      <polyline
        points={pts}
        fill="none"
        stroke="var(--color-ink)"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function ConfigStep({
  report,
  model,
  dataset,
  config,
  onConfig,
  onTrained,
}: {
  report: GpuReport | null;
  model: ModelEntry | null;
  dataset: Dataset | null;
  config: TrainConfig;
  onConfig: (c: TrainConfig) => void;
  onTrained?: () => void;
}) {
  const est = useMemo(() => (model ? estimateMB(model, config) : 0), [model, config]);
  const budget = report?.weightsBudgetMB ?? 512;
  const over = est > budget;
  const ready = model && dataset && !over;

  // --- training state -----------------------------------------------------
  const [textCol, setTextCol] = useState<string | null>(null);
  const [labelCol, setLabelCol] = useState<string | null>(null);
  const [phase, setPhase] = useState<'idle' | 'embed' | 'fetch' | 'tok' | 'train' | 'done'>('idle');
  const [embedPct, setEmbedPct] = useState(0);
  const [fetchPct, setFetchPct] = useState(0);
  const [rank, setRank] = useState(8);
  const [history, setHistory] = useState<{ epoch: number; loss: number; acc: number }[]>([]);
  const [result, setResult] = useState<TrainedHead | null>(null);
  const [ftResult, setFtResult] = useState<FtArtifact | null>(null);
  const [liveBackend, setLiveBackend] = useState<string | null>(null);
  const [trainErr, setTrainErr] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => () => workerRef.current?.terminate(), []);

  const cols = dataset?.columns ?? [];
  const tCol = textCol ?? cols[0] ?? '';
  const lCol = labelCol ?? cols[cols.length - 1] ?? '';

  const trainRows = useMemo(() => {
    if (!dataset) return [];
    return dataset.rows
      .filter((r) => (r[tCol] ?? '').trim() && (r[lCol] ?? '').trim())
      .slice(0, MAX_ROWS);
  }, [dataset, tCol, lCol]);

  const labels = useMemo(
    () => [...new Set(trainRows.map((r) => r[lCol].trim()))],
    [trainRows, lCol],
  );

  const busy = phase !== 'idle' && phase !== 'done';
  const trainable = !busy && trainRows.length >= 8 && labels.length >= 2 && labels.length <= 64;

  const startTraining = async () => {
    if (!dataset) return;
    setTrainErr(null);
    setHistory([]);
    setResult(null);
    try {
      setPhase('embed');
      setEmbedPct(0);
      const x = await embedTexts(
        trainRows.map((r) => r[tCol].trim()),
        (d, t) => setEmbedPct(d / t),
      );
      const y = Uint32Array.from(trainRows.map((r) => labels.indexOf(r[lCol].trim())));

      setPhase('train');
      const worker = new Worker(new URL('../lib/train.worker.ts', import.meta.url), {
        type: 'module',
      });
      workerRef.current = worker;
      const done = await new Promise<{ w: Float32Array; b: Float32Array; backend: 'cpu' | 'wgsl'; loss: number; acc: number; valAcc: number | null; valN: number; trainN: number }>(
        (resolve, reject) => {
          worker.onmessage = (e: MessageEvent<TrainMsg>) => {
            const m = e.data;
            if (m.type === 'progress')
              setHistory((h) => [...h, { epoch: m.epoch, loss: m.loss, acc: m.acc }]);
            else if (m.type === 'done') resolve(m);
            else reject(new Error(m.message));
          };
          worker.onerror = (e) => reject(new Error(e.message));
          const req: TrainRequest = {
            type: 'train',
            init: {
              x,
              y,
              n: trainRows.length,
              d: EMBED_DIM,
              c: labels.length,
              lr: config.lr,
              batch: config.batch,
              epochs: config.epochs,
              seed: 1337,
              backend: report?.ok ? 'auto' : 'cpu',
              valFrac: 0.2,
            },
          };
          worker.postMessage(req, [x.buffer]);
        },
      );
      worker.terminate();
      workerRef.current = null;

      const head: TrainedHead = {
        w: done.w,
        b: done.b,
        labels,
        encoder: EMBED_MODEL,
        backend: done.backend,
        loss: done.loss,
        acc: done.acc,
        valAcc: done.valAcc ?? undefined,
        valN: done.valN,
        trainN: done.trainN,
        trainedAt: new Date().toISOString(),
        rows: trainRows.length,
      };
      setResult(head);
      setPhase('done');
    } catch (e) {
      setTrainErr(e instanceof Error ? e.message : String(e));
      setPhase('idle');
    }
  };

  /** LoRA / full fine-tune through our own encoder backend. */
  const startFinetune = async () => {
    if (!dataset) return;
    setTrainErr(null);
    setHistory([]);
    setFtResult(null);
    setLiveBackend(null);
    try {
      const worker = new Worker(
        new URL('../lib/xformer/ft.worker.ts', import.meta.url),
        { type: 'module' },
      );
      workerRef.current = worker;
      setPhase('fetch');
      const done = await new Promise<{
        params: Record<string, Float32Array>;
        headW: Float32Array;
        headB: Float32Array;
        backend: 'cpu' | 'wgsl';
        loss: number;
        acc: number;
        valAcc: number | null;
        valN: number;
        trainN: number;
      }>((resolve, reject) => {
        worker.onmessage = (e: MessageEvent<FtMsg>) => {
          const m = e.data;
          if (m.type === 'weights') setFetchPct(m.frac);
          else if (m.type === 'tokenizing') setPhase('tok');
          else if (m.type === 'epoch') {
            setPhase('train');
            setLiveBackend(m.backend);
            setHistory((h) => [...h, { epoch: m.epoch, loss: m.loss, acc: m.acc }]);
          } else if (m.type === 'done') resolve(m);
          else reject(new Error(m.message));
        };
        worker.onerror = (e) => reject(new Error(e.message));
        const req: FtRequest = {
          type: 'train',
          texts: trainRows.map((r) => r[tCol].trim()),
          labels: Uint32Array.from(trainRows.map((r) => labels.indexOf(r[lCol].trim()))),
          nClasses: labels.length,
          lr: config.lr,
          epochs: config.epochs,
          mode: config.mode === 'full' ? 'full' : 'lora',
          rank,
          seed: 1337,
          backend: report?.ok ? 'auto' : 'cpu',
          maxSeq: Math.min(config.seqLen, 128),
          valFrac: 0.2,
        };
        worker.postMessage(req);
      });
      worker.terminate();
      workerRef.current = null;

      const artifact: FtArtifact = {
        params: done.params,
        headW: done.headW,
        headB: done.headB,
        labels,
        mode: config.mode === 'full' ? 'full' : 'lora',
        rank,
        backend: done.backend,
        loss: done.loss,
        acc: done.acc,
        valAcc: done.valAcc ?? undefined,
        valN: done.valN,
        trainN: done.trainN,
        rows: trainRows.length,
        maxSeq: Math.min(config.seqLen, 128),
        trainedAt: new Date().toISOString(),
      };
      setFtArtifact(artifact);
      setPhase('done');
    } catch (e) {
      setTrainErr(e instanceof Error ? e.message : String(e));
      setPhase('idle');
    }
  };

  const exportHead = () => {
    if (!result) return;
    const bytes = writeSafetensors(
      {
        'head.weight': { shape: [EMBED_DIM, result.labels.length], data: result.w },
        'head.bias': { shape: [result.labels.length], data: result.b },
      },
      {
        encoder: result.encoder,
        labels: result.labels.join('\n'),
        loss: String(result.loss),
        acc: String(result.acc),
        ...(result.valAcc != null ? { valAcc: String(result.valAcc) } : {}),
        valN: String(result.valN ?? 0),
        trainN: String(result.trainN ?? result.rows),
        rows: String(result.rows),
        trainedAt: result.trainedAt,
        backend: result.backend,
        anvil: 'head-v1',
      },
    );
    downloadBytes('anvil-head.safetensors', bytes);
  };

  const exportFt = () => {
    if (!ftResult) return;
    const tensors: Record<string, { shape: number[]; data: Float32Array }> = {
      'head.weight': {
        shape: [EMBED_DIM, ftResult.labels.length],
        data: ftResult.headW,
      },
      'head.bias': { shape: [ftResult.labels.length], data: ftResult.headB },
    };
    for (const [k, v] of Object.entries(ftResult.params))
      tensors[k] = { shape: paramShape(k, MINILM_L6, ftResult.rank), data: v };
    downloadBytes(
      'anvil-ft.safetensors',
      writeSafetensors(tensors, {
        encoder: 'sentence-transformers/all-MiniLM-L6-v2',
        labels: ftResult.labels.join('\n'),
        mode: ftResult.mode,
        rank: String(ftResult.rank),
        maxSeq: String(ftResult.maxSeq),
        loss: String(ftResult.loss),
        acc: String(ftResult.acc),
        ...(ftResult.valAcc != null ? { valAcc: String(ftResult.valAcc) } : {}),
        valN: String(ftResult.valN ?? 0),
        trainN: String(ftResult.trainN ?? ftResult.rows),
        rows: String(ftResult.rows),
        trainedAt: ftResult.trainedAt,
        backend: ftResult.backend,
        anvil: 'ft-v1',
      }),
    );
  };

  const exportSpec = () => {
    if (!model || !dataset) return;
    const spec = {
      tool: 'anvil',
      specVersion: 1,
      createdAt: new Date().toISOString(),
      hardware: {
        tier: report?.tier ?? 0,
        vendor: report?.vendor ?? null,
        architecture: report?.architecture ?? null,
        fp16: report?.fp16 ?? false,
        weightsBudgetMB: budget,
      },
      model: { hf: model.hf, paramsM: model.paramsM, arch: model.arch },
      dataset: {
        name: dataset.name,
        sector: dataset.sector,
        columns: dataset.columns,
        rows: dataset.rows.length,
      },
      training: config,
    };
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `anvil-job-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Step
      index="05"
      title="Configure & train"
      lede="Two engines are live: a linear head over frozen MiniLM embeddings, or real encoder backprop (LoRA / full fine-tune) through our own WGSL/CPU backend. Metrics report a held-out 20% split — val accuracy is what the model does on rows it never trained on."
    >
      <div className="grid gap-3 lg:grid-cols-5">
        {/* left: method + hyperparams */}
        <div className="rounded-2xl border border-line bg-card p-5 lg:col-span-3">
          <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">Method</p>
          <div className="mb-6 flex flex-col gap-2">
            {(['head', 'lora', 'full'] as const).map((m) => (
              <label
                key={m}
                className={`flex cursor-pointer items-center gap-3 rounded-xl border px-4 py-3 transition-colors ${
                  config.mode === m ? 'border-ink bg-ink-dim/50' : 'border-line hover:border-faint'
                }`}
              >
                <input
                  type="radio"
                  name="mode"
                  className="accent-ink"
                  checked={config.mode === m}
                  onChange={() => onConfig({ ...config, mode: m })}
                />
                <span className="text-[13px] text-dim">{MODE_LABELS[m]}</span>
                <Tag tone="ink">{m === 'head' ? 'engine v1' : 'engine v2'}</Tag>
              </label>
            ))}
          </div>

          {config.mode === 'lora' && (
            <div className="mb-6 flex items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-widest text-faint">
                lora rank
              </span>
              <div className="flex gap-1.5">
                {[4, 8, 16, 32].map((r) => (
                  <button
                    key={r}
                    onClick={() => setRank(r)}
                    className={`rounded-full border px-3 py-1 font-mono text-[11px] transition-colors ${
                      rank === r ? 'border-ink bg-ink-dim text-ink' : 'border-line text-dim'
                    }`}
                  >
                    r={r}
                  </button>
                ))}
              </div>
            </div>
          )}

          <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
            Hyperparameters
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <label className="flex flex-col gap-1.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-faint">lr</span>
              <select
                value={config.lr}
                onChange={(e) => onConfig({ ...config, lr: Number(e.target.value) })}
                className="rounded-lg border border-line bg-paper px-2.5 py-2 font-mono text-[12px] text-fg"
              >
                {LR_OPTIONS.map((lr) => (
                  <option key={lr} value={lr}>
                    {lr.toExponential(0)}
                  </option>
                ))}
              </select>
            </label>
            {(
              [
                ['epochs', 'epochs', 1, 40, 1],
                ['batch', 'batch', 1, 64, 1],
                ['seqLen', 'seq len', 128, 4096, 128],
              ] as const
            ).map(([key, label, min, max, step]) => (
              <label key={key} className="flex flex-col gap-1.5">
                <span className="font-mono text-[10px] uppercase tracking-widest text-faint">
                  {label}
                </span>
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={config[key]}
                  onChange={(e) =>
                    onConfig({
                      ...config,
                      [key]: Math.max(min, Math.min(max, Number(e.target.value) || min)),
                    })
                  }
                  className="rounded-lg border border-line bg-paper px-2.5 py-2 font-mono text-[12px] text-fg"
                />
              </label>
            ))}
          </div>

          {/* column mapping for the head trainer */}
          {dataset && (
            <div className="mt-6 grid grid-cols-2 gap-3">
              {(
                [
                  ['text', tCol, setTextCol],
                  ['label', lCol, setLabelCol],
                ] as const
              ).map(([label, val, set]) => (
                <label key={label} className="flex flex-col gap-1.5">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-faint">
                    {label} column
                  </span>
                  <select
                    value={val}
                    onChange={(e) => set(e.target.value)}
                    className="rounded-lg border border-line bg-paper px-2.5 py-2 font-mono text-[12px] text-fg"
                  >
                    {cols.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* right: budget + actions */}
        <div className="flex flex-col rounded-2xl border border-line bg-card p-5 lg:col-span-2">
          {model ? (
            <>
              <p className="mb-4 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                Memory estimate · {config.mode}
              </p>
              <div className="mb-2 flex items-baseline justify-between font-mono text-[12px]">
                <span
                  className={
                    over ? 'text-fg underline decoration-dashed underline-offset-4' : 'text-fg'
                  }
                >
                  {fmtBytes(est)}
                </span>
                <span className="text-faint">/ {fmtBytes(budget)}</span>
              </div>
              <div className="mb-2 h-1.5 w-full rounded-full bg-line">
                <div
                  className={`h-full rounded-full transition-all ${over ? 'bar-over' : 'bg-ink'}`}
                  style={{ width: `${Math.min(100, (est / budget) * 100)}%` }}
                />
              </div>
              <p className="mb-6 font-mono text-[10px] leading-relaxed text-faint">
                {over
                  ? 'Over budget — drop to LoRA, shrink batch or seq len, or pick a smaller model.'
                  : 'Within budget. Estimate is crude: weights + optimizer states + activation reserve.'}
              </p>
            </>
          ) : (
            <p className="mb-6 font-mono text-[11px] leading-relaxed text-dim">
              No roster model picked — head training still works (it runs on the MiniLM
              embedder, not the roster model).
            </p>
          )}

          <div className="mt-auto flex flex-col gap-2.5">
            {phase === 'idle' || phase === 'done' ? (
              <button
                onClick={config.mode === 'head' ? startTraining : startFinetune}
                disabled={!trainable}
                className="flex items-center justify-center gap-2 rounded-full bg-fg px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Play size={13} weight="fill" />
                {config.mode === 'head' ? 'Train head' : config.mode === 'lora' ? `LoRA r=${rank}` : 'Full fine-tune'}{' '}
                — {trainRows.length} rows · {labels.length} classes · 80/20 split
              </button>
            ) : phase === 'embed' ? (
              <div className="rounded-full border border-line px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-dim">
                embedding… {Math.round(embedPct * 100)}%
              </div>
            ) : phase === 'fetch' ? (
              <div className="rounded-full border border-line px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-dim">
                fetching weights… {Math.round(fetchPct * 100)}%
              </div>
            ) : phase === 'tok' ? (
              <div className="rounded-full border border-line px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-dim">
                tokenizing…
              </div>
            ) : (
              <div className="flex items-center justify-center gap-2 rounded-full border border-ink/60 bg-ink-dim px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-ink">
                <SpinnerGap size={13} className="animate-spin" />
                training{liveBackend ? ` · ${liveBackend}` : ''} · epoch{' '}
                {Math.ceil(history[history.length - 1]?.epoch ?? 0)}/{config.epochs}
              </div>
            )}

            {history.length > 0 && (
              <div className="rounded-xl border border-line bg-paper p-3">
                <div className="mb-1 flex items-center justify-between font-mono text-[10px] text-faint">
                  <span>train loss {history[history.length - 1].loss.toFixed(3)}</span>
                  <span>train acc {(history[history.length - 1].acc * 100).toFixed(0)}%</span>
                </div>
                <LossCurve history={history} />
              </div>
            )}

            {trainErr && (
              <p className="rounded-xl border border-dashed border-fg/40 bg-paper px-4 py-3 font-mono text-[11px] text-fg">
                {trainErr}
              </p>
            )}

            {phase === 'done' && (result || ftResult) && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
                role="status"
                className="flex items-center gap-3 rounded-xl border border-ink/30 bg-ink px-4 py-3 text-card"
              >
                <CheckCircle size={18} weight="fill" className="shrink-0" />
                <div className="min-w-0">
                  <p className="font-mono text-[11px] uppercase tracking-[0.15em]">
                    Training complete
                  </p>
                  <p className="mt-0.5 truncate font-mono text-[10px] tracking-wide text-card/60">
                    {(result ?? ftResult)!.backend} ·{' '}
                    {result ? 'head' : ftResult!.mode} ·{' '}
                    {(result ?? ftResult)!.valAcc != null
                      ? `val ${((result ?? ftResult)!.valAcc! * 100).toFixed(0)}% on held-out rows`
                      : 'no val split'}{' '}
                    — artifact in session, export or send to Run
                  </p>
                </div>
              </motion.div>
            )}

            {result && (
              <div className="flex flex-col gap-2">
                <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-widest">
                  <Tag tone="ink">trained · {result.backend}</Tag>
                  <Tag>loss {result.loss.toFixed(3)}</Tag>
                  <Tag>train {(result.acc * 100).toFixed(0)}% · n={result.trainN ?? result.rows}</Tag>
                  {result.valAcc != null ? (
                    <Tag tone="ink">
                      val {(result.valAcc * 100).toFixed(0)}% · n={result.valN}
                    </Tag>
                  ) : (
                    <Tag tone="faint">no val split</Tag>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={exportHead}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full border border-fg px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-fg transition-colors hover:bg-fg hover:text-card"
                  >
                    <Download size={13} />
                    .safetensors
                  </button>
                  <button
                    onClick={() => {
                      setTrainedHead(result);
                      onTrained?.();
                    }}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-ink px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-white transition-opacity hover:opacity-85"
                  >
                    Use in Run →
                  </button>
                </div>
              </div>
            )}

            {ftResult && (
              <div className="flex flex-col gap-2">
                <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-widest">
                  <Tag tone="ink">
                    {ftResult.mode} · {ftResult.backend}
                  </Tag>
                  <Tag>loss {ftResult.loss.toFixed(3)}</Tag>
                  <Tag>train {(ftResult.acc * 100).toFixed(0)}% · n={ftResult.trainN ?? ftResult.rows}</Tag>
                  {ftResult.valAcc != null ? (
                    <Tag tone="ink">
                      val {(ftResult.valAcc * 100).toFixed(0)}% · n={ftResult.valN}
                    </Tag>
                  ) : (
                    <Tag tone="faint">no val split</Tag>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={exportFt}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full border border-fg px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-fg transition-colors hover:bg-fg hover:text-card"
                  >
                    <Download size={13} />
                    .safetensors
                  </button>
                  <button
                    onClick={() => {
                      setFtArtifact(ftResult);
                      onTrained?.();
                    }}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-ink px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-white transition-opacity hover:opacity-85"
                  >
                    Use in Run →
                  </button>
                </div>
              </div>
            )}

            <button
              onClick={exportSpec}
              disabled={!model || !dataset}
              className="flex items-center justify-center gap-2 rounded-full border border-line px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-dim transition-colors hover:border-fg hover:text-fg disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Download size={14} />
              Export job spec
            </button>
            <p className="flex items-center justify-between pt-1 font-mono text-[10px] text-faint">
              <span>{model ? model.name : 'no roster pick'}</span>
              <span>
                {dataset ? `${dataset.rows.length} rows` : 'no dataset'}{' '}
                {ready ? <Tag tone="ink">ready</Tag> : <Tag tone="line">incomplete</Tag>}
              </span>
            </p>
          </div>
        </div>
      </div>
    </Step>
  );
}
