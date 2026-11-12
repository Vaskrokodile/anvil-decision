import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { ArrowRight, CheckCircle, Download, Play } from '@phosphor-icons/react';
import type { GpuReport } from '../lib/gpu';
import { fmtBytes } from '../lib/gpu';
import { analyzeDataset, type Dataset } from '../lib/dataset';
import { SECTORS } from '../data/sectors';
import { type ModelEntry, type TrainMode } from '../data/models';
import { Tag } from '../components/ui';
import Step from './Step';
import type { TrainConfig } from './Studio';
import { embedTexts, EMBED_DIM, EMBED_MODEL } from '../lib/embed';
import type { TrainMsg, TrainRequest } from '../lib/train.worker';
import { writeSafetensors, downloadBytes } from '../lib/safetensors';
import { setTrainedHead, type TrainedHead } from '../lib/trainedHead';
import type { FtMsg, FtRequest } from '../lib/xformer/ft.worker';
import { TASK_MISSING } from '../lib/xformer/ft.worker';
import { setFtArtifact, type FtArtifact } from '../lib/xformer/ft';
import { MINILM_L6, DEBERTA_V3_SMALL, paramShape, poolForArch, type XformerSpec } from '../lib/xformer/spec';
import { setWorkspaceItem } from '../lib/workspace';
import { stratifiedSplit } from '../lib/split';
import type { EvalSet } from '../lib/evaluate';
import TrainingMonitor, { type TrainPoint } from './TrainingMonitor';
import { bundleEntries, headCard, ftCard, headBundleExtras, ftBundleExtras } from '../lib/bundle';

function estimateMB(
  cfg: TrainConfig,
  rank: number,
  classes: number,
  spec: XformerSpec = MINILM_L6,
): number {
  const { layers, dModel, dFfn, heads } = spec;
  const fullParams = layers * (4 * dModel * dModel + 2 * dModel * dFfn + 9 * dModel + dFfn) + 2 * dModel;
  const headParams = dModel * classes + classes;
  const encoderAdapters = layers * rank * (10 * dModel + 2 * dFfn);
  const tunableEncoderParams = cfg.mode === 'full' ? fullParams : cfg.mode === 'lora' ? encoderAdapters : 0;
  const trainableParams = tunableEncoderParams + headParams;
  const seq = Math.min(cfg.seqLen, 128);
  const activationFloats =
    cfg.mode === 'head'
      ? 0
      : layers * (9 * dModel * seq + 2 * dFfn * seq + heads * seq * seq);
  const parameterStateBytes = cfg.mode === 'full' ? 12 : 16;
  const optimizerMB = (trainableParams * parameterStateBytes) / 1_000_000;
  const fullHeadWeightsMB = cfg.mode === 'full' ? (headParams * 4) / 1_000_000 : 0;
  const activationMB = (activationFloats * 4 * 1.5) / (1024 * 1024);
  return 91 + optimizerMB + fullHeadWeightsMB + activationMB + (cfg.mode === 'head' ? 1 : 0);
}

const LR_OPTIONS = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2];

const MODE_LABELS: Record<TrainMode, string> = {
  head: 'Head — linear probe over frozen MiniLM embeddings',
  lora: 'LoRA — adapters on the encoder (own forward/backward, WGSL or CPU)',
  full: 'Full fine-tune — transformer weights move; token embeddings stay frozen',
};

const MAX_ROWS = 512;

/** Held-out rows per task — the saved set the Compare page scores both engines on. */
function buildEvalSet(
  kind: 'ft' | 'head',
  texts: string[],
  tasks: { name: string; labels: string[]; y: Uint32Array }[],
  valIdx: number[][],
): EvalSet {
  const held = [...new Set(valIdx.flat())].sort((a, b) => a - b);
  return {
    kind,
    createdAt: new Date().toISOString(),
    tasks: tasks.map(({ name, labels }) => ({ name, labels })),
    rows: held.map((i) => ({
      text: texts[i],
      gold: tasks.map((t, k) => (valIdx[k].includes(i) ? t.y[i] : -1)),
    })),
  };
}

export default function ConfigStep({
  report,
  model,
  dataset,
  config,
  onConfig,
  onStorageError,
  onTrained,
}: {
  report: GpuReport | null;
  model: ModelEntry | null;
  dataset: Dataset | null;
  config: TrainConfig;
  onConfig: (c: TrainConfig) => void;
  onStorageError?: (message: string) => void;
  onTrained?: () => void;
}) {
  const budget = report?.ok ? report.weightsBudgetMB : null;

  // --- training state -----------------------------------------------------
  const [textCol, setTextCol] = useState<string | null>(null);
  const [labelCol, setLabelCol] = useState<string | null>(null);
  const [extraTaskCols, setExtraTaskCols] = useState<string[]>([]);
  const [arch, setArch] = useState<'minilm' | 'deberta'>('minilm');
  const [phase, setPhase] = useState<'idle' | 'embed' | 'fetch' | 'tok' | 'train' | 'done'>('idle');
  const [embedPct, setEmbedPct] = useState(0);
  const [fetchPct, setFetchPct] = useState(0);
  const [rank, setRank] = useState(8);
  const [history, setHistory] = useState<TrainPoint[]>([]);
  const [totalSteps, setTotalSteps] = useState<number | null>(null);
  const [result, setResult] = useState<TrainedHead | null>(null);
  const [ftResult, setFtResult] = useState<FtArtifact | null>(null);
  const [liveBackend, setLiveBackend] = useState<string | null>(null);
  const [parity, setParity] = useState<{ backend: 'cpu' | 'wgsl'; cos: number } | null>(null);
  const [trainErr, setTrainErr] = useState<string | null>(null);
  const reportStorage = onStorageError ?? setTrainErr;
  const workerRef = useRef<Worker | null>(null);
  const runIdRef = useRef(0);
  const cancelRef = useRef<(() => void) | null>(null);

  const cancelTraining = () => {
    runIdRef.current++;
    workerRef.current?.terminate();
    workerRef.current = null;
    cancelRef.current?.();
    cancelRef.current = null;
    setPhase('idle');
    setTrainErr(null);
  };

  useEffect(
    () => () => {
      runIdRef.current++;
      workerRef.current?.terminate();
      cancelRef.current?.();
    },
    [],
  );

  const cols = dataset?.columns ?? [];
  const health = useMemo(() => (dataset ? analyzeDataset(dataset) : null), [dataset]);
  const schema = SECTORS.find((sector) => sector.id === dataset?.sector)?.schema ?? [];
  const schemaText = schema.find((field) => field.role === 'input')?.field;
  const schemaLabel = schema.find((field) => field.role === 'label')?.field;
  const tCol = textCol && cols.includes(textCol) ? textCol : health?.textCol ?? schemaText ?? cols[0] ?? '';
  const lCol =
    labelCol && cols.includes(labelCol)
      ? labelCol
      : health?.labelGuess ?? schemaLabel ?? cols[cols.length - 1] ?? '';

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

  /** Candidate extra task columns: ≠ text col, 2..64 uniques, ≥8 labeled rows. */
  const taskCandidates = useMemo(() => {
    if (!dataset) return [];
    return dataset.columns
      .filter((c) => c !== tCol && c !== lCol)
      .map((col) => {
        const vals = trainRows.map((r) => (r[col] ?? '').trim()).filter(Boolean);
        return { col, uniques: new Set(vals).size, n: vals.length };
      })
      .filter((x) => x.uniques >= 2 && x.uniques <= 64 && x.n >= 8);
  }, [dataset, tCol, lCol, trainRows]);

  const taskCols = useMemo(
    () => [lCol, ...extraTaskCols.filter((c) => taskCandidates.some((x) => x.col === c))],
    [lCol, extraTaskCols, taskCandidates],
  );

  /** Multi-task request payload — one {name, labels, y} per label column. */
  const reqTasks = useMemo(() => {
    if (taskCols.length < 2) return undefined;
    return taskCols.map((col) => {
      const taskLabels = [
        ...new Set(trainRows.map((r) => (r[col] ?? '').trim()).filter(Boolean)),
      ];
      const y = Uint32Array.from(
        trainRows.map((r) => {
          const v = (r[col] ?? '').trim();
          return v ? taskLabels.indexOf(v) : TASK_MISSING;
        }),
      );
      return { name: col, labels: taskLabels, y };
    });
  }, [taskCols, trainRows]);

  const busy = phase !== 'idle' && phase !== 'done';
  const trainable = !busy && tCol !== lCol && trainRows.length >= 8 && labels.length >= 2 && labels.length <= 64;
  const activeSpec = arch === 'deberta' && config.mode !== 'head' ? DEBERTA_V3_SMALL : MINILM_L6;
  const est = useMemo(
    () => estimateMB(config, rank, labels.length, activeSpec),
    [config, rank, labels.length, activeSpec],
  );
  const over = report?.ok === true && budget !== null && est > budget;
  const ready = dataset !== null && trainable;

  const startTraining = async () => {
    if (!dataset) return;
    const runId = ++runIdRef.current;
    setTrainErr(null);
    setHistory([]);
    setTotalSteps(null);
    setResult(null);
    setFtResult(null);
    try {
      setPhase('embed');
      setEmbedPct(0);
      const x = await embedTexts(
        trainRows.map((r) => r[tCol].trim()),
        (d, t) => setEmbedPct(d / t),
      );
      if (runId !== runIdRef.current) return;
      const y = Uint32Array.from(trainRows.map((r) => labels.indexOf(r[lCol].trim())));

      setPhase('train');
      const worker = new Worker(new URL('../lib/train.worker.ts', import.meta.url), {
        type: 'module',
      });
      workerRef.current = worker;
      const done = await new Promise<{ w: Float32Array; b: Float32Array; backend: 'cpu' | 'wgsl'; loss: number; acc: number; valAcc: number | null; valN: number; trainN: number }>(
        (resolve, reject) => {
          cancelRef.current = () => reject(new Error('Training cancelled.'));
          worker.onmessage = (e: MessageEvent<TrainMsg>) => {
            const m = e.data;
            if (m.type === 'progress') {
              setLiveBackend(m.backend);
              setHistory((h) => [...h, { epoch: m.epoch, loss: m.loss, acc: m.acc, t: performance.now() }]);
            }
            else if (m.type === 'done') {
              cancelRef.current = null;
              resolve(m);
            } else {
              cancelRef.current = null;
              reject(new Error(m.message));
            }
          };
          worker.onerror = (e) => {
            cancelRef.current = null;
            reject(new Error(e.message));
          };
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
      if (runId !== runIdRef.current) return;
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
      // register + persist immediately — the artifact survives step switches/reloads
      setTrainedHead(head);
      void setWorkspaceItem('trainedHead', head).catch(() =>
        reportStorage('Trained for this tab, but local persistence failed. Export before leaving.'),
      );
      const yHead = Uint32Array.from(trainRows.map((r) => labels.indexOf(r[lCol].trim())));
      void setWorkspaceItem(
        'evalSet',
        buildEvalSet('head', trainRows.map((r) => r[tCol].trim()), [{ name: lCol, labels, y: yHead }], [
          stratifiedSplit(yHead, 0.2, 1337).val,
        ]),
      ).catch(() => reportStorage('Trained, but the held-out rows could not be saved; compare will be unavailable.'));
      setPhase('done');
    } catch (e) {
      if (runId !== runIdRef.current) return;
      workerRef.current?.terminate();
      workerRef.current = null;
      cancelRef.current = null;
      setTrainErr(e instanceof Error ? e.message : String(e));
      setPhase('idle');
    }
  };

  /** LoRA / full fine-tune through our own encoder backend. */
  const startFinetune = async () => {
    if (!dataset) return;
    const runId = ++runIdRef.current;
    setTrainErr(null);
    setHistory([]);
    setTotalSteps(null);
    setResult(null);
    setFtResult(null);
    setLiveBackend(null);
    setParity(null);
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
        tasks?: {
          name: string;
          labels: string[];
          headW: Float32Array;
          headB: Float32Array;
          valAcc: number | null;
          valN: number;
          valIdx: number[];
        }[];
      }>((resolve, reject) => {
        cancelRef.current = () => reject(new Error('Training cancelled.'));
        worker.onmessage = (e: MessageEvent<FtMsg>) => {
          const m = e.data;
          if (m.type === 'weights') setFetchPct(m.frac);
          else if (m.type === 'tokenizing') setPhase('tok');
          else if (m.type === 'parity') setParity({ backend: m.backend, cos: m.cos });
          else if (m.type === 'epoch') {
            setPhase('train');
            setLiveBackend(m.backend);
            setTotalSteps(m.steps);
            setHistory((h) => [...h, { epoch: m.epoch, loss: m.loss, acc: m.acc, t: performance.now() }]);
          } else if (m.type === 'done') {
            cancelRef.current = null;
            resolve(m);
          } else {
            cancelRef.current = null;
            reject(new Error(m.message));
          }
        };
        worker.onerror = (e) => {
          cancelRef.current = null;
          reject(new Error(e.message));
        };
        const req: FtRequest = {
          type: 'train',
          texts: trainRows.map((r) => r[tCol].trim()),
          labels: Uint32Array.from(trainRows.map((r) => labels.indexOf(r[lCol].trim()))),
          nClasses: labels.length,
          tasks: reqTasks,
          arch,
          lr: config.lr,
          epochs: config.epochs,
          mode: config.mode === 'full' ? 'full' : 'lora',
          rank,
          seed: 1337,
          backend: report?.ok ? 'auto' : 'cpu',
          batch: config.batch,
          maxSeq: Math.min(config.seqLen, 128),
          valFrac: 0.2,
        };
        worker.postMessage(req);
      });
      if (runId !== runIdRef.current) return;
      worker.terminate();
      workerRef.current = null;

      const artifact: FtArtifact = {
        params: done.params,
        headW: done.headW,
        headB: done.headB,
        labels,
        tasks: done.tasks?.map((t) => ({
          name: t.name,
          labels: t.labels,
          headW: t.headW,
          headB: t.headB,
          valAcc: t.valAcc ?? undefined,
          valN: t.valN,
        })),
        arch,
        pool: poolForArch(arch),
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
      setFtResult(artifact);
      setFtArtifact(artifact);
      void setWorkspaceItem('ftArtifact', artifact).catch(() =>
        reportStorage('Trained for this tab, but local persistence failed. Export before leaving.'),
      );
      const yPrimary = Uint32Array.from(trainRows.map((r) => labels.indexOf(r[lCol].trim())));
      void setWorkspaceItem(
        'evalSet',
        buildEvalSet(
          'ft',
          trainRows.map((r) => r[tCol].trim()),
          reqTasks ?? [{ name: lCol, labels, y: yPrimary }],
          done.tasks ? done.tasks.map((t) => t.valIdx) : [stratifiedSplit(yPrimary, 0.2, 1337).val],
        ),
      ).catch(() => reportStorage('Trained, but the held-out rows could not be saved; compare will be unavailable.'));
      setPhase('done');
    } catch (e) {
      if (runId !== runIdRef.current) return;
      workerRef.current?.terminate();
      workerRef.current = null;
      cancelRef.current = null;
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
    downloadBytes(
      'anvil-head-bundle.zip',
      bundleEntries(
        'anvil-head.safetensors',
        bytes,
        headCard(result),
        headBundleExtras(result),
      ),
    );
  };

  const exportFt = () => {
    if (!ftResult) return;
    const tensors: Record<string, { shape: number[]; data: Float32Array }> = {};
    const multi = ftResult.tasks && ftResult.tasks.length > 1;
    if (multi) {
      ftResult.tasks!.forEach((t, i) => {
        tensors[`head.${i}.weight`] = {
          shape: [EMBED_DIM, t.labels.length],
          data: t.headW,
        };
        tensors[`head.${i}.bias`] = { shape: [t.labels.length], data: t.headB };
      });
    } else {
      tensors['head.weight'] = {
        shape: [EMBED_DIM, ftResult.labels.length],
        data: ftResult.headW,
      };
      tensors['head.bias'] = { shape: [ftResult.labels.length], data: ftResult.headB };
    }
    const spec = ftResult.arch === 'deberta' ? DEBERTA_V3_SMALL : MINILM_L6;
    for (const [k, v] of Object.entries(ftResult.params))
      tensors[k] = { shape: paramShape(k, spec, ftResult.rank), data: v };
    const stBytes = writeSafetensors(tensors, {
      encoder:
        ftResult.arch === 'deberta'
          ? 'cross-encoder/nli-deberta-v3-small'
          : 'sentence-transformers/all-MiniLM-L6-v2',
      arch: ftResult.arch ?? 'minilm',
      pool: ftResult.pool ?? 'cls',
      labels: ftResult.labels.join('\n'),
      ...(multi
        ? {
            tasks: JSON.stringify(
              ftResult.tasks!.map((t) => ({ name: t.name, labels: t.labels })),
            ),
          }
        : {}),
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
    });
    downloadBytes(
      'anvil-ft-bundle.zip',
      bundleEntries('anvil-ft.safetensors', stBytes, ftCard(ftResult), ftBundleExtras()),
    );
  };

  return (
    <Step
      index="05"
      title="Configure & train"
      lede="Train a MiniLM classifier: a linear head over frozen embeddings, or LoRA/full fine-tune of the encoder. Check extra label columns for one head per task over a shared encoder. WGSL is used when available with CPU fallback; validation is held out. Roster checkpoints are not fine-tuned here."
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

          {config.mode !== 'head' && (
            <div className="mb-6">
              <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                Substrate
              </p>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {(
                  [
                    ['minilm', 'MiniLM-L6', 'WGSL + CPU · ~91 MB · fast'],
                    ['deberta', 'DeBERTa-v3-small', 'CPU only · ~570 MB · disentangled attn'],
                  ] as const
                ).map(([key, name, note]) => (
                  <button
                    key={key}
                    onClick={() => setArch(key)}
                    disabled={busy}
                    className={`flex flex-col items-start gap-0.5 rounded-xl border px-3 py-2.5 text-left transition-colors disabled:opacity-50 ${
                      arch === key ? 'border-ink bg-ink-dim' : 'border-line hover:border-faint'
                    }`}
                  >
                    <span className={`font-mono text-[11px] ${arch === key ? 'text-ink' : 'text-fg'}`}>
                      {name}
                    </span>
                    <span className="font-mono text-[9.5px] text-faint">{note}</span>
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
                ['seqLen', 'seq len', 32, 128, 32],
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

          {/* extra task columns — multi-task heads over the shared encoder */}
          {dataset && config.mode !== 'head' && taskCandidates.length > 0 && (
            <div className="mt-4">
              <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-faint">
                extra task columns · multi-task
              </p>
              <div className="flex flex-wrap gap-1.5">
                {taskCandidates.map((x) => {
                  const on = extraTaskCols.includes(x.col);
                  return (
                    <button
                      key={x.col}
                      onClick={() =>
                        setExtraTaskCols((cur) =>
                          on ? cur.filter((c) => c !== x.col) : [...cur, x.col],
                        )
                      }
                      className={`rounded-full border px-3 py-1 font-mono text-[11px] transition-colors ${
                        on ? 'border-ink bg-ink-dim text-ink' : 'border-line text-dim hover:border-faint'
                      }`}
                    >
                      {x.col} · {x.uniques}
                    </button>
                  );
                })}
              </div>
              <p className="mt-1.5 font-mono text-[10px] text-faint">
                {reqTasks
                  ? `${reqTasks.length} heads over one shared encoder — unlabeled rows are skipped per task`
                  : 'check extra label columns — each becomes its own head on the shared encoder'}
              </p>
            </div>
          )}
          {dataset && config.mode === 'head' && extraTaskCols.length > 0 && (
            <p className="mt-4 font-mono text-[10px] text-faint">
              extra task columns apply to LoRA / full only — head mode trains the label column.
            </p>
          )}
        </div>

        {/* right: budget + actions */}
        <div className="flex flex-col rounded-2xl border border-line bg-card p-5 lg:col-span-2">
          <p className="mb-4 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
            {activeSpec === DEBERTA_V3_SMALL ? 'DeBERTa' : 'MiniLM'} working-set estimate ·{' '}
            {config.mode}
          </p>
          <p className="mb-4 font-mono text-[10px] leading-relaxed text-dim">
            This trainer updates {activeSpec === DEBERTA_V3_SMALL ? 'DeBERTa-v3-small' : 'MiniLM-L6'}
            {reqTasks ? ` and ${reqTasks.length} task heads` : ' and a single-label classifier'}.
            The roster choice{model ? ` (${model.name})` : ''} is not fine-tuned by this run.
          </p>
          <div className="mb-2 flex items-baseline justify-between font-mono text-[12px]">
            <span className={over ? 'text-fg underline decoration-dashed underline-offset-4' : 'text-fg'}>
              {fmtBytes(est)}
            </span>
            <span className="text-faint">/ {budget == null ? 'no GPU budget' : fmtBytes(budget)}</span>
          </div>
          <div className="mb-2 h-1.5 w-full rounded-full bg-line">
            <div
              className={`h-full rounded-full transition-all ${over ? 'bar-over' : 'bg-ink'}`}
              style={{ width: `${budget ? Math.min(100, (est / budget) * 100) : 0}%` }}
            />
          </div>
          <p className="mb-6 font-mono text-[10px] leading-relaxed text-faint">
            {over
              ? 'Estimate exceeds the adapter budget. The worker may fall back to CPU; actual memory varies by browser.'
              : 'Rough estimate of base weights, optimizer state and encoder activations; not a hard limit.'}
          </p>

          <div className="mt-auto flex flex-col gap-2.5">
            {phase === 'idle' || phase === 'done' ? (
              <button
                onClick={config.mode === 'head' ? startTraining : startFinetune}
                disabled={!trainable}
                className="flex items-center justify-center gap-2 rounded-full bg-fg px-4 py-3 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Play size={13} weight="fill" />
                {config.mode === 'head' ? 'Train head' : config.mode === 'lora' ? `LoRA r=${rank}` : 'Full fine-tune'}{' '}
                — {trainRows.length} rows ·{' '}
                {reqTasks ? `${reqTasks.length} tasks` : `${labels.length} classes`} · 80/20 split
              </button>
            ) : null}
            {busy && (
              <button
                onClick={cancelTraining}
                className="self-end rounded-full border border-line px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest text-dim hover:border-fg hover:text-fg"
              >
                Cancel training
              </button>
            )}

            {parity && (
              <p className="font-mono text-[10px] leading-relaxed text-faint">
                {parity.backend === 'wgsl'
                  ? `wgsl parity verified on-device (cos ${parity.cos.toFixed(4)} vs CPU)`
                  : `wgsl failed device parity (cos ${parity.cos.toFixed(4)}) — running on CPU`}
              </p>
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
                className="flex flex-wrap items-center gap-3 rounded-xl border border-ink/30 bg-ink px-4 py-3 text-card"
              >
                <CheckCircle size={18} weight="fill" className="shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="font-mono text-[11px] uppercase tracking-[0.15em]">
                    Training complete
                  </p>
                  <p className="mt-0.5 font-mono text-[10px] tracking-wide text-card/60">
                    {(result ?? ftResult)!.backend} ·{' '}
                    {result ? 'head' : ftResult!.mode} ·{' '}
                    {(result ?? ftResult)!.valAcc != null
                      ? `val ${((result ?? ftResult)!.valAcc! * 100).toFixed(0)}% on held-out rows`
                      : 'no val split'}{' '}
                    — saved locally
                  </p>
                </div>
                <a
                  href="#/compare"
                  className="flex shrink-0 items-center gap-2 rounded-full bg-card px-4 py-2 font-mono text-[10px] uppercase tracking-[0.15em] text-ink transition-opacity hover:opacity-85"
                >
                  Compare vs base
                  <ArrowRight size={12} />
                </a>
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
                    bundle .zip
                  </button>
                  <button
                    onClick={() => {
                      setTrainedHead(result);
                      onTrained?.();
                    }}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-ink px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85"
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
                    bundle .zip
                  </button>
                  <button
                    onClick={() => {
                      setFtArtifact(ftResult);
                      onTrained?.();
                    }}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-ink px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85"
                  >
                    Use in Run →
                  </button>
                </div>
              </div>
            )}


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

      {phase !== 'idle' && (busy || history.length > 0) && (
        <div className="mt-3">
          <TrainingMonitor
            mode={config.mode === 'head' ? 'head' : 'ft'}
            phase={phase}
            embedPct={embedPct}
            fetchPct={fetchPct}
            points={history}
            epochs={config.epochs}
            steps={totalSteps}
            backend={liveBackend}
          />
        </div>
      )}
    </Step>
  );
}
