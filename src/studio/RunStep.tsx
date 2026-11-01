import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Play, SpinnerGap } from '@phosphor-icons/react';
import Step from './Step';
import { Tag } from '../components/ui';
import type { ModelEntry } from '../data/models';
import type { Dataset } from '../lib/dataset';
import {
  RUNNABLE,
  isJevRunnable,
  jevInfo,
  loadJev,
  isDecideOnnx,
  type Answer,
  type OpenJevInfo,
  type OpenJevRuntime,
  type Question,
} from '../lib/jev';
import { getTrainedHead, setTrainedHead, headFromSafetensors, inferHead, type TrainedHead } from '../lib/trainedHead';
import { getFtArtifact, setFtArtifact, ftFromSafetensors, inferFtTasks, type FtArtifact, type FtTaskResult } from '../lib/xformer/ft';
import { readSafetensors } from '../lib/safetensors';
import { getWorkspaceItem, setWorkspaceItem } from '../lib/workspace';
import { loadDecide, type DecideSession } from '../lib/gliner2';
import { loadDecoder, type DecoderSession } from '../lib/decoder';
import { releaseSessions } from '../lib/sessions';

/** Question packs per sector — the "typed decisions" contract in miniature. */
const PACKS: Record<string, { label: string; questions: Question[] }> = {
  decision: {
    label: 'Ops triage',
    questions: [
      {
        type: 'choice',
        instructions: 'What is the user asking for?',
        options: ['refund request', 'cancel subscription', 'order status', 'technical issue', 'other'],
      },
      {
        type: 'score',
        instructions: 'How urgent is this?',
        options: ['low', 'medium', 'high', 'critical'],
      },
      { type: 'noul', instructions: 'The user is frustrated or angry.' },
    ],
  },
  labelling: {
    label: 'Entity tagging',
    questions: [
      {
        type: 'choice',
        instructions: 'What is the primary entity?',
        options: ['person', 'organization', 'location', 'product', 'event'],
      },
      { type: 'noul', instructions: 'The text contains personally identifiable information.' },
    ],
  },
  trust: {
    label: 'Safety check',
    questions: [
      { type: 'noul', instructions: 'The text contains personally identifiable information.' },
      { type: 'score', instructions: 'How safe is this content?', options: ['safe', 'borderline', 'unsafe'] },
    ],
  },
  scientific: {
    label: 'Paper triage',
    questions: [
      {
        type: 'choice',
        instructions: 'What kind of claim is this?',
        options: ['hypothesis', 'methodology', 'result', 'review', 'dataset'],
      },
      { type: 'noul', instructions: 'The text reports an empirical finding.' },
    ],
  },
  general: {
    label: 'Topic sort',
    questions: [
      {
        type: 'choice',
        instructions: 'What is the topic?',
        options: ['business', 'technology', 'science', 'sports', 'politics', 'other'],
      },
      { type: 'noul', instructions: 'The tone is positive.' },
    ],
  },
};

const DEFAULT_STATE =
  'I was charged twice for the same order and nobody answers my emails. I want my money back now.';

function Bar({ p }: { p: number }) {
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-line">
      <div className="h-full rounded-full bg-ink" style={{ width: `${Math.round(p * 100)}%` }} />
    </div>
  );
}

function classifierAnswer(labels: string[], probabilities: Record<string, number>): Answer {
  const choice = labels.reduce((best, label) =>
    (probabilities[label] ?? 0) > (probabilities[best] ?? 0) ? label : best,
  labels[0] ?? '');
  return {
    type: 'choice',
    choice,
    confidence: probabilities[choice] ?? 0,
    probabilities,
  };
}

function AnswerCard({ q, a }: { q: Question; a: Answer }) {
  return (
    <div className="rounded-2xl border border-line bg-card p-5">
      <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
        {q.type} · {q.instructions}
      </p>
      {a.type === 'choice' && (
        <div className="flex flex-col gap-2.5">
          {Object.entries(a.probabilities)
            .sort((x, y) => y[1] - x[1])
            .map(([opt, p]) => (
              <div key={opt} className="grid grid-cols-[10rem_1fr_3.5rem] items-center gap-3">
                <span
                  className={`truncate text-[13px] ${opt === a.choice ? 'font-medium text-fg' : 'text-dim'}`}
                >
                  {opt}
                </span>
                <Bar p={p} />
                <span className="text-right font-mono text-[11px] text-dim">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          <p className="mt-1 font-mono text-[11px] text-faint">
            → <span className="text-ink">{a.choice}</span> @ {(a.confidence * 100).toFixed(1)}%
          </p>
        </div>
      )}
      {a.type === 'score' && (
        <div>
          <div className="mb-3 flex items-baseline gap-3">
            <span className="font-display text-3xl text-fg">{a.level}</span>
            <span className="font-mono text-[11px] text-faint">
              score {a.score.toFixed(2)} · {(a.confidence * 100).toFixed(0)}% conf
            </span>
          </div>
          <div className="flex flex-col gap-2">
            {Object.entries(a.probabilities).map(([lvl, p]) => (
              <div key={lvl} className="grid grid-cols-[10rem_1fr_3.5rem] items-center gap-3">
                <span className={`truncate text-[13px] ${lvl === a.level ? 'text-fg' : 'text-dim'}`}>
                  {lvl}
                </span>
                <Bar p={p} />
                <span className="text-right font-mono text-[11px] text-dim">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {a.type === 'noul' && (
        <div>
          <div className="mb-3 flex items-baseline gap-3">
            <span className="font-display text-3xl text-fg">{a.answer ? 'yes' : 'no'}</span>
            <span className="font-mono text-[11px] text-faint">
              p(yes) {(a.probability * 100).toFixed(1)}%
            </span>
          </div>
          <Bar p={a.probability} />
        </div>
      )}
    </div>
  );
}

export default function RunStep({
  model,
  dataset,
  onNext,
  onRan,
}: {
  model: ModelEntry | null;
  dataset: Dataset | null;
  onNext: () => void;
  onRan?: () => void;
}) {
  const lmRepo =
    model?.arch === 'decoder-lm' && model.runtime !== 'planned' ? (model.onnx ?? model.hf) : null;
  const [head, setHead] = useState<TrainedHead | null>(() => getTrainedHead());
  const [ft, setFt] = useState<FtArtifact | null>(() => getFtArtifact());
  const [engine, setEngine] = useState<'jev' | 'head' | 'ft' | 'lm'>(
    ft ? 'ft' : head ? 'head' : lmRepo ? 'lm' : 'jev',
  );
  const [fallbackRepo, setFallbackRepo] = useState(RUNNABLE[0].id);
  const repo = isJevRunnable(model?.onnx) ? (model?.onnx as string) : fallbackRepo;

  const [info, setInfo] = useState<OpenJevInfo | null>(null);
  const [runtime, setRuntime] = useState<OpenJevRuntime | null>(null);
  const decideRef = useRef<DecideSession | null>(null);
  const decoderRef = useRef<DecoderSession | null>(null);
  const [decideRt, setDecideRt] = useState<{ device: string; dtype: string } | null>(null);
  const [decoderRt, setDecoderRt] = useState<{ device: string; dtype: string } | null>(null);
  const [phase, setPhase] = useState<'idle' | 'loading' | 'ready' | 'deciding'>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const datasetPrefill = useMemo(() => {
    const row = dataset?.rows[0];
    if (!row) return null;
    const cols = dataset?.columns ?? [];
    let best = '';
    for (const c of cols) if ((row[c] ?? '').length > best.length) best = row[c];
    return best || null;
  }, [dataset]);

  const [state, setState] = useState(datasetPrefill ?? DEFAULT_STATE);
  const [packOverride, setPackOverride] = useState<string | null>(null);
  const pack = packOverride ?? model?.sectors[0] ?? 'decision';
  const [customNoul, setCustomNoul] = useState('');
  const [results, setResults] = useState<{ q: Question[]; a: Answer[]; ms: number } | null>(null);
  const [headResult, setHeadResult] = useState<{ probs: Record<string, number>; ms: number } | null>(null);
  const [taskResults, setTaskResults] = useState<{ tasks: FtTaskResult[]; ms: number } | null>(null);

  useEffect(() => {
    let live = true;
    Promise.all([getWorkspaceItem('trainedHead'), getWorkspaceItem('ftArtifact')])
      .then(([savedHead, savedFt]) => {
        if (!live) return;
        if (savedHead && !getTrainedHead()) {
          setTrainedHead(savedHead);
          setHead(savedHead);
          if (!savedFt) setEngine('head');
        }
        if (savedFt && !getFtArtifact()) {
          setFtArtifact(savedFt);
          setFt(savedFt);
          setEngine('ft');
        }
      })
      .catch(() => setError('Could not restore saved classifiers from this browser.'));
    return () => {
      live = false;
    };
  }, []);

  const loadHeadFile = async (f: File) => {
    try {
      const { tensors, meta } = readSafetensors(await f.arrayBuffer());
      if (
        meta.anvil === 'ft-v1' ||
        Object.keys(tensors).some(
          (k) =>
            k.startsWith('lora.') ||
            k.startsWith('encoder.layer.') ||
            k.startsWith('embeddings.LayerNorm.'),
        )
      ) {
        const a = ftFromSafetensors(tensors, meta);
        setFtArtifact(a);
        setFt(a);
        setEngine('ft');
        void setWorkspaceItem('ftArtifact', a).catch(() =>
          setError('Loaded for this tab, but could not save the classifier locally.'),
        );
        setError(null);
        return;
      }
      const h = headFromSafetensors(tensors, meta);
      setTrainedHead(h);
      setHead(h);
      setEngine('head');
      void setWorkspaceItem('trainedHead', h).catch(() =>
        setError('Loaded for this tab, but could not save the classifier locally.'),
      );
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load that .safetensors file.');
    }
  };

  /** Free JEV/Decide/decoder sessions when the user moves to a classifier engine. */
  const releaseHeavy = async () => {
    await releaseSessions('none');
    decideRef.current = null;
    decoderRef.current = null;
    setRuntime(null);
    setDecideRt(null);
    setDecoderRt(null);
  };

  const runFt = async () => {
    if (!ft) return;
    setPhase('deciding');
    setError(null);
    setHeadResult(null);
    setTaskResults(null);
    try {
      const t0 = performance.now();
      const [tasks] = await Promise.all([
        inferFtTasks(state, ft, { preferWgsl: true }),
        releaseHeavy(),
      ]);
      setTaskResults({ tasks, ms: performance.now() - t0 });
      setPhase('ready');
      onRan?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('ready');
    }
  };

  const runHead = async () => {
    if (!head) return;
    setPhase('deciding');
    setError(null);
    setHeadResult(null);
    try {
      const t0 = performance.now();
      const [probs] = await Promise.all([inferHead(state, head), releaseHeavy()]);
      setHeadResult({ probs, ms: performance.now() - t0 });
      setPhase('ready');
      onRan?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase(head ? 'ready' : 'idle');
    }
  };

  const pickRepo = (id: string) => {
    setFallbackRepo(id);
    setInfo(null);
    setRuntime(null);
    setDecideRt(null);
    decideRef.current = null;
    setPhase('idle');
  };

  const decideMode = isDecideOnnx(repo);

  // fetch download/cache info whenever the target repo changes (jev family only)
  useEffect(() => {
    if (isDecideOnnx(repo)) return;
    let live = true;
    jevInfo(repo)
      .then((i) => live && setInfo(i))
      .catch(() => live && setInfo(null));
    return () => {
      live = false;
    };
  }, [repo]);

  const questions = useMemo(() => {
    const qs = [...PACKS[pack].questions];
    const extra = customNoul.trim();
    if (extra) qs.push({ type: 'noul', instructions: extra });
    return qs;
  }, [pack, customNoul]);

  const load = async () => {
    setPhase('loading');
    setProgress(0);
    setError(null);
    try {
      if (engine === 'lm') {
        if (!lmRepo) throw new Error('Pick a decoder-LM roster entry to use the generation engine.');
        await releaseSessions('decoder');
        const s = await loadDecoder(lmRepo, { onProgress: (p) => setProgress(p.progress) });
        decoderRef.current = s;
        setDecoderRt({ device: s.device, dtype: s.dtype });
      } else if (decideMode) {
        await releaseSessions('decide');
        decoderRef.current = null;
        setDecoderRt(null);
        const s = await loadDecide(repo, { onProgress: (p) => setProgress(p.progress) });
        decideRef.current = s;
        setDecideRt({ device: s.device, dtype: s.dtype });
      } else {
        await releaseSessions('jev');
        decoderRef.current = null;
        setDecoderRt(null);
        decideRef.current = null;
        setDecideRt(null);
        const j = await loadJev(repo, (p) => setProgress(p.progress));
        setRuntime(j.runtime);
      }
      setPhase('ready');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('idle');
    }
  };

  const run = async () => {
    setPhase('deciding');
    setError(null);
    try {
      const t0 = performance.now();
      let answers: Answer[];
      if (engine === 'lm') {
        const s = decoderRef.current ?? (lmRepo ? await loadDecoder(lmRepo) : null);
        if (!s) throw new Error('Pick a decoder-LM roster entry to use the generation engine.');
        answers = await s.decide(state, questions);
      } else if (decideMode) {
        answers = await (decideRef.current ?? (await loadDecide(repo))).decide(state, questions);
      } else {
        answers = await (await loadJev(repo)).decide(state, questions);
      }
      setResults({ q: questions, a: answers, ms: performance.now() - t0 });
      setPhase('ready');
      onRan?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('ready');
    }
  };

  const loadPct = Math.round(progress * 100);
  const downloadMB = info ? Math.round(info.downloadSize / 1e6) : null;
  const classifierLabels = engine === 'ft' ? ft?.labels ?? [] : head?.labels ?? [];

  return (
    <Step
      index="04"
      title="Run it on real text"
      lede="Run JEV decision checkpoints for typed answers, let a decoder LM generate them, or use your trained MiniLM classifier for a JEV-shaped choice result. Classifier training currently supports one fixed label set, not multi-question score/noul tasks."
      onNext={onNext}
      nextLabel="Configure training"
    >
      {/* engine picker */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-faint">engine</span>
        <button
          onClick={() => {
            setEngine('jev');
            setHeadResult(null);
          }}
          className={`rounded-full border px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors ${
            engine === 'jev' ? 'border-fg bg-fg text-card' : 'border-line text-dim hover:text-fg'
          }`}
        >
          jev decision model
        </button>
        <button
          onClick={() => {
            setEngine('head');
            setResults(null);
          }}
          disabled={!head}
          className={`rounded-full border px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors disabled:opacity-40 ${
            engine === 'head' ? 'border-ink bg-ink-dim text-ink' : 'border-line text-dim hover:text-fg'
          }`}
        >
          anvil head {head ? `· ${head.labels.length} classes` : '· none trained'}
        </button>
        <button
          onClick={() => {
            setEngine('ft');
            setResults(null);
            setHeadResult(null);
          }}
          disabled={!ft}
          className={`rounded-full border px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors disabled:opacity-40 ${
            engine === 'ft' ? 'border-ink bg-ink-dim text-ink' : 'border-line text-dim hover:text-fg'
          }`}
        >
          anvil-ft {ft ? `· ${ft.mode} ${ft.backend}` : '· none'}
        </button>
        {lmRepo && (
          <button
            onClick={() => {
              setEngine('lm');
              setResults(null);
              setHeadResult(null);
            }}
            className={`rounded-full border px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors ${
              engine === 'lm' ? 'border-fg bg-fg text-card' : 'border-line text-dim hover:text-fg'
            }`}
          >
            decoder lm
          </button>
        )}
        <label className="cursor-pointer rounded-full border border-line px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-dim transition-colors hover:border-fg hover:text-fg">
          load .safetensors
          <input
            type="file"
            accept=".safetensors"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void loadHeadFile(f);
              e.target.value = '';
            }}
          />
        </label>
      </div>

      {/* model line */}
      <div className="mb-6 flex flex-wrap items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
          checkpoint
        </span>
        {engine === 'head' ? (
          <Tag tone="ink">anvil head — {head?.encoder ?? 'MiniLM'} + linear</Tag>
        ) : engine === 'ft' ? (
          <Tag tone="ink">
            anvil-ft — MiniLM {ft?.mode}
            {ft?.mode === 'lora' ? ` r=${ft.rank}` : ''} + head
          </Tag>
        ) : engine === 'lm' ? (
          <Tag tone="ink">{lmRepo}</Tag>
        ) : isJevRunnable(model?.onnx) ? (
          <Tag tone="ink">{model?.onnx}</Tag>
        ) : (
          <>
            <Tag>
              {model?.onnx ? 'onnx exists — driver pending' : 'reference model'}
            </Tag>
            <select
              value={repo}
              onChange={(e) => pickRepo(e.target.value)}
              className="rounded-lg border border-line bg-paper px-2.5 py-1.5 font-mono text-[11px] text-fg"
            >
              {RUNNABLE.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label} — {r.note}
                </option>
              ))}
            </select>
          </>
        )}
        {engine === 'jev' && info && (
          <Tag tone={info.isCached ? 'ink' : 'dim'}>
            {info.isCached ? 'cached' : `~${downloadMB} MB download`}
          </Tag>
        )}
        {engine === 'jev' && runtime && (
          <Tag tone="solid">
            {runtime.device} · {runtime.dtype}
          </Tag>
        )}
        {engine === 'jev' && decideRt && (
          <Tag tone="solid">
            {decideRt.device} · {decideRt.dtype}
          </Tag>
        )}
        {engine === 'lm' && decoderRt && (
          <Tag tone="solid">
            {decoderRt.device} · {decoderRt.dtype}
          </Tag>
        )}
      </div>

      {/* state + questions */}
      <div className="mb-5 grid gap-3 lg:grid-cols-5">
        <div className="rounded-2xl border border-line bg-card p-5 lg:col-span-3">
          <div className="mb-3 flex items-center justify-between">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-faint">state</p>
            {datasetPrefill && <Tag tone="dim">prefilled from {dataset?.name}</Tag>}
          </div>
          <textarea
            value={state}
            onChange={(e) => setState(e.target.value)}
            rows={5}
            spellCheck={false}
            className="w-full resize-y rounded-lg border border-line bg-paper p-3 font-mono text-[13px] leading-relaxed text-fg focus:outline-none"
            placeholder="Paste the text to decide on…"
          />
          <div className="mt-3 flex items-center gap-3">
            {engine === 'head' || engine === 'ft' ? (
              phase === 'deciding' ? (
                <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-dim">
                  <SpinnerGap size={14} className="animate-spin text-ink" />
                  {engine === 'ft' ? 'encoder forward…' : 'embed + forward…'}
                </span>
              ) : (
                <button
                  onClick={engine === 'ft' ? runFt : runHead}
                  disabled={!state.trim() || (engine === 'ft' ? !ft : !head)}
                  className="flex items-center gap-2 rounded-full bg-fg px-5 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:opacity-40"
                >
                  <Play size={13} weight="fill" />
                  Classify
                </button>
              )
            ) : phase === 'ready' ? (
              <button
                onClick={run}
                disabled={!state.trim()}
                className="flex items-center gap-2 rounded-full bg-fg px-5 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:opacity-40"
              >
                <Play size={13} weight="fill" />
                {engine === 'lm' ? 'Generate' : 'Decide'}
              </button>
            ) : phase === 'deciding' ? (
              <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-dim">
                <SpinnerGap size={14} className="animate-spin text-ink" />
                {engine === 'lm' ? 'generating…' : 'one forward pass…'}
              </span>
            ) : (
              <button
                onClick={load}
                disabled={phase === 'loading'}
                className="flex items-center gap-2 rounded-full bg-fg px-5 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:opacity-40"
              >
                {phase === 'loading' ? (
                  <SpinnerGap size={13} className="animate-spin" />
                ) : (
                  <ArrowRight size={13} />
                )}
                {phase === 'loading'
                  ? `fetching ${loadPct}%`
                  : info?.isCached
                    ? 'Load (cached)'
                    : `Load model${downloadMB ? ` — ${downloadMB} MB` : ''}`}
              </button>
            )}
            {phase === 'loading' && (
              <div className="h-1 flex-1 overflow-hidden rounded-full bg-line">
                <div
                  className="h-full rounded-full bg-ink transition-all"
                  style={{ width: `${loadPct}%` }}
                />
              </div>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-line bg-card p-5 lg:col-span-2">
          {engine === 'head' || engine === 'ft' ? (
            <>
              <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                {ft?.tasks && ft.tasks.length > 1 && engine === 'ft'
                  ? `tasks · ${ft.tasks.length}`
                  : 'label set'}
              </p>
              {engine === 'ft' && ft?.tasks && ft.tasks.length > 1 ? (
                <div className="flex flex-col gap-2.5">
                  {ft.tasks.map((t) => (
                    <div key={t.name}>
                      <p className="mb-1 font-mono text-[9px] uppercase tracking-widest text-ink">
                        {t.name}
                        {t.valAcc != null && (
                          <span className="ml-2 text-faint">val {(t.valAcc * 100).toFixed(0)}%</span>
                        )}
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {t.labels.map((l) => (
                          <span
                            key={l}
                            className="rounded-full border border-ink/60 bg-ink-dim px-2.5 py-0.5 font-mono text-[10px] text-ink"
                          >
                            {l}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {((engine === 'ft' ? ft?.labels : head?.labels) ?? []).map((l) => (
                    <span
                      key={l}
                      className="rounded-full border border-ink/60 bg-ink-dim px-2.5 py-0.5 font-mono text-[10px] text-ink"
                    >
                      {l}
                    </span>
                  ))}
                </div>
              )}
              <p className="mt-3 font-mono text-[10px] leading-relaxed text-faint">
                {engine === 'ft'
                  ? ft
                    ? `${ft.mode} on ${ft.arch === 'deberta' ? 'DeBERTa-v3-small' : 'MiniLM'} · ${ft.rows} rows · ${ft.backend} · loss ${ft.loss.toFixed(3)}${ft.valAcc != null ? ` · val ${(ft.valAcc * 100).toFixed(0)}%` : ''}`
                    : 'fine-tune in step 05 or load a .safetensors export'
                  : head
                    ? `trained ${head.rows} rows · ${head.backend} · loss ${head.loss.toFixed(3)}${head.valAcc != null ? ` · val ${(head.valAcc * 100).toFixed(0)}%` : ''}`
                    : 'train a head in step 05 or load a .safetensors export'}
              </p>
            </>
          ) : (
            <>
          <div className="mb-3 flex items-center justify-between">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-faint">questions</p>
            <select
              value={pack}
              onChange={(e) => setPackOverride(e.target.value)}
              className="rounded-lg border border-line bg-paper px-2 py-1 font-mono text-[11px] text-fg"
            >
              {Object.entries(PACKS).map(([id, p]) => (
                <option key={id} value={id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <ul className="flex flex-col gap-2">
            {PACKS[pack].questions.map((q, i) => (
              <li key={i} className="rounded-lg border border-line-soft bg-paper px-3 py-2">
                <span className="font-mono text-[9px] uppercase tracking-widest text-ink">
                  {q.type}
                </span>
                <p className="text-[12.5px] text-fg">{q.instructions}</p>
                {'options' in q && (
                  <p className="mt-0.5 font-mono text-[10px] text-faint">{q.options.join(' · ')}</p>
                )}
              </li>
            ))}
          </ul>
          <input
            value={customNoul}
            onChange={(e) => setCustomNoul(e.target.value)}
            placeholder="Add a yes/no statement…"
            className="mt-3 w-full rounded-lg border border-line bg-paper px-3 py-2 font-mono text-[11.5px] text-fg placeholder:text-faint focus:outline-none"
          />
            </>
          )}
        </div>
      </div>

      {error && (
        <p className="mb-5 rounded-xl border border-dashed border-fg/40 bg-paper px-4 py-3 font-mono text-[12px] text-fg">
          {error}
        </p>
      )}

      {/* ft result — one card per task */}
      {taskResults && (
        <div className="mb-5">
          <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
            trained classifier · {taskResults.tasks.length} task
            {taskResults.tasks.length > 1 ? 's' : ''} · {taskResults.ms.toFixed(0)} ms · choice
            output
          </p>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {taskResults.tasks.map((t) => (
              <AnswerCard
                key={t.name}
                q={{ type: 'choice', instructions: t.name, options: t.labels }}
                a={classifierAnswer(t.labels, t.probs)}
              />
            ))}
          </div>
        </div>
      )}

      {/* head result */}
      {headResult && (
        <div className="mb-5">
          <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
            trained classifier · {headResult.ms.toFixed(0)} ms · choice output
          </p>
          <AnswerCard
            q={{ type: 'choice', instructions: 'Select the most likely training label.', options: classifierLabels }}
            a={classifierAnswer(classifierLabels, headResult.probs)}
          />
        </div>
      )}

      {/* results */}
      {results && (
        <div>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
            answers · {results.ms.toFixed(0)} ms forward pass
          </p>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {results.a.map((a, i) => (
              <AnswerCard key={i} q={results.q[i]} a={a} />
            ))}
          </div>
        </div>
      )}
    </Step>
  );
}
