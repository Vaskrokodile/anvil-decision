import { useEffect, useMemo, useRef, useState } from 'react';
import {
  FileArrowUp,
  Table,
  SpinnerGap,
  Database,
  DownloadSimple,
  Broom,
} from '@phosphor-icons/react';
import { SECTORS } from '../data/sectors';
import {
  parseFile,
  parseJsonl,
  templateRows,
  ParseError,
  analyzeDataset,
  dedupeDataset,
  type Dataset,
} from '../lib/dataset';
import {
  parseDatasetRef,
  fetchSplits,
  fetchRows,
  HfError,
  type HfSplit,
} from '../lib/hfDatasets';
import { Tag } from '../components/ui';
import Step from './Step';

const nowIso = () => new Date().toISOString();

interface StarterMeta {
  file: string;
  name: string;
  desc: string;
  sector: string;
  dataset: string;
  license: string;
  rows: number;
  classes: number;
  labels: string[];
}

export default function DatasetStep({
  dataset,
  onDataset,
  onNext,
}: {
  dataset: Dataset | null;
  onDataset: (d: Dataset) => void;
  onNext: () => void;
}) {
  const [sectorId, setSectorId] = useState('decision');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // starter sets
  const [starters, setStarters] = useState<StarterMeta[] | null>(null);
  const [starterBusy, setStarterBusy] = useState<string | null>(null);

  // hf import
  const [hfInput, setHfInput] = useState('');
  const [hfId, setHfId] = useState<string | null>(null);
  const [hfSplits, setHfSplits] = useState<HfSplit[]>([]);
  const [hfSel, setHfSel] = useState(0);
  const [hfPhase, setHfPhase] = useState<'idle' | 'splits' | 'rows'>('idle');
  const [hfGot, setHfGot] = useState(0);
  const [hfError, setHfError] = useState<string | null>(null);

  const sector = SECTORS.find((s) => s.id === sectorId)!;
  const health = useMemo(() => (dataset ? analyzeDataset(dataset) : null), [dataset]);

  useEffect(() => {
    let live = true;
    fetch('/datasets/index.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => live && setStarters(j?.datasets ?? null))
      .catch(() => live && setStarters(null));
    return () => {
      live = false;
    };
  }, []);

  const scaffold = () => {
    setError(null);
    onDataset({
      name: `${sector.id}-dataset`,
      sector: sector.id,
      columns: sector.schema.map((f) => f.field),
      rows: templateRows(sector.schema),
      source: 'template',
      createdAt: nowIso(),
    });
  };

  const ingest = async (file: File) => {
    setError(null);
    setBusy(true);
    try {
      const { columns, rows } = await parseFile(file);
      onDataset({
        name: file.name.replace(/\.[^.]+$/, ''),
        sector: sector.id,
        columns,
        rows,
        source: 'import',
        createdAt: nowIso(),
      });
    } catch (e) {
      setError(e instanceof ParseError ? e.message : 'Could not parse that file.');
    } finally {
      setBusy(false);
    }
  };

  const loadStarter = async (s: StarterMeta) => {
    setError(null);
    setStarterBusy(s.file);
    try {
      const r = await fetch(`/datasets/${s.file}`);
      if (!r.ok) throw new Error('missing');
      const { columns, rows } = parseJsonl(await r.text());
      setSectorId(s.sector);
      onDataset({
        name: s.name.toLowerCase().replace(/\s+/g, '-'),
        sector: s.sector,
        columns,
        rows,
        source: 'starter',
        createdAt: nowIso(),
      });
    } catch {
      setError(`Could not load ${s.file} — run scripts/fetch-starter.mts to regenerate it.`);
    } finally {
      setStarterBusy(null);
    }
  };

  const hfConnect = async () => {
    setHfError(null);
    const id = parseDatasetRef(hfInput);
    if (!id) {
      setHfError('Not a dataset reference — use owner/name or a huggingface.co link.');
      return;
    }
    setHfPhase('splits');
    setHfSplits([]);
    try {
      const splits = await fetchSplits(id);
      setHfId(id);
      setHfSplits(splits);
      setHfSel(Math.max(0, splits.findIndex((s) => s.split === 'train')));
      setHfPhase('idle');
    } catch (e) {
      setHfError(e instanceof HfError ? e.message : 'Could not reach Hugging Face.');
      setHfPhase('idle');
    }
  };

  const hfLoad = async () => {
    if (!hfId || hfSplits.length === 0) return;
    const { config, split } = hfSplits[hfSel];
    setHfError(null);
    setHfPhase('rows');
    setHfGot(0);
    try {
      const { columns, rows } = await fetchRows(hfId, config, split, 512, setHfGot);
      onDataset({
        name: `${hfId.split('/')[1]}-${split}`,
        sector: sector.id,
        columns,
        rows,
        source: 'hf',
        createdAt: nowIso(),
      });
    } catch (e) {
      setHfError(e instanceof HfError ? e.message : 'Row download failed.');
    } finally {
      setHfPhase('idle');
    }
  };

  return (
    <Step
      index="03"
      title="Build the dataset"
      lede="Load a real starter set, pull any public Hugging Face dataset, scaffold a template, or drop in your own file. Rows are parsed locally — nothing uploads."
      onNext={onNext}
      nextDisabled={!dataset || dataset.rows.length === 0}
    >
      {/* sector pick */}
      <div className="mb-5 flex flex-wrap gap-2">
        {SECTORS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSectorId(s.id)}
            className={`rounded-full border px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors ${
              sectorId === s.id
                ? 'border-fg bg-fg text-card'
                : 'border-line text-dim hover:text-fg'
            }`}
          >
            {s.name}
          </button>
        ))}
      </div>

      <div className="mb-6 flex items-start justify-between gap-4 rounded-2xl border border-line-soft bg-paper p-4">
        <div>
          <p className="mb-1 text-[13px] text-dim">{sector.desc}</p>
          <div className="flex flex-wrap gap-1.5">
            {sector.schema.map((f) => (
              <Tag key={f.field} tone={f.role === 'label' ? 'blue' : 'dim'}>
                {f.field}
              </Tag>
            ))}
          </div>
        </div>
        <button
          onClick={scaffold}
          className="flex shrink-0 items-center gap-2 rounded-full border border-line px-4 py-2 font-mono text-[10px] uppercase tracking-[0.15em] text-dim transition-colors hover:border-fg hover:text-fg"
        >
          <Table size={13} />
          Scaffold {sector.schema.length}-field template
        </button>
      </div>

      {/* starter sets */}
      {starters && starters.length > 0 && (
        <div className="mb-6">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
            starter sets · real data
          </p>
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {starters.map((s) => (
              <button
                key={s.file}
                onClick={() => void loadStarter(s)}
                disabled={starterBusy !== null}
                className="group flex flex-col items-start gap-1.5 rounded-2xl border border-line bg-card p-4 text-left transition-colors hover:border-faint disabled:opacity-50"
              >
                <span className="flex w-full items-center justify-between gap-2">
                  <span className="font-mono text-[11px] text-fg">{s.name}</span>
                  {starterBusy === s.file ? (
                    <SpinnerGap size={13} className="animate-spin text-blue" />
                  ) : (
                    <Database size={13} className="text-faint transition-colors group-hover:text-blue" />
                  )}
                </span>
                <span className="text-[12px] leading-snug text-dim">{s.desc}</span>
                <span className="font-mono text-[10px] uppercase tracking-widest text-faint">
                  {s.rows} rows · {s.classes} classes · {s.dataset}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* dropzone */}
      <button
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) void ingest(f);
        }}
        className={`mb-6 flex w-full flex-col items-center gap-2 rounded-2xl border border-dashed px-6 py-10 transition-colors ${
          dragging ? 'border-blue bg-blue-dim' : 'border-line bg-paper hover:border-faint'
        }`}
      >
        {busy ? (
          <SpinnerGap size={22} className="animate-spin text-blue" />
        ) : (
          <FileArrowUp size={22} className="text-dim" />
        )}
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-dim">
          {busy ? 'Parsing…' : 'Drop .jsonl / .csv / .json / .txt or click to browse'}
        </span>
        <span className="font-mono text-[10px] text-faint">
          parsed in-browser · never uploaded
        </span>
        <input
          ref={inputRef}
          type="file"
          accept=".jsonl,.ndjson,.csv,.json,.txt"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void ingest(f);
            e.target.value = '';
          }}
        />
      </button>

      {/* hf import */}
      <div className="mb-6 rounded-2xl border border-line bg-paper p-4">
        <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
          import from hugging face
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={hfInput}
            onChange={(e) => setHfInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void hfConnect()}
            placeholder="owner/dataset or huggingface.co/datasets/…"
            spellCheck={false}
            className="min-w-0 flex-1 rounded-lg border border-line bg-card px-3 py-2 font-mono text-[12px] text-fg placeholder:text-faint focus:outline-none focus:border-faint"
          />
          {hfSplits.length > 0 && (
            <select
              value={hfSel}
              onChange={(e) => setHfSel(Number(e.target.value))}
              className="rounded-lg border border-line bg-card px-2.5 py-2 font-mono text-[12px] text-fg"
            >
              {hfSplits.map((s, i) => (
                <option key={`${s.config}:${s.split}`} value={i}>
                  {s.config} · {s.split}
                </option>
              ))}
            </select>
          )}
          {hfSplits.length === 0 ? (
            <button
              onClick={() => void hfConnect()}
              disabled={hfPhase !== 'idle' || !hfInput.trim()}
              className="flex items-center gap-2 rounded-full border border-line px-4 py-2 font-mono text-[10px] uppercase tracking-[0.15em] text-dim transition-colors hover:border-fg hover:text-fg disabled:opacity-40"
            >
              {hfPhase === 'splits' ? (
                <SpinnerGap size={12} className="animate-spin" />
              ) : (
                <Database size={12} />
              )}
              Connect
            </button>
          ) : (
            <button
              onClick={() => void hfLoad()}
              disabled={hfPhase !== 'idle'}
              className="flex items-center gap-2 rounded-full bg-fg px-4 py-2 font-mono text-[10px] uppercase tracking-[0.15em] text-card transition-opacity hover:opacity-85 disabled:opacity-40"
            >
              {hfPhase === 'rows' ? (
                <>
                  <SpinnerGap size={12} className="animate-spin" />
                  {hfGot} rows…
                </>
              ) : (
                <>
                  <DownloadSimple size={12} />
                  Load ≤512 rows
                </>
              )}
            </button>
          )}
        </div>
        {hfError && (
          <p className="mt-2 font-mono text-[11px] text-fg">{hfError}</p>
        )}
      </div>

      {error && (
        <p className="mb-6 rounded-2xl border border-dashed border-fg/40 bg-paper px-4 py-3 font-mono text-[12px] text-fg">
          {error}
        </p>
      )}

      {/* preview */}
      {dataset && (
        <div className="overflow-hidden rounded-2xl border border-line bg-card">
          <div className="flex items-center justify-between border-b border-line bg-paper px-4 py-2.5">
            <span className="font-mono text-[11px] text-fg">{dataset.name}</span>
            <span className="font-mono text-[10px] uppercase tracking-widest text-faint">
              {dataset.rows.length} rows · {dataset.columns.length} cols · {dataset.source}
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse text-left">
              <thead>
                <tr className="border-b border-line-soft">
                  {dataset.columns.map((c) => (
                    <th
                      key={c}
                      className="px-3 py-2 font-mono text-[10px] font-normal uppercase tracking-[0.15em] text-faint"
                    >
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.rows.slice(0, 8).map((r, i) => (
                  <tr key={i} className="border-b border-line-soft last:border-0">
                    {dataset.columns.map((c) => (
                      <td
                        key={c}
                        className="max-w-[280px] truncate px-3 py-2 font-mono text-[11.5px] text-dim"
                      >
                        {r[c] || <span className="text-faint">—</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {dataset.rows.length > 8 && (
            <p className="border-t border-line-soft px-4 py-2 font-mono text-[10px] text-faint">
              + {dataset.rows.length - 8} more rows
            </p>
          )}

          {/* health */}
          {health && (
            <div className="border-t border-line bg-paper px-4 py-3">
              <div className="mb-2 flex flex-wrap items-center gap-1.5">
                <Tag tone={health.dupes > 0 ? 'line' : 'blue'}>
                  {health.dupes > 0 ? `${health.dupes} dupes` : 'no dupes'}
                </Tag>
                {health.textCol && <Tag tone="dim">text: {health.textCol}</Tag>}
                {health.labelGuess && (
                  <Tag tone="blue">
                    label: {health.labelGuess} · {health.classCounts.length} classes
                  </Tag>
                )}
                {health.dupes > 0 && (
                  <button
                    onClick={() => onDataset(dedupeDataset(dataset))}
                    className="ml-auto flex items-center gap-1.5 rounded-full border border-line px-3 py-1 font-mono text-[10px] uppercase tracking-[0.15em] text-dim transition-colors hover:border-fg hover:text-fg"
                  >
                    <Broom size={11} />
                    drop dupes
                  </button>
                )}
              </div>
              {health.classCounts.length > 0 && (
                <div className="mb-2 flex flex-col gap-1">
                  {health.classCounts.slice(0, 8).map(([label, cnt]) => (
                    <div key={label} className="grid grid-cols-[9rem_1fr_3rem] items-center gap-2">
                      <span className="truncate font-mono text-[10.5px] text-dim">{label}</span>
                      <div className="h-1 overflow-hidden rounded-full bg-line">
                        <div
                          className="h-full rounded-full bg-blue"
                          style={{
                            width: `${Math.round((cnt / health.classCounts[0][1]) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-right font-mono text-[10px] text-faint">{cnt}</span>
                    </div>
                  ))}
                  {health.classCounts.length > 8 && (
                    <p className="font-mono text-[10px] text-faint">
                      + {health.classCounts.length - 8} more classes
                    </p>
                  )}
                </div>
              )}
              {health.warnings.length > 0 && (
                <div className="flex flex-col gap-0.5">
                  {health.warnings.map((w) => (
                    <p key={w} className="font-mono text-[10.5px] text-faint">
                      · {w}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Step>
  );
}
