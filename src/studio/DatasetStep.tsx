import { useRef, useState } from 'react';
import { FileArrowUp, Table, SpinnerGap } from '@phosphor-icons/react';
import { SECTORS } from '../data/sectors';
import {
  parseFile,
  templateRows,
  ParseError,
  type Dataset,
} from '../lib/dataset';
import { Tag } from '../components/ui';
import Step from './Step';

const nowIso = () => new Date().toISOString();

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

  const sector = SECTORS.find((s) => s.id === sectorId)!;

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

  return (
    <Step
      index="03"
      title="Build the dataset"
      lede="Pick a sector for its schema, scaffold a template to fill in, or drop in your own file. Rows are parsed locally — nothing uploads."
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
        </div>
      )}
    </Step>
  );
}
