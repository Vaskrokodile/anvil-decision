import { useState } from 'react';
import { MODELS, fitsTier, ARCH_LABELS, type Sector } from '../data/models';
import type { GpuReport } from '../lib/gpu';
import { fmtBytes } from '../lib/gpu';
import { Tag, FitBadge } from '../components/ui';
import Step from './Step';

const SECTOR_TABS: { id: Sector | 'all'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'decision', label: 'Decision' },
  { id: 'labelling', label: 'Labelling' },
  { id: 'scientific', label: 'Scientific' },
  { id: 'trust', label: 'Trust' },
  { id: 'general', label: 'General' },
];

export default function ModelStep({
  report,
  modelId,
  onPick,
  onNext,
}: {
  report: GpuReport | null;
  modelId: string | null;
  onPick: (hf: string) => void;
  onNext: () => void;
}) {
  const [tab, setTab] = useState<Sector | 'all'>('all');
  const tier = report?.ok ? report.tier : 0;

  const rows = MODELS.filter((m) => tab === 'all' || m.sectors.includes(tab));

  return (
    <Step
      index="02"
      title="Pick a base model"
      lede={
        report?.ok
          ? `Tier ${tier} assigned — models are marked fits / marginal / won’t fit against it.`
          : 'Probe your hardware first — or browse anyway.'
      }
      onNext={onNext}
      nextDisabled={!modelId}
    >
      <div className="mb-5 flex flex-wrap gap-2">
        {SECTOR_TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-full border px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors ${
              tab === t.id
                ? 'border-fg bg-fg text-card'
                : 'border-line text-dim hover:text-fg'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {rows.map((m) => {
          const fit = report?.ok ? fitsTier(m, tier) : 'unknown';
          const selected = modelId === m.hf;
          const disabled = fit === 'no';
          return (
            <button
              key={m.hf}
              disabled={disabled}
              onClick={() => onPick(m.hf)}
              className={`group rounded-2xl border bg-card p-6 text-left transition-colors ${
                selected
                  ? 'border-ink bg-ink-dim/50'
                  : disabled
                    ? 'cursor-not-allowed border-line-soft opacity-45'
                    : 'border-line hover:border-faint hover:bg-paper'
              }`}
            >
              <div className="mb-2 flex items-center justify-between gap-3">
                <span className="text-[15px] font-medium text-fg">{m.name}</span>
                <FitBadge fit={fit} />
              </div>
              <p className="mb-3 font-mono text-[10.5px] text-faint">{m.hf}</p>
              <p className="mb-4 text-[13px] leading-relaxed text-dim">{m.blurb}</p>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-faint">
                <span>{m.paramsM}M</span>
                <span>{fmtBytes(m.weightsMB)}</span>
                <span>{ARCH_LABELS[m.arch]}</span>
                <span>min T{m.minTier}</span>
                <span className="flex gap-1">
                  {m.trainMode.map((t) => (
                    <Tag key={t}>{t}</Tag>
                  ))}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </Step>
  );
}
