import { useState } from 'react';
import { motion } from 'motion/react';
import { ArrowUpRight } from '@phosphor-icons/react';
import { MODELS, HONORABLE_MENTIONS, ARCH_LABELS, fitsTier, type Sector } from '../data/models';
import { useGpuReport } from '../lib/useGpu';
import { fmtBytes } from '../lib/gpu';
import { SectionHeader, Tag, FitBadge } from './ui';

const FILTERS: { id: Sector | 'all' | 'fastino'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'fastino', label: 'Fastino' },
  { id: 'decision', label: 'Decision' },
  { id: 'labelling', label: 'Labelling' },
  { id: 'scientific', label: 'Scientific' },
  { id: 'trust', label: 'Trust' },
  { id: 'general', label: 'General' },
];

export default function Roster() {
  const [filter, setFilter] = useState<Sector | 'all' | 'fastino'>('all');
  const gpu = useGpuReport();

  const rows = MODELS.filter((m) => {
    if (filter === 'all') return true;
    if (filter === 'fastino') return m.family === 'fastino';
    return m.sectors.includes(filter);
  });

  return (
    <section id="roster" className="bg-paper py-24">
      <div className="mx-auto max-w-[1340px] px-6">
        <SectionHeader
          index="03 — roster"
          title="Sub-1B specialists, weighed for your GPU"
          lede="Eleven candidates at or under the 1B ceiling. Fastino's GLiNER2.5 line covers decisions, extraction and guardrails; decoder LMs round out the general sector. The fit column is computed live against your probed adapter."
        />

        <div className="mb-5 flex flex-wrap gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={`rounded-full border px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] transition-colors ${
                filter === f.id
                  ? 'border-fg bg-fg text-card'
                  : 'border-line text-dim hover:border-faint hover:text-fg'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        <div className="overflow-hidden rounded-2xl border border-line bg-card">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[980px] border-collapse text-left">
              <thead>
                <tr className="border-b border-line">
                  {['Model', 'HF repo', 'Params', 'Weights', 'Arch', 'Min tier', 'Train', 'Runtime', 'Fit'].map(
                    (h) => (
                      <th
                        key={h}
                        className="px-5 py-4 font-mono text-[9.5px] font-normal uppercase tracking-[0.2em] text-faint"
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {rows.map((m, i) => (
                  <motion.tr
                    key={m.hf}
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.03 }}
                    className="group border-b border-line-soft last:border-0 hover:bg-paper"
                  >
                    <td className="px-5 py-4">
                      <a
                        href={`https://huggingface.co/${m.hf}`}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-1.5 text-[14px] font-medium text-fg hover:underline"
                      >
                        {m.name}
                        <ArrowUpRight
                          size={12}
                          className="text-faint opacity-0 transition-opacity group-hover:opacity-100"
                        />
                      </a>
                    </td>
                    <td className="px-5 py-4 font-mono text-[11px] text-dim">{m.hf}</td>
                    <td className="px-5 py-4 font-mono text-[12px] text-fg">{m.paramsM}M</td>
                    <td className="px-5 py-4 font-mono text-[12px] text-dim">
                      {fmtBytes(m.weightsMB)}
                    </td>
                    <td className="px-5 py-4 font-mono text-[11px] text-dim">
                      {ARCH_LABELS[m.arch]}
                    </td>
                    <td className="px-5 py-4 font-mono text-[11px] text-dim">T{m.minTier}</td>
                    <td className="px-5 py-4">
                      <div className="flex gap-1.5">
                        {m.trainMode.map((t) => (
                          <Tag key={t}>{t}</Tag>
                        ))}
                      </div>
                    </td>
                    <td className="px-5 py-4 font-mono text-[11px] text-dim">{m.runtime}</td>
                    <td className="px-5 py-4">
                      <FitBadge fit={gpu ? fitsTier(m, gpu.tier) : 'unknown'} />
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p className="mt-4 font-mono text-[11px] text-faint">
          Also worth a look — {HONORABLE_MENTIONS.map((m) => m.name).join(' · ')}
        </p>
      </div>
    </section>
  );
}
