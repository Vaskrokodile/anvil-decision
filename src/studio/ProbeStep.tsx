import { Cpu, Warning } from '@phosphor-icons/react';
import type { GpuReport } from '../lib/gpu';
import { fmtBytes } from '../lib/gpu';
import { TIER_LABELS } from '../data/models';
import { Tag } from '../components/ui';
import Step from './Step';

function Row({ k, v, mono = true }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-line-soft py-2.5 last:border-0">
      <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-faint">{k}</span>
      <span className={`text-right text-[13px] text-fg ${mono ? 'font-mono' : ''}`}>{v}</span>
    </div>
  );
}

export default function ProbeStep({
  report,
  onNext,
}: {
  report: GpuReport | null;
  onNext: () => void;
}) {
  return (
    <Step
      index="01"
      title="Hardware probe"
      lede="Reading navigator.gpu — adapter identity, buffer limits, fp16 support. This decides what you’re allowed to train."
      onNext={onNext}
      nextDisabled={!report}
    >
      {report === null ? (
        <div className="flex items-center gap-3 rounded-2xl border border-line bg-card p-5 font-mono text-[12px] text-dim">
          <span className="h-2 w-2 animate-pulse rounded-full bg-ink" />
          Requesting adapter…
        </div>
      ) : !report.ok ? (
        <div className="rounded-2xl border border-dashed border-fg/40 bg-paper p-5 text-fg">
          <p className="mb-2 flex items-center gap-2 text-[14px] font-medium text-fg">
            <Warning size={16} /> WebGPU unavailable
          </p>
          <p className="mb-4 text-[13px] text-dim">{report.error}</p>
          <p className="font-mono text-[11px] text-faint">
            Fallback path: WASM/CPU inference only — training is disabled on this machine.
          </p>
        </div>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {/* adapter identity */}
          <div className="rounded-2xl border border-line bg-card p-5">
            <p className="mb-4 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
              <Cpu size={14} className="text-fg" /> Adapter
            </p>
            <Row k="vendor" v={report.vendor ?? 'unknown'} />
            <Row k="architecture" v={report.architecture ?? 'unknown'} />
            <Row k="device" v={report.device ?? 'masked'} />
            <Row k="description" v={report.description ?? 'masked'} />
            <Row k="fallback adapter" v={String(report.isFallbackAdapter ?? 'unknown')} />
            <Row k="shader-f16" v={report.fp16 ? 'supported' : 'not reported'} />
            <Row k="device memory" v={report.deviceMemoryGB ? `~${report.deviceMemoryGB} GB` : 'not reported'} />
            <Row k="cpu cores" v={String(report.cores ?? 'unknown')} />
          </div>

          {/* limits + verdict */}
          <div className="flex flex-col rounded-2xl border border-line bg-card">
            <div className="flex-1 p-5">
              <p className="mb-4 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                Limits that matter
              </p>
              <Row
                k="max storage buffer"
                v={
                  report.maxStorageBufferBindingSize
                    ? `${fmtBytes(report.maxStorageBufferBindingSize / (1024 * 1024))}`
                    : 'unknown'
                }
              />
              <Row
                k="max buffer"
                v={report.maxBufferSize ? fmtBytes(report.maxBufferSize / (1024 * 1024)) : 'unknown'}
              />
              <Row
                k="workgroups / dim"
                v={String(report.maxComputeWorkgroupsPerDimension ?? 'unknown')}
              />
              <Row
                k="invocations / workgroup"
                v={String(report.maxComputeInvocationsPerWorkgroup ?? 'unknown')}
              />
            </div>
            <div className="rounded-b-2xl border-t border-line bg-paper p-5">
              <div className="mb-3 flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                  Assigned tier
                </span>
                <Tag tone={report.tier >= 3 ? 'ink' : report.tier === 2 ? 'line' : 'faint'}>
                  {TIER_LABELS[report.tier]}
                </Tag>
              </div>
              <Row k="weights budget" v={fmtBytes(report.weightsBudgetMB)} />
              <Row k="full-ft param ceiling" v={`≈ ${report.trainableBudgetM}M params`} />
              <p className="mt-3 font-mono text-[10px] leading-relaxed text-faint">
                LoRA extends the ceiling ~10× past the full-FT number. Budgets are conservative on
                purpose — activations and browser overhead eat the rest.
              </p>
            </div>
          </div>
        </div>
      )}
    </Step>
  );
}
