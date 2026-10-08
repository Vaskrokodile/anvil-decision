import { ArrowUpRight } from '@phosphor-icons/react';
import { useGpuReport } from '../lib/useGpu';
import { TIER_LABELS } from '../data/models';

const LINKS: [string, string][] = [
  ['Roster', '#roster'],
  ['Sectors', '#sectors'],
  ['Pipeline', '#pipeline'],
];

export default function Nav({ onStudio }: { onStudio: () => void }) {
  const gpu = useGpuReport();

  return (
    <header className="absolute inset-x-0 top-0 z-50">
      <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between px-6">
        {/* logo — rotated square + spaced wordmark */}
        <a href="#/" className="flex items-center gap-3">
          <span className="inline-block h-3.5 w-3.5 rotate-45 rounded-[3px] bg-ice" />
          <span className="font-sans text-[15px] font-semibold tracking-[0.22em] text-ice">
            ANVIL
          </span>
        </a>

        {/* bracketed links */}
        <nav className="hidden items-center gap-8 md:flex">
          {LINKS.map(([label, href]) => (
            <a
              key={href}
              href={href}
              className="font-mono text-[11px] uppercase tracking-[0.18em] text-ice-dim transition-colors hover:text-ice"
            >
              [ {label} ]
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-5">
          {/* live GPU probe status */}
          <span className="hidden items-center gap-2 rounded-full border border-ice-line px-3.5 py-1.5 font-mono text-[10px] uppercase tracking-widest text-ice-faint lg:flex">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                gpu === null
                  ? 'animate-pulse bg-ice'
                  : gpu.ok
                    ? 'bg-ice'
                    : 'bg-ice-line'
              }`}
            />
            {gpu === null
              ? 'probing…'
              : gpu.ok
                ? TIER_LABELS[gpu.tier]
                : 'no webgpu'}
          </span>

          <button
            type="button"
            onClick={onStudio}
            className="flex items-center gap-1.5 rounded-full bg-ice px-4 py-2 font-mono text-[11px] font-medium uppercase tracking-[0.15em] text-abyss transition-opacity hover:opacity-85"
          >
            Studio
            <ArrowUpRight size={12} />
          </button>
        </div>
      </div>
    </header>
  );
}
