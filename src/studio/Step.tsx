import type { ReactNode } from 'react';
import { ArrowRight } from '@phosphor-icons/react';

export default function Step({
  index,
  title,
  lede,
  children,
  onNext,
  nextLabel = 'Continue',
  nextDisabled,
}: {
  index: string;
  title: string;
  lede?: string;
  children: ReactNode;
  onNext?: () => void;
  nextLabel?: string;
  nextDisabled?: boolean;
}) {
  return (
    <div className="p-6 sm:p-8">
      <header className="mb-8 flex items-start justify-between gap-6">
        <div>
          <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.25em] text-faint">
            [ step {index} ]
          </p>
          <h2 className="text-2xl font-medium tracking-tight text-fg">{title}</h2>
          {lede && <p className="mt-2 max-w-[56ch] text-[14px] text-dim">{lede}</p>}
        </div>
        {onNext && (
          <button
            onClick={onNext}
            disabled={nextDisabled}
            className="mt-1 flex shrink-0 items-center gap-2 rounded-full border border-line px-5 py-2 font-mono text-[11px] uppercase tracking-[0.15em] text-fg transition-colors enabled:hover:border-fg disabled:cursor-not-allowed disabled:opacity-40"
          >
            {nextLabel}
            <ArrowRight size={13} />
          </button>
        )}
      </header>
      {children}
    </div>
  );
}
