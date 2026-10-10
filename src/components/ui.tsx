import type { ReactNode } from 'react';
import { motion } from 'motion/react';

/** Bracketed label + serif display heading — the Nexora-style section opener. */
export function SectionHeader({
  index,
  title,
  lede,
  center = false,
}: {
  index: string;
  title: ReactNode;
  lede?: string;
  center?: boolean;
}) {
  return (
    <div className={`mb-14 ${center ? 'text-center' : ''}`}>
      <motion.p
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true, margin: '-80px' }}
        className="mb-5 font-mono text-[10.5px] uppercase tracking-[0.28em] text-faint"
      >
        [ {index} ]
      </motion.p>
      <motion.h2
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.55 }}
        className={`font-display text-[clamp(2rem,4.4vw,3.6rem)] leading-[1.04] tracking-[-0.015em] text-fg ${
          center ? 'mx-auto max-w-[22ch]' : 'max-w-[20ch]'
        }`}
      >
        {title}
      </motion.h2>
      {lede && (
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.5, delay: 0.12 }}
          className={`mt-5 text-[15px] leading-relaxed text-dim ${center ? 'mx-auto max-w-[58ch]' : 'max-w-[58ch]'}`}
        >
          {lede}
        </motion.p>
      )}
    </div>
  );
}

/**
 * Tag — fill level carries meaning:
 * solid = selected, ink = fits/positive, line = marginal, dim = neutral,
 * faint = disabled/negative.
 */
export function Tag({
  children,
  tone = 'dim',
}: {
  children: ReactNode;
  tone?: 'solid' | 'ink' | 'line' | 'dim' | 'faint';
}) {
  const tones = {
    solid: 'border-fg bg-fg text-card',
    ink: 'border-ink/60 bg-ink-dim text-ink',
    line: 'border-fg/30 text-fg',
    dim: 'border-line text-dim',
    faint: 'border-dashed border-line text-faint',
  };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function FitBadge({ fit }: { fit: 'fits' | 'marginal' | 'no' | 'unknown' }) {
  if (fit === 'unknown') return <Tag>unprobed</Tag>;
  const map = {
    fits: ['ink', 'fits'],
    marginal: ['line', 'marginal'],
    no: ['faint', 'won’t fit'],
  } as const;
  const [tone, label] = map[fit];
  return <Tag tone={tone}>{label}</Tag>;
}
