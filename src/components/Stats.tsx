import { useEffect, useRef, useState } from 'react';
import { animate, motion, useInView } from 'motion/react';
import { MODELS, TIER_LABELS, fitsTier } from '../data/models';
import { useGpuReport } from '../lib/useGpu';

/** Serif figure that counts 0 → `to` the first time it scrolls into view. */
function CountUp({ to }: { to: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!inView) return;
    const controls = animate(0, to, {
      duration: 1.2,
      ease: 'easeOut',
      onUpdate: (v) => setValue(Math.round(v)),
    });
    return () => controls.stop();
  }, [inView, to]);

  return <span ref={ref}>{value}</span>;
}

export default function Stats() {
  const gpu = useGpuReport();
  const fitCount = gpu
    ? MODELS.filter((m) => fitsTier(m, gpu.tier) === 'fits').length
    : null;

  return (
    <section className="bg-paper py-24">
      <div className="mx-auto max-w-[1340px] px-6">
        <motion.h2
          initial={{ opacity: 0, y: 14 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.6 }}
          className="mx-auto max-w-[24ch] text-center font-display text-[clamp(1.9rem,3.6vw,3rem)] leading-[1.08] tracking-[-0.01em] text-fg"
        >
          Probed. Matched. Trained. In the tab.
        </motion.h2>

        {/* stat row — hairline top + vertical dividers, Nexora-style */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.55, delay: 0.1 }}
          className="mx-auto mt-14 grid max-w-3xl grid-cols-3 divide-x divide-line border-t border-line pt-10"
        >
          <div className="px-4 text-center sm:px-8">
            <p className="font-display text-4xl leading-none text-fg sm:text-5xl">
              <CountUp to={MODELS.length} />
            </p>
            <p className="mt-3 text-[12px] leading-snug text-dim sm:text-[13px]">
              models in the roster
            </p>
          </div>
          <div className="px-4 text-center sm:px-8">
            <p className="font-display text-4xl leading-none text-fg sm:text-5xl">
              {fitCount === null ? '–' : <CountUp to={fitCount} />}
            </p>
            <p className="mt-3 text-[12px] leading-snug text-dim sm:text-[13px]">
              fit your probed GPU
            </p>
          </div>
          <div className="px-4 text-center sm:px-8">
            <p className="font-display text-4xl leading-none text-fg sm:text-5xl">0</p>
            <p className="mt-3 text-[12px] leading-snug text-dim sm:text-[13px]">
              bytes leave the machine
            </p>
          </div>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="mx-auto mt-12 max-w-2xl text-center text-[14.5px] leading-relaxed text-dim"
        >
          The roster is weighed against your adapter&rsquo;s real limits — buffer
          sizes, fp16 support, memory budgets — not marketing tier lists.
        </motion.p>

        {/* trusted strip — the probed machine, printed like a spec plate */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mx-auto mt-12 grid max-w-4xl grid-cols-2 divide-x divide-line rounded-2xl border border-line bg-card sm:grid-cols-4"
        >
          {(
            [
              ['adapter', gpu?.vendor ?? 'unprobed'],
              ['tier', gpu?.ok ? TIER_LABELS[gpu.tier] : '—'],
              ['params', '74M — 1.24B'],
              ['runtime', 'onnx · wgsl'],
            ] as const
          ).map(([k, v]) => (
            <div key={k} className="px-5 py-4">
              <p className="mb-1 font-mono text-[9.5px] uppercase tracking-[0.22em] text-faint">
                {k}
              </p>
              <p className="truncate font-mono text-[12px] uppercase tracking-[0.08em] text-fg">
                {v}
              </p>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
