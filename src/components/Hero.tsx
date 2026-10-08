import { lazy, Suspense } from 'react';
import { motion } from 'motion/react';
import { useGpuReport } from '../lib/useGpu';
import { TIER_LABELS } from '../data/models';

const AuroraCanvas = lazy(() => import('./AuroraCanvas'));

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 },
};

export default function Hero({ onStudio }: { onStudio: () => void }) {
  const gpu = useGpuReport();

  const adapter =
    gpu === null
      ? 'probing…'
      : gpu.ok
        ? `${gpu.vendor ?? 'gpu'} · ${TIER_LABELS[gpu.tier]}`
        : 'no webgpu';

  const specs: [string, string][] = [
    ['adapter', adapter],
    ['params', '74M — 1.24B'],
    ['runtime', 'onnx · wgsl'],
    ['data path', 'local only'],
  ];

  return (
    <section className="relative min-h-svh overflow-hidden bg-abyss text-ice">
      {/* aurora scene — back to front: sky, dome, planet, stars, dot grid */}
      <div className="aurora-sky absolute inset-0" aria-hidden="true" />
      <div className="aurora-dome absolute inset-0" aria-hidden="true" />
      <div className="planet absolute inset-0" aria-hidden="true" />
      <Suspense fallback={null}>
        <AuroraCanvas className="absolute inset-0" />
      </Suspense>
      <div
        className="dot-matrix pointer-events-none absolute inset-0"
        aria-hidden="true"
      />

      {/* content — pointer-events pass through to the star layer except on CTAs */}
      <div className="pointer-events-none relative z-10 flex min-h-svh flex-col pt-16">
        <div className="mx-auto flex w-full max-w-[1440px] flex-1 flex-col items-center justify-center px-6 py-16 text-center">
          <motion.p
            {...fadeUp}
            transition={{ duration: 0.5 }}
            className="font-mono text-[11px] uppercase tracking-[0.28em] text-ice-dim"
          >
            [ in-browser fine-tuning · webgpu ]
          </motion.p>

          <motion.h1
            {...fadeUp}
            transition={{ duration: 0.6, delay: 0.08 }}
            className="mt-7 max-w-[16ch] font-display text-[clamp(2.8rem,6.5vw,5.5rem)] leading-[1.02] tracking-[-0.02em] text-ice"
          >
            Train small models on your{' '}
            <span className="serif-i text-blue-bright">own metal.</span>
          </motion.h1>

          <motion.p
            {...fadeUp}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="mt-7 max-w-[52ch] text-[15px] leading-relaxed text-ice-dim sm:text-[16px]"
          >
            Anvil probes your GPU, matches it to a roster of sub-1B specialist
            models — decision heads, extractors, guardrails — and fine-tunes them
            in the browser. Your data never leaves the machine.
          </motion.p>

          <motion.div
            {...fadeUp}
            transition={{ duration: 0.6, delay: 0.24 }}
            className="pointer-events-auto mt-10 flex flex-wrap items-center justify-center gap-3"
          >
            <button
              type="button"
              onClick={onStudio}
              className="flex items-center gap-2.5 rounded-full bg-ice px-7 py-3.5 font-mono text-[12px] font-medium uppercase tracking-[0.14em] text-abyss transition-opacity hover:opacity-85"
            >
              Open the Studio
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-abyss" />
            </button>
            <a
              href="#roster"
              className="rounded-full border border-ice/25 px-7 py-3.5 font-mono text-[12px] uppercase tracking-[0.14em] text-ice transition-colors hover:border-ice/60"
            >
              Browse the roster
            </a>
          </motion.div>

          {/* scroll cue — vertical hairline fading toward the planet */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.55 }}
            className="mt-12 h-14 w-px bg-gradient-to-b from-ice/40 to-transparent"
          />
        </div>

        {/* spec strip — full-bleed hairline, probed values */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.4 }}
          className="border-t border-ice-line"
        >
          <div className="mx-auto grid w-full max-w-[1440px] grid-cols-2 gap-x-6 gap-y-5 px-6 py-6 sm:grid-cols-4">
            {specs.map(([k, v]) => (
              <div key={k}>
                <p className="mb-1.5 font-mono text-[9.5px] uppercase tracking-[0.24em] text-ice-faint">
                  {k}
                </p>
                <p className="truncate font-mono text-[12px] uppercase tracking-[0.1em] text-ice">
                  {v}
                </p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
