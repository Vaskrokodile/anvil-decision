import { motion } from 'motion/react';
import { SectionHeader, Tag } from './ui';

const STEPS = [
  {
    n: '01',
    title: 'Probe',
    body: 'navigator.gpu hands over the adapter: vendor, architecture, buffer limits, fp16 support. No benchmark theatre — we read the real limits and tier the machine.',
    detail: 'adapter.info · adapter.limits · shader-f16',
    live: true,
  },
  {
    n: '02',
    title: 'Select',
    body: 'The roster is gated by your tier. Models you can’t train are marked, not hidden — the honest ceiling is part of the product.',
    detail: 'params ≤ 1.24B · LoRA / head / full',
    live: true,
  },
  {
    n: '03',
    title: 'Build',
    body: 'Sector templates give you the schema; import JSONL, CSV or text and it never leaves the tab. Datasets persist to OPFS.',
    detail: '.jsonl · .csv · .txt → OPFS',
    live: true,
  },
  {
    n: '04',
    title: 'Train',
    body: 'Engine v1 trains a linear head over frozen MiniLM embeddings — WGSL compute in a Worker, CPU fallback, exports .safetensors. LoRA on the encoder is next.',
    detail: 'wgsl/CPU head · adamw · .safetensors',
    live: true,
  },
];

export default function Pipeline() {
  return (
    <section id="pipeline" className="bg-paper py-24">
      <div className="mx-auto max-w-[1340px] px-6">
        <SectionHeader
          index="04 — pipeline"
          title="Four steps, no servers"
          lede="What’s live today vs. what ships next is printed on the tin. Probe, select and dataset-building work now; the WGSL training engine is the build in progress."
        />

        <ol className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s, i) => (
            <motion.li
              key={s.n}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.45, delay: i * 0.07 }}
              className="flex flex-col rounded-2xl border border-line bg-card p-7 transition-colors hover:bg-white"
            >
              <div className="mb-12 flex items-center justify-between">
                <span className="font-mono text-[11px] tracking-widest text-faint">
                  [ {s.n} ]
                </span>
                <Tag tone={s.live ? 'solid' : 'faint'}>{s.live ? 'live' : 'next'}</Tag>
              </div>
              <h3 className="mb-3 text-xl font-medium tracking-[-0.02em] text-fg">{s.title}</h3>
              <p className="mb-6 flex-1 text-[13.5px] leading-relaxed text-dim">{s.body}</p>
              <p className="border-t border-line-soft pt-4 font-mono text-[10px] uppercase tracking-[0.15em] text-faint">
                {s.detail}
              </p>
            </motion.li>
          ))}
        </ol>
      </div>
    </section>
  );
}
