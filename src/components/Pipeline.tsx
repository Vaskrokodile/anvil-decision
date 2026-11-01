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
    body: 'The roster provides hardware-fit estimates and inference candidates. It does not select the weights used by the current trainer.',
    detail: 'GLiNER2 Decide driver · JEV runtimes',
    live: true,
  },
  {
    n: '03',
    title: 'Build',
    body: 'Edit a sector template, import JSONL, CSV or text, or load a public Hugging Face dataset. Draft datasets are saved in this browser’s local database.',
    detail: '.jsonl · .csv · .txt · local IndexedDB',
    live: true,
  },
  {
    n: '04',
    title: 'Train',
    body: 'Train a choice head or fine-tune MiniLM-L6 with LoRA/full mode. These runs update MiniLM, not the selected roster checkpoint; validation uses held-out rows.',
    detail: 'CPU / WGSL · choice artifacts · .safetensors',
    live: true,
  },
];

export default function Pipeline() {
  return (
    <section id="pipeline" className="bg-paper py-24">
      <div className="mx-auto max-w-[1340px] px-6">
        <SectionHeader
          index="04 — pipeline"
          title="Local training, clear limits"
          lede="Probe hardware, choose an inference candidate, build a reusable local dataset, then train and evaluate a MiniLM choice classifier in the browser."
        />

        <ol className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s, i) => (
            <motion.li
              key={s.n}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.45, delay: i * 0.07 }}
              className="flex flex-col rounded-2xl border border-line bg-card p-7 transition-colors hover:bg-line-soft"
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
