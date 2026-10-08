import { motion } from 'motion/react';
import { SECTORS } from '../data/sectors';

export default function Sectors() {
  return (
    <section id="sectors" className="bg-paper py-24">
      <div className="mx-auto max-w-[1340px] px-6">
        {/* split header — serif left, dim lede right */}
        <div className="mb-14 grid gap-8 lg:grid-cols-[1.2fr_1fr] lg:items-end">
          <div>
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true, margin: '-80px' }}
              className="mb-5 font-mono text-[10.5px] uppercase tracking-[0.28em] text-faint"
            >
              [ 02 — sectors ]
            </motion.p>
            <motion.h2
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.55 }}
              className="max-w-[18ch] font-display text-[clamp(2rem,4.4vw,3.6rem)] leading-[1.04] tracking-[-0.015em] text-fg"
            >
              Datasets shaped by sector
            </motion.h2>
          </div>
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.5, delay: 0.12 }}
            className="max-w-[46ch] text-[14px] leading-relaxed text-dim lg:justify-self-end"
          >
            Each sector ships a schema — the fields your records need — plus a
            starter template and the models that fit it. Bring your own and it
            becomes a sixth.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
          {SECTORS.map((s, i) => (
            <motion.article
              key={s.id}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.45, delay: i * 0.05 }}
              className="card-aurora flex flex-col rounded-2xl border border-line bg-card p-6"
            >
              <p className="mb-8 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
                [ {s.code} ]
              </p>
              <h3 className="mb-2.5 text-lg font-medium tracking-[-0.02em] text-fg">
                {s.name}
              </h3>
              <p className="mb-6 flex-1 text-[13px] leading-relaxed text-dim">
                {s.desc}
              </p>

              <div className="mb-5 flex flex-wrap gap-1.5">
                {s.schema.map((f) => (
                  <span
                    key={f.field}
                    className={`rounded-full border px-2.5 py-0.5 font-mono text-[10px] tracking-wide ${
                      f.role === 'label'
                        ? 'border-blue/60 text-blue'
                        : 'border-line text-dim'
                    }`}
                  >
                    {f.field}
                  </span>
                ))}
              </div>

              <div className="border-t border-line-soft pt-4">
                <p className="font-mono text-[10px] leading-relaxed text-faint">
                  {s.examples.join('  ·  ')}
                </p>
              </div>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
