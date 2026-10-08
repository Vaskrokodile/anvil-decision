import { useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Check, Circle } from '@phosphor-icons/react';
import { useGpuReport } from '../lib/useGpu';
import { MODELS } from '../data/models';
import type { Dataset } from '../lib/dataset';
import type { TrainMode } from '../data/models';
import ProbeStep from './ProbeStep';
import ModelStep from './ModelStep';
import DatasetStep from './DatasetStep';
import RunStep from './RunStep';
import ConfigStep from './ConfigStep';

export interface TrainConfig {
  mode: TrainMode;
  lr: number;
  epochs: number;
  batch: number;
  seqLen: number;
}

const STEPS = ['Probe', 'Model', 'Dataset', 'Run', 'Configure'] as const;

export default function Studio() {
  const report = useGpuReport();
  const [step, setStep] = useState(0);
  const [modelId, setModelId] = useState<string | null>(null);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [ranOnce, setRanOnce] = useState(false);
  const [config, setConfig] = useState<TrainConfig>({
    mode: 'lora',
    lr: 1e-4,
    epochs: 3,
    batch: 8,
    seqLen: 512,
  });

  const model = useMemo(() => MODELS.find((m) => m.hf === modelId) ?? null, [modelId]);

  const done = [
    report?.ok === true,
    model !== null,
    dataset !== null && dataset.rows.length > 0,
    ranOnce,
    model !== null && dataset !== null,
  ];

  return (
    <div className="mx-auto flex min-h-svh w-full max-w-[1400px] px-3 pb-6 pt-24 sm:px-4">
      {/* step rail */}
      <aside className="sticky top-24 hidden h-fit w-52 shrink-0 flex-col overflow-hidden rounded-2xl border border-line bg-card md:flex">
        <p className="border-b border-line px-4 py-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">
          Studio / job
        </p>
        {STEPS.map((label, i) => {
          const active = i === step;
          return (
            <button
              key={label}
              onClick={() => setStep(i)}
              className={`flex items-center gap-3 border-b border-line-soft px-4 py-3.5 text-left last:border-0 ${
                active ? 'bg-paper text-fg' : 'text-dim hover:bg-paper/60 hover:text-fg'
              }`}
            >
              {done[i] ? (
                <Check size={12} className="shrink-0 text-blue" weight="bold" />
              ) : (
                <Circle
                  size={9}
                  weight="fill"
                  className={`shrink-0 ${active ? 'text-blue' : 'text-faint'}`}
                />
              )}
              <span className="font-mono text-[11px] uppercase tracking-[0.15em]">{label}</span>
              <span className="ml-auto font-mono text-[10px] text-faint">0{i + 1}</span>
            </button>
          );
        })}
        <div className="px-4 py-3">
          <p className="font-mono text-[10px] leading-relaxed text-faint">
            Everything runs client-side. Closing the tab keeps cached weights; datasets stay in
            OPFS.
          </p>
        </div>
      </aside>

      {/* active panel */}
      <main className="min-w-0 flex-1 rounded-2xl border border-line bg-card md:ml-3">
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.22 }}
          >
            {step === 0 && <ProbeStep report={report} onNext={() => setStep(1)} />}
            {step === 1 && (
              <ModelStep
                report={report}
                modelId={modelId}
                onPick={setModelId}
                onNext={() => setStep(2)}
              />
            )}
            {step === 2 && (
              <DatasetStep
                dataset={dataset}
                onDataset={setDataset}
                onNext={() => setStep(3)}
              />
            )}
            {step === 3 && (
              <RunStep
                model={model}
                dataset={dataset}
                onNext={() => setStep(4)}
                onRan={() => setRanOnce(true)}
              />
            )}
            {step === 4 && (
              <ConfigStep
                report={report}
                model={model}
                dataset={dataset}
                config={config}
                onConfig={setConfig}
                onTrained={() => setStep(3)}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
