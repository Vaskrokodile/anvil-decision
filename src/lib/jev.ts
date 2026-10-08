import {
  OpenJev,
  choice,
  score,
  noul,
  type Answer,
  type LoadProgress,
  type OpenJevInfo,
  type OpenJevRuntime,
  type Question,
} from 'open-jev';

export { choice, score, noul };
export type { Answer, LoadProgress, OpenJevInfo, OpenJevRuntime, Question };

/**
 * Runnable browser checkpoints — models open-jev can drive (families `kev` and
 * `open_jev`; the `gliner2` config section — e.g. GLiNER2.5-Decide-ONNX — needs
 * a dedicated encoder and is a later milestone).
 */
export const RUNNABLE: { id: string; label: string; note: string }[] = [
  {
    id: 'onnx-community/GLiNER2.5-Decide-ONNX',
    label: 'GLiNER2.5 Decide',
    note: 'fastino · ~0.87 GB fp16 · 512 ctx',
  },
  {
    id: 'kev-0.6b',
    label: 'kev 0.6B',
    note: 'qwen3-based · ~0.38 GB · 8k ctx',
  },
  {
    id: 'open-jev',
    label: 'open-jev',
    note: 'deberta-v3-large · ~0.49 GB · 512 ctx',
  },
  {
    id: 'kev-4b',
    label: 'kev 4B',
    note: 'most accurate · ~2.3 GB · discrete GPUs only',
  },
];

export function isJevRunnable(repo: string | undefined): boolean {
  return repo != null && RUNNABLE.some((r) => r.id === repo);
}

/** Whether a repo is the Decide ONNX export (driven by gliner2.ts, not open-jev). */
export function isDecideOnnx(repo: string | undefined): boolean {
  return repo === 'onnx-community/GLiNER2.5-Decide-ONNX';
}

let instance: OpenJev | null = null;
let instanceModel: string | null = null;

/** Metadata for a model config without downloading it (cache status + size). */
export function jevInfo(model: string): Promise<OpenJevInfo> {
  return OpenJev.info({ model, dtype: 'auto', device: 'auto' });
}

/**
 * Load (or reuse) a decision model. `auto` picks fp16 on WebGPU with
 * shader-f16, q4 elsewhere; `webgpu` device with wasm fallback.
 */
export async function loadJev(
  model: string,
  onProgress?: (p: LoadProgress) => void,
): Promise<OpenJev> {
  if (instance && instanceModel === model) return instance;
  if (instance) await instance.dispose().catch(() => undefined);
  instance = await OpenJev.load({
    model,
    dtype: 'auto',
    device: 'auto',
    onProgress,
  });
  instanceModel = model;
  return instance;
}

export async function disposeJev() {
  if (instance) await instance.dispose().catch(() => undefined);
  instance = null;
  instanceModel = null;
}
