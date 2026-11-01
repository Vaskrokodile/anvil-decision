import type { TunerMode } from './spec.ts';

/** One named classification task — a dataset label column → its own head. */
export interface FtTask {
  name: string;
  labels: string[];
  headW: Float32Array;
  headB: Float32Array;
  valAcc?: number;
  valN?: number;
}

export interface FtArtifact {
  params: Record<string, Float32Array>;
  headW: Float32Array;
  headB: Float32Array;
  labels: string[];
  /** Multi-task artifacts: per-task heads. headW/headB/labels mirror task 0. */
  tasks?: FtTask[];
  /** encoder substrate — absent on pre-deberta artifacts */
  arch?: 'minilm' | 'deberta';
  mode: TunerMode;
  rank: number;
  backend: 'cpu' | 'wgsl';
  loss: number;
  acc: number;
  valAcc?: number;
  valN?: number;
  trainN?: number;
  rows: number;
  maxSeq: number;
  trainedAt: string;
}

export function ftFromSafetensors(
  tensors: Record<string, { shape: number[]; data: Float32Array | BigInt64Array }>,
  meta: Record<string, string>,
): FtArtifact {
  let tasks: FtTask[] | undefined;
  if (meta.tasks) {
    // multi-task export: head.<i>.weight/bias + a JSON task table in meta
    const table = JSON.parse(meta.tasks) as { name: string; labels: string[] }[];
    tasks = table.map((t, i) => {
      const w = tensors[`head.${i}.weight`];
      const b = tensors[`head.${i}.bias`];
      if (!w || !b) throw new Error(`safetensors missing head.${i} tensors for task "${t.name}"`);
      return {
        name: t.name,
        labels: t.labels,
        headW: w.data as Float32Array,
        headB: b.data as Float32Array,
      };
    });
  }
  const headW = tasks ? { data: tasks[0].headW } : tensors['head.weight'];
  const headB = tasks ? { data: tasks[0].headB } : tensors['head.bias'];
  if (!headW || !headB) throw new Error('safetensors missing head tensors');
  const isEncoderKey = (k: string) =>
    k.startsWith('encoder.layer.') || k.startsWith('embeddings.LayerNorm.');
  // canonical encoder keys only exist in full-mode exports — infer when meta is absent
  const mode: TunerMode =
    meta.mode === 'full' || (meta.mode == null && Object.keys(tensors).some(isEncoderKey))
      ? 'full'
      : 'lora';
  const params: Record<string, Float32Array> = {};
  for (const [k, v] of Object.entries(tensors)) {
    if (k.startsWith('lora.') || (mode === 'full' && isEncoderKey(k))) {
      params[k] = v.data as Float32Array;
    }
  }
  if (Object.keys(params).length === 0) {
    throw new Error(
      mode === 'full'
        ? 'safetensors has no encoder.layer.* tensors — not a full fine-tune artifact'
        : 'safetensors has no lora.* tensors — not a LoRA artifact',
    );
  }
  return {
    params,
    headW: headW.data as Float32Array,
    headB: headB.data as Float32Array,
    labels: tasks ? tasks[0].labels : (meta.labels ?? '').split('\n').filter(Boolean),
    tasks,
    arch: meta.arch === 'deberta' ? 'deberta' : undefined,
    mode,
    rank: Number(meta.rank ?? 8),
    backend: 'cpu',
    loss: Number(meta.loss ?? 0),
    acc: Number(meta.acc ?? 0),
    valAcc: meta.valAcc != null ? Number(meta.valAcc) : undefined,
    valN: meta.valN != null ? Number(meta.valN) : undefined,
    trainN: meta.trainN != null ? Number(meta.trainN) : undefined,
    rows: Number(meta.rows ?? 0),
    maxSeq: Number(meta.maxSeq ?? 128),
    trainedAt: meta.trainedAt ?? '',
  };
}
