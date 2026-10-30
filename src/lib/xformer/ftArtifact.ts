import type { TunerMode } from './spec.ts';

export interface FtArtifact {
  params: Record<string, Float32Array>;
  headW: Float32Array;
  headB: Float32Array;
  labels: string[];
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
  const headW = tensors['head.weight'];
  const headB = tensors['head.bias'];
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
    labels: (meta.labels ?? '').split('\n').filter(Boolean),
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
