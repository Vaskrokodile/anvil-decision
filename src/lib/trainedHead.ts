import { embedTexts, EMBED_DIM } from './embed';
import { softmaxCE } from './head';

/**
 * The artifact the studio actually trains: a linear head over frozen MiniLM
 * embeddings, plus the label set. Kept in-module for the session; exportable
 * as safetensors, reloadable.
 */
export interface TrainedHead {
  w: Float32Array; // [d, c]
  b: Float32Array; // [c]
  labels: string[];
  encoder: string;
  backend: 'cpu' | 'wgsl';
  loss: number;
  acc: number;
  /** held-out accuracy at end of training — undefined for older artifacts */
  valAcc?: number;
  valN?: number;
  trainN?: number;
  trainedAt: string;
  rows: number;
}

let current: TrainedHead | null = null;

export function setTrainedHead(h: TrainedHead | null) {
  current = h;
}
export function getTrainedHead() {
  return current;
}

export function headFromSafetensors(
  tensors: Record<string, { shape: number[]; data: Float32Array | BigInt64Array }>,
  meta: Record<string, string>,
): TrainedHead {
  const w = tensors['head.weight'];
  const b = tensors['head.bias'];
  if (!w || !b) throw new Error('safetensors missing head.weight / head.bias');
  return {
    w: w.data as Float32Array,
    b: b.data as Float32Array,
    labels: (meta.labels ?? '').split('\n').filter(Boolean),
    encoder: meta.encoder ?? 'unknown',
    backend: 'cpu',
    loss: Number(meta.loss ?? 0),
    acc: Number(meta.acc ?? 0),
    valAcc: meta.valAcc != null ? Number(meta.valAcc) : undefined,
    valN: meta.valN != null ? Number(meta.valN) : undefined,
    trainN: meta.trainN != null ? Number(meta.trainN) : undefined,
    trainedAt: meta.trainedAt ?? '',
    rows: Number(meta.rows ?? 0),
  };
}

/** Embed one text and run the trained head → per-label probabilities. */
export async function inferHead(
  text: string,
  head: TrainedHead,
): Promise<Record<string, number>> {
  const x = await embedTexts([text]);
  const c = head.labels.length;
  const logits = new Float32Array(c);
  for (let j = 0; j < c; j++) {
    let s = head.b[j];
    for (let k = 0; k < EMBED_DIM; k++) s += x[k] * head.w[k * c + j];
    logits[j] = s;
  }
  // softmax
  let max = -Infinity;
  for (let j = 0; j < c; j++) max = Math.max(max, logits[j]);
  let sum = 0;
  for (let j = 0; j < c; j++) sum += Math.exp(logits[j] - max);
  const out: Record<string, number> = {};
  for (let j = 0; j < c; j++) out[head.labels[j]] = Math.exp(logits[j] - max) / sum;
  return out;
}

/** Per-row loss helper (used by tests). */
export function headLoss(logits: Float32Array, label: number, c: number) {
  return softmaxCE(logits, 0, c, label);
}
