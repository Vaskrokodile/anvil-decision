import { AutoTokenizer, type PreTrainedTokenizer } from '@huggingface/transformers';
import type { EncoderBackend, EncoderInit } from './spec';
import type { FtArtifact } from './ftArtifact';
export { ftFromSafetensors } from './ftArtifact';
export type { FtArtifact } from './ftArtifact';
import { MINILM_L6 } from './spec';
import { loadMiniLMWeights } from './weights';
import { CpuEncoder } from './cpu';
import { WgslEncoder } from './wgsl';

/**
 * A fine-tuned MiniLM artifact: LoRA adapters (or full weights) + linear head.
 * The inference path runs OUR encoder backend (not onnx) so the tuned params
 * actually apply.
 */
const ENCODER_ID = 'sentence-transformers/all-MiniLM-L6-v2';

let session: { enc: EncoderBackend; tok: PreTrainedTokenizer; artifact: FtArtifact } | null = null;
let store: FtArtifact | null = null;
let weightsCache: Promise<Record<string, Float32Array>> | null = null;

export function setFtArtifact(a: FtArtifact | null) {
  store = a;
  if (!a) session = null;
}
export function getFtArtifact() {
  return store;
}

async function ensureSession(
  a: FtArtifact,
  preferWgsl: boolean,
  onWeights?: (frac: number) => void,
) {
  if (session && session.artifact === a) return session;
  if (session) session.enc.dispose?.();
  session = null;
  const [weights, tok] = await Promise.all([
    (weightsCache ??= loadMiniLMWeights(onWeights)),
    AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2'),
  ]);
  const init: EncoderInit = {
    spec: MINILM_L6,
    weights,
    mode: a.mode,
    rank: a.rank,
    seed: 0,
    maxSeq: Math.min(a.maxSeq, MINILM_L6.maxPos),
    initParams: a.params,
  };
  let enc: EncoderBackend;
  if (!preferWgsl) {
    enc = new CpuEncoder();
    enc.init(init);
  } else {
    try {
      enc = await WgslEncoder.create(init);
    } catch {
      enc = new CpuEncoder();
      enc.init(init);
    }
  }
  session = { enc, tok, artifact: a };
  return session;
}

/** Run the tuned encoder + head on text → per-label probabilities. */
export async function inferFt(
  text: string,
  a: FtArtifact,
  opts: { preferWgsl?: boolean; onWeights?: (f: number) => void } = {},
): Promise<Record<string, number>> {
  const { enc, tok } = await ensureSession(a, opts.preferWgsl !== false, opts.onWeights);
  const T = Math.min(a.maxSeq, MINILM_L6.maxPos);
  const e = await tok(text, {
    padding: 'max_length',
    truncation: true,
    max_length: T,
    return_tensor: false,
  });
  const ids = new Uint32Array(T);
  const mask = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    ids[t] = Number((e.input_ids as ArrayLike<number>)[t] ?? 0);
    mask[t] = Number((e.attention_mask as ArrayLike<number>)[t] ?? 0);
  }
  const hidden = await enc.forward(ids, mask);
  const c = a.labels.length;
  const d = MINILM_L6.dModel;
  const logits = new Float32Array(c);
  for (let j = 0; j < c; j++) {
    let s = a.headB[j];
    for (let k = 0; k < d; k++) s += hidden[k] * a.headW[k * c + j];
    logits[j] = s;
  }
  let mx = -Infinity;
  for (let j = 0; j < c; j++) mx = Math.max(mx, logits[j]);
  let sum = 0;
  for (let j = 0; j < c; j++) sum += Math.exp(logits[j] - mx);
  const out: Record<string, number> = {};
  for (let j = 0; j < c; j++) out[a.labels[j]] = Math.exp(logits[j] - mx) / sum;
  return out;
}

export const FT_ENCODER = ENCODER_ID;
