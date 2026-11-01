import { AutoTokenizer, type PreTrainedTokenizer } from '@huggingface/transformers';
import type { EncoderBackend, EncoderInit } from './spec';
import type { FtArtifact } from './ftArtifact';
export { ftFromSafetensors } from './ftArtifact';
export type { FtArtifact } from './ftArtifact';
import { MINILM_L6, DEBERTA_V3_SMALL, poolHidden } from './spec';
import { loadMiniLMWeights, loadDebertaWeights, DEBERTA_TOKENIZER } from './weights';
import { CpuEncoder } from './cpu';
import { DebertaCpuEncoder } from './deberta';
import { WgslEncoder } from './wgsl';
import { parityCheckedEncoder } from './parity';

/**
 * A fine-tuned MiniLM artifact: LoRA adapters (or full weights) + linear head.
 * The inference path runs OUR encoder backend (not onnx) so the tuned params
 * actually apply.
 */
const ENCODER_ID = 'sentence-transformers/all-MiniLM-L6-v2';

let session: { enc: EncoderBackend; tok: PreTrainedTokenizer; artifact: FtArtifact } | null = null;
let store: FtArtifact | null = null;

export function setFtArtifact(a: FtArtifact | null) {
  store = a;
  if (!a) session = null;
}
export function getFtArtifact() {
  return store;
}

const specOf = (a: FtArtifact) => (a.arch === 'deberta' ? DEBERTA_V3_SMALL : MINILM_L6);

// weight downloads are ~90MB (minilm) / ~565MB (deberta) — cache per arch
const weightsCache: Record<string, Promise<Record<string, Float32Array>> | undefined> = {};

async function ensureSession(
  a: FtArtifact,
  preferWgsl: boolean,
  onWeights?: (frac: number) => void,
) {
  if (session && session.artifact === a) return session;
  if (session) session.enc.dispose?.();
  session = null;
  const arch = a.arch ?? 'minilm';
  const spec = specOf(a);
  const [weights, tok] = await Promise.all([
    (weightsCache[arch] ??= arch === 'deberta'
      ? loadDebertaWeights(onWeights)
      : loadMiniLMWeights(onWeights)),
    AutoTokenizer.from_pretrained(
      arch === 'deberta' ? DEBERTA_TOKENIZER : 'Xenova/all-MiniLM-L6-v2',
    ),
  ]);
  const init: EncoderInit = {
    spec,
    weights,
    mode: a.mode,
    rank: a.rank,
    seed: 0,
    maxSeq: Math.min(a.maxSeq, spec.maxPos),
    initParams: a.params,
  };
  let enc: EncoderBackend;
  if (!preferWgsl || arch === 'deberta') {
    // deberta: CPU-only until a WGSL disentangled-attention backend exists
    enc = arch === 'deberta' ? new DebertaCpuEncoder() : new CpuEncoder();
    enc.init(init);
  } else {
    try {
      const wgsl = await WgslEncoder.create(init);
      // same real-device parity gate as the training worker — a shader bug
      // that slipped through would corrupt inference just as badly
      const T = Math.min(a.maxSeq, spec.maxPos);
      const probe = await tok('anvil parity probe', {
        padding: 'max_length',
        truncation: true,
        max_length: T,
        return_tensor: false,
      });
      const ids = new Uint32Array(T);
      const mask = new Float32Array(T);
      for (let t = 0; t < T; t++) {
        ids[t] = Number((probe.input_ids as ArrayLike<number>)[t] ?? 0);
        mask[t] = Number((probe.attention_mask as ArrayLike<number>)[t] ?? 0);
      }
      enc = (await parityCheckedEncoder(wgsl, init, ids, mask)).enc;
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
  const spec = specOf(a);
  const { enc, tok } = await ensureSession(a, opts.preferWgsl !== false, opts.onWeights);
  const T = Math.min(a.maxSeq, spec.maxPos);
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
  const pooled = poolHidden(hidden, mask, T, spec.dModel, a.pool ?? 'cls');
  return headProbs(pooled, a.headW, a.headB, a.labels, spec.dModel);
}

function headProbs(
  cls: Float32Array,
  headW: Float32Array,
  headB: Float32Array,
  labels: string[],
  d: number,
): Record<string, number> {
  const c = labels.length;
  const logits = new Float32Array(c);
  for (let j = 0; j < c; j++) {
    let s = headB[j];
    for (let k = 0; k < d; k++) s += cls[k] * headW[k * c + j];
    logits[j] = s;
  }
  let mx = -Infinity;
  for (let j = 0; j < c; j++) mx = Math.max(mx, logits[j]);
  let sum = 0;
  for (let j = 0; j < c; j++) sum += Math.exp(logits[j] - mx);
  const out: Record<string, number> = {};
  for (let j = 0; j < c; j++) out[labels[j]] = Math.exp(logits[j] - mx) / sum;
  return out;
}

export interface FtTaskResult {
  name: string;
  labels: string[];
  probs: Record<string, number>;
}

/** Per-task probabilities — single-task artifacts yield one result. */
export async function inferFtTasks(
  text: string,
  a: FtArtifact,
  opts: { preferWgsl?: boolean; onWeights?: (f: number) => void } = {},
): Promise<FtTaskResult[]> {
  const tasks = a.tasks ?? [
    { name: 'label', labels: a.labels, headW: a.headW, headB: a.headB },
  ];
  const spec = specOf(a);
  const { enc, tok } = await ensureSession(a, opts.preferWgsl !== false, opts.onWeights);
  const T = Math.min(a.maxSeq, spec.maxPos);
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
  const pooled = poolHidden(hidden, mask, T, spec.dModel, a.pool ?? 'cls');
  return tasks.map((t) => ({
    name: t.name,
    labels: t.labels,
    probs: headProbs(pooled, t.headW, t.headB, t.labels, spec.dModel),
  }));
}

export const FT_ENCODER = ENCODER_ID;
