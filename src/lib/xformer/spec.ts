/**
 * Shared contract for the in-browser transformer encoder — the thing LoRA
 * and full fine-tune actually backprop through. Two backends implement this:
 * a CPU reference (testable in Node) and a WGSL compute backend.
 *
 * Model: sentence-transformers/all-MiniLM-L6-v2 — BERT encoder, 6 layers,
 * d=384, 12 heads × 32, FFN 1536, vocab 30522, max pos 512, GELU (erf).
 */

export interface XformerSpec {
  layers: number;
  dModel: number;
  heads: number;
  dHead: number;
  dFfn: number;
  vocab: number;
  maxPos: number;
  lnEps: number;
}

export const MINILM_L6: XformerSpec = {
  layers: 6,
  dModel: 384,
  heads: 12,
  dHead: 32,
  dFfn: 1536,
  vocab: 30522,
  maxPos: 512,
  lnEps: 1e-12,
};

/** flat name → F32 data, straight out of model.safetensors */
export type Weights = Record<string, Float32Array>;

/** Canonical safetensors key names for BERT checkpoints. */
export const W = {
  word: 'embeddings.word_embeddings.weight', // [vocab, d]
  pos: 'embeddings.position_embeddings.weight', // [maxPos, d]
  tokType: 'embeddings.token_type_embeddings.weight', // [2, d]
  embLnW: 'embeddings.LayerNorm.weight',
  embLnB: 'embeddings.LayerNorm.bias',
  layer: (i: number) => ({
    qW: `encoder.layer.${i}.attention.self.query.weight`, // [d, d] (out,in)
    qB: `encoder.layer.${i}.attention.self.query.bias`,
    kW: `encoder.layer.${i}.attention.self.key.weight`,
    kB: `encoder.layer.${i}.attention.self.key.bias`,
    vW: `encoder.layer.${i}.attention.self.value.weight`,
    vB: `encoder.layer.${i}.attention.self.value.bias`,
    oW: `encoder.layer.${i}.attention.output.dense.weight`,
    oB: `encoder.layer.${i}.attention.output.dense.bias`,
    ln1W: `encoder.layer.${i}.attention.output.LayerNorm.weight`,
    ln1B: `encoder.layer.${i}.attention.output.LayerNorm.bias`,
    f1W: `encoder.layer.${i}.intermediate.dense.weight`, // [ffn, d]
    f1B: `encoder.layer.${i}.intermediate.dense.bias`,
    f2W: `encoder.layer.${i}.output.dense.weight`, // [d, ffn]
    f2B: `encoder.layer.${i}.output.dense.bias`,
    ln2W: `encoder.layer.${i}.output.LayerNorm.weight`,
    ln2B: `encoder.layer.${i}.output.LayerNorm.bias`,
  }),
};

export type TunerMode = 'lora' | 'full';

/**
 * Reconstruct a trainable param's shape from its name — needed because
 * `trainableWeights()` returns flat arrays (safetensors export + reload).
 * lora names: `lora.<i>.<q|k|v|o|f1|f2>.a|b` (a: [in,r], b: [r,out]).
 * full names: the canonical W.* keys.
 */
export function paramShape(name: string, spec: XformerSpec, rank: number): number[] {
  const d = spec.dModel;
  const f = spec.dFfn;
  if (name.startsWith('lora.')) {
    const proj = name.split('.')[2];
    const [inDim, outDim] = proj === 'f1' ? [d, f] : proj === 'f2' ? [f, d] : [d, d];
    return name.endsWith('.a') ? [inDim, rank] : [rank, outDim];
  }
  if (name.includes('intermediate.dense.weight')) return [f, d];
  if (name.includes('output.dense.weight')) return name.includes('attention') ? [d, d] : [d, f];
  if (name.includes('attention.self') && name.endsWith('weight')) return [d, d];
  if (name.includes('embeddings')) return name.endsWith('weight') ? [d] : [d];
  if (name.includes('intermediate')) return [f]; // f1 bias
  return [d]; // other biases + LayerNorm
}

export interface EncoderInit {
  spec: XformerSpec;
  weights: Weights;
  mode: TunerMode;
  /** LoRA rank (ignored for 'full') */
  rank: number;
  seed: number;
  /** max padded sequence length the backend must support */
  maxSeq: number;
  /** pre-trained trainable params to load at init (adapter reload for inference) */
  initParams?: Record<string, Float32Array>;
}

export interface EncoderBackend {
  kind: 'cpu' | 'wgsl';
  init(i: EncoderInit): Promise<void> | void;
  /**
   * Forward one padded sequence. ids/mask length T (≤ maxSeq).
   * Returns hidden states [T, dModel] (row-major).
   */
  forward(inputIds: Uint32Array, mask: Float32Array): Promise<Float32Array>;
  /**
   * Backprop one sequence: takes dL/d(hidden) [T, dModel] (zeros except the
   * positions that carry loss — e.g. CLS only for classification) and
   * accumulates parameter gradients internally.
   */
  backward(hiddenGrad: Float32Array): Promise<void>;
  /** AdamW step on all trainable params using accumulated grads; zeroes grads. */
  applyGrads(lr: number): Promise<void>;
  /** Snapshot the trainable tensors (LoRA A/B pairs, or all weights in full). */
  trainableWeights(): Promise<Record<string, Float32Array>>;
  dispose?(): void;
}
