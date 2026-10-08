// Real-weights parity check: CpuEncoder.forward vs transformers.js MiniLM.
// Run: node scripts/parity.mts
import { AutoTokenizer, pipeline } from '@huggingface/transformers';
import { CpuEncoder } from '../src/lib/xformer/cpu.ts';
import { loadMiniLMWeights } from '../src/lib/xformer/weights.ts';
import { MINILM_L6 } from '../src/lib/xformer/spec.ts';

const TEXTS = [
  'The cat sat on the mat.',
  'I was charged twice for the same order and want a refund.',
];

const weights = await loadMiniLMWeights();
console.log('weights loaded:', Object.keys(weights).length, 'tensors');

const tok = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

const enc = new CpuEncoder();
enc.init({ spec: MINILM_L6, weights, mode: 'lora', rank: 8, seed: 0, maxSeq: 128 });

for (const text of TEXTS) {
  const e = await tok(text, { padding: 'max_length', truncation: true, max_length: 128, return_tensor: false });
  const ids = Uint32Array.from((e.input_ids as number[]).map(Number));
  const mask = Float32Array.from((e.attention_mask as number[]).map(Number));
  const hidden = await enc.forward(ids, mask);

  // mean-pool over masked tokens + l2 normalize
  const d = MINILM_L6.dModel;
  const pooled = new Float32Array(d);
  let nTok = 0;
  for (let t = 0; t < 128; t++) {
    if (!mask[t]) continue;
    nTok++;
    for (let k = 0; k < d; k++) pooled[k] += hidden[t * d + k];
  }
  for (let k = 0; k < d; k++) pooled[k] /= nTok;
  let norm = 0;
  for (let k = 0; k < d; k++) norm += pooled[k] * pooled[k];
  norm = Math.sqrt(norm);
  for (let k = 0; k < d; k++) pooled[k] /= norm;

  const ref = (await pipe(text, { pooling: 'mean', normalize: true })).data as Float32Array;

  let dot = 0,
    maxAbs = 0;
  for (let k = 0; k < d; k++) {
    dot += pooled[k] * ref[k];
    maxAbs = Math.max(maxAbs, Math.abs(pooled[k] - ref[k]));
  }
  console.log(JSON.stringify({ text: text.slice(0, 30), cosSim: +dot.toFixed(6), maxAbsDiff: +maxAbs.toFixed(6) }));
}
