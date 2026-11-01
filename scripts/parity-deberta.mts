// Real-weights parity check: DebertaCpuEncoder (disentangled attention)
// vs transformers.js ONNX. The nli-deberta-v3-small ONNX export emits only
// NLI logits, so parity runs end-to-end: our encoder hidden[CLS] → pooler
// dense → classifier → logits, compared against the ONNX logits.
// Downloads ~565 MB safetensors + ~200 MB ONNX — network required.
// Run: node scripts/parity-deberta.mts
import { AutoTokenizer, AutoModel } from '@huggingface/transformers';
import { DebertaCpuEncoder } from '../src/lib/xformer/deberta.ts';
import { loadDebertaWeights, DEBERTA_TOKENIZER, DEBERTA_HEAD } from '../src/lib/xformer/weights.ts';
import { DEBERTA_V3_SMALL } from '../src/lib/xformer/spec.ts';

const TEXTS = [
  'The cat sat on the mat.',
  'I was charged twice for the same order and want a refund.',
];

const T = 32;
console.log('loading deberta weights (~565MB)…');
const weights = await loadDebertaWeights(undefined, Object.values(DEBERTA_HEAD));
console.log('weights loaded:', Object.keys(weights).length, 'tensors');

const tok = await AutoTokenizer.from_pretrained(DEBERTA_TOKENIZER);
const model = await AutoModel.from_pretrained(DEBERTA_TOKENIZER, { dtype: 'fp32' });

const enc = new DebertaCpuEncoder();
enc.init({ spec: DEBERTA_V3_SMALL, weights, mode: 'lora', rank: 8, seed: 0, maxSeq: T });

const d = DEBERTA_V3_SMALL.dModel;
const poolW = weights[DEBERTA_HEAD.poolW];
const poolB = weights[DEBERTA_HEAD.poolB];
const clfW = weights[DEBERTA_HEAD.clfW];
const clfB = weights[DEBERTA_HEAD.clfB];
const nCls = clfB.length;

const SQRT2 = Math.SQRT2;
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    t *
    (0.254829592 +
      t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-ax * ax));
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / SQRT2));

/** logits = classifier(gelu(pooler_dense(cls))) — ContextPooler act = gelu */
function ourLogits(cls: Float32Array): Float64Array {
  const pooled = new Float64Array(d);
  for (let o = 0; o < d; o++) {
    let acc = poolB[o];
    for (let k = 0; k < d; k++) acc += cls[k] * poolW[o * d + k];
    pooled[o] = gelu(acc);
  }
  const logits = new Float64Array(nCls);
  for (let c = 0; c < nCls; c++) {
    let acc = clfB[c];
    for (let k = 0; k < d; k++) acc += pooled[k] * clfW[c * d + k];
    logits[c] = acc;
  }
  return logits;
}

let worst = 0;
for (const text of TEXTS) {
  const e = await tok(text, {
    padding: 'max_length',
    truncation: true,
    max_length: T,
  });
  const ids = Uint32Array.from(e.input_ids.data as ArrayLike<number>, Number);
  const mask = Float32Array.from(e.attention_mask.data as ArrayLike<number>, Number);
  const hidden = await enc.forward(ids, mask);
  const ours = ourLogits(hidden.slice(0, d));

  const t = await model({ input_ids: e.input_ids, attention_mask: e.attention_mask });
  const ref = t.logits.data as Float32Array;

  let maxAbs = 0;
  for (let c = 0; c < nCls; c++) maxAbs = Math.max(maxAbs, Math.abs(ours[c] - ref[c]));
  worst = Math.max(worst, maxAbs);
  console.log(
    JSON.stringify({
      text: text.slice(0, 28),
      ours: [...ours].map((x) => +x.toFixed(4)),
      onnx: [...ref].map((x) => +x.toFixed(4)),
      maxAbsDiff: +maxAbs.toFixed(5),
    }),
  );
}
if (worst > 0.02) {
  console.log(`FAIL: max logit diff ${worst.toFixed(4)} > 0.02`);
  process.exit(1);
}
console.log('parity PASSED');
