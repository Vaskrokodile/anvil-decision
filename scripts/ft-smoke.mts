// CPU fine-tune smoke: real MiniLM weights + LoRA r4 + CLS head, verify loss drops.
// Run: node scripts/ft-smoke.mts
import { AutoTokenizer } from '@huggingface/transformers';
import { CpuEncoder } from '../src/lib/xformer/cpu.ts';
import { loadMiniLMWeights } from '../src/lib/xformer/weights.ts';
import { MINILM_L6 } from '../src/lib/xformer/spec.ts';

const ROWS: [string, number][] = [
  ['I love this product, it works perfectly', 0],
  ['absolutely fantastic experience, highly recommend', 0],
  ['great quality and fast shipping', 0],
  ['really happy with my purchase', 0],
  ['excellent service, will buy again', 0],
  ['superb quality, exceeded expectations', 0],
  ['wonderful item, very satisfied', 0],
  ['best purchase I made this year', 0],
  ['terrible product, broke immediately', 1],
  ['awful quality, complete waste of money', 1],
  ['very disappointed, does not work', 1],
  ['horrible experience, never again', 1],
  ['worst item ever, totally useless', 1],
  ['poor quality, regret buying', 1],
  ['broken on arrival, want a refund', 1],
  ['junk, do not buy this', 1],
];
const T = 32, D = 384, C = 2, LR = 5e-3, EPOCHS = 6;

const weights = await loadMiniLMWeights();
const tok = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
const enc = new CpuEncoder();
enc.init({ spec: MINILM_L6, weights, mode: 'lora', rank: 4, seed: 1, maxSeq: T });

// head
const w = new Float32Array(D * C).map(() => (Math.random() - 0.5) * 0.05);
const b = new Float32Array(C);
const gw = new Float32Array(D * C), gb = new Float32Array(C);

const seqs = ROWS.map(([t]) => {
  const e = tok(t, { padding: 'max_length', truncation: true, max_length: T, return_tensor: false });
  return {
    ids: Uint32Array.from((e.input_ids as number[]).map(Number)),
    mask: Float32Array.from((e.attention_mask as number[]).map(Number)),
  };
});

const logits = new Float32Array(C), dz = new Float32Array(C);
for (let ep = 0; ep < EPOCHS; ep++) {
  let loss = 0, hit = 0;
  const order = [...ROWS.keys()];
  for (let off = 0; off < order.length; off += 4) {
    for (const r of order.slice(off, off + 4)) {
      const hidden = await enc.forward(seqs[r].ids, seqs[r].mask);
      const cls = hidden.slice(0, D);
      for (let j = 0; j < C; j++) {
        let s = b[j];
        for (let k = 0; k < D; k++) s += cls[k] * w[k * C + j];
        logits[j] = s;
      }
      const y = ROWS[r][1];
      const mx = Math.max(...logits);
      let sum = 0; for (let j = 0; j < C; j++) sum += Math.exp(logits[j] - mx);
      const p1 = Math.exp(logits[y] - mx) / sum;
      loss += -Math.log(p1);
      if (logits[1 - y] < logits[y]) hit++;
      for (let j = 0; j < C; j++) dz[j] = Math.exp(logits[j] - mx) / sum - (j === y ? 1 : 0);
      const dcls = new Float32Array(D);
      for (let j = 0; j < C; j++) { gb[j] += dz[j]; for (let k = 0; k < D; k++) { gw[k * C + j] += cls[k] * dz[j]; dcls[k] += dz[j] * w[k * C + j]; } }
      const hg = new Float32Array(T * D);
      hg.set(dcls, 0);
      await enc.backward(hg);
    }
    await enc.applyGrads(LR);
    // head sgd step (plain, fine for smoke)
    for (let p = 0; p < D * C; p++) { w[p] -= (LR * gw[p]) / 4; gw[p] = 0; }
    for (let j = 0; j < C; j++) { b[j] -= (LR * gb[j]) / 4; gb[j] = 0; }
  }
  console.log(`epoch ${ep + 1}  loss ${(loss / ROWS.length).toFixed(3)}  acc ${(hit / ROWS.length).toFixed(2)}`);
}
