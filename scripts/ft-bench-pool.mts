/**
 * ft-bench-pool.mts — real-data LoRA bench: CLS vs mean pooling.
 * Trains MiniLM-L6 + LoRA on the bundled emotion dataset twice with identical
 * seeds and reports held-out accuracy for each pooling strategy.
 * Run: node scripts/ft-bench-pool.mts   (downloads ~91MB weights once)
 */
import { readFileSync } from 'node:fs';
import { AutoTokenizer } from '@huggingface/transformers';
import { CpuEncoder } from '../src/lib/xformer/cpu.ts';
import { loadMiniLMWeights } from '../src/lib/xformer/weights.ts';
import { MINILM_L6 } from '../src/lib/xformer/spec.ts';
import { softmaxCE } from '../src/lib/head.ts';
import { mulberry, stratifiedSplit } from '../src/lib/split.ts';

const T = 64;
const D = MINILM_L6.dModel;
const LR = 5e-3;
const EPOCHS = 4;
const RANK = 8;
const SEED = 1337;

// --- data ----------------------------------------------------------------
const rows = readFileSync('public/datasets/emotion.jsonl', 'utf8')
  .split('\n')
  .filter(Boolean)
  .map((l) => JSON.parse(l) as { text: string; label: string });
const labelSet = [...new Set(rows.map((r) => r.label))];
const y = Uint32Array.from(rows.map((r) => labelSet.indexOf(r.label)));
const C = labelSet.length;
console.log(`${rows.length} rows, ${C} classes: ${labelSet.join(', ')}`);

const { train, val } = stratifiedSplit(y, 0.2, SEED);
console.log(`train ${train.length} / val ${val.length}`);

// --- head (mirrors ft.worker Head) ----------------------------------------
class Head {
  w: Float32Array;
  b = new Float32Array(C);
  gw = new Float32Array(D * C);
  gb = new Float32Array(C);
  mw = new Float32Array(D * C);
  vw = new Float32Array(D * C);
  mb = new Float32Array(C);
  vb = new Float32Array(C);
  t = 0;
  constructor(rng: () => number) {
    const scale = Math.sqrt(2 / (D + C));
    this.w = new Float32Array(D * C).map(() => (rng() * 2 - 1) * scale);
  }
  logits(x: Float32Array, out: Float32Array) {
    for (let j = 0; j < C; j++) {
      let s = this.b[j];
      for (let k = 0; k < D; k++) s += x[k] * this.w[k * C + j];
      out[j] = s;
    }
  }
  backward(x: Float32Array, dz: Float32Array) {
    const dx = new Float32Array(D);
    for (let j = 0; j < C; j++) {
      this.gb[j] += dz[j];
      for (let k = 0; k < D; k++) {
        this.gw[k * C + j] += x[k] * dz[j];
        dx[k] += dz[j] * this.w[k * C + j];
      }
    }
    return dx;
  }
  apply(lr: number, n: number) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.t++;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    const inv = 1 / n;
    for (let p = 0; p < D * C; p++) {
      const g = this.gw[p] * inv;
      this.mw[p] = b1 * this.mw[p] + (1 - b1) * g;
      this.vw[p] = b2 * this.vw[p] + (1 - b2) * g * g;
      this.w[p] -= (lr * this.mw[p]) / (bc1 * (Math.sqrt(this.vw[p] / bc2) + eps));
      this.gw[p] = 0;
    }
    for (let j = 0; j < C; j++) {
      const g = this.gb[j] * inv;
      this.mb[j] = b1 * this.mb[j] + (1 - b1) * g;
      this.vb[j] = b2 * this.vb[j] + (1 - b2) * g * g;
      this.b[j] -= (lr * this.mb[j]) / (bc1 * (Math.sqrt(this.vb[j] / bc2) + eps));
      this.gb[j] = 0;
    }
  }
}

const pool = (hidden: Float32Array, mask: Float32Array, mean: boolean): Float32Array => {
  if (!mean) return hidden.slice(0, D);
  const out = new Float32Array(D);
  let n = 0;
  for (let t = 0; t < T; t++) {
    if (!mask[t]) continue;
    n++;
    for (let k = 0; k < D; k++) out[k] += hidden[t * D + k];
  }
  for (let k = 0; k < D; k++) out[k] /= Math.max(1, n);
  return out;
};
const poolGrad = (dx: Float32Array, mask: Float32Array, mean: boolean): Float32Array => {
  const hg = new Float32Array(T * D);
  if (!mean) {
    hg.set(dx, 0);
    return hg;
  }
  let n = 0;
  for (let t = 0; t < T; t++) n += mask[t];
  const inv = 1 / Math.max(1, n);
  for (let t = 0; t < T; t++) {
    if (!mask[t]) continue;
    for (let k = 0; k < D; k++) hg[t * D + k] = dx[k] * inv;
  }
  return hg;
};

async function run(meanPool: boolean, seqs: { ids: Uint32Array; mask: Float32Array }[]) {
  const weights = await loadMiniLMWeights();
  const enc = new CpuEncoder();
  enc.init({ spec: MINILM_L6, weights, mode: 'lora', rank: RANK, seed: SEED, maxSeq: T });
  const rng = mulberry(SEED);
  const head = new Head(rng);
  const order = [...train];
  const logits = new Float32Array(C);
  const dz = new Float32Array(C);
  for (let ep = 0; ep < EPOCHS; ep++) {
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    let loss = 0, hit = 0;
    for (let off = 0; off < order.length; off += 8) {
      const batch = order.slice(off, off + 8);
      for (const r of batch) {
        const hidden = await enc.forward(seqs[r].ids, seqs[r].mask);
        const x = pool(hidden, seqs[r].mask, meanPool);
        head.logits(x, logits);
        loss += softmaxCE(logits, 0, C, y[r]);
        let best = 0;
        for (let j = 1; j < C; j++) if (logits[j] > logits[best]) best = j;
        if (best === y[r]) hit++;
        let mx = -Infinity, sum = 0;
        for (let j = 0; j < C; j++) mx = Math.max(mx, logits[j]);
        for (let j = 0; j < C; j++) { dz[j] = Math.exp(logits[j] - mx); sum += dz[j]; }
        for (let j = 0; j < C; j++) dz[j] = dz[j] / sum - (j === y[r] ? 1 : 0);
        const dx = head.backward(x, dz);
        await enc.backward(poolGrad(dx, seqs[r].mask, meanPool));
      }
      await enc.applyGrads(LR, batch.length);
      head.apply(LR, batch.length);
    }
    // held-out
    let vhit = 0;
    for (const r of val) {
      const hidden = await enc.forward(seqs[r].ids, seqs[r].mask);
      head.logits(pool(hidden, seqs[r].mask, meanPool), logits);
      let best = 0;
      for (let j = 1; j < C; j++) if (logits[j] > logits[best]) best = j;
      if (best === y[r]) vhit++;
    }
    console.log(
      `${meanPool ? 'mean' : 'cls '} ep${ep + 1}  loss ${(loss / train.length).toFixed(3)}  trainAcc ${(hit / train.length).toFixed(3)}  valAcc ${(vhit / val.length).toFixed(3)}`,
    );
  }
  let vhit = 0;
  for (const r of val) {
    const hidden = await enc.forward(seqs[r].ids, seqs[r].mask);
    head.logits(pool(hidden, seqs[r].mask, meanPool), logits);
    let best = 0;
    for (let j = 1; j < C; j++) if (logits[j] > logits[best]) best = j;
    if (best === y[r]) vhit++;
  }
  return vhit / val.length;
}

const tok = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
const seqs: { ids: Uint32Array; mask: Float32Array }[] = [];
for (const r of rows) {
  const e = await tok(r.text, { padding: 'max_length', truncation: true, max_length: T, return_tensor: false });
  seqs.push({
    ids: Uint32Array.from((e.input_ids as number[]).map(Number)),
    mask: Float32Array.from((e.attention_mask as number[]).map(Number)),
  });
}
console.log('tokenized; training cls run…');
const clsAcc = await run(false, seqs);
console.log('training mean-pool run…');
const meanAcc = await run(true, seqs);
console.log(`\nRESULT  cls valAcc ${(clsAcc * 100).toFixed(1)}%   mean valAcc ${(meanAcc * 100).toFixed(1)}%`);
