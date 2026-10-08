import { AutoTokenizer } from '@huggingface/transformers';
import type { EncoderBackend, EncoderInit } from './spec';
import { MINILM_L6 } from './spec';
import { loadMiniLMWeights } from './weights';
import { CpuEncoder } from './cpu';
import { WgslEncoder } from './wgsl';
import { softmaxCE } from '../head';

export interface FtRequest {
  type: 'train';
  /** texts already truncated client-side for display; tokenized here */
  texts: string[];
  labels: Uint32Array;
  nClasses: number;
  lr: number;
  epochs: number;
  mode: 'lora' | 'full';
  rank: number;
  seed: number;
  backend: 'auto' | 'cpu';
  /** padded length, ≤ spec.maxPos */
  maxSeq: number;
}

export type FtMsg =
  | { type: 'weights'; frac: number }
  | { type: 'tokenizing' }
  | { type: 'epoch'; epoch: number; epochs: number; loss: number; acc: number; backend: string }
  | {
      type: 'done';
      params: Record<string, Float32Array>;
      headW: Float32Array;
      headB: Float32Array;
      backend: 'cpu' | 'wgsl';
      loss: number;
      acc: number;
    }
  | { type: 'error'; message: string };

const ctx = self as unknown as {
  onmessage: (e: MessageEvent<FtRequest>) => void;
  postMessage: (m: FtMsg, transfer?: Transferable[]) => void;
};

function mulberry(seed: number) {
  let s = seed >>> 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Tiny linear head on the CLS row — trained jointly, AdamW inline. */
class Head {
  d: number;
  c: number;
  w: Float32Array;
  b: Float32Array;
  gw: Float32Array;
  gb: Float32Array;
  mw: Float32Array;
  vw: Float32Array;
  mb: Float32Array;
  vb: Float32Array;
  t = 0;
  constructor(d: number, c: number, rng: () => number) {
    this.d = d;
    this.c = c;
    const scale = Math.sqrt(2 / (d + c));
    this.w = new Float32Array(d * c).map(() => (rng() * 2 - 1) * scale);
    this.b = new Float32Array(c);
    this.gw = new Float32Array(d * c);
    this.gb = new Float32Array(c);
    this.mw = new Float32Array(d * c);
    this.vw = new Float32Array(d * c);
    this.mb = new Float32Array(c);
    this.vb = new Float32Array(c);
  }
  logits(cls: Float32Array, out: Float32Array) {
    for (let j = 0; j < this.c; j++) {
      let s = this.b[j];
      for (let k = 0; k < this.d; k++) s += cls[k] * this.w[k * this.c + j];
      out[j] = s;
    }
  }
  /** accumulate grads; returns dL/dcls */
  backward(cls: Float32Array, dz: Float32Array) {
    const dcls = new Float32Array(this.d);
    for (let j = 0; j < this.c; j++) {
      this.gb[j] += dz[j];
      for (let k = 0; k < this.d; k++) {
        this.gw[k * this.c + j] += cls[k] * dz[j];
        dcls[k] += dz[j] * this.w[k * this.c + j];
      }
    }
    return dcls;
  }
  apply(lr: number, n: number) {
    const b1 = 0.9,
      b2 = 0.999,
      eps = 1e-8;
    this.t++;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    const inv = 1 / n;
    for (let p = 0; p < this.d * this.c; p++) {
      const g = this.gw[p] * inv;
      this.mw[p] = b1 * this.mw[p] + (1 - b1) * g;
      this.vw[p] = b2 * this.vw[p] + (1 - b2) * g * g;
      this.w[p] -= (lr * this.mw[p]) / (bc1 * (Math.sqrt(this.vw[p] / bc2) + eps));
      this.gw[p] = 0;
    }
    for (let j = 0; j < this.c; j++) {
      const g = this.gb[j] * inv;
      this.mb[j] = b1 * this.mb[j] + (1 - b1) * g;
      this.vb[j] = b2 * this.vb[j] + (1 - b2) * g * g;
      this.b[j] -= (lr * this.mb[j]) / (bc1 * (Math.sqrt(this.vb[j] / bc2) + eps));
      this.gb[j] = 0;
    }
  }
}

async function run(req: FtRequest, post: (m: FtMsg, t?: Transferable[]) => void) {
  const weights = await loadMiniLMWeights((f) => post({ type: 'weights', frac: f }));

  post({ type: 'tokenizing' });
  const tok = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
  const T = req.maxSeq;
  const allIds: Uint32Array[] = [];
  const allMask: Float32Array[] = [];
  for (const text of req.texts) {
    const enc = await tok(text, {
      padding: 'max_length',
      truncation: true,
      max_length: T,
      return_tensor: false,
    });
    const ids = new Uint32Array(T);
    const mask = new Float32Array(T);
    const raw = enc.input_ids as number[] | bigint[] | ArrayLike<number>;
    const rawMask = enc.attention_mask as number[] | ArrayLike<number>;
    for (let t = 0; t < T; t++) {
      ids[t] = Number(raw[t] ?? 0);
      mask[t] = Number(rawMask[t] ?? 0);
    }
    allIds.push(ids);
    allMask.push(mask);
  }

  const init: EncoderInit = {
    spec: MINILM_L6,
    weights,
    mode: req.mode,
    rank: req.rank,
    seed: req.seed,
    maxSeq: T,
  };
  let enc: EncoderBackend;
  if (req.backend === 'cpu') {
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

  const d = MINILM_L6.dModel;
  const c = req.nClasses;
  const rng = mulberry(req.seed);
  const head = new Head(d, c, rng);
  const order = Array.from({ length: req.texts.length }, (_, i) => i);
  const BATCH = 8;
  const dz = new Float32Array(c);
  const logits = new Float32Array(c);
  let lastLoss = 0,
    lastAcc = 0;

  for (let ep = 0; ep < req.epochs; ep++) {
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    let epLoss = 0,
      epHit = 0,
      seen = 0;
    for (let off = 0; off < order.length; off += BATCH) {
      const rows = order.slice(off, off + BATCH);
      for (const r of rows) {
        const hidden = await enc.forward(allIds[r], allMask[r]);
        const cls = hidden.slice(0, d);
        head.logits(cls, logits);
        epLoss += softmaxCE(logits, 0, c, req.labels[r]);
        let best = 0;
        for (let j = 1; j < c; j++) if (logits[j] > logits[best]) best = j;
        if (best === req.labels[r]) epHit++;
        // dz = softmax - onehot
        let mx = -Infinity;
        for (let j = 0; j < c; j++) mx = Math.max(mx, logits[j]);
        let sum = 0;
        for (let j = 0; j < c; j++) {
          dz[j] = Math.exp(logits[j] - mx);
          sum += dz[j];
        }
        for (let j = 0; j < c; j++) dz[j] = dz[j] / sum - (j === req.labels[r] ? 1 : 0);
        const dcls = head.backward(cls, dz);
        const hg = new Float32Array(T * d);
        hg.set(dcls, 0);
        await enc.backward(hg);
      }
      await enc.applyGrads(req.lr);
      head.apply(req.lr, rows.length);
      seen += rows.length;
      post({
        type: 'epoch',
        epoch: ep + seen / order.length,
        epochs: req.epochs,
        loss: epLoss / seen,
        acc: epHit / seen,
        backend: enc.kind,
      });
    }
    lastLoss = epLoss / order.length;
    lastAcc = epHit / order.length;
    post({
      type: 'epoch',
      epoch: ep + 1,
      epochs: req.epochs,
      loss: lastLoss,
      acc: lastAcc,
      backend: enc.kind,
    });
  }

  const params = await enc.trainableWeights();
  const msg = {
    type: 'done' as const,
    params,
    headW: head.w,
    headB: head.b,
    backend: enc.kind,
    loss: lastLoss,
    acc: lastAcc,
  };
  const transfers = [
    head.w.buffer,
    head.b.buffer,
    ...Object.values(params).map((p) => p.buffer as ArrayBuffer),
  ] as Transferable[];
  post(msg, transfers);
  enc.dispose?.();
}

ctx.onmessage = (e) => {
  const post = (m: FtMsg, t?: Transferable[]) => ctx.postMessage(m, t);
  if (e.data.type !== 'train') return;
  run(e.data, post).catch((err) =>
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) }),
  );
};
