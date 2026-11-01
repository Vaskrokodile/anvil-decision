import { AutoTokenizer } from '@huggingface/transformers';
import type { EncoderBackend, EncoderInit } from './spec';
import { MINILM_L6, DEBERTA_V3_SMALL, poolForArch, poolHidden, poolGrad } from './spec';
import { loadMiniLMWeights, loadDebertaWeights, DEBERTA_TOKENIZER } from './weights';
import { CpuEncoder } from './cpu';
import { DebertaCpuEncoder } from './deberta';
import { WgslEncoder } from './wgsl';
import { parityCheckedEncoder } from './parity';
import { softmaxCE } from '../head';
import { mulberry, stratifiedSplit } from '../split';

/** Rows without a label for a task carry this sentinel in `y`. */
export const TASK_MISSING = 0xffffffff;

export interface FtRequest {
  type: 'train';
  /** texts already truncated client-side for display; tokenized here */
  texts: string[];
  labels: Uint32Array;
  nClasses: number;
  /**
   * Multi-task mode: each entry is a dataset label column → its own head.
   * `y` holds TASK_MISSING for unlabeled rows. Overrides labels/nClasses.
   */
  tasks?: { name: string; labels: string[]; y: Uint32Array }[];
  /** training substrate — deberta runs CPU-only (no WGSL backend yet) */
  arch?: 'minilm' | 'deberta';
  lr: number;
  batch?: number;
  epochs: number;
  mode: 'lora' | 'full';
  rank: number;
  seed: number;
  backend: 'auto' | 'cpu';
  /** padded length, ≤ spec.maxPos */
  maxSeq: number;
  /** fraction of rows held out for validation (0 disables the split) */
  valFrac?: number;
}

export type FtMsg =
  | { type: 'weights'; frac: number }
  | { type: 'tokenizing' }
  | { type: 'parity'; backend: 'cpu' | 'wgsl'; cos: number }
  | { type: 'epoch'; epoch: number; epochs: number; loss: number; acc: number; backend: string }
  | {
      type: 'done';
      params: Record<string, Float32Array>;
      headW: Float32Array;
      headB: Float32Array;
      backend: 'cpu' | 'wgsl';
      loss: number;
      acc: number;
      /** held-out accuracy — null when the split left no val rows */
      valAcc: number | null;
      valN: number;
      trainN: number;
      /** Multi-task: per-task heads + held-out metrics (task 0 mirrors headW). */
      tasks?: {
        name: string;
        labels: string[];
        headW: Float32Array;
        headB: Float32Array;
        valAcc: number | null;
        valN: number;
      }[];
    }
  | { type: 'error'; message: string };

const ctx = self as unknown as {
  onmessage: (e: MessageEvent<FtRequest>) => void;
  postMessage: (m: FtMsg, transfer?: Transferable[]) => void;
};

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
  const arch = req.arch ?? 'minilm';
  const spec = arch === 'deberta' ? DEBERTA_V3_SMALL : MINILM_L6;
  const pool = poolForArch(arch);
  const weights = arch === 'deberta'
    ? await loadDebertaWeights((f) => post({ type: 'weights', frac: f }))
    : await loadMiniLMWeights((f) => post({ type: 'weights', frac: f }));

  post({ type: 'tokenizing' });
  const tok = await AutoTokenizer.from_pretrained(
    arch === 'deberta' ? DEBERTA_TOKENIZER : 'Xenova/all-MiniLM-L6-v2',
  );
  const T = Math.max(1, Math.min(req.maxSeq, spec.maxPos));
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
    spec,
    weights,
    mode: req.mode,
    rank: req.rank,
    seed: req.seed,
    maxSeq: T,
  };
  let enc: EncoderBackend;
  if (req.backend === 'cpu' || arch === 'deberta') {
    // deberta is CPU-only for now — disentangled attention has no WGSL backend
    enc = arch === 'deberta' ? new DebertaCpuEncoder() : new CpuEncoder();
    enc.init(init);
  } else {
    try {
      const wgsl = await WgslEncoder.create(init);
      const checked = await parityCheckedEncoder(wgsl, init, allIds[0], allMask[0]);
      enc = checked.enc;
      post({ type: 'parity', backend: enc.kind, cos: checked.cos });
    } catch {
      enc = new CpuEncoder();
      enc.init(init);
    }
  }

  const d = spec.dModel;
  const rng = mulberry(req.seed);

  // Normalize: single-task becomes a one-entry task list.
  const tasks = (req.tasks ?? [
    { name: 'label', labels: Array.from({ length: req.nClasses }, (_, i) => String(i)), y: req.labels },
  ]).map((td) => {
    const validIdx: number[] = [];
    const yv: number[] = [];
    td.y.forEach((c, i) => {
      if (c !== TASK_MISSING) {
        validIdx.push(i);
        yv.push(c);
      }
    });
    const { train, val } = stratifiedSplit(Uint32Array.from(yv), req.valFrac ?? 0.2, req.seed);
    return {
      ...td,
      head: new Head(d, td.labels.length, rng),
      trainIdx: train.map((li) => validIdx[li]),
      valIdx: val.map((li) => validIdx[li]),
      trainSet: new Set<number>(),
      batchN: 0,
      valAcc: null as number | null,
    };
  });
  for (const t of tasks) t.trainSet = new Set(t.trainIdx);
  // union of all tasks' train rows — one forward per row feeds every head that
  // labels it, gradients sum into the shared encoder
  const order = [...new Set(tasks.flatMap((t) => t.trainIdx))];
  const BATCH = Math.max(1, Math.min(64, Math.floor(req.batch ?? 8)));
  let lastLoss = 0,
    lastAcc = 0;

  for (let ep = 0; ep < req.epochs; ep++) {
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    let epLoss = 0,
      epHit = 0,
      epPreds = 0,
      seen = 0;
    for (let off = 0; off < order.length; off += BATCH) {
      const rows = order.slice(off, off + BATCH);
      for (const r of rows) {
        const hidden = await enc.forward(allIds[r], allMask[r]);
        const pooled = poolHidden(hidden, allMask[r], T, d, pool);
        const hg = new Float32Array(T * d);
        let touched = false;
        for (const t of tasks) {
          if (!t.trainSet.has(r)) continue;
          touched = true;
          t.batchN++;
          const c = t.labels.length;
          const logits = new Float32Array(c);
          t.head.logits(pooled, logits);
          epLoss += softmaxCE(logits, 0, c, t.y[r]);
          let best = 0;
          for (let j = 1; j < c; j++) if (logits[j] > logits[best]) best = j;
          if (best === t.y[r]) epHit++;
          epPreds++;
          // dz = softmax - onehot
          const dz = new Float32Array(c);
          let mx = -Infinity;
          for (let j = 0; j < c; j++) mx = Math.max(mx, logits[j]);
          let sum = 0;
          for (let j = 0; j < c; j++) {
            dz[j] = Math.exp(logits[j] - mx);
            sum += dz[j];
          }
          for (let j = 0; j < c; j++) dz[j] = dz[j] / sum - (j === t.y[r] ? 1 : 0);
          const dpooled = t.head.backward(pooled, dz);
          const dhg = poolGrad(dpooled, allMask[r], T, d, pool);
          for (let k = 0; k < hg.length; k++) hg[k] += dhg[k];
        }
        if (touched) await enc.backward(hg);
      }
      await enc.applyGrads(req.lr, rows.length);
      for (const t of tasks) {
        if (t.batchN > 0) t.head.apply(req.lr, t.batchN);
        t.batchN = 0;
      }
      seen += rows.length;
      post({
        type: 'epoch',
        epoch: ep + seen / order.length,
        epochs: req.epochs,
        loss: epLoss / Math.max(1, epPreds),
        acc: epHit / Math.max(1, epPreds),
        backend: enc.kind,
      });
    }
    lastLoss = epLoss / Math.max(1, epPreds);
    lastAcc = epHit / Math.max(1, epPreds);
    post({
      type: 'epoch',
      epoch: ep + 1,
      epochs: req.epochs,
      loss: lastLoss,
      acc: lastAcc,
      backend: enc.kind,
    });
  }

  // held-out eval: forward each task's val rows, no grads
  for (const t of tasks) {
    if (t.valIdx.length === 0) continue;
    let hit = 0;
    const c = t.labels.length;
    const logits = new Float32Array(c);
    for (const r of t.valIdx) {
      const hidden = await enc.forward(allIds[r], allMask[r]);
      t.head.logits(poolHidden(hidden, allMask[r], T, d, pool), logits);
      let best = 0;
      for (let j = 1; j < c; j++) if (logits[j] > logits[best]) best = j;
      if (best === t.y[r]) hit++;
    }
    t.valAcc = hit / t.valIdx.length;
  }

  const params = await enc.trainableWeights();
  const primary = tasks[0];
  const msg = {
    type: 'done' as const,
    params,
    headW: primary.head.w,
    headB: primary.head.b,
    backend: enc.kind,
    loss: lastLoss,
    acc: lastAcc,
    valAcc: primary.valAcc,
    valN: primary.valIdx.length,
    trainN: primary.trainIdx.length,
    ...(req.tasks
      ? {
          tasks: tasks.map((t) => ({
            name: t.name,
            labels: t.labels,
            headW: t.head.w,
            headB: t.head.b,
            valAcc: t.valAcc,
            valN: t.valIdx.length,
          })),
        }
      : {}),
  };
  const transfers = [
    primary.head.w.buffer,
    primary.head.b.buffer,
    ...(req.tasks ? tasks.flatMap((t) => [t.head.w.buffer, t.head.b.buffer]) : []),
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
