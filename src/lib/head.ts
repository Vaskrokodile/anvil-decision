/**
 * Trainable head over frozen embeddings — backend interface + CPU reference
 * implementation (linear → softmax CE → AdamW). Pure typed-array math; runs
 * identically in a Worker or in Node for tests.
 */

export interface TrainInit {
  /** embeddings [n, d] row-major */
  x: Float32Array;
  /** class index per row */
  y: Uint32Array;
  n: number;
  d: number;
  c: number;
  lr: number;
  batch: number;
  epochs: number;
  seed: number;
  backend: 'cpu' | 'wgsl' | 'auto';
  /** fraction of rows held out for validation (0 disables the split) */
  valFrac?: number;
}

export interface StepReport {
  epoch: number;
  step: number;
  loss: number;
  acc: number;
}

export interface HeadBackend {
  kind: 'cpu' | 'wgsl';
  init(i: TrainInit): Promise<void> | void;
  /** one optimizer step over the given batch row indices */
  step(idx: Uint32Array): Promise<void>;
  /** logits for the whole set — used for epoch-level loss/acc reporting */
  logits(): Promise<Float32Array>; // [n, c]
  weights(): Promise<{ w: Float32Array; b: Float32Array }>;
  dispose?(): void;
}

/* ------------------------------------------------------------------ */

export function softmaxCE(logits: Float32Array, off: number, c: number, label: number) {
  let max = -Infinity;
  for (let j = 0; j < c; j++) max = Math.max(max, logits[off + j]);
  let sum = 0;
  for (let j = 0; j < c; j++) sum += Math.exp(logits[off + j] - max);
  const logsum = max + Math.log(sum);
  return logsum - logits[off + label]; // -log p(label)
}

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

export class CpuHead implements HeadBackend {
  kind = 'cpu' as const;
  private i!: TrainInit;
  private w!: Float32Array; // [d, c]
  private b!: Float32Array; // [c]
  private mw!: Float32Array;
  private vw!: Float32Array;
  private mb!: Float32Array;
  private vb!: Float32Array;
  private t = 0;

  init(i: TrainInit) {
    this.i = i;
    const rng = mulberry(i.seed);
    const scale = Math.sqrt(2 / (i.d + i.c)); // he-ish
    this.w = new Float32Array(i.d * i.c).map(() => (rng() * 2 - 1) * scale);
    this.b = new Float32Array(i.c);
    this.mw = new Float32Array(i.d * i.c);
    this.vw = new Float32Array(i.d * i.c);
    this.mb = new Float32Array(i.c);
    this.vb = new Float32Array(i.c);
  }

  private forward(xo: number, into: Float32Array) {
    const { d, c } = this.i;
    for (let j = 0; j < c; j++) {
      let s = this.b[j];
      for (let k = 0; k < d; k++) s += this.i.x[xo + k] * this.w[k * c + j];
      into[j] = s;
    }
  }

  async step(idx: Uint32Array) {
    const { d, c, x, y } = this.i;
    const B = idx.length;
    const dw = new Float32Array(d * c);
    const db = new Float32Array(c);
    const dz = new Float32Array(B * c);
    const logits = new Float32Array(c);

    for (let r = 0; r < B; r++) {
      const row = idx[r];
      const xo = row * d;
      this.forward(xo, logits);
      // softmax - onehot
      let max = -Infinity;
      for (let j = 0; j < c; j++) max = Math.max(max, logits[j]);
      let sum = 0;
      for (let j = 0; j < c; j++) {
        const e = Math.exp(logits[j] - max);
        dz[r * c + j] = e;
        sum += e;
      }
      for (let j = 0; j < c; j++) dz[r * c + j] = dz[r * c + j] / sum - (j === y[row] ? 1 : 0);
      // accumulate db and dw
      for (let j = 0; j < c; j++) db[j] += dz[r * c + j];
      for (let k = 0; k < d; k++) {
        const xv = x[xo + k];
        if (xv === 0) continue;
        const base = k * c;
        const dzo = r * c;
        for (let j = 0; j < c; j++) dw[base + j] += xv * dz[dzo + j];
      }
    }

    // AdamW
    const { lr } = this.i;
    const b1 = 0.9,
      b2 = 0.999,
      eps = 1e-8,
      wd = 0.01;
    this.t++;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    const inv = 1 / B;
    for (let p = 0; p < d * c; p++) {
      const g = dw[p] * inv + wd * this.w[p];
      this.mw[p] = b1 * this.mw[p] + (1 - b1) * g;
      this.vw[p] = b2 * this.vw[p] + (1 - b2) * g * g;
      this.w[p] -= (lr * this.mw[p]) / (bc1 * (Math.sqrt(this.vw[p] / bc2) + eps));
    }
    for (let j = 0; j < c; j++) {
      const g = db[j] * inv;
      this.mb[j] = b1 * this.mb[j] + (1 - b1) * g;
      this.vb[j] = b2 * this.vb[j] + (1 - b2) * g * g;
      this.b[j] -= (lr * this.mb[j]) / (bc1 * (Math.sqrt(this.vb[j] / bc2) + eps));
    }
  }

  async logits() {
    const { n, d, c } = this.i;
    const out = new Float32Array(n * c);
    const row = new Float32Array(c);
    for (let r = 0; r < n; r++) {
      this.forward(r * d, row);
      out.set(row, r * c);
    }
    return out;
  }

  async weights() {
    return { w: this.w.slice(), b: this.b.slice() };
  }
}

/** Mean CE loss + accuracy over a logits matrix; `idx` restricts to a row subset. */
export function evalLogits(
  logits: Float32Array,
  y: Uint32Array,
  c: number,
  idx?: ArrayLike<number>,
) {
  const n = idx ? idx.length : y.length;
  let loss = 0;
  let hits = 0;
  for (let i = 0; i < n; i++) {
    const r = idx ? idx[i] : i;
    loss += softmaxCE(logits, r * c, c, y[r]);
    let best = 0;
    for (let j = 1; j < c; j++) if (logits[r * c + j] > logits[r * c + best]) best = j;
    if (best === y[r]) hits++;
  }
  return { loss: loss / n, acc: hits / n };
}
