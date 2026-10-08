/**
 * cpu.ts — pure-TypeScript CPU reference encoder (BERT-style) with LoRA or
 * full fine-tuning. Numeric oracle for the WGSL backend and the no-GPU
 * fallback. Readable > fast: everything is plain Float32Array loops.
 *
 * Conventions:
 *  - HF linear weights are [out, in] row-major:  y = x·Wᵀ + b.
 *  - Attention heads are contiguous column slices of the [T,d] q/k/v
 *    projections; probs are stored [H,T,T].
 *  - LoRA on q/k/v/o/f1/f2: frozen W plus trainable a [in,r] (σ=0.01,
 *    weight-decayed) and b [r,out] (zeros, no decay), scale s = alpha/r = 2.
 *  - Embedding tables (word/pos/tokType) are frozen in every mode.
 *
 * NOTE: the './spec.ts' extension is deliberate — it lets Node ≥22 run this
 * file directly (type stripping) for grad-checks without a bundler.
 */

import {
  W,
  type EncoderBackend,
  type EncoderInit,
  type Weights,
  type XformerSpec,
} from './spec.ts';

type F = Float32Array;

/** masked-out logits get this bias — exp() underflows to exactly 0 */
const MASK_BIAS = -1e9;

/* ------------------------------------------------------------------ */
/* small numeric helpers                                               */

/** mulberry32 — same seeded RNG as head.ts so LoRA init is reproducible. */
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

/**
 * erf via Abramowitz & Stegun 7.1.26 — |ε| ≤ 1.5e-7, plenty for the ~1e-6
 * numeric-parity budget. (JS has no Math.erf.)
 */
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

const SQRT2 = Math.SQRT2;
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

/** exact-erf GELU: g(x) = x·Φ(x) = 0.5·x·(1 + erf(x/√2)) */
function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / SQRT2));
}

/** g'(x) = Φ(x) + x·φ(x), φ = standard normal pdf */
function geluGrad(x: number): number {
  return 0.5 * (1 + erf(x / SQRT2)) + x * INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}

/** y[t,o] = Σ_k x[t,k]·w[o,k] + b[o]  — HF layout w [out,in]. */
function linear(x: F, w: F, b: F, T: number, iD: number, oD: number): F {
  const y = new Float32Array(T * oD);
  for (let t = 0; t < T; t++) {
    const xo = t * iD;
    const yo = t * oD;
    for (let o = 0; o < oD; o++) {
      let acc = b[o];
      const wo = o * iD;
      for (let k = 0; k < iD; k++) acc += x[xo + k] * w[wo + k];
      y[yo + o] = acc;
    }
  }
  return y;
}

/** lh[t,j] = Σ_k x[t,k]·a[k,j] — LoRA bottleneck activations, a is [in,r]. */
function loraHidden(x: F, a: F, T: number, iD: number, r: number): F {
  const lh = new Float32Array(T * r);
  for (let t = 0; t < T; t++) {
    const xo = t * iD;
    const ho = t * r;
    for (let j = 0; j < r; j++) {
      let acc = 0;
      for (let k = 0; k < iD; k++) acc += x[xo + k] * a[k * r + j];
      lh[ho + j] = acc;
    }
  }
  return lh;
}

/** y[t,o] += s·Σ_j lh[t,j]·b[j,o] — LoRA output side, b is [r,out]. */
function loraApply(y: F, lh: F, b: F, T: number, r: number, oD: number, s: number): void {
  for (let t = 0; t < T; t++) {
    const ho = t * r;
    const yo = t * oD;
    for (let o = 0; o < oD; o++) {
      let acc = 0;
      for (let j = 0; j < r; j++) acc += lh[ho + j] * b[j * oD + o];
      y[yo + o] += s * acc;
    }
  }
}

interface LnCache {
  xhat: F;
  rstd: F;
}

/** Row LayerNorm. Returns y plus the xhat/rstd the backward pass needs. */
function lnFwd(x: F, w: F, b: F, T: number, d: number, eps: number): { y: F } & LnCache {
  const y = new Float32Array(T * d);
  const xhat = new Float32Array(T * d);
  const rstd = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let mu = 0;
    for (let k = 0; k < d; k++) mu += x[o + k];
    mu /= d;
    let va = 0;
    for (let k = 0; k < d; k++) {
      const c = x[o + k] - mu;
      va += c * c;
    }
    va /= d;
    const rs = 1 / Math.sqrt(va + eps);
    rstd[t] = rs;
    for (let k = 0; k < d; k++) {
      const h = (x[o + k] - mu) * rs;
      xhat[o + k] = h;
      y[o + k] = h * w[k] + b[k];
    }
  }
  return { y, xhat, rstd };
}

/**
 * LayerNorm backward. dy [T,d] → dx [T,d]:
 *   dx = rstd·(g − mean(g) − xhat·mean(g·xhat)),   g = dy∘w
 * Accumulates dw/db rowsums when gw/gb are provided (full mode).
 */
function lnBwd(
  dy: F,
  xhat: F,
  rstd: F,
  w: F,
  T: number,
  d: number,
  gw: F | undefined,
  gb: F | undefined,
): F {
  const dx = new Float32Array(T * d);
  const g = new Float32Array(d);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let s1 = 0;
    let s2 = 0;
    for (let k = 0; k < d; k++) {
      const v = dy[o + k] * w[k];
      g[k] = v;
      s1 += v;
      s2 += v * xhat[o + k];
    }
    const rs = rstd[t];
    const m1 = s1 / d;
    const m2 = s2 / d;
    for (let k = 0; k < d; k++) dx[o + k] = (g[k] - m1 - xhat[o + k] * m2) * rs;
    if (gw) for (let k = 0; k < d; k++) gw[k] += dy[o + k] * xhat[o + k];
    if (gb) for (let k = 0; k < d; k++) gb[k] += dy[o + k];
  }
  return dx;
}

interface LoraGrads {
  a: F;
  b: F;
  ga: F;
  gb: F;
  lh: F;
}

/**
 * Backward through y = x·Wᵀ + b (+ s·(x·a)·b′ LoRA branch).
 * Returns dx [T,iD]; accumulates dw/db (full mode) and/or ga/gb (lora mode).
 *
 *   dx[t,k]  = Σ_o dy[t,o]·w[o,k]  +  Σ_j dl[t,j]·a[k,j]
 *   dw[o,k] += Σ_t dy[t,o]·x[t,k];   db[o] += Σ_t dy[t,o]
 *   dl[t,j]  = s·Σ_o dy[t,o]·b′[j,o]
 *   ga[k,j] += Σ_t x[t,k]·dl[t,j]        ≡ s·(Xᵀdy)·b′ᵀ
 *   gb[j,o] += s·Σ_t lh[t,j]·dy[t,o]     ≡ s·aᵀ·(Xᵀdy)
 *
 * The frozen base w still contributes to dx — it is in the compute graph
 * even when it takes no grads.
 */
function linBwd(
  dy: F,
  x: F,
  T: number,
  iD: number,
  oD: number,
  w: F,
  dw: F | undefined,
  db: F | undefined,
  lora: LoraGrads | undefined,
  s: number,
): F {
  const dx = new Float32Array(T * iD);
  for (let t = 0; t < T; t++) {
    const yo = t * oD;
    const xo = t * iD;
    for (let o = 0; o < oD; o++) {
      const g = dy[yo + o];
      if (g === 0) continue;
      const wo = o * iD;
      for (let k = 0; k < iD; k++) dx[xo + k] += g * w[wo + k];
    }
  }
  if (dw) {
    for (let o = 0; o < oD; o++) {
      const wo = o * iD;
      for (let t = 0; t < T; t++) {
        const g = dy[t * oD + o];
        if (g === 0) continue;
        const xo = t * iD;
        for (let k = 0; k < iD; k++) dw[wo + k] += g * x[xo + k];
      }
    }
  }
  if (db) {
    for (let t = 0; t < T; t++) {
      const yo = t * oD;
      for (let o = 0; o < oD; o++) db[o] += dy[yo + o];
    }
  }
  if (lora) {
    const { a, b, ga, gb, lh } = lora;
    const r = a.length / iD;
    const dl = new Float32Array(T * r);
    for (let t = 0; t < T; t++) {
      const yo = t * oD;
      const ho = t * r;
      for (let j = 0; j < r; j++) {
        let acc = 0;
        const bo = j * oD;
        for (let o = 0; o < oD; o++) acc += dy[yo + o] * b[bo + o];
        dl[ho + j] = s * acc;
      }
    }
    for (let t = 0; t < T; t++) {
      const xo = t * iD;
      const ho = t * r;
      for (let k = 0; k < iD; k++) {
        const xv = x[xo + k];
        if (xv === 0) continue;
        const ao = k * r;
        for (let j = 0; j < r; j++) ga[ao + j] += xv * dl[ho + j];
      }
    }
    for (let t = 0; t < T; t++) {
      const yo = t * oD;
      const ho = t * r;
      for (let j = 0; j < r; j++) {
        const hv = s * lh[ho + j];
        if (hv === 0) continue;
        const bo = j * oD;
        for (let o = 0; o < oD; o++) gb[bo + o] += hv * dy[yo + o];
      }
    }
    for (let t = 0; t < T; t++) {
      const xo = t * iD;
      const ho = t * r;
      for (let k = 0; k < iD; k++) {
        const ao = k * r;
        let acc = 0;
        for (let j = 0; j < r; j++) acc += dl[ho + j] * a[ao + j];
        dx[xo + k] += acc;
      }
    }
  }
  return dx;
}

/**
 * Multi-head scaled dot-product attention.
 *   P[h,i,j] = softmax_j( q[h,i]·k[h,j]/√dH + (1−mask[j])·MASK_BIAS )
 *   ctx[h,i] = Σ_j P[h,i,j]·v[h,j]     (merged back into [T,d])
 */
function attend(
  q: F,
  k: F,
  v: F,
  mask: F,
  T: number,
  d: number,
  H: number,
  dH: number,
): { p: F; ctx: F } {
  const p = new Float32Array(H * T * T);
  const ctx = new Float32Array(T * d);
  const inv = 1 / Math.sqrt(dH);
  for (let h = 0; h < H; h++) {
    const hd = h * dH;
    for (let i = 0; i < T; i++) {
      const qi = i * d + hd;
      const po = (h * T + i) * T;
      let mx = -Infinity;
      for (let j = 0; j < T; j++) {
        const kj = j * d + hd;
        let acc = 0;
        for (let u = 0; u < dH; u++) acc += q[qi + u] * k[kj + u];
        const sc = acc * inv + (1 - mask[j]) * MASK_BIAS;
        p[po + j] = sc;
        if (sc > mx) mx = sc;
      }
      let sum = 0;
      for (let j = 0; j < T; j++) {
        const e = Math.exp(p[po + j] - mx);
        p[po + j] = e;
        sum += e;
      }
      const invSum = 1 / sum;
      for (let j = 0; j < T; j++) {
        const wgt = p[po + j] * invSum;
        p[po + j] = wgt;
        if (wgt === 0) continue;
        const vj = j * d + hd;
        const ci = i * d + hd;
        for (let u = 0; u < dH; u++) ctx[ci + u] += wgt * v[vj + u];
      }
    }
  }
  return { p, ctx };
}

/**
 * Attention backward. With ctx[h,i] = Σ_j P[i,j]·v[j] and P = softmax(S):
 *   dP[i,j] = dctx[i]·v[j];          dV[j] = Σ_i P[i,j]·dctx[i]
 *   dS[i,j] = P[i,j]·(dP[i,j] − Σ_l P[i,l]·dP[i,l]) / √dH
 *   dQ[i]   = Σ_j dS[i,j]·k[j];      dK[j] = Σ_i dS[i,j]·q[i]
 * Masked j have P≈0, so their dS vanish with no special-casing.
 */
function attendBwd(
  dctx: F,
  q: F,
  k: F,
  v: F,
  p: F,
  T: number,
  d: number,
  H: number,
  dH: number,
): { dq: F; dk: F; dv: F } {
  const dq = new Float32Array(T * d);
  const dk = new Float32Array(T * d);
  const dv = new Float32Array(T * d);
  const inv = 1 / Math.sqrt(dH);
  const ds = new Float32Array(T); // per-row dP→dS scratch
  for (let h = 0; h < H; h++) {
    const hd = h * dH;
    for (let i = 0; i < T; i++) {
      const po = (h * T + i) * T;
      const ci = i * d + hd;
      for (let j = 0; j < T; j++) {
        const wgt = p[po + j];
        if (wgt === 0) continue;
        const vj = j * d + hd;
        for (let u = 0; u < dH; u++) dv[vj + u] += wgt * dctx[ci + u];
      }
    }
    for (let i = 0; i < T; i++) {
      const po = (h * T + i) * T;
      const ci = i * d + hd;
      let dot = 0;
      for (let j = 0; j < T; j++) {
        const vj = j * d + hd;
        let acc = 0;
        for (let u = 0; u < dH; u++) acc += dctx[ci + u] * v[vj + u];
        ds[j] = acc; // dP[i,j]
        dot += p[po + j] * acc;
      }
      for (let j = 0; j < T; j++) ds[j] = p[po + j] * (ds[j] - dot) * inv;
      const qi = i * d + hd;
      for (let j = 0; j < T; j++) {
        const g = ds[j];
        if (g === 0) continue;
        const kj = j * d + hd;
        for (let u = 0; u < dH; u++) {
          dq[qi + u] += g * k[kj + u];
          dk[kj + u] += g * q[qi + u];
        }
      }
    }
  }
  return { dq, dk, dv };
}

/* ------------------------------------------------------------------ */
/* parameter / cache bookkeeping                                       */

interface Param {
  data: F;
  grad: F;
  m: F;
  v: F;
  /** AdamW weight decay applies (matrices + LoRA-A only) */
  decay: boolean;
}

interface LoraProj {
  a: Param; // [in, r]
  b: Param; // [r, out]
}

interface LoraL {
  q: LoraProj;
  k: LoraProj;
  v: LoraProj;
  o: LoraProj;
  f1: LoraProj;
  f2: LoraProj;
}

interface LayerW {
  qW: F;
  qB: F;
  kW: F;
  kB: F;
  vW: F;
  vB: F;
  oW: F;
  oB: F;
  ln1W: F;
  ln1B: F;
  f1W: F;
  f1B: F;
  f2W: F;
  f2B: F;
  ln2W: F;
  ln2B: F;
}

interface LayerCache {
  xIn: F; // layer input (also q/k/v linear input)
  q: F;
  k: F;
  v: F;
  probs: F; // [H,T,T]
  ctx: F; // merged attention output, o-proj input
  ln1: LnCache;
  xAttn: F; // post-LN1, ffn input
  h1pre: F; // pre-GELU
  h1: F; // post-GELU, f2 input
  ln2: LnCache;
  lh: { q?: F; k?: F; v?: F; o?: F; f1?: F; f2?: F }; // lora bottlenecks
}

interface FwdCache {
  T: number;
  embLn: LnCache;
  layers: LayerCache[];
}

/* ------------------------------------------------------------------ */

export class CpuEncoder implements EncoderBackend {
  kind = 'cpu' as const;

  private spec!: XformerSpec;
  private weights!: Weights;
  private rank = 0;
  private loraScale = 0;
  private maxSeq = 0;
  private layerW: LayerW[] = [];
  private lora: LoraL[] = [];
  private params: Record<string, Param> = {};
  private adamT = 0;
  private cache: FwdCache | null = null;

  init(i: EncoderInit) {
    this.spec = i.spec;
    this.weights = i.weights;
    this.rank = i.rank;
    this.maxSeq = i.maxSeq;
    const s = i.spec;
    if (s.dModel !== s.heads * s.dHead) {
      throw new Error(`spec: dModel ${s.dModel} ≠ heads·dHead ${s.heads * s.dHead}`);
    }
    // resolve + validate weight tensors up front
    const need = [W.word, W.pos, W.tokType, W.embLnW, W.embLnB];
    for (let l = 0; l < s.layers; l++) {
      const n = W.layer(l);
      need.push(
        n.qW, n.qB, n.kW, n.kB, n.vW, n.vB, n.oW, n.oB,
        n.ln1W, n.ln1B, n.f1W, n.f1B, n.f2W, n.f2B, n.ln2W, n.ln2B,
      );
    }
    const missing = need.filter((k) => !(k in i.weights));
    if (missing.length) {
      throw new Error(`weights missing ${missing.length} tensors, e.g. ${missing[0]}`);
    }

    const w = i.weights;
    this.layerW = [];
    for (let l = 0; l < s.layers; l++) {
      const n = W.layer(l);
      this.layerW.push({
        qW: w[n.qW], qB: w[n.qB], kW: w[n.kW], kB: w[n.kB], vW: w[n.vW], vB: w[n.vB],
        oW: w[n.oW], oB: w[n.oB], ln1W: w[n.ln1W], ln1B: w[n.ln1B],
        f1W: w[n.f1W], f1B: w[n.f1B], f2W: w[n.f2W], f2B: w[n.f2B],
        ln2W: w[n.ln2W], ln2B: w[n.ln2B],
      });
    }

    this.params = {};
    this.lora = [];
    if (i.mode === 'full') {
      // everything except the embedding tables trains; wd on plain matrices
      // only — biases and LayerNorm params get decay = false.
      const reg = (name: string) => {
        const t = w[name];
        this.params[name] = {
          data: t,
          grad: new Float32Array(t.length),
          m: new Float32Array(t.length),
          v: new Float32Array(t.length),
          decay: name.endsWith('weight') && !name.includes('LayerNorm'),
        };
      };
      reg(W.embLnW);
      reg(W.embLnB);
      for (let l = 0; l < s.layers; l++) {
        const n = W.layer(l);
        for (const key of [
          n.qW, n.qB, n.kW, n.kB, n.vW, n.vB, n.oW, n.oB,
          n.ln1W, n.ln1B, n.f1W, n.f1B, n.f2W, n.f2B, n.ln2W, n.ln2B,
        ]) {
          reg(key);
        }
      }
    } else {
      // LoRA: bases stay frozen; a ~ N(0, 0.01²), b = 0; alpha = 2r → s = 2.
      const r = i.rank;
      if (!(r > 0)) throw new Error('lora mode needs rank > 0');
      const alpha = 2 * r;
      this.loraScale = alpha / r;
      const rng = mulberry(i.seed);
      const randn = () => {
        const u1 = 1 - rng(); // ∈ (0,1] — log(u1) is never -∞
        const u2 = rng();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      };
      const d = s.dModel;
      const f = s.dFfn;
      for (let l = 0; l < s.layers; l++) {
        const mk = (proj: string, iD: number, oD: number): LoraProj => {
          const a = new Float32Array(iD * r);
          for (let p = 0; p < a.length; p++) a[p] = randn() * 0.01;
          const b = new Float32Array(r * oD);
          const pa: Param = {
            data: a, grad: new Float32Array(a.length),
            m: new Float32Array(a.length), v: new Float32Array(a.length),
            decay: true,
          };
          const pb: Param = {
            data: b, grad: new Float32Array(b.length),
            m: new Float32Array(b.length), v: new Float32Array(b.length),
            decay: false,
          };
          this.params[`lora.${l}.${proj}.a`] = pa;
          this.params[`lora.${l}.${proj}.b`] = pb;
          return { a: pa, b: pb };
        };
        this.lora.push({
          q: mk('q', d, d),
          k: mk('k', d, d),
          v: mk('v', d, d),
          o: mk('o', d, d),
          f1: mk('f1', d, f),
          f2: mk('f2', f, d),
        });
      }
    }
    // adapter reload: overwrite freshly-initialised trainables (inference path)
    if (i.initParams) {
      for (const [k, v] of Object.entries(i.initParams)) {
        const p = this.params[k];
        if (p && v.length === p.data.length) p.data.set(v);
      }
    }
    this.cache = null;
    this.adamT = 0;
  }

  async forward(inputIds: Uint32Array, mask: Float32Array): Promise<F> {
    const s = this.spec;
    const d = s.dModel;
    const H = s.heads;
    const dH = s.dHead;
    const ff = s.dFfn;
    const T = inputIds.length;
    if (mask.length !== T) throw new Error(`mask length ${mask.length} ≠ ids ${T}`);
    if (T > this.maxSeq || T > s.maxPos) {
      throw new Error(`sequence length ${T} exceeds maxSeq/maxPos`);
    }

    const w = this.weights;
    const word = w[W.word];
    const pos = w[W.pos];
    const tt = w[W.tokType];
    const emb = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      const id = inputIds[t];
      if (id >= s.vocab) throw new Error(`token id ${id} ≥ vocab ${s.vocab}`);
      const wo = id * d;
      const o = t * d;
      for (let k = 0; k < d; k++) emb[o + k] = word[wo + k] + pos[o + k] + tt[k];
    }
    const embLn = lnFwd(emb, w[W.embLnW], w[W.embLnB], T, d, s.lnEps);
    let x = embLn.y;

    const layers: LayerCache[] = [];
    const sc = this.loraScale;
    const r = this.rank;
    for (let l = 0; l < s.layers; l++) {
      const lw = this.layerW[l];
      const lp = this.lora[l]; // undefined in full mode
      const lh: LayerCache['lh'] = {};
      const xIn = x;

      const q = linear(xIn, lw.qW, lw.qB, T, d, d);
      const k = linear(xIn, lw.kW, lw.kB, T, d, d);
      const v = linear(xIn, lw.vW, lw.vB, T, d, d);
      if (lp) {
        lh.q = loraHidden(xIn, lp.q.a.data, T, d, r);
        loraApply(q, lh.q, lp.q.b.data, T, r, d, sc);
        lh.k = loraHidden(xIn, lp.k.a.data, T, d, r);
        loraApply(k, lh.k, lp.k.b.data, T, r, d, sc);
        lh.v = loraHidden(xIn, lp.v.a.data, T, d, r);
        loraApply(v, lh.v, lp.v.b.data, T, r, d, sc);
      }

      const { p, ctx } = attend(q, k, v, mask, T, d, H, dH);

      const attn = linear(ctx, lw.oW, lw.oB, T, d, d);
      if (lp) {
        lh.o = loraHidden(ctx, lp.o.a.data, T, d, r);
        loraApply(attn, lh.o, lp.o.b.data, T, r, d, sc);
      }
      const s1 = new Float32Array(T * d);
      for (let j = 0; j < s1.length; j++) s1[j] = xIn[j] + attn[j];
      const ln1 = lnFwd(s1, lw.ln1W, lw.ln1B, T, d, s.lnEps);
      const xAttn = ln1.y;

      const h1pre = linear(xAttn, lw.f1W, lw.f1B, T, d, ff);
      if (lp) {
        lh.f1 = loraHidden(xAttn, lp.f1.a.data, T, d, r);
        loraApply(h1pre, lh.f1, lp.f1.b.data, T, r, ff, sc);
      }
      const h1 = new Float32Array(T * ff);
      for (let j = 0; j < h1.length; j++) h1[j] = gelu(h1pre[j]);
      const h2 = linear(h1, lw.f2W, lw.f2B, T, ff, d);
      if (lp) {
        lh.f2 = loraHidden(h1, lp.f2.a.data, T, ff, r);
        loraApply(h2, lh.f2, lp.f2.b.data, T, r, d, sc);
      }
      const s2 = new Float32Array(T * d);
      for (let j = 0; j < s2.length; j++) s2[j] = xAttn[j] + h2[j];
      const ln2 = lnFwd(s2, lw.ln2W, lw.ln2B, T, d, s.lnEps);

      layers.push({ xIn, q, k, v, probs: p, ctx, ln1, xAttn, h1pre, h1, ln2, lh });
      x = ln2.y;
    }
    this.cache = { T, embLn, layers };
    return x;
  }

  async backward(hiddenGrad: Float32Array): Promise<void> {
    const c = this.cache;
    if (!c) throw new Error('backward() needs a matching forward() first');
    const s = this.spec;
    const d = s.dModel;
    const H = s.heads;
    const dH = s.dHead;
    const ff = s.dFfn;
    const T = c.T;
    if (hiddenGrad.length !== T * d) {
      throw new Error(`hiddenGrad length ${hiddenGrad.length} ≠ ${T * d}`);
    }
    const g = (name: string) => this.params[name]?.grad;
    const lg = (p: LoraProj | undefined, lh: F | undefined): LoraGrads | undefined =>
      p && lh ? { a: p.a.data, b: p.b.data, ga: p.a.grad, gb: p.b.grad, lh } : undefined;
    const sc = this.loraScale;

    let dx = new Float32Array(hiddenGrad);
    for (let l = s.layers - 1; l >= 0; l--) {
      const lc = c.layers[l];
      const lw = this.layerW[l];
      const lp = this.lora[l];
      const n = W.layer(l);

      // out = LN2(xAttn + h2) — residual split of ds2 happens below
      const ds2 = lnBwd(dx, lc.ln2.xhat, lc.ln2.rstd, lw.ln2W, T, d, g(n.ln2W), g(n.ln2B));
      // h2 = h1·f2Wᵀ + f2B
      const dh1 = linBwd(ds2, lc.h1, T, ff, d, lw.f2W, g(n.f2W), g(n.f2B), lg(lp?.f2, lc.lh.f2), sc);
      // h1 = gelu(h1pre)
      const dh1pre = new Float32Array(T * ff);
      for (let j = 0; j < dh1pre.length; j++) dh1pre[j] = dh1[j] * geluGrad(lc.h1pre[j]);
      // h1pre = xAttn·f1Wᵀ + f1B ; + residual
      const dxA = linBwd(dh1pre, lc.xAttn, T, d, ff, lw.f1W, g(n.f1W), g(n.f1B), lg(lp?.f1, lc.lh.f1), sc);
      for (let j = 0; j < dxA.length; j++) dxA[j] += ds2[j];
      // xAttn = LN1(xIn + attn)
      const ds1 = lnBwd(dxA, lc.ln1.xhat, lc.ln1.rstd, lw.ln1W, T, d, g(n.ln1W), g(n.ln1B));
      // attn = ctx·oWᵀ + oB
      const dctx = linBwd(ds1, lc.ctx, T, d, d, lw.oW, g(n.oW), g(n.oB), lg(lp?.o, lc.lh.o), sc);
      // attention → dq, dk, dv in [T,d]
      const { dq, dk, dv } = attendBwd(dctx, lc.q, lc.k, lc.v, lc.probs, T, d, H, dH);
      // q/k/v projections all read xIn; + residual
      const dxq = linBwd(dq, lc.xIn, T, d, d, lw.qW, g(n.qW), g(n.qB), lg(lp?.q, lc.lh.q), sc);
      const dxk = linBwd(dk, lc.xIn, T, d, d, lw.kW, g(n.kW), g(n.kB), lg(lp?.k, lc.lh.k), sc);
      const dxv = linBwd(dv, lc.xIn, T, d, d, lw.vW, g(n.vW), g(n.vB), lg(lp?.v, lc.lh.v), sc);
      const dxIn = new Float32Array(T * d);
      for (let j = 0; j < dxIn.length; j++) dxIn[j] = dxq[j] + dxk[j] + dxv[j] + ds1[j];
      dx = dxIn;
    }
    // embedding LN: accumulate grads (full mode); dx into the frozen
    // embedding tables is the chain endpoint — computed then discarded.
    const w = this.weights;
    lnBwd(dx, c.embLn.xhat, c.embLn.rstd, w[W.embLnW], T, d, g(W.embLnW), g(W.embLnB));
    this.cache = null; // release the activation cache
  }

  async applyGrads(lr: number): Promise<void> {
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const wd = 0.01;
    this.adamT++;
    const bc1 = 1 - Math.pow(b1, this.adamT);
    const bc2 = 1 - Math.pow(b2, this.adamT);
    for (const p of Object.values(this.params)) {
      const { data, grad, m, v, decay } = p;
      const w = decay ? wd : 0;
      for (let i = 0; i < data.length; i++) {
        const gi = grad[i];
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        const mh = m[i] / bc1;
        const vh = v[i] / bc2;
        data[i] -= lr * (mh / (Math.sqrt(vh) + eps) + w * data[i]);
        grad[i] = 0;
      }
    }
  }

  async trainableWeights(): Promise<Record<string, F>> {
    const out: Record<string, F> = {};
    for (const [name, p] of Object.entries(this.params)) out[name] = p.data.slice();
    return out;
  }
}
