/**
 * deberta.ts — CPU reference encoder for DeBERTa-v3 (disentangled attention)
 * with LoRA or full fine-tuning. Mirrors cpu.ts conventions:
 *  - HF linear weights are [out, in] row-major.
 *  - share_att_key: pos_key = key_proj(relEmbLN), pos_query = query_proj(relEmbLN).
 *  - score[i,j] = (q_i·k_j + q_i·pk[Δ] + k_j·pq[Δ]) / √(dH·3)
 *    where Δ(i,j) = clamp(i − j + span, 0, 2span−1) for BOTH pos terms:
 *    HF gathers p2c over the key axis then transposes, which flips the index
 *    to the same Δ as c2p.
 *  - rel_embeddings and word embeddings stay frozen in every mode (embedding
 *    tables are never trained — same rule as cpu.ts).
 *
 * The './spec.ts' extension is deliberate — Node ≥22 runs this via type
 * stripping for grad-checks without a bundler.
 */

import {
  DW,
  type DebertaSpec,
  type EncoderBackend,
  type EncoderInit,
  type Weights,
} from './spec.ts';
import {
  MASK_BIAS,
  mulberry,
  gelu,
  geluGrad,
  linear,
  loraHidden,
  loraApply,
  lnFwd,
  lnBwd,
  linBwd,
  type F,
  type LnCache,
  type LoraProj,
  type Param,
} from './cpu.ts';

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

interface LoraL {
  q: LoraProj;
  k: LoraProj;
  v: LoraProj;
  o: LoraProj;
  f1: LoraProj;
  f2: LoraProj;
}

interface LayerCache {
  xIn: F;
  q: F;
  k: F;
  v: F;
  pq: F; // [P, d] pos-query = query_proj(relEmbLN)
  pk: F; // [P, d] pos-key   = key_proj(relEmbLN)
  probs: F; // [H,T,T]
  ctx: F;
  ln1: LnCache;
  xAttn: F;
  h1pre: F;
  h1: F;
  ln2: LnCache;
  lh: { q?: F; k?: F; v?: F; o?: F; f1?: F; f2?: F; pq?: F; pk?: F };
}

interface FwdCache {
  T: number;
  embLn: LnCache;
  relLn: { y: F } & LnCache;
  layers: LayerCache[];
}

/** rel-pos gather index — same Δ for both c2p and p2c terms. */
function posIndex(i: number, j: number, span: number): number {
  const d = i - j + span;
  return d < 0 ? 0 : d > 2 * span - 1 ? 2 * span - 1 : d;
}

/**
 * Disentangled attention forward.
 *   score[h,i,j] = (q·k + q·pk[Δ] + k·pq[Δ])·inv + maskBias,  Δ = clamp(i−j+span)
 *   P = softmax_j(score);  ctx = P·v
 */
function attendRel(
  q: F,
  k: F,
  v: F,
  pq: F,
  pk: F,
  mask: F,
  T: number,
  d: number,
  H: number,
  dH: number,
  span: number,
  inv: number,
): { p: F; ctx: F } {
  const p = new Float32Array(H * T * T);
  const ctx = new Float32Array(T * d);
  for (let h = 0; h < H; h++) {
    const hd = h * dH;
    for (let i = 0; i < T; i++) {
      const qi = i * d + hd;
      const po = (h * T + i) * T;
      let mx = -Infinity;
      for (let j = 0; j < T; j++) {
        const kj = j * d + hd;
        const dp = posIndex(i, j, span) * d + hd;
        const pko = dp;
        const pqo = dp;
        let acc = 0;
        let ac2p = 0;
        let ap2c = 0;
        for (let u = 0; u < dH; u++) {
          acc += q[qi + u] * k[kj + u];
          ac2p += q[qi + u] * pk[pko + u];
          ap2c += k[kj + u] * pq[pqo + u];
        }
        const sc = (acc + ac2p + ap2c) * inv + (1 - mask[j]) * MASK_BIAS;
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
 * Backward. With S = softmax⁻¹(P) per row:
 *   dP[i,j] = dctx[i]·v[j];   dS[i,j] = P[i,j]·(dP − Σ_l P[i,l]dP[i,l])
 *   dq[i]  += Σ_j dS·(k_j + pk[Δ])·inv            dk[j] += Σ_i dS·(q_i + pq[Δ])·inv
 *   dpk[Δ] += dS·q_i·inv                          dpq[Δ] += dS·k_j·inv
 */
function attendRelBwd(
  dctx: F,
  q: F,
  k: F,
  v: F,
  pq: F,
  pk: F,
  p: F,
  T: number,
  d: number,
  H: number,
  dH: number,
  span: number,
  inv: number,
): { dq: F; dk: F; dv: F; dpq: F; dpk: F } {
  const dq = new Float32Array(T * d);
  const dk = new Float32Array(T * d);
  const dv = new Float32Array(T * d);
  const P = 2 * span;
  const dpq = new Float32Array(P * d);
  const dpk = new Float32Array(P * d);
  const ds = new Float32Array(T);
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
        ds[j] = acc;
        dot += p[po + j] * acc;
      }
      for (let j = 0; j < T; j++) ds[j] = p[po + j] * (ds[j] - dot) * inv;
      const qi = i * d + hd;
      for (let j = 0; j < T; j++) {
        const g = ds[j];
        if (g === 0) continue;
        const kj = j * d + hd;
        const dp = posIndex(i, j, span) * d + hd;
        const pko = dp;
        const pqo = dp;
        for (let u = 0; u < dH; u++) {
          dq[qi + u] += g * (k[kj + u] + pk[pko + u]);
          dk[kj + u] += g * (q[qi + u] + pq[pqo + u]);
          dpk[pko + u] += g * q[qi + u];
          dpq[pqo + u] += g * k[kj + u];
        }
      }
    }
  }
  return { dq, dk, dv, dpq, dpk };
}

export class DebertaCpuEncoder implements EncoderBackend {
  kind = 'cpu' as const;

  private spec!: DebertaSpec;
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
    this.spec = i.spec as DebertaSpec;
    this.weights = i.weights;
    this.rank = i.rank;
    this.maxSeq = i.maxSeq;
    const s = this.spec;
    if (s.dModel !== s.heads * s.dHead) {
      throw new Error(`spec: dModel ${s.dModel} ≠ heads·dHead ${s.heads * s.dHead}`);
    }
    const need = [DW.word, DW.embLnW, DW.embLnB, DW.relEmb, DW.relLnW, DW.relLnB];
    for (let l = 0; l < s.layers; l++) {
      const n = DW.layer(l);
      need.push(
        n.qW, n.qB, n.kW, n.kB, n.vW, n.vB, n.oW, n.oB,
        n.ln1W, n.ln1B, n.f1W, n.f1B, n.f2W, n.f2B, n.ln2W, n.ln2B,
      );
    }
    const missing = need.filter((k) => !(k in i.weights));
    if (missing.length) {
      throw new Error(`weights missing ${missing.length} tensors, e.g. ${missing[0]}`);
    }
    if (i.weights[DW.relEmb].length !== 2 * s.posSpan * s.dModel) {
      throw new Error(
        `rel_embeddings ${i.weights[DW.relEmb].length} ≠ 2·posSpan·d ${2 * s.posSpan * s.dModel}`,
      );
    }

    const w = i.weights;
    this.layerW = [];
    for (let l = 0; l < s.layers; l++) {
      const n = DW.layer(l);
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
      reg(DW.embLnW);
      reg(DW.embLnB);
      reg(DW.relLnW);
      reg(DW.relLnB);
      for (let l = 0; l < s.layers; l++) {
        const n = DW.layer(l);
        for (const key of [
          n.qW, n.qB, n.kW, n.kB, n.vW, n.vB, n.oW, n.oB,
          n.ln1W, n.ln1B, n.f1W, n.f1B, n.f2W, n.f2B, n.ln2W, n.ln2B,
        ]) {
          reg(key);
        }
      }
    } else {
      const r = i.rank;
      if (!(r > 0)) throw new Error('lora mode needs rank > 0');
      const alpha = 2 * r;
      this.loraScale = alpha / r;
      const rng = mulberry(i.seed);
      const randn = () => {
        const u1 = 1 - rng();
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
            data: b, grad: new Float32Array(b.length), m: new Float32Array(b.length),
            v: new Float32Array(b.length), decay: false,
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
    const span = s.posSpan;
    const P = 2 * span;
    const inv = 1 / Math.sqrt(dH * s.scaleFactor);
    const T = inputIds.length;
    if (mask.length !== T) throw new Error(`mask length ${mask.length} ≠ ids ${T}`);
    if (T > this.maxSeq || T > s.maxPos) {
      throw new Error(`sequence length ${T} exceeds maxSeq/maxPos`);
    }

    const w = this.weights;
    const word = w[DW.word];
    const emb = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      const id = inputIds[t];
      if (id >= s.vocab) throw new Error(`token id ${id} ≥ vocab ${s.vocab}`);
      const wo = id * d;
      const o = t * d;
      for (let k = 0; k < d; k++) emb[o + k] = word[wo + k];
    }
    const embLn = lnFwd(emb, w[DW.embLnW], w[DW.embLnB], T, d, s.lnEps);
    let x = embLn.y;

    // norm_rel_ebd = layer_norm: rel table is LN'd once, then projected by
    // each layer's own query/key weights (share_att_key).
    const relLn = lnFwd(w[DW.relEmb], w[DW.relLnW], w[DW.relLnB], P, d, s.lnEps);
    const relY = relLn.y;

    const layers: LayerCache[] = [];
    const sc = this.loraScale;
    const r = this.rank;
    for (let l = 0; l < s.layers; l++) {
      const lw = this.layerW[l];
      const lp = this.lora[l];
      const lh: LayerCache['lh'] = {};
      const xIn = x;

      const q = linear(xIn, lw.qW, lw.qB, T, d, d);
      const k = linear(xIn, lw.kW, lw.kB, T, d, d);
      const v = linear(xIn, lw.vW, lw.vB, T, d, d);
      // pos projections share the q/k weights + adapters
      const pq = linear(relY, lw.qW, lw.qB, P, d, d);
      const pk = linear(relY, lw.kW, lw.kB, P, d, d);
      if (lp) {
        lh.q = loraHidden(xIn, lp.q.a.data, T, d, r);
        loraApply(q, lh.q, lp.q.b.data, T, r, d, sc);
        lh.k = loraHidden(xIn, lp.k.a.data, T, d, r);
        loraApply(k, lh.k, lp.k.b.data, T, r, d, sc);
        lh.v = loraHidden(xIn, lp.v.a.data, T, d, r);
        loraApply(v, lh.v, lp.v.b.data, T, r, d, sc);
        lh.pq = loraHidden(relY, lp.q.a.data, P, d, r);
        loraApply(pq, lh.pq, lp.q.b.data, P, r, d, sc);
        lh.pk = loraHidden(relY, lp.k.a.data, P, d, r);
        loraApply(pk, lh.pk, lp.k.b.data, P, r, d, sc);
      }

      const { p, ctx } = attendRel(q, k, v, pq, pk, mask, T, d, H, dH, span, inv);

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

      layers.push({ xIn, q, k, v, pq, pk, probs: p, ctx, ln1, xAttn, h1pre, h1, ln2, lh });
      x = ln2.y;
    }
    this.cache = { T, embLn, relLn, layers };
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
    const span = s.posSpan;
    const P = 2 * span;
    const inv = 1 / Math.sqrt(dH * s.scaleFactor);
    const T = c.T;
    if (hiddenGrad.length !== T * d) {
      throw new Error(`hiddenGrad length ${hiddenGrad.length} ≠ ${T * d}`);
    }
    const g = (name: string) => this.params[name]?.grad;
    const lg = (p: LoraProj | undefined, lh: F | undefined) =>
      p && lh ? { a: p.a.data, b: p.b.data, ga: p.a.grad, gb: p.b.grad, lh } : undefined;
    const sc = this.loraScale;
    const relY = c.relLn.y;

    // shared rel-embedding LN gradient accumulator
    let dRelY = new Float32Array(P * d);

    let dx = new Float32Array(hiddenGrad);
    for (let l = s.layers - 1; l >= 0; l--) {
      const lc = c.layers[l];
      const lw = this.layerW[l];
      const lp = this.lora[l];
      const n = DW.layer(l);

      const ds2 = lnBwd(dx, lc.ln2.xhat, lc.ln2.rstd, lw.ln2W, T, d, g(n.ln2W), g(n.ln2B));
      const dh1 = linBwd(ds2, lc.h1, T, ff, d, lw.f2W, g(n.f2W), g(n.f2B), lg(lp?.f2, lc.lh.f2), sc);
      const dh1pre = new Float32Array(T * ff);
      for (let j = 0; j < dh1pre.length; j++) dh1pre[j] = dh1[j] * geluGrad(lc.h1pre[j]);
      const dxA = linBwd(dh1pre, lc.xAttn, T, d, ff, lw.f1W, g(n.f1W), g(n.f1B), lg(lp?.f1, lc.lh.f1), sc);
      for (let j = 0; j < dxA.length; j++) dxA[j] += ds2[j];
      const ds1 = lnBwd(dxA, lc.ln1.xhat, lc.ln1.rstd, lw.ln1W, T, d, g(n.ln1W), g(n.ln1B));
      const dctx = linBwd(ds1, lc.ctx, T, d, d, lw.oW, g(n.oW), g(n.oB), lg(lp?.o, lc.lh.o), sc);
      const { dq, dk, dv, dpq, dpk } = attendRelBwd(
        dctx, lc.q, lc.k, lc.v, lc.pq, lc.pk, lc.probs, T, d, H, dH, span, inv,
      );
      const dxq = linBwd(dq, lc.xIn, T, d, d, lw.qW, g(n.qW), g(n.qB), lg(lp?.q, lc.lh.q), sc);
      const dxk = linBwd(dk, lc.xIn, T, d, d, lw.kW, g(n.kW), g(n.kB), lg(lp?.k, lc.lh.k), sc);
      const dxv = linBwd(dv, lc.xIn, T, d, d, lw.vW, g(n.vW), g(n.vB), lg(lp?.v, lc.lh.v), sc);
      // pos projections backprop through the SAME q/k weights and adapters —
      // qW/kW (and lora a/b) collect grads from both content and pos inputs.
      const dRelQ = linBwd(dpq, relY, P, d, d, lw.qW, g(n.qW), g(n.qB), lg(lp?.q, lc.lh.pq), sc);
      const dRelK = linBwd(dpk, relY, P, d, d, lw.kW, g(n.kW), g(n.kB), lg(lp?.k, lc.lh.pk), sc);
      for (let j = 0; j < dRelY.length; j++) dRelY[j] += dRelQ[j] + dRelK[j];

      const dxIn = new Float32Array(T * d);
      for (let j = 0; j < dxIn.length; j++) dxIn[j] = dxq[j] + dxk[j] + dxv[j] + ds1[j];
      dx = dxIn;
    }
    const w = this.weights;
    // encoder.LayerNorm on the rel table — grad path ends at the frozen
    // rel_embeddings, but the LN params train in full mode.
    lnBwd(dRelY, c.relLn.xhat, c.relLn.rstd, w[DW.relLnW], P, d, g(DW.relLnW), g(DW.relLnB));
    // embeddings LN (full mode); dx into frozen word embeddings is discarded.
    lnBwd(dx, c.embLn.xhat, c.embLn.rstd, w[DW.embLnW], T, d, g(DW.embLnW), g(DW.embLnB));
    this.cache = null;
  }

  async applyGrads(lr: number, batchSize = 1): Promise<void> {
    const gradScale = 1 / Math.max(1, batchSize);
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
        const gi = grad[i] * gradScale;
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
