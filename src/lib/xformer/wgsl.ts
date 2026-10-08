import { W, type EncoderBackend, type EncoderInit } from './spec';

/**
 * WGSL (WebGPU compute) backend for the BERT encoder — forward AND backward.
 * This is the engine that makes in-browser LoRA / full fine-tuning real.
 *
 * Conventions (mirroring spec.ts):
 *  - HF weights are [out, in] row-major: y[t,o] = Σ_i x[t,i]·W[o,i] + b[o].
 *  - Attention heads are a view into [T, D]: head h occupies columns
 *    h·DH .. h·DH+DH-1. No physical transpose — kernels index with h = d/DH.
 *  - GELU is the exact erf form 0.5x(1+erf(x/√2)); erf is evaluated with the
 *    Abramowitz–Stegun 7.1.26 polynomial (|ε| ≤ 1.5e-7). Backward uses the
 *    analytic derivative gelu'(x) = Φ(x) + x·φ(x) where Φ is computed with
 *    the SAME erf approximation, so fwd/bwd are self-consistent.
 *  - LoRA: per (layer, proj∈{q,k,v,o,f1,f2}) trainable A[in,r] ~ U(-σ,σ)
 *    with σ = √(2/(in+out))·0.1 (mulberry32 seeded by init.seed) and
 *    B[r,out] = 0; scale s = α/r = 2 baked into the shaders as LS.
 *  - All dx-producing kernels ACCUMULATE (+=) into zeroed scratch so that
 *    residual forks and q/k/v fan-in need no extra pass; the two residual
 *    branches of a LayerNorm backward get the same value written twice by
 *    ln_bwd_dx (one += into the running accumulator, one = into the sub-path
 *    dY buffer).
 *  - Only T is a runtime uniform; every other dim is baked as a WGSL const.
 *    mean/rstd for LayerNorm live packed in one buffer mr[2·MAXT] to stay
 *    under the 8-storage-buffer-per-stage limit (add_ln uses exactly 8).
 */

const PROJS = ['q', 'k', 'v', 'o', 'f1', 'f2'] as const;
type Proj = (typeof PROJS)[number];

/** one trainable tensor + its grad/m/v AdamW companions */
interface Train {
  name: string;
  p: GPUBuffer;
  g: GPUBuffer;
  m: GPUBuffer;
  v: GPUBuffer;
  len: number;
  bgAdam: GPUBindGroup;
  bgZero: GPUBindGroup;
}

interface Layer {
  /** canonical name → weight buffer (frozen in lora mode, trainable in full) */
  w: Record<string, GPUBuffer>;
  /** lora mode only: A[in,r] / B[r,out] per proj */
  a: Record<Proj, GPUBuffer>;
  b: Record<Proj, GPUBuffer>;
  /** per-layer activation cache, all sized for maxSeq */
  xin: GPUBuffer;
  q: GPUBuffer;
  k: GPUBuffer;
  v: GPUBuffer;
  xa: Record<Proj, GPUBuffer>; // [maxSeq, R] x·A caches (lora)
  P: GPUBuffer; // [H, maxSeq, maxSeq] post-softmax probs
  ctx: GPUBuffer;
  ln1In: GPUBuffer; // xin + attnOut (LN1 input)
  ln1Out: GPUBuffer; // LN1 output = FFN input
  mr1: GPUBuffer; // [2·MAXT] mean | rstd
  h1pre: GPUBuffer; // [maxSeq, F] pre-GELU
  h1: GPUBuffer; // [maxSeq, F] post-GELU (f2 input)
  ln2In: GPUBuffer; // ln1Out + h2 (LN2 input)
  mr2: GPUBuffer;
  bg: Record<string, GPUBindGroup>;
}

const f32lit = (v: number) => {
  const s = String(v);
  return s.includes('.') || s.includes('e') || s.includes('E') ? s : s + '.0';
};

export class WgslEncoder implements EncoderBackend {
  kind = 'wgsl' as const;

  private dev!: GPUDevice;
  private i!: EncoderInit;

  // baked dims
  private D = 0;
  private F = 0;
  private H = 0;
  private DH = 0;
  private R = 0;
  private MAXT = 0;
  private EPS = 0;
  private LS = 2; // LoRA scale alpha/r with alpha=2r

  private T = 0; // current sequence length (set by forward, used by backward)
  private t = 0; // adam step
  private pending: Promise<unknown> = Promise.resolve();

  private p: Record<string, GPUComputePipeline> = {};
  private layers: Layer[] = [];
  private train: Train[] = [];

  // shared buffers
  private uT!: GPUBuffer;
  private uAdamW!: GPUBuffer; // wd = 0.01
  private uAdamZ!: GPUBuffer; // wd = 0
  private idsB!: GPUBuffer;
  private maskB!: GPUBuffer;
  private embWord!: GPUBuffer;
  private embPos!: GPUBuffer;
  private embTok!: GPUBuffer;
  private embLnW!: GPUBuffer;
  private embLnB!: GPUBuffer;
  private embSum!: GPUBuffer; // [T,D] raw embedding sum (embLN input)
  private mrE!: GPUBuffer;
  private x0!: GPUBuffer; // running hidden state
  private x1!: GPUBuffer; // second state buffer (in/out must not alias)
  private attnO!: GPUBuffer;
  private h2!: GPUBuffer;
  private dupSink!: GPUBuffer;
  // backward scratch
  private dXcur!: GPUBuffer;
  private dXacc!: GPUBuffer;
  private dMid!: GPUBuffer;
  private dH2!: GPUBuffer;
  private dSub!: GPUBuffer;
  private dCtx!: GPUBuffer;
  private dQ!: GPUBuffer;
  private dK!: GPUBuffer;
  private dV!: GPUBuffer;
  private dH1!: GPUBuffer;
  private dH1pre!: GPUBuffer;
  private dYB!: GPUBuffer;
  private dPbuf!: GPUBuffer;
  private dS!: GPUBuffer;
  private readB!: GPUBuffer;
  // shared bind groups
  private bg: Record<string, GPUBindGroup> = {};

  static async create(i: EncoderInit): Promise<WgslEncoder> {
    const e = new WgslEncoder();
    if (!navigator.gpu) throw new Error('wgsl: no navigator.gpu');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('wgsl: no adapter');
    e.dev = await adapter.requestDevice();
    await e.init(i);
    return e;
  }

  /** serialize all public calls so the shared staging buffer never double-maps */
  private run<T>(job: () => Promise<T>): Promise<T> {
    const r = this.pending.then(job);
    this.pending = r.then(
      () => undefined,
      () => undefined,
    );
    return r;
  }

  // ------------------------------------------------------------------ init

  async init(i: EncoderInit) {
    this.i = i;
    const dev = this.dev;
    const s = i.spec;
    this.D = s.dModel;
    this.F = s.dFfn;
    this.H = s.heads;
    this.DH = s.dHead;
    this.R = i.mode === 'lora' ? i.rank : 0;
    this.MAXT = i.maxSeq;
    this.EPS = s.lnEps;
    if (this.H * this.DH !== this.D) throw new Error('wgsl: heads*dHead != dModel');
    if (this.MAXT < 1) throw new Error('wgsl: maxSeq < 1');
    if (i.mode === 'lora' && this.R < 1) throw new Error('wgsl: lora rank < 1');

    const U = GPUBufferUsage;
    const buf = (floats: number, usage: number, data?: ArrayBufferView) => {
      const b = dev.createBuffer({ size: Math.max(16, floats * 4), usage });
      if (data) dev.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
      return b;
    };
    const need = (name: string) => {
      const t = i.weights[name];
      if (!t) throw new Error(`wgsl: missing weight ${name}`);
      return t;
    };
    // sanity-check tensor shapes against the spec — a silently truncated
    // weight would corrupt everything downstream
    const check = (name: string, floats: number) => {
      if (need(name).length !== floats)
        throw new Error(`wgsl: ${name} has ${need(name).length} floats, expected ${floats}`);
    };
    check(W.word, s.vocab * this.D);
    check(W.pos, s.maxPos * this.D);
    check(W.tokType, 2 * this.D);
    check(W.embLnW, this.D);
    check(W.embLnB, this.D);
    for (let L = 0; L < s.layers; L++) {
      const lw = W.layer(L);
      for (const [k, n] of [
        [lw.qW, this.D * this.D], [lw.qB, this.D], [lw.kW, this.D * this.D], [lw.kB, this.D],
        [lw.vW, this.D * this.D], [lw.vB, this.D], [lw.oW, this.D * this.D], [lw.oB, this.D],
        [lw.ln1W, this.D], [lw.ln1B, this.D], [lw.f1W, this.F * this.D], [lw.f1B, this.F],
        [lw.f2W, this.D * this.F], [lw.f2B, this.D], [lw.ln2W, this.D], [lw.ln2B, this.D],
      ] as [string, number][]) check(k, n);
    }

    this.buildPipelines();

    // ---- shared buffers -------------------------------------------------
    const TD = this.MAXT * this.D;
    const TF = this.MAXT * this.F;
    const HTT = this.H * this.MAXT * this.MAXT;
    this.uT = buf(4, U.UNIFORM | U.COPY_DST);
    this.uAdamW = buf(8, U.UNIFORM | U.COPY_DST);
    this.uAdamZ = buf(8, U.UNIFORM | U.COPY_DST);
    this.idsB = buf(this.MAXT, U.STORAGE | U.COPY_DST);
    this.maskB = buf(this.MAXT, U.STORAGE | U.COPY_DST);
    this.embWord = buf(s.vocab * this.D, U.STORAGE | U.COPY_DST, need(W.word));
    this.embPos = buf(s.maxPos * this.D, U.STORAGE | U.COPY_DST, need(W.pos));
    this.embTok = buf(2 * this.D, U.STORAGE | U.COPY_DST, need(W.tokType));
    const embFlags = i.mode === 'full' ? U.STORAGE | U.COPY_DST | U.COPY_SRC : U.STORAGE | U.COPY_DST;
    this.embLnW = buf(this.D, embFlags, need(W.embLnW));
    this.embLnB = buf(this.D, embFlags, need(W.embLnB));
    this.embSum = buf(TD, U.STORAGE);
    this.mrE = buf(2 * this.MAXT, U.STORAGE);
    this.x0 = buf(TD, U.STORAGE | U.COPY_SRC);
    this.x1 = buf(TD, U.STORAGE);
    this.attnO = buf(TD, U.STORAGE);
    this.h2 = buf(TD, U.STORAGE);
    this.dupSink = buf(TD, U.STORAGE);
    this.dXcur = buf(TD, U.STORAGE | U.COPY_DST);
    this.dXacc = buf(TD, U.STORAGE);
    this.dMid = buf(TD, U.STORAGE);
    this.dH2 = buf(TD, U.STORAGE);
    this.dSub = buf(TD, U.STORAGE);
    this.dCtx = buf(TD, U.STORAGE);
    this.dQ = buf(TD, U.STORAGE);
    this.dK = buf(TD, U.STORAGE);
    this.dV = buf(TD, U.STORAGE);
    this.dH1 = buf(TF, U.STORAGE);
    this.dH1pre = buf(TF, U.STORAGE);
    this.dYB = buf(Math.max(1, this.MAXT * this.R), U.STORAGE);
    this.dPbuf = buf(HTT, U.STORAGE);
    this.dS = buf(HTT, U.STORAGE);
    const readFloats = Math.max(TD, this.D * this.F);
    this.readB = buf(readFloats, U.MAP_READ | U.COPY_DST);

    // ---- per-layer weights + caches + lora ------------------------------
    // deterministic mulberry32 (same pattern as gpuTrain.ts)
    let rs = i.seed >>> 0;
    const rng = () => {
      rs = (rs + 0x6d2b79f5) | 0;
      let t = Math.imul(rs ^ (rs >>> 15), 1 | rs);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const loraInit = (inn: number, out: number) => {
      const sig = Math.sqrt(2 / (inn + out)) * 0.1;
      const a = new Float32Array(inn * this.R);
      for (let n = 0; n < a.length; n++) a[n] = (rng() * 2 - 1) * sig;
      return a;
    };
    const inOf = (p: Proj) => (p === 'f2' ? this.F : this.D);
    const outOf = (p: Proj) => (p === 'f1' ? this.F : this.D);

    const wFlags = i.mode === 'full' ? U.STORAGE | U.COPY_DST | U.COPY_SRC : U.STORAGE | U.COPY_DST;
    const addTrain = (name: string, p: GPUBuffer, len: number, wd: boolean) => {
      const g = buf(len, U.STORAGE);
      const m = buf(len, U.STORAGE);
      const v = buf(len, U.STORAGE);
      const bgAdam = dev.createBindGroup({
        layout: this.pipeAdam(len).getBindGroupLayout(0),
        entries: [wd ? this.uAdamW : this.uAdamZ, g, p, m, v].map((b, n) => ({
          binding: n,
          resource: { buffer: b },
        })),
      });
      const bgZero = dev.createBindGroup({
        layout: this.pipeZero(len).getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: g } }],
      });
      this.train.push({ name, p, g, m, v, len, bgAdam, bgZero });
    };

    for (let L = 0; L < s.layers; L++) {
      const lw = W.layer(L);
      const keys = [
        lw.qW, lw.qB, lw.kW, lw.kB, lw.vW, lw.vB, lw.oW, lw.oB,
        lw.ln1W, lw.ln1B, lw.f1W, lw.f1B, lw.f2W, lw.f2B, lw.ln2W, lw.ln2B,
      ];
      const w: Record<string, GPUBuffer> = {};
      for (const k of keys) w[k] = buf(need(k).length, wFlags, need(k));

      const layer: Layer = {
        w,
        a: {} as Record<Proj, GPUBuffer>,
        b: {} as Record<Proj, GPUBuffer>,
        xa: {} as Record<Proj, GPUBuffer>,
        xin: buf(TD, U.STORAGE),
        q: buf(TD, U.STORAGE),
        k: buf(TD, U.STORAGE),
        v: buf(TD, U.STORAGE),
        P: buf(HTT, U.STORAGE),
        ctx: buf(TD, U.STORAGE),
        ln1In: buf(TD, U.STORAGE),
        ln1Out: buf(TD, U.STORAGE),
        mr1: buf(2 * this.MAXT, U.STORAGE),
        h1pre: buf(TF, U.STORAGE),
        h1: buf(TF, U.STORAGE),
        ln2In: buf(TD, U.STORAGE),
        mr2: buf(2 * this.MAXT, U.STORAGE),
        bg: {},
      };
      this.layers.push(layer);

      if (i.mode === 'lora') {
        for (const pr of PROJS) {
          const inn = inOf(pr);
          const out = outOf(pr);
          layer.a[pr] = buf(inn * this.R, U.STORAGE | U.COPY_DST | U.COPY_SRC, loraInit(inn, out));
          layer.b[pr] = buf(this.R * out, U.STORAGE | U.COPY_DST | U.COPY_SRC); // zeros
          layer.xa[pr] = buf(this.MAXT * this.R, U.STORAGE);
          addTrain(`lora.${L}.${pr}.a`, layer.a[pr], inn * this.R, true);
          addTrain(`lora.${L}.${pr}.b`, layer.b[pr], this.R * out, false);
        }
      } else {
        for (const k of keys) {
          const isMat = k.endsWith('.weight') && !k.includes('LayerNorm');
          addTrain(k, w[k], need(k).length, isMat);
        }
        if (L === s.layers - 1) {
          // embLN params are trainable in full mode (embeddings themselves frozen)
          addTrain(W.embLnW, this.embLnW, this.D, false);
          addTrain(W.embLnB, this.embLnB, this.D, false);
        }
      }
    }

    // adapter reload: overwrite freshly-initialised trainables (inference path)
    if (i.initParams) {
      for (const t of this.train) {
        const v = i.initParams[t.name];
        if (v && v.length === t.len) {
          dev.queue.writeBuffer(t.p, 0, v.buffer, v.byteOffset, v.byteLength);
        }
      }
    }

    this.buildBindGroups();
  }

  // -------------------------------------------------------------- pipelines

  /** WGSL prologue: baked dims + uniform seq + erf/gelu helpers */
  private prologue(): string {
    const { D, F, H, DH, R, MAXT, EPS, LS } = this;
    return /* wgsl */ `
struct Seq { T: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> seq: Seq;
const D: u32 = ${D}u;
const F: u32 = ${F}u;
const H: u32 = ${H}u;
const DH: u32 = ${DH}u;
const R: u32 = ${Math.max(1, R)}u;
const MAXT: u32 = ${MAXT}u;
const EPS: f32 = ${f32lit(EPS)};
const RSQRT_DH: f32 = ${f32lit(1 / Math.sqrt(DH))};
const LS: f32 = ${f32lit(LS)};
const RSQRT2: f32 = 0.70710678118654752440;
const RSQRT2PI: f32 = 0.39894228040143267794; // 1/sqrt(2π)

// Abramowitz–Stegun 7.1.26 erf approximation, |ε| <= 1.5e-7
fn erfA(x: f32) -> f32 {
  let sgn = sign(x);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return sgn * (1.0 - poly * exp(-ax * ax));
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erfA(x * RSQRT2)); }
// d/dx [0.5x(1+erf(x/√2))] = Φ(x) + x·φ(x); Φ uses the same erfA for consistency
fn geluDx(x: f32) -> f32 {
  return 0.5 * (1.0 + erfA(x * RSQRT2)) + x * RSQRT2PI * exp(-0.5 * x * x);
}
`;
  }

  private mk(name: string, code: string, entry: string) {
    const mod = this.dev.createShaderModule({ code });
    this.p[name] = this.dev.createComputePipeline({
      layout: 'auto',
      compute: { module: mod, entryPoint: entry },
    });
  }

  private buildPipelines() {
    const C = this.prologue();
    const mk = (n: string, code: string, e: string) => this.mk(n, code, e);

    mk('embed', `${C}
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> word: array<f32>;
@group(0) @binding(3) var<storage, read> pos: array<f32>;
@group(0) @binding(4) var<storage, read> tok: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
// emb[t,d] = word[ids[t],d] + pos[t,d] + tok[0,d]
@compute @workgroup_size(8, 8)
fn embed(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x; let t = g.y;
  if (d >= D || t >= seq.T) { return; }
  out[t * D + d] = word[ids[t] * D + d] + pos[t * D + d] + tok[d];
}`, 'embed');

    mk('ln', `${C}
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> gam: array<f32>;
@group(0) @binding(3) var<storage, read> bet: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> mr: array<f32>; // [2*MAXT] mean|rstd
// one thread per row: out = LN(inp); caches mean/rstd
@compute @workgroup_size(64)
fn lnFwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= seq.T) { return; }
  let base = t * D;
  var mean = 0.0;
  for (var d = 0u; d < D; d++) { mean += inp[base + d]; }
  mean /= f32(D);
  var va = 0.0;
  for (var d = 0u; d < D; d++) { let c = inp[base + d] - mean; va += c * c; }
  va /= f32(D);
  let rstd = 1.0 / sqrt(va + EPS);
  mr[t] = mean;
  mr[MAXT + t] = rstd;
  for (var d = 0u; d < D; d++) {
    out[base + d] = (inp[base + d] - mean) * rstd * gam[d] + bet[d];
  }
}`, 'lnFwd');

    mk('addln', `${C}
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> gam: array<f32>;
@group(0) @binding(4) var<storage, read> bet: array<f32>;
@group(0) @binding(5) var<storage, read_write> lnIn: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> dup: array<f32>;
@group(0) @binding(8) var<storage, read_write> mr: array<f32>;
// one thread per row: lnIn = a+b (cached); out = dup = LN(lnIn)
@compute @workgroup_size(64)
fn addLn(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= seq.T) { return; }
  let base = t * D;
  var mean = 0.0;
  for (var d = 0u; d < D; d++) {
    let s = a[base + d] + b[base + d];
    lnIn[base + d] = s;
    mean += s;
  }
  mean /= f32(D);
  var va = 0.0;
  for (var d = 0u; d < D; d++) { let c = lnIn[base + d] - mean; va += c * c; }
  va /= f32(D);
  let rstd = 1.0 / sqrt(va + EPS);
  mr[t] = mean;
  mr[MAXT + t] = rstd;
  for (var d = 0u; d < D; d++) {
    let o = (lnIn[base + d] - mean) * rstd * gam[d] + bet[d];
    out[base + d] = o;
    dup[base + d] = o;
  }
}`, 'addLn');

    const linFwd = (IN: number, OUT: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>; // [OUT, IN]
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
const IN: u32 = ${IN}u;
const OUT: u32 = ${OUT}u;
// y[t,o] = Σ_i x[t,i]·w[o,i] + b[o]
@compute @workgroup_size(8, 8)
fn linFwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let o = g.x; let t = g.y;
  if (o >= OUT || t >= seq.T) { return; }
  var s = b[o];
  for (var i = 0u; i < IN; i++) { s += x[t * IN + i] * w[o * IN + i]; }
  y[t * OUT + o] = s;
}`;
    mk('linDD', C + linFwd(this.D, this.D), 'linFwd');
    mk('linDF', C + linFwd(this.D, this.F), 'linFwd');
    mk('linFD', C + linFwd(this.F, this.D), 'linFwd');

    if (this.i.mode === 'lora') {
      const loraA = (IN: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>; // [IN, R]
@group(0) @binding(3) var<storage, read_write> xa: array<f32>;
const IN: u32 = ${IN}u;
// xa[t,r] = Σ_i x[t,i]·A[i,r]
@compute @workgroup_size(8, 8)
fn loraA(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x; let t = g.y;
  if (r >= R || t >= seq.T) { return; }
  var s = 0.0;
  for (var i = 0u; i < IN; i++) { s += x[t * IN + i] * A[i * R + r]; }
  xa[t * R + r] = s;
}`;
      const loraB = (OUT: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> xa: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>; // [R, OUT]
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
const OUT: u32 = ${OUT}u;
// y[t,o] += LS · Σ_r xa[t,r]·B[r,o]
@compute @workgroup_size(8, 8)
fn loraB(@builtin(global_invocation_id) g: vec3<u32>) {
  let o = g.x; let t = g.y;
  if (o >= OUT || t >= seq.T) { return; }
  var s = 0.0;
  for (var r = 0u; r < R; r++) { s += xa[t * R + r] * B[r * OUT + o]; }
  y[t * OUT + o] += LS * s;
}`;
      mk('laD', C + loraA(this.D), 'loraA');
      mk('laF', C + loraA(this.F), 'loraA');
      mk('lbD', C + loraB(this.D), 'loraB');
      mk('lbF', C + loraB(this.F), 'loraB');
    }

    mk('scores', `${C}
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> msk: array<f32>;
@group(0) @binding(4) var<storage, read_write> P: array<f32>; // [H,T,T]
// P[h,i,j] = q_h[i]·k_h[j]/√DH + (mask[j]==0 ? -1e4 : 0)
@compute @workgroup_size(8, 8)
fn scores(@builtin(global_invocation_id) g: vec3<u32>) {
  let j = g.x; let i = g.y; let h = g.z;
  if (i >= seq.T || j >= seq.T || h >= H) { return; }
  var s = 0.0;
  for (var d = 0u; d < DH; d++) { s += q[i * D + h * DH + d] * k[j * D + h * DH + d]; }
  s *= RSQRT_DH;
  if (msk[j] == 0.0) { s -= 1e4; }
  P[(h * seq.T + i) * seq.T + j] = s;
}`, 'scores');

    mk('softmax', `${C}
@group(0) @binding(1) var<storage, read_write> P: array<f32>;
// one thread per (h,i) row over j — in-place softmax
@compute @workgroup_size(64)
fn softmaxRows(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x;
  if (r >= H * seq.T) { return; }
  let base = r * seq.T;
  var m = P[base];
  for (var j = 1u; j < seq.T; j++) { m = max(m, P[base + j]); }
  var sum = 0.0;
  for (var j = 0u; j < seq.T; j++) { let e = exp(P[base + j] - m); P[base + j] = e; sum += e; }
  for (var j = 0u; j < seq.T; j++) { P[base + j] /= sum; }
}`, 'softmaxRows');

    mk('ctx', `${C}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> ctx: array<f32>;
// ctx[i, h·DH+dh] = Σ_j P[h,i,j]·v[j, h·DH+dh]
@compute @workgroup_size(8, 8)
fn ctxFwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x; let i = g.y;
  if (d >= D || i >= seq.T) { return; }
  let h = d / DH;
  var s = 0.0;
  for (var j = 0u; j < seq.T; j++) { s += P[(h * seq.T + i) * seq.T + j] * v[j * D + d]; }
  ctx[i * D + d] = s;
}`, 'ctxFwd');

    mk('gelu', `${C}
@group(0) @binding(1) var<storage, read> pre: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(8, 8)
fn geluFwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let f = g.x; let t = g.y;
  if (f >= F || t >= seq.T) { return; }
  out[t * F + f] = gelu(pre[t * F + f]);
}`, 'geluFwd');

    mk('cp', `${C}
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
// dst[i] = src[i] for i < T*D
@compute @workgroup_size(256)
fn cp(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= seq.T * D) { return; }
  dst[i] = src[i];
}`, 'cp');

    // ---- backward ------------------------------------------------------
    mk('lnDx', `${C}
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> mr: array<f32>;
@group(0) @binding(4) var<storage, read> gam: array<f32>;
@group(0) @binding(5) var<storage, read_write> dxA: array<f32>;
@group(0) @binding(6) var<storage, read_write> dxB: array<f32>;
// per row: dxhat = dout·γ; dx = rstd·(dxhat − mean(dxhat) − xhat·mean(dxhat·xhat))
// writes the same dx into dxA (+= residual accumulator) and dxB (= sub-path dY)
@compute @workgroup_size(64)
fn lnBwdDx(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= seq.T) { return; }
  let base = t * D;
  let mean = mr[t];
  let rstd = mr[MAXT + t];
  var m1 = 0.0;
  var m2 = 0.0;
  for (var d = 0u; d < D; d++) {
    let dy = dout[base + d] * gam[d];
    let xh = (inp[base + d] - mean) * rstd;
    m1 += dy;
    m2 += dy * xh;
  }
  m1 /= f32(D);
  m2 /= f32(D);
  for (var d = 0u; d < D; d++) {
    let dy = dout[base + d] * gam[d];
    let xh = (inp[base + d] - mean) * rstd;
    let v = (dy - m1 - xh * m2) * rstd;
    dxA[base + d] += v;
    dxB[base + d] = v;
  }
}`, 'lnBwdDx');

    mk('lnP', `${C}
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> mr: array<f32>;
@group(0) @binding(4) var<storage, read_write> dg: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
// per column: dγ[d] += Σ_t dout·xhat ; dβ[d] += Σ_t dout
@compute @workgroup_size(64)
fn lnBwdP(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x;
  if (d >= D) { return; }
  var sg = 0.0;
  var sb = 0.0;
  for (var t = 0u; t < seq.T; t++) {
    let xh = (inp[t * D + d] - mr[t]) * mr[MAXT + t];
    sg += dout[t * D + d] * xh;
    sb += dout[t * D + d];
  }
  dg[d] += sg;
  db[d] += sb;
}`, 'lnBwdP');

    const linBwdDx = (IN: number, OUT: number, lora: boolean) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>; // [OUT, IN]
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
${lora ? `@group(0) @binding(4) var<storage, read> A: array<f32>; // [IN, R]
@group(0) @binding(5) var<storage, read> dyb: array<f32>; // [T, R] = dY·Bᵀ` : ''}
const IN: u32 = ${IN}u;
const OUT: u32 = ${OUT}u;
// dx[t,i] += Σ_o dY[t,o]·w[o,i] ${lora ? '+ LS·Σ_r dyb[t,r]·A[i,r]' : ''}
@compute @workgroup_size(8, 8)
fn linBwdDx(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; let t = g.y;
  if (i >= IN || t >= seq.T) { return; }
  var s = 0.0;
  for (var o = 0u; o < OUT; o++) { s += dY[t * OUT + o] * w[o * IN + i]; }
${lora ? `  var s2 = 0.0;
  for (var r = 0u; r < R; r++) { s2 += dyb[t * R + r] * A[i * R + r]; }
  s += LS * s2;` : ''}
  dx[t * IN + i] += s;
}`;
    const linBwdDw = (IN: number, OUT: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;
const IN: u32 = ${IN}u;
const OUT: u32 = ${OUT}u;
// dw[o,i] += Σ_t dY[t,o]·x[t,i]
@compute @workgroup_size(8, 8)
fn linBwdDw(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; let o = g.y;
  if (i >= IN || o >= OUT) { return; }
  var s = 0.0;
  for (var t = 0u; t < seq.T; t++) { s += dY[t * OUT + o] * x[t * IN + i]; }
  dw[o * IN + i] += s;
}`;
    const linBwdDb = (OUT: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;
const OUT: u32 = ${OUT}u;
// db[o] += Σ_t dY[t,o]
@compute @workgroup_size(64)
fn linBwdDb(@builtin(global_invocation_id) g: vec3<u32>) {
  let o = g.x;
  if (o >= OUT) { return; }
  var s = 0.0;
  for (var t = 0u; t < seq.T; t++) { s += dY[t * OUT + o]; }
  db[o] += s;
}`;

    if (this.i.mode === 'lora') {
      mk('bdxDD', C + linBwdDx(this.D, this.D, true), 'linBwdDx');
      mk('bdxDF', C + linBwdDx(this.D, this.F, true), 'linBwdDx');
      mk('bdxFD', C + linBwdDx(this.F, this.D, true), 'linBwdDx');
      const dyb = (OUT: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>; // [R, OUT]
@group(0) @binding(3) var<storage, read_write> dyb: array<f32>; // [T, R]
const OUT: u32 = ${OUT}u;
// dyb[t,r] = Σ_o dY[t,o]·B[r,o]
@compute @workgroup_size(8, 8)
fn loraDyb(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x; let t = g.y;
  if (r >= R || t >= seq.T) { return; }
  var s = 0.0;
  for (var o = 0u; o < OUT; o++) { s += dY[t * OUT + o] * B[r * OUT + o]; }
  dyb[t * R + r] = s;
}`;
      const loraDb = (OUT: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> xa: array<f32>; // [T, R]
@group(0) @binding(2) var<storage, read> dY: array<f32>;
@group(0) @binding(3) var<storage, read_write> dB: array<f32>; // [R, OUT]
const OUT: u32 = ${OUT}u;
// dB[r,o] += LS·Σ_t xa[t,r]·dY[t,o]
@compute @workgroup_size(8, 8)
fn loraDbBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let o = g.x; let r = g.y;
  if (o >= OUT || r >= R) { return; }
  var s = 0.0;
  for (var t = 0u; t < seq.T; t++) { s += xa[t * R + r] * dY[t * OUT + o]; }
  dB[r * OUT + o] += LS * s;
}`;
      const loraDa = (IN: number) => /* wgsl */ `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dyb: array<f32>; // [T, R]
@group(0) @binding(3) var<storage, read_write> dA: array<f32>; // [IN, R]
const IN: u32 = ${IN}u;
// dA[i,r] += LS·Σ_t x[t,i]·dyb[t,r]
@compute @workgroup_size(8, 8)
fn loraDaBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x; let i = g.y;
  if (r >= R || i >= IN) { return; }
  var s = 0.0;
  for (var t = 0u; t < seq.T; t++) { s += x[t * IN + i] * dyb[t * R + r]; }
  dA[i * R + r] += LS * s;
}`;
      mk('dybD', C + dyb(this.D), 'loraDyb');
      mk('dybF', C + dyb(this.F), 'loraDyb');
      mk('ldbD', C + loraDb(this.D), 'loraDbBwd');
      mk('ldbF', C + loraDb(this.F), 'loraDbBwd');
      mk('ldaD', C + loraDa(this.D), 'loraDaBwd');
      mk('ldaF', C + loraDa(this.F), 'loraDaBwd');
    } else {
      mk('bdxDD', C + linBwdDx(this.D, this.D, false), 'linBwdDx');
      mk('bdxDF', C + linBwdDx(this.D, this.F, false), 'linBwdDx');
      mk('bdxFD', C + linBwdDx(this.F, this.D, false), 'linBwdDx');
      mk('bdwDD', C + linBwdDw(this.D, this.D), 'linBwdDw');
      mk('bdwDF', C + linBwdDw(this.D, this.F), 'linBwdDw');
      mk('bdwFD', C + linBwdDw(this.F, this.D), 'linBwdDw');
      mk('bdbD', C + linBwdDb(this.D), 'linBwdDb');
      mk('bdbF', C + linBwdDb(this.F), 'linBwdDb');
    }

    mk('dpBwd', `${C}
@group(0) @binding(1) var<storage, read> dCtx: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> dP: array<f32>;
// dP[h,i,j] = Σ_dh dCtx[i,h·DH+dh]·v[j,h·DH+dh]
@compute @workgroup_size(8, 8)
fn dpBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let j = g.x; let i = g.y; let h = g.z;
  if (i >= seq.T || j >= seq.T || h >= H) { return; }
  var s = 0.0;
  for (var d = 0u; d < DH; d++) { s += dCtx[i * D + h * DH + d] * v[j * D + h * DH + d]; }
  dP[(h * seq.T + i) * seq.T + j] = s;
}`, 'dpBwd');

    mk('smBwd', `${C}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> dP: array<f32>;
@group(0) @binding(3) var<storage, read_write> dS: array<f32>;
// per (h,i) row: dS_j = p_j·(dP_j − Σ_k p_k·dP_k) · (1/√DH)  ← score scale folded here
@compute @workgroup_size(64)
fn smBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x;
  if (r >= H * seq.T) { return; }
  let base = r * seq.T;
  var dot = 0.0;
  for (var j = 0u; j < seq.T; j++) { dot += P[base + j] * dP[base + j]; }
  for (var j = 0u; j < seq.T; j++) {
    dS[base + j] = P[base + j] * (dP[base + j] - dot) * RSQRT_DH;
  }
}`, 'smBwd');

    mk('dqBwd', `${C}
@group(0) @binding(1) var<storage, read> dS: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> dq: array<f32>;
// dq[i, h·DH+dh] = Σ_j dS[h,i,j]·k[j, h·DH+dh]   (dS already carries 1/√DH)
@compute @workgroup_size(8, 8)
fn dqBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x; let i = g.y;
  if (d >= D || i >= seq.T) { return; }
  let h = d / DH;
  var s = 0.0;
  for (var j = 0u; j < seq.T; j++) { s += dS[(h * seq.T + i) * seq.T + j] * k[j * D + d]; }
  dq[i * D + d] = s;
}`, 'dqBwd');

    mk('dkBwd', `${C}
@group(0) @binding(1) var<storage, read> dS: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> dk: array<f32>;
// dk[j, h·DH+dh] = Σ_i dS[h,i,j]·q[i, h·DH+dh]
@compute @workgroup_size(8, 8)
fn dkBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x; let j = g.y;
  if (d >= D || j >= seq.T) { return; }
  let h = d / DH;
  var s = 0.0;
  for (var i = 0u; i < seq.T; i++) { s += dS[(h * seq.T + i) * seq.T + j] * q[i * D + d]; }
  dk[j * D + d] = s;
}`, 'dkBwd');

    mk('dvBwd', `${C}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> dCtx: array<f32>;
@group(0) @binding(3) var<storage, read_write> dv: array<f32>;
// dv[j, h·DH+dh] = Σ_i P[h,i,j]·dCtx[i, h·DH+dh]
@compute @workgroup_size(8, 8)
fn dvBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x; let j = g.y;
  if (d >= D || j >= seq.T) { return; }
  let h = d / DH;
  var s = 0.0;
  for (var i = 0u; i < seq.T; i++) { s += P[(h * seq.T + i) * seq.T + j] * dCtx[i * D + d]; }
  dv[j * D + d] = s;
}`, 'dvBwd');

    mk('geluBwd', `${C}
@group(0) @binding(1) var<storage, read> dOut: array<f32>;
@group(0) @binding(2) var<storage, read> pre: array<f32>;
@group(0) @binding(3) var<storage, read_write> dIn: array<f32>;
// dIn = dOut · gelu'(pre)
@compute @workgroup_size(8, 8)
fn geluBwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let f = g.x; let t = g.y;
  if (f >= F || t >= seq.T) { return; }
  dIn[t * F + f] = dOut[t * F + f] * geluDx(pre[t * F + f]);
}`, 'geluBwd');
  }

  /** memset pipeline cached by (baked) length */
  private zeroPipes = new Map<number, GPUComputePipeline>();
  private pipeZero(len: number): GPUComputePipeline {
    let p = this.zeroPipes.get(len);
    if (!p) {
      const mod = this.dev.createShaderModule({
        code: /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
const LEN: u32 = ${len}u;
@compute @workgroup_size(256)
fn zero(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= LEN) { return; }
  o[g.x] = 0.0;
}`,
      });
      p = this.dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'zero' } });
      this.zeroPipes.set(len, p);
    }
    return p;
  }

  /** AdamW pipeline cached by (baked) length — same update rule as gpuTrain.ts */
  private adamPipes = new Map<number, GPUComputePipeline>();
  private pipeAdam(len: number): GPUComputePipeline {
    let p = this.adamPipes.get(len);
    if (!p) {
      const mod = this.dev.createShaderModule({
        code: /* wgsl */ `
struct AdamP { lr: f32, bc1: f32, bc2: f32, wd: f32, b1: f32, b2: f32, eps: f32, pad: f32 };
@group(0) @binding(0) var<uniform> ap: AdamP;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> P: array<f32>;
@group(0) @binding(3) var<storage, read_write> M: array<f32>;
@group(0) @binding(4) var<storage, read_write> V: array<f32>;
const LEN: u32 = ${len}u;
@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= LEN) { return; }
  var gr = G[i];
  if (ap.wd != 0.0) { gr += ap.wd * P[i]; }
  let m = ap.b1 * M[i] + (1.0 - ap.b1) * gr;
  let v = ap.b2 * V[i] + (1.0 - ap.b2) * gr * gr;
  M[i] = m;
  V[i] = v;
  P[i] -= ap.lr * (m / ap.bc1) / (sqrt(v / ap.bc2) + ap.eps);
}`,
      });
      p = this.dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'adam' } });
      this.adamPipes.set(len, p);
    }
    return p;
  }

  // ------------------------------------------------------------ bind groups

  private bgFor(pipe: string, bufs: GPUBuffer[]): GPUBindGroup {
    return this.dev.createBindGroup({
      layout: this.p[pipe].getBindGroupLayout(0),
      entries: bufs.map((b, n) => ({ binding: n, resource: { buffer: b } })),
    });
  }

  private buildBindGroups() {
    const { i } = this;
    const lora = i.mode === 'lora';

    // shared groups
    this.bg.embed = this.bgFor('embed', [this.uT, this.idsB, this.embWord, this.embPos, this.embTok, this.embSum]);
    this.bg.embLn = this.bgFor('ln', [this.uT, this.embSum, this.embLnW, this.embLnB, this.x0, this.mrE]);
    // dx-scratch zero groups (full-buffer memset; sizes are static)
    this.bg.zAcc = this.dev.createBindGroup({
      layout: this.pipeZero(this.MAXT * this.D).getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.dXacc } }],
    });
    this.bg.zMid = this.dev.createBindGroup({
      layout: this.pipeZero(this.MAXT * this.D).getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.dMid } }],
    });
    this.bg.zCtx = this.dev.createBindGroup({
      layout: this.pipeZero(this.MAXT * this.D).getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.dCtx } }],
    });
    this.bg.zH1 = this.dev.createBindGroup({
      layout: this.pipeZero(this.MAXT * this.F).getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.dH1 } }],
    });

    for (let idx = 0; idx < this.layers.length; idx++) {
      const L = this.layers[idx];
      const w = W.layer(idx);
      const g = (name: string, pipe: string, bufs: GPUBuffer[]) => {
        L.bg[name] = this.bgFor(pipe, bufs);
      };
      const wbuf = (k: string) => L.w[k];

      // --- forward ---
      g('cp', 'cp', [this.uT, this.x0, L.xin]);
      g('qlin', 'linDD', [this.uT, L.xin, wbuf(w.qW), wbuf(w.qB), L.q]);
      g('klin', 'linDD', [this.uT, L.xin, wbuf(w.kW), wbuf(w.kB), L.k]);
      g('vlin', 'linDD', [this.uT, L.xin, wbuf(w.vW), wbuf(w.vB), L.v]);
      g('sco', 'scores', [this.uT, L.q, L.k, this.maskB, L.P]);
      g('sm', 'softmax', [this.uT, L.P]);
      g('ctx', 'ctx', [this.uT, L.P, L.v, L.ctx]);
      g('olin', 'linDD', [this.uT, L.ctx, wbuf(w.oW), wbuf(w.oB), this.attnO]);
      g('ln1', 'addln', [this.uT, this.x0, this.attnO, wbuf(w.ln1W), wbuf(w.ln1B), L.ln1In, this.x1, L.ln1Out, L.mr1]);
      g('f1lin', 'linDF', [this.uT, this.x1, wbuf(w.f1W), wbuf(w.f1B), L.h1pre]);
      g('gel', 'gelu', [this.uT, L.h1pre, L.h1]);
      g('f2lin', 'linFD', [this.uT, L.h1, wbuf(w.f2W), wbuf(w.f2B), this.h2]);
      g('ln2', 'addln', [this.uT, this.x1, this.h2, wbuf(w.ln2W), wbuf(w.ln2B), L.ln2In, this.x0, this.dupSink, L.mr2]);
      if (lora) {
        const loraIn: Record<Proj, GPUBuffer> = { q: L.xin, k: L.xin, v: L.xin, o: L.ctx, f1: this.x1, f2: L.h1 };
        const loraY: Record<Proj, GPUBuffer> = { q: L.q, k: L.k, v: L.v, o: this.attnO, f1: L.h1pre, f2: this.h2 };
        for (const pr of PROJS) {
          const inn = pr === 'f2' ? this.F : this.D;
          const out = pr === 'f1' ? this.F : this.D;
          g(`${pr}la`, inn === this.D ? 'laD' : 'laF', [this.uT, loraIn[pr], L.a[pr], L.xa[pr]]);
          g(`${pr}lb`, out === this.D ? 'lbD' : 'lbF', [this.uT, L.xa[pr], L.b[pr], loraY[pr]]);
        }
      }

      // --- backward ---
      g('ln2dx', 'lnDx', [this.uT, this.dXcur, L.ln2In, L.mr2, wbuf(w.ln2W), this.dMid, this.dH2]);
      g('ln1dx', 'lnDx', [this.uT, this.dMid, L.ln1In, L.mr1, wbuf(w.ln1W), this.dXacc, this.dSub]);
      g('gelb', 'geluBwd', [this.uT, this.dH1, L.h1pre, this.dH1pre]);
      g('dp', 'dpBwd', [this.uT, this.dCtx, L.v, this.dPbuf]);
      g('smb', 'smBwd', [this.uT, L.P, this.dPbuf, this.dS]);
      g('dq', 'dqBwd', [this.uT, this.dS, L.k, this.dQ]);
      g('dk', 'dkBwd', [this.uT, this.dS, L.q, this.dK]);
      g('dv', 'dvBwd', [this.uT, L.P, this.dCtx, this.dV]);
      g('cpg', 'cp', [this.uT, this.dXacc, this.dXcur]);

      // per-proj backward groups
      const dYof: Record<Proj, GPUBuffer> = { q: this.dQ, k: this.dK, v: this.dV, o: this.dSub, f1: this.dH1pre, f2: this.dH2 };
      const xOf: Record<Proj, GPUBuffer> = { q: L.xin, k: L.xin, v: L.xin, o: L.ctx, f1: L.ln1Out, f2: L.h1 };
      const dxOf: Record<Proj, GPUBuffer> = { q: this.dXacc, k: this.dXacc, v: this.dXacc, o: this.dCtx, f1: this.dMid, f2: this.dH1 };
      const wk: Record<Proj, [string, string]> = {
        q: [w.qW, w.qB], k: [w.kW, w.kB], v: [w.vW, w.vB],
        o: [w.oW, w.oB], f1: [w.f1W, w.f1B], f2: [w.f2W, w.f2B],
      };
      for (const pr of PROJS) {
        const inn = pr === 'f2' ? this.F : this.D;
        const out = pr === 'f1' ? this.F : this.D;
        const dxPipe = inn === this.D ? (out === this.D ? 'bdxDD' : 'bdxDF') : 'bdxFD';
        if (lora) {
          g(`b.${pr}dyb`, out === this.D ? 'dybD' : 'dybF', [this.uT, dYof[pr], L.b[pr], this.dYB]);
          g(`b.${pr}db`, out === this.D ? 'ldbD' : 'ldbF', [this.uT, L.xa[pr], dYof[pr], this.gbuf(`lora.${idx}.${pr}.b`)]);
          g(`b.${pr}da`, inn === this.D ? 'ldaD' : 'ldaF', [this.uT, xOf[pr], this.dYB, this.gbuf(`lora.${idx}.${pr}.a`)]);
          g(`b.${pr}dx`, dxPipe, [this.uT, dYof[pr], wbuf(wk[pr][0]), dxOf[pr], L.a[pr], this.dYB]);
        } else {
          g(`b.${pr}dx`, dxPipe, [this.uT, dYof[pr], wbuf(wk[pr][0]), dxOf[pr]]);
          g(`b.${pr}dw`, inn === this.D ? (out === this.D ? 'bdwDD' : 'bdwDF') : 'bdwFD', [this.uT, dYof[pr], xOf[pr], this.gbuf(wk[pr][0])]);
          g(`b.${pr}db`, out === this.D ? 'bdbD' : 'bdbF', [this.uT, dYof[pr], this.gbuf(wk[pr][1])]);
        }
      }
      if (!lora) {
        g('ln2p', 'lnP', [this.uT, this.dXcur, L.ln2In, L.mr2, this.gbuf(w.ln2W), this.gbuf(w.ln2B)]);
        g('ln1p', 'lnP', [this.uT, this.dMid, L.ln1In, L.mr1, this.gbuf(w.ln1W), this.gbuf(w.ln1B)]);
      }
    }
    if (!lora) {
      this.bg.embLnP = this.bgFor('lnP', [this.uT, this.dXcur, this.embSum, this.mrE, this.gbuf(W.embLnW), this.gbuf(W.embLnB)]);
    }
  }

  /** grad buffer lookup by trainable name */
  private gbuf(name: string): GPUBuffer {
    const t = this.train.find((x) => x.name === name);
    if (!t) throw new Error(`wgsl: no trainable ${name}`);
    return t.g;
  }

  // ------------------------------------------------------------- dispatchers

  private disp(pass: GPUComputePassEncoder, pipe: string, bg: GPUBindGroup, x: number, y = 1, z = 1) {
    pass.setPipeline(this.p[pipe]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(x, y, z);
  }

  private g8(n: number) {
    return Math.ceil(n / 8);
  }

  private fwdLayer(pass: GPUComputePassEncoder, L: Layer) {
    const { T } = this;
    const d8 = this.g8(this.D);
    const t8 = this.g8(T);
    const f8 = this.g8(this.F);
    const r8 = this.g8(Math.max(1, this.R));
    const lora = this.i.mode === 'lora';
    const b = (n: string) => L.bg[n];

    this.disp(pass, 'cp', b('cp'), Math.ceil((T * this.D) / 256)); // xin[L] = x0
    for (const pr of ['q', 'k', 'v'] as const) {
      this.disp(pass, 'linDD', b(`${pr}lin`), d8, t8);
      if (lora) {
        this.disp(pass, 'laD', b(`${pr}la`), r8, t8);
        this.disp(pass, 'lbD', b(`${pr}lb`), d8, t8);
      }
    }
    this.disp(pass, 'scores', b('sco'), t8, t8, this.H);
    this.disp(pass, 'softmax', b('sm'), Math.ceil((this.H * T) / 64));
    this.disp(pass, 'ctx', b('ctx'), d8, t8);
    this.disp(pass, 'linDD', b('olin'), d8, t8);
    if (lora) {
      this.disp(pass, 'laD', b('ola'), r8, t8);
      this.disp(pass, 'lbD', b('olb'), d8, t8);
    }
    this.disp(pass, 'addln', b('ln1'), Math.ceil(T / 64)); // x = LN1(x + attnOut) → x1
    this.disp(pass, 'linDF', b('f1lin'), f8, t8);
    if (lora) {
      this.disp(pass, 'laD', b('f1la'), r8, t8);
      this.disp(pass, 'lbF', b('f1lb'), f8, t8);
    }
    this.disp(pass, 'gelu', b('gel'), f8, t8);
    this.disp(pass, 'linFD', b('f2lin'), d8, t8);
    if (lora) {
      this.disp(pass, 'laF', b('f2la'), r8, t8);
      this.disp(pass, 'lbD', b('f2lb'), d8, t8);
    }
    this.disp(pass, 'addln', b('ln2'), Math.ceil(T / 64)); // x = LN2(ln1Out + h2) → x0
  }

  private bwdProj(pass: GPUComputePassEncoder, L: Layer, pr: Proj) {
    const { T } = this;
    const lora = this.i.mode === 'lora';
    const inn = pr === 'f2' ? this.F : this.D;
    const out = pr === 'f1' ? this.F : this.D;
    const b = (n: string) => L.bg[n];
    const i8 = this.g8(inn);
    const o8 = this.g8(out);
    const t8 = this.g8(T);
    const r8 = this.g8(Math.max(1, this.R));
    if (lora) {
      this.disp(pass, out === this.D ? 'dybD' : 'dybF', b(`b.${pr}dyb`), r8, t8);
      this.disp(pass, out === this.D ? 'ldbD' : 'ldbF', b(`b.${pr}db`), o8, r8);
      this.disp(pass, inn === this.D ? 'ldaD' : 'ldaF', b(`b.${pr}da`), r8, i8);
      this.disp(pass, inn === this.D ? (out === this.D ? 'bdxDD' : 'bdxDF') : 'bdxFD', b(`b.${pr}dx`), i8, t8);
    } else {
      this.disp(pass, inn === this.D ? (out === this.D ? 'bdxDD' : 'bdxDF') : 'bdxFD', b(`b.${pr}dx`), i8, t8);
      this.disp(pass, inn === this.D ? (out === this.D ? 'bdwDD' : 'bdwDF') : 'bdwFD', b(`b.${pr}dw`), i8, o8);
      this.disp(pass, out === this.D ? 'bdbD' : 'bdbF', b(`b.${pr}db`), Math.ceil(out / 64));
    }
  }

  private bwdLayer(pass: GPUComputePassEncoder, L: Layer) {
    const { T } = this;
    const lora = this.i.mode === 'lora';
    const b = (n: string) => L.bg[n];
    const t8 = this.g8(T);
    const d8 = this.g8(this.D);
    const f8 = this.g8(this.F);
    const TD = this.MAXT * this.D;
    const TF = this.MAXT * this.F;

    // zero the accumulators for this layer
    const zp = (pipe: GPUComputePipeline, bgz: GPUBindGroup, len: number) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bgz);
      pass.dispatchWorkgroups(Math.ceil(len / 256));
    };
    zp(this.pipeZero(TD), this.bg.zAcc, TD);
    zp(this.pipeZero(TD), this.bg.zMid, TD);
    zp(this.pipeZero(TD), this.bg.zCtx, TD);
    zp(this.pipeZero(TF), this.bg.zH1, TF);

    // LN2 backward: dMid += resid branch, dH2 = ffn branch
    this.disp(pass, 'lnDx', b('ln2dx'), Math.ceil(T / 64));
    if (!lora) this.disp(pass, 'lnP', b('ln2p'), Math.ceil(this.D / 64));
    // f2 backward (dY=dH2) → dx into dH1
    this.bwdProj(pass, L, 'f2');
    this.disp(pass, 'geluBwd', b('gelb'), f8, t8); // dH1pre = dH1·gelu'(h1pre)
    // f1 backward (dY=dH1pre) → dx into dMid
    this.bwdProj(pass, L, 'f1');
    // LN1 backward: dXacc += resid branch, dSub = attn branch (dY for o-proj)
    this.disp(pass, 'lnDx', b('ln1dx'), Math.ceil(T / 64));
    if (!lora) this.disp(pass, 'lnP', b('ln1p'), Math.ceil(this.D / 64));
    // o backward (dY=dSub) → dx into dCtx
    this.bwdProj(pass, L, 'o');
    // attention backward
    this.disp(pass, 'dpBwd', b('dp'), t8, t8, this.H);
    this.disp(pass, 'smBwd', b('smb'), Math.ceil((this.H * T) / 64));
    this.disp(pass, 'dqBwd', b('dq'), d8, t8);
    this.disp(pass, 'dkBwd', b('dk'), d8, t8);
    this.disp(pass, 'dvBwd', b('dv'), d8, t8);
    // q/k/v proj backward → dx into dXacc
    this.bwdProj(pass, L, 'q');
    this.bwdProj(pass, L, 'k');
    this.bwdProj(pass, L, 'v');
    // dXacc → dXcur for the next (lower) layer
    this.disp(pass, 'cp', b('cpg'), Math.ceil((T * this.D) / 256));
  }

  // ----------------------------------------------------------------- public

  forward(inputIds: Uint32Array, mask: Float32Array): Promise<Float32Array> {
    return this.run(async () => {
      const T = inputIds.length;
      if (T === 0 || T > this.MAXT) throw new Error(`wgsl: T=${T} out of range 1..${this.MAXT}`);
      if (mask.length !== T) throw new Error('wgsl: mask length != ids length');
      this.T = T;
      const q = this.dev.queue;
      q.writeBuffer(this.idsB, 0, inputIds.buffer, inputIds.byteOffset, T * 4);
      q.writeBuffer(this.maskB, 0, mask.buffer, mask.byteOffset, T * 4);
      q.writeBuffer(this.uT, 0, new Uint32Array([T, 0, 0, 0]));

      const enc = this.dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.disp(pass, 'embed', this.bg.embed, this.g8(this.D), this.g8(T));
      this.disp(pass, 'ln', this.bg.embLn, Math.ceil(T / 64));
      for (const L of this.layers) this.fwdLayer(pass, L);
      pass.end();
      enc.copyBufferToBuffer(this.x0, 0, this.readB, 0, T * this.D * 4);
      q.submit([enc.finish()]);

      await this.readB.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(this.readB.getMappedRange().slice(0, T * this.D * 4));
      this.readB.unmap();
      return out;
    });
  }

  backward(hiddenGrad: Float32Array): Promise<void> {
    return this.run(async () => {
      if (this.T === 0) throw new Error('wgsl: backward before forward');
      if (hiddenGrad.length !== this.T * this.D) {
        throw new Error(`wgsl: hiddenGrad length ${hiddenGrad.length} != T*D=${this.T * this.D}`);
      }
      const q = this.dev.queue;
      q.writeBuffer(this.dXcur, 0, hiddenGrad.buffer, hiddenGrad.byteOffset, hiddenGrad.byteLength);

      const enc = this.dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let idx = this.layers.length - 1; idx >= 0; idx--) this.bwdLayer(pass, this.layers[idx]);
      if (this.i.mode === 'full') {
        // embLN param grads (dx into frozen embeddings is dropped)
        this.disp(pass, 'lnP', this.bg.embLnP, Math.ceil(this.D / 64));
      }
      pass.end();
      q.submit([enc.finish()]);
      await q.onSubmittedWorkDone();
    });
  }

  applyGrads(lr: number): Promise<void> {
    return this.run(async () => {
      const t = ++this.t;
      const bc1 = 1 - Math.pow(0.9, t);
      const bc2 = 1 - Math.pow(0.999, t);
      const q = this.dev.queue;
      q.writeBuffer(this.uAdamW, 0, new Float32Array([lr, bc1, bc2, 0.01, 0.9, 0.999, 1e-8, 0]));
      q.writeBuffer(this.uAdamZ, 0, new Float32Array([lr, bc1, bc2, 0.0, 0.9, 0.999, 1e-8, 0]));

      const enc = this.dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const tr of this.train) {
        pass.setPipeline(this.pipeAdam(tr.len));
        pass.setBindGroup(0, tr.bgAdam);
        pass.dispatchWorkgroups(Math.ceil(tr.len / 64));
      }
      for (const tr of this.train) {
        pass.setPipeline(this.pipeZero(tr.len));
        pass.setBindGroup(0, tr.bgZero);
        pass.dispatchWorkgroups(Math.ceil(tr.len / 256));
      }
      pass.end();
      q.submit([enc.finish()]);
      await q.onSubmittedWorkDone();
    });
  }

  private async readBuf(src: GPUBuffer, floats: number): Promise<Float32Array> {
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, this.readB, 0, floats * 4);
    this.dev.queue.submit([enc.finish()]);
    await this.readB.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(this.readB.getMappedRange().slice(0, floats * 4));
    this.readB.unmap();
    return v;
  }

  trainableWeights(): Promise<Record<string, Float32Array>> {
    return this.run(async () => {
      const out: Record<string, Float32Array> = {};
      for (const tr of this.train) out[tr.name] = await this.readBuf(tr.p, tr.len);
      return out;
    });
  }

  dispose() {
    this.dev.destroy();
  }
}
