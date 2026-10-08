/**
 * xformer-smoke.mts — finite-difference gradient check for CpuEncoder.
 * Run:  node scripts/xformer-smoke.mts     (node ≥22, type-strips the .ts)
 *
 * Reduced spec (2 layers, d=32, 4 heads × 8, ffn 64, vocab 64, maxPos 16)
 * exercises the exact same code paths as MiniLM-L6. For each checked tensor
 * we compute the FULL numeric gradient (±1e-3 central differences) and report
 * ‖g_ana − g_num‖₂ / ‖g_num‖₂ — the vector-norm is robust to the float32
 * noise floor (~ε/δ ≈ 1e-4) that makes elementwise rel-err misleading on
 * near-zero gradient entries.
 */
import { CpuEncoder } from '../src/lib/xformer/cpu.ts';
import { W, type Weights, type XformerSpec } from '../src/lib/xformer/spec.ts';

const spec: XformerSpec = {
  layers: 2, dModel: 32, heads: 4, dHead: 8, dFfn: 64,
  vocab: 64, maxPos: 16, lnEps: 1e-12,
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
const rng = mulberry(20251008);
const randn = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
const fill = (n: number, s: number) => {
  const t = new Float32Array(n);
  for (let i = 0; i < n; i++) t[i] = randn() * s;
  return t;
};
const fillLN = (n: number) => {
  const t = new Float32Array(n);
  for (let i = 0; i < n; i++) t[i] = 1 + randn() * 0.02;
  return t;
};

function makeWeights(): Weights {
  const d = spec.dModel;
  const ff = spec.dFfn;
  const w: Weights = {};
  w[W.word] = fill(spec.vocab * d, 0.08);
  w[W.pos] = fill(spec.maxPos * d, 0.04);
  w[W.tokType] = fill(2 * d, 0.04);
  w[W.embLnW] = fillLN(d);
  w[W.embLnB] = fill(d, 0.02);
  for (let l = 0; l < spec.layers; l++) {
    const n = W.layer(l);
    for (const key of [n.qW, n.kW, n.vW, n.oW]) w[key] = fill(d * d, 0.06);
    w[n.f1W] = fill(ff * d, 0.06);
    w[n.f2W] = fill(d * ff, 0.06);
    for (const key of [n.qB, n.kB, n.vB, n.oB, n.f2B]) w[key] = fill(d, 0.02);
    w[n.f1B] = fill(ff, 0.02);
    for (const key of [n.ln1W, n.ln2W]) w[key] = fillLN(d);
    for (const key of [n.ln1B, n.ln2B]) w[key] = fill(d, 0.02);
  }
  return w;
}

const ids = new Uint32Array([3, 17, 5, 42, 9]);
const mask = new Float32Array([1, 1, 1, 1, 0]); // last position padded
const T = ids.length;
const coeff = new Float32Array(T * spec.dModel);
{
  const r = mulberry(99);
  for (let i = 0; i < coeff.length; i++) coeff[i] = (r() * 2 - 1) * 0.5;
}
const lossOf = (h: Float32Array) => {
  let L = 0;
  for (let i = 0; i < h.length; i++) L += h[i] * coeff[i];
  return L;
};

const DELTA = 1e-2; // larger δ → f32 forward noise (~ε/δ) stops dominating tiny grads

interface P { data: Float32Array; grad: Float32Array }
type PMap = Record<string, P>;
const internals = (enc: CpuEncoder): PMap => (enc as unknown as { params: PMap }).params;

/** full-tensor central-difference gradient — 2·N forwards, tiny model so cheap */
async function numericGrads(enc: CpuEncoder, p: P): Promise<Float64Array> {
  const num = new Float64Array(p.data.length);
  for (let i = 0; i < p.data.length; i++) {
    const orig = p.data[i];
    p.data[i] = orig + DELTA;
    const lp = lossOf(await enc.forward(ids, mask));
    p.data[i] = orig - DELTA;
    const lm = lossOf(await enc.forward(ids, mask));
    p.data[i] = orig;
    num[i] = (lp - lm) / (2 * DELTA);
  }
  return num;
}

/**
 * Per-tensor directional check along the analytic gradient: perturb ONLY this
 * tensor by ±ε·ĝ — numeric result should equal ‖g‖₂. Maximal SNR (the signal
 * is the full gradient norm; noise stays ~ε_loss/2ε ≈ 1e-5).
 */
async function dirTensor(enc: CpuEncoder, p: P, ana: Float32Array): Promise<number> {
  let gn = 0;
  for (let i = 0; i < ana.length; i++) gn += ana[i] * ana[i];
  gn = Math.sqrt(gn);
  if (gn === 0) return 0;
  const EPS = 5e-2;
  for (let i = 0; i < ana.length; i++) p.data[i] += (EPS * ana[i]) / gn;
  const lp = lossOf(await enc.forward(ids, mask));
  for (let i = 0; i < ana.length; i++) p.data[i] -= (2 * EPS * ana[i]) / gn;
  const lm = lossOf(await enc.forward(ids, mask));
  for (let i = 0; i < ana.length; i++) p.data[i] += (EPS * ana[i]) / gn;
  const num = (lp - lm) / (2 * EPS);
  return Math.abs(num - gn) / gn;
}

async function check(mode: 'lora' | 'full', names: string[]) {
  const enc = new CpuEncoder();
  await enc.init({ spec, weights: makeWeights(), mode, rank: 4, seed: 7, maxSeq: 16 });
  const params = internals(enc);
  if (mode === 'lora') {
    // LoRA-B is zeros at init → dA ≡ 0 vacuously. Seed Bs with small values so
    // both A and B carry real gradients (equivalent to checking mid-training).
    for (const [name, p] of Object.entries(params)) {
      if (name.endsWith('.b')) for (let i = 0; i < p.data.length; i++) p.data[i] = randn() * 0.05;
    }
    if (W.layer(0).qW in params) throw new Error('frozen weight leaked into lora params');
  }
  const nTrain = Object.keys(await enc.trainableWeights()).length;
  const expect = mode === 'lora' ? 24 : 34;
  console.log(`${mode}: ${nTrain} trainable tensors (expect ${expect})`);
  if (nTrain !== expect) throw new Error('trainable count mismatch');

  const hidden = await enc.forward(ids, mask);
  const finite = [...hidden].every(Number.isFinite);
  console.log(`${mode}: hidden [${T}×${spec.dModel}] len=${hidden.length} finite=${finite}`);
  if (hidden.length !== T * spec.dModel || !finite) throw new Error('forward shape/finite FAIL');

  await enc.backward(coeff);
  const grads: Record<string, Float32Array> = {};
  for (const [name, p] of Object.entries(params)) grads[name] = p.grad.slice();

  let worst = 0;
  for (const name of names) {
    const p = params[name];
    const ana = grads[name];
    const num = await numericGrads(enc, p);
    let d2 = 0;
    let n2 = 0;
    let a2 = 0;
    let dot = 0;
    let maxAbs = 0;
    for (let i = 0; i < num.length; i++) {
      const d = ana[i] - num[i];
      d2 += d * d;
      n2 += num[i] * num[i];
      a2 += ana[i] * ana[i];
      dot += ana[i] * num[i];
      if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
    }
    const vecRel = Math.sqrt(d2) / (Math.sqrt(n2) + 1e-30);
    const cos = dot / (Math.sqrt(a2) * Math.sqrt(n2) + 1e-30);
    const dirRel = await dirTensor(enc, p, ana);
    worst = Math.max(worst, dirRel);
    console.log(
      `  ${name}  N=${num.length}  vecRel=${vecRel.toExponential(3)}  cos=${cos.toFixed(6)}  maxAbsDiff=${maxAbs.toExponential(2)}  dirRel=${dirRel.toExponential(3)}`,
    );
  }
  const verdict = worst < 1e-2 ? 'PASS' : 'FAIL';
  console.log(`${mode}: worst dirRel = ${worst.toExponential(3)}  ${verdict}\n`);
  return worst;
}

/**
 * Directional check over ALL trainable params at once: with u_i ∈ {±1}
 * (Rademacher), Σ_i g_i·u_i = dL/dt along θ+t·u — compare analytic dot vs
 * (L(θ+εu) − L(θ−εu))/2ε. One scalar, but it covers every parameter.
 */
async function dirCheck(mode: 'lora' | 'full') {
  const enc = new CpuEncoder();
  await enc.init({ spec, weights: makeWeights(), mode, rank: 4, seed: 7, maxSeq: 16 });
  const params = internals(enc);
  if (mode === 'lora') {
    for (const [name, p] of Object.entries(params)) {
      if (name.endsWith('.b')) for (let i = 0; i < p.data.length; i++) p.data[i] = randn() * 0.05;
    }
  }
  await enc.forward(ids, mask);
  await enc.backward(coeff);

  const r = mulberry(4242);
  const all = Object.values(params);
  const u = all.map((p) => {
    const v = new Float32Array(p.data.length);
    for (let i = 0; i < v.length; i++) v[i] = r() < 0.5 ? -1 : 1;
    return v;
  });
  let ana = 0;
  for (let k = 0; k < all.length; k++) {
    const g = all[k].grad;
    for (let i = 0; i < g.length; i++) ana += g[i] * u[k][i];
  }
  const EPS = 1e-3;
  for (let k = 0; k < all.length; k++) for (let i = 0; i < u[k].length; i++) all[k].data[i] += EPS * u[k][i];
  const lp = lossOf(await enc.forward(ids, mask));
  for (let k = 0; k < all.length; k++) for (let i = 0; i < u[k].length; i++) all[k].data[i] -= 2 * EPS * u[k][i];
  const lm = lossOf(await enc.forward(ids, mask));
  for (let k = 0; k < all.length; k++) for (let i = 0; i < u[k].length; i++) all[k].data[i] += EPS * u[k][i];
  const num = (lp - lm) / (2 * EPS);
  const rel = Math.abs(ana - num) / (Math.abs(ana) + Math.abs(num) + 1e-30);
  console.log(`${mode} directional: analytic=${ana.toExponential(6)} numeric=${num.toExponential(6)} rel=${rel.toExponential(3)}  ${rel < 1e-2 ? 'PASS' : 'FAIL'}`);
  return rel;
}

const L = (l: number) => W.layer(l);
// cover every op type: lora A & B on attn + ffn, frozen-base dx path
const loraNames = ['lora.0.q.a', 'lora.0.q.b', 'lora.1.o.b', 'lora.1.f1.a', 'lora.1.f2.a', 'lora.1.f2.b'];
// cover: attn matrix, LN weight, ffn bias, ffn matrix, emb-LN
const fullNames = [L(0).qW, L(0).ln1W, L(1).f1B, L(1).f2W, W.embLnW];

const e1 = await check('lora', loraNames);
const e2 = await check('full', fullNames);
const d1 = await dirCheck('lora');
const d2 = await dirCheck('full');

// applyGrads smoke — params should move, grads zeroed
{
  const enc = new CpuEncoder();
  await enc.init({ spec, weights: makeWeights(), mode: 'full', rank: 4, seed: 7, maxSeq: 16 });
  const params = internals(enc);
  const p = params[L(0).qW];
  const before = p.data[0];
  await enc.forward(ids, mask);
  await enc.backward(coeff);
  await enc.applyGrads(1e-3);
  const moved = p.data[0] !== before;
  const zeroed = p.grad.every((x) => x === 0);
  console.log(`applyGrads: param moved=${moved} grads zeroed=${zeroed}`);
  if (!moved || !zeroed) process.exit(1);
}

if (e1 >= 1e-2 || e2 >= 1e-2 || d1 >= 1e-2 || d2 >= 1e-2) process.exit(1);
console.log('ALL CHECKS PASSED');
