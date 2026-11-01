/**
 * deberta-smoke.mts — finite-difference gradient check for DebertaCpuEncoder
 * (disentangled attention: qk + c2p + p2c terms, shared pos projections).
 * Run:  node scripts/deberta-smoke.mts     (node ≥22, type-strips the .ts)
 *
 * Reduced spec (2 layers, d=32, 4 heads × 8, ffn 64, vocab 64, span 8)
 * exercises every code path: pos gather/scatter, qW/kW grads from BOTH the
 * content and pos projection inputs, shared rel-embedding LN backward.
 */
import { DebertaCpuEncoder } from '../src/lib/xformer/deberta.ts';
import { DW, type DebertaSpec, type Weights } from '../src/lib/xformer/spec.ts';

const spec: DebertaSpec = {
  layers: 2, dModel: 32, heads: 4, dHead: 8, dFfn: 64,
  vocab: 64, maxPos: 16, lnEps: 1e-7, posSpan: 8, scaleFactor: 3,
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
const rng = mulberry(20251009);
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
  w[DW.word] = fill(spec.vocab * d, 0.08);
  w[DW.embLnW] = fillLN(d);
  w[DW.embLnB] = fill(d, 0.02);
  w[DW.relEmb] = fill(2 * spec.posSpan * d, 0.06);
  w[DW.relLnW] = fillLN(d);
  w[DW.relLnB] = fill(d, 0.02);
  for (let l = 0; l < spec.layers; l++) {
    const n = DW.layer(l);
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

const DELTA = 1e-2;

interface P { data: Float32Array; grad: Float32Array }
type PMap = Record<string, P>;
const internals = (enc: DebertaCpuEncoder): PMap => (enc as unknown as { params: PMap }).params;

async function numericGrads(enc: DebertaCpuEncoder, p: P): Promise<Float64Array> {
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

async function dirTensor(enc: DebertaCpuEncoder, p: P, ana: Float32Array): Promise<number> {
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
  const enc = new DebertaCpuEncoder();
  await enc.init({ spec, weights: makeWeights(), mode, rank: 4, seed: 7, maxSeq: 16 });
  const params = internals(enc);
  if (mode === 'lora') {
    for (const [name, p] of Object.entries(params)) {
      if (name.endsWith('.b')) for (let i = 0; i < p.data.length; i++) p.data[i] = randn() * 0.05;
    }
  }
  const nTrain = Object.keys(await enc.trainableWeights()).length;
  const expect = mode === 'lora' ? 24 : 36; // 36 = 16×2 layer keys + embLn + relLn
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
    let d2 = 0, n2 = 0, a2 = 0, dot = 0, maxAbs = 0;
    for (let i = 0; i < num.length; i++) {
      const dd = ana[i] - num[i];
      d2 += dd * dd;
      n2 += num[i] * num[i];
      a2 += ana[i] * ana[i];
      dot += ana[i] * num[i];
      if (Math.abs(dd) > maxAbs) maxAbs = Math.abs(dd);
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

const L = (l: number) => DW.layer(l);
// lora: q/k adapters carry pos-path gradients too — check them specifically
const loraNames = ['lora.0.q.a', 'lora.0.q.b', 'lora.0.k.a', 'lora.1.o.b', 'lora.1.f1.a', 'lora.1.f2.b'];
// full: qW/kW (content+pos grads), LN weight, ffn, emb-LN, rel-LN
const fullNames = [L(0).qW, L(0).kW, L(1).ln1W, L(1).f2W, DW.embLnW, DW.relLnW];

const e1 = await check('lora', loraNames);
const e2 = await check('full', fullNames);

if (e1 >= 1e-2 || e2 >= 1e-2) process.exit(1);
console.log('ALL CHECKS PASSED');
