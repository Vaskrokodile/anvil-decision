import type { HeadBackend, TrainInit } from './head';

/**
 * WGSL head trainer — engine v0. The encoder stays frozen (embeddings are
 * uploaded once); the linear head trains on-GPU:
 *   logits = X[b]·W + b,   dz = softmax(logits) − onehot(y),
 *   dW = X[b]ᵀ·dz / B,     db = colsum(dz) / B,   AdamW update.
 * Matrices here are small (d=384, c≤64, B≤64) so kernels are naive — the
 * point is a correct, inspectable WGSL autograd slice, not a speed record.
 */

const WGSL = /* wgsl */ `
struct Dims { n: u32, d: u32, c: u32, bsz: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> X: array<f32>;        // [n,d]
@group(0) @binding(2) var<storage, read> YH: array<f32>;       // [n,c] onehot
@group(0) @binding(3) var<storage, read> IDX: array<u32>;      // [bsz]
@group(0) @binding(4) var<storage, read_write> W: array<f32>;  // [d,c]
@group(0) @binding(5) var<storage, read_write> Bv: array<f32>; // [c]
@group(0) @binding(6) var<storage, read_write> LG: array<f32>; // [bsz,c] logits
@group(0) @binding(7) var<storage, read_write> DZ: array<f32>; // [bsz,c]
@group(0) @binding(8) var<storage, read_write> DW: array<f32>; // [d,c]
@group(0) @binding(9) var<storage, read_write> DB: array<f32>; // [c]
`;

const FWD = /* wgsl */ `
${'' /* logits[r,j] = Σ_k X[idx[r],k]·W[k,j] + Bv[j] */}
@compute @workgroup_size(8, 8)
fn fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.y; let j = gid.x;
  if (r >= dims.bsz || j >= dims.c) { return; }
  let row = IDX[r];
  var s = Bv[j];
  for (var k = 0u; k < dims.d; k++) {
    s += X[row * dims.d + k] * W[k * dims.c + j];
  }
  LG[r * dims.c + j] = s;
}
`;

const BWD_DZ = /* wgsl */ `
${'' /* one thread per row: dz = softmax(logits) - onehot */}
@compute @workgroup_size(64)
fn bwdDz(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= dims.bsz) { return; }
  let row = IDX[r];
  let base = r * dims.c;
  var m = LG[base];
  for (var j = 1u; j < dims.c; j++) { m = max(m, LG[base + j]); }
  var sum = 0.0;
  for (var j = 0u; j < dims.c; j++) { sum += exp(LG[base + j] - m); }
  for (var j = 0u; j < dims.c; j++) {
    DZ[base + j] = exp(LG[base + j] - m) / sum - YH[row * dims.c + j];
  }
}
`;

const BWD_DW = /* wgsl */ `
${'' /* dW[k,j] = Σ_r X[idx[r],k]·dz[r,j] / B */}
@compute @workgroup_size(8, 8)
fn bwdDw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.y; let j = gid.x;
  if (k >= dims.d || j >= dims.c) { return; }
  var s = 0.0;
  for (var r = 0u; r < dims.bsz; r++) {
    s += X[IDX[r] * dims.d + k] * DZ[r * dims.c + j];
  }
  DW[k * dims.c + j] = s / f32(dims.bsz);
}
`;

const BWD_DB = /* wgsl */ `
${'' /* db[j] = Σ_r dz[r,j] / B */}
@compute @workgroup_size(64)
fn bwdDb(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= dims.c) { return; }
  var s = 0.0;
  for (var r = 0u; r < dims.bsz; r++) { s += DZ[r * dims.c + j]; }
  DB[j] = s / f32(dims.bsz);
}
`;

const ADAM = /* wgsl */ `
struct AdamP { lr: f32, bc1: f32, bc2: f32, wd: f32, b1: f32, b2: f32, eps: f32, pad: f32 };
@group(1) @binding(0) var<uniform> ap: AdamP;
@group(1) @binding(1) var<storage, read> G: array<f32>;        // grad
@group(1) @binding(2) var<storage, read_write> P: array<f32>;  // param
@group(1) @binding(3) var<storage, read_write> M: array<f32>;
@group(1) @binding(4) var<storage, read_write> V: array<f32>;
@group(1) @binding(5) var<uniform> len: u32;
@group(1) @binding(6) var<uniform> applyWd: u32;

@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= len) { return; }
  var g = G[i];
  if (applyWd == 1u) { g += ap.wd * P[i]; }
  let m = ap.b1 * M[i] + (1.0 - ap.b1) * g;
  let v = ap.b2 * V[i] + (1.0 - ap.b2) * g * g;
  M[i] = m; V[i] = v;
  P[i] -= ap.lr * (m / ap.bc1) / (sqrt(v / ap.bc2) + ap.eps);
}
`;

export class WgslHead implements HeadBackend {
  kind = 'wgsl' as const;

  private dev!: GPUDevice;
  private i!: TrainInit;
  private t = 0;

  private pFwd!: GPUComputePipeline;
  private pDz!: GPUComputePipeline;
  private pDw!: GPUComputePipeline;
  private pDb!: GPUComputePipeline;
  private pAdam!: GPUComputePipeline;

  private uDims!: GPUBuffer;
  private uAdam!: GPUBuffer;
  private uLen!: GPUBuffer;
  private uWd!: GPUBuffer;
  private xB!: GPUBuffer;
  private yB!: GPUBuffer;
  private idxB!: GPUBuffer;
  private wB!: GPUBuffer;
  private bB!: GPUBuffer;
  private lgB!: GPUBuffer;
  private dzB!: GPUBuffer;
  private dwB!: GPUBuffer;
  private dbB!: GPUBuffer;
  private mwB!: GPUBuffer;
  private vwB!: GPUBuffer;
  private mbB!: GPUBuffer;
  private vbB!: GPUBuffer;
  private readB!: GPUBuffer;

  private bindMain!: GPUBindGroup;
  private bindAdamW!: GPUBindGroup;
  private bindAdamB!: GPUBindGroup;

  static async create(i: TrainInit): Promise<WgslHead> {
    const h = new WgslHead();
    if (!navigator.gpu) throw new Error('no navigator.gpu');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no adapter');
    h.dev = await adapter.requestDevice();
    await h.init(i);
    return h;
  }

  async init(i: TrainInit) {
    this.i = i;
    const dev = this.dev;
    const { n, d, c } = i;
    const B = Math.max(1, i.batch);
    const F = GPUBufferUsage;

    const buf = (size: number, usage: number, data?: ArrayBufferView) => {
      const b = dev.createBuffer({ size: Math.max(16, size), usage });
      if (data) dev.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
      return b;
    };

    // one-hot labels
    const yh = new Float32Array(n * c);
    for (let r = 0; r < n; r++) yh[r * c + i.y[r]] = 1;

    // init W (same heuristic as CPU: uniform ±sqrt(2/(d+c)))
    let s = i.seed >>> 0;
    const rng = () => {
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const scale = Math.sqrt(2 / (d + c));
    const w0 = new Float32Array(d * c).map(() => (rng() * 2 - 1) * scale);

    this.uDims = buf(16, F.UNIFORM | F.COPY_DST, new Uint32Array([n, d, c, B]));
    this.uAdam = buf(32, F.UNIFORM | F.COPY_DST);
    this.uLen = buf(16, F.UNIFORM | F.COPY_DST);
    this.uWd = buf(16, F.UNIFORM | F.COPY_DST);
    this.xB = buf(i.x.byteLength, F.STORAGE, i.x);
    this.yB = buf(yh.byteLength, F.STORAGE, yh);
    this.idxB = buf(B * 4, F.STORAGE | F.COPY_DST);
    this.wB = buf(d * c * 4, F.STORAGE | F.COPY_SRC | F.COPY_DST, w0);
    this.bB = buf(c * 4, F.STORAGE | F.COPY_SRC | F.COPY_DST);
    this.lgB = buf(B * c * 4, F.STORAGE | F.COPY_SRC);
    this.dzB = buf(B * c * 4, F.STORAGE);
    this.dwB = buf(d * c * 4, F.STORAGE);
    this.dbB = buf(c * 4, F.STORAGE);
    this.mwB = buf(d * c * 4, F.STORAGE);
    this.vwB = buf(d * c * 4, F.STORAGE);
    this.mbB = buf(c * 4, F.STORAGE);
    this.vbB = buf(c * 4, F.STORAGE);
    this.readB = buf(Math.max(B * c, d * c) * 4, F.MAP_READ | F.COPY_DST);

    const mk = (code: string, entry: string) => {
      const mod = dev.createShaderModule({ code });
      return dev.createComputePipeline({
        layout: 'auto',
        compute: { module: mod, entryPoint: entry },
      });
    };
    this.pFwd = mk(WGSL + FWD, 'fwd');
    this.pDz = mk(WGSL + BWD_DZ, 'bwdDz');
    this.pDw = mk(WGSL + BWD_DW, 'bwdDw');
    this.pDb = mk(WGSL + BWD_DB, 'bwdDb');
    this.pAdam = mk(ADAM, 'adam');

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.uDims } },
      { binding: 1, resource: { buffer: this.xB } },
      { binding: 2, resource: { buffer: this.yB } },
      { binding: 3, resource: { buffer: this.idxB } },
      { binding: 4, resource: { buffer: this.wB } },
      { binding: 5, resource: { buffer: this.bB } },
      { binding: 6, resource: { buffer: this.lgB } },
      { binding: 7, resource: { buffer: this.dzB } },
      { binding: 8, resource: { buffer: this.dwB } },
      { binding: 9, resource: { buffer: this.dbB } },
    ];
    this.bindMain = dev.createBindGroup({ layout: this.pFwd.getBindGroupLayout(0), entries });

    const adamEntries = (G: GPUBuffer, P: GPUBuffer, M: GPUBuffer, V: GPUBuffer): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: this.uAdam } },
      { binding: 1, resource: { buffer: G } },
      { binding: 2, resource: { buffer: P } },
      { binding: 3, resource: { buffer: M } },
      { binding: 4, resource: { buffer: V } },
      { binding: 5, resource: { buffer: this.uLen } },
      { binding: 6, resource: { buffer: this.uWd } },
    ];
    this.bindAdamW = dev.createBindGroup({
      layout: this.pAdam.getBindGroupLayout(0),
      entries: adamEntries(this.dwB, this.wB, this.mwB, this.vwB),
    });
    this.bindAdamB = dev.createBindGroup({
      layout: this.pAdam.getBindGroupLayout(0),
      entries: adamEntries(this.dbB, this.bB, this.mbB, this.vbB),
    });
  }

  async step(idx: Uint32Array) {
    const { dev, i } = this;
    const B = Math.min(i.batch, idx.length);
    dev.queue.writeBuffer(this.idxB, 0, idx.buffer, idx.byteOffset, B * 4);
    dev.queue.writeBuffer(this.uDims, 12, new Uint32Array([B]));

    // adam params: [lr, bc1, bc2, wd, b1, b2, eps, pad]
    const t = ++this.t;
    const bc1 = 1 - Math.pow(0.9, t);
    const bc2 = 1 - Math.pow(0.999, t);
    dev.queue.writeBuffer(
      this.uAdam,
      0,
      new Float32Array([i.lr, bc1, bc2, 0.01, 0.9, 0.999, 1e-8, 0]),
    );

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.bindMain);
    const c8 = Math.ceil(i.c / 8),
      d8 = Math.ceil(i.d / 8);
    pass.setPipeline(this.pFwd);
    pass.dispatchWorkgroups(c8, Math.ceil(B / 8));
    pass.setPipeline(this.pDz);
    pass.dispatchWorkgroups(Math.ceil(B / 64));
    pass.setPipeline(this.pDw);
    pass.dispatchWorkgroups(c8, d8);
    pass.setPipeline(this.pDb);
    pass.dispatchWorkgroups(Math.ceil(i.c / 64));
    pass.end();
    dev.queue.submit([enc.finish()]);

    // AdamW — weight (with decay) then bias (without); uniforms set per submit
    const adamDispatch = async (bind: GPUBindGroup, len: number, wd: number) => {
      dev.queue.writeBuffer(this.uLen, 0, new Uint32Array([len]));
      dev.queue.writeBuffer(this.uWd, 0, new Uint32Array([wd]));
      const e2 = dev.createCommandEncoder();
      const p2 = e2.beginComputePass();
      p2.setPipeline(this.pAdam);
      p2.setBindGroup(0, bind);
      p2.dispatchWorkgroups(Math.ceil(len / 64));
      p2.end();
      dev.queue.submit([e2.finish()]);
    };
    await adamDispatch(this.bindAdamW, i.d * i.c, 1);
    await adamDispatch(this.bindAdamB, i.c, 0);
  }

  /** Read all logits by streaming rows through the batch forward pass. */
  async logits() {
    const { dev, i } = this;
    const out = new Float32Array(i.n * i.c);
    const B = Math.max(1, i.batch);
    for (let off = 0; off < i.n; off += B) {
      const m = Math.min(B, i.n - off);
      const idx = new Uint32Array(B);
      for (let r = 0; r < m; r++) idx[r] = off + r;
      dev.queue.writeBuffer(this.idxB, 0, idx.buffer, 0, B * 4);
      dev.queue.writeBuffer(this.uDims, 12, new Uint32Array([m]));
      const enc = dev.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.pFwd);
      p.setBindGroup(0, this.bindMain);
      p.dispatchWorkgroups(Math.ceil(i.c / 8), Math.ceil(B / 8));
      p.end();
      enc.copyBufferToBuffer(this.lgB, 0, this.readB, 0, B * i.c * 4);
      dev.queue.submit([enc.finish()]);
      await this.readB.mapAsync(GPUMapMode.READ);
      out.set(new Float32Array(this.readB.getMappedRange().slice(0, m * i.c * 4)), off * i.c);
      this.readB.unmap();
    }
    return out;
  }

  async weights() {
    const { dev, i } = this;
    const read = async (src: GPUBuffer, bytes: number) => {
      const enc = dev.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, this.readB, 0, bytes);
      dev.queue.submit([enc.finish()]);
      await this.readB.mapAsync(GPUMapMode.READ);
      const v = new Float32Array(this.readB.getMappedRange().slice(0, bytes));
      this.readB.unmap();
      return v;
    };
    return { w: await read(this.wB, i.d * i.c * 4), b: await read(this.bB, i.c * 4) };
  }

  dispose() {
    this.dev.destroy();
  }
}
