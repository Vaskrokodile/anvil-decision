import { CpuHead, evalLogits, type HeadBackend, type TrainInit } from './head';
import { WgslHead } from './gpuTrain';

export interface TrainRequest {
  type: 'train';
  init: TrainInit;
}
export interface TrainProgress {
  type: 'progress';
  epoch: number;
  epochs: number;
  loss: number;
  acc: number;
  backend: 'cpu' | 'wgsl';
}
export interface TrainDone {
  type: 'done';
  w: Float32Array;
  b: Float32Array;
  backend: 'cpu' | 'wgsl';
  loss: number;
  acc: number;
}
export interface TrainError {
  type: 'error';
  message: string;
}
export type TrainMsg = TrainProgress | TrainDone | TrainError;

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

async function train(init: TrainInit, post: (m: TrainMsg) => void) {
  let backend: HeadBackend;
  if (init.backend !== 'cpu') {
    try {
      backend = await WgslHead.create(init);
    } catch {
      backend = new CpuHead();
      backend.init(init);
    }
  } else {
    backend = new CpuHead();
    backend.init(init);
  }

  const rng = mulberry(init.seed);
  const order = Array.from({ length: init.n }, (_, k) => k);
  const B = Math.max(1, init.batch);
  let last = { loss: 0, acc: 0 };

  for (let ep = 0; ep < init.epochs; ep++) {
    // fisher-yates shuffle
    for (let k = init.n - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    for (let off = 0; off < init.n; off += B) {
      await backend.step(Uint32Array.from(order.slice(off, off + B)));
    }
    const lg = await backend.logits();
    last = evalLogits(lg, init.y, init.c);
    post({
      type: 'progress',
      epoch: ep + 1,
      epochs: init.epochs,
      loss: last.loss,
      acc: last.acc,
      backend: backend.kind,
    });
  }

  const { w, b } = await backend.weights();
  post({ type: 'done', w, b, backend: backend.kind, loss: last.loss, acc: last.acc });
  backend.dispose?.();
}

const ctx = self as unknown as {
  onmessage: (e: MessageEvent<TrainRequest>) => void;
  postMessage: (m: TrainMsg, transfer?: Transferable[]) => void;
};

ctx.onmessage = (e) => {
  const post = (m: TrainMsg) => ctx.postMessage(m);
  if (e.data.type !== 'train') return;
  train(e.data.init, post).catch((err) =>
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) }),
  );
};
