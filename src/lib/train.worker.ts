import { CpuHead, evalLogits, type HeadBackend, type TrainInit } from './head';
import { WgslHead } from './gpuTrain';
import { mulberry, stratifiedSplit } from './split';

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
  /** held-out accuracy — null when the split left no val rows */
  valAcc: number | null;
  valN: number;
  trainN: number;
}
export interface TrainError {
  type: 'error';
  message: string;
}
export type TrainMsg = TrainProgress | TrainDone | TrainError;

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
  const { train: trainIdx, val: valIdx } = stratifiedSplit(
    init.y,
    init.valFrac ?? 0.2,
    init.seed,
  );
  const order = trainIdx;
  const B = Math.max(1, init.batch);
  let last = { loss: 0, acc: 0 };

  for (let ep = 0; ep < init.epochs; ep++) {
    // fisher-yates shuffle
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    for (let off = 0; off < order.length; off += B) {
      await backend.step(Uint32Array.from(order.slice(off, off + B)));
    }
    const lg = await backend.logits();
    last = evalLogits(lg, init.y, init.c, trainIdx);
    post({
      type: 'progress',
      epoch: ep + 1,
      epochs: init.epochs,
      loss: last.loss,
      acc: last.acc,
      backend: backend.kind,
    });
  }

  const lg = await backend.logits();
  const val =
    valIdx.length > 0 ? evalLogits(lg, init.y, init.c, valIdx) : null;
  const { w, b } = await backend.weights();
  post({
    type: 'done',
    w,
    b,
    backend: backend.kind,
    loss: last.loss,
    acc: last.acc,
    valAcc: val?.acc ?? null,
    valN: valIdx.length,
    trainN: trainIdx.length,
  });
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
