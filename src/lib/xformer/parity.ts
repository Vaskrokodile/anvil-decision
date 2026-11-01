import type { EncoderBackend, EncoderInit } from './spec';
import { CpuEncoder } from './cpu';

/**
 * Real-device parity gate: run one forward through the WGSL backend and the
 * CPU reference on identical inputs, compare cosine of the hidden states.
 * WGSL kernels are verified structurally only — this catches shader-precision
 * or dispatch bugs on the user's actual GPU before they corrupt a run.
 */
export const PARITY_MIN_COS = 0.999;

export async function parityCheckedEncoder(
  wgsl: EncoderBackend,
  init: EncoderInit,
  ids: Uint32Array,
  mask: Float32Array,
): Promise<{ enc: EncoderBackend; cos: number }> {
  const cpu: EncoderBackend = new CpuEncoder();
  cpu.init(init);
  const hw = await wgsl.forward(ids, mask);
  const hc = await cpu.forward(ids, mask);
  let dot = 0,
    aw = 0,
    ac = 0;
  for (let i = 0; i < hw.length; i++) {
    dot += hw[i] * hc[i];
    aw += hw[i] * hw[i];
    ac += hc[i] * hc[i];
  }
  const cos = dot / (Math.sqrt(aw) * Math.sqrt(ac) + 1e-30);
  if (cos < PARITY_MIN_COS) {
    wgsl.dispose?.();
    return { enc: cpu, cos };
  }
  cpu.dispose?.();
  return { enc: wgsl, cos };
}
