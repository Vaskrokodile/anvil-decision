/**
 * Hardware probe — the one part of Anvil that is already 100% real.
 * Queries navigator.gpu for adapter info/limits/features and classifies
 * the machine into a training tier used to gate model selection.
 */

export type Tier = 0 | 1 | 2 | 3 | 4;

export interface GpuReport {
  ok: boolean;
  tier: Tier;
  error?: string;
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  isFallbackAdapter?: boolean;
  fp16: boolean;
  deviceMemoryGB?: number;
  cores?: number;
  /** limits that matter for training */
  maxStorageBufferBindingSize?: number;
  maxBufferSize?: number;
  maxComputeWorkgroupsPerDimension?: number;
  maxComputeInvocationsPerWorkgroup?: number;
  /** derived budgets */
  weightsBudgetMB: number;
  trainableBudgetM: number;
  probedAt: string;
}

const MIB = 1024 * 1024;
const GIB = 1024 * 1024 * 1024;

/**
 * Tier model, from what the adapter can bind per storage buffer.
 * maxStorageBufferBindingSize is the best honest proxy for VRAM headroom:
 * the guaranteed floor is 128 MiB; real discrete adapters report 1–2 GiB+.
 *
 * Training costs ~4× params in fp32 (w + grad + Adam m,v) plus activations,
 * so budgets stay deliberately conservative.
 */
function classify(r: Omit<GpuReport, 'tier' | 'weightsBudgetMB' | 'trainableBudgetM' | 'probedAt'>): {
  tier: Tier;
  weightsBudgetMB: number;
  trainableBudgetM: number;
} {
  if (!r.ok) return { tier: 0, weightsBudgetMB: 0, trainableBudgetM: 0 };

  const buf = r.maxStorageBufferBindingSize ?? 0;
  const mem = r.deviceMemoryGB ?? 4;
  let score = 0;

  if (buf >= 2 * GIB) score += 3;
  else if (buf >= 512 * MIB) score += 2;
  else if (buf >= 128 * MIB) score += 1;

  // discrete-class vendors; 'apple' counts as discrete-class for unified mem
  if (r.vendor && ['nvidia', 'amd', 'apple', 'intel'].includes(r.vendor)) score += 1;
  if (mem >= 8) score += 1;
  if (r.isFallbackAdapter) score = 0;

  const tier = Math.min(4, Math.max(1, score)) as Tier;

  // Budgets: what we let a run use, not what the chip physically has.
  // Weights budget ~ a fraction of the largest bindable buffer.
  const weightsBudgetMB = Math.max(
    128,
    Math.floor((Math.min(buf || 128 * MIB, 3 * GIB) * 0.6) / MIB),
  );
  // Trainable params (full FT) ≈ budget / 4 bytes / 4 states — halved again for activations.
  const trainableBudgetM = Math.max(
    10,
    Math.floor(((weightsBudgetMB * MIB) / 4 / 8) / 1_000_000) * 10,
  );

  return { tier, weightsBudgetMB, trainableBudgetM };
}

export async function probeGpu(): Promise<GpuReport> {
  const probedAt = new Date().toISOString();
  const deviceMemoryGB = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const cores = navigator.hardwareConcurrency;

  if (!('gpu' in navigator) || !navigator.gpu) {
    return {
      ok: false,
      tier: 0,
      error: 'navigator.gpu unavailable — WebGPU needs Chrome/Edge 113+ or Safari 26+.',
      fp16: false,
      deviceMemoryGB,
      cores,
      weightsBudgetMB: 0,
      trainableBudgetM: 0,
      probedAt,
    };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return {
        ok: false,
        tier: 0,
        error: 'WebGPU present but no adapter — GPU may be blocklisted by the browser.',
        fp16: false,
        deviceMemoryGB,
        cores,
        weightsBudgetMB: 0,
        trainableBudgetM: 0,
        probedAt,
      };
    }

    const info = adapter.info;
    const features = new Set<string>();
    adapter.features.forEach((f) => features.add(f));
    const limits = adapter.limits;

    const partial = {
      ok: true as const,
      vendor: info?.vendor || undefined,
      architecture: info?.architecture || undefined,
      device: info?.device || undefined,
      description: info?.description || undefined,
      isFallbackAdapter: info?.isFallbackAdapter ?? undefined,
      fp16: features.has('shader-f16'),
      deviceMemoryGB,
      cores,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxBufferSize: limits.maxBufferSize,
      maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    };

    return { ...partial, ...classify(partial), probedAt };
  } catch (e) {
    return {
      ok: false,
      tier: 0,
      error: e instanceof Error ? e.message : 'adapter request failed',
      fp16: false,
      deviceMemoryGB,
      cores,
      weightsBudgetMB: 0,
      trainableBudgetM: 0,
      probedAt,
    };
  }
}

export function fmtBytes(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}
