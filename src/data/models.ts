export type Sector =
  | 'decision'
  | 'scientific'
  | 'labelling'
  | 'trust'
  | 'general';

export type Arch = 'token-cls' | 'extractor' | 'decoder-lm' | 'guardrail';

export type TrainMode = 'head' | 'lora' | 'full';

export interface ModelEntry {
  /** Hugging Face repo id */
  hf: string;
  name: string;
  paramsM: number;
  /** Approx. fp32 weights on disk, MB */
  weightsMB: number;
  arch: Arch;
  family: 'fastino' | 'community';
  sectors: Sector[];
  /** Minimum hardware tier to consider training it locally (1..4) */
  minTier: 1 | 2 | 3 | 4;
  trainMode: TrainMode[];
  /** Inference runtime available in-browser today */
  runtime: 'onnx-webgpu' | 'wgsl' | 'planned';
  /** Browser-loadable ONNX repo (open-jev compatible), if one exists */
  onnx?: string;
  blurb: string;
}

export const TIER_LABELS = [
  'T0 · no WebGPU',
  'T1 · entry',
  'T2 · standard',
  'T3 · capable',
  'T4 · discrete GPU',
] as const;

export const MODELS: ModelEntry[] = [
  {
    hf: 'fastino/gliner2.5-small-v1',
    name: 'GLiNER2.5 Small',
    paramsM: 74,
    weightsMB: 296,
    arch: 'extractor',
    family: 'fastino',
    sectors: ['labelling', 'scientific'],
    minTier: 1,
    trainMode: ['head', 'full'],
    runtime: 'planned',
    blurb:
      'DeBERTa-v3-xsmall boundary extractor. Entities, records and relations in one pass. Small enough for full fine-tuning in a tab.',
  },
  {
    hf: 'fastino/gliner2.5-base-v1',
    name: 'GLiNER2.5 Base',
    paramsM: 194,
    weightsMB: 776,
    arch: 'extractor',
    family: 'fastino',
    sectors: ['labelling', 'scientific'],
    minTier: 2,
    trainMode: ['head', 'lora'],
    runtime: 'planned',
    blurb:
      'Default English multi-task checkpoint. Joint entity–relation extraction, constrained classification, span attributes.',
  },
  {
    hf: 'fastino/gliner2.5-multi-v1',
    name: 'GLiNER2.5 Multi',
    paramsM: 287,
    weightsMB: 1148,
    arch: 'extractor',
    family: 'fastino',
    sectors: ['labelling', 'scientific', 'general'],
    minTier: 2,
    trainMode: ['lora'],
    runtime: 'planned',
    blurb:
      'Multilingual (mDeBERTa) multi-task extractor. The general-purpose workhorse for cross-language labelling pipelines.',
  },
  {
    hf: 'fastino/GLiNER2-Guardrails-PII-Multi',
    name: 'GLiNER2 Guardrails PII',
    paramsM: 300,
    weightsMB: 1200,
    arch: 'guardrail',
    family: 'fastino',
    sectors: ['trust', 'labelling'],
    minTier: 2,
    trainMode: ['head', 'lora'],
    runtime: 'planned',
    blurb:
      'PII detection and safety moderation in a single forward pass. Fine-tune it on your own redaction policy.',
  },
  {
    hf: 'fastino/GLiNER2.5-multi-Decide',
    name: 'GLiNER2.5 Decide Multi',
    paramsM: 300,
    weightsMB: 1200,
    arch: 'token-cls',
    family: 'fastino',
    sectors: ['decision', 'general'],
    minTier: 2,
    trainMode: ['lora'],
    runtime: 'planned',
    blurb:
      'Multilingual decision head: typed labels, confidence scores, and constraint enforcement between labels.',
  },
  {
    hf: 'fastino/GLiNER2.5-Decide',
    name: 'GLiNER2.5 Decide',
    paramsM: 340,
    weightsMB: 1740,
    arch: 'token-cls',
    family: 'fastino',
    sectors: ['decision', 'scientific'],
    minTier: 3,
    trainMode: ['lora', 'full'],
    runtime: 'onnx-webgpu',
    onnx: 'onnx-community/GLiNER2.5-Decide-ONNX',
    blurb:
      'The flagship small decision model. Typed decisions with confidence, span + relation output, full and LoRA fine-tuning.',
  },
  {
    hf: 'fastino/GLiNER2.5-Decide-1B',
    name: 'GLiNER2.5 Decide 1B',
    paramsM: 1024,
    weightsMB: 4096,
    arch: 'token-cls',
    family: 'fastino',
    sectors: ['decision'],
    minTier: 4,
    trainMode: ['lora'],
    runtime: 'planned',
    blurb:
      'Top of the local envelope. The largest Decide checkpoint — for discrete GPUs with real VRAM headroom.',
  },
  {
    hf: 'onnx-community/SmolLM2-135M-Instruct',
    name: 'SmolLM2 135M',
    paramsM: 135,
    weightsMB: 540,
    arch: 'decoder-lm',
    family: 'community',
    sectors: ['general'],
    minTier: 1,
    trainMode: ['full', 'lora'],
    runtime: 'onnx-webgpu',
    blurb:
      'Smallest usable instruct LM. Ships as q4f16 ONNX — runs today via Transformers.js, trainable head-first.',
  },
  {
    hf: 'onnx-community/Qwen2.5-0.5B-Instruct',
    name: 'Qwen2.5 0.5B',
    paramsM: 494,
    weightsMB: 1976,
    arch: 'decoder-lm',
    family: 'community',
    sectors: ['general', 'scientific'],
    minTier: 3,
    trainMode: ['lora'],
    runtime: 'onnx-webgpu',
    blurb:
      'Best quality-per-parameter in the half-billion bracket. Strong instruction following; LoRA-tune on discrete GPUs.',
  },
  {
    hf: 'Qwen/Qwen3-0.6B',
    name: 'Qwen3 0.6B',
    paramsM: 600,
    weightsMB: 2400,
    arch: 'decoder-lm',
    family: 'community',
    sectors: ['general', 'scientific'],
    minTier: 3,
    trainMode: ['lora'],
    runtime: 'onnx-webgpu',
    blurb:
      'Hybrid-thinking tiny model. The right pick when decisions need a reasoning trace before the label.',
  },
  {
    hf: 'onnx-community/Llama-3.2-1B-Instruct',
    name: 'Llama 3.2 1B',
    paramsM: 1240,
    weightsMB: 4960,
    arch: 'decoder-lm',
    family: 'community',
    sectors: ['general'],
    minTier: 4,
    trainMode: ['lora'],
    runtime: 'onnx-webgpu',
    blurb:
      'The 1B ceiling. Widest tooling support of any browser-runnable LM; LoRA adapters only, and only on real GPUs.',
  },
];

/** Worth knowing, not in the primary roster. */
export const HONORABLE_MENTIONS: ModelEntry[] = [
  {
    hf: 'onnx-community/SmolLM2-360M-Instruct',
    name: 'SmolLM2 360M',
    paramsM: 360,
    weightsMB: 1440,
    arch: 'decoder-lm',
    family: 'community',
    sectors: ['general'],
    minTier: 2,
    trainMode: ['lora'],
    runtime: 'onnx-webgpu',
    blurb: 'Middle child of the Smol line — noticeably smarter than 135M.',
  },
  {
    hf: 'onnx-community/gemma-3-270m-it-ONNX',
    name: 'Gemma 3 270M',
    paramsM: 270,
    weightsMB: 1080,
    arch: 'decoder-lm',
    family: 'community',
    sectors: ['general'],
    minTier: 2,
    trainMode: ['lora'],
    runtime: 'onnx-webgpu',
    blurb: 'Google’s purpose-built edge model; extremely memory-efficient arch.',
  },
  {
    hf: 'fastino/gliguard-LLMGuardrails-300M',
    name: 'GLiGuard 300M',
    paramsM: 200,
    weightsMB: 800,
    arch: 'guardrail',
    family: 'fastino',
    sectors: ['trust'],
    minTier: 2,
    trainMode: ['head'],
    runtime: 'planned',
    blurb: 'Prompt-injection and jailbreak screening for the labelling pipeline.',
  },
];

export const ARCH_LABELS: Record<Arch, string> = {
  'token-cls': 'decision head',
  extractor: 'extractor',
  'decoder-lm': 'decoder LM',
  guardrail: 'guardrail',
};

export function fitsTier(m: ModelEntry, tier: number): 'fits' | 'marginal' | 'no' {
  if (tier >= m.minTier) return 'fits';
  if (tier === m.minTier - 1 && m.trainMode.includes('lora')) return 'marginal';
  return 'no';
}
