import {
  pipeline,
  type FeatureExtractionPipeline,
  type ProgressInfo,
} from '@huggingface/transformers';

/**
 * Frozen embedder — MiniLM-L6 mean-pooled normalized embeddings (384-dim).
 * Loaded lazily; transformers.js caches weights in the browser so later runs
 * are instant. WebGPU when available, WASM otherwise.
 */
export const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
export const EMBED_DIM = 384;

let pipe: FeatureExtractionPipeline | null = null;
let loading: Promise<FeatureExtractionPipeline> | null = null;

export type EmbedProgress = (info: ProgressInfo) => void;

export async function loadEmbedder(
  onProgress?: EmbedProgress,
): Promise<FeatureExtractionPipeline> {
  if (pipe) return pipe;
  if (!loading) {
    loading = pipeline('feature-extraction', EMBED_MODEL, {
      dtype: 'fp32',
      progress_callback: onProgress,
    }) as Promise<FeatureExtractionPipeline>;
  }
  pipe = await loading;
  return pipe;
}

/** Embed a batch of texts → Float32Array [n, 384], mean-pooled + normalized. */
export async function embedTexts(
  texts: string[],
  onRow?: (done: number, total: number) => void,
): Promise<Float32Array> {
  const p = await loadEmbedder();
  const out = new Float32Array(texts.length * EMBED_DIM);
  const B = 16;
  for (let i = 0; i < texts.length; i += B) {
    const batch = texts.slice(i, i + B);
    const emb = await p(batch, { pooling: 'mean', normalize: true });
    const data = emb.data as Float32Array;
    out.set(data, i * EMBED_DIM);
    onRow?.(Math.min(i + B, texts.length), texts.length);
  }
  return out;
}
