import { readSafetensors } from '../safetensors.ts';
import { W, type Weights } from './spec.ts';

export const MINILM_SAFETENSORS =
  'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors';

/** Expected tensor names (BERT layout) we need for the encoder forward/backward. */
export function requiredKeys(layers: number): string[] {
  const keys = [W.word, W.pos, W.tokType, W.embLnW, W.embLnB];
  for (let i = 0; i < layers; i++) {
    const l = W.layer(i);
    keys.push(l.qW, l.qB, l.kW, l.kB, l.vW, l.vB, l.oW, l.oB, l.ln1W, l.ln1B,
      l.f1W, l.f1B, l.f2W, l.f2B, l.ln2W, l.ln2B);
  }
  return keys;
}

/**
 * Fetch + parse MiniLM weights (≈91 MB fp32). Handles the optional `bert.`
 * prefix some checkpoints carry. Progress callback gets 0..1 download fraction.
 */
export async function loadMiniLMWeights(
  onProgress?: (frac: number) => void,
): Promise<Weights> {
  const res = await fetch(MINILM_SAFETENSORS);
  if (!res.ok || !res.body) throw new Error(`weights fetch failed: ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress?.(got / total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  const { tensors } = readSafetensors(buf.buffer);

  // normalize keys: strip a leading "bert." if present
  const out: Weights = {};
  const need = new Set(requiredKeys(6));
  for (const [name, t] of Object.entries(tensors)) {
    const key = name.startsWith('bert.') ? name.slice(5) : name;
    if (need.has(key) && t.dtype === 'F32') out[key] = t.data as Float32Array;
  }
  const missing = [...need].filter((k) => !(k in out));
  if (missing.length) throw new Error(`safetensors missing ${missing.length} tensors, e.g. ${missing[0]}`);
  return out;
}
