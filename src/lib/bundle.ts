import { zipStore, type ZipEntry } from './zip';
import type { TrainedHead } from './trainedHead';
import type { FtArtifact } from './xformer/ft';

/**
 * Export bundles: the .safetensors weights + a machine-readable model card +
 * (for head artifacts) a runnable Node predictor that only needs
 * @huggingface/transformers — no Anvil code.
 */

const enc = new TextEncoder();
const utf8 = (s: string) => enc.encode(s);

/** Standalone predictor for head artifacts: embed + linear layer, ~no deps. */
function predictSnippet(labels: string[]): string {
  return `// anvil-head predict — usage: node predict.mjs "some text"
// needs: npm i @huggingface/transformers   (model downloads once, then cached)
import { pipeline } from '@huggingface/transformers';
import { readFileSync } from 'node:fs';

const LABELS = ${JSON.stringify(labels)};
const MODEL = 'Xenova/all-MiniLM-L6-v2';

// minimal f32 safetensors reader
function readSafetensors(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = Number(dv.getBigUint64(0, true));
  const meta = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + n)));
  const tensors = {};
  for (const [name, t] of Object.entries(meta)) {
    if (name === '__metadata__') continue;
    const [s, e] = t.data_offsets;
    tensors[name] = {
      shape: t.shape,
      data: new Float32Array(buf.buffer, buf.byteOffset + 8 + n + s, (e - s) / 4),
    };
  }
  return tensors;
}

const path = new URL('./anvil-head.safetensors', import.meta.url);
const { 'head.weight': W, 'head.bias': B } = readSafetensors(readFileSync(path));
const C = LABELS.length;

const pipe = await pipeline('feature-extraction', MODEL);
const text = process.argv[2] ?? 'hello world';
const emb = (await pipe(text, { pooling: 'mean', normalize: true })).data;

const logits = Array.from({ length: C }, (_, j) => {
  let s = B.data[j];
  for (let k = 0; k < emb.length; k++) s += emb[k] * W.data[k * C + j];
  return s;
});
const mx = Math.max(...logits);
const exps = logits.map((x) => Math.exp(x - mx));
const sum = exps.reduce((a, b) => a + b, 0);
for (const [i, l] of LABELS.entries())
  console.log(\`\${l.padEnd(24)} \${((exps[i] / sum) * 100).toFixed(1)}%\`);
`;
}

const FT_NOTES = `# anvil-ft artifact

This bundle contains a fine-tuned MiniLM-L6 encoder artifact:

- \`head.*\` — linear classifier head(s) over the [CLS] row
- \`lora.<layer>.<q|k|v|o|f1|f2>.<a|b>\` — LoRA adapters (lora mode)
- \`encoder.layer.*\`, \`embeddings.LayerNorm.*\` — full-weight deltas (full mode)

Adapter application happens *inside* the encoder forward pass, so the artifact
cannot run on a stock ONNX Runtime — load it in Anvil's Run step
(#/studio → 04 Run → "load .safetensors"), which applies the tuned params in
its own WGSL/CPU encoder backend.
`;

export function headCard(head: TrainedHead): Record<string, unknown> {
  return {
    format: 'anvil-bundle-v1',
    kind: 'head',
    arch: 'all-MiniLM-L6-v2 embeddings (frozen) + linear head',
    encoder: head.encoder,
    labels: head.labels,
    metrics: {
      loss: head.loss,
      acc: head.acc,
      valAcc: head.valAcc ?? null,
      valN: head.valN ?? 0,
      trainN: head.trainN ?? head.rows,
    },
    rows: head.rows,
    backend: head.backend,
    trainedAt: head.trainedAt,
    files: {
      'anvil-head.safetensors': 'head.weight [384, C] + head.bias [C], f32',
      'predict.mjs': 'standalone Node predictor (npm i @huggingface/transformers)',
    },
  };
}

export function ftCard(ft: FtArtifact): Record<string, unknown> {
  const deberta = ft.arch === 'deberta';
  return {
    format: 'anvil-bundle-v1',
    kind: 'ft',
    arch: `${deberta ? 'deberta-v3-small' : 'all-MiniLM-L6-v2'} encoder · ${ft.mode}${ft.mode === 'lora' ? ` r=${ft.rank}` : ''} + head${ft.tasks && ft.tasks.length > 1 ? `s (${ft.tasks.length} tasks)` : ''}`,
    archFamily: ft.arch ?? 'minilm',
    encoder: deberta
      ? 'cross-encoder/nli-deberta-v3-small'
      : 'sentence-transformers/all-MiniLM-L6-v2',
    mode: ft.mode,
    rank: ft.rank,
    maxSeq: ft.maxSeq,
    labels: ft.labels,
    tasks: ft.tasks?.map((t) => ({ name: t.name, labels: t.labels })),
    metrics: {
      loss: ft.loss,
      acc: ft.acc,
      valAcc: ft.valAcc ?? null,
      valN: ft.valN ?? 0,
      trainN: ft.trainN ?? ft.rows,
      ...(ft.tasks && ft.tasks.length > 1
        ? {
            perTask: ft.tasks.map((t) => ({
              name: t.name,
              valAcc: t.valAcc ?? null,
              valN: t.valN ?? 0,
            })),
          }
        : {}),
    },
    rows: ft.rows,
    backend: ft.backend,
    trainedAt: ft.trainedAt,
    runtime: 'anvil (WGSL/CPU encoder backend) — see USING.md',
    files: {
      'anvil-ft.safetensors': 'tuned encoder params + head tensors, f32',
      'USING.md': 'adapter layout + how to run',
    },
  };
}

export function bundleEntries(
  stName: string,
  stBytes: Uint8Array,
  card: Record<string, unknown>,
  extra: ZipEntry[],
): Uint8Array {
  return zipStore([
    { name: stName, data: stBytes },
    { name: 'model.card.json', data: utf8(JSON.stringify(card, null, 2)) },
    ...extra,
  ]);
}

export function headBundleExtras(head: TrainedHead): ZipEntry[] {
  return [{ name: 'predict.mjs', data: utf8(predictSnippet(head.labels)) }];
}

export function ftBundleExtras(): ZipEntry[] {
  return [{ name: 'USING.md', data: utf8(FT_NOTES) }];
}
