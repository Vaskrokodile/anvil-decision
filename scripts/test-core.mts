import { strict as assert } from 'node:assert';
import { analyzeDataset, dedupeDataset, parseCsv, type Dataset } from '../src/lib/dataset.ts';
import { createJobSpec, parseJobSpec } from '../src/lib/jobSpec.ts';
import { ftFromSafetensors } from '../src/lib/xformer/ftArtifact.ts';
import { readSafetensors, writeSafetensors } from '../src/lib/safetensors.ts';

const csv = parseCsv('text,label\r\n"two, parts\nsecond line",yes\r\nthird,no\r\n');
assert.deepEqual(csv.columns, ['text', 'label']);
assert.equal(csv.rows[0].text, 'two, parts\nsecond line');
assert.throws(() => parseCsv('text,label\n"unfinished,yes'), /unclosed quoted field/);

const dataset: Dataset = {
  name: 'test',
  sector: 'custom',
  columns: ['text', 'label'],
  rows: [
    { text: 'hello ', label: 'yes' },
    { text: 'hello', label: 'yes ' },
    { text: 'world', label: 'no' },
  ],
  source: 'import',
  createdAt: new Date(0).toISOString(),
};
assert.equal(analyzeDataset(dataset).dupes, 1);
assert.equal(dedupeDataset(dataset).rows.length, 2);

const job = createJobSpec({
  targetModel: null,
  dataset,
  training: { mode: 'lora', lr: 1e-4, epochs: 3, batch: 8, seqLen: 128 },
  rank: 8,
  textColumn: 'text',
  labelColumn: 'label',
});
const restored = parseJobSpec(JSON.stringify(job));
assert.equal(restored.dataset.rows.length, dataset.rows.length);
assert.equal(restored.training.seqLen, 128);
assert.equal(restored.rank, 8);
assert.throws(() => parseJobSpec(JSON.stringify({ ...job, specVersion: 2 })), /Not a supported/);

const fullKey = 'encoder.layer.0.attention.self.query.weight';
const baseTensors = {
  'head.weight': { shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) },
  'head.bias': { shape: [2], data: new Float32Array([0.1, 0.2]) },
};
const fullBytes = writeSafetensors(
  {
    ...baseTensors,
    [fullKey]: { shape: [1], data: new Float32Array([5]) },
  },
  { mode: 'full', labels: 'yes\nno' },
);
const fullRead = readSafetensors(fullBytes.buffer as ArrayBuffer);
const full = ftFromSafetensors(fullRead.tensors, fullRead.meta);
assert.equal(full.mode, 'full');
assert.deepEqual(Array.from(full.params[fullKey]), [5]);
assert.deepEqual(Array.from(full.headW), [1, 2, 3, 4]);

const loraBytes = writeSafetensors(
  {
    ...baseTensors,
    'lora.0.q.a': { shape: [1], data: new Float32Array([6]) },
  },
  { mode: 'lora', labels: 'yes\nno' },
);
const loraRead = readSafetensors(loraBytes.buffer as ArrayBuffer);
const lora = ftFromSafetensors(loraRead.tensors, loraRead.meta);
assert.equal(lora.mode, 'lora');
assert.deepEqual(Array.from(lora.params['lora.0.q.a']), [6]);

console.log('dataset parsing, job bundle, and head/LoRA/full artifact tests passed');
