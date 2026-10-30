import { strict as assert } from 'node:assert';
import { ftFromSafetensors } from '../src/lib/xformer/ftArtifact.ts';
import { readSafetensors, writeSafetensors } from '../src/lib/safetensors.ts';

const fullKey = 'encoder.layer.0.attention.self.query.weight';
const tensors = {
  'head.weight': { shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) },
  'head.bias': { shape: [2], data: new Float32Array([0.1, 0.2]) },
  [fullKey]: { shape: [1], data: new Float32Array([5]) },
};
const fullBytes = writeSafetensors(tensors, { mode: 'full', labels: 'a\nb' });
const fullRead = readSafetensors(fullBytes.buffer as ArrayBuffer);
const full = ftFromSafetensors(fullRead.tensors, fullRead.meta);
assert.equal(full.mode, 'full');
assert.deepEqual(Array.from(full.params[fullKey]), [5]);
assert.deepEqual(Array.from(full.headW), [1, 2, 3, 4]);

const loraBytes = writeSafetensors(
  {
    'head.weight': tensors['head.weight'],
    'head.bias': tensors['head.bias'],
    'lora.0.q.a': { shape: [1], data: new Float32Array([6]) },
  },
  { mode: 'lora', labels: 'a\nb' },
);
const loraRead = readSafetensors(loraBytes.buffer as ArrayBuffer);
const lora = ftFromSafetensors(loraRead.tensors, loraRead.meta);
assert.equal(lora.mode, 'lora');
assert.deepEqual(Array.from(lora.params['lora.0.q.a']), [6]);

console.log('full and LoRA safetensors round-trips passed');
