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

// multi-task round-trip: head.<i>.* tensors + tasks table in meta
const mtBytes = writeSafetensors(
  {
    'head.0.weight': { shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) },
    'head.0.bias': { shape: [2], data: new Float32Array([0.1, 0.2]) },
    'head.1.weight': { shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) },
    'head.1.bias': { shape: [3], data: new Float32Array([0.1, 0.2, 0.3]) },
    'lora.0.q.a': { shape: [1], data: new Float32Array([7]) },
  },
  {
    mode: 'lora',
    anvil: 'ft-v1',
    tasks: JSON.stringify([
      { name: 'intent', labels: ['a', 'b'] },
      { name: 'urgency', labels: ['x', 'y', 'z'] },
    ]),
  },
);
const mtRead = readSafetensors(mtBytes.buffer as ArrayBuffer);
const mt = ftFromSafetensors(mtRead.tensors, mtRead.meta);
assert.equal(mt.tasks?.length, 2);
assert.deepEqual(mt.tasks?.[0].labels, ['a', 'b']);
assert.deepEqual(Array.from(mt.tasks?.[1].headW ?? []), [1, 2, 3, 4, 5, 6]);
assert.deepEqual(Array.from(mt.headW), [1, 2, 3, 4]); // task 0 mirrors headW
assert.deepEqual(mt.labels, ['a', 'b']);

console.log('full, LoRA and multi-task safetensors round-trips passed');
