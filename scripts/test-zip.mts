import { strict as assert } from 'node:assert';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import { zipStore } from '../src/lib/zip.ts';

const zip = zipStore([
  { name: 'model.card.json', data: new TextEncoder().encode('{"a":1}') },
  { name: 'weights/predict.mjs', data: new TextEncoder().encode('console.log(1)') },
]);

// EOCD tail check
const dv = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
assert.equal(dv.getUint32(zip.length - 22, true), 0x06054b50, 'EOCD signature');
assert.equal(dv.getUint16(zip.length - 12, true), 2, 'entry count');

// local header sig at start
assert.equal(dv.getUint32(0, true), 0x04034b50, 'local header signature');

// real-world check: extract with bsdtar (ships with Windows) or PowerShell
const dir = mkdtempSync(join(tmpdir(), 'anvil-zip-'));
const f = join(dir, 'test.zip');
writeFileSync(f, zip);
let listing = '';
try {
  listing = execSync(`tar -tf "${f}"`).toString();
} catch {
  const out = join(dir, 'out');
  execSync(
    `powershell -NoProfile -Command "Expand-Archive -LiteralPath '${f}' -DestinationPath '${out}'"`,
  );
  listing = execSync(`powershell -NoProfile -Command "(Get-ChildItem -Recurse '${out}').FullName"`).toString();
}
assert.match(listing, /model\.card\.json/);
assert.match(listing, /predict\.mjs/);
console.log('zip writer verified:', listing.trim().split(/\r?\n/).join(', '));
