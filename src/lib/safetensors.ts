/**
 * Minimal safetensors writer/reader — F32 tensors only (all we emit).
 * Format: u64 LE header length, JSON header (name → {dtype,shape,data_offsets}
 * + optional "__metadata__"), then concatenated raw little-endian f32 data.
 */

export interface TensorDef {
  shape: number[];
  data: Float32Array | BigInt64Array;
  dtype?: 'F32' | 'I64';
}

export function writeSafetensors(
  tensors: Record<string, TensorDef>,
  meta: Record<string, string> = {},
): Uint8Array {
  const names = Object.keys(tensors);
  let offset = 0;
  const headerObj: Record<string, unknown> = {};
  for (const name of names) {
    const t = tensors[name];
    headerObj[name] = {
      dtype: 'F32',
      shape: t.shape,
      data_offsets: [offset, offset + t.data.byteLength],
    };
    offset += t.data.byteLength;
  }
  if (Object.keys(meta).length) headerObj.__metadata__ = meta;

  let header = JSON.stringify(headerObj);
  // pad with spaces so (8 + headerLen) % 8 === 0
  const pad = (8 - ((8 + header.length) % 8)) % 8;
  header += ' '.repeat(pad);

  const enc = new TextEncoder();
  const hBytes = enc.encode(header);
  const out = new Uint8Array(8 + hBytes.length + offset);
  const dv = new DataView(out.buffer);
  dv.setBigUint64(0, BigInt(hBytes.length), true);
  out.set(hBytes, 8);
  let o = 8 + hBytes.length;
  for (const name of names) {
    out.set(new Uint8Array(tensors[name].data.buffer, tensors[name].data.byteOffset, tensors[name].data.byteLength), o);
    o += tensors[name].data.byteLength;
  }
  return out;
}

export function readSafetensors(buf: ArrayBuffer): {
  tensors: Record<string, TensorDef>;
  meta: Record<string, string>;
} {
  const dv = new DataView(buf);
  const hLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hLen)));
  const meta = (header.__metadata__ ?? {}) as Record<string, string>;
  const tensors: Record<string, TensorDef> = {};
  for (const [name, t] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const def = t as { dtype: string; shape: number[]; data_offsets: [number, number] };
    const [s, e] = def.data_offsets;
    const slice = buf.slice(8 + hLen + s, 8 + hLen + e);
    if (def.dtype === 'I64') {
      tensors[name] = { shape: def.shape, data: new BigInt64Array(slice), dtype: 'I64' };
    } else if (def.dtype === 'F32') {
      tensors[name] = { shape: def.shape, data: new Float32Array(slice), dtype: 'F32' };
    } else {
      throw new Error(`unsupported dtype ${def.dtype}`);
    }
  }
  return { tensors, meta };
}

export function downloadBytes(name: string, bytes: Uint8Array) {
  const url = URL.createObjectURL(new Blob([bytes.buffer as ArrayBuffer], { type: 'application/octet-stream' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
