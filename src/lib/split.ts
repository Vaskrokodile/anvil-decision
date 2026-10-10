/**
 * Seeded stratified train/val split over row class indices. Every class keeps
 * at least one row in train; singleton classes never leave train. Shared by
 * both training workers so "val acc" means the same thing everywhere.
 */

export function mulberry(seed: number) {
  let s = seed >>> 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function stratifiedSplit(
  y: Uint32Array,
  valFrac = 0.2,
  seed = 1337,
): { train: number[]; val: number[] } {
  const frac = Math.max(0, Math.min(0.5, valFrac));
  const byClass = new Map<number, number[]>();
  y.forEach((c, i) => {
    const a = byClass.get(c) ?? [];
    a.push(i);
    byClass.set(c, a);
  });
  const rng = mulberry(seed);
  const train: number[] = [];
  const val: number[] = [];
  for (const idx of byClass.values()) {
    for (let k = idx.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [idx[k], idx[j]] = [idx[j], idx[k]];
    }
    const take = Math.min(idx.length - 1, Math.floor(idx.length * frac));
    val.push(...idx.slice(0, Math.max(0, take)));
    train.push(...idx.slice(Math.max(0, take)));
  }
  // global shuffle so batches mix classes
  for (let k = train.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [train[k], train[j]] = [train[j], train[k]];
  }
  for (let k = val.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [val[k], val[j]] = [val[j], val[k]];
  }
  return { train, val };
}
