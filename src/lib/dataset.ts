/**
 * Dataset builder primitives — client-side parsing of imported files.
 * Everything stays in memory (OPFS persistence is a later milestone).
 */

export interface DatasetRow {
  [key: string]: string;
}

export interface Dataset {
  name: string;
  sector: string;
  columns: string[];
  rows: DatasetRow[];
  source: 'import' | 'template' | 'starter' | 'hf';
  createdAt: string;
}

export class ParseError extends Error {}

/** Minimal RFC-4180-ish CSV parser — quoted fields, escaped quotes, commas. */
function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (c === '"') {
        inQuotes = false;
      } else {
        cur += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ',') {
      out.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

export function parseCsv(text: string): { columns: string[]; rows: DatasetRow[] } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new ParseError('CSV needs a header row and at least one data row.');
  const columns = parseCsvLine(lines[0]);
  if (columns.length === 0 || columns.some((c) => !c))
    throw new ParseError('Header row has empty column names.');
  const rows = lines.slice(1).map((l) => {
    const cells = parseCsvLine(l);
    const row: DatasetRow = {};
    columns.forEach((c, i) => (row[c] = cells[i] ?? ''));
    return row;
  });
  return { columns, rows };
}

export function parseJsonl(text: string): { columns: string[]; rows: DatasetRow[] } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  const rows: DatasetRow[] = [];
  const colSet = new Set<string>();
  for (const [i, line] of lines.entries()) {
    let obj: unknown;
    try {
      obj = JSON.parse(line);
    } catch {
      throw new ParseError(`Line ${i + 1} is not valid JSON.`);
    }
    if (typeof obj !== 'object' || obj === null || Array.isArray(obj))
      throw new ParseError(`Line ${i + 1} is not a JSON object.`);
    const row: DatasetRow = {};
    for (const [k, v] of Object.entries(obj)) {
      colSet.add(k);
      row[k] = typeof v === 'string' ? v : JSON.stringify(v);
    }
    rows.push(row);
  }
  if (rows.length === 0) throw new ParseError('No rows found.');
  return { columns: [...colSet], rows };
}

/** Plain text → one `text` column, one row per line. */
export function parseTxt(text: string): { columns: string[]; rows: DatasetRow[] } {
  const rows = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((t) => ({ text: t }));
  if (rows.length === 0) throw new ParseError('File is empty.');
  return { columns: ['text'], rows };
}

export async function parseFile(file: File): Promise<{ columns: string[]; rows: DatasetRow[] }> {
  const text = await file.text();
  const name = file.name.toLowerCase();
  if (name.endsWith('.jsonl') || name.endsWith('.ndjson')) return parseJsonl(text);
  if (name.endsWith('.csv')) return parseCsv(text);
  if (name.endsWith('.json')) {
    try {
      const arr = JSON.parse(text) as unknown;
      if (!Array.isArray(arr)) throw new ParseError('JSON file must be an array of objects.');
      return parseJsonl(arr.map((r) => JSON.stringify(r)).join('\n'));
    } catch (e) {
      if (e instanceof ParseError) throw e;
      throw new ParseError('File is not valid JSON.');
    }
  }
  if (name.endsWith('.txt')) return parseTxt(text);
  throw new ParseError('Unsupported format — use .jsonl, .csv, .json or .txt.');
}

/** Scaffold a template dataset from a sector schema. */
export function templateRows(schema: { field: string; role: string }[], n = 24): DatasetRow[] {
  return Array.from({ length: n }, (_, i) => {
    const row: DatasetRow = {};
    for (const f of schema) row[f.field] = f.role === 'input' ? `sample ${i + 1}` : '';
    return row;
  });
}

/* ---------------- dataset health ---------------- */

export interface ColumnStat {
  name: string;
  nonEmpty: number;
  unique: number;
  /** top values for label-shaped columns (≤ 64 uniques) */
  topValues: [string, number][];
  avgLen: number;
}

export interface DatasetHealth {
  rows: number;
  dupes: number;
  colStats: ColumnStat[];
  /** best-guess text column (highest avg length) and label column */
  textCol: string | null;
  labelGuess: string | null;
  /** class counts for the guessed label column, largest first */
  classCounts: [string, number][];
  warnings: string[];
}

/**
 * One pass over the rows: per-column empties/uniques/lengths, exact-duplicate
 * count, and plain-language warnings. A column is "label-shaped" when it has
 * 2–64 distinct values — what the trainer can actually consume.
 */
export function analyzeDataset(d: Dataset): DatasetHealth {
  const n = d.rows.length;
  const uniques = new Map<string, Map<string, number>>();
  const sums = new Map<string, number>();
  const nonEmpty = new Map<string, number>();
  for (const c of d.columns) {
    uniques.set(c, new Map());
    sums.set(c, 0);
    nonEmpty.set(c, 0);
  }
  const seen = new Set<string>();
  let dupes = 0;
  for (const r of d.rows) {
    let key = '';
    for (const c of d.columns) {
      const v = (r[c] ?? '').trim();
      key += `${c}\u001f${v}\u001f`;
      const m = uniques.get(c)!;
      m.set(v, (m.get(v) ?? 0) + 1);
      sums.set(c, sums.get(c)! + v.length);
      if (v) nonEmpty.set(c, nonEmpty.get(c)! + 1);
    }
    if (seen.has(key)) dupes++;
    else seen.add(key);
  }

  const colStats: ColumnStat[] = d.columns.map((c) => {
    const m = uniques.get(c)!;
    return {
      name: c,
      nonEmpty: nonEmpty.get(c)!,
      unique: m.size,
      topValues:
        m.size >= 2 && m.size <= 64
          ? [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8)
          : [],
      avgLen: n ? Math.round(sums.get(c)! / n) : 0,
    };
  });

  const textCol =
    colStats.length === 0
      ? null
      : colStats.reduce((a, b) => (b.avgLen > a.avgLen ? b : a)).name;
  const labelGuess =
    colStats
      .filter((c) => c.name !== textCol && c.unique >= 2 && c.unique <= 64)
      .sort((a, b) => a.unique - b.unique)[0]?.name ?? null;

  const classCounts = labelGuess
    ? ([...uniques.get(labelGuess)!.entries()]
        .filter(([v]) => v !== '')
        .sort((a, b) => b[1] - a[1]) as [string, number][])
    : [];

  const warnings: string[] = [];
  if (dupes > 0) warnings.push(`${dupes} duplicate row${dupes === 1 ? '' : 's'}`);
  for (const c of colStats) {
    const empty = n - c.nonEmpty;
    if (empty > 0 && empty / n > 0.25)
      warnings.push(`"${c.name}" is empty in ${Math.round((empty / n) * 100)}% of rows`);
  }
  if (labelGuess) {
    const top = classCounts[0];
    if (top && top[1] / n > 0.8)
      warnings.push(
        `"${labelGuess}" is ${Math.round((top[1] / n) * 100)}% "${top[0]}" — imbalanced`,
      );
    if (classCounts.some(([, cnt]) => cnt < 5))
      warnings.push('some classes have < 5 rows — they may not survive the val split');
  } else {
    warnings.push('no label-shaped column found (needs 2–64 distinct values)');
  }
  if (n > 512) warnings.push(`training uses the first 512 of ${n} rows`);

  return {
    rows: n,
    dupes,
    colStats,
    textCol,
    labelGuess,
    classCounts,
    warnings,
  };
}

/** Return a copy with exact-duplicate rows (all columns equal) removed. */
export function dedupeDataset(d: Dataset): Dataset {
  const seen = new Set<string>();
  const rows = d.rows.filter((r) => {
    const key = d.columns.map((c) => `${c}${(r[c] ?? '').trim()}`).join('');
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return { ...d, rows };
}
