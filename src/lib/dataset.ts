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
  source: 'import' | 'template';
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
