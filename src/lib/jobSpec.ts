import type { Dataset, DatasetRow } from './dataset';
import type { ModelEntry, TrainMode } from '../data/models';

export interface JobTrainingConfig {
  mode: TrainMode;
  lr: number;
  epochs: number;
  batch: number;
  seqLen: number;
}

export interface AnvilJobSpec {
  tool: 'anvil';
  specVersion: 1;
  createdAt: string;
  trainer: 'sentence-transformers/all-MiniLM-L6-v2';
  targetModel: string | null;
  dataset: Dataset;
  training: JobTrainingConfig;
  rank: number;
  columns: { text: string; label: string };
}

export function createJobSpec(input: {
  targetModel: ModelEntry | null;
  dataset: Dataset;
  training: JobTrainingConfig;
  rank: number;
  textColumn: string;
  labelColumn: string;
}): AnvilJobSpec {
  return {
    tool: 'anvil',
    specVersion: 1,
    createdAt: new Date().toISOString(),
    trainer: 'sentence-transformers/all-MiniLM-L6-v2',
    targetModel: input.targetModel?.hf ?? null,
    dataset: input.dataset,
    training: input.training,
    rank: input.rank,
    columns: { text: input.textColumn, label: input.labelColumn },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseDataset(value: unknown): Dataset {
  if (!isRecord(value)) throw new Error('Job is missing its dataset.');
  const { name, sector, columns, rows, source, createdAt } = value;
  if (
    typeof name !== 'string' ||
    typeof sector !== 'string' ||
    typeof createdAt !== 'string' ||
    !Array.isArray(columns) ||
    !columns.every((column): column is string => typeof column === 'string') ||
    columns.length === 0 ||
    !Array.isArray(rows) ||
    rows.length > 100_000 ||
    !['import', 'template', 'starter', 'hf'].includes(String(source))
  ) {
    throw new Error('Job dataset has an invalid shape.');
  }
  const parsedRows: DatasetRow[] = rows.map((row) => {
    if (!isRecord(row)) throw new Error('Job dataset contains an invalid row.');
    const parsed: DatasetRow = {};
    for (const column of columns) {
      const cell = row[column];
      if (cell !== undefined && typeof cell !== 'string') {
        throw new Error(`Job dataset column "${column}" must contain text values.`);
      }
      parsed[column] = cell ?? '';
    }
    return parsed;
  });
  return {
    name,
    sector,
    columns,
    rows: parsedRows,
    source: source as Dataset['source'],
    createdAt,
  };
}

export function parseJobSpec(text: string): AnvilJobSpec {
  const value: unknown = JSON.parse(text);
  if (!isRecord(value) || value.tool !== 'anvil' || value.specVersion !== 1) {
    throw new Error('Not a supported Anvil job file.');
  }
  const config = value.training;
  const columns = value.columns;
  if (!isRecord(config) || !isRecord(columns)) throw new Error('Job is missing training settings.');
  const mode = config.mode;
  const lr = config.lr;
  const epochs = config.epochs;
  const batch = config.batch;
  const seqLen = config.seqLen;
  const rank = value.rank;
  const targetModel = value.targetModel;
  const textColumn = columns.text;
  const labelColumn = columns.label;
  if (
    !['head', 'lora', 'full'].includes(String(mode)) ||
    typeof lr !== 'number' || !Number.isFinite(lr) || lr <= 0 || lr > 1 ||
    typeof epochs !== 'number' || !Number.isInteger(epochs) || epochs < 1 || epochs > 40 ||
    typeof batch !== 'number' || !Number.isInteger(batch) || batch < 1 || batch > 64 ||
    typeof seqLen !== 'number' || !Number.isInteger(seqLen) || seqLen < 32 || seqLen > 128 ||
    ![4, 8, 16, 32].includes(Number(rank)) ||
    (targetModel !== null && typeof targetModel !== 'string') ||
    typeof textColumn !== 'string' || typeof labelColumn !== 'string'
  ) {
    throw new Error('Job training settings are invalid.');
  }
  const dataset = parseDataset(value.dataset);
  if (!dataset.columns.includes(textColumn) || !dataset.columns.includes(labelColumn)) {
    throw new Error('Job column mapping does not match its dataset.');
  }
  return {
    tool: 'anvil',
    specVersion: 1,
    createdAt: typeof value.createdAt === 'string' ? value.createdAt : '',
    trainer: 'sentence-transformers/all-MiniLM-L6-v2',
    targetModel,
    dataset,
    training: { mode: mode as TrainMode, lr, epochs, batch, seqLen },
    rank: Number(rank),
    columns: { text: textColumn, label: labelColumn },
  };
}
