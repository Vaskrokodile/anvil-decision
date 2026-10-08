import {
  AutoConfig,
  AutoModel,
  AutoTokenizer,
  Tensor,
  type PreTrainedTokenizer,
  type ProgressInfo,
} from '@huggingface/transformers';
import type { Answer, LoadProgress, Question } from 'open-jev';

/**
 * Driver for `gliner2`-config ONNX exports of the GLiNER2.5-Decide
 * classification path (open-jev 0.1.2 only knows `open_jev`/`kev` configs).
 *
 * Wire format — reproduced from fastino-ai/GLiNER2 `gliner2/processor.py`
 * (`_transform_schema`, `_format_input_with_mapping`, `_normalize_text`) and
 * checked against `conversion/verify_parity.py` in the model repo:
 *
 *   one schema block per question:
 *     "(" "[P]" prompt "(" ("[L]" label)* ")" ")"
 *   blocks joined by "[SEP_STRUCT]", then "[SEP_TEXT]", then the state words:
 *     ( [P] Which team? [DESCRIPTION] billing: charges ( [L] billing [L] tech ) )
 *     [SEP_STRUCT] ( [P] Angry? ( [L] no [L] yes ) ) [SEP_TEXT] the export … .
 *
 * - Every element above is tokenized INDEPENDENTLY with
 *   `add_special_tokens: false` and the ids are concatenated — the processor
 *   calls `tokenizer.tokenize(item)` per schema token / per word. Prompt and
 *   labels keep their case. No [CLS]/[SEP] anywhere.
 * - `prompt` = question instructions, plus ` [DESCRIPTION] <label>: <desc>`
 *   appended per described option (option order). Descriptions live in the
 *   prompt, not in the label tokens.
 * - `noul` becomes a two-label task with open-jev's NOUL ordering ["no","yes"];
 *   `score` keeps its ordered levels as labels.
 * - The state text is normalized first: empty -> ".", otherwise a terminal
 *   "." is appended unless it already ends with ".", "!" or "?". Then it is
 *   split with the processor's WhitespaceTokenSplitter regex and each word is
 *   lowercased (see WORD_SPLITTER below).
 * - `marker_positions` (int64) = sequence index of every "[L]" token, in
 *   question order. Output `logits` is [batch, markers] — one logit per
 *   marker; softmax within each question's contiguous slice gives the
 *   distribution (Python divides logits by `temperature` first; default 1.0).
 * - `max_len` (512) is the encoder's positional ceiling: schema is always kept
 *   whole and the state is cut at the end, on whole-word boundaries.
 */
export const DECIDE_REPO = 'onnx-community/GLiNER2.5-Decide-ONNX';

/** Repo ids that resolve to the Decide ONNX export (base repo -> ONNX mirror). */
const DECIDE_REPOS: Record<string, string> = {
  [DECIDE_REPO]: DECIDE_REPO,
  'fastino/GLiNER2.5-Decide': DECIDE_REPO,
};

/** True when `repo` names a GLiNER2.5-Decide checkpoint this driver can run. */
export function isDecideRepo(repo: string | undefined): boolean {
  return repo != null && repo in DECIDE_REPOS;
}

export type DecideDtype = 'fp32' | 'fp16' | 'q4' | 'q4f16' | 'auto';
export type DecideDevice = 'auto' | 'webgpu' | 'wasm' | 'cpu';

export type DecideLoadOptions = {
  /** Weight variant. `auto` = fp16 on WebGPU with shader-f16, else q4. */
  dtype?: DecideDtype;
  /** Backend. `auto` = webgpu when available, cpu under Node, else wasm. */
  device?: DecideDevice;
  onProgress?: (p: LoadProgress) => void;
};

export type DecideCallOptions = {
  /** Softmax temperature (default: the config's `gliner2.temperature`, 1.0). */
  temperature?: number;
  /** `cut` (default) drops trailing state words; `error` throws instead. */
  truncation?: 'cut' | 'error';
};

type DecideModel = {
  (inputs: Record<string, Tensor>): Promise<{ logits: Tensor }>;
  dispose?: () => Promise<unknown>;
  sessions?: Record<string, { inputNames?: string[] } | undefined>;
};

type Gliner2Section = { max_len?: number; temperature?: number };

const NOUL_LABELS = ['no', 'yes'] as const;
const SEP_STRUCT = '[SEP_STRUCT]';
const SEP_TEXT = '[SEP_TEXT]';
const P_TOKEN = '[P]';
const L_TOKEN = '[L]';
const DESC_TOKEN = '[DESCRIPTION]';

// gliner2.processing.word_splitter.WhitespaceTokenSplitter: Python `\w`
// (alphanumerics + "_") plus all combining marks and ZWNJ/ZWJ.
const WORD_CHAR = '\\p{L}\\p{N}\\p{M}_\\u200c\\u200d';
const WORD_SPLITTER = new RegExp(
  `https?://\\S+|www\\.\\S+` +
    `|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}` +
    `|@[a-z0-9_]+` +
    `|[${WORD_CHAR}]+(?:[-_][${WORD_CHAR}]+)*` +
    `|\\S`,
  'giu',
);

function normalizeState(text: string): string {
  if (!text) return '.';
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

/** Lowercased word list, matching `word_splitter(text, lower=True)`. */
function splitWords(text: string): string[] {
  const words: string[] = [];
  for (const m of text.matchAll(WORD_SPLITTER)) words.push(m[0].toLowerCase());
  return words;
}

function validateQuestion(question: Question, index: number): void {
  const label = `#${index}`;
  if (!question || typeof question !== 'object') {
    throw new Error(`Question ${label} must be an object.`);
  }
  if (typeof question.instructions !== 'string' || question.instructions.trim() === '') {
    throw new Error(`Question ${label} needs non-empty instructions.`);
  }
  if (question.type === 'noul') return;
  if (question.type !== 'choice' && question.type !== 'score') {
    throw new Error(
      `Question ${label} has unknown type "${String(
        (question as { type: unknown }).type,
      )}". Expected "choice", "score" or "noul".`,
    );
  }
  if (!Array.isArray(question.options) || question.options.length === 0) {
    throw new Error(`Question ${label} (${question.type}) needs at least one option.`);
  }
  const seen = new Set<string>();
  for (const option of question.options) {
    if (typeof option !== 'string' || option.trim() === '') {
      throw new Error(`Question ${label} has an empty or non-string option.`);
    }
    if (seen.has(option)) {
      throw new Error(`Question ${label} has a duplicate option "${option}".`);
    }
    seen.add(option);
  }
}

function softmax(logits: number[], temperature: number): number[] {
  if (logits.length === 0) return [];
  const max = Math.max(...logits);
  const scaled = logits.map((x) => Math.exp((x - max) / temperature));
  const sum = scaled.reduce((a, b) => a + b, 0);
  return scaled.map((x) => x / sum);
}

function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

function toRecord(keys: readonly string[], values: number[]): Record<string, number> {
  const record: Record<string, number> = {};
  keys.forEach((key, i) => {
    record[key] = values[i];
  });
  return record;
}

/** Same answer decoding as open-jev's `decodeAnswer`. */
function decodeAnswer(question: Question, logits: number[], temperature: number): Answer {
  const probs = softmax(logits, temperature);
  if (question.type === 'choice') {
    const best = argmax(probs);
    return {
      type: 'choice',
      choice: question.options[best],
      confidence: probs[best],
      probabilities: toRecord(question.options, probs),
    };
  }
  if (question.type === 'score') {
    const levels = question.options;
    const expected = probs.reduce((sum, p, i) => sum + p * i, 0);
    const last = levels.length - 1;
    return {
      type: 'score',
      score: expected,
      normalized: last > 0 ? expected / last : 0,
      level: levels[Math.min(last, Math.max(0, Math.round(expected)))],
      confidence: probs[argmax(probs)],
      probabilities: toRecord(levels, probs),
    };
  }
  const pYes = probs[NOUL_LABELS.indexOf('yes')];
  return {
    type: 'noul',
    answer: pYes >= 0.5,
    probability: pYes,
    confidence: Math.max(pYes, 1 - pYes),
  };
}

/** Schema labels + prompt text for one question. */
function promptFor(question: Question): { prompt: string; labels: readonly string[] } {
  if (question.type === 'noul') {
    return { prompt: question.instructions, labels: NOUL_LABELS };
  }
  let prompt = question.instructions;
  if (question.type === 'choice' && question.descriptions) {
    for (const option of question.options) {
      const desc = question.descriptions[option];
      if (desc) prompt += ` ${DESC_TOKEN} ${option}: ${desc}`;
    }
  }
  return { prompt, labels: question.options };
}

function int64Row(values: number[]): Tensor {
  return new Tensor('int64', BigInt64Array.from(values, BigInt), [1, values.length]);
}

let fp16Support: Promise<boolean> | null = null;

function webgpuFp16(): Promise<boolean> {
  fp16Support ??= (async () => {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter?.features.has('shader-f16') ?? false;
    } catch {
      return false;
    }
  })();
  return fp16Support;
}

function isNode(): boolean {
  return typeof (globalThis as { process?: { versions?: { node?: string } } }).process?.versions
    ?.node === 'string';
}

function resolveDevice(device: DecideDevice | undefined): 'webgpu' | 'wasm' | 'cpu' {
  if (device && device !== 'auto') return device;
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) return 'webgpu';
  if (isNode()) return 'cpu';
  return 'wasm';
}

async function resolveDtype(
  dtype: DecideDtype | undefined,
  device: 'webgpu' | 'wasm' | 'cpu',
): Promise<'fp32' | 'fp16' | 'q4' | 'q4f16'> {
  if (dtype && dtype !== 'auto') return dtype;
  if (device === 'webgpu' && (await webgpuFp16())) return 'fp16';
  return 'q4';
}

type MarkerIds = {
  open: number[];
  close: number[];
  p: number[];
  l: number[];
  sepStruct: number[];
  sepText: number[];
};

export class DecideSession {
  readonly repo: string;
  readonly device: 'webgpu' | 'wasm' | 'cpu';
  readonly dtype: 'fp32' | 'fp16' | 'q4' | 'q4f16';
  /** Total subword budget (schema + state), from `gliner2.max_len`. */
  readonly maxLen: number;
  /** Default softmax temperature, from `gliner2.temperature`. */
  readonly temperature: number;
  /** Set when the latest `decide()` had to drop trailing state words. */
  lastTruncated = false;

  private readonly model: DecideModel;
  private readonly tokenizer: PreTrainedTokenizer;
  private readonly markers: MarkerIds;
  private queue: Promise<unknown> = Promise.resolve();
  private disposed = false;

  constructor(
    model: DecideModel,
    tokenizer: PreTrainedTokenizer,
    markers: MarkerIds,
    info: Pick<DecideSession, 'repo' | 'device' | 'dtype' | 'maxLen' | 'temperature'>,
  ) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.markers = markers;
    this.repo = info.repo;
    this.device = info.device;
    this.dtype = info.dtype;
    this.maxLen = info.maxLen;
    this.temperature = info.temperature;
  }

  /** Tokenize one schema element / word exactly as the processor does. */
  private encode(text: string): number[] {
    return this.tokenizer(text, {
      add_special_tokens: false,
      return_tensor: false,
    }).input_ids;
  }

  async decide(
    state: string,
    questions: readonly Question[],
    options: DecideCallOptions = {},
  ): Promise<Answer[]> {
    if (this.disposed) throw new Error('DecideSession has been disposed.');
    if (typeof state !== 'string') {
      throw new Error('decide() expects the state to be a string.');
    }
    if (!Array.isArray(questions) || questions.length === 0) {
      throw new Error('decide() needs at least one question.');
    }
    questions.forEach(validateQuestion);
    const temperature = Math.max(1e-6, options.temperature ?? this.temperature);
    const result = this.enqueue(() =>
      this.run(state, questions, temperature, options.truncation ?? 'cut'),
    );
    this.queue = result.catch(() => undefined);
    return result;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    try {
      await this.queue;
    } catch {
      // A pending decision failed; nothing left to wait for.
    }
    await this.model.dispose?.();
  }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    return this.queue.then(task, task);
  }

  private async run(
    state: string,
    questions: readonly Question[],
    temperature: number,
    truncation: 'cut' | 'error',
  ): Promise<Answer[]> {
    const ids: number[] = [];
    const groups: number[][] = [];

    const append = (text: string): void => {
      ids.push(...this.encode(text));
    };
    const appendIds = (part: number[]): void => {
      ids.push(...part);
    };

    // ( [P] prompt ( [L] label … ) ) blocks, joined by [SEP_STRUCT].
    questions.forEach((question, index) => {
      if (index > 0) appendIds(this.markers.sepStruct);
      const { prompt, labels } = promptFor(question);
      appendIds(this.markers.open);
      appendIds(this.markers.p);
      append(prompt);
      appendIds(this.markers.open);
      const group: number[] = [];
      for (const label of labels) {
        group.push(ids.length);
        appendIds(this.markers.l);
        append(label);
      }
      groups.push(group);
      appendIds(this.markers.close);
      appendIds(this.markers.close);
    });

    appendIds(this.markers.sepText);

    // State words fill the remaining subword budget; cut at the end.
    let budget = this.maxLen - ids.length;
    if (budget < 0) {
      throw new Error(
        `The questions alone need ${ids.length} tokens, exceeding the ${this.maxLen} token context. Ask fewer questions or shorten the instructions/options.`,
      );
    }
    const words = splitWords(normalizeState(state));
    let truncated = false;
    for (const word of words) {
      const wordIds = this.encode(word);
      if (wordIds.length === 0) continue;
      if (wordIds.length > budget) {
        truncated = true;
        break;
      }
      ids.push(...wordIds);
      budget -= wordIds.length;
    }
    if (truncated && truncation === 'error') {
      throw new Error(
        `State was cut to fit the ${this.maxLen} token context. Shorten the state or the questions, or set truncation to "cut".`,
      );
    }
    this.lastTruncated = truncated;

    const markerPositions = groups.flat();
    const { logits } = await this.model({
      input_ids: int64Row(ids),
      attention_mask: int64Row(ids.map(() => 1)),
      marker_positions: int64Row(markerPositions),
    });
    const scores = Array.from(logits.to('float32').data as ArrayLike<number>);

    let offset = 0;
    return questions.map((question, i) => {
      const slice = scores.slice(offset, offset + groups[i].length);
      offset += groups[i].length;
      return decodeAnswer(question, slice, temperature);
    });
  }
}

function encodeMarkerIds(tokenizer: PreTrainedTokenizer): MarkerIds {
  const encode = (text: string): number[] =>
    tokenizer(text, { add_special_tokens: false, return_tensor: false }).input_ids;
  const markers: MarkerIds = {
    open: encode('('),
    close: encode(')'),
    p: encode(P_TOKEN),
    l: encode(L_TOKEN),
    sepStruct: encode(SEP_STRUCT),
    sepText: encode(SEP_TEXT),
  };
  for (const [name, part] of [
    ['[P]', markers.p],
    ['[L]', markers.l],
    ['[SEP_STRUCT]', markers.sepStruct],
    ['[SEP_TEXT]', markers.sepText],
  ] as const) {
    if (part.length !== 1) {
      throw new Error(`Tokenizer does not map ${name} to a single id — not a gliner2 export.`);
    }
  }
  return markers;
}

let instance: DecideSession | null = null;
let instanceKey: string | null = null;

/**
 * Load (or reuse) a GLiNER2.5-Decide ONNX session. Repos without a `gliner2`
 * export shape fail fast with a descriptive error.
 */
export async function loadDecide(
  repo: string = DECIDE_REPO,
  opts: DecideLoadOptions = {},
): Promise<DecideSession> {
  const modelId = DECIDE_REPOS[repo] ?? repo;
  const key = `${modelId}|${opts.dtype ?? 'auto'}|${opts.device ?? 'auto'}`;
  if (instance && instanceKey === key) return instance;
  if (instance) await instance.dispose().catch(() => undefined);

  const config = await AutoConfig.from_pretrained(modelId);
  const gliner2 = (config as unknown as { gliner2?: Gliner2Section }).gliner2;

  const device = resolveDevice(opts.device);
  const dtype = await resolveDtype(opts.dtype, device);

  const onProgress = opts.onProgress;
  let lastProgress = -1;
  const progressCallback = (info: ProgressInfo): void => {
    if (!onProgress || info.status !== 'progress_total') return;
    const progress = Math.round(Math.max(0, Math.min(1, info.progress / 100)) * 100) / 100;
    if (progress === lastProgress) return;
    lastProgress = progress;
    onProgress({ progress, loaded: info.loaded, total: info.total });
  };

  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  const markers = encodeMarkerIds(tokenizer);
  const model = (await AutoModel.from_pretrained(modelId, {
    config,
    dtype,
    device,
    progress_callback: progressCallback,
  })) as unknown as DecideModel;

  const inputs = model.sessions?.model?.inputNames ?? [];
  for (const name of ['input_ids', 'attention_mask', 'marker_positions']) {
    if (!inputs.includes(name)) {
      await model.dispose?.().catch(() => undefined);
      throw new Error(
        `Not a Decide export: ONNX graph is missing the "${name}" input ` +
          `(got: ${inputs.join(', ') || 'none'}).`,
      );
    }
  }

  const session = new DecideSession(model, tokenizer, markers, {
    repo: modelId,
    device,
    dtype,
    maxLen: Math.max(8, Math.floor(gliner2?.max_len ?? 512)),
    temperature: gliner2?.temperature ?? 1,
  });
  instance = session;
  instanceKey = key;
  return session;
}

export async function disposeDecide(): Promise<void> {
  if (instance) await instance.dispose().catch(() => undefined);
  instance = null;
  instanceKey = null;
}
