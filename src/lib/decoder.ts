import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type ProgressInfo,
} from '@huggingface/transformers';
import type { Answer, LoadProgress, Question } from './jev';

/**
 * Driver for decoder-LM roster entries (SmolLM2, Qwen, Llama) — they answer the
 * same typed question packs as the JEV checkpoints, via one greedy generation
 * pass instead of the marker-logit forward pass.
 *
 * Output contract: the model is asked for numbered lines `N: <option>`; each
 * line is fuzzy-matched back to that question's options. Confidence is honest —
 * 1.0 only when the match is unambiguous, lower when it had to be recovered.
 */
export type DecoderDtype = 'fp32' | 'fp16' | 'q4' | 'q4f16' | 'auto';
export type DecoderDevice = 'auto' | 'webgpu' | 'wasm' | 'cpu';

export type DecoderLoadOptions = {
  dtype?: DecoderDtype;
  device?: DecoderDevice;
  onProgress?: (p: LoadProgress) => void;
};

function isNode(): boolean {
  return typeof (globalThis as { process?: { versions?: { node?: string } } }).process?.versions
    ?.node === 'string';
}

function resolveDevice(device: DecoderDevice | undefined): 'webgpu' | 'wasm' | 'cpu' {
  if (device && device !== 'auto') return device;
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) return 'webgpu';
  if (isNode()) return 'cpu';
  return 'wasm';
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

async function resolveDtype(
  dtype: DecoderDtype | undefined,
  device: 'webgpu' | 'wasm' | 'cpu',
): Promise<'fp32' | 'fp16' | 'q4' | 'q4f16'> {
  if (dtype && dtype !== 'auto') return dtype;
  return device === 'webgpu' && (await webgpuFp16()) ? 'q4f16' : 'q4';
}

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

/** Find which option the generated segment picked. -1 when nothing matched. */
function matchOption(segment: string, options: readonly string[]): { idx: number; clean: boolean } {
  const seg = norm(segment);
  if (!seg) return { idx: -1, clean: false };
  let best = -1;
  let hits = 0;
  options.forEach((opt, i) => {
    const n = norm(opt);
    if (n && seg.includes(n)) {
      hits++;
      // prefer the longest match when several options collide
      if (best < 0 || n.length > norm(options[best]).length) best = i;
    }
  });
  return { idx: best, clean: hits === 1 };
}

/** Slice the generated text into one segment per numbered question line. */
function answerSegments(text: string, count: number): string[] {
  const lines = text.split('\n');
  const segs: string[] = [];
  let cur = '';
  for (const line of lines) {
    const m = line.match(/^\s*(?:answer\s*)?(\d{1,2})\s*[:.\-)]\s*(.*)$/i);
    if (m && Number(m[1]) >= 1 && Number(m[1]) <= count) {
      segs.push(cur);
      cur = m[2];
    } else if (segs.length > 0) {
      cur += ' ' + line.trim();
    }
  }
  segs.push(cur);
  // segs[0] is pre-answer chatter; question i reads segs[i]
  return segs.slice(1, 1 + count);
}

function decodeAnswers(questions: readonly Question[], generated: string): Answer[] {
  const segs = answerSegments(generated, questions.length);
  return questions.map((question, i) => {
    const seg = segs[i] ?? '';
    if (question.type === 'noul') {
      const yes = matchOption(seg, ['yes', 'true', 'correct', 'it does', 'there is']);
      const no = matchOption(seg, ['no', 'false', 'not', 'none']);
      let pYes = 0.5;
      if (yes.idx >= 0 && no.idx < 0) pYes = 0.9;
      else if (no.idx >= 0 && yes.idx < 0) pYes = 0.1;
      else if (yes.idx >= 0 && no.idx >= 0) pYes = seg.toLowerCase().indexOf('yes') < seg.toLowerCase().indexOf('no') ? 0.6 : 0.4;
      return {
        type: 'noul',
        answer: pYes >= 0.5,
        probability: pYes,
        confidence: Math.max(pYes, 1 - pYes),
      };
    }
    const { idx, clean } = matchOption(seg, question.options);
    const probabilities: Record<string, number> = {};
    if (idx >= 0) {
      question.options.forEach((opt, j) => {
        probabilities[opt] = j === idx ? (clean ? 1 : 0.6) : clean ? 0 : 0.4 / (question.options.length - 1);
      });
    } else {
      question.options.forEach((opt) => {
        probabilities[opt] = 1 / question.options.length;
      });
    }
    const pick = idx >= 0 ? idx : 0;
    const confidence = idx >= 0 ? (clean ? 1 : 0.6) : 0;
    if (question.type === 'score') {
      const last = question.options.length - 1;
      return {
        type: 'score',
        score: pick,
        normalized: last > 0 ? pick / last : 0,
        level: question.options[pick],
        confidence,
        probabilities,
      };
    }
    return {
      type: 'choice',
      choice: question.options[pick],
      confidence,
      probabilities,
    };
  });
}

function promptFor(state: string, questions: readonly Question[]): { role: string; content: string }[] {
  const list = questions
    .map((q, i) => {
      const kind = q.type === 'noul' ? 'yes/no' : q.type;
      const opts = q.type === 'noul' ? 'yes | no' : q.options.join(' | ');
      return `${i + 1}. (${kind}) ${q.instructions} Options: ${opts}`;
    })
    .join('\n');
  return [
    {
      role: 'system',
      content:
        'You answer questions about a text by picking exactly one option per question. ' +
        'Reply with numbered lines only, like "1: option". No explanations.',
    },
    { role: 'user', content: `Text:\n"""\n${state}\n"""\n\nQuestions:\n${list}` },
  ];
}

export class DecoderSession {
  readonly repo: string;
  readonly device: 'webgpu' | 'wasm' | 'cpu';
  readonly dtype: 'fp32' | 'fp16' | 'q4' | 'q4f16';
  /** Tokens produced by the latest decide() — surfaced for latency display. */
  lastTokens = 0;

  private readonly model: PreTrainedModel;
  private readonly tokenizer: PreTrainedTokenizer;
  private queue: Promise<unknown> = Promise.resolve();
  private disposed = false;

  constructor(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    info: Pick<DecoderSession, 'repo' | 'device' | 'dtype'>,
  ) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.repo = info.repo;
    this.device = info.device;
    this.dtype = info.dtype;
  }

  async decide(
    state: string,
    questions: readonly Question[],
    options: { maxNewTokens?: number; onToken?: (n: number) => void } = {},
  ): Promise<Answer[]> {
    if (this.disposed) throw new Error('DecoderSession has been disposed.');
    const result = this.enqueue(() => this.run(state, questions, options));
    this.queue = result.catch(() => undefined);
    return result;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    try {
      await this.queue;
    } catch {
      // A pending generation failed; nothing left to wait for.
    }
    await this.model.dispose();
  }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    return this.queue.then(task, task);
  }

  private async run(
    state: string,
    questions: readonly Question[],
    options: { maxNewTokens?: number; onToken?: (n: number) => void },
  ): Promise<Answer[]> {
    const messages = promptFor(state, questions);
    const inputs = this.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
    });
    this.lastTokens = 0;
    let generated = '';
    const streamer = new TextStreamer(this.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        generated += text;
        this.lastTokens++;
        options.onToken?.(this.lastTokens);
      },
    });
    await this.model.generate({
      ...inputs,
      max_new_tokens: options.maxNewTokens ?? questions.length * 16 + 32,
      do_sample: false,
      streamer,
      return_dict_in_generate: true,
    });
    return decodeAnswers(questions, generated);
  }
}

let instance: DecoderSession | null = null;
let instanceKey: string | null = null;

/** Load (or reuse) a decoder-LM session for chat-style question answering. */
export async function loadDecoder(
  repo: string,
  opts: DecoderLoadOptions = {},
): Promise<DecoderSession> {
  const key = `${repo}|${opts.dtype ?? 'auto'}|${opts.device ?? 'auto'}`;
  if (instance && instanceKey === key) return instance;
  if (instance) await instance.dispose().catch(() => undefined);

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

  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(repo),
    AutoModelForCausalLM.from_pretrained(repo, {
      dtype,
      device,
      progress_callback: progressCallback,
    }),
  ]);

  const session = new DecoderSession(model, tokenizer, { repo, device, dtype });
  instance = session;
  instanceKey = key;
  return session;
}

export async function disposeDecoder(): Promise<void> {
  if (instance) await instance.dispose().catch(() => undefined);
  instance = null;
  instanceKey = null;
}
