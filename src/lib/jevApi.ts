/**
 * OpenJEV decision API client — shared by the Node dataset teacher and the
 * in-browser JevBench runner. Plain fetch, no SDK. The key is passed in by the
 * caller; this module never stores or logs it.
 */

export const JEV_ENDPOINT = 'https://api.openjev.sh/v1/systemone';

export interface ChoiceQ {
  type: 'choice';
  instructions: string;
  criteria: Record<string, string | null>;
}
export interface NoulQ {
  type: 'noul';
  instructions: string;
  criteria?: { true?: string; false?: string };
}
export interface ScoreQ {
  type: 'score';
  instructions: string;
  criteria: string[];
}
export type JevQuestion = ChoiceQ | NoulQ | ScoreQ;

export interface JevAnswer {
  type: 'choice' | 'noul' | 'score';
  choice?: string;
  probabilities?: Record<string, number>;
  confidence?: number;
  noul?: number;
  score?: number;
  legend?: unknown;
}

export class JevError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function jevDecide(
  apiKey: string,
  state: string | object,
  questions: Record<string, JevQuestion>,
  opts: { signal?: AbortSignal; retries?: number } = {},
): Promise<{ answers: Record<string, JevAnswer>; ms: number }> {
  const t0 = performance.now();
  const retries = opts.retries ?? 4;
  for (let attempt = 0; ; attempt++) {
    let r: Response;
    try {
      r = await fetch(JEV_ENDPOINT, {
        method: 'POST',
        headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'openjev', state, questions }),
        signal: opts.signal,
      });
    } catch (e) {
      if (attempt < retries) {
        await sleep(800 * 2 ** attempt);
        continue;
      }
      throw new JevError(`Network error: ${(e as Error).message}`, 0);
    }
    if (r.ok) {
      const j = (await r.json()) as { answers: Record<string, JevAnswer> };
      return { answers: j.answers, ms: performance.now() - t0 };
    }
    if (r.status === 401) throw new JevError('Invalid or missing API key (401).', 401);
    if (r.status === 422) throw new JevError('Request rejected (422) — check the question shape.', 422);
    if ((r.status === 429 || r.status >= 500) && attempt < retries) {
      const ra = Number(r.headers.get('Retry-After'));
      await sleep(ra > 0 ? ra * 1000 : 1000 * 2 ** attempt);
      continue;
    }
    throw new JevError(`Jev API returned ${r.status}.`, r.status);
  }
}

/** Probability the model puts on the option `label` for a choice answer. */
export function choiceProb(a: JevAnswer | undefined, label: string): number {
  return a?.probabilities?.[label] ?? 0;
}
