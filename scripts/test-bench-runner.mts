import { runItems, summarize, type BenchItem } from '../src/bench/jevbench.ts';

let fails = 0;
const check = (name: string, ok: boolean, extra = '') => {
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? '  ' + extra : ''}`);
};

const items: BenchItem[] = [
  { id: 'a', family: 'f', expected: 'refund', labels: ['cancel', 'refund'], state: 'charged twice', question: { type: 'choice', instructions: 'Which?', criteria: { cancel: 'x', refund: 'y' } } },
  { id: 'b', family: 'f', expected: 'yes', labels: ['no', 'yes'], state: 'it is so', question: { type: 'noul', instructions: 'True?', criteria: { true: 'yes', false: 'no' } } },
  { id: 'c', family: 'f', expected: 2, labels: ['0', '1', '2', '3'], state: 'minor bug', question: { type: 'score', instructions: 'Rate', criteria: ['none', 'minor', 'major', 'severe'] } },
  { id: 'd', family: 'f', expected: 'cancel', labels: ['cancel', 'refund'], state: 'boom', question: { type: 'choice', instructions: 'Which?', criteria: { cancel: 'x', refund: 'y' } } },
];

const realFetch = globalThis.fetch;
let calls: { body: any; auth: string | null }[] = [];

function mockFetch(answerFor: (body: any) => unknown, status = 200) {
  globalThis.fetch = (async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body));
    calls.push({ body, auth: new Headers(init.headers).get('Authorization') });
    if (status !== 200) return new Response(JSON.stringify({ error: 'x' }), { status });
    return new Response(JSON.stringify({ model: 'openjev', answers: answerFor(body), usage: {} }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  }) as typeof fetch;
}

// 1) happy path: sends the right question shapes and judges each answer
calls = [];
mockFetch((body) => {
  const q = body.questions.q;
  if (q.type === 'choice') {
    const probs = Object.keys(q.criteria).reduce((o: any, k) => ((o[k] = k === 'refund' ? 0.9 : 0.1), o), {});
    return { q: { type: 'choice', choice: probs.refund > probs.cancel ? 'refund' : 'cancel', probabilities: probs, confidence: 0.8 } };
  }
  if (q.type === 'noul') return { q: { type: 'noul', noul: 0.95 } };
  return { q: { type: 'score', score: 2, probabilities: { '0': 0, '1': 0.1, '2': 0.8, '3': 0.1 }, confidence: 0.7 } };
});
const seen: string[] = [];
const res = await runItems('KEY123', items, 'easy', (r) => seen.push(r.id), new AbortController().signal, 2);
check('all items ran', res.length === 4, `got ${res.length}`);
check('auth header is Bearer key', calls.every((c) => c.auth === 'Bearer KEY123'));
check('one question per request, model openjev', calls.every((c) => c.body.model === 'openjev' && Object.keys(c.body.questions).length === 1));
check('score criteria sent as array', calls.some((c) => c.body.questions.q.type === 'score' && Array.isArray(c.body.questions.q.criteria)));
const byId = Object.fromEntries(res.map((r) => [r.id, r]));
check('choice a correct', byId.a.correct === true && byId.a.predicted === 'refund');
check('choice d wrong (model says refund, gold cancel)', byId.d.correct === false && byId.d.predicted === 'refund');
check('noul b correct (0.95 => yes)', byId.b.correct === true && Math.abs(byId.b.conf - 0.9) < 1e-9);
check('score c correct via index keys', byId.c.correct === true && byId.c.predicted === '2');
const s = summarize(res, new Map(items.map((i) => [i.id, i])));
check('accuracy 3/4', Math.abs(s.accuracy - 0.75) < 1e-9, `acc=${s.accuracy}`);

// 2) invalid key: whole run aborts with JevError 401
mockFetch(() => ({}), 401);
let aborted = false;
try {
  await runItems('BAD', items, 'easy', () => {}, new AbortController().signal, 2);
} catch (e) {
  aborted = (e as { status?: number }).status === 401;
}
check('401 aborts run with status 401', aborted);

// 3) server error on one item (422) is recorded, run continues
let n = 0;
globalThis.fetch = (async () => {
  n++;
  if (n === 2) return new Response('{"error":"bad"}', { status: 422 });
  return new Response(JSON.stringify({ answers: { q: { type: 'noul', noul: 0.2 } } }), { status: 200 });
}) as typeof fetch;
const partial = await runItems('K', items, 'easy', () => {}, new AbortController().signal, 1);
check('one error recorded, others kept', partial.filter((r) => r.error).length === 1 && partial.length === 4);

// 4) abort signal stops early
const ac = new AbortController();
mockFetch(() => ({ q: { type: 'noul', noul: 0.5 } }));
const got = await runItems('K', items, 'easy', (_r, done) => { if (done === 1) ac.abort(); }, ac.signal, 1);
check('abort stops remaining items', got.length < items.length, `got ${got.length}`);

globalThis.fetch = realFetch;
console.log(fails ? `\n${fails} FAILED` : '\nall e2e checks passed');
process.exit(fails ? 1 : 0);

