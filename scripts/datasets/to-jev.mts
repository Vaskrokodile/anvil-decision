// Convert {text,label} datasets into Jev-format training records.
//
// Record (one JSON object per line in public/jev-datasets/<name>.jsonl):
//   {
//     "id":        "<dataset>-000123",
//     "split":     "train" | "val",           // stratified 80/20, seed 1337
//     "state":     "<text the decision is about>",
//     "questions": { "<qid>": { "type": "choice"|"score"|"noul",
//                               "instructions": "...",
//                               "criteria": {...}  (choice, noul)  |  [...] (score) } },
//     "answers":   { "<qid>": { "type": "choice", "choice": "<label>",
//                               "probabilities": {label: p, ...}, "confidence": c }
//                    | { "type": "score", "score": s, "legend": [...],
//                        "probabilities": {level: p, ...}, "confidence": c }
//                    | { "type": "noul", "noul": p } }
//   }
// Request shape matches the OpenJEV API (POST /v1/systemone): state + questions.
// Answer targets are label-smoothed (gold gets 0.9, the rest share 0.1), and
// confidence = 1 - H(p)/ln K as the API defines it. Score targets are a
// distribution over ordered levels peaked at the gold level, with score = Σ i·pᵢ.
//
// Run: node scripts/datasets/to-jev.mts
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { stratifiedSplit } from '../../src/lib/split.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const DATA = join(ROOT, 'public', 'datasets');
const OUT = join(ROOT, 'public', 'jev-datasets');
const EPS = 0.1;

type Spec =
  | { kind: 'choice'; qid: string; instructions: string; criteria: Record<string, string>; file: string; map?: Record<string, string>; description: string }
  | { kind: 'score'; qid: string; instructions: string; levels: string[]; file: string; description: string }
  | { kind: 'noul'; qid: string; instructions: string; criteria: { true: string; false: string }; file: string; yes: string[]; description: string };

const SPECS: Record<string, Spec> = {
  'jev-refund-decision': {
    kind: 'choice', qid: 'decision', file: 'jev-refund-decision.jsonl', description: 'Refund requests: approve, deny, or escalate under the written policy.',
    instructions: 'Given the policy and the case facts, what should the support agent do with this refund request?',
    criteria: {
      approve: 'Meets the written policy criteria, evidence is present, amount within front-line authority.',
      deny: 'Clearly outside policy with no discretion: window expired, item excluded, no eligible purchase, or misuse.',
      escalate: 'Policy allows discretion or an exception, amount over authority, fraud or safety signals, or conflicting facts.',
    },
  },
  'jev-contract-risk': {
    kind: 'score', qid: 'risk', file: 'jev-contract-risk.jsonl', description: 'Contract clauses rated by risk to the party the reviewer represents.',
    instructions: 'How much risk does this clause create for the party the reviewer represents, if signed as written?',
    levels: [
      'low: balanced or standard, reasonable caps, normal notice, clear scope',
      'medium: one-sided or ambiguous but fixable, a short unilateral right, vague deliverable, easy-to-miss renewal',
      'high: materially dangerous: uncapped liability or indemnity, all IP assigned, long lock-in, broad non-compete, one-way termination',
    ],
  },
  'jev-claim-support': {
    kind: 'choice', qid: 'support', file: 'jev-claim-support.jsonl', description: 'Claim plus evidence passage: supported, contradicted, or insufficient.',
    instructions: 'Does the evidence passage support the claim?',
    criteria: {
      supported: 'The passage states or directly entails the claim, including values, direction, population and strength.',
      contradicted: 'The passage states something incompatible with the claim: opposite direction, different value, or a scope that rules it out.',
      insufficient: 'On the same topic but does not settle the claim: omits the needed fact, only correlates, or covers a different population or period.',
    },
  },
  'jev-moderation-threads': {
    kind: 'choice', qid: 'severity', file: 'jev-moderation-threads.jsonl', description: 'Comment threads; the target (last) comment rated safe, borderline, or unsafe.',
    instructions: 'Rate the target comment (the last segment) in the context of the thread.',
    criteria: {
      safe: 'Ordinary discussion, disagreement, blunt feedback, or banter that does not target a person.',
      borderline: 'Gray zone a moderator would review: hostile-reading sarcasm, flame-bait, pile-on tone, insults aimed at a behaviour rather than a person.',
      unsafe: 'Clearly violating: direct personal insults or harassment, threats, dehumanising generalisations, scams, or doxxing.',
    },
  },
  'jev-urgency': {
    kind: 'score', qid: 'urgency', file: 'jev-urgency.jsonl', description: 'Long case threads graded by how soon a human must act.',
    instructions: 'How soon does this case need a human to act?',
    levels: ['low: no deadline, general question', 'medium: needs an answer this week', 'high: blocking work or money within days', 'critical: active harm or loss now'],
  },
  'jev-case-routing': {
    kind: 'choice', qid: 'team', file: 'jev-case-routing.jsonl', description: 'Long support threads routed to the owning team.',
    instructions: 'Which team should own this case?',
    criteria: {
      billing: 'Charges, invoices, refunds, payment methods, subscription price.',
      technical: 'Errors, outages, bugs, API failures, integrations, crashes.',
      shipping: 'Parcels, tracking, delivery windows, lost or damaged in transit.',
      account_security: 'Login or 2FA problems, suspicious access, password reset, takeover.',
      sales: 'Pricing for new accounts, quotes, demos, plan upgrades, procurement.',
      trust_safety: 'Abuse reports, fraud rings, policy violations by users or sellers, impersonation.',
    },
  },
  'jev-policy-check': {
    kind: 'choice', qid: 'compliance', file: 'jev-policy-check.jsonl', description: 'Policy plus agent transcript: did the last action comply?',
    instructions: 'Did the agent\'s last action comply with the policy shown?',
    criteria: {
      compliant: 'The action follows the rule; conditions are met and documented.',
      violation: 'The action clearly breaks a rule: cap breached, unverified disclosure, skipped approval, out-of-scope edit.',
      needs_review: 'Policy silent or ambiguous, or facts missing or conflicting; the agent should hold and escalate.',
    },
  },
  'jev-grounding': {
    kind: 'score', qid: 'grounding', file: 'jev-grounding.jsonl', description: 'Sources plus an answer: grounded, partially grounded, or ungrounded.',
    instructions: 'How well is the assistant\'s final answer supported by the source passages?',
    levels: ['ungrounded: contradicts the sources or invents uncovered content', 'partially_grounded: main claim supported, a specific value or condition is not', 'grounded: every material claim appears in or follows from the sources'],
  },
  'general-topic': {
    kind: 'choice', qid: 'topic', file: 'general-topic.jsonl', description: 'News articles by topic.',
    instructions: 'What is the main topic of this news article?',
    criteria: {
      World: 'International politics, conflicts, diplomacy, world events.',
      Sports: 'Sporting events, teams, athletes, competitions.',
      Business: 'Companies, markets, earnings, economy, business deals.',
      'Sci/Tech': 'Science, technology, software, internet, research.',
    },
  },
  'general-emotion': {
    kind: 'choice', qid: 'emotion', file: 'general-emotion.jsonl', description: 'Short English texts by the dominant emotion expressed.',
    instructions: 'Which emotion does the writer express most strongly?',
    criteria: {
      sadness: 'Grief, disappointment, loneliness, feeling down.',
      joy: 'Happiness, delight, pleasure, feeling good.',
      love: 'Affection, caring, romance, warmth toward someone.',
      anger: 'Irritation, resentment, outrage, feeling wronged.',
      fear: 'Worry, anxiety, feeling threatened or unsafe.',
      surprise: 'Being astonished or caught off guard.',
    },
  },
  'general-sentiment': {
    kind: 'noul', qid: 'positive', file: 'general-sentiment.jsonl', description: 'Movie review sentences: is the opinion positive?',
    instructions: 'Does this sentence express a positive opinion about the subject?',
    criteria: { true: 'The sentence expresses a positive opinion.', false: 'The sentence expresses a negative opinion.' },
    yes: ['positive'],
  },
};

const LN = (x: number) => Math.log(x);

function choiceTarget(labels: string[], gold: string) {
  const K = labels.length;
  const p: Record<string, number> = {};
  for (const l of labels) p[l] = l === gold ? 1 - EPS : K > 1 ? EPS / (K - 1) : 0;
  return { probabilities: p, confidence: confOf(Object.values(p)) };
}

function confOf(ps: number[]) {
  const K = ps.length;
  if (K <= 1) return 1;
  let H = 0;
  for (const q of ps) if (q > 0) H -= q * LN(q);
  return Math.max(0, Math.min(1, 1 - H / LN(K)));
}

function scoreTarget(levels: string[], goldIdx: number) {
  const w = levels.map((_, i) => Math.exp(-1.2 * Math.abs(i - goldIdx)));
  const z = w.reduce((a, b) => a + b, 0);
  const ps = w.map((x) => x / z);
  const probabilities: Record<string, number> = {};
  levels.forEach((_, i) => (probabilities[String(i)] = ps[i]));
  const score = ps.reduce((s, q, i) => s + i * q, 0);
  return { probabilities, score, confidence: confOf(ps) };
}

function readRows(file: string) {
  const path = join(DATA, file);
  if (!existsSync(path)) return null;
  return readFileSync(path, 'utf8')
    .replace(/^\uFEFF/, '')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l) as { text: string; label: string });
}

mkdirSync(OUT, { recursive: true });
const index: Record<string, unknown>[] = [];

for (const [name, spec] of Object.entries(SPECS)) {
  const rows = readRows(spec.file);
  if (!rows) {
    console.log(`${name}: source ${spec.file} missing — skipped`);
    continue;
  }

  const labelsAll =
    spec.kind === 'choice'
      ? Object.keys(spec.criteria)
      : spec.kind === 'score'
        ? spec.levels.map((lv) => lv.split(':')[0].trim())
        : ['yes', 'no'];
  const labelIndex = new Map<string, number>();
  if (spec.kind === 'choice') Object.keys(spec.criteria).forEach((l, i) => labelIndex.set(l, i));
  if (spec.kind === 'score') spec.levels.forEach((_, i) => labelIndex.set(String(i), i));

  // map source labels → canonical keys
  const canonical = (label: string): string | null => {
    if (spec.kind === 'score') {
      const idx = spec.levels.findIndex((lv) => lv.split(':')[0].trim() === label);
      return idx >= 0 ? String(idx) : null;
    }
    if (spec.kind === 'choice') return (spec.kind === 'choice' && label in spec.criteria) ? label : null;
    return null;
  };

  const usable = rows
    .map((r) => {
      if (spec.kind === 'noul') return { state: r.text, yes: spec.yes.includes(r.label), label: r.label };
      const c = canonical(r.label);
      return c === null ? null : { state: r.text, label: c };
    })
    .filter((x): x is { state: string; label: string; yes?: boolean } => x !== null);

  // stratified split on canonical labels (noul → yes/no)
  const yArr = Uint32Array.from(usable.map((u) => (spec.kind === 'noul' ? (u.yes ? 1 : 0) : (labelIndex.get(u.label) ?? 0))));
  const { train, val } = stratifiedSplit(yArr, 0.2, 1337);
  const splitOf = new Map<number, 'train' | 'val'>();
  train.forEach((i) => splitOf.set(i, 'train'));
  val.forEach((i) => splitOf.set(i, 'val'));

  const out: string[] = [];
  const counts = { train: 0, val: 0 };
  usable.forEach((u, i) => {
    const questions: Record<string, unknown> = {};
    let answer: Record<string, unknown>;
    if (spec.kind === 'choice') {
      questions[spec.qid] = { type: 'choice', instructions: spec.instructions, criteria: spec.criteria };
      const t = choiceTarget(Object.keys(spec.criteria), u.label);
      answer = { type: 'choice', choice: u.label, probabilities: t.probabilities, confidence: round(t.confidence) };
    } else if (spec.kind === 'score') {
      questions[spec.qid] = { type: 'score', instructions: spec.instructions, criteria: spec.levels };
      const t = scoreTarget(spec.levels, Number(u.label));
      answer = { type: 'score', score: round(t.score), legend: spec.levels, probabilities: roundObj(t.probabilities), confidence: round(t.confidence) };
    } else {
      questions[spec.qid] = { type: 'noul', instructions: spec.instructions, criteria: spec.criteria };
      const p = u.yes ? 1 - EPS : EPS;
      answer = { type: 'noul', noul: round(p) };
    }
    const split = splitOf.get(i) ?? 'train';
    counts[split]++;
    const rec = {
      id: `${name}-${String(i).padStart(6, '0')}`,
      split,
      state: u.state,
      questions,
      answers: { [spec.qid]: answer },
    };
    out.push(JSON.stringify(rec));
  });

  writeFileSync(join(OUT, `${name}.jsonl`), out.join('\n') + '\n', 'utf8');
  index.push({
    file: `${name}.jsonl`,
    name,
    description: spec.description,
    question: { qid: spec.qid, type: spec.kind },
    records: out.length,
    train: counts.train,
    val: counts.val,
    labels: spec.kind === 'noul' ? ['yes', 'no'] : labelsAll,
    source: spec.file,
  });
  console.log(`${name}: ${out.length} records (train ${counts.train} · val ${counts.val}) ${spec.kind}`);
}

function round(x: number) {
  return Math.round(x * 1e6) / 1e6;
}
function roundObj(o: Record<string, number>) {
  return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, round(v)]));
}

writeFileSync(
  join(OUT, 'index.json'),
  JSON.stringify(
    {
      format: 'OpenJEV-shaped training records: {id, split, state, questions, answers}. Request = state + questions; target = answers. Choice answers carry probabilities+confidence; score answers carry score+legend+probabilities+confidence; noul answers carry a yes-probability. Targets are label-smoothed (gold 0.9). Confidence = 1 - H(p)/ln K.',
      generatedBy: 'scripts/datasets/to-jev.mts',
      seed: 1337,
      split: '80/20 stratified',
      datasets: index,
    },
    null,
    2,
  ) + '\n',
  'utf8',
);
console.log('index.json written');
