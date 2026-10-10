import { toJevQuestion, judge, summarize, sampleItems, chanceOf, type BenchItem, type ItemResult } from '../src/bench/jevbench.ts';

let fails = 0;
const eq = (name: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${ok ? '' : `  got=${JSON.stringify(got)} want=${JSON.stringify(want)}`}`);
};

const choice: BenchItem = {
  id: 'c1', family: 'intent', expected: 'refund', labels: ['cancel', 'refund', 'status'], state: 's',
  question: { type: 'choice', instructions: 'Which?', criteria: { cancel: 'a', refund: 'b', status: 'c' } },
};
const noul: BenchItem = {
  id: 'n1', family: 'fact', expected: 'yes', labels: ['no', 'yes'], state: 's',
  question: { type: 'noul', instructions: 'True?', criteria: { true: 'yes', false: 'no' } },
};
const score: BenchItem = {
  id: 's1', family: 'ord', expected: 2, labels: ['0', '1', '2', '3'], state: 's',
  question: { type: 'score', instructions: 'Rate', criteria: ['none', 'minor', 'major', 'severe'] },
};

eq('toJevQuestion choice keeps object criteria', toJevQuestion(choice).criteria, { cancel: 'a', refund: 'b', status: 'c' });
eq('toJevQuestion score keeps array criteria', toJevQuestion(score).criteria, ['none', 'minor', 'major', 'severe']);
eq('toJevQuestion noul type', toJevQuestion(noul).type, 'noul');

const jc = judge(choice, { type: 'choice', choice: 'refund', probabilities: { cancel: 0.1, refund: 0.8, status: 0.1 }, confidence: 0.7 });
eq('choice correct', [jc.correct, jc.predicted, Number(jc.pGold.toFixed(2))], [true, 'refund', 0.8]);
const jw = judge(choice, { type: 'choice', choice: 'cancel', probabilities: { cancel: 0.6, refund: 0.3, status: 0.1 }, confidence: 0.5 });
eq('choice wrong', [jw.correct, jw.predicted], [false, 'cancel']);

const jn = judge(noul, { type: 'noul', noul: 0.9 });
eq('noul yes correct, conf 0.8', [jn.correct, Number(jn.conf.toFixed(2))], [true, 0.8]);
const jnn = judge(noul, { type: 'noul', noul: 0.4 });
eq('noul predicts no', [jnn.predicted, jnn.correct], ['no', false]);

const js1 = judge(score, { type: 'score', score: 2.1, probabilities: { '0': 0, '1': 0.1, '2': 0.7, '3': 0.2 }, confidence: 0.6 });
eq('score index keys correct', [js1.correct, js1.predicted], [true, '2']);
const js2 = judge(score, { type: 'score', score: 2.1, probabilities: { none: 0, minor: 0.1, major: 0.7, severe: 0.2 } });
eq('score legend keys map to index', [js2.predicted, js2.correct], ['2', true]);

eq('chance noul', chanceOf(noul), 0.5);
eq('chance choice', Number(chanceOf(choice).toFixed(4)), Number((1 / 3).toFixed(4)));

// summarize: easy tier 3-way choice, all correct at 0.9 conf -> acc 1, intelligence 100, ECE ~0.1
const items = new Map([[choice.id, choice], [noul.id, noul], [score.id, score]]);
const mk = (id: string, tier: 'easy' | 'hard', correct: boolean, conf: number): ItemResult => ({
  id, tier, correct, pGold: correct ? conf : 0, conf, predicted: null, expected: 'x', ms: 100,
});
const sum = summarize([mk('c1', 'easy', true, 0.9), mk('c1', 'easy', true, 0.9), mk('c1', 'hard', false, 0.9)], items);
eq('accuracy 2/3', Number(sum.accuracy.toFixed(3)), 0.667);
eq('easy intelligence 100 (acc 1 above chance)', Math.round(sum.perTier.easy.intelligence), 100);
eq('hard intelligence clipped at 0', sum.perTier.hard.intelligence, 0);
eq('ECE = |acc 2/3 - conf 0.9|', Number(sum.ece.toFixed(3)), 0.233);
eq('error rows excluded', summarize([{ ...mk('c1', 'easy', false, 0), error: 'x' }], items).accuracy, 0);

// sampling: deterministic and bounded
const arr = Array.from({ length: 50 }, (_, i) => i);
eq('sample deterministic', sampleItems(arr, 10), sampleItems(arr, 10));
eq('sample size', sampleItems(arr, 10).length, 10);
eq('sample all when n>=len', sampleItems(arr, 99).length, 50);

console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);

