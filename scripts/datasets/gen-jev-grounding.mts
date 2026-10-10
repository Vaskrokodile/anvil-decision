// Deterministic generator for public/datasets/jev-grounding.jsonl (2,500 rows).
//
// Each row is one `text` line: a session header, a question, 2-3 source passages
// (short numbered excerpts), earlier chit-chat turns, and the assistant's FINAL
// answer. The gold label says how well that final answer is supported by the sources.
//   grounded             every material claim is in, or follows directly from, the sources.
//   partially_grounded   the main claim is supported, but a specific (a number, a date,
//                        a condition, a quantifier) is altered or added without support.
//   ungrounded           the main claim contradicts the sources, or is not in them at all.
//
// Construction: each domain provides a correct claim, a claim with an altered value,
// an unsupported condition sentence, contradicting sentences and invented sentences.
// Sources always carry the core fact plus other facts; the answer is assembled from
// the variant for the row's label. Sentences used as "unsupported" in one row can
// appear as true source material in another row, so no phrasing is tied to one class.
//
// Run: node scripts/datasets/gen-jev-grounding.mts
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

type Label = 'grounded' | 'partially_grounded' | 'ungrounded';
type Facts = Record<string, string>;

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT = join(ROOT, 'public', 'datasets', 'jev-grounding.jsonl');
const SEED = 0x6f9d21;
const PER_LABEL: Record<Label, number> = { grounded: 834, partially_grounded: 833, ungrounded: 833 };

// ---------------------------------------------------------------- rng / utils

class Rng {
  s: number;
  constructor(seed: number) {
    this.s = seed >>> 0;
  }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  int(lo: number, hi: number): number {
    return lo + Math.floor(this.next() * (hi - lo + 1));
  }
  pick<T>(a: readonly T[]): T {
    return a[Math.floor(this.next() * a.length)];
  }
  chance(p: number): boolean {
    return this.next() < p;
  }
  shuffle<T>(a: T[]): T[] {
    for (let i = a.length - 1; i > 0; i--) {
      const j = this.int(0, i);
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  digits(n: number): string {
    let s = '';
    for (let i = 0; i < n; i++) s += String(this.int(0, 9));
    return s;
  }
}

const pad = (n: number): string => String(n).padStart(2, '0');
const money = (n: number): string => `$${String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}`;

// ---------------------------------------------------------------- shared noise

// Earlier-turn chit-chat (label-neutral). Each pair is [user line, assistant line].
const CHAT: [string, string][] = [
  ['User: Thanks, that is a helpful start.', 'Assistant: Glad it helps. Ask away if anything is unclear.'],
  ['User: Can you keep answers short?', 'Assistant: Yes, I will keep them brief.'],
  ['User: My reading glasses are on the desk somewhere.', 'Assistant: Hope you find them soon.'],
  ['User: I am reading this on my phone during the commute.', 'Assistant: Then I will keep the formatting simple.'],
  ['User: Do you know if the printer on floor two works?', 'Assistant: I can only answer from the excerpts, and they do not cover printers.'],
  ['User: Thanks for the quick reply earlier.', 'Assistant: You are welcome.'],
  ['User: Our team meeting moved to Thursday.', 'Assistant: Noted. I will keep that in mind for the schedule.'],
  ['User: I have a cold, so my typing is slow today.', 'Assistant: Take your time; there is no rush.'],
  ['User: Please also remind me to call the desk later.', 'Assistant: I cannot set reminders, but you can note it on your list.'],
  ['User: Is the lift working on the east side?', 'Assistant: The excerpts do not mention the lift; the facilities desk can confirm that.'],
  ['User: Okay, moving on to something else.', 'Assistant: Sure, what would you like to look at next?'],
  ['User: The coffee machine on our floor is broken again.', 'Assistant: Sorry to hear that. I can help with the documents, not the machine.'],
  ['User: Do you ever get tired of answering questions?', 'Assistant: I do not get tired, so keep them coming.'],
  ['User: Can you read the excerpt title out for me?', 'Assistant: Sure, the title is shown in the source label above the passage.'],
  ['User: I am on a short break, so I will read this later.', 'Assistant: The answer will still be here when you come back.'],
  ['User: Did you get the attachment I sent yesterday?', 'Assistant: I only see the text in this session, not earlier attachments.'],
  ['User: Can you suggest a title for my report?', 'Assistant: How about "Notes on the current excerpts"?'],
  ['User: My manager likes bullet points, just so you know.', 'Assistant: Noted; I will use bullets next time.'],
  ['User: Our office moved to the third floor last month.', 'Assistant: Congratulations on the move.'],
  ['User: Quick question, how do I reset my password?', 'Assistant: That is outside these excerpts, so I cannot say for certain.'],
  ['User: Thanks, I can see the excerpt labels now.', 'Assistant: Good, the labels show which source each line came from.'],
  ['User: Sorry for the typos in my last message.', 'Assistant: No problem at all.'],
  ['User: Is it raining where you are?', 'Assistant: I do not have a location, so I cannot tell.'],
  ['User: I just finished my lunch, so I am back now.', 'Assistant: Welcome back. Where were we?'],
  ['User: That is all for the moment, thanks.', 'Assistant: You are welcome. Come back any time.'],
];

// Non-factual lines that can sit at the end of a source passage.
const SOURCE_NOISE: string[] = [
  'Last reviewed 2025-11-02 by the content desk.',
  'Doc ID HB-2231, page 3 of 9.',
  'This excerpt is provided for reference only.',
  'Printed copies are uncontrolled.',
  'Contact desk: 555-0134 during weekday hours.',
  'Layout note: headings are omitted in this extract.',
  'Translation notes appear in the appendix.',
  'Excerpt generated by the document portal.',
  'Revision history is kept by the document owner.',
  'Some tables were flattened for readability.',
  'The portal shows the newest revision first.',
  'Index terms: guidance, summary, reference.',
  'Feedback on this page goes to the content desk inbox.',
  'Page footers were removed from this extract.',
  'Section numbers are kept as printed in the source.',
  'The excerpt may omit notes that appear elsewhere in the document.',
];

const LEADS = ['', '', '', 'Short answer: ', 'Per the sources: ', 'Based on the passages: ', 'From the documents: ', 'In short: ', 'Answer: ', 'According to the excerpts: '];

// ---------------------------------------------------------------- domains

interface Dom {
  titles: string[];
  make: (r: Rng) => Facts; // must set: val (correct), alt (wrong, never in sources)
  question: (f: Facts) => string;
  core: (f: Facts, v: string) => string;
  claims: ((f: Facts, v: string) => string)[];
  others: (f: Facts) => string[]; // 3 true source sentences
  conds: (f: Facts) => string[]; // unsupported extras (partial)
  contras: (f: Facts) => string[]; // contradict the core (ungrounded)
  invents: (f: Facts) => string[]; // topics not in the sources (ungrounded)
}

const SHIPPING: Dom = {
  titles: ['Shipping FAQ', 'Delivery zones sheet', 'Dispatch notes'],
  make(r) {
    const z = r.pick(['B', 'C', 'D', 'E']);
    const lo = r.int(1, 4);
    const hi = lo + r.int(1, 3);
    return { z, val: `${lo} to ${hi} business days`, alt: `${lo + 4} to ${hi + 4} business days`, thr: money(r.pick([60, 75, 90])) };
  },
  question: (f) => `How long does standard delivery to Zone ${f.z} take?`,
  core: (f, v) => `Standard delivery to Zone ${f.z} takes ${v} after the order ships.`,
  claims: [
    (f, v) => `Standard delivery to Zone ${f.z} takes ${v} once the order has shipped.`,
    (f, v) => `Customers in Zone ${f.z} should expect ${v} for standard delivery after dispatch.`,
    (f, v) => `Zone ${f.z} deliveries take ${v} from the day the order leaves the depot.`,
    (f, v) => `Standard shipping to Zone ${f.z} is ${v} after dispatch.`,
  ],
  others: (f) => [
    'Orders ship from the Harrowgate depot on weekdays only.',
    'Tracking links are emailed within 12 hours of dispatch.',
    `Customs fees are not included in the Zone ${f.z} price.`,
  ],
  conds: (f) => [`Shipping is free on orders over ${f.thr}.`, 'Gift wrapping is available for an extra fee.'],
  contras: (f) => [`Standard delivery to Zone ${f.z} is same-day for every order.`, `Zone ${f.z} does not receive standard delivery at all.`, `Zone ${f.z} deliveries arrive the same afternoon the order is placed.`],
  invents: (f) => [`Express couriers guarantee arrival within 24 hours in Zone ${f.z}.`, 'Evening collection from a locker in Port Veyra is also available.'],
};

const RETURNS: Dom = {
  titles: ['Returns policy', 'Refund timing', 'Store help article'],
  make(r) {
    const days = r.pick([14, 30, 45, 60]);
    const alt = days === 14 ? 30 : days === 30 ? 45 : days === 45 ? 60 : 30;
    return { val: `${days} days`, alt: `${alt} days` };
  },
  question: () => 'What is the return window for unopened items?',
  core: (f, v) => `Unopened items may be returned within ${v} for a refund to the original payment method.`,
  claims: [
    (f, v) => `Unopened items can be returned within ${v}, with the refund going to the original payment method.`,
    (f, v) => `The return window for unopened goods is ${v}, refunded to the card used at checkout.`,
    (f, v) => `Unopened goods qualify for return within ${v} of delivery.`,
    (f, v) => `You have ${v} to return unopened items for a refund to the original payment method.`,
  ],
  others: () => [
    'Opened software and gift cards are excluded from returns.',
    'Return shipping labels are emailed once the return request is approved.',
    'Refunds usually post within 5 business days of inspection.',
  ],
  conds: () => ['A receipt is required for every return.', 'Returns must be packed in the original box.'],
  contras: () => ['Unopened items cannot be returned; only store credit is issued.', 'Returns are refunded only as store credit, never to the original card.', 'Unopened items are never refunded; they can only be exchanged for the same product.'],
  invents: () => ['Returns can be dropped off at any partner pharmacy in the city.', 'A restocking fee of 15 percent applies to every return.'],
};

const WARRANTY: Dom = {
  titles: ['Warranty terms', 'Product care sheet', 'Support FAQ'],
  make(r) {
    const prod = r.pick(['Tidewell Pro 3 kettle', 'Marrow Lane desk lamp', 'Quillon 40 blender', 'Brisk Harbour fan']);
    const m = r.pick([12, 18, 24, 36]);
    const alt = m === 12 ? 24 : m === 18 ? 36 : m === 24 ? 12 : 18;
    return { prod, val: `${m} months`, alt: `${alt} months` };
  },
  question: (f) => `How long is the warranty on the ${f.prod}?`,
  core: (f, v) => `The ${f.prod} is covered for ${v} against manufacturing defects.`,
  claims: [
    (f, v) => `The ${f.prod} has ${v} of cover for manufacturing defects.`,
    (f, v) => `Manufacturing defects on the ${f.prod} are covered for ${v}.`,
    (f, v) => `Defects from manufacture are covered for ${v} on the ${f.prod}.`,
    (f, v) => `The ${f.prod} warranty period for manufacturing faults is ${v}.`,
  ],
  others: () => [
    'Batteries and filters are consumables and are not covered.',
    'Claims can be started on the portal, which speeds up review.',
    'Claims are reviewed within five working days.',
  ],
  conds: () => ['The warranty also covers accidental drops and liquid damage.', 'Registration within 30 days is required to keep the warranty active.'],
  contras: (f) => [`The ${f.prod} has no warranty against manufacturing defects.`, `Manufacturing defects are excluded from the ${f.prod} warranty.`, `The ${f.prod} is not covered for manufacturing defects at any time.`],
  invents: () => ['Repairs are handled by a partner workshop in Port Veyra.', 'Replacement units ship within 48 hours of approval.'],
};

const CLINIC: Dom = {
  titles: ['Clinic hours', 'Patient information', 'Visiting guide'],
  make(r) {
    const clinic = r.pick(['Maple Row Clinic', 'Fernwick Family Practice', 'Northgate Dental Centre', 'Quayside Health Hub']);
    const days = r.pick(['Monday to Friday', 'Tuesday to Saturday']);
    const t = r.pick([
      ['08:00', '16:30'],
      ['07:30', '15:00'],
      ['09:00', '17:30'],
    ]);
    const alt = t[0] === '08:00' ? '09:00-17:30' : t[0] === '07:30' ? '08:30-16:00' : '07:30-15:00';
    return { clinic, days, val: `${t[0]}-${t[1]}`, alt };
  },
  question: (f) => `When is ${f.clinic} open?`,
  core: (f, v) => `${f.clinic} is open ${f.days}, from ${v}.`,
  claims: [
    (f, v) => `${f.clinic} is open ${f.days} from ${v}.`,
    (f, v) => `You can visit ${f.clinic} ${f.days}, between ${v}.`,
    (f, v) => `${f.clinic} welcomes patients ${f.days}, during ${v}.`,
    (f, v) => `The opening hours of ${f.clinic} are ${v} on ${f.days}.`,
  ],
  others: () => [
    'Appointments can be booked online up to 60 days in advance.',
    'The clinic is closed on public holidays.',
    'The clinic has a step-free entrance on the east side.',
  ],
  conds: () => ['Walk-ins are welcome at any time during those hours.', 'Saturday mornings are extended for vaccines only.'],
  contras: (f) => [`${f.clinic} does not see patients on any weekday.`, `${f.clinic} has no regular opening hours and runs by appointment only.`, `${f.clinic} is closed to all patients every week.`],
  invents: (f) => ['Free parking is available in the north lot for patients.', `${f.clinic} also runs a pharmacy counter on the ground floor.`],
};

const LIBRARY: Dom = {
  titles: ['Borrowing rules', 'Library FAQ', 'Membership guide'],
  make(r) {
    const w = r.pick([2, 3, 4]);
    const alt = w === 2 ? 4 : w === 3 ? 6 : 2;
    return { val: `${w} weeks`, alt: `${alt} weeks` };
  },
  question: () => 'How long can a member keep a borrowed book?',
  core: (f, v) => `Books can be borrowed for ${v} and renewed once if no one has reserved them.`,
  claims: [
    (f, v) => `A member can keep a book for ${v}.`,
    (f, v) => `Loans run for ${v}, and a book can be renewed once when nobody is waiting for it.`,
    (f, v) => `Borrowers have ${v} with a book, and one renewal is allowed if it is not reserved.`,
    (f, v) => `A loan lasts ${v}; renewal is possible once, provided no one has reserved the title.`,
  ],
  others: () => [
    'Reference books and journals stay in the building.',
    'Reserved items are held for three days after they are returned.',
    'Members can manage loans through the library app.',
  ],
  conds: () => ['Late returns cost 25 cents per day.', 'Overdue books must be paid for before any new loan.'],
  contras: () => ['Books cannot be taken home; every title is reference-only.', 'Members may keep books indefinitely with no due date.', 'Loans last one day only, and renewal is never allowed.'],
  invents: () => ['Audiobooks can be downloaded free of charge for anyone in the county.', 'Interlibrary requests usually take about two months to arrive.'],
};

const LICENSE: Dom = {
  titles: ['Licensing terms', 'Plan comparison', 'Admin handbook'],
  make(r) {
    const seats = r.pick([5, 10, 25]);
    const alt = seats === 5 ? 10 : seats === 10 ? 25 : 5;
    return { val: `${seats} seats`, alt: `${alt} seats` };
  },
  question: () => 'How many seats does a Team license cover?',
  core: (f, v) => `Each Team license covers ${v} for named users and renews every twelve months.`,
  claims: [
    (f, v) => `A Team license covers ${v} for named users.`,
    (f, v) => `The Team plan includes ${v}, assigned to named users.`,
    (f, v) => `A Team plan has ${v} available to named users.`,
    (f, v) => `Each Team license is sized at ${v} for named users.`,
  ],
  others: () => [
    'Business licenses include single sign-on and audit logs.',
    'Seats can be reassigned once every 30 days.',
    "Invoices are issued in the account's billing currency.",
  ],
  conds: () => ['Unused seats roll over for twelve months.', 'Seat counts can be increased mid-term at no charge.'],
  contras: () => ['Team licenses are sold per device with no seat limit.', 'The Team plan has no seat count and is priced per gigabyte.', 'Team licenses cover unlimited users and seats.'],
  invents: () => ['Enterprise customers receive a dedicated support engineer.', 'Annual payment includes a 20 percent discount on add-ons.'],
};

const DEPOSIT: Dom = {
  titles: ['Tenancy guide', 'Move-out checklist', 'Deposit rules'],
  make(r) {
    const d = r.pick([14, 21, 28]);
    const alt = d === 14 ? 35 : d === 21 ? 42 : 45;
    return { val: `${d} days`, alt: `${alt} days` };
  },
  question: () => 'When is the deposit returned after move-out?',
  core: (f, v) => `The deposit is returned within ${v} after move-out, less documented damage.`,
  claims: [
    (f, v) => `The deposit comes back within ${v} of move-out, minus any documented damage.`,
    (f, v) => `Tenants get the deposit back within ${v} after leaving, less recorded damage.`,
    (f, v) => `The landlord returns the deposit within ${v} after move-out, less recorded damage.`,
    (f, v) => `After move-out, the deposit is refunded within ${v}, minus documented damage.`,
  ],
  others: () => [
    'The deposit equals two weeks of rent.',
    'Keys must be returned to the building office on the last day.',
    'Inspection photos are taken at check-in and check-out.',
  ],
  conds: () => ['Deductions are capped at half of the deposit.', 'A cleaning certificate is needed before any refund is paid.'],
  contras: () => ['Deposits are non-refundable once a lease is signed.', 'The deposit is only returned if the tenant renews the lease.', 'The deposit is kept in full at the end of every tenancy.'],
  invents: () => ['Pets are permitted on the third floor with an extra deposit.', 'Interest on the deposit is paid annually at 2 percent.'],
};

const ENROLMENT: Dom = {
  titles: ['Admissions calendar', 'Enrolment guide', 'Pupil services'],
  make(r) {
    const val = `${r.int(3, 27)} ${r.pick(['March', 'April', 'May'])}`;
    const alt = `${r.int(1, 28)} ${r.pick(['January', 'October'])}`;
    return { val, alt };
  },
  question: () => 'When do enrolment applications close?',
  core: (f, v) => `Enrolment applications close on ${v}.`,
  claims: [
    (f, v) => `The enrolment deadline falls on ${v}.`,
    (f, v) => `Applications for enrolment must be in by ${v}.`,
    (f, v) => `The last day to submit an enrolment application is ${v}.`,
    (f, v) => `Enrolment applications are due by ${v}.`,
  ],
  others: () => [
    'Applications must include a birth certificate and a proof of address.',
    'Offers are sent by email within two weeks of the closing date.',
    'Siblings are considered in the same round as other applicants.',
  ],
  conds: () => ['Late applications are accepted with a fee of $40.', 'An interview is required for every applicant.'],
  contras: () => ['Enrolment applications have no closing date and are accepted all year.', 'Applications close on the first day of term, so none are accepted after it starts.', 'Enrolment applications are reviewed on a first-come basis with no deadline.'],
  invents: () => ['Siblings receive a ten percent fee discount.', 'Uniforms are provided free to every enrolled pupil.'],
};

const PER_DIEM: Dom = {
  titles: ['Travel policy', 'Expense rules', 'Trip handbook'],
  make(r) {
    const m = r.pick([45, 55, 65]);
    return { val: `${money(m)} per day`, alt: `${money(m + 15)} per day` };
  },
  question: () => 'What is the daily meal limit on approved trips?',
  core: (f, v) => `Meals are reimbursed up to ${v} on approved trips.`,
  claims: [
    (f, v) => `Approved trips reimburse meals up to ${v}.`,
    (f, v) => `The daily meal allowance on approved trips is ${v}.`,
    (f, v) => `On an approved trip, meals are covered up to ${v} each day.`,
    (f, v) => `Employees may claim meals on approved trips up to ${v} per day.`,
  ],
  others: () => [
    'Receipts are required for any single expense over $25.',
    'Hotel rates above the city cap need a written exception.',
    'Mileage is reimbursed at the published monthly rate.',
  ],
  conds: () => ['Alcohol purchases are reimbursable within the meal allowance.', 'Tips above 20 percent are reimbursed in full.'],
  contras: () => ['Meals are not reimbursable on any approved trip.', 'Meal costs are never reimbursed; only lodging is covered.', 'Meal allowances are removed for every approved trip.'],
  invents: () => ['Rideshare costs above $100 need a vice-president signature.', 'Employees may claim a fixed car allowance each quarter.'],
};

const DOMAINS: Dom[] = [SHIPPING, RETURNS, WARRANTY, CLINIC, LIBRARY, LICENSE, DEPOSIT, ENROLMENT, PER_DIEM];

// ---------------------------------------------------------------- row builder

function pickNew<T>(r: Rng, arr: readonly T[], used: Set<T>): T {
  const free = arr.filter((x) => !used.has(x));
  const x = free.length ? r.pick(free) : r.pick(arr);
  used.add(x);
  return x;
}

function buildRow(i: number, label: Label, r: Rng, dom: Dom): string {
  let f = dom.make(r);
  const core = dom.core(f, f.val);
  const others = dom.others(f);
  const conds = dom.conds(f);
  const contras = dom.contras(f);
  const invents = dom.invents(f);
  const claim = r.pick(dom.claims);

  // the sentence(s) that make the final answer's main claim
  let main: string;
  let exclude = new Set<string>(); // special sentences that must NOT be in the sources
  if (label === 'grounded') {
    main = claim(f, f.val);
  } else if (label === 'partially_grounded') {
    if (r.chance(0.5)) {
      main = claim(f, f.alt); // altered specific
    } else {
      const c = r.pick(conds);
      exclude.add(c);
      main = `${claim(f, f.val)} ${c}`; // unsupported addition
    }
  } else {
    const k = r.pick(r.chance(0.5) ? contras : invents);
    exclude.add(k);
    main = k;
  }

  // sources: core + 3 true sentences, optionally one extra cond/invent (never the answer's)
  const pool: string[] = [core, ...r.shuffle([...others])];
  if (r.chance(0.5)) {
    const extra = conds.filter((x) => !exclude.has(x));
    if (extra.length) pool.push(r.pick(extra));
  }
  // invented-topic sentences are harmless extra facts when they are in the sources;
  // they are never the answer's sentence, so they spread across classes as true material
  for (const x of invents) if (!exclude.has(x) && r.chance(0.45)) pool.push(x);
  // make sure the altered value is nowhere in the sources
  const altRe = new RegExp(`(?<![\\d.])${f.alt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\d])`);
  if (altRe.test(pool.join(' '))) return ''; // altered value would appear in sources: retry

  // passages: 2-3 excerpts, each with one or two sentences and optional noise
  const nPass = r.int(2, 3);
  const titles = r.shuffle([...dom.titles]).slice(0, nPass);
  const groups: string[][] = Array.from({ length: nPass }, () => []);
  pool.forEach((s, k) => groups[k % nPass].push(s));
  const usedNoise = new Set<string>();
  const passages = groups.map((g, k) => {
    if (r.chance(0.8)) g.push(pickNew(r, SOURCE_NOISE, usedNoise));
    if (r.chance(0.5)) g.push(pickNew(r, SOURCE_NOISE, usedNoise));
    return `[S${k + 1} ${titles[k]}] ${g.join(' ')}`;
  });

  // earlier turns: one chit-chat pair, sometimes plus a lone line
  const [cu, ca] = r.pick(CHAT);
  const earlier = [cu, ca];
  const [cu2, ca2] = r.pick(CHAT.filter(([u]) => u !== cu));
  earlier.push(cu2, ca2);
  if (r.chance(0.4)) {
    const lone = r.pick(CHAT.filter(([u]) => !earlier.includes(u)))[0];
    earlier.push(lone);
  }

  // final answer: the main claim, then 0-2 true supporting sentences from the sources
  const lead = r.pick(LEADS);
  const nSupport = r.int(1, 2); // same for every class so length does not reveal the label
  const supports = r.shuffle([...others]).slice(0, nSupport);
  const answer = [`${lead}${main}`, ...supports].join(' ');

  const date = `2026-${pad(r.int(1, 12))}-${pad(r.int(1, 28))}`;
  const header = `Session GS-${100000 + i * 11 + r.int(0, 10)} | retrieved ${date} ${pad(r.int(8, 17))}:${pad(r.int(0, 59))} | index shard ${r.int(1, 9)}`;
  return `${header} || Question: ${dom.question(f)} || Sources: ${passages.join(' | ')} || Earlier turns: ${earlier.join(' | ')} || Assistant final answer: ${answer}`;
}

// ---------------------------------------------------------------- main

const r = new Rng(SEED);
const labels: Label[] = [];
for (const l of Object.keys(PER_LABEL) as Label[]) for (let k = 0; k < PER_LABEL[l]; k++) labels.push(l);
r.shuffle(labels);

const seen = new Set<string>();
const prefixes = new Set<string>();
const out: string[] = [];
labels.forEach((label, i) => {
  const dom = r.pick(DOMAINS);
  let text = buildRow(i, label, r, dom);
  let guard = 0;
  while (!text || seen.has(text) || prefixes.has(text.split(' ').slice(0, 8).join(' '))) {
    text = buildRow(i, label, r, dom);
    if (++guard > 50) throw new Error(`could not make a unique row at ${i}`);
  }
  if (/[\r\n]/.test(text)) throw new Error('newline in text');
  seen.add(text);
  prefixes.add(text.split(' ').slice(0, 8).join(' '));
  out.push(JSON.stringify({ text, label }));
});

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, out.join('\n') + '\n', 'utf8');

const counts: Record<string, number> = {};
const lens: Record<string, number[]> = {};
for (const line of out) {
  const o = JSON.parse(line) as { text: string; label: string };
  counts[o.label] = (counts[o.label] ?? 0) + 1;
  (lens[o.label] ??= []).push(o.text.length);
}
console.log(`wrote ${out.length} rows -> ${OUT}`);
for (const [l, n] of Object.entries(counts)) {
  const a = lens[l].sort((x, y) => x - y);
  console.log(`  ${l.padEnd(19)} n=${n}  len min/med/max ${a[0]}/${a[Math.floor(a.length / 2)]}/${a[a.length - 1]}`);
}
