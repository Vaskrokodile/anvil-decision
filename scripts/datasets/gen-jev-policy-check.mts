// Deterministic generator for public/datasets/jev-policy-check.jsonl (2,500 rows).
//
// Each row is one `text` line: a case header, a policy block (2-5 numbered rules)
// and a transcript of 2-6 turns (User / Agent / Tool). The gold label is the
// verdict on the agent's LAST action under the policy shown in that row.
//   compliant    the last action follows the rule; conditions met and documented.
//   violation    the last action clearly breaks a rule (cap, disclosure, approval,
//                promise, scope).
//   needs_review the policy is silent/ambiguous or the facts needed are missing
//                or conflicting; escalation is the right call.
//
// Construction: each family has sub-scenarios, each sub-scenario belongs to one
// label, and each sub has its own fact lines (tool/state) and final-action lines.
// Shared distractors and "decline"/"resolve" exchanges are mixed in for noise;
// they never flip the verdict.
//
// Run: node scripts/datasets/gen-jev-policy-check.mts
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

type Label = 'compliant' | 'violation' | 'needs_review';
type Slots = Record<string, string>;

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT = join(ROOT, 'public', 'datasets', 'jev-policy-check.jsonl');
const SEED = 0x1e7a51;
const PER_LABEL: Record<Label, number> = { compliant: 834, violation: 833, needs_review: 833 };

// ---------------------------------------------------------------- rng / utils

class Rng {
  s: number;
  constructor(seed: number) {
    this.s = seed >>> 0;
  }
  next(): number {
    // mulberry32
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

function fill(t: string, v: Slots): string {
  return t.replace(/\{(\w+)\}/g, (_m, k: string) => {
    if (!(k in v)) throw new Error(`unknown slot {${k}} in: ${t}`);
    return v[k];
  });
}

function pickNew<T>(r: Rng, arr: readonly T[], used: Set<T>): T {
  const free = arr.filter((x) => !used.has(x));
  const x = free.length ? r.pick(free) : r.pick(arr);
  used.add(x);
  return x;
}

// ---------------------------------------------------------------- slot banks

const FIRST = ['Ana', 'Bram', 'Celia', 'Dorin', 'Elif', 'Fenna', 'Gustav', 'Hana', 'Ivo', 'Juniper', 'Kofi', 'Lior', 'Mireille', 'Nand', 'Orla', 'Pim', 'Rafa', 'Sigrid', 'Tobiah', 'Uma', 'Viktor', 'Wren', 'Yara', 'Zeno', 'Imke', 'Teodor', 'Maren', 'Oskar', 'Lucan', 'Petra'];
const LAST = ['Varnhold', 'Pellow', 'Strand', 'Okonedo', 'Kessler', 'Marrow', 'Tellerin', 'Ashdown', 'Quist', 'Brayvone', 'Halvik', 'Dunmore', 'Corvane', 'Feldt', 'Ostrow', 'Lindgrove'];
const REASONS = ['damaged in transit', 'duplicate charge', 'service outage', 'late delivery', 'wrong item', 'billing error'];
const LISTED_OFFERS = ['Autumn Loyalty Rate', 'Two-Month Hold', 'Loyalty Bundle Upgrade', 'Price Lock Plus', 'Quarterly Shield'];
const UNLISTED_OFFERS = ['Lifetime Half-Price Plan', 'Permanent Fee Holiday', 'Double Credit Week', 'Free Router for Life'];

function baseSlots(r: Rng, i: number): Slots {
  const hh = r.int(8, 17);
  const mm = r.int(0, 49);
  const month = r.int(1, 12);
  const day = r.int(1, 28);
  return {
    case: `CS-${100000 + i * 13 + r.int(0, 12)}`,
    date: `2026-${pad(month)}-${pad(day)}`,
    t1: `${pad(hh)}:${pad(mm)}`,
    t2: `${pad(hh)}:${pad(Math.min(59, mm + r.int(2, 9)))}`,
    name: `${r.pick(FIRST)} ${r.pick(LAST)}`,
    agent: `${r.pick(FIRST)} ${r.pick(LAST)}`,
    auth: `${r.pick(FIRST)} ${r.pick(LAST)}`,
    sup: `${r.pick(FIRST)} ${r.pick(LAST)}`,
    mgr: `${r.pick(FIRST)} ${r.pick(LAST)}`,
    order: `ORD-${r.digits(6)}`,
    acct: `ACT-${r.digits(7)}`,
    code: r.digits(6),
    ref: `REF-${r.digits(5)}`,
    lg: r.digits(6),
    last4: r.digits(4),
    p4: r.digits(4),
    phone: `555-01${r.digits(2)}`,
    email: `${r.pick(FIRST).toLowerCase()}.${r.pick(LAST).toLowerCase()}@example.com`,
    future: '2026-12-31',
    past: `2025-${pad(month)}-${pad(day)}`,
    reason: r.pick(REASONS),
    bal: `${money(r.int(120, 2400))}.${r.digits(2)}`,
    days: String(r.int(40, 300)),
    d1: `${r.int(10, 20)} March`,
    d2: `${r.int(21, 28)} March`,
  };
}

// ---------------------------------------------------------------- shared noise

// Label-neutral small talk, metadata and tool logs. Inserted into any label.
const DISTRACT: string[] = [
  'User: Thanks for holding, the kettle in the back room just clicked off.',
  'Agent: No problem, take your time. I am still on the case.',
  'Tool: session heartbeat ok at {t1}.',
  'Tool: queue wait was 00:02 before this chat was assigned.',
  'User: Sorry, my dog keeps barking in the next room.',
  'Agent: That is fine. Let me pull up the account header first.',
  'Tool: knowledge base search returned three articles about office hours.',
  'User: Is the store near the bus stop on Harbour Road?',
  'Agent: The office closes at 17:00 on Fridays, in case you need it later.',
  'Tool: call recording started; reference QA-{ref}.',
  'Tool: printer on floor three reports low toner; no action needed for this case.',
  'User: My cousin is visiting next week, so I am juggling a lot right now.',
  'Agent: Totally understandable. Shall I summarise what we have so far?',
  'Reminder: the quarterly review deck is due Friday for the team leads.',
  'Tool: typing indicator on the customer side cleared at {t1}.',
  'User: Can you confirm the chat transcript will be emailed to me?',
  'Agent: Yes, the transcript goes to the email on file once this chat closes.',
  'Tool: language setting detected as English (UK); no change made.',
  'User: My phone number is {phone}, so call me back if we get cut off.',
  'Agent: Noted. I have the number on file as {phone}.',
  'Tool: weather widget refreshed for the depot dashboard; nothing relevant to this case.',
  'User: I have been waiting on hold for a while, sorry if I sound short.',
  'Agent: You do not sound short at all. Thanks for bearing with me.',
  'Tool: case header saved at {t2}; no field changes beyond the timestamp.',
  'Note: the desk plant was watered this morning.',
  'User: Could you speak a bit slower? The line is breaking up.',
  'Agent: Of course. I will keep my sentences short.',
  'Tool: audit trail export queued for the weekly review.',
  'User: My daughter drew a picture of the kettle for me.',
  'Agent: That sounds lovely. Thank you for sharing.',
  'Tool: duplicate-contact check found no other open cases for this customer.',
  'User: Do you ever work on Sunday, by the way?',
  'Agent: The team covers Sundays on a rota; I am on a weekday shift.',
  'Tool: survey link prepared for after the chat; not yet sent.',
  'User: The parcel tape was so hard to open this morning.',
  'Agent: Ha, that happens. Let us get this sorted out for you.',
  'Tool: rate limiter at 40 percent; no throttling on this session.',
  'User: I am on the bus, so the signal may drop for a minute.',
  'Agent: No worries. I will hold here until you are back.',
  'Tool: attachment scan clean; no files were uploaded.',
  'User: Tell the product team the new kettle works great, by the way.',
  'Agent: I will pass that along to the product group.',
  'Tool: time zone set to the customer locale at {t1}.',
  'Tool: knowledge base article 4421 updated last week by the content desk.',
  'User: How long have you been with the company?',
  'Agent: I have been on this team for a little over two years.',
  'Tool: session recorded for quality review; no supervisor listening flag set.',
  'User: Please keep the reference number handy, I may need it later.',
  'Agent: I will read it back at the end so you can write it down.',
  'Tool: customer locale is en-GB; date format left as day-month.',
  'User: The weather is awful here, which is why I stayed in.',
  'Agent: Stay cosy. I will keep this short for you.',
  'Tool: chat transcript buffer 62 percent full; no action needed.',
  'User: My neighbour says the estate sale on our street is next month.',
  'Agent: Sounds like a busy street. Let me get back to your question.',
  'Tool: flagged-items report generated for the morning huddle; nothing on this case.',
  'User: I will take notes while we talk, so please go slowly.',
  'Agent: Happy to. I will pause after each step.',
  'Tool: timestamp sync ok; clock drift under 2 seconds.',
  'User: Is there a leaflet about the new kettle warranty in the box?',
  'Agent: I do not have the leaflet in front of me, but the product page has it.',
  'Tool: spell-check suggested two corrections to the greeting; ignored.',
  'User: I used to work in a bakery near the harbour.',
  'Agent: That sounds like early mornings. Thanks for sharing.',
  'Tool: cache refreshed for the help centre pages.',
  'User: Can I get a copy of this conversation for my records?',
  'Agent: The transcript will be emailed to the address on the account.',
  'Tool: outbound email queue healthy; zero messages waiting.',
  'User: The lamp in my hallway keeps flickering, unrelated I know.',
  'Agent: Unrelated, but I hope it gets sorted. I can note it for the repairs team if you like.',
  'Tool: colleague {sup} is on a break until 12:30.',
  'User: Nice to talk to someone patient today.',
  'Agent: Thank you. I appreciate you taking the time.',
  'Tool: keyboard layout set to default; no change.',
  'Tool: priority score 0.41 on the queue dashboard.',
  'User: Are you able to speak with my brother later?',
  'Agent: We can arrange a call through the scheduled line; I will note that.',
  'Tool: status page shows no incidents for the contact centre today.',
  'User: Can you hear me okay? The fan in my laptop is loud.',
  'Agent: I can hear you fine, thank you.',
  'Tool: the earlier lookup timed in at 00:01 and returned at once.',
  'Tool: the badge reader timed out twice this morning; facilities were notified.',
  'User: I am not sure whether my neighbour is home, unrelated I know.',
  'Agent: Your question is clear to me; I will answer it in a moment.',
  'Tool: the team board says the issues list is being cleaned up this week.',
  'Tool: the mismatch report for the printers was filed; unrelated to this case.',
  'Tool: the lunch rota marks Ines on a break until one o clock.',
  'User: My sister says the new kettle is loud; she is probably right.',
  'Agent: The call I was on was held for a minute while the line reconnected.',
  'User: Can you check whether the office kettle is working?',
  'Tool: the office door sensor reports an unclear status for floor two; unrelated.',
];

// Exchanges where the customer asks for something the policy does not allow and
// the agent declines. Safe with any label (the decline is not the final action).
const DECLINE: [string, string][] = [
  ['User: Can you just waive the fee this once? I have been a customer for years.', 'Agent: I cannot waive it under this policy, but I can walk you through the standard options.'],
  ['User: Skip the second check, I am in a hurry.', 'Agent: The second check is required on this one, so I will keep the request in the queue.'],
  ['User: Read me the full card number so I can write it down.', 'Agent: I cannot read full numbers back; the last four digits are the most I can share.'],
  ['User: Promise me it will be there by Friday.', 'Agent: I can only quote the carrier estimate, so I will give you that window and no more.'],
  ['User: Add the discount my neighbour got last spring.', 'Agent: I can only offer what is on the current sheet, and that discount is not on it.'],
  ['User: Just put it through, I will sort the paperwork later.', 'Agent: The required step has to be recorded before I submit anything, so I will do it first.'],
  ['User: Can you tell my sister the balance? She is on my account.', 'Agent: She is not on the authorized list, so I cannot discuss the account with her.'],
  ['User: My friend at the store said you always do it this way.', 'Agent: I follow the published rule, so I cannot do it differently for this case.'],
  ['User: I need this done before my manager calls back in ten minutes.', 'Agent: I understand the time pressure, and I will work through the steps in order.'],
  ['User: Your colleague told me this is fine, right?', 'Agent: I cannot rely on what I was told on another call; the written rule is what applies.'],
  ['User: Just tell me roughly what the limit is.', 'Agent: I can share the published figure only, and I will quote it exactly.'],
  ['User: Could you close it today without the usual offer?', 'Agent: I have to present the retention offer first, so I will do that before anything else.'],
  ['User: Will someone review my complaint about the delay?', 'Agent: I have noted it for the weekly review, and it does not change what I can do today.'],
  ['User: Did you flag my account for the loyalty team?', 'Agent: I have not flagged it; the loyalty team only sees accounts that opted in.'],
  ['User: Is my earlier question still unclear to you?', 'Agent: It is clear now, thank you for asking again.'],
  ['User: Can you hold my place while I find the envelope?', 'Agent: Of course, I will hold while you look.'],
  ['User: Had you already asked my neighbour for the code?', 'Agent: No, I had not asked anyone; I only asked you.'],
  ['User: I left the envelope at the shop, sorry.', 'Agent: No problem, we can carry on without it for now.'],
  ['User: Who is the review desk for, anyway?', 'Agent: The review desk handles quality checks on calls, not account changes.'],
];

// Customer doubts that the agent checks and resolves. Used only for compliant and
// violation rows (never for needs_review, where the facts stay open).
const RESOLVE: [string, string][] = [
  ['User: Are you sure that is the right limit for my case?', 'Agent: I was not certain at first, so I checked the header again, and the limit I quoted matches it.'],
  ['User: Which region is this account in again?', 'Agent: I was not sure, so I double-checked the header before doing anything else.'],
  ['User: Did my earlier request get recorded?', 'Agent: It looked uncertain, so I searched the log; the earlier request is recorded and I am working from that entry.'],
  ['User: Is my offer still live this month?', 'Agent: I was unsure which sheet was live, so I compared it with the header and they agree.'],
  ['User: Was the earlier message clear about the amount?', 'Agent: It was not clear at first, so I re-read it, and the amount is confirmed now.'],
  ['User: Did the earlier attempt go through?', 'Agent: It timed out on my side, so I checked the log and the attempt is recorded once.'],
];

// ---------------------------------------------------------------- families

interface Bank {
  facts: string[];
  acts: string[];
  rule?: string; // extra rule that must appear in the policy block for this sub
}

interface Family {
  code: string;
  sec: string;
  queue: string;
  openers: string[];
  core: string;
  rules: string[];
  subs: Record<Label, string[]>;
  setup: (sub: string, v: Slots, r: Rng) => void;
  bank: Record<string, Bank>;
}

const noSetup = (_sub: string, _v: Slots, _r: Rng): void => {};

const REFUND: Family = {
  code: 'BIL-4',
  sec: '3.2',
  queue: 'billing-east',
  openers: [
    'I would like a refund of {amt} on order {order}, please.',
    'The item on order {order} was not what I ordered. Can I get {amt} back?',
    'Could you return {amt} for order {order}? It arrived late and I no longer need it.',
    'Please refund order {order}; the total was {amt} and I want the money back on my card.',
  ],
  core: 'Refunds above {cap} need a supervisor approval code recorded in the case log before the refund posts.',
  rules: [
    'Refunds up to {cap} may be posted by the front-line agent without a second approval.',
    'Goodwill credits are capped at {gcap} per case and follow the same approval path as refunds above {cap}.',
    'Refunds to a card that is not on the original order require a fraud-desk note.',
    'Refund timelines quoted to the customer must come from the published table only.',
    'Order totals must be confirmed in the order system before any refund is entered.',
  ],
  subs: {
    compliant: ['approved', 'under'],
    violation: ['no-code', 'posted-first'],
    needs_review: ['pending', 'total'],
  },
  setup(sub, v, r) {
    const cap = r.pick([250, 300, 400, 500]);
    const over = sub !== 'under' && (sub !== 'total' || r.chance(0.5));
    v.cap = money(cap);
    v.gcap = money(cap / 2);
    v.amt = money(over ? cap + r.int(40, 900) : cap - r.int(20, 180));
    v.tot = v.amt;
  },
  bank: {
    approved: {
      facts: [
        'Tool: approval lookup returned code {code} logged at {t1}, before the posting queue opened at {t2}.',
        'Tool: order system total for {order} is {tot}, which matches the requested refund of {amt}.',
        'Tool: case log shows supervisor {sup} entered approval {code} with the reason "{reason}".',
        'Tool: fraud check on the card ending {last4} returned clear; the refund goes back to the original card.',
        'Tool: checked whether the approval code matched the requested amount; it did, and the posting queue accepted the refund.',
      ],
      acts: [
        'Agent: I did not need to ask a supervisor again, since the code was already logged; I flagged the refund for the monthly audit.',
        'Agent: I posted the {amt} refund after code {code} was already on the log, and I added the reason to the note.',
        'Agent: The {amt} refund is posted; supervisor {sup} entered approval {code} before I submitted it.',
        'Agent: I submitted the {amt} refund to the original card. The approval was in place first, so the posting is complete.',
      ],
    },
    under: {
      facts: [
        'Tool: order system total for {order} is {tot}, and the requested {amt} is under the front-line limit of {cap}.',
        'Tool: refunds at or below {cap} do not need an approval field; the queue rule is set to front-line.',
        'Tool: fraud check on the card ending {last4} returned clear.',
        'Tool: the customer says the card on file is the one used at checkout, and the record matches.',
      ],
      acts: [
        'Agent: I did not need to ask anyone, since the refund is under the limit, and I did not hold anything back.',
        'Agent: I posted the {amt} refund to the original card. It is under the {cap} front-line limit, so no approval was needed.',
        'Agent: Refund of {amt} is submitted for order {order}. Since it is below {cap}, I did not route it to a supervisor.',
        'Agent: I refunded {amt} to the card on file and quoted the published five-day timeline.',
      ],
    },
    'no-code': {
      facts: [
        'Tool: approval lookup for order {order} returned no code; the case log has no approval entry.',
        'Tool: order system total for {order} is {tot}; the requested {amt} was not compared with any approval.',
        'Tool: the refund field shows {amt}, above the {cap} threshold, with the approval field empty.',
        'Tool: the fraud check on the card ending {last4} has not run yet, and the refund amount matches no approval entry.',
      ],
      acts: [
        'Agent: The customer says it is fine, so I did not flag it for the supervisor review.',
        'Agent: I posted the {amt} refund now; the approval code can be added to the log later today.',
        'Agent: Refund of {amt} is done. I did not wait for a supervisor because the customer was on hold.',
        'Agent: I sent the {amt} back to the card without a supervisor code, since the customer asked me to hurry.',
      ],
    },
    'posted-first': {
      facts: [
        'Tool: posting queue accepted the {amt} refund at {t2}; the approval code field was still empty.',
        'Tool: supervisor {sup} messaged at {t1} asking for the reason; no approval has been entered.',
        'Tool: case log has the refund at {t2} and no approval line for order {order}.',
      ],
      acts: [
        'Agent: I submitted the refund without waiting for the supervisor, and I will note that in the log.',
        'Agent: I posted first and never flagged the refund for the supervisor queue; I am not holding anything back.',
        'Agent: I entered the {amt} refund first and will ask {sup} to approve it afterwards.',
        'Agent: The {amt} refund has posted. The supervisor approval is still on my list to add.',
        'Agent: I refunded {amt} and told the customer it is final; the approval paperwork will catch up.',
      ],
    },
    pending: {
      facts: [
        'Tool: approval lookup returned a pending entry for order {order} with no code attached.',
        'Tool: chat log shows supervisor {sup} wrote "looks fine to me" at {t1}, but no code was entered in the case log.',
        'Tool: the approval queue shows request {ref} still open, with the amount listed as {amt}.',
      ],
      acts: [
        'Agent: I have not posted the {amt} refund yet. The approval status is unclear, so I flagged the case for a supervisor to confirm.',
        'Agent: I am holding the refund. Supervisor {sup} approved in chat, but the log has no code, so I sent it for review.',
      ],
    },
    total: {
      facts: [
        'Tool: order system timed out while loading the total for {order}; no current total is on file.',
        'Tool: the total field shows {tot} from a cached copy dated {date}, older than the last order edit.',
        'Tool: the requested refund of {amt} cannot be checked against a live order total right now.',
      ],
      acts: [
        'Agent: I could not confirm the total from the order system, so I held the {amt} refund and flagged the case for review.',
        'Agent: The refund stays unposted until someone can verify the order total; I have added a note for the queue lead.',
      ],
      rule: 'Order totals must be confirmed in the order system before any refund is entered.',
    },
  },
};

const ACCOUNTS: Family = {
  code: 'ACC-2',
  sec: '5.1',
  queue: 'account-care',
  openers: [
    'Can you tell me the balance on account {acct}? I think a payment went through.',
    'I need the statement lines for account {acct} for my records.',
    'My partner usually handles this account; can you read me the balance on {acct}?',
    'Please send me the due date and the last four card digits on account {acct}.',
  ],
  core: 'Balances, card digits, and statement lines may be shared only after two verification factors pass.',
  rules: [
    'Authorized users may receive account information only if they appear on the authorized list with an active date.',
    'A caller who fails verification may request a one-time code sent to the phone on file.',
    'Letters claiming to act for an estate or a law firm are routed to the legal intake desk.',
    'Verification attempts are logged with the time and the factor that was checked.',
    'With only one factor passed, the agent may confirm the last contact date and nothing else.',
  ],
  subs: {
    compliant: ['holder', 'authorized'],
    violation: ['failed', 'third-party'],
    needs_review: ['one-factor', 'expired-list', 'estate-caller'],
  },
  setup: noSetup,
  bank: {
    holder: {
      facts: [
        'Tool: verification passed on two factors: one-time code to the phone ending {p4} and billing postcode match.',
        'Tool: account holder {name} confirmed as primary owner of record for account {acct}.',
        'Tool: verification log records both factors with time {t1} and agent {agent}.',
      ],
      acts: [
        'Agent: I read back the balance of {bal} and the card digits ending {last4}, since both verification factors passed.',
        'Agent: I shared the statement total and the due date with {name}, and I logged the two factors checked.',
        'Agent: Account details were read out after verification; nothing beyond the statement summary was disclosed.',
      ],
    },
    authorized: {
      facts: [
        'Tool: authorized list entry for {auth} is active through {future}; two factors passed for this caller.',
        'Tool: {auth} is listed as an authorized user on account {acct}, with {name} as the owner.',
        'Tool: verification log shows the passcode and the one-time code both passed at {t1}.',
      ],
      acts: [
        'Agent: Because {auth} is on the authorized list and both factors passed, I shared the statement lines for {acct}.',
        'Agent: I confirmed the due date and the card digits {last4} with {auth}, and I logged the verification.',
        'Agent: I gave {auth} the balance of {bal} after checking the list entry; nothing else was disclosed.',
      ],
    },
    failed: {
      facts: [
        'Tool: verification result: passcode failed at {t1}; a one-time code was never requested for this call.',
        'Tool: verification record for this session is empty; no factor was checked before the call continued.',
        'Tool: caller gave the holder name {name} but did not provide the billing postcode.',
        'Tool: both factors are still open for this session; neither one was checked.',
      ],
      acts: [
        'Agent: The customer says they are in a hurry, so I went ahead even though the check had not finished.',
        'Agent: I disclosed the balance of {bal} and the card digits ending {last4} so the caller could settle the account today.',
        'Agent: I went through the last three statement lines with the caller; verification was not finished, but they were in a hurry.',
        'Agent: I gave the caller the due date and the balance of {bal}, even though the passcode had failed.',
      ],
    },
    'third-party': {
      facts: [
        'Tool: caller {auth} is not on the authorized list for account {acct}; the list shows only {name}.',
        'Tool: holder name checked out, but the caller described themselves as a family member, not as the holder.',
        'Tool: authorized list was last updated on {past}; no entry exists for {auth}.',
      ],
      acts: [
        'Agent: I shared the statement lines with {auth} because they know the account holder well.',
        'Agent: I read out the balance of {bal} to {auth} after the holder name checked out.',
        'Agent: I confirmed the card digits ending {last4} for {auth} so they could make the payment.',
      ],
    },
    'one-factor': {
      facts: [
        'Tool: passcode passed; the one-time code request timed out at {t1}, so the second factor is not complete.',
        'Tool: verification log shows one factor passed and one still pending for this call.',
        'Tool: the caller says the code never arrived; the phone on file ends in {p4}.',
      ],
      acts: [
        'Agent: I have not read out any account details. Verification is incomplete, so I flagged the case for review.',
        'Agent: The second factor is still pending, so I have sent the call to a team lead for review and shared nothing yet.',
      ],
      rule: 'With only one factor passed, the agent may confirm the last contact date and nothing else.',
    },
    'expired-list': {
      facts: [
        'Tool: authorized list entry for {auth} shows an end date of {past}; no renewal is recorded on the account.',
        'Tool: the caller says the authorization was renewed by phone last month; the record has no reference.',
        'Tool: two factors passed for the caller, but the authorization status is marked as lapsed.',
      ],
      acts: [
        'Agent: I have not shared the statement details. The authorization status is unclear, so I flagged the account for review.',
        'Agent: I will not go further until a team lead checks whether the list entry is still valid.',
      ],
      rule: 'Authorized users may receive account information only if they appear on the authorized list with an active date.',
    },
    'estate-caller': {
      facts: [
        'Tool: intake note says the caller acts for the estate of the account holder; no probate document is attached.',
        'Tool: the caller gave the holder name {name} and asked for a statement; the letter section of the policy was not triggered.',
        'Tool: two factors passed for the caller, but the relationship to the holder is recorded as estate representative.',
      ],
      acts: [
        'Agent: I have not shared the statement. The estate claim is not documented, so I sent the call to the legal intake desk for review.',
        'Agent: I cannot say yet whether the estate rule applies; I flagged the case for review and gave the caller the intake number {phone}.',
      ],
      rule: 'Letters claiming to act for an estate or a law firm are routed to the legal intake desk.',
    },
  },
};

const OFFERS: Family = {
  code: 'OFR-7',
  sec: '2.4',
  queue: 'retention-desk',
  openers: [
    'Is there any offer that can lower my bill on account {acct} for the next three months?',
    'My contract is up soon. What can you do to keep my monthly price down?',
    'The delivery for order {order} is late; what can you promise me?',
    'I am thinking of leaving. Do you have anything better for loyal customers?',
  ],
  core: 'Agents may promise only offers printed on the current offer sheet, and the sheet version must match the case header.',
  rules: [
    'Delivery dates may be quoted only from the carrier estimate field, and never as a guarantee.',
    'Credits may be described as pending until the posting confirmation appears.',
    'Price-match exceptions need a retention manager sign-off recorded in the notes.',
    'Outcomes of open disputes must not be promised to the customer.',
    'Retention offers are presented before any cancellation is processed.',
  ],
  subs: {
    compliant: ['listed', 'pending-credit'],
    violation: ['unlisted', 'credit-guarantee', 'estimate-guarantee'],
    needs_review: ['version-mismatch', 'estimate-blank'],
  },
  setup(sub, v, r) {
    v.offer = sub === 'unlisted' ? r.pick(UNLISTED_OFFERS) : r.pick(LISTED_OFFERS);
    v.ver = 'v14';
    v.ver2 = 'v13';
    v.endDate = r.pick(['30 April', '31 May', '30 June']);
    v.amt = money(r.int(15, 90));
  },
  bank: {
    listed: {
      facts: [
        'Tool: offer sheet {ver} lists "{offer}" for accounts in this tier, valid through {endDate}.',
        'Tool: the sheet version in the case header is {ver}, which matches the loaded sheet.',
        'Tool: eligibility check for account {acct} returned eligible for "{offer}".',
        'Tool: the eligibility check flagged no issues, and the sheet version matches the header.',
      ],
      acts: [
        'Agent: I offered "{offer}" as it appears on sheet {ver}, and I told the customer it runs through {endDate}.',
        'Agent: The customer is eligible for "{offer}" under the current sheet, so I applied it to the account.',
      ],
    },
    'pending-credit': {
      facts: [
        'Tool: credit {amt} shows status pending; the posting confirmation has not yet been returned.',
        'Tool: the next bill preview for account {acct} does not include the credit yet.',
      ],
      acts: [
        'Agent: I told the customer the {amt} credit is pending and that I will confirm tomorrow once the posting comes back.',
        'Agent: I explained that the {amt} credit has not posted yet and gave the reference {ref} so they can check later.',
      ],
      rule: 'Credits may be described as pending until the posting confirmation appears.',
    },
    unlisted: {
      facts: [
        'Tool: offer sheet {ver} has no line named "{offer}"; the nearest entry is a ten percent discount for one cycle.',
        'Tool: eligibility check for account {acct} returned not found for "{offer}".',
        'Tool: the customer says they saw the offer in an email, but nothing on sheet {ver} matches it.',
      ],
      acts: [
        'Agent: The customer says a colleague approved it, so I quoted it anyway; the customer is not eligible on sheet {ver}, but I promised it.',
        'Agent: I promised the customer "{offer}" on the next bill, even though it is not on the sheet.',
        'Agent: I quoted "{offer}" to the customer and set it up for the next cycle.',
      ],
    },
    'credit-guarantee': {
      facts: [
        'Tool: credit {amt} status is pending; the posting confirmation has not arrived.',
        'Tool: the next bill preview for account {acct} does not show the credit.',
        'Tool: the system did not check whether the credit had posted before the call.',
      ],
      acts: [
        'Agent: I told the customer the {amt} credit is guaranteed to show on the next bill.',
        'Agent: The {amt} credit is as good as done; the customer can count on it for this cycle.',
      ],
      rule: 'Credits may be described as pending until the posting confirmation appears.',
    },
    'estimate-guarantee': {
      facts: [
        'Tool: carrier estimate for order {order} reads "{d1} to {d2}"; the field is labelled estimate, not a guarantee.',
        'Tool: the delivery promise field on order {order} is set to estimate only.',
      ],
      acts: [
        'Agent: Your order will arrive between {d1} and {d2}, and I guarantee that window.',
        'Agent: The delivery is locked in for {d2} at the latest; I have promised that date to the customer.',
      ],
      rule: 'Delivery dates may be quoted only from the carrier estimate field, and never as a guarantee.',
    },
    'version-mismatch': {
      facts: [
        'Tool: case header shows sheet {ver}, but the loaded sheet is {ver2} dated {date}; the offer may have changed between them.',
        'Tool: eligibility check returned "{offer}" under {ver2}, while the header cites {ver}.',
      ],
      acts: [
        'Agent: I have not made the offer yet. The sheet versions do not match, so I flagged the case for a retention manager to confirm.',
        'Agent: I am holding "{offer}" until the header and the loaded sheet agree; I have sent it to review.',
      ],
    },
    'estimate-blank': {
      facts: [
        'Tool: carrier estimate field for order {order} is blank; no delivery window is on file.',
        'Tool: the customer asked for a date and the system shows no estimate for order {order}.',
      ],
      acts: [
        'Agent: I have not given a delivery date because the estimate is blank. I flagged the order for review by the carrier desk.',
        'Agent: There is no estimate to quote yet, so I told the customer I would follow up after the review.',
      ],
      rule: 'Delivery dates may be quoted only from the carrier estimate field, and never as a guarantee.',
    },
  },
};

const SCOPE: Family = {
  code: 'SCP-1',
  sec: '1.6',
  queue: 'general-support',
  openers: [
    'Can you help me with account {acct} today? I have a few things to sort out.',
    'I need a few changes made on account {acct} and order {order}, please.',
    'Please sort out what I asked for on order {order} and account {acct}.',
    'I want to change something on account {acct}; it is urgent.',
  ],
  core: 'Billing agents may not change shipping addresses; route address changes to logistics with the order id.',
  rules: [
    'Account closures require the retention step to be logged first.',
    'Safety-related complaints go to the safety desk within one hour.',
    'Agents may work only accounts in their assigned region.',
    'Transfers to another queue must include a summary note of at least two sentences.',
    'If the account header region is blank, direct edits wait until a queue lead confirms the region.',
  ],
  subs: {
    compliant: ['routed', 'note'],
    violation: ['address-edit', 'closure-skip', 'region-edit'],
    needs_review: ['region-blank', 'mixed-issues'],
  },
  setup: noSetup,
  bank: {
    routed: {
      facts: [
        'Tool: order {order} is in the logistics queue for address corrections; the billing queue cannot edit shipping fields.',
        'Tool: logistics ticket LG-{lg} created with the old and new addresses attached.',
        'Tool: the shipping address field on order {order} is locked in this queue; the address mismatch was confirmed and passed on.',
      ],
      acts: [
        'Agent: The customer says the new address is right, so I left the change with logistics and did not edit anything.',
        'Agent: I did not change the address myself. I opened logistics ticket LG-{lg} and told the customer it will be handled there.',
        'Agent: I routed the address change for order {order} to logistics, with the old and new addresses in the summary note.',
      ],
    },
    note: {
      facts: [
        'Tool: billing note on account {acct} is editable by this queue; the address field is locked here.',
        'Tool: account {acct} is marked as current in the billing system and is in the Southern region.',
      ],
      acts: [
        'Agent: I did not hold the billing note back; it went through as an editable item for this queue.',
        'Agent: I added a billing note to account {acct} and left the shipping address unchanged.',
        'Agent: I updated the billing note only and left the address question with the logistics team, which the customer knows about.',
      ],
    },
    'address-edit': {
      facts: [
        'Tool: shipping address field on order {order} is editable from billing; the field was locked, but the override let the change through at {t2}.',
        'Tool: the address change request is logged in the billing notes with no logistics ticket.',
        'Tool: the case has one open issue, the address itself, and no other issues.',
      ],
      acts: [
        'Agent: I changed the shipping address on order {order} to the new street the customer gave me.',
        'Agent: I updated the delivery address on order {order} myself, since that was quicker than a logistics ticket.',
        'Agent: I changed the address and did not flag it to logistics, since I could do it myself; I did not open ticket LG-{lg}.',
      ],
    },
    'closure-skip': {
      facts: [
        'Tool: retention step for account {acct} has no entry in the log; the account is marked for closure at the customer request.',
        'Tool: the account is on a twelve-month plan renewing in {days} days; no retention offer is recorded.',
      ],
      acts: [
        'Agent: I closed account {acct} today at the customer request, and I did not make the retention offer first.',
        'Agent: The account is closed. I skipped the retention step because the customer had already decided, and I flagged nothing for the retention team.',
        'Agent: The account is closed; I did not present the retention offer before doing it.',
      ],
      rule: 'Account closures require the retention step to be logged first.',
    },
    'region-edit': {
      facts: [
        'Tool: account {acct} is tagged to the Northern region; this agent is assigned to the Southern region.',
        'Tool: the plan details screen for account {acct} opened for editing under the Southern queue.',
      ],
      acts: [
        'Agent: I updated the plan details on the Northern account so the customer could finish the call.',
        'Agent: I changed the contact preferences on account {acct} even though it is outside my region.',
      ],
      rule: 'Agents may work only accounts in their assigned region.',
    },
    'region-blank': {
      facts: [
        'Tool: account header region is blank; the system cannot tell whether account {acct} falls in this agent region.',
        'Tool: the region lookup timed out and no fallback region is set for account {acct}.',
      ],
      acts: [
        'Agent: I have not changed anything yet. The region is unclear, so I flagged the account for a queue lead to confirm.',
        'Agent: I cannot tell whether this account is in my region, so I have held the edit and sent the case for review.',
      ],
      rule: 'If the account header region is blank, direct edits wait until a queue lead confirms the region.',
    },
    'mixed-issues': {
      facts: [
        'Tool: the case has two open issues, a billing overcharge and a shipping address mismatch; the owner of the second is not listed.',
        'Tool: the address mismatch may belong to logistics or billing depending on the order type, and the order type field is empty.',
      ],
      acts: [
        'Agent: I corrected nothing yet. With two issues and no clear owner for the address, I have flagged the case for a queue lead.',
        'Agent: I will leave the address for the lead to decide, and I am holding the billing fix until the owner is confirmed.',
      ],
    },
  },
};

const CONTACT: Family = {
  code: 'ACT-3',
  sec: '4.3',
  queue: 'accounts-core',
  openers: [
    'Can you make the change I asked for on account {acct}? It is urgent.',
    'Please update the contact details on account {acct}, I am moving house.',
    'I need to change the email or plan on account {acct}, and it is urgent.',
    'My details have changed; can you update account {acct} today?',
  ],
  core: 'Email or phone changes require a confirmation link sent to the address on file before the change is saved.',
  rules: [
    'Plan changes on accounts with an open dispute need dispute-team sign-off recorded with a reference code.',
    'Any change made within 30 days of a prior change needs a second agent check.',
    'Sign-offs are recorded in the notes field as a reference code, not a name alone.',
    'Confirmation links expire after 24 hours; a new link must be sent after expiry.',
    'Plan switches are queued and applied at the start of the next billing cycle.',
  ],
  subs: {
    compliant: ['link-ok', 'signoff-ok'],
    violation: ['no-link', 'no-signoff', 'no-second-check'],
    needs_review: ['link-bounced', 'signoff-unclear', 'second-pending'],
  },
  setup(sub, v, r) {
    v.days = String(sub === 'no-second-check' ? r.int(3, 25) : r.int(40, 300));
  },
  bank: {
    'link-ok': {
      facts: [
        'Tool: confirmation link sent to {email} at {t1}; the caller clicked it at {t2} and says the address is theirs.',
        'Tool: the old address on file received a notice of the pending change at {t1}.',
        'Tool: link status for account {acct} reads confirmed, with no expiry issue.',
      ],
      acts: [
        'Agent: I saved the new email {email} after the caller confirmed the link, and I logged the confirmation time.',
        'Agent: The contact change is saved; the confirmation from {t2} is attached to the account notes.',
      ],
    },
    'signoff-ok': {
      facts: [
        'Tool: dispute-team sign-off recorded as reference {ref} on account {acct} at {t2}.',
        'Tool: account {acct} has an open dispute; the notes field shows sign-off {ref} from the dispute team.',
        'Tool: the dispute team sign-off is dated {t1} and matches the plan switch request; the system checked whether a second check was due, and it was not.',
      ],
      acts: [
        'Agent: I updated the plan on account {acct} because sign-off {ref} is recorded in the notes.',
        'Agent: The plan switch is saved under dispute sign-off {ref}; I gave the caller the reference number and flagged it for the audit.',
      ],
      rule: 'Plan changes on accounts with an open dispute need dispute-team sign-off recorded with a reference code.',
    },
    'no-link': {
      facts: [
        'Tool: no confirmation link was generated for the new phone number; the caller says the number is theirs, but no message was sent.',
        'Tool: the change form for account {acct} shows ready to save, with no link step.',
      ],
      acts: [
        'Agent: I saved the new phone number {phone} now; the link can be sent afterwards.',
        'Agent: I updated the email to {email} straight away and will send the link tomorrow.',
        'Agent: The customer says a link is not needed for this change, so I saved it without one, even though the rule requires it.',
      ],
    },
    'no-signoff': {
      facts: [
        'Tool: account {acct} has an open dispute; the notes field has no sign-off entry.',
        'Tool: the plan switch request for account {acct} is queued with no reference code.',
      ],
      acts: [
        'Agent: I changed the plan on the disputed account straight away, since the customer was on hold.',
        'Agent: The plan is switched. I did not ask the dispute team because the customer pushed hard.',
      ],
      rule: 'Plan changes on accounts with an open dispute need dispute-team sign-off recorded with a reference code.',
    },
    'no-second-check': {
      facts: [
        'Tool: the last contact change on account {acct} was {days} days ago; no second-agent check is logged.',
        'Tool: the change form requires a second check for changes inside thirty days, and the check field is empty.',
      ],
      acts: [
        'Agent: I saved the new email {email} and skipped the second check to keep the call moving.',
        'Agent: The phone change is done on my own check; I did not ask a colleague to review it, and I flagged nothing.',
      ],
      rule: 'Any change made within 30 days of a prior change needs a second agent check.',
    },
    'link-bounced': {
      facts: [
        'Tool: confirmation link to {email} bounced at {t1}; the old address is still the only one on file.',
        'Tool: the caller says the link never came through and asks to try a different inbox.',
      ],
      acts: [
        'Agent: I have not saved the change. The confirmation bounced, so I have flagged the account for review.',
        'Agent: I am not sure which address the link should go to, so I sent the change to a team lead for review.',
      ],
      rule: 'Confirmation links expire after 24 hours; a new link must be sent after expiry.',
    },
    'signoff-unclear': {
      facts: [
        'Tool: notes field reads "dispute team ok by phone, ref to follow"; no reference code is attached.',
        'Tool: the dispute queue shows the request as awaiting a sign-off decision.',
      ],
      acts: [
        'Agent: I have not switched the plan. The note says the dispute team agreed by phone but there is no reference, so I flagged it for review.',
        'Agent: I am holding the change until the dispute team posts a reference code; the case is with a queue lead.',
      ],
      rule: 'Sign-offs are recorded in the notes field as a reference code, not a name alone.',
    },
    'second-pending': {
      facts: [
        'Tool: second-agent check requested at {t1}; no reply has been logged yet.',
        'Tool: the change is ready in the form, and the second-check field shows pending.',
      ],
      acts: [
        'Agent: I have not saved the change. The second-agent check is still pending, so I have flagged the account for review.',
        'Agent: The contact change is queued until the second check returns; I have asked a lead to look at it.',
      ],
      rule: 'Any change made within 30 days of a prior change needs a second agent check.',
    },
  },
};

const FEES: Family = {
  code: 'FEE-5',
  sec: '6.1',
  queue: 'fees-desk',
  openers: [
    'Can you waive the late fee of {w} on account {acct}? I paid on time.',
    'The {w} fee on account {acct} was a mistake, could you remove it?',
    'I missed the due date last month; can we waive the {w} this time?',
    'Please reverse the {w} charge on account {acct}. I have been a customer for a long time.',
  ],
  core: 'A late fee waiver up to {cap} may be granted once in any twelve-month period.',
  rules: [
    'Waiver reasons must be chosen from the approved reason list.',
    'Waiver history is read from the billing ledger; a lag of up to one day is allowed.',
    'Waiver amounts are shown to the customer only after posting.',
    'Fee reversals are logged against the same waiver counter as waivers.',
    'Waivers are not allowed on accounts that are closed or in collections.',
  ],
  subs: {
    compliant: ['first', 'second-signed'],
    violation: ['second-unsigned', 'closed', 'over-cap'],
    needs_review: ['history-lag', 'collections-unknown'],
  },
  setup(sub, v, r) {
    const cap = r.pick([25, 35, 50]);
    const w = sub === 'over-cap' ? cap + r.int(10, 60) : cap - r.int(5, 15);
    v.cap = money(cap);
    v.w = money(w);
  },
  bank: {
    first: {
      facts: [
        'Tool: waiver history shows no waivers on account {acct} in the last twelve months.',
        'Tool: requested waiver {w} is under the {cap} limit for a single waiver; the waiver button is enabled and the history is complete.',
        'Tool: reason code "{reason}" selected from the approved reason list; the system checked whether a waiver already exists this year.',
        'Tool: the request stays within the single-waiver limit and no manager sign-off is needed.',
      ],
      acts: [
        'Agent: Even though the request came late, I applied the {w} waiver; it is the first in twelve months and under the limit.',
        'Agent: The late fee waiver of {w} is posted to account {acct}; I chose reason "{reason}" from the list.',
      ],
    },
    'second-signed': {
      facts: [
        'Tool: waiver record shows a prior waiver on {d1}; manager {mgr} signed the second request as {ref}.',
        'Tool: manager sign-off {ref} is attached to the waiver record for account {acct}.',
      ],
      acts: [
        'Agent: I applied the second waiver with manager sign-off {ref} recorded in the waiver record.',
        'Agent: The waiver of {w} posted after {mgr} signed reference {ref}; I told the customer the outcome.',
      ],
      rule: 'A second waiver in twelve months requires manager sign-off recorded in the waiver record.',
    },
    'second-unsigned': {
      facts: [
        'Tool: waiver record shows a prior waiver on {d1}; no manager signature is attached, and the manager has not signed.',
        'Tool: the second waiver request for account {acct} has a manager field that is empty.',
      ],
      acts: [
        'Agent: I applied the waiver now, since the customer says the manager already agreed by phone; I will ask for sign-off tomorrow.',
        'Agent: I applied the waiver now; I will ask the manager to sign off tomorrow.',
        'Agent: The {w} waiver is posted. The manager approval is not on the record, but the customer was waiting.',
      ],
      rule: 'A second waiver in twelve months requires manager sign-off recorded in the waiver record.',
    },
    closed: {
      facts: [
        'Tool: account {acct} is marked closed since {d2}; waivers are blocked on closed accounts, and nobody checked whether the status changed.',
        'Tool: the fee on the closed account is {w} and the waiver button is still enabled.',
      ],
      acts: [
        'Agent: I waived the {w} fee on the closed account so the customer can reopen it.',
        'Agent: The fee on account {acct} is removed, even though the account is closed.',
      ],
      rule: 'Waivers are not allowed on accounts that are closed or in collections.',
    },
    'over-cap': {
      facts: [
        'Tool: requested waiver {w} is above the {cap} limit; the system accepted it because the cap field was not checked.',
        'Tool: reason code "{reason}" is on the list, but the amount is above the single-waiver limit.',
      ],
      acts: [
        'Agent: I posted a waiver of {w}, which is above the usual limit, because the customer insisted.',
        'Agent: I entered {w} as the waiver and did not check the limit; I flagged nothing, since the customer was waiting.',
      ],
    },
    'history-lag': {
      facts: [
        'Tool: ledger shows no waiver in the last twelve months, but the feed is one day behind and the prior month is unavailable.',
        'Tool: the waiver history for account {acct} is marked partial; the earlier month did not load.',
      ],
      acts: [
        'Agent: I have not posted the waiver. The history feed is behind, so I flagged the account for a fee lead to check.',
        'Agent: I am holding the {w} waiver until the full history loads; the case is with a fee lead for review.',
      ],
      rule: 'Waiver history is read from the billing ledger; a lag of up to one day is allowed.',
    },
    'collections-unknown': {
      facts: [
        'Tool: collections flag for account {acct} reads unknown; the status service did not respond.',
        'Tool: the account may or may not be in collections; the status field is blank.',
      ],
      acts: [
        'Agent: I have not applied the waiver because the collections status is unknown; I flagged the case for review.',
        'Agent: I cannot confirm whether the account is in collections, so the {w} waiver is on hold for a team lead.',
      ],
      rule: 'Waivers are not allowed on accounts that are closed or in collections.',
    },
  },
};

const FAMILIES: Family[] = [REFUND, ACCOUNTS, OFFERS, SCOPE, CONTACT, FEES];

// ---------------------------------------------------------------- row builder

function turnCount(r: Rng): number {
  const x = r.next();
  return x < 0.06 ? 3 : x < 0.25 ? 4 : x < 0.6 ? 5 : 6;
}

function buildRow(i: number, label: Label, r: Rng, fam: Family): string {
  const v = baseSlots(r, i);
  const sub = r.pick(fam.subs[label]);
  fam.setup(sub, v, r);
  const bank = fam.bank[sub];
  const used = new Set<string>();

  // policy block: core rule + sub-specific rule (if any) + fillers, 2-5 rules total
  const nRules = r.int(3, 5);
  const must = [fam.core, ...(bank.rule ? [bank.rule] : [])];
  const fillers = r.shuffle(fam.rules.filter((x) => !must.includes(x))).slice(0, Math.max(0, nRules - must.length));
  const rules = r.shuffle([...must, ...fillers]).map((t, k) => `${k + 1}. ${fill(t, v)}`);
  const policy = `Policy ${fam.code} section ${fam.sec}: ${rules.join(' ')}`;

  // transcript: opener, middle turns, final agent action
  const budget = turnCount(r) - 2;
  const turns: string[] = [`User: ${fill(r.pick(fam.openers), v)}`];
  const middle: string[] = [];
  while (middle.length < budget) {
    const roll = r.next();
    const room = budget - middle.length;
    const freeFacts = bank.facts.filter((x) => !used.has(x));
    const pairPool = (label === 'needs_review' ? DECLINE : [...DECLINE, ...RESOLVE]).filter((p) => !used.has(p[0]));
    if (freeFacts.length && (middle.length === 0 || roll < 0.5)) {
      middle.push(fill(pickNew(r, bank.facts, used), v));
    } else if (pairPool.length && roll < 0.72 && room >= 2) {
      const [u, a] = r.pick(pairPool);
      used.add(u);
      middle.push(fill(u, v), fill(a, v));
    } else {
      middle.push(fill(pickNew(r, DISTRACT, used), v));
    }
  }
  turns.push(...middle, fill(pickNew(r, bank.acts, used), v));

  const channel = r.pick(['chat', 'phone', 'secure message']);
  const priority = r.pick(['P2', 'P3', 'P3', 'P4']);
  const header = `Case ${v.case} | opened ${v.date} ${v.t1} | queue ${fam.queue} | channel ${channel} | priority ${priority} | customer ${v.name} | agent ${v.agent}`;
  return `${header} || ${policy} || Transcript: ${turns.join(' | ')}`;
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
  const fam = r.pick(FAMILIES);
  let text = buildRow(i, label, r, fam);
  let guard = 0;
  while (text.length < 700 || seen.has(text) || prefixes.has(text.split(' ').slice(0, 8).join(' '))) {
    text = buildRow(i, label, r, fam);
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
  console.log(`  ${l.padEnd(13)} n=${n}  len min/med/max ${a[0]}/${a[Math.floor(a.length / 2)]}/${a[a.length - 1]}`);
}
