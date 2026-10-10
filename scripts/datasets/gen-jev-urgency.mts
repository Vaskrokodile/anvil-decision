// Deterministic generator for public/datasets/jev-urgency.jsonl
// Label = how soon a case needs a human: low | medium | high | critical.
// Long multi-message support threads. Topic context is shared across levels,
// so the level has to come from the urgency signals, not from the subject matter.
// Run: node scripts/datasets/gen-jev-urgency.mts
// Only Node built-ins. Fictional names/companies only; example.com emails; 555-01xx phones.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../../public/datasets/jev-urgency.jsonl');
const SEED = 0x7a11c0de;
const PER_CLASS = 625; // 4 x 625 = 2,500 rows
const MIN_CHARS = 500;
const MAX_CHARS = 1400;
const BANK_MIN = 60;

type Label = 'low' | 'medium' | 'high' | 'critical';
type Role = 'customer' | 'agent' | 'auto' | 'note';
type Env = Record<string, string>;

const LABELS: Label[] = ['low', 'medium', 'high', 'critical'];

// ---------- seeded RNG helpers ----------
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(SEED);
const int = (lo: number, hi: number): number => lo + Math.floor(rnd() * (hi - lo + 1));
const pad2 = (n: number): string => String(n).padStart(2, '0');
function pick<T>(xs: readonly T[]): T {
  return xs[Math.floor(rnd() * xs.length)];
}
function weighted<T>(xs: readonly T[], w: readonly number[]): T {
  let r = rnd() * w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < xs.length; i++) {
    r -= w[i];
    if (r < 0) return xs[i];
  }
  return xs[xs.length - 1];
}
function sampleN<T>(xs: readonly T[], n: number): T[] {
  const pool = [...xs];
  const out: T[] = [];
  for (let i = 0; i < n && pool.length > 0; i++) out.push(pool.splice(Math.floor(rnd() * pool.length), 1)[0]);
  return out;
}
function shuffle<T>(xs: T[]): void {
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }
}
function shuffleCopy<T>(xs: readonly T[]): T[] {
  const c = [...xs];
  shuffle(c);
  return c;
}
function gauss(): number {
  const u = rnd() || 1e-9;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
}
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

// ---------- slot banks ----------
const FIRST = ['Dana', 'Marcus', 'Priya', 'Tomasz', 'Lena', 'Omar', 'Grace', 'Ilse', 'Rafael', 'Mei', 'Kofi', 'Hannah', 'Viktor', 'Aisha', 'Ben', 'Sofia', 'Yusuf', 'Clara', 'Diego', 'Nadia', 'Ruth', 'Felix', 'Imani', 'Jonah', 'Leticia', 'Anton', 'Wren', 'Caleb', 'Noor', 'Pablo', 'Hedda', 'Sam', 'Oksana', 'Teodor', 'Maya', 'Elias', 'Farah', 'Gideon', 'Ines', 'Kenji', 'Lara', 'Mateo', 'Odile', 'Pia', 'Quinn', 'Rosa', 'Silas', 'Tove', 'Uma', 'Vera', 'Wes', 'Yara', 'Zane', 'Adela', 'Bruno', 'Corin', 'Delia', 'Emeka', 'Freya', 'Galen', 'Helga'];
const LAST = ['Whitfield', 'Okafor', 'Lindqvist', 'Marchetti', 'Haddad', 'Brennan', 'Kowalczyk', 'Nakamura', 'Osei', 'Fairbanks', 'Delacroix', 'Ahmadi', 'Thorsen', 'Quintero', 'Reyes', 'Dunmore', 'Halvorsen', 'Mbeki', 'Petrakis', 'Sorensen', 'Villanueva', 'Chandler', 'Ruiz', 'Abernathy', 'Kaur', 'Lehtonen', 'Oyelaran', 'Brandt', 'Castellano', 'Ferreira', 'Gallagher', 'Hollis', 'Ivanova', 'Jastrzebski', 'Kessler', 'Lombardi', 'Moreau', 'Novak', 'Pemberton', 'Rourke', 'Szabo', 'Tanaka', 'Underwood', 'Varga', 'Winslow', 'Yilmaz', 'Zielinski', 'Ashdown', 'Beaumont', 'Crawley'];
const AGENT = ['Renata', 'Jun', 'Callum', 'Esme', 'Dmitri', 'Ayo', 'Sigrid', 'Theo', 'Marisol', 'Ravi', 'Keiko', 'Bastien'];
const CO_A = ['Harbor Lane', 'Northgate', 'Cobalt Field', 'Marrow Creek', 'Lattice Row', 'Quill and Pine', 'Tallow Ridge', 'Ember Park', 'Gilded Reed', 'Foxglove', 'Juniper Yard', 'Stonebridge', 'Copperleaf'];
const CO_B = ['Outfitters', 'Print Works', 'Dental Group', 'Cycle Co.', 'Bakehouse', 'Hardware', 'Studio', 'Analytics', 'Pet Supply', 'Garden Supply', 'Logistics', 'Labs', 'Textiles', 'Coffee Roasters', 'Clinic Partners', 'Builders'];
const PRODUCTS = ['Kestrel Tablet', 'Relay Desk', 'LedgerLoop', 'Pinecast Sync', 'Harbor Reports', 'Fieldnote Pro', 'Quarry Analytics', 'Mosaic CRM', 'Tidewell Scheduler', 'Cinder Forms'];
const CITIES = ['Columbus', 'Tacoma', 'Lyon', 'Leeds', 'Rotterdam', 'Dayton', 'Bristol', 'Ghent', 'Tampa', 'Eugene', 'Malmo', 'Brno', 'Porto', 'Kingston', 'Aarhus'];
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

function money(cents: number): string {
  const d = Math.floor(cents / 100);
  const c = cents % 100;
  return `$${String(d).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}.${pad2(c)}`;
}

function makeEnv(): Env {
  const cust = pick(FIRST);
  const custLast = pick(LAST);
  return {
    cust,
    custLast,
    agent: pick(AGENT),
    cc: pick(FIRST.filter((f) => f !== cust)),
    email: `${cust.toLowerCase()}.${custLast.toLowerCase()}@example.com`,
    co: `${pick(CO_A)} ${pick(CO_B)}`,
    tkt: `TKT-${int(10000, 99999)}`,
    order: `ORD-${int(100000, 999999)}`,
    acct: `AC-${int(1000, 9999)}`,
    amt: money(int(4000, 980000)),
    amt2: money(int(4000, 980000)),
    date: `${pick(MONTHS)} ${int(1, 28)}`,
    date2: `${pick(MONTHS)} ${int(1, 28)}`,
    prod: pick(PRODUCTS),
    city: pick(CITIES),
    phone: `555-01${pad2(int(10, 99))}`,
    wd: pick(WEEKDAYS),
    n: String(int(2, 40)),
    pct: String(int(3, 38)),
  };
}

function fill(s: string, env: Env): string {
  return s.replace(/\{(\w+)\}/g, (_, k: string) => {
    const v = env[k];
    if (v === undefined) throw new Error(`unknown slot {${k}} in: ${s}`);
    return v;
  });
}

// ---------- urgency signal banks: >= 60 distinct sentences per level ----------
const SIGNALS: Record<Label, string[]> = {
  critical: [
    'Money is leaving our operating account right now; there are three outgoing transfers to unknown payees in the last ten minutes.',
    'Our entire checkout is down for every customer in the EU, and we are losing sales by the minute.',
    'Someone is logged into the admin console right now and deleting records as I write this.',
    'A court filing deadline is today at 5 PM, and we need the export of those records before then.',
    'The payroll run is processing right now with a wrong bank routing number, and the batch is about to be sent.',
    'We have a live data breach: an export of customer emails is being downloaded from an unknown IP address at this moment.',
    'The outage is hitting every tenant on the platform, and our support line is full of people who cannot work.',
    'A patient on our care programme may miss a dose because the medication portal is showing the wrong dosage right now.',
    'Fraudulent card charges are appearing every few minutes on our merchant account, and the daily limit is still open.',
    'Our smart lock vendor says remote unlocks are firing for unauthorised users at the warehouse right now.',
    "The regulator's response is due by midnight tonight, and our submission portal keeps returning internal errors.",
    'Ransomware has encrypted two file servers, and the backup job is still writing into them.',
    'We are watching a wire transfer of {amt} leave the account as we speak; please freeze it.',
    'Every order placed in the last hour has been charged twice, and it is still happening.',
    'A threat of violence has been posted about our office today, and staff are still inside the building.',
    'The payment processor has frozen our payouts, and our drivers are paid at 6 PM tonight, which we cannot cover.',
    'The emergency alert feed has stopped sending to 40,000 subscribers.',
    'An attacker is using a stolen admin token to change bank details on supplier invoices right now.',
    "The records export is showing other patients' data on the front-desk screens at this moment.",
    'Our production database is replicating corrupted rows to every region as we speak.',
    "Unauthorised transactions are appearing on my card every few minutes, and I cannot block it in the app.",
    "Customers can see each other's order history on the storefront this minute, and we have had three complaints already.",
    'The gas leak alarm integration went silent in one of our buildings, and the site contact has not acknowledged it.',
    'A pending payment batch of {amt} to a new account is scheduled to release in twenty minutes.',
    'The legal hold deadline is tonight, and the retention job is purging deleted emails every hour.',
    'The fleet tracker lost connection to trucks carrying hazardous material, and the drivers are not responding.',
    'A group is posting our staff home addresses in real time, and two employees have already been contacted.',
    'Every call to our payments service is failing, and merchants are seeing failed charges at the till right now.',
    'The attacker has our VPN credentials and is moving across the network this morning.',
    'Stop the transfers: a scammer has convinced our controller to send {amt} to a new account, and it leaves at noon.',
    'We are in the middle of a live outage on the voting-day system, and users cannot submit their forms.',
    'The patient portal is showing the wrong allergy list right now, and clinicians are relying on it.',
    'The subscription billing job charges customers a second time on each retry, and it runs again at the top of the hour.',
    'Our entire site has been defaced by an attacker who still has write access.',
    'Someone changed the payout destination on our account, and our next payout of {amt} leaves at 4 PM today.',
    'The emergency dispatch integration is down, so calls are being logged on paper across the county.',
    'We are losing a container of perishable product every hour because the temperature alarm has no one listening.',
    'Customers in three countries are being charged in the wrong currency right now, and the rate is rising.',
    'A court-ordered deletion must finish before close of business today, and the job has failed twice.',
    'The compromised plugin is injecting a redirect into every checkout page at this moment.',
    'A former employee is still logged in and downloading the client list as I type.',
    'Our billing system overcharged 6,000 customers in the last hour, and the cause is still active.',
    "The children's app is showing inappropriate content to users right now, and parents have reported it.",
    'The fraud ring is cashing out gift cards faster than we can block them, about {amt} in the last hour.',
    'A rogue admin has locked us out of our own tenant and is changing the MFA policy for every user.',
    'We are under an active denial-of-service attack, and the storefront has been unreachable for twenty minutes.',
    'The insurance submission window closes at 11:59 tonight, and our portal rejects every upload with an internal error.',
    'A ransom note was left on the shared drive this morning, and files are still being renamed as I write.',
    'The building access system has opened every door in the warehouse since 2 AM, and the security desk cannot lock them.',
    'Our app is sending the wrong medication reminders to users this evening, and it is still live.',
    'The pharmacy terminal is charging the full price twice on each sale at the counter in front of customers.',
    'A supplier is actively draining our credit line with unauthorised orders, and the limit resets at midnight.',
    'Our backups were deleted an hour ago, and the attacker still has the admin session open.',
    'An unknown device is streaming our camera feeds to an external address at this moment.',
    'The payroll batch went out twice this morning, and the second copy is clearing into staff accounts as we speak.',
    'Our on-call engineer is unreachable, and the production cluster is refusing every write request right now.',
    'We see a fund transfer loop: the same {amt} is being sent back and forth between two accounts every minute.',
    'The alarm system at our clinic is armed, but the sensors report a break-in in progress.',
    'The phishing campaign is harvesting staff passwords right now, and three people have already entered theirs.',
    'Our control panel shows water pressure dropping in the east wing, and the pump alerts are not reaching anyone.',
    'The gift card codes printed this morning are being redeemed on the site faster than we can cancel them.',
    'The compliance deadline passes at midnight, and the attestation upload is failing for the whole legal team.',
    'Our emergency contact list was exposed publicly ten minutes ago, and the page is still live.',
    'The vendor portal is paying out duplicate invoices every few minutes, and the total is already {amt}.',
  ],
  high: [
    'The overcharge of {amt} needs to be reversed before our payment run on the 20th, or the supplier will hold the order.',
    'We were billed {amt} in error on the renewal, and the card will be charged again on Friday unless this is corrected.',
    "This is the third time I've written about the blocked import, and our go-live is on Thursday.",
    'Our payroll file must be corrected before the Friday cutoff, or nine people will be paid late.',
    'We cannot publish the catalogue until the feed is fixed, and the seasonal launch is on the 14th.',
    'Our contract requires a response within four hours, and we are already past two hours on this ticket.',
    "I'm following up again on {tkt}; nobody has replied since {date}, and the renewal is due Monday.",
    'The shipment has to clear customs by Friday, or we lose the event slot and the deposit.',
    'We cannot close the month-end books until the reconciliation report runs, and the audit starts Monday.',
    'Our store opens Saturday, and the point-of-sale system is refusing card payments on two terminals.',
    'Please prioritise this; the client demo is tomorrow morning and the dashboard is empty.',
    'This has blocked three of our staff for two days, and their deliverables are due at the end of the week.',
    'The refund has not arrived after the promised three working days, and I need it to cover rent this month.',
    'We will miss the contract signing on {date} if the signed PDF cannot be exported by tomorrow.',
    'The contract requires the quarterly report by Thursday, and the data export keeps timing out.',
    'Two of our client accounts have been overcharged, and they want corrections before their month closes on the 30th.',
    'Our production release is frozen until the API returns valid responses, which puts the launch at risk this week.',
    "I've called twice and written three times about the duplicate invoices; the supplier is threatening to stop deliveries.",
    'Inventory counts are wrong for the weekend sale, and the stores need corrected stock by Friday morning.',
    'The field crew cannot clock in, and the job is being billed to the client; we need this unlocked within the day.',
    'The renewal date is in four days, and the discount approval is still sitting on your side.',
    'Onboarding is stalled for twelve accounts, and each day adds to the backlog we owe them.',
    "I'd appreciate a fix before the holiday weekend; without it, dispatch will be working from spreadsheets.",
    'We have a board presentation on Monday, and the revenue dashboard still shows last quarter numbers.',
    'The accreditation review is at the end of the week, and the audit trail export is still missing entries.',
    'The tax filing is due on the 15th, and we cannot submit until the corrected statement is reissued.',
    'This is the second week the parcel has been stuck, and the event it was meant for is this Saturday.',
    'The integration has been failing since Monday, and we are behind on about 2,000 orders that must sync before the weekend.',
    "Please escalate: our contract's uptime credit is at risk if this outage runs past the weekend.",
    "We're two follow-ups in without a reply, and the vendor deadline is Wednesday at noon.",
    'The duplicate charges will push two customers over their credit limits by Friday if we do not reverse them.',
    'I need the corrected invoice before the payment run on the 20th, or the supplier will put us on hold.',
    'Staff cannot reach the scheduling app, and the weekend shift roster has to go out by Thursday.',
    'We are missing the permit paperwork the city needs by Wednesday, and the crew is booked for Thursday.',
    'This has affected our ability to take bookings for the weekend; it started on {date} and has not moved.',
    "Our clinic cannot send appointment reminders, and the backlog grows daily; we need it working before Monday's clinic.",
    'Can we get a response today? I have been waiting since {date}, and the decision is needed by the end of the week.',
    'Our licence expires in three days, and the activation key still returns invalid.',
    'We have been without the reporting feature for four days, and the regional managers ask every morning.',
    'The lab results feed is two days behind, and the clinical team needs it before rounds on Monday.',
    'The seasonal sale starts Friday, and the discount codes do not validate, so customers are abandoning carts.',
    "I'm writing for the third time: the replacement order must land before the event on the 12th.",
    'We signed the lease for the new site starting Monday, and the account setup has not finished.',
    'Please expedite the approval; our grant funds are only valid until the end of this week.',
    "We cannot ship the rest of the order until the export clears, and the buyer's penalty clause starts on Friday.",
    'The client has asked twice for corrected numbers, and their board meets Thursday.',
    "We've been blocked on this since {date}, and every day it stays broken pushes another project back.",
    'The payroll-linked report is three days late, and our finance lead wants it on his desk by Wednesday.',
    'The integration partner has missed two agreed fix dates, and our contract penalties begin on the first of the month.',
    'Our field technicians have no access to job sheets, and two jobs are already overdue this week.',
    "I'm the third person on our team to chase this, and we need a status by the end of the day.",
    'The new hires cannot log their hours, and payroll closes Thursday without them.',
    'We have a customer escalation open for a week, and their executive team is asking why nothing has moved.',
    'Our training course starts Monday with 60 attendees, and the course materials will not load for them.',
    'Our inventory sync has been stale for three days, and wholesale buyers are cancelling orders.',
    'The vendor has put our account on hold until the disputed invoice is resolved, and deliveries stop after this week.',
    'This is blocking our quarterly close, and the auditor has already asked us twice for the schedule.',
    "The retailer's deadline for the labels is Wednesday, and the printing template keeps failing.",
    'The audit log export failed again, and the compliance team needs it before the review meeting on Friday.',
    'We are waiting on the corrected certificate so our new location can open its doors this weekend.',
    'The coach bookings for the team trip are due to be paid by the end of the week, but the invoice is wrong.',
    'Our customer portal has been down for the regional sales team for two days, and they cannot close deals.',
    'Please fix the SSO mapping before the new contractors start on Monday; they cannot get in.',
    'We have a product recall notice due to the regulator on Friday, and the batch report is still incomplete.',
    'The emergency grant report is due this week, and the funder will withhold the next tranche if it is late.',
    "I've now followed up on this four times, and the clinic manager wants a firm answer before the weekend.",
  ],
  medium: [
    'Could you confirm the {amt} credit will be applied to the next invoice when you have a moment this week?',
    'The refund of {amt} is with your team; a reply this week about the timing would be appreciated.',
    'Could you let me know by Friday whether the change can be applied before the next billing cycle?',
    "It's a bit annoying but not blocking us; we would like a fix sometime in the next two weeks.",
    'When you have a moment this week, could you look at why the monthly report is off by a few percent?',
    'We need the updated quote before our budget review on the 21st, so early next week would be ideal.',
    'The export works, but the column order changed from last month and breaks our spreadsheet macro.',
    'Our team can work around it for now, but we would like a permanent fix within the next fortnight.',
    'The refund is taking longer than usual; a reply this week would be appreciated.',
    'We are hoping to have this sorted before the quarterly review on {date}, which is about ten days away.',
    'The tracking is a little delayed, so I would like an update by the end of the week.',
    'This is a minor inconvenience for the team, but it has been happening most mornings.',
    'Please confirm the new address by Wednesday so the vendor can update their paperwork next week.',
    "I'd like to resolve this before the end of the month, when we close our books.",
    'It would help to get this corrected in the next week or so, before the new hires start.',
    'The reminder emails go out twice, which is clumsy but not a big deal; a fix in a couple of weeks would be fine.',
    'Our renewal is on the 28th, so I would like the discount confirmed sometime this week.',
    'The dashboard loads slowly in the afternoons. Not critical, but it is getting in the way.',
    'Could someone check the permission settings before our onboarding session next Tuesday?',
    'We need the signed amendment back within two weeks to keep our schedule.',
    'The label looks off, though the parcel still reached us, so this can wait until next week.',
    'Please review the revised statement when you get a chance this week; we will pay after we see it.',
    'The sync delay is annoying, and our team is working around it, but a timeline by Friday would help.',
    'We would like to know this week whether the feature will be available for our next planning cycle.',
    "It's not urgent, but we would appreciate an answer before the first of next month.",
    'The duplicate entry in the report is a nuisance; a correction in the next few days would be ideal.',
    'Could you look into the reset issue this week? The staff member can still work, but it is slow.',
    'We have a vendor review on the 10th and would like the audit export in hand before then.',
    'The invoice number format changed, which confuses our clerks; I would like a clarification by Thursday.',
    "I'm hoping for a reply within the week so I can plan the migration for the following sprint.",
    'The app feels clunky on older phones, and our field staff have been complaining for a while.',
    'The plan change needs approval by the 15th; please let me know whether that is possible.',
    'A summary by Friday would help our planning, but I am not blocked on it.',
    'The warehouse can receive the pallet next Tuesday, so the delivery issue should be settled before then.',
    'We are missing one field in the import template, which means some manual cleanup each week.',
    'Notifications are going to the wrong team inbox, which is confusing people; I would like it sorted in a couple of weeks.',
    'I would like this handled within the next ten days; a meeting on the 24th depends on it.',
    'This causes extra work on our side every Monday, so a fix when there is capacity would be appreciated.',
    'Could you tell us this week whether the parts can be replaced under warranty? Our installer is booked for the 22nd.',
    'The licence count is off by two seats, which we can live with for a couple of weeks.',
    'The summary email arrives about an hour late; it is not critical, but it happens every day.',
    'The photos on the product page are the wrong size; please fix them when someone is free in the next couple of weeks.',
    'Our internal audit is in three weeks, so getting this resolved in the next week or so would be good.',
    'I need to confirm the invoice before we close next week, so an update by Wednesday would help.',
    'Please send the updated price list by the middle of next week so we can plan our promotion.',
    "The training session is in ten days, and I'd like the sandbox access confirmed before then.",
    'The approval form is waiting for a reply from your side; I would like a decision before the board pack goes out on the 18th.',
    'A colleague is out next week, so please send the answer to me directly by Thursday.',
    'Could you confirm this week that the upgrade path will work for our version? We want to schedule it for the 2nd.',
    "The renewal reminder went to the old address, and I would like the record corrected before next month's cycle.",
    'The team would like an overview of the new reporting options within two weeks, if possible.',
    'We have a small backlog of duplicate tickets, and clearing them by the end of next week would be great.',
    'The tax category on the form looks wrong; can someone check it before the filing next Friday?',
    'Our kickoff meeting is on the 9th, and I would like the project template in hand by then.',
    'The newsletter sign-up form double-submits; it is a nuisance that we would like fixed within the fortnight.',
    'Please let us know by Wednesday whether the training credits can roll over to next quarter.',
    'The contract clause needs a clarification within the week so that legal can sign off on the renewal.',
    "We'd like a status note by Friday on the three open items from the call on {date}.",
    'The workflow emails are slow to arrive, which is not critical, but we need them reliable before the pilot in two weeks.',
    'The access request for the new analyst is pending, and I would like it approved within a week.',
    'The vendor has asked for the revised purchase order by the 19th; could someone check our draft this week?',
    'The glitch with the calendar invite is minor, but we would like it resolved before the next planning session.',
    'Our partner asked for an update on the migration timeline before their sprint planning next week.',
    'The receipt template prints the old address; a fix in the next ten days would be fine.',
    'We need someone to confirm the holiday closure dates for our schedule before the 25th.',
    "I'd appreciate a reply sometime this week about whether the feature is on the roadmap for Q3.",
  ],
  low: [
    'Just curious how the {amt} plan compares with the Business tier; no decision is needed on our side.',
    'For our records, what was the list price of the {amt} bundle last year? Whenever you get to it.',
    'Whenever you get a chance, could you tell me whether the dark theme is on the roadmap?',
    'No rush at all; I was just curious how the export handles very old date formats.',
    'Feedback for the team: the new menu layout is nicer, though the icons are a bit small.',
    'Not urgent, but is there a guide to the reporting shortcuts you would recommend?',
    "I'd love to know if there's a way to rename a folder without breaking links, eventually.",
    'Just a general question about how long the archive keeps old attachments; no specific deadline.',
    'This is more of a suggestion: a weekly digest option would be nice someday.',
    "Whenever someone has a moment, I'd like to hear about the best way to tag our contacts.",
    'No problem if it takes a while; I just wondered whether the mobile app supports landscape mode.',
    'Thanks for the great webinar last month. Would you consider recording the next one as well?',
    "There's no hurry, but could you share the link to the API changelog when it's convenient?",
    "I'm writing to say that the onboarding video was helpful, and I'd like to suggest subtitles.",
    "Whenever it suits you, I'd be grateful for a quick overview of what the reporting module includes.",
    'Just curious whether the training materials are available in Spanish. No deadline on this.',
    'Could you add a dark-mode option to the list of feature requests? Not time-sensitive at all.',
    "I don't need anything right now; I'm just exploring whether the product fits our future plans.",
    'Low priority, but the help article about permissions could use an example with a table.',
    'Please pass along our compliments to the support agent who helped us last week.',
    'Eventually it would be nice to customise the dashboard colours, but nothing is waiting on it.',
    "I'm sending this for the record: the button label is a little confusing, though it works fine.",
    'When there is a spare moment, could someone explain the difference between the two export formats?',
    'The newsletter had a broken link in the footer; obviously not a big deal, just letting you know.',
    "We'd like to hear about upcoming changes to the connector catalogue whenever they're announced.",
    'No need to reply quickly; I just want to know if the team has a public roadmap.',
    'Would you mind sharing the average onboarding time for new teams when you have a chance?',
    'Just a thought for the product team: keyboard shortcuts for filters would be handy.',
    'I know this is not a priority for you, but a list of supported file types would be useful one day.',
    "Whenever convenient, could you send the brochure? We're not looking to buy anything yet.",
    'Our team enjoys the monthly release notes; keep them coming whenever you publish.',
    "I'm happy to wait; there's no deadline, and I'd just like a general answer about data residency options.",
    'It would be lovely to have a sample template for quarterly reviews, sometime when you have time.',
    "This isn't blocking anything; I just noticed the help centre search doesn't find plurals.",
    "Nothing is due on this. I'm curious whether anyone on the team has used the Kestrel integration before.",
    'Thanks, that answers my question. One more curiosity: is the logo export available as SVG?',
    'Please note this is just a comment about the training video narration; take it whenever you have capacity.',
    "I wondered if you ever plan to add a language option for the summary emails, but there's no timeline I need.",
    'The feedback form asks for suggestions; mine is that the pricing page could show annual savings more clearly.',
    "Whenever you are free, I'd like to talk about feature ideas; I'm in no rush at all.",
    'Our office would like a printed copy of the user guide at some point; no date needed.',
    'Is there a community forum where customers share templates? Just browsing, nothing to resolve.',
    "I'm not sure a feature like this exists, so this is simply a curiosity question for later.",
    "Please take your time with this; I'd be happy with a reply in a month or whenever.",
    'Could you share the names of the webinar speakers when you have a moment? No action needed on our side.',
    'The release notes were clear. Would you consider adding a glossary at some point?',
    'Idle question: do you plan to support more languages for the help centre?',
    "We're thinking about next year's budget, but nothing here needs an answer soon.",
    "I'd like a copy of the case study you published last spring, whenever it's convenient.",
    'The product tour could be shorter; that is just my opinion, no action needed.',
    'Could you tell me whether there are any plans for a mobile widget? No timeline needed.',
    'Out of curiosity, what sort of customers typically use the advanced filters?',
    'Nothing is broken; I wanted to say the onboarding checklist made our first week much smoother.',
    'Would it be possible to receive the newsletter in plain text as well? Whenever suits you.',
    "We're still deciding whether to adopt this, so there's no rush on any answer.",
    'A brief note of thanks to the team for the quick fix last month; no reply needed.',
    "The icon set looks a bit dated, but that's cosmetic and can be considered for a future update.",
    "I'd be interested in a beta invite someday; no rush and no specific use case yet.",
    'Please add a note to the knowledge base about the old export defaults when you get around to it.',
    'Could you share the list of integration partners for our records? Whenever is fine.',
    'I would like to understand how the data is categorised, purely for general interest at this stage.',
    'Our intern is writing a report on tools like yours; a general overview would help, no deadline.',
    'The sample dataset was helpful; I would love more examples in different languages at some point.',
    'We do not need this resolved, but the wording of the error on the help page is a bit vague.',
    "It's a small curiosity: why does the dashboard use a different colour for archived items?",
    'Thanks for the quick reply. No further action needed; I just wanted to say thank you.',
  ],
};

// Negated urgency words: shared by every level (a passing mention that is explicitly not a driver).
const CONF_NEG = [
  "I know 'urgent' appeared in the subject line of my first email, but nothing is blocked and there is no deadline.",
  "People here keep calling it critical, but honestly it's just a question with no due date.",
  'It might look like a deadline because we mention the event, but the event itself is not affected at all.',
  'My manager asked me to flag it, but nobody has told me there is a date attached or any impact yet.',
  "The word 'blocking' came up in an internal chat, but on reflection nothing is actually blocked.",
  'Our renewal is next year, so there is no real timing pressure from that side.',
  'The client mentioned a deadline last month, but it has moved to next year and nothing hangs on this answer.',
  "Earlier I wrote 'emergency' by mistake; it was a typo for a different question and that matter is closed.",
  "That 'escalation' label was added automatically by the form, and I'm not asking for anything to be escalated.",
  "Nothing here is on fire, despite what the subject says; I just picked a loud subject line.",
  "The 'P1' tag on the old ticket was a mistake from the intake form and has since been removed.",
];
// Low-level side remarks: a weaker, passing note inside a more urgent thread.
const CONF_LOW = [
  "Separately, I'd like a copy of the newsletter template whenever convenient, with no deadline on that.",
  "Unrelated, but the help centre search doesn't find plurals; that can wait.",
  'Not urgent in any way, but a dark-mode option would be a nice future feature.',
  "On a side note, I'd love a tour of the reporting module someday; nothing is due.",
  'Also, the footer link is slightly off; that can go on the list for later.',
  "The feedback survey can wait until next quarter; it isn't part of this problem.",
  "Totally separate: the logo file in the brand kit is a little blurry, whenever you get to it.",
  "Also, I'm curious about the public roadmap, but no reply is needed on that point.",
  'Side question that can be answered at leisure: do you have a template for quarterly letters?',
  'Not related to the issue, but the webinar recording was a little quiet; no action needed.',
];

// Neutral topical context: the same pool for every level.
const CONTEXT = [
  'The order in question is for twelve units of the Kestrel tablet bundle.',
  "We're on the Business plan with twenty-two seats and two admin users.",
  'The export is set to run monthly and lands in a shared finance folder.',
  'Our contact at the vendor is moving offices this month.',
  'The integration connects our order system to the warehouse management tool.',
  'The problem first appeared after we updated the browser extension on the reception computers.',
  'Most of the team works from {city} on weekdays, with a few remote staff in other time zones.',
  'About 3,400 records sit in the affected table, most of them created in the last year.',
  'The duplicate entries are in the vendor master list, which feeds the purchase orders.',
  'The renewal paperwork was signed by our operations director last quarter.',
  'We use the mobile app mainly for stock counts on the shop floor.',
  'Schedule B of the contract lists the support hours and the escalation contacts.',
  'The spring catalogue is printed in {city}, and the file goes to the printer each Friday.',
  'Our reports go to three regional managers and an external auditor.',
  'Our integration partner built the connector two years ago and has since moved on to other work.',
  'The account was created by a previous admin, so some details may be out of date.',
  'We manage about 140 products across four storefronts.',
  'The invoice references purchase order {order} from the first quarter.',
  'Our finance team uses the summary sheet every month to reconcile the regional totals.',
  'The sandbox environment mirrors production, minus the customer data.',
  'Our staff rotate through the support desk, so the contact person may change.',
  'We have three sites: the head office in {city}, a satellite office, and the warehouse.',
  'The dashboard is shared with about thirty people across sales and operations.',
  'The team uses the Relay Desk workspace for planning and the Harbor Reports module for reporting.',
  'Our onboarding was run by a consultant who has since left the project.',
  'We keep attachments for seven years, mostly contracts and signed forms.',
  'We typically review the reports on the second Monday of each month.',
  'The admin console has four roles: owner, admin, editor and viewer.',
  'We run weekly syncs at the start of the week and a full refresh at the end of the month.',
  'The mailbox forwards copies to a shared inbox that three people check.',
  'Our staff use the desktop client on laptops that are about three years old.',
  'The Kestrel display in the lobby still runs an old slideshow.',
  'We have a long-standing arrangement with a regional courier for the local runs.',
  'The team uses custom fields for region, priority tier and account owner.',
  'Our contract with the reseller was signed in the spring.',
  'Each store has its own user list, which is synced nightly from the head office.',
  'The reporting template has eleven tabs and a pivot table on the front.',
  'We have been a customer for about four years and have upgraded twice.',
  'The product catalogue includes both seasonal and permanent lines.',
  'The customer service team handles about 300 messages a week across channels.',
];

// Shared neutral material (same role as the routing generator's distractor bank).
const DISTRACTORS = [
  "I'm writing from our office in {city}, where the coffee machine has been broken all week.",
  'We moved to a new floor in the spring, so some of the contact details I have may be out of date.',
  'I can send screenshots if they would help, though I have not attached anything yet.',
  '{cc} is copied on this message in case I am away from my desk.',
  'The weather here has been strange, with rain every afternoon followed by sudden sunshine.',
  'We are finishing our offsite soon, so my calendar is packed.',
  "I'll be on a train for part of the afternoon, so replies may be slow.",
  'It was helpful to see the steps laid out clearly in your last message.',
  'If a call is easier, my number is {phone}.',
  "I've been with the company for about six years, so I know most of the internal tools.",
  'The team photo from the summer retreat is still pinned in our shared channel.',
  'Please ignore the automatic footer on my earlier message; it came from our mail tool.',
  'Our office dog has been a bit of a distraction, but in a good way.',
  'I have a long list of other things on my desk, so short answers are fine.',
  'We are also planning a trip to {city} in the autumn, so any hotel tips would be welcome.',
  'I finally finished the novel my sister recommended, and it was far better than I expected.',
  'The kids have started swimming lessons, so our evenings are very busy at the moment.',
  'I read your blog post about spreadsheet folders and it made me reorganise ours.',
  'The Wi-Fi in the meeting room drops every half hour, which is a separate annoyance.',
  'I am not sure who else handles this kind of thing, so I am writing to the general mailbox.',
  'We are a small team of nine people, so everyone wears a few different hats.',
  'I am out of the office on {wd} afternoons, so a reply to the morning address works best.',
  'Last year we changed our logo, and a few old templates still show the previous version.',
  'The lunch for the all-hands arrived late, which left everyone grumpy this morning.',
  'The printer on the second floor jams whenever someone uses thick card stock.',
  'I have copied the earlier exchange below so you have the full context.',
  'Below is the message I sent previously, for reference.',
  'Forwarded from my inbox, unedited, for your records.',
  'On {date} you wrote that the team was reviewing the file, so I wanted to check in.',
  'I hope your week is going well so far.',
  'Apologies for the typos; I am typing this on my phone during a commute.',
  'We have no other topics to raise at the moment.',
  'The company newsletter this month featured a profile of the warehouse crew, which was a nice read.',
  'I am happy to jump on a call if that is quicker, though email works too.',
  'Our office is closed on {wd} for a holiday, so please keep that in mind.',
  'The parking rules near our building changed recently and they are a bit confusing.',
  'The sandwich shop downstairs changed its opening hours, which has made lunch harder.',
  'My manager, {agent}, asked me to forward this so that it is on the record.',
  'Our shared drive holds about four thousand files, mostly old drafts nobody opens.',
  'I am on the early shift this week, which explains the odd timestamps on my messages.',
  'The spreadsheet I sent last time had a few formatting quirks, sorry about that.',
  'This is my first time writing to your team, so please bear with me on the details.',
  "The sign on our front door still says 'closed for renovation', which confuses visitors.",
  'A former colleague used to work at a company like yours, so I heard good things about your team.',
  'We hosted a school group for a tour of the facility recently, which was a lot of fun.',
  'Thanks to whoever wrote the help article on exports; it was clear and easy to follow.',
  'My background is in accounting, so I am comfortable with detail but not with the jargon.',
  'The conference in {city} was great, and I collected far too many stickers for the kids.',
  'We have had a busy few weeks with seasonal hiring, so I have been juggling a lot.',
  'Please keep this thread in one place, because my colleagues will be checking it too.',
  "I'm copying the general inbox as well, in case it is monitored more often than mine.",
  'The office coffee club has a new blend this month, which is more interesting than my inbox.',
  'I started a new hobby, pottery, and my first attempts have been wobbly.',
  'There was roadworks outside the office, which blocked the bus this morning.',
  'The Kestrel display in our lobby has been showing an old slideshow for a month.',
  'I am in the field for a few days, so I will read messages in batches.',
  'Our office has a new recycling bin that nobody quite understands yet.',
  'I appreciate your patience while I gathered all of these details together.',
  'The team lunch on Thursday was a good excuse to try the new noodle place.',
  'A cat walks across my keyboard during calls, so apologies for any stray characters.',
  'Our analyst {cc} has been on this for a while and can add more if needed.',
  "There's a small typo in the subject line of my earlier email, but the body is correct.",
  'The sun came out just as I started writing, which is a nice change for once.',
  'Thanks again for your help; I appreciate the time your team puts into these emails.',
  'Our customer survey in March had a high response rate, which we were proud of.',
  'We are a family-run business that started in a garage about twenty years ago.',
  'I am forwarding this to a second inbox so the note stays with our internal history.',
  'The company picnic photos are in the shared album if anyone wants to see them.',
  'We used a different tool before, and the migration took a while to settle down.',
  'I am not sure this is the right form, so I have written it as a plain email.',
  'The best contact on our side is the general mailbox at {email}, though I am also reachable directly.',
  'Our head of operations is copied because she signed off on the original arrangement.',
  'We have used the product for a little over a year, and it has mostly been a good experience.',
  'The last call dropped twice, but otherwise the conversation went well.',
  'I have tried to keep this short, so I left out the background on our old system.',
  'Thanks for keeping the thread tidy; it makes the history easy to follow.',
  'Attached is a photo of our desk setup, unrelated to this request but a fun one.',
  'The footer below is from our mail system, not from me personally.',
  'Our building has a new visitor badge system that takes about a minute to print.',
  'Our team lead likes to read these threads on the train, so the emails are kept brief.',
  'I have been reading about the new city cycle lanes, and they look much safer than before.',
  'Thanks for the quick acknowledgement earlier; it set my mind at ease for the day.',
];
const GREETINGS = ['Hi team.', 'Hello.', 'Hi support.', 'Good morning.', 'Dear support team.', 'Hi there.', 'Hello again.', 'Hey folks!', 'Hi, sorry for the long message.', 'Hello, thanks for taking this on.', 'Morning.'];
const SIGNOFFS = ['Thanks, {cust}.', 'Many thanks, {cust} {custLast}', 'Regards, {cust} {custLast}', 'Cheers, {cust}', 'Best, {cust}', 'Kind regards, {cust} {custLast}, {co}', 'Thank you, {cust}. My number is {phone}.'];
const AGENT_ACKS = [
  'Thanks for the detail, that really helps us picture what you are seeing.',
  'We have read through everything you sent and we are looking into it now.',
  'Could you send one more example so we can compare it against our logs?',
  'Thanks for your patience while we review this with the right people.',
  'Happy to help. Let us know if anything else comes up in the meantime.',
  'We have reviewed the attachment you sent over and noted the key points.',
  'Can you confirm the best phone number to reach you on if we need a quick call?',
  'Thanks for being so clear about the history; it makes it much easier to follow.',
  'Could you confirm the name that appears on the paperwork you sent us?',
  "I've added your details to the case notes so the next person has the full background.",
  'Thanks for sending that over. We will update you once there is something concrete to share.',
  'I can see why this has been annoying to deal with, and I appreciate you laying it out.',
];
const AUTO_LINES = [
  'Thank you for writing to us. Your message has been logged and given a reference number.',
  'This mailbox is monitored during business hours, and replies go out in the order they arrive.',
  'This is an automated message; please do not reply to this address directly.',
  'Our team has received your request and a colleague will review the details.',
  'Out of office: I am away from the office with limited access to email until my return.',
  'You are receiving this message because a case was opened on your behalf.',
  'For general questions, the help centre has guides that may answer them.',
  'We have recorded your contact details and will follow up using the information on file.',
];
const NOTE_LINES = [
  'the customer has been with us for several years and has no earlier escalations on file.',
  'the contact details on this record were last checked in the spring.',
  'no prior notes from the previous owner of this relationship.',
  'the account record was copied from an older system during the spring migration.',
  'the customer is on the standard support plan.',
  'the call-back number was not verified; the caller asked for a written reply only.',
  'checked the history: two earlier conversations, both closed by the customer.',
  'the reference was copied from an old spreadsheet, so the formatting may look odd.',
];

const GENERIC_SUBJECTS = ['Question about {prod}', 'Follow-up on {tkt}', 'Update on {order}', 'Access question', 'Billing check', 'Integration question', 'Feedback on {prod}', 'Request for information', 'Renewal paperwork', 'Re: {tkt}'];
const LEVEL_SUBJECTS: Record<Label, string[]> = {
  low: ['Quick question', 'Feedback on the release', 'Whenever you get a chance', 'Curious about the roadmap'],
  medium: ['Need an update this week', 'Review request', 'Follow-up on last month'],
  high: ['Follow-up #2, still waiting', 'Blocked on {prod}', 'Second request: please prioritise'],
  critical: ['Live issue, please check now', 'Possible breach in progress', 'Money moving right now', 'Outage affecting our customers'],
};

// ---------- thread composition ----------
interface Msg {
  role: Role;
  when: string;
  pre: string;
  body: string[];
  post: string;
}

function render(env: Env, subject: string, msgs: Msg[]): string {
  const parts = [`Case ${env.tkt} | Subject: ${subject}`];
  for (const m of msgs) {
    const head =
      m.role === 'customer'
        ? `Customer ${env.cust} ${env.custLast} (${m.when})`
        : m.role === 'agent'
          ? `Agent ${env.agent} (${m.when})`
          : m.role === 'auto'
            ? `Auto-reply (${m.when})`
            : `Internal note from ${env.agent} (${m.when})`;
    const body = [m.pre, ...m.body, m.post].filter((s) => s.length > 0).join(' ');
    parts.push(`${head}: ${body}`);
  }
  return parts.join(' | ');
}

function stamp(i: number): string {
  return `${pad2(9 + i * 2 + int(0, 1))}:${pad2(int(0, 59))}`;
}

function tryBuild(label: Label, env: Env, subject: string): string | null {
  const target = Math.round(clamp(gauss() * 180 + 790, 540, 1200));
  const nTurns = weighted([2, 3, 4, 5, 6], [0.14, 0.3, 0.26, 0.18, 0.12]);
  const roles: Role[] = ['customer'];
  for (let i = 1; i < nTurns; i++) {
    if (i === 1) roles.push(rnd() < 0.12 ? 'auto' : 'agent');
    else if (i % 2 === 0) roles.push('customer');
    else roles.push(rnd() < 0.2 ? 'note' : 'agent');
  }
  const msgs: Msg[] = roles.map((role, i) => ({
    role,
    when: `${env.date}, ${stamp(i)}`,
    pre: '',
    body: [],
    post: '',
  }));
  const custIdx = msgs.map((m, i) => (m.role === 'customer' ? i : -1)).filter((i) => i >= 0);
  const F = (s: string): string => fill(s, env);
  const deck = shuffleCopy([...DISTRACTORS, ...CONTEXT]);
  const acks = shuffleCopy(AGENT_ACKS);
  const autos = shuffleCopy(AUTO_LINES);
  const notes = shuffleCopy(NOTE_LINES);

  msgs[0].pre = pick(GREETINGS);
  // 1-3 signal sentences for the row's level, placed in customer turns.
  const k = weighted([1, 2, 3], [0.35, 0.45, 0.2]);
  for (const s of sampleN(SIGNALS[label], k)) msgs[pick(custIdx)].body.push(F(s));
  // Passing confounders (weaker or explicitly not a driver).
  if (rnd() < 0.18) msgs[pick(custIdx)].body.push(F(pick(CONF_NEG)));
  if (label !== 'low' && rnd() < 0.2) msgs[pick(custIdx)].body.push(F(pick(CONF_LOW)));
  // Neutral filler per turn; every turn gets at least one sentence.
  for (const m of msgs) {
    if (m.role === 'customer') {
      if (m.body.length === 0 || rnd() < 0.5) m.body.push(F(deck.shift()!));
    } else if (m.role === 'agent') {
      m.body.push(F(acks.shift()!));
    } else if (m.role === 'auto') {
      m.body.push(F(autos.shift()!));
    } else {
      m.body.push(F(notes.shift()!));
    }
  }
  const lastCust = custIdx[custIdx.length - 1];
  if (rnd() < 0.6) msgs[lastCust].post = F(pick(SIGNOFFS));

  const padTargets = msgs.map((m, i) => i).filter((i) => msgs[i].role !== 'auto');
  let guard = 0;
  while (render(env, subject, msgs).length < target) {
    if (++guard > 40 || deck.length === 0) return null;
    msgs[pick(padTargets)].body.push(F(deck.shift()!));
  }
  for (const m of msgs) shuffle(m.body);
  const text = render(env, subject, msgs);
  if (text.length < MIN_CHARS || text.length > MAX_CHARS) return null;
  return text;
}

function pickSubject(label: Label, env: Env): string {
  const pool = rnd() < 0.5 ? GENERIC_SUBJECTS : LEVEL_SUBJECTS[label];
  return fill(pick(pool), env);
}

const seenText = new Set<string>();
const seenPrefix = new Set<string>();
const prefix8 = (t: string): string => t.split(' ').slice(0, 8).join(' ').toLowerCase();

function makeRow(label: Label): string {
  for (let attempt = 0; attempt < 500; attempt++) {
    const env = makeEnv();
    const subject = pickSubject(label, env);
    const text = tryBuild(label, env, subject);
    if (text === null) continue;
    const p = prefix8(text);
    if (seenText.has(text) || seenPrefix.has(p)) continue;
    seenText.add(text);
    seenPrefix.add(p);
    return text;
  }
  throw new Error(`could not build a unique row for ${label}`);
}

// ---------- validate banks ----------
for (const label of LABELS) {
  const uniq = new Set(SIGNALS[label]);
  if (uniq.size !== SIGNALS[label].length) throw new Error(`duplicate signal in ${label}`);
  if (uniq.size < BANK_MIN) throw new Error(`${label} has only ${uniq.size} signal sentences`);
}
{
  const probe = makeEnv();
  for (const label of LABELS) for (const s of SIGNALS[label]) fill(s, probe);
  for (const s of [...DISTRACTORS, ...CONTEXT, ...CONF_NEG, ...CONF_LOW, ...SIGNOFFS]) fill(s, probe);
  for (const label of LABELS) for (const s of LEVEL_SUBJECTS[label]) fill(s, probe);
  for (const s of GENERIC_SUBJECTS) fill(s, probe);
}

// ---------- main ----------
const rows: { text: string; label: Label }[] = [];
for (const label of LABELS) for (let i = 0; i < PER_CLASS; i++) rows.push({ text: makeRow(label), label });
shuffle(rows);

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, rows.map((r) => JSON.stringify({ text: r.text, label: r.label })).join('\n') + '\n', 'utf8');

const summary: string[] = [];
for (const label of LABELS) {
  const lens = rows.filter((r) => r.label === label).map((r) => r.text.length).sort((a, b) => a - b);
  summary.push(`${label.padEnd(10)} n=${lens.length} min=${lens[0]} med=${lens[Math.floor(lens.length / 2)]} max=${lens[lens.length - 1]}`);
}
console.log(`wrote ${rows.length} rows -> ${OUT}`);
console.log(summary.join('\n'));
