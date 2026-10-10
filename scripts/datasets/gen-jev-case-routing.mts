// Deterministic generator for public/datasets/jev-case-routing.jsonl
// Label = which team owns a support case (6 classes, long multi-message threads).
// Run: node scripts/datasets/gen-jev-case-routing.mts
// Only Node built-ins. Fictional names/companies only; example.com emails; 555-01xx phones.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../../public/datasets/jev-case-routing.jsonl');
const SEED = 0x5eed2601;
const PER_CLASS = 417; // 6 x 417 = 2,502 rows
const MIN_CHARS = 500;
const MAX_CHARS = 1400;
const BANK_MIN = 60;

type Label = 'billing' | 'technical' | 'shipping' | 'account_security' | 'sales' | 'trust_safety';
type Role = 'customer' | 'agent' | 'auto' | 'note';
type Env = Record<string, string>;

const LABELS: Label[] = ['billing', 'technical', 'shipping', 'account_security', 'sales', 'trust_safety'];

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

// ---------- signal banks: >= 60 distinct sentences per label ----------
const SIGNALS: Record<Label, string[]> = {
  billing: [
    "The second charge posted at 6:30 PM on {date}, after the first one had already cleared.",
    "I was charged {amt} twice on {date} for the same {prod} subscription.",
    "The invoice for {order} shows {amt} in tax even though our account is listed as tax-exempt.",
    "Can you refund the difference between the {amt} we paid and the {amt2} we were quoted?",
    "Our card on file expired, and the renewal of {amt} went through on a card we never added.",
    "The receipt has a line called 'service adjustment' for {amt} that nobody on our side can explain.",
    "We were billed for seven seats but only five people are active on the {prod} account.",
    "Please reverse the late fee of {amt} from {date}; the payment cleared the same day it was due.",
    "Accounts payable needs a corrected invoice with our purchase order number printed on it.",
    "The annual plan renewed at {amt} instead of the discounted rate written in our contract.",
    "There is a pending charge of {amt} from your company on our statement that is not on the invoice list.",
    "Could someone check why the VAT line changed from zero to twenty percent on the latest invoice?",
    "Our bookkeeper needs a credit note for the duplicate payment made on {date}.",
    "The payment method is a debit card that our bank keeps declining, so each renewal bounces and then retries.",
    "Please stop the automatic charge; we moved to a purchase order and the card should not be billed again.",
    "I want to dispute the {amt} debit on my statement, because the cancellation was confirmed on {date}.",
    "The proration on our mid-cycle upgrade is wrong: we were billed for the full month and still got no credit.",
    "The subscription price went from {amt} to {amt2} per month without the notice you promised in the spring.",
    "Our finance team reconciled the statement and found a {amt} refund that never reached the card.",
    "The reversal of {amt} shows up as a chargeback on our side, which means we are also paying a dispute fee.",
    "Can you send the invoice history for the last twelve months so we can close the books?",
    "The billing address on the invoice needs changing; the amount itself is correct.",
    "Why does the statement show two separate charges labelled 'seat fee' for the same user?",
    "We were refunded to the wrong card last month, and the original {amt} is still missing.",
    "The coupon from the spring offer was never applied, so the {amt} total on the invoice is too high.",
    "Please confirm whether the overage charge of {amt} is based on metered usage or on the flat plan.",
    "A card surcharge added {amt} to our renewal, and that line was not in the order form we signed.",
    "Our auditor needs a breakdown of every line on invoice {order}, especially the currency conversion fees.",
    "The refund of {amt} has been pending on your side for nine days, and the portal still says processing.",
    "We keep getting reminders about an overdue balance of {amt} that we already paid by wire on {date}.",
    "The tax ID on file is wrong, so the invoice carries the wrong sales tax category and the total is off.",
    "Please send the refund to the original payment method rather than issuing store credit.",
    "The premium support line is billed monthly, even though our contract says it is included in the annual fee.",
    "The billing portal says the balance is {amt}, but the statement PDF shows {amt2} for the same period.",
    "Please void the unpaid invoice {order}; it was issued under the wrong legal entity name.",
    "The chargeback fee of {amt} arrived after we had already refunded the customer in full.",
    "We are being charged in dollars although the contract says the invoices are in euros.",
    "The new card was never saved after the old one was replaced, so renewal failed and the retry charged us twice.",
    "I can't find where the {amt} discount from the renewal offer appears on the next invoice.",
    "Our cost centre code is missing from the invoice, so accounting can't approve the payment.",
    "Please explain the 'platform fee' line; it was not on the quote we accepted on {date}.",
    "We paid the balance, but the invoice still shows {amt} outstanding and late interest keeps being added.",
    "The gift card balance of {amt} was not applied at checkout, and the full price went to the card.",
    "Could you split the {amt} charge across two cost centres and reissue both invoices?",
    "The refund for the cancelled add-on is {amt}; the confirmation says processed but nothing has hit our bank.",
    "The upgrade was billed twice because the first payment attempt timed out and the second one went through too.",
    "Please confirm that my payment of {amt} on {date} covered the invoice and not only the late fee.",
    "Our tax exemption certificate was uploaded, but the invoices still carry sales tax.",
    "The invoice total includes a charge for a plan we downgraded from on {date}.",
    "We need a refund for the unused prepaid credits of {amt} before the account is closed.",
    "The monthly statement lists our payment as received on {date}, but the bank shows it cleared on a different date.",
    "Please change the invoice currency to dollars; the amounts were converted at a rate our contract does not mention.",
    "We were charged a restocking fee of {amt} on a return that our records show was never accepted at your warehouse.",
    "A credit of {amt} from the cancelled order is still sitting on the account and can't be applied to the new invoice.",
    "The finance portal refuses to download invoices older than eighteen months, so I can't file them.",
    "Our procurement card paid the renewal, but the invoice was addressed to a different department.",
    "The discount code from our contract was typed into the wrong field, so the full {amt} was billed for the month.",
    "Why was the setup fee of {amt} added to our renewal when we have been customers for three years?",
    "The payment schedule on the order form said quarterly, but the system bills us every month.",
    "I received a bill for {amt} for add-on storage we never switched on; please adjust it.",
    "Our payment to invoice {order} bounced back as a failed transfer, and the late fee was added anyway.",
    "The invoice shows a rounding adjustment of {amt} that makes the total harder to match to our purchase order.",
    "We were overcharged on overage: the rate applied was double the rate in the signed order.",
    "Could you reissue the statement with the correct payment terms? It says net 15 instead of net 45.",
    "The refund request from {date} still shows 'awaiting approval' and the finance inbox has not replied.",
  ],
  technical: [
    "Since the 2:05 PM deploy on {date}, every export has come back with an error after about forty seconds.",
    "The export job fails every time with a 500 error after about forty seconds of processing.",
    "Our nightly sync to the warehouse has returned HTTP 504 from the API for three nights running.",
    "Dashboard charts render blank in Chrome 124 but load correctly in Firefox on the same machine.",
    "Since the last release the mobile app crashes on launch whenever the offline cache is larger than 200 MB.",
    "Webhook deliveries are timing out, and the queue on the receiving side now holds about twelve thousand events.",
    "The CSV importer silently drops any row that contains a comma inside a quoted field.",
    "Search results take over eight seconds now, up from under one second at the start of the month.",
    "Our integration with the inventory system broke after the schema change that went out on {date}.",
    "The REST endpoint for orders returns an empty array even though the orders show up in the web interface.",
    "Pagination in the API skips every second page once the page size is set above fifty.",
    "The SDK throws a NullReferenceException inside the batch writer whenever the queue is empty.",
    "Scheduled reports stopped generating after the cluster upgrade, and nothing relevant appears in the logs.",
    "The API gateway returns 503 for roughly a third of requests coming from our EU region.",
    "The connector to our ticketing tool stopped pushing updates, and the last successful run was {date}.",
    "Attachments over 25 MB fail to upload with a generic error and no detail in the console.",
    "Memory on the worker nodes climbs steadily until the process restarts, roughly every six hours.",
    "The rate limiter rejects our calls at about forty requests a minute, although our plan allows far more.",
    "Time zones are wrong in the scheduled emails: a 9 AM send goes out at 4 AM.",
    "After the update the PDF export drops embedded fonts, and the totals column prints as hash symbols.",
    "The {prod} app freezes when we scroll past the 500th row in a table view.",
    "Our staging environment serves stale inventory data because the cache never invalidates.",
    "Dashboards show zero events for the last two hours, even though the ingestion logs show steady traffic.",
    "Version 4.8 has a regression: the formula engine returns #VALUE for any cell that references another sheet.",
    "The bulk update endpoint returns 200 but the changes never persist to the database.",
    "Our SFTP drop folder stopped picking up files after the server rebuild on {date}.",
    "Two jobs have been stuck in 'running' for days and block everything queued behind them.",
    "The GraphQL schema now marks fields as nullable that were non-null last week, which breaks our typed client.",
    "The upload progress bar reaches 100 percent while the file is still transferring for several minutes.",
    "Python SDK 2.3 raises a TypeError whenever a list is passed to the metadata argument.",
    "Report latency averages six seconds per query since {date}; our internal target is one second.",
    "Product images return 404 after the CDN migration, so about 900 listings show broken thumbnails.",
    "The sandbox key returns 'version unsupported' for every call to the newer endpoints.",
    "The embedded widget throws a JavaScript error, and the checkout iframe never finishes loading.",
    "The integration writes duplicate records whenever a retry happens inside the same second.",
    "The data pipeline fails with a checksum mismatch on parquet files written by the new exporter.",
    "Scrolling in the desktop client stutters badly on laptops with integrated graphics.",
    "The import reports success, but the record count in the target is 14 percent lower than the source.",
    "The status page says all systems are operational, yet our API calls hang for minutes at a time.",
    "The build pipeline fails at the packaging step with an unhandled promise rejection.",
    "Mobile push notifications arrive about an hour late, and some never arrive at all.",
    "The search index is missing every document created after {date}, so new records do not show up.",
    "The custom field mapping returns null for every date-type field in the connector.",
    "The CLI crashes with a segmentation fault on arm64 builds, while the x86 build is fine.",
    "Background sync on the iOS client stops after the app has been idle for a few minutes.",
    "The API returns 422 'invalid encoding' for UTF-8 names that contain accented characters.",
    "Monthly rollups double count events that arrive exactly at midnight UTC.",
    "The chart export endpoint sends an empty PNG whenever the chart has more than three series.",
    "Our self-hosted runner cannot reach the artifact service; the connection resets after the TLS handshake.",
    "A data model migration left two columns swapped in the reporting view.",
    "The bulk CSV import sets the wrong locale on new customers, which breaks their number formats.",
    "Autosave in the editor fails silently, and documents revert to the version saved on {date}.",
    "The reconciliation script hits a deadlock on the ledger table once per run.",
    "A bug in the date parser turns 2 March into 3 February for every row imported from spreadsheets.",
    "Webhook signatures fail verification on payloads that contain emoji, so our handler rejects them.",
    "The sync agent crashes with an out-of-memory error when a folder holds more than eighty thousand files.",
    "The indexing job runs every night, but the index size stops growing after the first gigabyte.",
    "The API client leaks a socket on each retry, and after a few hundred calls the service refuses connections.",
    "The batch endpoint silently truncates request bodies above 1 MB, so the last records are lost.",
    "Our custom report template throws a parse error on any formula that uses nested brackets.",
    "The geocoding integration returns coordinates in the wrong hemisphere for every address in the database.",
    "The reporting database returns different totals for the same query when it is run twice in a row.",
    "Scheduled exports write files with a byte order mark, and our parser fails on the first column header.",
    "Event timestamps are stored as local time after the migration, so every daily job is shifted by a few hours.",
    "Our test sandbox endpoint returns 500 whenever the request includes a custom header with an underscore.",
  ],
  shipping: [
    "The last scan showed the parcel at the hub at 7:45 AM on {date}, and nothing has moved since.",
    "The insured value on box {order} is {amt}, and the carrier wants proof of the contents before it opens a claim.",
    "The consignment was declared at {amt}, but it has been held at the border and nobody has told us why.",
    "Tracking for {order} has shown 'label created' since {date}, and the parcel has not been scanned once.",
    "The box arrived with a crushed corner, and the {prod} screen inside is cracked.",
    "The carrier says the delivery window was missed because the driver could not find the loading dock.",
    "Can we change the delivery address on {order} to our warehouse in {city}? It has not left yet.",
    "The parcel is marked delivered, but nothing was left at the door and there is no photo of the drop.",
    "Our pallet of twelve units went to the wrong depot in {city} and has been sitting there for a week.",
    "The shipment is stuck in customs with a status of 'awaiting documents' since {date}.",
    "We would like the freight held for pickup at the terminal instead of delivered to the site.",
    "Two of the eight cartons in shipment {order} never arrived, although the packing slip lists all eight.",
    "The courier scanned the package at the hub, but the scan history has a twelve-day gap afterwards.",
    "Please reroute the parcel; the recipient moved offices and the new suite number is different.",
    "The delivery estimate slips by a day every time I refresh the tracking page.",
    "The packaging was torn open in transit and a bottle has leaked into the box.",
    "The carrier ignored the appointment window we booked for the dock delivery on {date}.",
    "The parcel was signed for by someone in the building lobby who does not work for our company.",
    "Please send a replacement for the damaged units from {order}; the carrier has already closed its claim.",
    "The label weight is off by four kilos, so the carrier re-weighed the box at the depot and the new size is heavier.",
    "Can you confirm the expected delivery window for the replacement pallet arriving at the {city} warehouse?",
    "The tracking number in the confirmation email does not match the one shown in the carrier app.",
    "The parcel went to our old address even though I updated the delivery details last week.",
    "The return label expired before the item was collected, so the return is sitting at the sender's depot.",
    "The order was marked shipped, but the carrier has no record of ever picking it up.",
    "The freight forwarder says the container is still at the port and the cut-off for loading is {date}.",
    "Two boxes were delivered to the building next door, and the neighbour refuses to hand them over.",
    "The courier left the pallet outside in the rain and the cardboard is soaked through.",
    "Could you add a signature requirement to deliveries? The last parcel was left unattended by the door.",
    "The shipping partner's estimate says {date}, but the tracking shows the crate was never loaded onto the truck.",
    "Half the units from the spring order arrived dented, and the seals on three cases are broken.",
    "The status jumped from 'in transit' to 'returned to sender' without a scan at any facility in between.",
    "We need the delivery moved to Saturday, because the receiving team is only on site on weekends.",
    "The label printed with the wrong postcode, so the parcel keeps bouncing between two hubs.",
    "The last-mile partner says it will not attempt the drop unless the gate access code is provided.",
    "I am still waiting on the parcel that was due on {date}; the carrier page only says 'exception'.",
    "Please advise whether the transit insurance claim can be opened for box {order}.",
    "The cold chain sensor shows the crate reached 14 degrees in transit, so the vials may not be usable.",
    "The driver refused to unload the pallet because the dock was too low for the truck.",
    "Can you ask the depot to hold the parcel until I am back on {date}?",
    "Our store manager asks why the restock shipment is three days behind the window we were given.",
    "Two pallets were delivered, but the pallet count on the bill of lading is one short.",
    "The parcel was scanned into the wrong country and now sits under a different tracking number.",
    "The partner missed the drop because the access code for the loading gate did not work.",
    "The receiving bay refused the shipment because the pallets were not shrink-wrapped.",
    "The carrier claims the box was never handed over, but our warehouse has a signed receipt from that day.",
    "Please confirm the carrier name and new tracking number for {order}, since the original link is dead.",
    "The return of the faulty dock unit was delivered to our old warehouse in {city}.",
    "The box shows 'damaged in sorting' on the tracking page, and the contents are a broken glass display.",
    "The parcel has been in a hub for eleven days with no scans, and the sender keeps asking us for proof of delivery.",
    "The forklift could not reach the pallet inside the trailer, so the driver took it back to the yard.",
    "The delivery was rescheduled twice without anyone telling us, and the crew turned up at the wrong gate.",
    "Our replacement tyres were routed through a consolidation point that is closed on weekends.",
    "Please hold the shipment at the airport until we confirm the delivery window with the buyer.",
    "The manifest lists nine boxes, but the truck only unloaded seven of them.",
    "The carrier has the wrong phone number on file, so the driver cannot call before the drop.",
    "The last update said the parcel was out for delivery, yet it has been sitting at the local depot.",
    "The consignment was damaged by a forklift at the cross-dock, and the photos show crushed corners.",
    "The package keeps getting returned to us because the building intercom does not accept deliveries.",
    "Freight has been stuck at the border since {date} waiting for a transit document we have already sent.",
    "A carton of samples arrived with the lid open and the contents scattered; the carrier has the photos.",
    "We never received the second half of the order, which the courier said went out on {date}.",
    "Please move our delivery to the loading dock on the north side; the front entrance is closed for works.",
    "The parcel tracking jumps between two cities every hour, which suggests the consignment is being moved in circles.",
    "The local courier says it cannot deliver to our postcode this week because the route is suspended.",
    "The label is correct, but the carrier scanned the wrong barcode during pickup at our loading bay.",
    "The pallet of replacement filters was left in the lobby, unsigned, and the security desk has no record of it.",
  ],
  account_security: [
    "I got a 2FA prompt I never triggered at 3:12 AM from a device that signed in from {city}.",
    "My authenticator app was reset when I changed phones, and the backup codes no longer work.",
    "Someone has been signing in to our admin account from an unknown IP address since {date}.",
    "I never requested the password reset email that arrived on {date}, and I think someone is trying to get in.",
    "The security key was removed from our account, and the audit log doesn't say who removed it.",
    "The finance manager's mailbox forwarded a login notification to an address that nobody here recognises.",
    "Under my data access request, I need a full export of all personal data you hold about me.",
    "The account recovery form asks for a phone number that is no longer on the account.",
    "Two former staff were removed from the workspace, but their sessions and API tokens still work.",
    "Please lock the account; I think my credentials were pasted into a public forum.",
    "We found a session from an unfamiliar browser that refreshed its token three times overnight.",
    "The 2FA text codes arrive about ten minutes late, by which time they have already expired.",
    "The SSO setup rejects the new SAML certificate and shows 'invalid signature' to every user.",
    "I want my personal data deleted and written confirmation of what was stored, under my privacy request.",
    "Someone I don't know changed the recovery email on our owner account last night.",
    "After I changed my password I was still signed in on a laptop I don't own.",
    "I was locked out after six failed sign-in attempts, and the unlock link returns a 404 page.",
    "The passkey I registered does not work on my new phone, and the fallback code is rejected.",
    "My mother's account was taken over: the login email was replaced and the phone number changed.",
    "Please list which third-party apps can read our mailbox data; we never approved the newest one.",
    "The audit log shows a full customer export at 2:40 AM by a user nobody here recognises.",
    "We need an access review, because a former contractor still holds admin rights in the workspace.",
    "The recovery codes I printed from the settings page are blank, so I cannot get back in after the phone reset.",
    "I suspect my password appeared in a breach list; please force a reset and revoke all sessions.",
    "Why has my account email changed to a domain I have never used? I never saw a confirmation message.",
    "The 2FA challenge loops after I approve it on my phone, so I can never finish signing in.",
    "We believe a shared service account was used to download the payroll export on {date}.",
    "Please send me a copy of everything you store about my account, including support chat history.",
    "The login alert says a new device was added from {city}, but nobody on our team has been there.",
    "I submitted my data access request last week and the verification email never arrived.",
    "Our SSO admin is locked out, and the break-glass account has also been disabled.",
    "The person asking us to reset passwords on their behalf is not on our staff list.",
    "Someone added a new authenticator device to my account while I was on a flight.",
    "The sign-in page accepts my password and then sends me to a copy of your login page that looks wrong.",
    "The account has a verified phone number I don't recognise, and the email was swapped on {date}.",
    "Please revoke the API key that appeared last night; I didn't create it and it has full scope.",
    "I see two active logins from different countries, and I cannot end either session from the settings page.",
    "The reset link expired before I could use it, and every new link says it was already used.",
    "Our access logs show the same session token used from three IP addresses within one minute.",
    "Has anyone changed our two-step verification settings? The admin list looks the same but the policy differs.",
    "I want to know what data was accessed when my account was opened from {city} on {date}.",
    "A stranger has been added as a co-owner of our shared drive, and their login came from abroad.",
    "The sign-in attempts come from a list of addresses that match a known credential stuffing campaign.",
    "We need the account's data export list so we can answer a subject access request from one of our customers.",
    "The one-time code I am sent is always rejected, even though the clock on my phone is correct.",
    "A customer security questionnaire asked whether our admin accounts use 2FA, and we need to confirm the status.",
    "The mailbox has a forwarding rule we never set up, sending copies of messages to an outside address.",
    "I am not sure whether the password change on {date} was made by me, and the audit trail is ambiguous.",
    "Our hardware keys stopped working after the firmware update, and the fallback is an emailed code we cannot receive.",
    "The unlock request for the locked service account has been pending, and no person has picked it up.",
    "We found a saved login for our billing portal on a shared computer in the lobby, so it needs a reset.",
    "Someone is requesting password resets for our staff every few minutes, and the reset emails bounce back to our inbox.",
    "The account was accessed from a new device through an old session cookie; please terminate that session.",
    "Could you confirm which employee's credentials were used for the export that ran on {date}?",
    "We were asked to approve a sign-in on our phone that we never started, and the prompt keeps repeating.",
    "The sign-in history shows a login from an address we have never used, two days before the reset.",
    "I am asking for a copy of all the data processed about my account and the processors involved.",
    "Every time I try to change my password, the form says the old one is still valid, so the change never happened.",
    "My recovery phone number was changed without my consent, and a text confirming the change went to a number I no longer use.",
    "The account owner left the company last month, but their login still receives the security codes for the admin console.",
    "We need help disabling legacy authentication on the mailbox, because sign-ins over that protocol are unexplained.",
    "The sign-in alert said a password was changed at 11 PM, and that was not me.",
    "Please check whether a third party has been granted access to our API tokens through the consent screen.",
    "Our employees are getting login prompts they never started, and two of them approved one by mistake.",
  ],
  sales: [
    "Could we book a 10:30 AM call on {date} to walk our procurement lead through the options?",
    "A competitor quoted us {amt} for the same seat count, so can you match that for a three-year term?",
    "Our finance lead set a ceiling of {amt} for the first year, and we'd like to see what fits under it.",
    "Could you hold the proposal at {amt} for the pilot until our budget review, which is next month?",
    "We would like a quote for 150 seats on the Enterprise plan, billed annually; last year's figure was {amt}.",
    "Could you send a quote for 150 seats on the Enterprise plan, billed annually?",
    "We would like to book a demo for our operations leads, ideally on a Tuesday or a Wednesday.",
    "What is the per-seat price if we commit to three years instead of one?",
    "Our procurement team needs the vendor registration form and a W-9 before a purchase order can be raised.",
    "We are evaluating the platform for a new branch in {city} and would like to talk about volume pricing.",
    "Is there a partner programme for agencies that resell the reporting module to their own clients?",
    "Please include professional services hours in the proposal, scoped for a six-week onboarding.",
    "We are interested in moving from Team to Business and want a comparison with our current setup.",
    "Would you be open to a co-marketing partnership? We run a webinar series with around 800 attendees.",
    "Our CFO wants a non-binding estimate before the budget meeting on {date}.",
    "Do you offer a startup discount for companies under twenty employees, and what would that look like on {prod}?",
    "Can you send the standard master services agreement so our legal team can start its review?",
    "We would like to add two regions to the contract we signed in the spring; what would that cost?",
    "I am comparing your product with two other tools and need the feature sheet for the analytics module.",
    "Our sales director asked whether a pilot with 25 licences is possible before we commit to anything.",
    "What are your payment terms for public-sector buyers? Our framework agreement requires net sixty.",
    "We want to discuss reseller margins for the EMEA territory and any minimum commitments.",
    "Please share case studies from logistics customers with a similar volume to ours.",
    "Can we schedule a call to discuss a custom plan with single sign-on and longer data retention?",
    "Our team grows from 40 to 120 people next quarter, and we want pricing that scales with that.",
    "We are preparing a request for proposal and want to know whether the platform meets section four.",
    "What does the Scale tier cost, prorated from the start of the year, if we upgrade from Business?",
    "Do you have a quote template with our logo and bank details on the last page?",
    "Our university wants a volume licence for the research labs; who handles education pricing?",
    "Could a sales engineer join our kickoff call to walk the team through the integration options?",
    "We would like to become a referral partner and need the commission schedule before we sign anything.",
    "Is it possible to add an annual prepay discount to the quote we received on {date}?",
    "The supplier onboarding questionnaire is attached; please return it so our vendor team can approve you.",
    "We are considering a multi-year deal and need the price hold extended to the end of the quarter.",
    "Please send the enterprise pricing sheet, including the volume tiers above 500 seats.",
    "I would like a trial environment for our agency with three client workspaces before we buy.",
    "We have budget approved for a new analytics platform and want a demo focused on forecasting.",
    "Can your team run a proof of concept with our sample data before we sign the order form?",
    "Our reseller agreement is up for renewal, and we want to discuss expanding into two new countries.",
    "Do you have a plan that bundles the API add-on with the team tier, and what does it cost?",
    "We would like a written quote covering migration services for {n} legacy databases.",
    "The buying committee wants a security questionnaire and pricing before the vendor shortlist vote.",
    "Could you confirm whether the Business plan includes unlimited guest users for our contractors?",
    "We are extending our contract to a second business unit and need a new order form at the same rate.",
    "What is the lead time to set up a dedicated account manager for an enterprise rollout?",
    "We are a nonprofit and wondered whether you have a discounted rate for registered charities.",
    "Please quote the professional training package for twenty admins, with travel included.",
    "We want to compare annual and monthly pricing for the Pro plan before our board signs off on the spend.",
    "Could we get a meeting with someone from your partnerships team about a joint webinar in the autumn?",
    "The procurement portal requires an itemised quote with unit prices, so we need that before requesting approval.",
    "We're looking at white-label options for our customers and would like to understand the licensing terms.",
    "Do you have a volume price for the Kestrel Tablet bundle if we order fifty units for our field teams?",
    "Could you put together a three-tier option (basic, standard, premium) so we can compare them for the committee?",
    "Our group is consolidating five subscriptions into one enterprise deal; what would the combined price be?",
    "We would like to know whether the training credits can be converted into an extra demo seat for our sales team.",
    "I'm writing on behalf of a distributor who wants to sell the product in Scandinavia and needs a margin proposal.",
    "What discount applies if we sign a two-year term but want to add seats during the first year?",
    "We need a sales call to understand whether the platform fits a franchise model with 30 locations.",
    "The buyer's checklist asks for pricing tiers, implementation timelines, and a reference customer; can you help?",
    "Would it be possible to get a custom quote that bundles hardware and the software subscription?",
    "Our company is interested in a strategic partnership and would like to meet your business development lead.",
    "Please send over the enterprise price list and the approved vendor documents for our procurement system.",
    "We are shortlisting three suppliers and need your best pricing for 200 named users by the end of the month.",
    "Could a rep attend our trade show booth in {city} to demo the product for prospective clients?",
    "What is the price difference between the standard and dedicated environment for a company our size?",
    "Our board wants to explore an enterprise agreement, so we need a term sheet and a proposed signing date.",
    "We'd like to expand from a pilot of ten users into a full rollout; what would the pricing look like?",
    "Do you have a pricing calculator for the Relay Desk add-on that we could share with our sales leadership?",
    "Interested in becoming a certified implementation partner; what are the requirements and the margin structure?",
  ],
  trust_safety: [
    "The listing went live at 1:20 AM and was taking orders from buyers before breakfast.",
    "A set of fake listings has taken in about {amt} from buyers who never received anything, and the seller is still active.",
    "The vendor collected {amt} in deposits for events that were never run, and the same account is still taking bookings.",
    "I want to report a seller on your marketplace who is listing counterfeit versions of our product.",
    "A user has been sending threatening messages to our staff every evening for two weeks.",
    "A group of accounts is posting identical five-star reviews to push a supplier's rating up.",
    "This profile is impersonating our CEO, with his photo and our logo, and messaging staff asking for payment.",
    "I found listings offering stolen card numbers in the marketplace, and the seller reference is {tkt}.",
    "Someone posted my home address and phone number in a comment thread; please remove it.",
    "A reseller account is selling prohibited items, and the listing photos show they are illegal in our state.",
    "Several accounts appear to have been created in bulk and are cashing out promotional credits.",
    "A user keeps sending unsolicited explicit images to members of our community channel.",
    "I received a message from a verified-looking account asking me to buy gift cards for a manager.",
    "Our community moderator was doxxed after a dispute, and the post containing her details is still up.",
    "I want to report a page that copies our brand and collects login details from visitors.",
    "The vendor sells counterfeit adapter cables and has ignored three takedown notices.",
    "We believe a marketplace ring is using fake delivery confirmations to get payouts released.",
    "A member of our group chat shared links to abusive imagery, and I'm reporting the whole channel.",
    "A user is running a harassment campaign against a streamer, with dozens of coordinated accounts.",
    "A phishing email using our name was sent to customers from a lookalike domain registered to an unknown party.",
    "A seller is using forged ID documents to pass verification and is listing restricted goods.",
    "I want to file a complaint about a vendor who posted hateful slurs about a religious group in product reviews.",
    "The account is part of a fraud ring that buys accounts with stolen identities; I have screenshots.",
    "Someone created a fake support agent profile to collect payment details from our users.",
    "A seller is advertising weapons parts and asking buyers to move the sale off the platform.",
    "We found a scraper network reposting our copyrighted training materials for resale.",
    "A user is blackmailing another member with private photos and threatening to publish them.",
    "The profile uses our logo and our executive's name and invites people to join a crypto giveaway.",
    "I have evidence that a reviewer was paid to leave one-star reviews on a competitor's listing.",
    "The account is collecting phone numbers and personal details of minors from a school forum.",
    "A group of sellers is coordinating fake bulk orders to trigger the seller bonus programme.",
    "This user keeps opening new accounts after each ban to keep sending abusive direct messages.",
    "I'm reporting a deepfake video of a local councillor that is spreading across the platform as real footage.",
    "Several listings are selling stolen gift cards, and buyers report the codes were already redeemed.",
    "The account is posting job offers that ask applicants to pay a training fee and share bank details.",
    "A member is posting threats of violence against a named teacher in our school's public group.",
    "The seller ships empty boxes and collects payouts before any delivery is recorded; I'd like this investigated.",
    "Our brand team found a lookalike domain used to impersonate our billing team and send fake invoices.",
    "A user is posting spam links to malware disguised as software updates across hundreds of threads.",
    "The page is selling access to other people's streaming accounts, which breaks your terms.",
    "Several users are sharing a guide on laundering money through the marketplace's refund flow.",
    "I'm reporting a repeat harasser who has been sending hateful messages to our moderators since {date}.",
    "A person posing as a journalist keeps requesting private information about our employees.",
    "We have a takedown request for content that exposes a child's private details, posted without consent.",
    "The vendor account changed its name and photo to pose as a well-known local bakery and is taking orders.",
    "There's an app in the store using our trademark that asks users for card details during setup.",
    "The account owner was banned for fraud last year and has come back under a different business name.",
    "Please investigate a scam group that targets older members with fake prize notices.",
    "The seller threatens to post bad reviews unless we pay for the listing to stay up.",
    "A sponsored post is impersonating a government benefits office and collecting national ID numbers.",
    "The forum admin account is rotating through fake names to spread a recruiting message for a banned group.",
    "Several accounts are posting the same doctored screenshots to discredit a local business owner.",
    "The marketplace seller is taking deposits for event tickets that were never issued, and buyers are complaining in bulk.",
    "A member is sharing other people's login credentials in a public thread and asking for more.",
    "I want to report a user who uploads hidden tracking software disguised as a photo editor.",
    "There's a group selling access to hacked school accounts, and the posts include screenshots of student records.",
    "The reviewer account is a copy of a verified customer's profile with a changed photo and a new name.",
    "The account invites children to private chats and asks them to keep the conversation secret.",
    "Multiple vendors use the same bank account and shipping address to pose as independent businesses.",
    "The store sells supplements with a fake certification seal, and the seal belongs to an agency that does not exist.",
    "A user is harassing a volunteer moderator with threats posted under several names across our forums.",
    "Our verification team found identity documents uploaded by seven different sellers, all with the same photo.",
    "The chat account claims to be our fraud department and asks users to confirm their card number.",
    "A reseller is cloning our product listings and redirecting buyers to a payment page on another domain.",
    "The creator's channel is hosting a stream of doxxing content aimed at a journalist, with thousands of viewers.",
    "A vendor is using bots to flood our search results with counterfeit listings each night.",
    "Someone is threatening to publish a former employee's medical details unless we restore their old account.",
  ],
};

// Topic phrases used only inside confounder sentences (a passing mention of another team's issue).
const TOPICS: Record<Label, string[]> = {
  billing: ['the invoice total', 'the card renewal', "last month's refund"],
  technical: ['the API timeout', 'the export error', 'the dashboard glitch'],
  shipping: ['the parcel tracking', 'the delivery window', 'the pallet that went to the wrong depot'],
  account_security: ['the password reset email', 'the 2FA prompt', 'the unfamiliar login alert'],
  sales: ['the pricing page', 'the demo request', 'the renewal quote'],
  trust_safety: ['the abusive comment thread', 'the fake seller listing', 'the impersonation account'],
};
const CONFOUNDER_FRAMES = [
  'Not the main issue, but {x} came up while I was reviewing this.',
  "I don't think {x} is related, though I wanted to mention it.",
  'Separate from this, {x} also happened last week, and it is not what I am asking about.',
  'Side note: {x} is unrelated to the question below.',
  "For completeness, I saw {x} too, but please don't treat it as the reason I'm writing.",
  "I'm mentioning {x} only because it was in the same email thread; it is not the cause here.",
  'Probably coincidence, but {x} showed up at the same time.',
  "Ignore {x} for now; it's a minor detail and I have already dealt with it.",
];

// Shared neutral material (usable with any label). Avoid team words in these.
const DISTRACTORS = [
  "I'm writing from our office in {city}, where the coffee machine has been broken all week.",
  'We moved to a new floor in the spring, so some of the contact details I have may be out of date.',
  'I can send screenshots if they would help, though I have not attached anything yet.',
  '{cc} is copied on this message in case I am away from my desk.',
  'The weather here has been strange, with rain every afternoon followed by sudden sunshine.',
  'We are finishing our offsite next week, so my calendar is packed.',
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

// Subjects per label (topical, the way real subject lines look).
const SUBJECTS: Record<Label, string[]> = {
  billing: ['Charge question on {order}', 'Invoice {order} does not match', 'Refund still pending', 'Duplicate payment on our account', 'Renewal billed at the wrong price', 'VAT line on latest invoice', 'Card declined but charged'],
  technical: ['Export failing with 500', 'Sync stopped after update', 'API returns empty results', 'Dashboard blank in Chrome', 'Import drops rows', 'Crash on launch after update', 'Slow reports since last release'],
  shipping: ['Parcel {order} has not arrived', 'Tracking stuck at label created', 'Damaged delivery on {order}', 'Change delivery address', 'Pallet delivered to wrong depot', 'Missing carton from shipment'],
  account_security: ['Unexpected 2FA prompt', 'Suspicious sign-in alert', 'Password reset I never requested', 'Data access request', 'Locked out of admin account', 'Recovery email changed'],
  sales: ['Quote for 150 seats', 'Demo request for ops team', 'Partnership enquiry', 'Pricing for multi-year plan', 'Procurement paperwork', 'Upgrade to Business tier'],
  trust_safety: ['Report: impersonation account', 'Abusive messages from a user', 'Counterfeit listing report', 'Fake seller profile', 'Scam group report', 'Policy violation by vendor'],
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
  const deck = shuffleCopy(DISTRACTORS);
  const acks = shuffleCopy(AGENT_ACKS);
  const autos = shuffleCopy(AUTO_LINES);
  const notes = shuffleCopy(NOTE_LINES);

  msgs[0].pre = pick(GREETINGS);
  // 1-3 signal sentences for the row's label, placed in customer turns.
  const k = weighted([1, 2, 3], [0.35, 0.45, 0.2]);
  for (const s of sampleN(SIGNALS[label], k)) msgs[pick(custIdx)].body.push(F(s));
  // Occasional passing mention of another team's issue, explicitly not the cause.
  if (rnd() < 0.18) {
    const other = pick(LABELS.filter((l) => l !== label));
    const frame = pick(CONFOUNDER_FRAMES).replace('{x}', pick(TOPICS[other]));
    msgs[pick(custIdx)].body.push(F(frame));
  }
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
  for (const m of msgs) shuffleInPlace(m.body);
  const text = render(env, subject, msgs);
  if (text.length < MIN_CHARS || text.length > MAX_CHARS) return null;
  return text;
}

function shuffleCopy<T>(xs: readonly T[]): T[] {
  const c = [...xs];
  shuffleInPlace(c);
  return c;
}
function shuffleInPlace<T>(xs: T[]): void {
  shuffle(xs);
}

const seenText = new Set<string>();
const seenPrefix = new Set<string>();
const prefix8 = (t: string): string => t.split(' ').slice(0, 8).join(' ').toLowerCase();

function makeRow(label: Label): string {
  for (let attempt = 0; attempt < 500; attempt++) {
    const env = makeEnv();
    const subject = fill(pick(SUBJECTS[label]), env);
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
  if (TOPICS[label].length < 2) throw new Error(`${label} needs topics`);
}
{
  const probe = makeEnv();
  for (const label of LABELS) for (const s of SIGNALS[label]) fill(s, probe);
  for (const s of [...DISTRACTORS, ...SIGNOFFS]) fill(s, probe);
  for (const f of CONFOUNDER_FRAMES) fill(f.replace('{x}', 'x'), probe);
  for (const label of LABELS) for (const s of SUBJECTS[label]) fill(s, probe);
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
  summary.push(`${label.padEnd(16)} n=${lens.length} min=${lens[0]} med=${lens[Math.floor(lens.length / 2)]} max=${lens[lens.length - 1]}`);
}
console.log(`wrote ${rows.length} rows -> ${OUT}`);
console.log(summary.join('\n'));
