import { disposeJev } from './jev';
import { disposeDecide } from './gliner2';
import { disposeDecoder } from './decoder';

export type HeavySession = 'jev' | 'decide' | 'decoder';

/**
 * The big inference sessions (JEV ~0.4–2.3 GB, Decide ~0.9 GB, decoder-LM
 * ~0.3–2.5 GB) share one memory budget. Loading a different heavyweight frees
 * the others; `keep` names the session being activated.
 */
export async function releaseSessions(keep: HeavySession | 'none'): Promise<void> {
  const jobs: Promise<void>[] = [];
  if (keep !== 'jev') jobs.push(disposeJev());
  if (keep !== 'decide') jobs.push(disposeDecide());
  if (keep !== 'decoder') jobs.push(disposeDecoder());
  await Promise.all(jobs);
}
