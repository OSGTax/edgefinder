import type { Kid, Position } from '../data/types';
import { FIELD_ORDER } from './play';

export interface Lineup {
  /** batting order: kid ids */
  order: string[];
  /** defense in FIELD_ORDER (P, C, 1B, 2B, 3B, SS, LF, CF, RF) */
  defense: string[];
}

type Scorer = (k: Kid) => number;

const FIT: Record<Position, Scorer> = {
  P: (k) => k.stats.pitching * 3 + k.stats.arm,
  C: (k) => k.stats.arm * 1.4 + k.stats.fielding * 1.2 - k.stats.speed * 0.4,
  SS: (k) => k.stats.fielding * 1.4 + k.stats.arm + k.stats.speed * 0.8,
  '2B': (k) => k.stats.fielding * 1.3 + k.stats.speed * 0.8,
  '3B': (k) => k.stats.arm * 1.3 + k.stats.fielding,
  CF: (k) => k.stats.speed * 1.5 + k.stats.fielding,
  RF: (k) => k.stats.arm * 1.2 + k.stats.speed * 0.6 + k.stats.fielding * 0.5,
  LF: (k) => k.stats.speed * 0.7 + k.stats.fielding * 0.7,
  '1B': (k) => k.stats.fielding * 0.6 + k.stats.power * 0.3 - k.stats.speed * 0.2,
};

// fill the hard spots first
const FILL_ORDER: Position[] = ['P', 'C', 'SS', 'CF', '2B', '3B', 'RF', 'LF', '1B'];

export function autoDefense(kids: Kid[]): string[] {
  const pool = [...kids];
  const chosen: Partial<Record<Position, Kid>> = {};
  for (const pos of FILL_ORDER) {
    let best = 0;
    for (let i = 1; i < pool.length; i++) if (FIT[pos](pool[i]) > FIT[pos](pool[best])) best = i;
    chosen[pos] = pool.splice(best, 1)[0];
  }
  return FIELD_ORDER.map((p) => chosen[p]!.id);
}

const bat = (k: Kid) => k.stats.contact * 1.1 + k.stats.power;

export function autoOrder(kids: Kid[]): string[] {
  const pool = [...kids];
  const take = (score: Scorer) => {
    let best = 0;
    for (let i = 1; i < pool.length; i++) if (score(pool[i]) > score(pool[best])) best = i;
    return pool.splice(best, 1)[0];
  };
  const order: Kid[] = [];
  order[0] = take((k) => k.stats.speed * 1.2 + k.stats.contact);
  order[2] = take(bat);
  order[3] = take((k) => k.stats.power * 1.5 + k.stats.contact * 0.5);
  order[1] = take((k) => k.stats.contact * 1.4 + k.stats.speed * 0.4);
  order[4] = take((k) => k.stats.power + k.stats.contact * 0.6);
  for (let i = 5; i < 9; i++) order[i] = take(bat);
  return order.map((k) => k.id);
}

export function autoLineup(kids: Kid[]): Lineup {
  return { order: autoOrder(kids), defense: autoDefense(kids) };
}
