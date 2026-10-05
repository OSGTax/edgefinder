import type { Kid, Position } from '../data/types';
import { FIELD_ORDER } from './play';

export interface Lineup {
  /** batting order: kid ids */
  order: string[];
  /** defense in FIELD_ORDER (P, C, 1B, 2B, 3B, SS, LF, CF, RF) */
  defense: string[];
}

type Scorer = (k: Kid) => number;

// How well each kid fits each spot, from their traits.
const FIT: Record<Position, Scorer> = {
  P: (k) => k.traits.pitching * 2 + k.traits.control * 1.2 + k.traits.arm * 0.3,
  C: (k) => k.traits.fielding * 1.2 + k.traits.arm * 0.8 - k.traits.speed * 0.5,
  SS: (k) => k.traits.fielding * 1.2 + k.traits.speed * 0.9 + k.traits.arm * 0.5,
  CF: (k) => k.traits.speed * 1.6 + k.traits.fielding * 0.8,
  '2B': (k) => k.traits.fielding * 1.2 + k.traits.speed * 0.7,
  '3B': (k) => k.traits.fielding * 1.1 + k.traits.arm * 0.8 + k.traits.power * 0.2,
  RF: (k) => k.traits.fielding * 0.7 + k.traits.arm * 0.8 + k.traits.speed * 0.5,
  LF: (k) => k.traits.speed * 0.7 + k.traits.fielding * 0.6,
  '1B': (k) => (k.traits.contact + k.traits.power) * 0.25 + k.traits.fielding * 0.4 - k.traits.speed * 0.3,
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

export function autoOrder(kids: Kid[]): string[] {
  const pool = [...kids];
  const take = (score: Scorer) => {
    let best = 0;
    for (let i = 1; i < pool.length; i++) if (score(pool[i]) > score(pool[best])) best = i;
    return pool.splice(best, 1)[0];
  };
  const order: Kid[] = [];
  order[3] = take((k) => k.traits.power * 1.6 + k.traits.contact * 0.4 - k.traits.speed * 0.2); // cleanup slugger
  order[0] = take((k) => k.traits.speed * 1.3 + k.traits.contact * 0.9); // leadoff: speed + contact
  order[2] = take((k) => k.traits.contact + k.traits.power * 0.8 + k.traits.speed * 0.2);
  order[1] = take((k) => k.traits.contact * 1.3 + k.traits.speed * 0.5);
  order[4] = take((k) => k.traits.power + k.traits.contact * 0.5);
  for (let i = 5; i < 9; i++) order[i] = take((k) => (k.traits.contact + k.traits.power) * 0.6 + k.traits.speed * 0.3);
  return order.map((k) => k.id);
}

export function autoLineup(kids: Kid[]): Lineup {
  return { order: autoOrder(kids), defense: autoDefense(kids) };
}
