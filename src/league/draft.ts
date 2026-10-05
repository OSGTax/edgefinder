import { Rng } from '../engine/rng';
import { KIDS } from '../data/kids';
import type { Kid } from '../data/types';

/** A single "how good is this kid" number for draft boards (1–10 scale). */
export function overall(k: Kid): number {
  const s = k.stats;
  return (s.contact * 1.25 + s.power + s.speed * 0.8 + s.arm * 0.6 + s.fielding * 0.8 + s.pitching * 0.9) / 5.35;
}

export interface Draft {
  pool: string[];
  mine: string[];
  theirs: string[];
  myTurn: boolean;
}

/** Playground pick: a pool of kids, two captains, alternating picks. */
export function startDraft(seed: number, size = 20): Draft {
  const rng = new Rng(seed);
  const ids = rng.shuffle(KIDS.map((k) => k.id)).slice(0, size);
  return { pool: ids, mine: [], theirs: [], myTurn: rng.chance(0.5) };
}

export function draftDone(d: Draft) {
  return d.mine.length >= 9 && d.theirs.length >= 9;
}

export function pick(d: Draft, id: string) {
  const i = d.pool.indexOf(id);
  if (i < 0) return;
  d.pool.splice(i, 1);
  (d.myTurn ? d.mine : d.theirs).push(id);
  // a full team stops picking; the other captain finishes up
  if (d.mine.length >= 9) d.myTurn = false;
  else if (d.theirs.length >= 9) d.myTurn = true;
  else d.myTurn = !d.myTurn;
}

/** The CPU captain wants a real pitcher early, then the best kid left. */
export function cpuPick(d: Draft, byId: (id: string) => Kid, rng: Rng): string {
  const team = d.theirs.map(byId);
  const hasPitcher = team.some((k) => k.stats.pitching >= 6);
  let best = d.pool[0], bestScore = -Infinity;
  for (const id of d.pool) {
    const k = byId(id);
    let score = overall(k) + rng.range(-0.6, 0.6);
    if (!hasPitcher && k.stats.pitching >= 6) score += 2;
    if (score > bestScore) { bestScore = score; best = id; }
  }
  return best;
}
