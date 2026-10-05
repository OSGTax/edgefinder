import { kid } from '../src/data/kids';
import { team } from '../src/data/teams';
import { yard } from '../src/data/yards';
import { autoLineup } from '../src/sim/lineup';
import { simulateMatch, type Match } from '../src/sim/match';
import type { Difficulty } from '../src/sim/types';

export function sim(awayId: string, homeId: string, seed: number, innings = 6, difficulty: Difficulty = 'pro'): Match {
  const a = team(awayId), h = team(homeId);
  return simulateMatch({
    away: { team: a, lineup: autoLineup(a.roster.map(kid)), human: false },
    home: { team: h, lineup: autoLineup(h.roster.map(kid)), human: false },
    yard: yard(h.yardId),
    innings,
    seed,
    difficulty,
  });
}

export interface Totals {
  games: number; runs: number; ab: number; h: number; hr: number; d: number; t: number;
  bb: number; so: number; pa: number; e: number; ties: number; maxRuns: number; secs: number;
}

export function totals(ms: Match[]): Totals {
  const t: Totals = { games: 0, runs: 0, ab: 0, h: 0, hr: 0, d: 0, t: 0, bb: 0, so: 0, pa: 0, e: 0, ties: 0, maxRuns: 0, secs: 0 };
  for (const m of ms) {
    t.games++;
    t.runs += m.score[0] + m.score[1];
    t.maxRuns = Math.max(t.maxRuns, m.score[0], m.score[1]);
    if (m.winner === -1) t.ties++;
    t.e += m.errors[0] + m.errors[1];
    for (const line of Object.values(m.box)) {
      t.ab += line.bat.ab; t.h += line.bat.h; t.hr += line.bat.hr; t.d += line.bat.d; t.t += line.bat.t;
      t.bb += line.bat.bb; t.so += line.bat.so; t.pa += line.bat.pa;
    }
  }
  return t;
}
