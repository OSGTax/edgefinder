import { describe, expect, it } from 'vitest';
import { TEAMS } from '../src/data/teams';
import { sim, totals } from './helpers';

describe('full-game simulation', () => {
  it('plays complete games without getting stuck, with believable kid-league numbers', () => {
    const games = [];
    const t0 = Date.now();
    for (let seed = 1; seed <= 24; seed++) {
      const [a, h] = seed % 2 ? [TEAMS[0], TEAMS[1]] : [TEAMS[1], TEAMS[0]];
      games.push(sim(a.id, h.id, seed * 7919));
    }
    const secs = (Date.now() - t0) / 1000;
    for (const g of games) expect(g.phase).toBe('over');
    const t = totals(games);
    const avg = t.h / t.ab;
    const perGame = t.runs / t.games / 2;
    const summary = {
      games: t.games,
      runsPerTeamGame: +perGame.toFixed(2),
      avg: +avg.toFixed(3),
      hrPerTeamGame: +(t.hr / t.games / 2).toFixed(2),
      xbhShare: +((t.d + t.t + t.hr) / Math.max(1, t.h)).toFixed(2),
      kRate: +(t.so / t.pa).toFixed(3),
      bbRate: +(t.bb / t.pa).toFixed(3),
      errorsPerGame: +(t.e / t.games).toFixed(2),
      ties: t.ties,
      maxRuns: t.maxRuns,
      secsPerGame: +(secs / t.games).toFixed(3),
    };
    console.log(summary);
    expect(avg).toBeGreaterThan(0.2);
    expect(avg).toBeLessThan(0.42);
    expect(perGame).toBeGreaterThan(2);
    expect(perGame).toBeLessThan(11);
    expect(summary.kRate).toBeLessThan(0.33);
    expect(summary.hrPerTeamGame).toBeGreaterThan(0.1);
  }, 120_000);

  it('is deterministic for a given seed', () => {
    const a = sim('mudcats', 'comets', 42);
    const b = sim('mudcats', 'comets', 42);
    expect(a.score).toEqual(b.score);
    expect(a.line).toEqual(b.line);
  });
});
