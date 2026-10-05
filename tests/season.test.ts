import { describe, expect, it } from 'vitest';
import { TEAMS } from '../src/data/teams';
import { createSeason, currentDay, inPlayoffs, leaders, nextUserGame, simThrough, standings } from '../src/league/season';

describe('season', () => {
  it('builds a fair round-robin schedule', () => {
    const s = createSeason('mudcats', { innings: 3, difficulty: 'pro', rounds: 2, seed: 9 });
    expect(s.games.length).toBe(56); // 28 pairings x 2
    const days = new Set(s.games.map((g) => g.day));
    expect(days.size).toBe(14);
    for (const d of days) {
      const todays = s.games.filter((g) => g.day === d);
      const teams = todays.flatMap((g) => [g.away, g.home]);
      expect(new Set(teams).size).toBe(8); // everybody plays exactly once a day
    }
    for (const t of TEAMS) {
      const mine = s.games.filter((g) => g.away === t.id || g.home === t.id);
      expect(mine.length).toBe(14);
      expect(mine.filter((g) => g.home === t.id).length).toBe(7);
    }
  });

  it('plays a whole season through the Lemonade Cup', () => {
    const s = createSeason('owls', { innings: 3, difficulty: 'pro', rounds: 1, seed: 4 });
    let guard = 0;
    while (!s.champion && guard++ < 20) simThrough(s, currentDay(s), true);
    expect(inPlayoffs(s)).toBe(true);
    expect(s.champion).not.toBeNull();
    expect(nextUserGame(s)).toBeNull();
    const st = standings(s);
    const total = [...st['Front Porch'], ...st['Back Fence']].reduce((a, r) => a + r.w + r.l + r.t, 0);
    expect(total).toBe(56); // 28 games, two sides each
    expect(leaders(s, 'hr').length).toBeGreaterThan(0);
    expect(leaders(s, 'avg').length).toBeGreaterThan(0);
  }, 60_000);
});
