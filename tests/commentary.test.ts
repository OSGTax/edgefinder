import { describe, expect, it } from 'vitest';
import { KIDS, kid } from '../src/data/kids';
import { team } from '../src/data/teams';
import { yard } from '../src/data/yards';
import { autoLineup } from '../src/sim/lineup';
import { Match } from '../src/sim/match';
import { Booth, shape, type Line } from '../src/ui/commentary';

/** Plays a CPU game step by step, feeding every event to a Booth like the game screen does. */
function broadcast(seed: number, innings: number): { lines: Line[]; perEvent: number[] } {
  const a = team('mudcats'), h = team('comets');
  const m = new Match({
    away: { team: a, lineup: autoLineup(a.roster.map(kid)), human: false },
    home: { team: h, lineup: autoLineup(h.roster.map(kid)), human: false },
    yard: yard(h.yardId),
    innings,
    seed,
    difficulty: 'pro',
    fast: true,
  });
  const booth = new Booth(seed);
  const lines: Line[] = [];
  const perEvent: number[] = [];
  const say = () => {
    for (const e of m.events) {
      const out = booth.react(e, m);
      perEvent.push(out.length);
      lines.push(...out);
    }
    m.events.length = 0;
  };
  say(); // the first batterUp is emitted by the constructor
  let steps = 0;
  while (m.phase !== 'over' && steps++ < 400000) {
    m.update(1 / 30);
    say();
  }
  expect(m.phase).toBe('over');
  return { lines, perEvent };
}

describe('announcer booth', () => {
  const games = [
    ...[11, 22, 33, 66, 88].map((s) => ({ seed: s, innings: 6 })),
    ...[44, 55].map((s) => ({ seed: s, innings: 3 })),
    { seed: 99, innings: 9 },
  ];

  for (const g of games) {
    it(`calls a ${g.innings}-inning game (seed ${g.seed}) with short, varied lines`, () => {
      const { lines, perEvent } = broadcast(g.seed, g.innings);
      expect(lines.length).toBeGreaterThan(g.innings * 8);
      for (const n of perEvent) expect(n).toBeLessThanOrEqual(2);
      for (const l of lines) {
        expect(['Chet', 'Dottie']).toContain(l.who);
        expect(l.text.trim().length).toBeGreaterThan(0);
        expect(l.text.length, l.text).toBeLessThanOrEqual(140);
        expect(l.text).not.toMatch(/undefined|NaN|\[object/);
      }
      // never the same line twice in a game, not even with different names in it
      const seen = new Map<string, string>();
      for (const l of lines) {
        const sh = shape(l.text);
        expect(seen.has(sh), `said twice: "${seen.get(sh)}" / "${l.text}"`).toBe(false);
        seen.set(sh, l.text);
      }
      // and mostly about somebody or something in particular: a kid, a team, the yard, the score
      const names = KIDS.flatMap((k) => [k.nick, k.first]);
      const specific = /Mendoza|grill|pool|splash|flamingo|picket|hedge|gnome|Biscuit|Bea|tee-ball|Channel|Mudcats|Comets|MUD|COM|juice|streetlights|lawn/;
      const about = lines.filter((l) => names.some((n) => l.text.includes(n)) || specific.test(l.text)).length;
      expect(about / lines.length, 'too many generic lines').toBeGreaterThan(0.8);
    });
  }

  it('treats a line with different names in it as the same line', () => {
    expect(shape('From right here on Cedar Lane... it\'s Kaboom!')).toBe(shape('From right here on Maple Street... it\'s Gus-Gus!'));
    expect(shape('Strikeout number 4 for Inny.')).toBe(shape('Strikeout number 6 for Pepper.'));
    expect(shape('Base hit for Bo!')).not.toBe(shape('Two-bagger for Bo!'));
  });

  it('is deterministic from the seed', () => {
    const x = broadcast(77, 3).lines.map((l) => l.text);
    const y = broadcast(77, 3).lines.map((l) => l.text);
    expect(x).toEqual(y);
  });
});
