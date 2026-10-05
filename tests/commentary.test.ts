import { describe, expect, it } from 'vitest';
import { kid } from '../src/data/kids';
import { team } from '../src/data/teams';
import { yard } from '../src/data/yards';
import { autoLineup } from '../src/sim/lineup';
import { Match } from '../src/sim/match';
import { Booth, type Line } from '../src/ui/commentary';

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
    ...[11, 22, 33].map((s) => ({ seed: s, innings: 6 })),
    ...[44, 55].map((s) => ({ seed: s, innings: 3 })),
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
      const counts = new Map<string, number>();
      for (const l of lines) counts.set(l.text, (counts.get(l.text) ?? 0) + 1);
      const [top, most] = [...counts].sort((x, y) => y[1] - x[1])[0];
      expect(most / lines.length, `"${top}" said ${most} times`).toBeLessThanOrEqual(0.1);
      // never the same line twice in a row
      for (let i = 1; i < lines.length; i++) expect(lines[i].text, `repeat at ${i}`).not.toBe(lines[i - 1].text);
      // and mostly fresh material
      expect(counts.size / lines.length).toBeGreaterThan(0.6);
    });
  }

  it('is deterministic from the seed', () => {
    const x = broadcast(77, 3).lines.map((l) => l.text);
    const y = broadcast(77, 3).lines.map((l) => l.text);
    expect(x).toEqual(y);
  });
});
