import { describe, expect, it } from 'vitest';
import { Rng } from '../src/engine/rng';
import { kid } from '../src/data/kids';
import { team } from '../src/data/teams';
import { yard } from '../src/data/yards';
import { autoLineup } from '../src/sim/lineup';
import { Match } from '../src/sim/match';
import { makePitch } from '../src/sim/pitching';
import type { MatchEvent } from '../src/sim/types';

// The phone controls lean on a few sim hooks: the pitch meter's accuracy,
// the timing read on every swing, and tapping through the pauses.

function spread(accuracy: number | undefined) {
  const rng = new Rng(7);
  const p = kid(team('comets').roster[0]);
  let sum = 0;
  for (let i = 0; i < 400; i++) {
    const pitch = makePitch(p, 'fastball', { x: 0, z: 2 }, null, 44, rng, 0, accuracy);
    sum += Math.hypot(pitch.target.x, pitch.target.z - 2);
  }
  return sum / 400;
}

function game(human: 'away' | 'home') {
  const a = team('mudcats'), h = team('comets');
  return new Match({
    away: { team: a, lineup: autoLineup(a.roster.map(kid)), human: human === 'away' },
    home: { team: h, lineup: autoLineup(h.roster.map(kid)), human: human === 'home' },
    yard: yard(h.yardId), innings: 3, seed: 5, difficulty: 'rookie',
  });
}

describe('phone control hooks', () => {
  it('a perfect pitch-meter release hits the spot better than a wild one', () => {
    const perfect = spread(1), plain = spread(undefined), wild = spread(0);
    expect(perfect).toBeLessThan(plain);
    expect(plain).toBeLessThan(wild);
  });

  it('every human swing reports its timing', () => {
    const m = game('away');
    const reads: MatchEvent[] = [];
    let guard = 0;
    while (reads.length < 12 && guard++ < 60000 && m.phase !== 'over') {
      if (m.humanBatting && m.phase === 'pitch' && !m.swingIn && m.pitchT > 0.2) m.swing(m.pitch!.arrival.x, m.pitch!.arrival.z, 'normal');
      m.update(1 / 60);
      for (const e of m.events) if (e.type === 'contact' || e.type === 'whiff' || e.type === 'foulTip') reads.push(e);
      m.events.length = 0;
    }
    expect(reads.length).toBeGreaterThan(5);
    for (const e of reads) expect('read' in e && e.read && Number.isFinite(e.read.timing)).toBe(true);
  });

  it('tapping skips a result hold', () => {
    const m = game('away');
    let guard = 0;
    while (m.phase !== 'result' && guard++ < 10000) { m.update(1 / 60); m.events.length = 0; }
    expect(m.phase).toBe('result');
    m.update(0.3);
    m.skip();
    m.update(1 / 60);
    expect(m.phase).not.toBe('result');
  });
});
