import { describe, expect, it } from 'vitest';
import { Rng } from '../src/engine/rng';
import { kid } from '../src/data/kids';
import { team } from '../src/data/teams';
import { yard } from '../src/data/yards';
import { autoLineup } from '../src/sim/lineup';
import { Match } from '../src/sim/match';
import { SWING_TIME } from '../src/sim/batting';
import type { Difficulty } from '../src/sim/types';

/**
 * A pretend human at the plate: perfect aim (Rookie aim assist), swing
 * timing with a person's typical error, and the CPU handling everything
 * else. Checks that hitting is learnable and that difficulty matters.
 */
function humanAtBats(difficulty: Difficulty, timingSigma: number, aimSigma: number, games = 4) {
  const tally = { pitches: 0, swings: 0, contact: 0, pa: 0, h: 0, ab: 0, so: 0, hr: 0 };
  const rng = new Rng(99);
  for (let g = 0; g < games; g++) {
    const a = team('mudcats'), h = team('comets');
    const m = new Match({
      away: { team: a, lineup: autoLineup(a.roster.map(kid)), human: true },
      home: { team: h, lineup: autoLineup(h.roster.map(kid)), human: false },
      yard: yard(h.yardId), innings: 6, seed: 1000 + g, difficulty, fast: true,
    });
    let plannedT: number | null = null;
    let plannedFor: unknown = null;
    let guard = 0;
    while (m.phase !== 'over' && guard++ < 400000) {
      if (m.humanPitching && m.phase === 'prePitch') m.selectPitch(m.pitcher.pitches[0], { x: rng.range(-0.5, 0.5), z: 2 });
      if (m.humanBatting && m.phase === 'pitch' && m.pitch && plannedFor !== m.pitch) {
        plannedFor = m.pitch;
        tally.pitches++;
        const arr = m.pitch.arrival, z = m.zone;
        const looksGood = Math.abs(arr.x) < z.half + 0.25 && arr.z > z.bottom - 0.25 && arr.z < z.top + 0.25;
        plannedT = looksGood || m.strikes === 2 ? m.pitch.Treal - SWING_TIME.normal + rng.gauss() * timingSigma : null;
      }
      if (plannedT !== null && m.phase === 'pitch' && m.pitchT >= plannedT && !m.swingIn) {
        const arr = m.pitch!.arrival;
        m.swing(arr.x + rng.gauss() * aimSigma, arr.z + rng.gauss() * aimSigma, 'normal');
        tally.swings++;
        plannedT = null;
      }
      const wasBatting = m.humanBatting;
      m.update(1 / 60);
      for (const e of m.events) if (e.type === 'contact' && wasBatting) tally.contact++;
      m.events.length = 0;
    }
    for (const id of m.cfg.away.lineup.order) {
      const b = m.box[id].bat;
      tally.pa += b.pa; tally.h += b.h; tally.ab += b.ab; tally.so += b.so; tally.hr += b.hr;
    }
  }
  return { ...tally, contactRate: tally.contact / Math.max(1, tally.swings), avg: tally.h / Math.max(1, tally.ab), kRate: tally.so / Math.max(1, tally.pa) };
}

describe('a human at the plate', () => {
  it('can hit on Rookie with ordinary timing', () => {
    const r = humanAtBats('rookie', 0.05, 0.05);
    console.log('rookie, ±50ms', r);
    expect(r.contactRate).toBeGreaterThan(0.55);
    expect(r.avg).toBeGreaterThan(0.25);
  }, 120_000);

  it('is harder on All-Star with the same timing and no aim help', () => {
    const rookie = humanAtBats('rookie', 0.05, 0.05);
    const allstar = humanAtBats('allstar', 0.05, 0.3);
    console.log('all-star, ±50ms, ±0.3ft aim', allstar);
    expect(allstar.contactRate).toBeLessThan(rookie.contactRate);
  }, 120_000);
});
