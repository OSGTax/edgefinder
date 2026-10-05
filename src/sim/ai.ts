import { clamp } from '../engine/math';
import type { Rng } from '../engine/rng';
import type { Kid, PitchType } from '../data/types';
import { SPECIAL_INFO } from '../data/types';
import { SWING_TIME, type SwingInput, type SwingKind } from './batting';
import { isStrike, PITCHES, type ActivePitch } from './pitching';
import type { Difficulty } from './types';

export interface Count { balls: number; strikes: number; outs: number }
export interface Zone { bottom: number; top: number; half: number }

const SKILL: Record<Difficulty, number> = { rookie: 1.4, pro: 1, allstar: 0.78 };

export interface PitchPlan { type: PitchType; aim: { x: number; z: number }; special: boolean }

export function cpuChoosePitch(p: Kid, count: Count, zone: Zone, hypeFull: boolean, diff: Difficulty, rng: Rng): PitchPlan {
  const pitches = p.pitches.length ? p.pitches : (['fastball'] as PitchType[]);
  const weights = pitches.map((t, i) => {
    let w = i === 0 ? 1.6 : 1;
    if (count.balls >= 3 && t !== 'fastball') w *= 0.4;
    if (count.strikes === 2 && t !== 'fastball') w *= 1.5;
    return w;
  });
  let r = rng.next() * weights.reduce((a, b) => a + b, 0);
  let type = pitches[0];
  for (let i = 0; i < pitches.length; i++) { r -= weights[i]; if (r <= 0) { type = pitches[i]; break; } }

  const mid = (zone.bottom + zone.top) / 2;
  const h = zone.top - zone.bottom;
  let aim: { x: number; z: number };
  if (count.balls === 3) aim = { x: rng.range(-0.35, 0.35), z: mid + rng.range(-0.2, 0.2) * h };
  else if (count.strikes === 2 && count.balls < 2 && rng.chance(0.45)) {
    // waste one off the edge
    aim = { x: (rng.chance(0.5) ? 1 : -1) * rng.range(0.95, 1.25), z: rng.chance(0.5) ? zone.bottom - 0.25 : mid };
  } else {
    aim = { x: (rng.chance(0.5) ? 1 : -1) * rng.range(0.3, 0.8), z: mid + rng.range(-0.42, 0.42) * h };
  }
  // easier CPU pitchers miss more over the middle
  const s = SKILL[diff];
  aim.x /= s > 1 ? 1.15 : 1;
  const special = hypeFull && SPECIAL_INFO[p.special].kind === 'pitch' && rng.chance(0.4);
  return { type, aim, special };
}

/**
 * CPU batter: read the pitch (with error), decide whether to swing, and pick
 * when and where. Returns null for a take.
 */
export function cpuDecideSwing(
  b: Kid, pitch: ActivePitch, count: Count, zone: Zone, hypeFull: boolean, diff: Difficulty, rng: Rng,
): SwingInput | null {
  const s = SKILL[diff];
  const def = PITCHES[pitch.type];
  const readErr = (0.12 + (10 - b.traits.hitting) * 0.018 + Math.abs(def.breakX) * 0.004 + def.wobble * 0.5) * s;
  const seenX = pitch.arrival.x + rng.gauss() * readErr;
  const seenZ = pitch.arrival.z + rng.gauss() * readErr * 0.9;
  const looksStrike = isStrike({ x: seenX, y: 0, z: seenZ }, zone);
  const offBy = Math.max(Math.abs(seenX) - zone.half, zone.bottom - seenZ, seenZ - zone.top, 0);

  let pSwing: number;
  if (looksStrike) pSwing = count.strikes === 2 ? 0.94 : count.balls === 3 && count.strikes < 2 ? 0.55 : 0.8;
  else {
    pSwing = offBy > 1 ? 0.03 : clamp(0.24 - offBy * 0.3 + (10 - b.traits.hitting) * 0.012, 0.02, 0.4);
    if (count.strikes === 2) pSwing += 0.15;
    if (count.balls === 3) pSwing *= 0.5;
  }
  if (!rng.chance(pSwing)) return null;

  let kind: SwingKind = 'normal';
  if (b.traits.hitting >= 7 && count.balls > count.strikes && rng.chance(0.45)) kind = 'power';
  if (b.traits.hitting <= 3 && b.traits.speed >= 8 && count.strikes < 2 && rng.chance(0.07)) kind = 'bunt';

  // timing: slow stuff fools you early, heat gets you late
  let bias = 0;
  if (pitch.type === 'changeup' || pitch.type === 'curve' || pitch.type === 'knuckler') bias -= 0.03 * (1.2 - b.traits.hitting / 10);
  if (pitch.special === 'loopy' || pitch.special === 'freeze') bias -= 0.07;
  if (pitch.special === 'heater' || pitch.mph > 62) bias += 0.025;
  const sigmaT = (0.04 + (10 - b.traits.hitting) * 0.0055) * s;
  const delta = bias + rng.gauss() * sigmaT;
  const special = hypeFull && SPECIAL_INFO[b.special].kind === 'bat' && rng.chance(0.45);
  const tSwing = kind === 'bunt' ? Math.max(0, pitch.Treal - 0.4) : pitch.Treal - SWING_TIME[kind] + delta;
  return {
    aimX: seenX + rng.gauss() * 0.06,
    aimZ: seenZ + rng.gauss() * 0.06,
    tSwing,
    kind,
    special,
  };
}
