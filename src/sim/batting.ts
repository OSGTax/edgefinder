import { clamp, DEG, MPH, type Vec3 } from '../engine/math';
import type { Rng } from '../engine/rng';
import type { Bats, Hand, Kid } from '../data/types';
import { pitchPos, type ActivePitch } from './pitching';

export type SwingKind = 'normal' | 'power' | 'bunt';

export interface SwingInput {
  /** sweet-spot aim at the plate, feet (x across, z height) */
  aimX: number;
  aimZ: number;
  /** wall-clock seconds after release when the swing started */
  tSwing: number;
  kind: SwingKind;
  special: boolean;
}

export type SwingOutcome =
  | { kind: 'miss'; timing: number; dist: number }
  | { kind: 'foulTip' }
  | { kind: 'contact'; contactPoint: Vec3; v: Vec3; ev: number; la: number; spray: number; quality: number };

export const SWING_TIME: Record<SwingKind, number> = { normal: 0.15, power: 0.18, bunt: 0 };

/** Which side a switch hitter bats from. */
export const batSide = (bats: Bats, pitcherHand: Hand): 'R' | 'L' =>
  bats === 'S' ? (pitcherHand === 'R' ? 'L' : 'R') : bats;

export interface SwingTuning {
  /** human-side forgiveness (rookie > 1) */
  window: number;
  radius: number;
}

/** Wall-clock time when the bat reaches the hitting zone. */
export const batArrival = (s: SwingInput) => s.tSwing + SWING_TIME[s.kind];

export function contactWindow(k: Kid, s: SwingInput, tune: SwingTuning) {
  let w = (0.05 + k.traits.contact * 0.004) * tune.window;
  let r = (0.3 + k.traits.contact * 0.019) * tune.radius;
  if (s.kind === 'power') { w *= 0.8; r *= 0.8; }
  if (s.kind === 'bunt') { w = 99; r *= 1.35; }
  if (s.special && k.special === 'eagleEye') { w *= 1.6; r *= 1.9; }
  if (s.special && k.special === 'laser') { w *= 1.2; r *= 1.15; }
  return { w, r };
}

export function resolveSwing(
  k: Kid, side: 'R' | 'L', pitch: ActivePitch, s: SwingInput, tune: SwingTuning, rng: Rng,
): SwingOutcome {
  const { w, r } = contactWindow(k, s, tune);
  const tHit = s.kind === 'bunt' ? pitch.Treal : batArrival(s);
  const delta = tHit - pitch.Treal; // >0 late, <0 early
  if (s.kind === 'bunt' && s.tSwing > pitch.Treal) return { kind: 'miss', timing: 1, dist: 0 };
  const ball = pitchPos(pitch, s.kind === 'bunt' ? pitch.Treal : clamp(tHit, 0, pitch.Treal + 0.2));
  const atPlate = pitch.arrival;
  const dx = atPlate.x - s.aimX;
  const dz = atPlate.z - s.aimZ;
  const dist = Math.hypot(dx, dz * 1.15);
  const tq = Math.abs(delta) / w;
  const sq = dist / r;
  if (tq > 1.35 || sq > 1.22) return { kind: 'miss', timing: delta, dist };
  if (tq > 1 || sq > 1) {
    if (rng.chance(0.75)) return { kind: 'foulTip' };
    return { kind: 'miss', timing: delta, dist };
  }
  // contact hitters get more out of a near-miss: the sweet spot is forgiving
  const miss = (0.5 * tq * tq + 0.45 * sq * sq) * (1.35 - k.traits.contact * 0.07);
  const quality = clamp(1 - miss + rng.gauss() * 0.05, 0.02, 1);
  const pull = side === 'R' ? -1 : 1;
  const dzRel = clamp(dz / r, -1.2, 1.2); // + when the bat is under the ball
  const dxRel = clamp(dx / r, -1.2, 1.2);

  let evMph: number;
  let la: number;
  let spray: number;
  if (s.kind === 'bunt') {
    evMph = 12 + rng.range(0, 10) + k.traits.contact * 0.4;
    la = -14 + dzRel * 14 + rng.gauss() * 5;
    spray = clamp(-s.aimX * 22 + rng.gauss() * 16, -60, 60);
  } else {
    const base = 38 + k.traits.power * 2.6 + k.traits.contact * 1.2 + (s.kind === 'power' ? 6 : 0);
    evMph = base * (0.55 + 0.45 * quality) + pitch.mph * 0.1;
    // good contact hitters square it up: tighter launch angles, more liners
    la = 10 + dzRel * 34 + rng.gauss() * (4 + (10 - k.traits.contact) * 0.6) + (s.kind === 'power' ? 5 : 0);
    spray = pull * (-delta / w) * 34 - pull * dxRel * 8 + rng.gauss() * 9;
    if (s.special && k.special === 'moonshot') {
      evMph = Math.max(evMph, base * 1.05) + 16;
      la = 31 + rng.gauss() * 3;
      spray = clamp(spray, -38, 38);
    }
    if (s.special && k.special === 'laser') {
      evMph = Math.max(evMph, base) + 10;
      la = 11 + rng.gauss() * 3;
    }
    if (s.special && k.special === 'eagleEye') evMph += 4;
  }
  la = clamp(la, -35, 75);
  spray = clamp(spray, -88, 88);
  const ev = evMph * MPH;
  const cl = Math.cos(la * DEG);
  const v = { x: ev * cl * Math.sin(spray * DEG), y: ev * cl * Math.cos(spray * DEG), z: ev * Math.sin(la * DEG) };
  const contactPoint = { x: ball.x * 0.5 + atPlate.x * 0.5, y: Math.max(-1, ball.y), z: Math.max(0.3, ball.z) };
  return { kind: 'contact', contactPoint, v, ev: evMph, la, spray, quality };
}
