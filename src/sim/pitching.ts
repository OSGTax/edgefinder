import { GRAVITY, MPH, type Vec3 } from '../engine/math';
import type { Rng } from '../engine/rng';
import type { Kid, PitchType, Special } from '../data/types';

export const PLATE_Y = 0;

export interface PitchDef {
  label: string;
  short: string;
  speed: number;     // multiplier on the pitcher's fastball
  breakX: number;    // ft/s², glove-side positive (sign flipped by pitcher hand)
  breakZ: number;    // ft/s², positive = fights gravity ("rise")
  control: number;   // extra aim scatter, ft
  wobble: number;    // knuckleball dance amplitude, ft
}

export const PITCHES: Record<PitchType, PitchDef> = {
  fastball: { label: 'Fastball', short: 'FB', speed: 1, breakX: -2, breakZ: 7, control: 0, wobble: 0 },
  curve: { label: 'Curveball', short: 'CB', speed: 0.78, breakX: 11, breakZ: -13, control: 0.06, wobble: 0 },
  changeup: { label: 'Changeup', short: 'CH', speed: 0.8, breakX: -6, breakZ: -5, control: 0.03, wobble: 0 },
  slider: { label: 'Slider', short: 'SL', speed: 0.88, breakX: 12, breakZ: -3, control: 0.05, wobble: 0 },
  sinker: { label: 'Sinker', short: 'SI', speed: 0.94, breakX: -8, breakZ: -9, control: 0.03, wobble: 0 },
  knuckler: { label: 'Knuckleball', short: 'KN', speed: 0.68, breakX: 0, breakZ: -2, control: 0.1, wobble: 0.38 },
};

export interface ActivePitch {
  type: PitchType;
  special: Special | null;
  p0: Vec3;
  v0: Vec3;
  a: Vec3;
  /** physics time to reach the plate */
  T: number;
  /** wall-clock duration (differs from T for the Brain Freeze) */
  Treal: number;
  warp: number;
  wob: { ax: number; az: number; fx: number; fz: number; px: number; pz: number };
  target: { x: number; z: number };
  arrival: Vec3;
  mph: number;
}

export function fastballMph(k: Kid) {
  return 40 + k.traits.pitching * 2.3 + k.traits.arm * 0.4;
}

export function releasePoint(k: Kid, moundDist: number): Vec3 {
  const hand = k.throws === 'R' ? 1 : -1;
  return { x: -1.6 * hand, y: moundDist - 3.2, z: 4.3 + k.look.height * 0.8 };
}

/** Pitcher's aim scatter in feet (1σ). */
export function pitchScatter(k: Kid, type: PitchType, fatigue: number) {
  return 0.33 + (10 - k.traits.control) * 0.065 + PITCHES[type].control + fatigue * 0.3;
}

export function makePitch(
  k: Kid,
  type: PitchType,
  aim: { x: number; z: number },
  special: Special | null,
  moundDist: number,
  rng: Rng,
  fatigue = 0,
  accuracy?: number,
): ActivePitch {
  const def = PITCHES[type];
  const hand = k.throws === 'R' ? 1 : -1;
  let mph = fastballMph(k) * def.speed;
  let breakX = def.breakX * hand;
  let breakZ = def.breakZ;
  let wobble = def.wobble;
  let scatter = pitchScatter(k, type, fatigue);
  let warp = 0;
  if (special === 'heater') { mph *= 1.28; breakZ = 9; scatter *= 0.7; }
  if (special === 'wobbler') { mph *= 0.9; wobble = 1.05; scatter *= 0.6; }
  if (special === 'loopy') { mph = 24; breakZ = 0; breakX = 0; scatter *= 0.6; }
  if (special === 'freeze') { warp = 0.93; scatter *= 0.7; }
  if (accuracy !== undefined) {
    // the player's pitch meter: a perfect release is tighter than the kid's
    // natural control and pops a little; a bad one sprays
    const a = Math.min(1, Math.max(0, accuracy));
    scatter *= 0.55 + (1 - a) * 1.25;
    mph *= 0.97 + a * 0.05;
  }

  const target = {
    x: aim.x + rng.gauss() * scatter,
    z: aim.z + rng.gauss() * scatter * 0.85,
  };
  const p0 = releasePoint(k, moundDist);
  const speed = mph * MPH;
  const T = (p0.y - PLATE_Y) / speed;
  const a = { x: breakX, y: 0, z: -GRAVITY + breakZ };
  const v0 = {
    x: (target.x - p0.x - 0.5 * a.x * T * T) / T,
    y: -speed,
    z: (target.z - p0.z - 0.5 * a.z * T * T) / T,
  };
  const wob = {
    ax: wobble * rng.range(0.6, 1), az: wobble * rng.range(0.4, 0.8),
    fx: rng.range(1.2, 2.4), fz: rng.range(1, 2),
    px: rng.range(0, Math.PI * 2), pz: rng.range(0, Math.PI * 2),
  };
  const Treal = warp > 0 ? T * 1.35 : T;
  const pitch: ActivePitch = { type, special, p0, v0, a, T, Treal, warp, wob, target, arrival: { x: 0, y: 0, z: 0 }, mph };
  pitch.arrival = pitchPos(pitch, Treal);
  return pitch;
}

/** Physics time for a given wall-clock time (handles the Brain Freeze stall). */
function physT(p: ActivePitch, t: number) {
  if (p.warp <= 0) return t;
  // fast out of the hand, nearly stops halfway, then zips to the plate
  const u = Math.min(1, t / p.Treal);
  const extra = t > p.Treal ? t - p.Treal : 0;
  return p.T * (u + (p.warp * Math.sin(2 * Math.PI * u)) / (2 * Math.PI)) + extra;
}

export function pitchPos(p: ActivePitch, t: number): Vec3 {
  const tp = physT(p, t);
  const s = Math.min(1, Math.max(0, tp / p.T));
  const wx = p.wob.ax * Math.sin(p.wob.fx * 2 * Math.PI * s + p.wob.px) * s;
  const wz = p.wob.az * Math.sin(p.wob.fz * 2 * Math.PI * s + p.wob.pz) * s;
  return {
    x: p.p0.x + p.v0.x * tp + 0.5 * p.a.x * tp * tp + wx,
    y: p.p0.y + p.v0.y * tp,
    z: p.p0.z + p.v0.z * tp + 0.5 * p.a.z * tp * tp + wz,
  };
}

export function isStrike(arr: Vec3, zone: { bottom: number; top: number; half: number }) {
  return Math.abs(arr.x) <= zone.half && arr.z >= zone.bottom - 0.12 && arr.z <= zone.top + 0.12;
}
