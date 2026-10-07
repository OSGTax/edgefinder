import { MathUtils } from 'three';
import { B, BONES, type BoneName } from './rig';

// Poses as flat number arrays, so building a kid's pose every frame costs no
// allocations. A pose holds an Euler rotation (x, y, z, order YXZ) per bone
// plus a hip offset. Poses are authored as readable objects (`PoseDef`) and
// compiled once at load; keyframe tracks sample into a caller-owned buffer.

export type Rot = [number, number, number];
export type PoseDef = Partial<Record<BoneName, Rot>> & { hipX?: number; hipY?: number; hipZ?: number };
export type Pose = Float32Array;

export const NB = BONES.length;
export const HX = NB * 3, HY = HX + 1, HZ = HX + 2;
const LEN = NB * 3 + 3;

const clamp = MathUtils.clamp;
export const smooth = (t: number) => { const c = clamp(t, 0, 1); return c * c * (3 - 2 * c); };

export function newPose(): Pose { return new Float32Array(LEN); }

export function compile(p: PoseDef): Pose {
  const out = newPose();
  for (const n of BONES) {
    const r = p[n];
    if (r) { const i = B[n] * 3; out[i] = r[0]; out[i + 1] = r[1]; out[i + 2] = r[2]; }
  }
  out[HX] = p.hipX ?? 0; out[HY] = p.hipY ?? 0; out[HZ] = p.hipZ ?? 0;
  return out;
}

export function copyPose(dst: Pose, src: Pose): Pose { dst.set(src); return dst; }

/** dst = a + (b − a)·k (dst may be a or b) */
export function lerpPose(dst: Pose, a: Pose, b: Pose, k: number): Pose {
  for (let i = 0; i < LEN; i++) dst[i] = a[i] + (b[i] - a[i]) * k;
  return dst;
}

/** Blend `src` over `dst` by w, only where `mask` (per bone, plus index NB for the hips offset) is set. */
export function blendPose(dst: Pose, src: Pose, w: number, mask?: Uint8Array) {
  if (w <= 0) return;
  for (let b = 0; b < NB; b++) {
    if (mask && !mask[b]) continue;
    const i = b * 3;
    dst[i] += (src[i] - dst[i]) * w; dst[i + 1] += (src[i + 1] - dst[i + 1]) * w; dst[i + 2] += (src[i + 2] - dst[i + 2]) * w;
  }
  if (!mask || mask[NB]) for (let i = HX; i < LEN; i++) dst[i] += (src[i] - dst[i]) * w;
}

/** dst += src·w (where masked) */
export function addPose(dst: Pose, src: Pose, w: number, mask?: Uint8Array) {
  if (w === 0) return;
  for (let b = 0; b < NB; b++) {
    if (mask && !mask[b]) continue;
    const i = b * 3;
    dst[i] += src[i] * w; dst[i + 1] += src[i + 1] * w; dst[i + 2] += src[i + 2] * w;
  }
  if (!mask || mask[NB]) for (let i = HX; i < LEN; i++) dst[i] += src[i] * w;
}

export function set(o: Pose, bone: number, x: number, y: number, z: number) {
  const i = bone * 3; o[i] = x; o[i + 1] = y; o[i + 2] = z;
}
export function add(o: Pose, bone: number, x: number, y: number, z: number) {
  const i = bone * 3; o[i] += x; o[i + 1] += y; o[i + 2] += z;
}

const MIRROR_OF = new Int8Array(NB);
{
  const pairs: [BoneName, BoneName][] = [
    ['eyeL', 'eyeR'], ['lidL', 'lidR'], ['shoulderL', 'shoulderR'], ['armL', 'armR'], ['foreL', 'foreR'], ['handL', 'handR'],
    ['thighL', 'thighR'], ['shinL', 'shinR'], ['footL', 'footR'],
  ];
  for (let b = 0; b < NB; b++) MIRROR_OF[b] = b;
  for (const [l, r] of pairs) { MIRROR_OF[B[l]] = B[r]; MIRROR_OF[B[r]] = B[l]; }
}

/** Left/right mirror (dst must not be src). */
export function mirrorPose(dst: Pose, src: Pose): Pose {
  for (let b = 0; b < NB; b++) {
    const i = b * 3, j = MIRROR_OF[b] * 3;
    dst[j] = src[i]; dst[j + 1] = -src[i + 1]; dst[j + 2] = -src[i + 2];
  }
  dst[HX] = -src[HX]; dst[HY] = src[HY]; dst[HZ] = src[HZ];
  return dst;
}

/** Bone masks: which bones a layer may touch (index NB = the hips offset). */
export function mask(names: (BoneName | 'hipOffset')[]): Uint8Array {
  const m = new Uint8Array(NB + 1);
  for (const n of names) m[n === 'hipOffset' ? NB : B[n]] = 1;
  return m;
}
export const UPPER = mask(['spine', 'chest', 'neck', 'head', 'shoulderL', 'armL', 'foreL', 'handL', 'shoulderR', 'armR', 'foreR', 'handR']);
export const ARMS = mask(['shoulderL', 'armL', 'foreL', 'handL', 'shoulderR', 'armR', 'foreR', 'handR']);
export const ALL = mask([...BONES, 'hipOffset']);

// ─────────────────────────────────────────────────────────────── tracks

/**
 * Easing into a key: 'io' smoothstep (default), 'in' accelerates (wind-ups
 * snapping into a release), 'out' decelerates (settling after a hit),
 * 'lin' constant, 'hold' stays on the previous key then cuts.
 */
export type Ease = 'io' | 'in' | 'out' | 'lin' | 'hold';
const EASE: Record<Ease, (t: number) => number> = {
  io: smooth,
  in: (t) => t * t * t,
  out: (t) => 1 - (1 - t) ** 3,
  lin: (t) => t,
  hold: (t) => (t < 1 ? 0 : 1),
};

export type Frame = [number, PoseDef] | [number, PoseDef, Ease];

/**
 * Keyframes. A bone missing from a key carries over from the nearest key
 * that has it, so a gesture only names the bones it moves; `touched` lists
 * those bones, so the track can be laid over another pose without flattening
 * everything else.
 */
export class Track {
  readonly times: Float32Array;
  readonly poses: Pose[];
  readonly ease: ((t: number) => number)[];
  readonly touched: Uint8Array;
  readonly end: number;

  constructor(frames: Frame[], fill = true) {
    this.times = new Float32Array(frames.map((f) => f[0]));
    this.ease = frames.map((f) => EASE[f[2] ?? 'io']);
    this.end = frames[frames.length - 1][0];
    this.touched = new Uint8Array(NB + 1);
    const defs = frames.map((f) => f[1]);
    for (const d of defs) {
      for (const n of BONES) if (d[n]) this.touched[B[n]] = 1;
      if (d.hipX !== undefined || d.hipY !== undefined || d.hipZ !== undefined) this.touched[NB] = 1;
    }
    if (fill) {
      // carry missing values from the nearest key that defines them
      const filled: PoseDef[] = defs.map((d) => ({ ...d }));
      const keys = [...BONES, 'hipX', 'hipY', 'hipZ'] as (keyof PoseDef)[];
      for (const k of keys) {
        for (let i = 0; i < filled.length; i++) {
          if (filled[i][k] !== undefined) continue;
          let src: PoseDef | undefined;
          for (let j = i - 1; j >= 0 && !src; j--) if (defs[j][k] !== undefined) src = defs[j];
          for (let j = i + 1; j < defs.length && !src; j++) if (defs[j][k] !== undefined) src = defs[j];
          if (src) (filled[i] as Record<string, unknown>)[k] = src[k];
        }
      }
      this.poses = filled.map(compile);
    } else this.poses = defs.map(compile);
  }

  sample(t: number, out: Pose): Pose {
    const n = this.times.length;
    if (t <= this.times[0]) return copyPose(out, this.poses[0]);
    for (let i = 0; i < n - 1; i++) {
      const t1 = this.times[i + 1];
      if (t <= t1) {
        const t0 = this.times[i];
        return lerpPose(out, this.poses[i], this.poses[i + 1], this.ease[i + 1]((t - t0) / (t1 - t0)));
      }
    }
    return copyPose(out, this.poses[n - 1]);
  }
}

/** A track where every key is a full pose (missing bones mean "no rotation"). */
export const fullTrack = (frames: Frame[]) => new Track(frames, false);
