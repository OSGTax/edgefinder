import { Euler, MathUtils, Quaternion, Vector3 } from 'three';
import type { KidModel } from './model';
import { BONES, type BoneName } from './rig';
import { pointBone, twoBoneIK } from './ik';
import type { Expression } from './face';

// Procedural animation for the kids. Every frame the game says what a kid
// is doing (a Mode plus a little context) and the animator builds a target
// pose from hand-tuned key poses and cycles, eases each bone toward it, then
// layers on head/eye tracking, blinking, breathing and arm IK (bat grips,
// glove reaches). Poses are authored for right-handers and mirrored.

export type Mode =
  | 'stand' | 'ready' | 'crouch' | 'run' | 'trot' | 'walk' | 'catch' | 'throw' | 'dive' | 'jump' | 'stumble'
  | 'slide' | 'cheer' | 'sad' | 'sit' | 'clap' | 'bat' | 'swing' | 'bunt' | 'windup' | 'follow' | 'out' | 'wave' | 'grill';

export interface AnimInput {
  mode: Mode;
  /** seconds in this mode (swing: since the swing started; windup: 0..WINDUP) */
  t: number;
  /** ground speed, ft/s */
  speed?: number;
  /** world-space point to look at */
  lookAt?: Vector3 | null;
  /** world-space point the glove hand should reach for */
  reach?: Vector3 | null;
  /** jump height (ft) for leaps */
  lift?: number;
  /** batting: aim height of the swing (ft) and swing kind */
  aimZ?: number;
  power?: boolean;
  /** windup: total duration */
  windup?: number;
  /** the kid bats/throws left-handed: mirror everything */
  lefty?: boolean;
  /** seat height when sitting */
  seat?: number;
}

type Rot = [number, number, number];
type Pose = Partial<Record<BoneName, Rot>> & { hipY?: number; hipZ?: number; hipX?: number };

const lerp = MathUtils.lerp;
const clamp = MathUtils.clamp;
const smooth = (t: number) => { const c = clamp(t, 0, 1); return c * c * (3 - 2 * c); };

/** Blend between keyframes [time, pose] at time t (smoothstep between keys). */
function keys(t: number, frames: [number, Pose][]): Pose {
  if (t <= frames[0][0]) return frames[0][1];
  for (let i = 0; i < frames.length - 1; i++) {
    const [t0, a] = frames[i], [t1, b] = frames[i + 1];
    if (t <= t1) return mix(a, b, smooth((t - t0) / (t1 - t0)));
  }
  return frames[frames.length - 1][1];
}

function mix(a: Pose, b: Pose, k: number): Pose {
  const out: Pose = {};
  const names = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const n of names) {
    const va = (a as Record<string, unknown>)[n], vb = (b as Record<string, unknown>)[n];
    if (typeof va === 'number' || typeof vb === 'number') {
      (out as Record<string, number>)[n] = lerp((va as number) ?? 0, (vb as number) ?? 0, k);
    } else {
      const ra = (va as Rot) ?? [0, 0, 0], rb = (vb as Rot) ?? [0, 0, 0];
      (out as Record<string, Rot>)[n] = [lerp(ra[0], rb[0], k), lerp(ra[1], rb[1], k), lerp(ra[2], rb[2], k)];
    }
  }
  return out;
}

const MIRROR: Partial<Record<BoneName, BoneName>> = {
  eyeL: 'eyeR', eyeR: 'eyeL', lidL: 'lidR', lidR: 'lidL', shoulderL: 'shoulderR', shoulderR: 'shoulderL',
  armL: 'armR', armR: 'armL', foreL: 'foreR', foreR: 'foreL', handL: 'handR', handR: 'handL',
  thighL: 'thighR', thighR: 'thighL', shinL: 'shinR', shinR: 'shinL', footL: 'footR', footR: 'footL',
};

function mirror(p: Pose): Pose {
  const out: Pose = { hipY: p.hipY, hipZ: p.hipZ, hipX: p.hipX !== undefined ? -p.hipX : undefined };
  for (const n of BONES) {
    const r = p[n];
    if (!r) continue;
    out[MIRROR[n] ?? n] = [r[0], -r[1], -r[2]];
  }
  return out;
}

// ─────────────────────────────────────────────────────────────── key poses
// (right-handed; +x rotation swings hanging limbs back / leans the torso
// forward; armL +z raises the left arm sideways, armR −z the right)

const STAND: Pose = {
  armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0],
  thighL: [0, 0, 0.03], thighR: [0, 0, -0.03],
};

const READY: Pose = {
  hipY: -0.32, hips: [0.35, 0, 0], spine: [0.15, 0, 0], chest: [0.05, 0, 0], neck: [-0.2, 0, 0], head: [-0.35, 0, 0],
  thighL: [-0.75, 0, 0.18], thighR: [-0.75, 0, -0.18], shinL: [1.0, 0, 0], shinR: [1.0, 0, 0], footL: [-0.25, 0, 0], footR: [-0.25, 0, 0],
  armL: [-0.95, 0, 0.12], foreL: [-0.55, 0, 0], armR: [-0.8, 0, -0.12], foreR: [-0.7, 0, 0],
};

const CROUCH: Pose = { // catcher
  hipY: -1.05, hips: [0.15, 0, 0], spine: [0.2, 0, 0], neck: [-0.1, 0, 0], head: [-0.25, 0, 0],
  thighL: [-1.75, 0, 0.42], thighR: [-1.75, 0, -0.42], shinL: [2.25, 0, 0], shinR: [2.25, 0, 0], footL: [-0.5, 0, 0], footR: [-0.5, 0, 0],
  armL: [-1.3, 0, 0.2], foreL: [-0.4, 0, 0], armR: [-0.6, 0, -0.3], foreR: [-1.2, 0, 0],
};

const SAD: Pose = {
  ...STAND, spine: [0.25, 0, 0], chest: [0.15, 0, 0], neck: [0.25, 0, 0], head: [0.45, 0, 0],
  shoulderL: [0, 0, -0.15], shoulderR: [0, 0, 0.15], armL: [-0.15, 0, 0.05], armR: [-0.15, 0, -0.05],
};

const SIT: Pose = {
  thighL: [-1.55, 0, 0.12], thighR: [-1.5, 0, -0.12], shinL: [1.45, 0, 0], shinR: [1.6, 0, 0],
  spine: [0.1, 0, 0], armL: [-0.55, 0, 0.18], armR: [-0.55, 0, -0.18], foreL: [-0.8, 0, 0], foreR: [-0.8, 0, 0],
};

const BAT_STANCE: Pose = {
  hipY: -0.22, hips: [0.12, -0.25, 0], spine: [0.2, -0.1, 0], chest: [0, -0.15, 0], neck: [0, 0.5, 0], head: [0.08, 0.8, 0],
  thighL: [-0.3, 0, 0.32], thighR: [-0.25, 0, -0.32], shinL: [0.55, 0, 0], shinR: [0.5, 0, 0], footL: [-0.1, 0.4, 0], footR: [-0.1, -0.3, 0],
  armL: [-0.9, 0, -0.3], foreL: [-1.2, 0, 0], armR: [-0.5, 0, -0.9], foreR: [-1.6, 0, 0],
};

// ─────────────────────────────────────────────────────────────── cycles

function runPose(t: number, speed: number, trot = false): Pose {
  const a = clamp(speed / (trot ? 26 : 20), trot ? 0.2 : 0.3, trot ? 0.6 : 1.05);
  const freq = 1.25 + speed * 0.055;
  const ph = t * freq * Math.PI * 2;
  const s = Math.sin(ph), c = Math.cos(ph);
  const knee = (x: number) => 0.35 + Math.max(0, x) * 1.5;
  return {
    hipY: -0.06 * a + Math.abs(c) * 0.16 * a, hips: [0.12 * a, s * 0.18 * a, 0], spine: [0.22 * a, -s * 0.12 * a, 0], chest: [0.05, -s * 0.12 * a, 0],
    neck: [-0.15 * a, s * 0.08 * a, 0], head: [-0.15 * a, s * 0.08 * a, 0],
    thighL: [-s * 0.95 * a - 0.15 * a, 0, 0.04], thighR: [s * 0.95 * a - 0.15 * a, 0, -0.04],
    shinL: [knee(c) * a + 0.1, 0, 0], shinR: [knee(-c) * a + 0.1, 0, 0],
    footL: [-0.2 * a + Math.max(0, -s) * 0.4 * a, 0, 0], footR: [-0.2 * a + Math.max(0, s) * 0.4 * a, 0, 0],
    armL: [s * 0.95 * a, 0, 0.18], armR: [-s * 0.95 * a, 0, -0.18],
    foreL: [-1.25 - Math.max(0, -s) * 0.3, 0, 0], foreR: [-1.25 - Math.max(0, s) * 0.3, 0, 0],
  };
}

function idlePose(t: number, seed: number): Pose {
  const w = Math.sin(t * 0.7 + seed) * 0.5 + 0.5; // weight shift
  return {
    ...STAND,
    hips: [0, 0, (w - 0.5) * 0.08], hipX: (w - 0.5) * 0.08,
    spine: [0.02, Math.sin(t * 0.33 + seed) * 0.1, -(w - 0.5) * 0.06],
    thighL: [0, 0, 0.03 + (0.5 - w) * 0.06], thighR: [0, 0, -0.03 + (0.5 - w) * 0.06],
    shinL: [w < 0.5 ? 0.15 : 0.02, 0, 0], shinR: [w > 0.5 ? 0.15 : 0.02, 0, 0],
  };
}

function cheerPose(t: number): Pose {
  const j = Math.abs(Math.sin(t * 6.5));
  return {
    hipY: j * 0.55 - 0.05, spine: [-0.15, 0, 0], neck: [-0.2, 0, 0], head: [-0.3, 0, 0],
    armL: [-0.3, 0, 2.5 + Math.sin(t * 13) * 0.25], armR: [-0.3, 0, -2.5 - Math.sin(t * 13 + 1) * 0.25],
    foreL: [-0.4, 0, 0], foreR: [-0.4, 0, 0],
    thighL: [-0.3 * j, 0, 0.1], thighR: [-0.3 * j, 0, -0.1], shinL: [0.6 * j, 0, 0], shinR: [0.6 * j, 0, 0], footL: [0.5 * j, 0, 0], footR: [0.5 * j, 0, 0],
  };
}

function clapPose(t: number): Pose {
  const k = Math.sin(t * 14) * 0.5 + 0.5;
  return {
    ...STAND, spine: [0.05, 0, 0],
    armL: [-0.95, 0, 0.45 - k * 0.3], armR: [-0.95, 0, -0.45 + k * 0.3], foreL: [-1.0, -0.6, 0], foreR: [-1.0, 0.6, 0],
    hipY: Math.abs(Math.sin(t * 7)) * 0.06,
  };
}

/** A grown-up at the grill: spatula hand flipping, the other on the hip. */
function grillPose(t: number): Pose {
  const flip = Math.max(0, Math.sin(t * 1.3)) ** 6;
  return {
    ...STAND, spine: [0.12, 0, 0], neck: [0.15, 0, 0], head: [0.2, 0, 0],
    armR: [-0.9 - flip * 0.5, 0, -0.2], foreR: [-0.8 + flip * 0.6, flip * 0.8, 0],
    armL: [0.1, 0, 0.55], foreL: [-1.6, 0.4, 0],
  };
}

function wavePose(t: number): Pose {
  return { ...STAND, armR: [-0.2, 0, -2.6], foreR: [0, 0, -0.4 + Math.sin(t * 9) * 0.5] };
}

// ─────────────────────────────────────────────────────────────── actions

function throwPose(t: number): Pose {
  return keys(t, [
    [0, { ...READY, hipY: -0.2 }],
    [0.12, { hipY: -0.12, hips: [0.05, -0.7, 0], spine: [0, -0.3, 0], chest: [-0.1, -0.35, 0], head: [0, 0.9, 0],
      armR: [0.4, 0, -1.6], foreR: [-1.7, 0, 0], armL: [-1.3, 0, 0.5], foreL: [-0.3, 0, 0],
      thighL: [-0.7, 0, 0.1], shinL: [0.9, 0, 0], thighR: [0.1, 0, -0.15], shinR: [0.3, 0, 0] }],
    [0.24, { hipY: -0.28, hips: [0.15, 0.2, 0], spine: [0.3, 0.3, 0], chest: [0.2, 0.35, 0], head: [-0.2, 0.1, 0],
      armR: [-2.4, 0, -0.5], foreR: [-0.4, 0, 0], armL: [-0.3, 0, 0.6], foreL: [-1.3, 0, 0],
      thighL: [-0.8, 0, 0.1], shinL: [0.4, 0, 0], thighR: [0.4, 0, -0.1], shinR: [0.9, 0, 0] }],
    [0.42, { hipY: -0.3, hips: [0.25, 0.45, 0], spine: [0.45, 0.25, 0], chest: [0.2, 0.2, 0], head: [-0.35, -0.1, 0],
      armR: [-0.7, 0, 0.5], foreR: [-0.5, 0, 0], armL: [0.2, 0, 0.4], foreL: [-1.0, 0, 0],
      thighL: [-0.6, 0, 0.1], shinL: [0.6, 0, 0], thighR: [0.6, 0, -0.1], shinR: [1.2, 0, 0], footR: [0.6, 0, 0] }],
    [0.75, READY],
  ]);
}

function divePose(t: number, kid: KidModel): Pose {
  const lay = -kid.p.hipY + 0.55;
  return keys(t, [
    [0, READY],
    [0.18, { hipY: -0.6, hips: [0.9, 0, 0], spine: [0.2, 0, 0], head: [-0.6, 0, 0], armL: [-2.6, 0, 0.2], armR: [-2.4, 0, -0.3], thighL: [-0.4, 0, 0.1], shinL: [0.5, 0, 0], thighR: [0.2, 0, -0.1], shinR: [0.6, 0, 0] }],
    [0.35, { hipY: lay, hipZ: 0.6, hips: [1.45, 0, 0], spine: [0.05, 0, 0], neck: [-0.4, 0, 0], head: [-0.7, 0, 0], armL: [-3.0, 0, 0.15], foreL: [-0.1, 0, 0], armR: [-2.8, 0, -0.3], foreR: [-0.3, 0, 0], thighL: [0.15, 0, 0.12], thighR: [0.25, 0, -0.12], shinL: [0.5, 0, 0], shinR: [0.9, 0, 0], footL: [0.8, 0, 0], footR: [0.8, 0, 0] }],
    [0.95, { hipY: lay, hipZ: 0.6, hips: [1.4, 0, 0.05], spine: [0, 0, 0], neck: [-0.5, 0, 0], head: [-0.7, 0.2, 0], armL: [-2.9, 0, 0.3], armR: [-2.0, 0, -0.6], thighL: [0.2, 0, 0.12], thighR: [0.2, 0, -0.12], shinL: [1.0, 0, 0], shinR: [0.9, 0, 0], footL: [0.8, 0, 0], footR: [0.8, 0, 0] }],
    [1.35, { hipY: -0.7, hips: [0.6, 0, 0], spine: [0.3, 0, 0], head: [-0.4, 0, 0], thighL: [-1.5, 0, 0.2], thighR: [-0.2, 0, -0.2], shinL: [2.0, 0, 0], shinR: [1.6, 0, 0], armL: [-1.2, 0, 0.3], armR: [-0.6, 0, -0.4] }],
    [1.7, READY],
  ]);
}

function jumpPose(t: number): Pose {
  return keys(t, [
    [0, { ...READY, hipY: -0.45 }],
    [0.12, { hipY: 0, spine: [-0.1, 0, 0], head: [-0.6, 0, 0], neck: [-0.3, 0, 0], armL: [-2.6, 0, 0.5], foreL: [-0.2, 0, 0], armR: [-0.6, 0, -0.9], foreR: [-0.6, 0, 0], thighL: [-0.9, 0, 0.1], shinL: [1.5, 0, 0], thighR: [0.2, 0, -0.1], shinR: [1.1, 0, 0], footL: [0.6, 0, 0], footR: [0.7, 0, 0] }],
    [0.5, { hipY: 0, spine: [0, 0, 0], head: [-0.4, 0, 0], armL: [-2.8, 0, 0.3], armR: [-0.4, 0, -0.8], thighL: [-0.5, 0, 0.1], shinL: [0.8, 0, 0], thighR: [-0.2, 0, -0.1], shinR: [0.6, 0, 0] }],
    [0.8, { ...READY, hipY: -0.5 }],
    [1.1, READY],
  ]);
}

function stumblePose(t: number): Pose {
  const f = Math.sin(t * 18);
  return keys(t, [
    [0, READY],
    [0.12, { hipY: -0.15, hips: [-0.15, 0.3, 0.1], spine: [-0.35, 0.2, 0.15], head: [0.3 * f, 0.4, 0.2], armL: [-1.8, 0, 1.2 + f * 0.4], armR: [-1.6, 0, -1.4 - f * 0.4], foreL: [-0.6, 0, 0], foreR: [-0.6, 0, 0], thighL: [-0.6, 0, 0.3], shinL: [0.5, 0, 0], thighR: [0.3, 0, -0.2], shinR: [0.4, 0, 0] }],
    [0.45, { hipY: -0.25, hips: [0.3, -0.2, -0.1], spine: [0.4, -0.2, 0], head: [0.2, -0.3, 0], armL: [-0.4, 0, 1.0], armR: [-0.5, 0, -1.0], thighL: [-0.3, 0, 0.2], shinL: [0.7, 0, 0], thighR: [-0.5, 0, -0.2], shinR: [0.9, 0, 0] }],
    [0.8, READY],
  ]);
}

function slidePose(t: number, kid: KidModel): Pose {
  const low = -kid.p.hipY + 0.55;
  return keys(t, [
    [0, runPose(0.1, 20)],
    [0.15, { hipY: low * 0.5, hips: [-0.6, 0, 0], spine: [0.1, 0, 0], armL: [-0.5, 0, 1.4], armR: [-0.3, 0, -1.6], thighL: [-1.2, 0, 0.1], shinL: [0.2, 0, 0], thighR: [-0.5, 0, -0.2], shinR: [1.6, 0, 0] }],
    [0.35, { hipY: low, hips: [-1.05, 0, 0], spine: [0.35, 0, 0], neck: [0.3, 0, 0], head: [0.5, 0, 0], armL: [-2.6, 0, 0.7], foreL: [-0.4, 0, 0], armR: [-2.4, 0, -0.9], foreR: [-0.5, 0, 0], thighL: [-0.5, 0, 0.12], shinL: [0.05, 0, 0], footL: [0.5, 0, 0], thighR: [-0.1, 0, -0.25], shinR: [1.8, 0, 0], footR: [0.3, 0, 0] }],
    [0.9, { hipY: low, hips: [-1.0, 0, 0], spine: [0.4, 0, 0], neck: [0.2, 0, 0], head: [0.4, 0, 0], armL: [-2.2, 0, 0.6], armR: [-0.6, 0, -0.9], thighL: [-0.5, 0, 0.12], shinL: [0.05, 0, 0], thighR: [-0.1, 0, -0.25], shinR: [1.8, 0, 0] }],
  ]);
}

/** A right-handed swing. t = seconds since the swing started; contact at `tc`. */
function swingPose(t: number, tc: number): Pose {
  return keys(t, [
    [0, { ...BAT_STANCE, hips: [0.12, -0.35, 0], chest: [0, -0.3, 0], thighL: [-0.25, 0, 0.5], shinL: [0.35, 0, 0] }],
    [tc * 0.55, { ...BAT_STANCE, hips: [0.15, 0.25, 0], spine: [0.25, 0.1, 0], chest: [0.05, 0.1, 0], head: [0.15, 0.55, 0], neck: [0, 0.3, 0], thighL: [-0.2, 0, 0.42], shinL: [0.25, 0, 0], thighR: [-0.35, 0, -0.3], shinR: [0.75, 0, 0], footR: [0.1, 0.5, 0] }],
    [tc, { ...BAT_STANCE, hipY: -0.26, hips: [0.15, 0.8, 0], spine: [0.25, 0.25, 0], chest: [0.05, 0.25, 0], head: [0.25, -0.05, 0], neck: [0.1, -0.2, 0], thighL: [-0.15, 0, 0.35], shinL: [0.1, 0, 0], thighR: [-0.45, 0.4, -0.25], shinR: [0.9, 0, 0], footR: [0.25, 0.9, 0] }],
    [tc + 0.12, { ...BAT_STANCE, hipY: -0.2, hips: [0.1, 1.25, 0], spine: [0.15, 0.35, 0], chest: [0, 0.3, 0], head: [0.1, -0.55, 0], neck: [0, -0.4, 0], thighL: [-0.1, 0, 0.3], shinL: [0.05, 0, 0], thighR: [-0.5, 0.6, -0.15], shinR: [1.0, 0, 0], footR: [0.5, 1.1, 0] }],
    [tc + 0.4, { ...BAT_STANCE, hipY: -0.12, hips: [0.05, 1.45, 0], spine: [0.05, 0.3, 0], chest: [-0.05, 0.3, 0], head: [0.05, -0.8, 0], neck: [0, -0.4, 0], thighL: [-0.05, 0, 0.28], shinL: [0.05, 0, 0], thighR: [-0.4, 0.6, -0.1], shinR: [0.8, 0, 0], footR: [0.6, 1.2, 0] }],
  ]);
}

/** A right-handed windup + delivery. u = 0..1 over the windup (release at 1), then follow-through seconds after. */
function windupPose(u: number): Pose {
  return keys(u, [
    [0, { ...STAND, head: [0.05, 0, 0], armL: [-0.75, 0, 0.35], foreL: [-1.3, -0.4, 0], armR: [-0.75, 0, -0.35], foreR: [-1.3, 0.4, 0] }],
    [0.18, { hipY: -0.05, hips: [0, -0.5, 0], spine: [-0.05, -0.2, 0], head: [0, 0.65, 0], armL: [-1.0, 0, 0.3], foreL: [-1.5, -0.4, 0], armR: [-1.0, 0, -0.3], foreR: [-1.5, 0.4, 0], thighR: [0, 0, -0.05] }],
    [0.45, { hipY: 0.02, hips: [-0.05, -1.25, 0], spine: [-0.12, -0.2, 0], chest: [0, -0.1, 0], head: [0.05, 1.2, 0], neck: [0, 0.2, 0],
      armL: [-1.1, 0, 0.4], foreL: [-1.6, -0.5, 0], armR: [-1.0, 0, -0.4], foreR: [-1.6, 0.5, 0],
      thighL: [-1.55, 0, 0.1], shinL: [1.5, 0, 0], footL: [0.5, 0, 0], thighR: [0.05, 0, -0.05], shinR: [0.25, 0, 0] }],
    [0.72, { hipY: -0.32, hips: [0.1, -0.7, 0], spine: [0.0, -0.35, 0], chest: [-0.15, -0.4, 0], head: [0, 0.95, 0], neck: [0, 0.3, 0],
      armL: [-1.5, 0, 0.35], foreL: [-0.3, 0, 0], armR: [0.6, 0, -1.55], foreR: [-1.6, 0, 0],
      thighL: [-0.9, 0, 0.25], shinL: [0.6, 0, 0], footL: [0.1, 0, 0], thighR: [0.15, 0, -0.25], shinR: [0.6, 0, 0] }],
    [0.9, { hipY: -0.42, hips: [0.2, 0.05, 0], spine: [0.25, 0.25, 0], chest: [0.1, 0.25, 0], head: [-0.15, 0.2, 0],
      armL: [-0.7, 0, 0.6], foreL: [-1.2, 0, 0], armR: [-1.4, 0, -1.2], foreR: [-1.2, 0, 0],
      thighL: [-0.85, 0, 0.2], shinL: [0.55, 0, 0], thighR: [0.45, 0, -0.2], shinR: [0.8, 0, 0] }],
    [1.0, { hipY: -0.45, hips: [0.3, 0.35, 0], spine: [0.4, 0.3, 0], chest: [0.15, 0.25, 0], head: [-0.3, 0, 0],
      armL: [-0.2, 0, 0.6], foreL: [-1.4, 0, 0], armR: [-2.5, 0, -0.45], foreR: [-0.35, 0, 0],
      thighL: [-0.8, 0, 0.2], shinL: [0.45, 0, 0], thighR: [0.55, 0, -0.15], shinR: [0.9, 0, 0], footR: [0.4, 0, 0] }],
  ]);
}

function followPose(t: number): Pose {
  return keys(t, [
    [0, windupPose(1)],
    [0.14, { hipY: -0.5, hips: [0.45, 0.5, 0], spine: [0.55, 0.25, 0], chest: [0.2, 0.2, 0], head: [-0.45, -0.1, 0],
      armL: [0.3, 0, 0.5], foreL: [-1.4, 0, 0], armR: [-0.9, 0, 0.55], foreR: [-0.4, 0, 0],
      thighL: [-0.75, 0, 0.2], shinL: [0.6, 0, 0], thighR: [0.85, 0, -0.1], shinR: [1.1, 0, 0], footR: [0.5, 0, 0] }],
    [0.45, { ...READY, hipY: -0.3 }],
  ]);
}

// ─────────────────────────────────────────────────────────────── the animator

const _e = new Euler(), _q = new Quaternion();
const _v = new Vector3(), _w = new Vector3(), _hp = new Vector3();

export class Animator {
  private cur: Record<string, Quaternion> = {};
  private hipOff = new Vector3();
  private blinkT = 2 + Math.random() * 3;
  private blinkPhase = -1;
  private seed = Math.random() * 10;
  private time = 0;
  private lastMode: Mode | null = null;
  /** set by the game: an expression override; otherwise picked from the mode */
  expression: Expression | null = null;
  /** world-space bat transform for the batting renderer (handle point + barrel direction) */
  readonly batHandle = new Vector3();
  readonly batDir = new Vector3(0, 1, 0);
  batActive = false;

  constructor(readonly kid: KidModel) {
    for (const n of BONES) this.cur[n] = new Quaternion();
  }

  update(dt: number, inp: AnimInput) {
    this.time += dt;
    const k = this.kid;
    const t = inp.t;
    let pose: Pose;
    let rate = 14;
    let expr: Expression = 'neutral';
    this.batActive = false;
    switch (inp.mode) {
      case 'stand': pose = idlePose(this.time, this.seed); rate = 6; break;
      case 'ready': pose = READY; rate = 10; expr = 'focus'; break;
      case 'crouch': pose = CROUCH; rate = 10; expr = 'focus'; break;
      case 'run': pose = runPose(this.time, inp.speed ?? 18); rate = 18; expr = 'focus'; break;
      case 'trot': pose = runPose(this.time, inp.speed ?? 10, true); rate = 14; expr = 'happy'; break;
      case 'walk': pose = runPose(this.time * 0.75, inp.speed ?? 5, true); rate = 10; break;
      case 'catch': pose = READY; rate = 16; expr = 'focus'; break;
      case 'throw': pose = throwPose(t); rate = 30; expr = 'focus'; break;
      case 'dive': pose = divePose(t, k); rate = 22; expr = t < 1 ? 'yell' : 'oops'; break;
      case 'jump': pose = jumpPose(t); rate = 22; expr = 'yell'; break;
      case 'stumble': pose = stumblePose(t); rate = 20; expr = 'oops'; break;
      case 'slide': pose = slidePose(t, k); rate = 20; expr = 'yell'; break;
      case 'cheer': pose = cheerPose(this.time); rate = 14; expr = 'yell'; break;
      case 'clap': pose = clapPose(this.time); rate = 14; expr = 'happy'; break;
      case 'wave': pose = wavePose(this.time); rate = 12; expr = 'happy'; break;
      case 'grill': pose = grillPose(this.time); rate = 8; expr = 'happy'; break;
      case 'sad': case 'out': pose = SAD; rate = 6; expr = 'sad'; break;
      case 'sit': {
        pose = { ...SIT, hipY: (inp.seat ?? 1.6) + 0.12 - k.p.hipY - 0.05 };
        const sway = Math.sin(this.time * 2.2 + this.seed) * 0.25;
        pose.shinL = [1.45 + sway, 0, 0];
        pose.shinR = [1.6 - sway, 0, 0];
        rate = 6;
        break;
      }
      case 'bat': pose = BAT_STANCE; rate = 10; expr = 'focus'; this.batActive = true;
        // a little bat waggle
        pose = { ...pose, chest: [0, -0.15 + Math.sin(this.time * 3) * 0.04, 0] };
        break;
      case 'swing': pose = swingPose(t, inp.power ? 0.18 : 0.15); rate = 45; expr = t < 0.3 ? 'yell' : 'focus'; this.batActive = true; break;
      case 'bunt': pose = { ...BAT_STANCE, hips: [0.1, 0.9, 0], chest: [0.05, 0.3, 0], head: [0.1, 0.1, 0], thighL: [-0.4, 0, 0.3], shinL: [0.6, 0, 0], thighR: [-0.4, 0, -0.3], shinR: [0.6, 0, 0], hipY: -0.35 }; rate = 16; expr = 'focus'; this.batActive = true; break;
      case 'windup': pose = windupPose(t / (inp.windup ?? 0.8)); rate = 40; expr = 'focus'; break;
      case 'follow': pose = followPose(t); rate = 30; expr = 'focus'; break;
      default: pose = STAND;
    }
    if (inp.lefty) pose = mirror(pose);
    if (inp.mode !== this.lastMode) this.lastMode = inp.mode;

    // ease bones toward the target pose
    const a = 1 - Math.exp(-dt * rate);
    for (const n of BONES) {
      const r = pose[n];
      const target = r ? _q.setFromEuler(_e.set(r[0], r[1], r[2], 'YXZ')) : _q.identity();
      this.cur[n].slerp(target, a);
      k.bones[n].quaternion.copy(this.cur[n]);
    }
    // hips offset (crouch / bob / jump / lie down)
    const s = k.p.s;
    _hp.set((pose.hipX ?? 0) * s, (pose.hipY ?? 0) * (inp.mode === 'sit' || inp.mode === 'dive' || inp.mode === 'slide' ? 1 : s) + (inp.lift ?? 0), (pose.hipZ ?? 0) * s);
    this.hipOff.lerp(_hp, Math.min(1, a * 1.2));
    k.bones.hips.position.copy(k.p.joints.hips).add(this.hipOff);

    // breathing
    const br = Math.sin(this.time * 2.1 + this.seed) * 0.025;
    k.bones.chest.quaternion.multiply(_q.setFromEuler(_e.set(br, 0, 0)));
    k.group.updateMatrixWorld(true);

    // bat grip / glove reach IK
    if (this.batActive) this.solveBat(inp);
    if (inp.reach && (inp.mode === 'catch' || inp.mode === 'ready' || inp.mode === 'run' || inp.mode === 'jump' || inp.mode === 'dive')) {
      const glove = inp.lefty ? 'R' : 'L';
      this.reachFor(glove, inp.reach, inp.mode === 'catch' ? 1 : 0.85);
    }
    if (inp.mode === 'windup' && t / (inp.windup ?? 0.8) < 0.6) {
      // hands together at the chest (ball in the glove)
      const cw = k.bones.chest.localToWorld(_w.set(0, 0.05 * s, 0.55 * s));
      this.reachFor('L', cw, 0.8);
      this.reachFor('R', cw.clone().add(_v.set(0, 0.03, 0)), 0.8);
    }

    // head + eyes track the look target
    if (inp.lookAt) this.look(inp.lookAt, inp.mode);
    this.blink(dt, this.expression ?? expr);
    const want = this.expression ?? expr;
    if (want !== k.expression) k.setExpression(want);
  }

  private reachFor(side: 'L' | 'R', target: Vector3, blend: number) {
    const k = this.kid;
    const arm = side === 'L' ? k.bones.armL : k.bones.armR;
    const fore = side === 'L' ? k.bones.foreL : k.bones.foreR;
    const hand = side === 'L' ? k.bones.handL : k.bones.handR;
    // elbows bend down and out
    const pole = k.bones.chest.localToWorld(_v.set(side === 'L' ? 1.4 : -1.4, -1.2, -0.4));
    twoBoneIK(arm, fore, hand, target, pole.clone(), blend);
  }

  /** Place the bat from the swing/stance and wrap both hands around the handle. */
  private solveBat(inp: AnimInput) {
    const k = this.kid;
    const s = k.p.s;
    const lefty = !!inp.lefty;
    const m = lefty ? -1 : 1;
    // in the kid's local frame (facing the plate, pitcher toward +x for a righty)
    let handle: Vector3, dir: Vector3;
    const sh = k.p.shoulderY;
    if (inp.mode === 'bunt') {
      handle = new Vector3(-0.35 * m, sh - 0.35 * s, 0.75 * s);
      dir = new Vector3(0.2 * m, 0.12, 1).normalize();
    } else if (inp.mode === 'swing') {
      const tc = inp.power ? 0.18 : 0.15;
      const u = inp.t;
      const aimY = clamp(((inp.aimZ ?? 2.2) - 2.2) * 0.35, -0.5, 0.6);
      const path: [number, Vector3, Vector3][] = [
        [0, new Vector3(-0.45, sh + 0.05 * s, 0.05), new Vector3(-0.6, 0.75, -0.35)],
        [tc * 0.55, new Vector3(-0.25, sh - 0.35 * s, 0.45 * s), new Vector3(-0.85, 0.05 + aimY * 0.5, 0.45)],
        [tc, new Vector3(0.12, sh - 0.6 * s + aimY * 0.3, 0.8 * s), new Vector3(0.2, -0.1 + aimY, 1)],
        [tc + 0.12, new Vector3(0.45, sh - 0.35 * s, 0.4 * s), new Vector3(0.95, 0.25, 0.1)],
        [tc + 0.4, new Vector3(0.45, sh + 0.05 * s, -0.15), new Vector3(0.1, 0.65, -0.85)],
      ];
      let i = 0;
      while (i < path.length - 2 && u > path[i + 1][0]) i++;
      const [t0, h0, d0] = path[i], [t1, h1, d1] = path[i + 1];
      const f = smooth((u - t0) / (t1 - t0));
      handle = h0.clone().lerp(h1, f);
      dir = d0.clone().lerp(d1, f).normalize();
    } else {
      const wag = Math.sin(this.time * 3) * 0.06;
      handle = new Vector3(-0.42, sh + 0.08 * s, 0.05);
      dir = new Vector3(-0.55 + wag, 0.8, -0.3).normalize();
    }
    handle.x *= m; dir.x *= m;
    // to world
    k.group.localToWorld(this.batHandle.copy(handle));
    this.batDir.copy(dir).transformDirection(k.group.matrixWorld);
    // bottom hand (glove-side) at the knob end, top hand just above it
    const bottom = lefty ? 'R' : 'L', top = lefty ? 'L' : 'R';
    const hb = this.batHandle.clone().addScaledVector(this.batDir, 0.05);
    const ht = this.batHandle.clone().addScaledVector(this.batDir, 0.32 * s);
    this.reachFor(bottom, hb, 1);
    this.reachFor(top, ht, 1);
    // hands point along the bat
    pointBone(k.bones.handL, _v.set(0, -1, 0), this.batDir.clone().cross(_w.set(0, 1, 0)).lengthSq() > 1e-4 ? this.batDir : this.batDir, 0.7);
    pointBone(k.bones.handR, _v.set(0, -1, 0), this.batDir, 0.7);
  }

  private look(target: Vector3, mode: Mode) {
    const k = this.kid;
    const head = k.bones.head;
    // turn the head (limited) toward the target, on top of the animated pose
    head.updateMatrixWorld(true);
    const local = head.parent!.worldToLocal(_v.copy(target));
    const hp = head.position;
    const dx = local.x - hp.x, dy = local.y - hp.y - k.p.headR, dz = local.z - hp.z;
    const yaw = Math.atan2(dx, dz), pitch = -Math.atan2(dy, Math.hypot(dx, dz));
    const busy = mode === 'swing' || mode === 'windup' || mode === 'throw' || mode === 'dive' || mode === 'slide' || mode === 'follow';
    const amt = busy ? 0.35 : 0.85;
    const curE = _e.setFromQuaternion(head.quaternion, 'YXZ');
    const ty = clamp(yaw, -1.3, 1.3), tp = clamp(pitch, -0.7, 0.6);
    curE.y = lerp(curE.y, ty, amt);
    curE.x = lerp(curE.x, tp, amt * 0.7);
    head.quaternion.setFromEuler(curE);
    head.updateMatrixWorld(true);
    // eyes do the rest
    for (const eb of [k.bones.eyeL, k.bones.eyeR]) {
      const lp = eb.parent!.worldToLocal(_w.copy(target)).sub(eb.position);
      const ey = clamp(Math.atan2(lp.x, lp.z), -0.55, 0.55), ex = clamp(-Math.atan2(lp.y, Math.hypot(lp.x, lp.z)), -0.4, 0.4);
      eb.quaternion.setFromEuler(_e.set(ex, ey, 0, 'YXZ'));
    }
  }

  private blink(dt: number, expr: Expression) {
    const k = this.kid;
    this.blinkT -= dt;
    if (this.blinkT <= 0 && this.blinkPhase < 0) { this.blinkPhase = 0; }
    let close = 0;
    if (this.blinkPhase >= 0) {
      this.blinkPhase += dt / 0.16;
      close = Math.sin(Math.min(1, this.blinkPhase) * Math.PI);
      if (this.blinkPhase >= 1) { this.blinkPhase = -1; this.blinkT = 1.5 + Math.random() * 3.5; }
    }
    const open = { neutral: -0.62, happy: -0.42, focus: -0.38, surprised: -0.9, sad: -0.3, yell: -0.5, smug: -0.32, oops: -0.7 }[expr];
    const ang = lerp(open, 0.62, close);
    k.bones.lidL.quaternion.setFromEuler(_e.set(ang, 0, 0));
    k.bones.lidR.quaternion.setFromEuler(_e.set(ang, 0, 0));
  }
}
