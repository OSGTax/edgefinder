import { Bone, Skeleton, Vector3 } from 'three';
import type { KidLook } from '../data/types';
import { kidHeightFt } from '../sim/field';

// The kids' skeleton and body proportions. Characters face +z, up is +y,
// so the kid's LEFT is +x and their RIGHT is −x. Feet stand on y = 0.

export const BONES = [
  'root', 'hips', 'spine', 'chest', 'neck', 'head', 'eyeL', 'eyeR', 'lidL', 'lidR',
  'shoulderL', 'armL', 'foreL', 'handL',
  'shoulderR', 'armR', 'foreR', 'handR',
  'thighL', 'shinL', 'footL',
  'thighR', 'shinR', 'footR',
] as const;
export type BoneName = (typeof BONES)[number];
export const B = Object.fromEntries(BONES.map((n, i) => [n, i])) as Record<BoneName, number>;

const PARENT: Record<BoneName, BoneName | null> = {
  root: null, hips: 'root', spine: 'hips', chest: 'spine', neck: 'chest', head: 'neck',
  eyeL: 'head', eyeR: 'head', lidL: 'head', lidR: 'head',
  shoulderL: 'chest', armL: 'shoulderL', foreL: 'armL', handL: 'foreL',
  shoulderR: 'chest', armR: 'shoulderR', foreR: 'armR', handR: 'foreR',
  thighL: 'hips', shinL: 'thighL', footL: 'shinL',
  thighR: 'hips', shinR: 'thighR', footR: 'shinR',
};

/** Head ellipsoid scale (x, y, z) per head shape. */
export const HEAD_SHAPE: Record<KidLook['head'], [number, number, number]> = {
  round: [1, 1, 1], oval: [0.93, 1.08, 0.97], square: [1.05, 0.97, 1.0], wide: [1.12, 0.95, 1.03],
};

/** Body measurements in feet, all derived from the kid's look. */
export interface Proportions {
  H: number;          // standing height to the top of the head
  s: number;          // body scale vs. a 4.6 ft kid
  hs: number;         // head scale (heads don't shrink as much — they're kids)
  wf: number;         // width factor from build
  belly: number;      // extra tummy (0..1)
  headR: number;
  joints: Record<BoneName, Vector3>;
  /** helpful extra points */
  eyeY: number; eyeX: number; eyeZ: number; eyeR: number;
  hipY: number; kneeY: number; ankleY: number; waistY: number; chestY: number; shoulderY: number; neckY: number;
  shoulderX: number; hipX: number;
  upperArm: number; foreArm: number;
  armR: number; legR: number;
  footLen: number;
}

export function proportions(look: KidLook): Proportions {
  const H = kidHeightFt(look.height);
  // big cartoon heads: solve the body scale so the top of the head lands at H
  const s0 = (H - 1.42) / 3.45;
  const hs = 1 + (s0 - 1) * 0.35;
  const headR = 0.74 * hs;
  const s = (H - 1.92 * headR) / 3.45;
  const wf = 0.86 + look.build * 0.42;
  const belly = Math.max(0, look.build - 0.6) * 1.8;
  const ankleY = 0.24 * s, kneeY = 1.12 * s, hipY = 2.0 * s;
  const waistY = 2.36 * s, chestY = 2.86 * s, shoulderY = 3.2 * s, neckY = 3.33 * s;
  const headBase = neckY + 0.12 * s;
  const shoulderX = 0.5 * wf + 0.06, hipX = 0.27 * wf;
  const upperArm = 0.62 * s, foreArm = 0.56 * s;
  const eyeR = 0.165 * hs;
  const headC = headBase + headR * 0.92;
  // eyes sit on the actual head surface (whatever its shape), about half proud of it
  const [sx, sy, sz] = HEAD_SHAPE[look.head] ?? HEAD_SHAPE.round;
  const ex = headR * 0.35 * Math.max(1, sx * 0.95), ey = headR * 0.05;
  const surf = 0.04 + headR * sz * Math.sqrt(Math.max(0, 1 - (ex / (headR * sx)) ** 2 - (ey / (headR * sy)) ** 2));
  const eyeZ = surf - eyeR * 0.42;
  const j: Record<BoneName, Vector3> = {
    root: new Vector3(0, 0, 0),
    hips: new Vector3(0, hipY + 0.05 * s, 0),
    spine: new Vector3(0, waistY, 0),
    chest: new Vector3(0, chestY, 0),
    neck: new Vector3(0, neckY, 0),
    head: new Vector3(0, headBase, 0.02),
    eyeL: new Vector3(ex, headC + ey, eyeZ),
    eyeR: new Vector3(-ex, headC + ey, eyeZ),
    lidL: new Vector3(ex, headC + ey, eyeZ),
    lidR: new Vector3(-ex, headC + ey, eyeZ),
    shoulderL: new Vector3(0.16 * wf, shoulderY - 0.04, 0),
    armL: new Vector3(shoulderX, shoulderY, 0),
    foreL: new Vector3(shoulderX + 0.09, shoulderY - upperArm, 0.02),
    handL: new Vector3(shoulderX + 0.15, shoulderY - upperArm - foreArm, 0.05),
    shoulderR: new Vector3(-0.16 * wf, shoulderY - 0.04, 0),
    armR: new Vector3(-shoulderX, shoulderY, 0),
    foreR: new Vector3(-shoulderX - 0.09, shoulderY - upperArm, 0.02),
    handR: new Vector3(-shoulderX - 0.15, shoulderY - upperArm - foreArm, 0.05),
    thighL: new Vector3(hipX, hipY, 0),
    shinL: new Vector3(hipX, kneeY, 0.03),
    footL: new Vector3(hipX, ankleY, 0),
    thighR: new Vector3(-hipX, hipY, 0),
    shinR: new Vector3(-hipX, kneeY, 0.03),
    footR: new Vector3(-hipX, ankleY, 0),
  };
  return {
    H, s, hs, wf, belly, headR, joints: j,
    eyeY: j.eyeL.y, eyeX: j.eyeL.x, eyeZ: j.eyeL.z, eyeR,
    hipY, kneeY, ankleY, waistY, chestY, shoulderY, neckY, shoulderX, hipX, upperArm, foreArm,
    armR: (0.17 + look.build * 0.06) * s, legR: (0.22 + look.build * 0.07) * s,
    footLen: 0.78 * s,
  };
}

/** Build the bone hierarchy at bind pose (no rotations, positions relative to parents). */
export function makeSkeleton(p: Proportions): { skeleton: Skeleton; bones: Bone[] } {
  const bones = BONES.map((name) => { const b = new Bone(); b.name = name; return b; });
  BONES.forEach((name, i) => {
    const par = PARENT[name];
    const world = p.joints[name];
    if (par) {
      const pw = p.joints[par];
      bones[i].position.copy(world).sub(pw);
      bones[B[par]].add(bones[i]);
    } else {
      bones[i].position.copy(world);
    }
  });
  bones[0].updateMatrixWorld(true);
  return { skeleton: new Skeleton(bones), bones };
}
