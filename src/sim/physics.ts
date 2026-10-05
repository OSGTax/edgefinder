import { GRAVITY, type Vec3 } from '../engine/math';
import type { Rng } from '../engine/rng';
import type { FenceSeg, Obstacle, Surface } from '../data/types';
import { SURFACE, surfaceAt, type Field } from './field';

export const DRAG_K = 0.0021; // quadratic drag coefficient for a kid's baseball, 1/ft

export interface Ball {
  p: Vec3;
  v: Vec3;
  drag: boolean;
  grounded: boolean;
  /** has touched ground, fence or an obstacle since the last contact/throw */
  touched: boolean;
  bounces: number;
  resting: boolean;
  dead: boolean;
  canopy: number;
}

export type BallEvent =
  | { type: 'bounce'; x: number; y: number; surface: Surface; speed: number }
  | { type: 'fence'; seg: FenceSeg; x: number; y: number; z: number; cleared: boolean }
  | { type: 'obstacle'; ob: Obstacle; x: number; y: number; z: number }
  | { type: 'canopy'; ob: Obstacle; x: number; y: number; z: number }
  | { type: 'water'; x: number; y: number }
  | { type: 'rest'; x: number; y: number };

export function newBall(p: Vec3, v: Vec3, drag = true): Ball {
  return { p: { ...p }, v: { ...v }, drag, grounded: false, touched: false, bounces: 0, resting: false, dead: false, canopy: -1 };
}

export function cloneBall(b: Ball): Ball {
  return { ...b, p: { ...b.p }, v: { ...b.v } };
}

const FENCE_BOUNCE: Record<string, number> = {
  picket: 0.35, wood: 0.4, chain: 0.22, hedge: 0.12, garage: 0.5, house: 0.45, sunflower: 0.08, reeds: 0.1, barn: 0.45,
};

/**
 * Advance the ball by dt. Pure physics: the caller decides what events mean
 * (home run, foul, ground-rule double...). `rng` adds a little chaos to
 * tree-canopy deflections; pass null for deterministic prediction.
 */
export function stepBall(b: Ball, f: Field, dt: number, rng: Rng | null, events?: BallEvent[]): void {
  if (b.resting || b.dead) return;
  const x0 = b.p.x, y0 = b.p.y, z0 = b.p.z;

  if (b.grounded) {
    const sp = Math.hypot(b.v.x, b.v.y);
    const surf = surfaceAt(f, b.p.x, b.p.y);
    if (surf === 'water') {
      b.dead = true;
      events?.push({ type: 'water', x: b.p.x, y: b.p.y });
      return;
    }
    const dec = SURFACE[surf].roll * dt;
    if (sp <= dec || sp < 0.4) {
      b.v.x = 0; b.v.y = 0;
      b.resting = true;
      events?.push({ type: 'rest', x: b.p.x, y: b.p.y });
      return;
    }
    const k = (sp - dec) / sp;
    b.v.x *= k; b.v.y *= k;
    b.p.x += b.v.x * dt;
    b.p.y += b.v.y * dt;
  } else {
    let ax = 0, ay = 0, az = -GRAVITY;
    if (b.drag) {
      const sp = Math.hypot(b.v.x, b.v.y, b.v.z);
      ax -= DRAG_K * sp * b.v.x;
      ay -= DRAG_K * sp * b.v.y;
      az -= DRAG_K * sp * b.v.z;
    }
    b.v.x += ax * dt; b.v.y += ay * dt; b.v.z += az * dt;
    b.p.x += b.v.x * dt; b.p.y += b.v.y * dt; b.p.z += b.v.z * dt;

    if (b.p.z <= 0 && b.v.z < 0) {
      b.p.z = 0;
      const surf = surfaceAt(f, b.p.x, b.p.y);
      if (surf === 'water') {
        b.dead = true;
        events?.push({ type: 'water', x: b.p.x, y: b.p.y });
        return;
      }
      const sp = SURFACE[surf];
      const impact = -b.v.z;
      b.touched = true;
      if (impact > 3) {
        b.v.z = impact * sp.bounce;
        b.v.x *= 1 - sp.scrub;
        b.v.y *= 1 - sp.scrub;
        b.bounces++;
        events?.push({ type: 'bounce', x: b.p.x, y: b.p.y, surface: surf, speed: impact });
        if (b.v.z < 2) { b.v.z = 0; b.grounded = true; }
      } else {
        b.v.z = 0;
        b.grounded = true;
      }
    }
  }

  // fences
  for (const seg of f.fence) {
    const ax = seg.a[0], ay = seg.a[1], bx = seg.b[0], by = seg.b[1];
    const rx = b.p.x - x0, ry = b.p.y - y0;
    const sx = bx - ax, sy = by - ay;
    const den = rx * sy - ry * sx;
    if (Math.abs(den) < 1e-9) continue;
    const t = ((ax - x0) * sy - (ay - y0) * sx) / den;
    const u = ((ax - x0) * ry - (ay - y0) * rx) / den;
    if (t < 0 || t > 1 || u < 0 || u > 1) continue;
    const zc = z0 + (b.p.z - z0) * t;
    const cx = x0 + rx * t, cy = y0 + ry * t;
    if (zc > seg.height + 0.15) {
      b.dead = true;
      events?.push({ type: 'fence', seg, x: cx, y: cy, z: zc, cleared: true });
      return;
    }
    // bounce off the fence: reflect the velocity about the segment normal
    const L = Math.hypot(sx, sy);
    let nx = -sy / L, ny = sx / L;
    if (nx * rx + ny * ry > 0) { nx = -nx; ny = -ny; }
    const e = FENCE_BOUNCE[seg.kind] ?? 0.35;
    const vn = b.v.x * nx + b.v.y * ny;
    b.v.x -= (1 + e) * vn * nx;
    b.v.y -= (1 + e) * vn * ny;
    b.v.z *= 0.7;
    b.p.x = cx + nx * 0.6;
    b.p.y = cy + ny * 0.6;
    b.p.z = zc;
    b.touched = true;
    events?.push({ type: 'fence', seg, x: cx, y: cy, z: zc, cleared: false });
    if (seg.splash) { b.dead = true; return; }
    break;
  }

  // obstacles
  for (let i = 0; i < f.obstacles.length; i++) {
    const ob = f.obstacles[i];
    if (ob.kind === 'canopy') {
      const dx = b.p.x - ob.x, dy = b.p.y - ob.y, dz = b.p.z - ob.h;
      const rz = ob.rz ?? ob.r * 0.7;
      const inside = (dx * dx + dy * dy) / (ob.r * ob.r) + (dz * dz) / (rz * rz) < 1;
      if (inside && b.canopy !== i) {
        b.canopy = i;
        b.touched = true;
        const damp = 0.3;
        const jitter = rng ? rng.range(-1, 1) : 0;
        const jitter2 = rng ? rng.range(-1, 1) : 0;
        const sp = Math.hypot(b.v.x, b.v.y, b.v.z) * damp;
        b.v.x = b.v.x * damp + jitter * sp * 0.6;
        b.v.y = b.v.y * damp + jitter2 * sp * 0.6;
        b.v.z = Math.min(b.v.z * damp, 4);
        b.grounded = false;
        events?.push({ type: 'canopy', ob, x: b.p.x, y: b.p.y, z: b.p.z });
      } else if (!inside && b.canopy === i) {
        b.canopy = -1;
      } else if (inside) {
        // leaves keep slowing it down
        b.v.x *= 1 - 2.5 * dt; b.v.y *= 1 - 2.5 * dt;
      }
    } else if (ob.kind === 'cylinder') {
      if (b.p.z > ob.h) continue;
      const dx = b.p.x - ob.x, dy = b.p.y - ob.y;
      const d = Math.hypot(dx, dy);
      if (d < ob.r && d > 1e-6) {
        const nx = dx / d, ny = dy / d;
        const vn = b.v.x * nx + b.v.y * ny;
        if (vn < 0) {
          const e = ob.bounce ?? 0.45;
          b.v.x -= (1 + e) * vn * nx;
          b.v.y -= (1 + e) * vn * ny;
        }
        b.p.x = ob.x + nx * (ob.r + 0.05);
        b.p.y = ob.y + ny * (ob.r + 0.05);
        b.touched = true;
        events?.push({ type: 'obstacle', ob, x: b.p.x, y: b.p.y, z: b.p.z });
      }
    } else {
      const rot = ob.rot ?? 0;
      const c = Math.cos(rot), s = Math.sin(rot);
      const dx = b.p.x - ob.x, dy = b.p.y - ob.y;
      const lx = dx * c + dy * s, ly = -dx * s + dy * c;
      const hw = ob.r, hd = ob.d ?? ob.r;
      if (Math.abs(lx) < hw && Math.abs(ly) < hd && b.p.z < ob.h) {
        const px = hw - Math.abs(lx), py = hd - Math.abs(ly), pz = ob.h - b.p.z;
        const e = ob.bounce ?? 0.35;
        b.touched = true;
        if (pz < px && pz < py && b.v.z <= 0) {
          b.p.z = ob.h;
          b.v.z = -b.v.z * e;
          if (b.v.z < 2) b.v.z = 2; // roll off the roof eventually
          b.grounded = false;
        } else {
          // local velocity
          let lvx = b.v.x * c + b.v.y * s, lvy = -b.v.x * s + b.v.y * c;
          let nlx = lx, nly = ly;
          if (px < py) { lvx = -lvx * e; nlx = Math.sign(lx) * (hw + 0.05); }
          else { lvy = -lvy * e; nly = Math.sign(ly) * (hd + 0.05); }
          b.v.x = lvx * c - lvy * s; b.v.y = lvx * s + lvy * c;
          b.p.x = ob.x + nlx * c - nly * s; b.p.y = ob.y + nlx * s + nly * c;
        }
        events?.push({ type: 'obstacle', ob, x: b.p.x, y: b.p.y, z: b.p.z });
      }
    }
  }

  if (Math.abs(b.p.x) > 600 || b.p.y > 600 || b.p.y < -200) b.dead = true;
}

export interface PathSample { t: number; x: number; y: number; z: number; touched: boolean; resting: boolean }

/** Simulate ahead without side effects (no canopy randomness). */
export function predict(b: Ball, f: Field, maxT = 7, dt = 1 / 60): PathSample[] {
  const sim = cloneBall(b);
  const out: PathSample[] = [{ t: 0, x: sim.p.x, y: sim.p.y, z: sim.p.z, touched: sim.touched, resting: sim.resting }];
  for (let t = dt; t <= maxT; t += dt) {
    stepBall(sim, f, dt, null);
    out.push({ t, x: sim.p.x, y: sim.p.y, z: sim.p.z, touched: sim.touched, resting: sim.resting });
    if (sim.dead || sim.resting) break;
  }
  return out;
}

/** Where a fly ball first comes down (z returns to `atZ` on the way down). */
export function landing(path: PathSample[], atZ = 0): PathSample | null {
  for (let i = 1; i < path.length; i++) {
    if (path[i].touched) return path[i];
    if (path[i].z <= atZ && path[i - 1].z > atZ) return path[i];
  }
  return null;
}
