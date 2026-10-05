import { lerp, type Vec3 } from '../engine/math';

export interface CamPose {
  pos: Vec3;
  target: Vec3;
  fov: number; // vertical, degrees
}

export interface Projected {
  x: number;
  y: number;
  /** camera-space depth (feet) */
  z: number;
  /** pixels per foot at this depth */
  s: number;
}

const NEAR = 0.6;

/** A simple perspective camera; world z is up. */
export class Camera {
  pose: CamPose = { pos: { x: 0, y: -16, z: 6 }, target: { x: 0, y: 40, z: 2.5 }, fov: 42 };
  w = 800;
  h = 450;
  private r = { x: 1, y: 0, z: 0 };
  private u = { x: 0, y: 0, z: 1 };
  private f = { x: 0, y: 1, z: 0 };
  private focal = 1;

  setViewport(w: number, h: number) {
    this.w = w;
    this.h = h;
  }

  update() {
    const { pos, target } = this.pose;
    let fx = target.x - pos.x, fy = target.y - pos.y, fz = target.z - pos.z;
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl; fy /= fl; fz /= fl;
    // right = f × up(0,0,1)
    let rx = fy, ry = -fx, rz = 0;
    const rl = Math.hypot(rx, ry) || 1;
    rx /= rl; ry /= rl;
    // up' = r × f
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    this.f = { x: fx, y: fy, z: fz };
    this.r = { x: rx, y: ry, z: rz };
    this.u = { x: ux, y: uy, z: uz };
    // fit the vertical FOV, but widen it on tall (portrait) screens
    const aspect = this.w / this.h;
    let fov = this.pose.fov;
    if (aspect < 1.3) fov = Math.min(100, fov * (1 + (1.3 - aspect) * 0.9));
    this.focal = this.h / 2 / Math.tan((fov * Math.PI) / 360);
  }

  /** Camera-space coordinates. */
  toCam(x: number, y: number, z: number) {
    const p = this.pose.pos;
    const dx = x - p.x, dy = y - p.y, dz = z - p.z;
    return {
      cx: dx * this.r.x + dy * this.r.y + dz * this.r.z,
      cy: dx * this.u.x + dy * this.u.y + dz * this.u.z,
      cz: dx * this.f.x + dy * this.f.y + dz * this.f.z,
    };
  }

  project(x: number, y: number, z: number): Projected | null {
    const c = this.toCam(x, y, z);
    if (c.cz < NEAR) return null;
    const s = this.focal / c.cz;
    return { x: this.w / 2 + c.cx * s, y: this.h / 2 - c.cy * s, z: c.cz, s };
  }

  /** Like project but clamps behind-camera points (for sorting / rough placement). */
  depth(x: number, y: number, z: number) {
    return this.toCam(x, y, z).cz;
  }

  /** Project a ground/world polygon, clipping it against the near plane. */
  projectPoly(pts: [number, number, number][]): [number, number][] {
    const cam = pts.map(([x, y, z]) => this.toCam(x, y, z));
    const out: { cx: number; cy: number; cz: number }[] = [];
    for (let i = 0; i < cam.length; i++) {
      const a = cam[i];
      const b = cam[(i + 1) % cam.length];
      const ain = a.cz >= NEAR, bin = b.cz >= NEAR;
      if (ain) out.push(a);
      if (ain !== bin) {
        const t = (NEAR - a.cz) / (b.cz - a.cz);
        out.push({ cx: a.cx + (b.cx - a.cx) * t, cy: a.cy + (b.cy - a.cy) * t, cz: NEAR });
      }
    }
    return out.map((c) => {
      const s = this.focal / c.cz;
      return [this.w / 2 + c.cx * s, this.h / 2 - c.cy * s];
    });
  }

  /** Screen y of the horizon. */
  horizonY() {
    const p = this.pose.pos;
    const far = this.project(p.x + this.f.x * 5000, p.y + this.f.y * 5000, 0);
    return far ? far.y : 0;
  }

  /** Cast a screen point onto the vertical plane y = planeY (e.g. the plate). */
  rayToPlaneY(sx: number, sy: number, planeY: number): { x: number; z: number } | null {
    const px = (sx - this.w / 2) / this.focal;
    const py = -(sy - this.h / 2) / this.focal;
    const dx = this.f.x + this.r.x * px + this.u.x * py;
    const dy = this.f.y + this.r.y * px + this.u.y * py;
    const dz = this.f.z + this.r.z * px + this.u.z * py;
    const o = this.pose.pos;
    if (Math.abs(dy) < 1e-6) return null;
    const t = (planeY - o.y) / dy;
    if (t <= 0) return null;
    return { x: o.x + dx * t, z: o.z + dz * t };
  }

  /** Cast a screen point onto the ground plane (z = h). */
  unproject(sx: number, sy: number, h = 0): { x: number; y: number } | null {
    const px = (sx - this.w / 2) / this.focal;
    const py = -(sy - this.h / 2) / this.focal;
    const dir = {
      x: this.f.x + this.r.x * px + this.u.x * py,
      y: this.f.y + this.r.y * px + this.u.y * py,
      z: this.f.z + this.r.z * px + this.u.z * py,
    };
    const o = this.pose.pos;
    if (Math.abs(dir.z) < 1e-6) return null;
    const t = (h - o.z) / dir.z;
    if (t <= 0) return null;
    return { x: o.x + dir.x * t, y: o.y + dir.y * t };
  }
}

export function lerpPose(a: CamPose, b: CamPose, t: number): CamPose {
  return {
    pos: { x: lerp(a.pos.x, b.pos.x, t), y: lerp(a.pos.y, b.pos.y, t), z: lerp(a.pos.z, b.pos.z, t) },
    target: { x: lerp(a.target.x, b.target.x, t), y: lerp(a.target.y, b.target.y, t), z: lerp(a.target.z, b.target.z, t) },
    fov: lerp(a.fov, b.fov, t),
  };
}
