// World units are feet. Origin is the point of home plate; +y runs toward
// second base / center field, +x toward the first-base side, +z is up.

export interface Vec2 { x: number; y: number }
export interface Vec3 { x: number; y: number; z: number }

export const v2 = (x: number, y: number): Vec2 => ({ x, y });
export const v3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

export const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const invLerp = (a: number, b: number, v: number) => (b === a ? 0 : (v - a) / (b - a));
export const smoothstep = (t: number) => {
  const c = clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
};
export const easeOutCubic = (t: number) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
export const easeInOut = (t: number) => {
  const c = clamp(t, 0, 1);
  return c < 0.5 ? 4 * c * c * c : 1 - Math.pow(-2 * c + 2, 3) / 2;
};

export const DEG = Math.PI / 180;
export const MPH = 5280 / 3600; // ft/s per mph
export const GRAVITY = 32.17; // ft/s²

export const dist2 = (a: Vec2, b: Vec2) => Math.hypot(a.x - b.x, a.y - b.y);
export const dist3 = (a: Vec3, b: Vec3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
export const len2 = (x: number, y: number) => Math.hypot(x, y);

export const copy3 = (a: Vec3): Vec3 => ({ x: a.x, y: a.y, z: a.z });
export const copy2 = (a: Vec2): Vec2 => ({ x: a.x, y: a.y });

/** Move `from` toward `to` by at most `step`; returns the new point. */
export function approach2(from: Vec2, to: Vec2, step: number): Vec2 {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const d = Math.hypot(dx, dy);
  if (d <= step || d === 0) return { x: to.x, y: to.y };
  return { x: from.x + (dx / d) * step, y: from.y + (dy / d) * step };
}

export function approachNum(v: number, target: number, step: number) {
  if (v < target) return Math.min(target, v + step);
  return Math.max(target, v - step);
}

/** Spray angle in degrees: 0 = straight to center, negative = left field. */
export const sprayAngle = (x: number, y: number) => Math.atan2(x, y) / DEG;

/** Segment intersection of p→q against a→b; returns t along p→q or null. */
export function segIntersect(
  px: number, py: number, qx: number, qy: number,
  ax: number, ay: number, bx: number, by: number,
): { t: number; u: number } | null {
  const rx = qx - px, ry = qy - py;
  const sx = bx - ax, sy = by - ay;
  const den = rx * sy - ry * sx;
  if (Math.abs(den) < 1e-9) return null;
  const t = ((ax - px) * sy - (ay - py) * sx) / den;
  const u = ((ax - px) * ry - (ay - py) * rx) / den;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return { t, u };
}

export function pointInPoly(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export const formatAvg = (h: number, ab: number) => {
  if (ab === 0) return '.000';
  const v = h / ab;
  return v >= 1 ? v.toFixed(3) : v.toFixed(3).slice(1);
};
