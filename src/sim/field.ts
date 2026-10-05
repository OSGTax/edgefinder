import { DEG, pointInPoly, segIntersect, type Vec2 } from '../engine/math';
import type { FenceSeg, Obstacle, Position, Surface, Yard } from '../data/types';

export interface SurfaceProps {
  /** vertical restitution on a bounce */
  bounce: number;
  /** fraction of horizontal speed lost per bounce */
  scrub: number;
  /** rolling deceleration, ft/s² */
  roll: number;
}

export const SURFACE: Record<Surface, SurfaceProps> = {
  grass: { bounce: 0.42, scrub: 0.16, roll: 6 },
  dirt: { bounce: 0.5, scrub: 0.1, roll: 4.5 },
  patio: { bounce: 0.62, scrub: 0.08, roll: 4.5 },
  sand: { bounce: 0.12, scrub: 0.6, roll: 30 },
  mud: { bounce: 0.05, scrub: 0.85, roll: 60 },
  water: { bounce: 0, scrub: 1, roll: 999 },
};

export interface Field {
  yard: Yard;
  base: number;      // base path length
  s: number;         // base coordinate offset (base / √2)
  bases: Vec2[];     // [home, first, second, third]
  mound: Vec2;
  fence: FenceSeg[];
  obstacles: Obstacle[];
  worn: [number, number][][]; // dirt worn into the grass (for physics + art)
  wornSurface: Surface;
  defaultSpots: Record<Position, Vec2>;
}

export function buildField(yard: Yard): Field {
  const base = yard.basePath;
  const s = base / Math.SQRT2;
  const bases: Vec2[] = [
    { x: 0, y: 0 },
    { x: s, y: s },
    { x: 0, y: 2 * s },
    { x: -s, y: s },
  ];
  const mound = { x: 0, y: yard.moundDist };
  const obstacles = obstaclesFromProps(yard);
  const worn = wornAreas(yard, bases, mound);
  const f: Field = {
    yard, base, s, bases, mound, fence: yard.fence, obstacles, worn,
    wornSurface: yard.infield === 'sand' ? 'sand' : 'dirt',
    defaultSpots: {} as Record<Position, Vec2>,
  };
  const of = (deg: number, frac: number) => {
    const d = fenceDistance(f, deg) * frac;
    return { x: Math.sin(deg * DEG) * d, y: Math.cos(deg * DEG) * d };
  };
  f.defaultSpots = {
    P: { x: 0, y: yard.moundDist },
    C: { x: 0, y: -4 },
    '1B': { x: s - 3, y: s + 9 },
    '2B': { x: 15, y: 2 * s - 9 },
    SS: { x: -16, y: 2 * s - 10 },
    '3B': { x: -s + 4, y: s + 7 },
    LF: of(-27, 0.72),
    CF: of(0, 0.72),
    RF: of(27, 0.72),
  };
  return f;
}

function circle(cx: number, cy: number, r: number, n = 12): [number, number][] {
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2;
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r] as [number, number];
  });
}

function strip(a: Vec2, b: Vec2, w: number): [number, number][] {
  const dx = b.x - a.x, dy = b.y - a.y;
  const L = Math.hypot(dx, dy);
  const nx = (-dy / L) * w, ny = (dx / L) * w;
  return [[a.x + nx, a.y + ny], [b.x + nx, b.y + ny], [b.x - nx, b.y - ny], [a.x - nx, a.y - ny]];
}

function wornAreas(yard: Yard, bases: Vec2[], mound: Vec2): [number, number][][] {
  const areas: [number, number][][] = [];
  const s = bases[1].x;
  if (yard.infield === 'dirt' || yard.infield === 'sand') {
    // whole skinned infield: diamond plus a margin, rounded toward the outfield
    const m = 10;
    const pts: [number, number][] = [[0, -8], [s + m * 0.7, s - m * 0.7]];
    for (let i = 0; i <= 10; i++) {
      const a = 45 - i * 9; // sweep right field → left field
      const r = 2 * s + m * 0.9;
      pts.push([Math.sin(a * DEG) * r, Math.cos(a * DEG) * r]);
    }
    pts.push([-s - m * 0.7, s - m * 0.7]);
    areas.push(pts);
    areas.push(circle(0, 0, 11));
    return areas;
  }
  areas.push(circle(0, -1, 9));
  areas.push(circle(mound.x, mound.y, 5));
  for (let i = 1; i <= 3; i++) areas.push(circle(bases[i].x, bases[i].y, yard.infield === 'grass' ? 3.5 : 5));
  if (yard.infield === 'paths') {
    for (let i = 0; i < 4; i++) areas.push(strip(bases[i], bases[(i + 1) % 4], 2.6));
  }
  return areas;
}

function obstaclesFromProps(yard: Yard): Obstacle[] {
  const out: Obstacle[] = [];
  for (const p of yard.props) {
    const s = p.scale ?? 1;
    switch (p.kind) {
      case 'tree':
        out.push({ kind: 'cylinder', x: p.x, y: p.y, r: 1.3 * s, h: 10 * s });
        out.push({ kind: 'canopy', x: p.x, y: p.y, r: 11 * s, h: 18 * s, rz: 8 * s });
        break;
      case 'treehouse':
        out.push({ kind: 'cylinder', x: p.x, y: p.y, r: 2 * s, h: 13 * s });
        out.push({ kind: 'canopy', x: p.x, y: p.y, r: 15 * s, h: 24 * s, rz: 10 * s });
        break;
      case 'shed':
        out.push({ kind: 'box', x: p.x, y: p.y, r: 6 * s, d: 5 * s, h: 9 * s, rot: p.rot ?? 0, bounce: 0.35 });
        break;
      case 'doghouse':
        out.push({ kind: 'box', x: p.x, y: p.y, r: 2.6 * s, d: 3 * s, h: 4 * s, rot: p.rot ?? 0, bounce: 0.3, effect: 'dog' });
        break;
      case 'car':
        out.push({ kind: 'box', x: p.x, y: p.y, r: 3.3 * s, d: 7.5 * s, h: 5 * s, rot: p.rot ?? 0, bounce: 0.45 });
        break;
      case 'hay':
        out.push({ kind: 'box', x: p.x, y: p.y, r: 2.6 * s, d: 1.6 * s, h: (p.variant ? 5.2 : 2.6) * s, rot: p.rot ?? 0, bounce: 0.1 });
        break;
      case 'birdbath':
        out.push({ kind: 'cylinder', x: p.x, y: p.y, r: 1.4 * s, h: 3.2 * s, bounce: 0.5 });
        break;
      default:
        break;
    }
  }
  return out;
}

export function surfaceAt(f: Field, x: number, y: number): Surface {
  const patches = f.yard.patches;
  for (let i = patches.length - 1; i >= 0; i--) {
    if (pointInPoly(x, y, patches[i].poly)) return patches[i].surface;
  }
  if (f.yard.infield === 'sand') {
    // the sandbox swallows the middle of the diamond
    if (Math.abs(x) + Math.abs(y - f.s) < f.s * 0.78) return 'sand';
  }
  for (const a of f.worn) if (pointInPoly(x, y, a)) return f.wornSurface === 'sand' ? 'dirt' : f.wornSurface;
  return 'grass';
}

export const isFair = (x: number, y: number) => y >= Math.abs(x) - 0.05;
/** Past first/third base — where fair/foul is decided on landing. */
export const pastBases = (f: Field, x: number, y: number) => Math.abs(x) + y > 2 * f.s;

/** Distance from home to the fence along a spray angle (deg). */
export function fenceDistance(f: Field, deg: number): number {
  const dx = Math.sin(deg * DEG) * 1000;
  const dy = Math.cos(deg * DEG) * 1000;
  let best = 1000;
  for (const seg of f.fence) {
    const hit = segIntersect(0, 0, dx, dy, seg.a[0], seg.a[1], seg.b[0], seg.b[1]);
    if (hit && hit.t * 1000 < best) best = hit.t * 1000;
  }
  return best;
}

export function fenceAt(f: Field, deg: number): FenceSeg | null {
  const dx = Math.sin(deg * DEG) * 1000;
  const dy = Math.cos(deg * DEG) * 1000;
  let best = Infinity;
  let out: FenceSeg | null = null;
  for (const seg of f.fence) {
    const hit = segIntersect(0, 0, dx, dy, seg.a[0], seg.a[1], seg.b[0], seg.b[1]);
    if (hit && hit.t < best) { best = hit.t; out = seg; }
  }
  return out;
}

export function insideFence(f: Field, x: number, y: number): boolean {
  const poly = f.fence.map((s) => s.a);
  return pointInPoly(x, y, poly);
}

/** Strike zone for a kid, in feet above the ground. */
export function strikeZone(heightFt: number) {
  return { bottom: heightFt * 0.29, top: heightFt * 0.58, half: 0.83 };
}

/** Kid height in feet from look.height (0..1). */
export const kidHeightFt = (h: number) => 4 + h * 1.25;
