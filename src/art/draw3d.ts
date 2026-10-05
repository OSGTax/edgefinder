import type { Camera } from '../render/camera';
import { INK } from '../data/palette';

// Small helpers for drawing flat-shaded 3D geometry through the camera.

export type P3 = [number, number, number];

export function poly(ctx: CanvasRenderingContext2D, cam: Camera, pts: P3[], fill: string | CanvasGradient | CanvasPattern | null, stroke: string | null = INK, lw = 1.5): boolean {
  const sp = cam.projectPoly(pts);
  if (sp.length < 3) return false;
  ctx.beginPath();
  ctx.moveTo(sp[0][0], sp[0][1]);
  for (let i = 1; i < sp.length; i++) ctx.lineTo(sp[i][0], sp[i][1]);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.lineJoin = 'round'; ctx.stroke(); }
  return true;
}

export function line3(ctx: CanvasRenderingContext2D, cam: Camera, a: P3, b: P3, color: string, worldWidth: number, minPx = 1) {
  const pa = cam.project(a[0], a[1], a[2]);
  const pb = cam.project(b[0], b[1], b[2]);
  if (!pa || !pb) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(minPx, worldWidth * (pa.s + pb.s) * 0.5);
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
}

/** A ground-plane ellipse/disc (approximated by a polygon). */
export function disc(ctx: CanvasRenderingContext2D, cam: Camera, x: number, y: number, z: number, rx: number, ry: number, fill: string, stroke: string | null = INK, rot = 0, n = 18) {
  const pts: P3[] = [];
  const c = Math.cos(rot), s = Math.sin(rot);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    const lx = Math.cos(a) * rx, ly = Math.sin(a) * ry;
    pts.push([x + lx * c - ly * s, y + lx * s + ly * c, z]);
  }
  poly(ctx, cam, pts, fill, stroke, 1.2);
}

export interface BoxStyle {
  top: string;
  sideA: string;
  sideB: string;
  stroke?: string | null;
  /** draw details on a visible side face: corners are [bl, br, tr, tl] */
  face?: (ctx: CanvasRenderingContext2D, cam: Camera, corners: P3[], which: number) => void;
  noTop?: boolean;
}

/** Axis-aligned-ish box (rotated about z), back faces culled. */
export function box(ctx: CanvasRenderingContext2D, cam: Camera, cx: number, cy: number, hw: number, hd: number, z0: number, z1: number, rot: number, st: BoxStyle) {
  const c = Math.cos(rot), s = Math.sin(rot);
  const corner = (lx: number, ly: number): [number, number] => [cx + lx * c - ly * s, cy + lx * s + ly * c];
  const base = [corner(-hw, -hd), corner(hw, -hd), corner(hw, hd), corner(-hw, hd)];
  const cp = cam.pose.pos;
  const faces: { i: number; d: number }[] = [];
  for (let i = 0; i < 4; i++) {
    const a = base[i], b = base[(i + 1) % 4];
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
    // outward normal
    const nx = mx - cx, ny = my - cy;
    if (nx * (cp.x - mx) + ny * (cp.y - my) > 0) faces.push({ i, d: Math.hypot(cp.x - mx, cp.y - my) });
  }
  faces.sort((p, q) => q.d - p.d);
  for (const { i } of faces) {
    const a = base[i], b = base[(i + 1) % 4];
    const corners: P3[] = [[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]];
    poly(ctx, cam, corners, i % 2 ? st.sideA : st.sideB, st.stroke === undefined ? INK : st.stroke, 1.4);
    st.face?.(ctx, cam, corners, i);
  }
  if (!st.noTop && cp.z > z1) poly(ctx, cam, base.map(([x, y]) => [x, y, z1] as P3), st.top, st.stroke === undefined ? INK : st.stroke, 1.4);
}

/** A gable roof over a box footprint (ridge along the local x axis). */
export function gableRoof(ctx: CanvasRenderingContext2D, cam: Camera, cx: number, cy: number, hw: number, hd: number, z: number, rise: number, rot: number, color: string, gable: string, over = 0.6) {
  const c = Math.cos(rot), s = Math.sin(rot);
  const w = (lx: number, ly: number, lz: number): P3 => [cx + lx * c - ly * s, cy + lx * s + ly * c, lz];
  const W = hw + over, D = hd + over;
  const r1 = w(-W, 0, z + rise), r2 = w(W, 0, z + rise);
  const slopes: P3[][] = [
    [w(-W, -D, z), w(W, -D, z), r2, r1],
    [w(W, D, z), w(-W, D, z), r1, r2],
  ];
  const gables: P3[][] = [
    [w(-hw, -hd, z), w(-hw, hd, z), w(-hw, 0, z + rise)],
    [w(hw, hd, z), w(hw, -hd, z), w(hw, 0, z + rise)],
  ];
  const cp = cam.pose.pos;
  const dist = (pts: P3[]) => {
    const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length, my = pts.reduce((a, p) => a + p[1], 0) / pts.length;
    return Math.hypot(cp.x - mx, cp.y - my);
  };
  const items = [...slopes.map((p) => ({ p, f: color })), ...gables.map((p) => ({ p, f: gable }))];
  items.sort((a, b) => dist(b.p) - dist(a.p));
  for (const it of items) poly(ctx, cam, it.p, it.f, INK, 1.4);
}

/** Shade a hex color: amt < 0 darkens, > 0 lightens. */
export function shade(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  if (amt < 0) { r *= 1 + amt; g *= 1 + amt; b *= 1 + amt; }
  else { r += (255 - r) * amt; g += (255 - g) * amt; b += (255 - b) * amt; }
  const h = (v: number) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0');
  return `#${h(r)}${h(g)}${h(b)}`;
}
