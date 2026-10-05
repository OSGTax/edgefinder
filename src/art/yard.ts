import { Rng, hashString } from '../engine/rng';
import { CHALK, INK } from '../data/palette';
import type { FenceSeg, Prop, Yard } from '../data/types';
import type { Field } from '../sim/field';
import type { Camera } from '../render/camera';
import { box, disc, gableRoof, line3, poly, shade, type P3 } from './draw3d';
import { drawGrownup } from './grownup';

export interface Drawable { z: number; draw: (ctx: CanvasRenderingContext2D) => void }

const SURF_COLOR: Record<string, (y: Yard) => string> = {
  dirt: (y) => y.theme.dirt,
  sand: () => '#ecd69c',
  mud: () => '#6b4a2b',
  water: () => '#4fb3e8',
  patio: () => '#c9c2b5',
  grass: (y) => y.theme.grass,
};

// ──────────────────────────────────────────────────────────── the sky

export function drawSky(ctx: CanvasRenderingContext2D, cam: Camera, yard: Yard, t: number) {
  const hy = cam.horizonY();
  const g = ctx.createLinearGradient(0, Math.min(0, hy - cam.h), 0, Math.max(10, hy));
  g.addColorStop(0, yard.theme.sky[0]);
  g.addColorStop(1, yard.theme.sky[1]);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, cam.w, cam.h);
  if (yard.theme.time === 'sunset') {
    ctx.fillStyle = 'rgba(255,170,90,0.25)';
    ctx.fillRect(0, Math.max(0, hy - cam.h * 0.25), cam.w, cam.h * 0.25);
  }
  // drifting puffy clouds
  const rng = new Rng(hashString(yard.id));
  for (let i = 0; i < 6; i++) {
    const speed = 4 + rng.range(0, 6);
    const x = ((rng.range(0, cam.w * 1.4) + t * speed) % (cam.w * 1.4)) - cam.w * 0.2;
    const y = hy - cam.h * rng.range(0.12, 0.5);
    if (y < -40) continue;
    const s = rng.range(0.6, 1.3) * Math.min(1, cam.w / 900);
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.beginPath();
    for (const [dx, dy, r] of [[0, 0, 26], [28, -10, 22], [52, 2, 20], [-26, 4, 18]]) {
      ctx.moveTo(x + dx * s + r * s, y + dy * s);
      ctx.arc(x + dx * s, y + dy * s, r * s, 0, Math.PI * 2);
    }
    ctx.fill();
  }
}

// ───────────────────────────────────────────────────────── the ground

export function drawGround(ctx: CanvasRenderingContext2D, cam: Camera, f: Field, t: number) {
  const y = f.yard;
  // neighbors' lawns beyond the fence
  poly(ctx, cam, [[-6000, -6000, 0], [6000, -6000, 0], [6000, 6000, 0], [-6000, 6000, 0]], shade(y.theme.grass, -0.18), null);
  const fencePts: P3[] = f.fence.map((s) => [s.a[0], s.a[1], 0]);
  const sp = cam.projectPoly(fencePts);
  if (sp.length >= 3) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(sp[0][0], sp[0][1]);
    for (let i = 1; i < sp.length; i++) ctx.lineTo(sp[i][0], sp[i][1]);
    ctx.closePath();
    ctx.fillStyle = y.theme.grass;
    ctx.fill();
    ctx.clip();
    if (y.theme.mowStripes) {
      for (let x = -160; x < 160; x += 24) {
        poly(ctx, cam, [[x, -60, 0], [x + 12, -60, 0], [x + 12, 260, 0], [x, 260, 0]], y.theme.grassAlt, null);
      }
    } else {
      // patchy, lived-in lawn
      const rng = new Rng(hashString(y.id) ^ 77);
      for (let i = 0; i < 26; i++) {
        disc(ctx, cam, rng.range(-120, 120), rng.range(10, 200), 0, rng.range(6, 18), rng.range(4, 12), y.theme.grassAlt, null, rng.range(0, 3), 10);
      }
    }
    for (const p of y.patches) {
      if (p.surface === 'water') continue; // drawn by the pool/pond prop
      poly(ctx, cam, p.poly.map(([a, b]) => [a, b, 0]), SURF_COLOR[p.surface](y), shade(SURF_COLOR[p.surface](y), -0.2), 1);
    }
    const worn = f.wornSurface === 'sand' ? '#e8cf8f' : y.theme.dirt;
    for (const a of f.worn) poly(ctx, cam, a.map(([a1, b1]) => [a1, b1, 0]), worn, null);
    if (y.infield === 'sand') {
      const s = f.s;
      const r = s * 0.78;
      poly(ctx, cam, [[0, s - r, 0], [r, s, 0], [0, s + r, 0], [-r, s, 0]], '#ecd69c', '#a77b45', 3);
    }
    // chalk lines (a little wobbly — Dad did them)
    const far = 260;
    line3(ctx, cam, [0, 0, 0.02], [far, far, 0.02], CHALK, 0.3);
    line3(ctx, cam, [0, 0, 0.02], [-far, far, 0.02], CHALK, 0.3);
    for (const sx of [-1, 1]) {
      const bx = sx * 2.4;
      poly(ctx, cam, [[bx - 1.4, -2.6, 0.02], [bx + 1.4, -2.6, 0.02], [bx + 1.4, 2.6, 0.02], [bx - 1.4, 2.6, 0.02]], null, CHALK, 1.2);
    }
    ctx.restore();
  }
  drawBases(ctx, cam, f, t);
}

type BaseStyle = 'bag' | 'frisbee' | 'pizza' | 'lid' | 'towel';
const BASES: Record<string, BaseStyle[]> = {
  mudpuddle: ['frisbee', 'pizza', 'towel'],
  poolparty: ['towel', 'towel', 'towel'],
  lilypad: ['bag', 'frisbee', 'bag'],
  junklot: ['lid', 'pizza', 'lid'],
  treehouse: ['bag', 'bag', 'bag'],
  grandmabea: ['pizza', 'frisbee', 'towel'],
  sandbox: ['frisbee', 'frisbee', 'frisbee'],
  sunflower: ['bag', 'lid', 'bag'],
};

function drawBases(ctx: CanvasRenderingContext2D, cam: Camera, f: Field, t: number) {
  void t;
  // home plate
  poly(ctx, cam, [[-0.71, 0.71, 0.05], [0.71, 0.71, 0.05], [0.71, 0, 0.05], [0, -0.71, 0.05], [-0.71, 0, 0.05]], '#ffffff', INK, 1.2);
  // pitcher's "rubber" (a 2x4)
  const m = f.mound;
  poly(ctx, cam, [[-1, m.y - 0.25, 0.05], [1, m.y - 0.25, 0.05], [1, m.y + 0.25, 0.05], [-1, m.y + 0.25, 0.05]], '#c69c6d', INK, 1);
  const styles = BASES[f.yard.id] ?? ['bag', 'bag', 'bag'];
  for (let i = 1; i <= 3; i++) {
    const b = f.bases[i];
    const st = styles[i - 1];
    switch (st) {
      case 'frisbee': disc(ctx, cam, b.x, b.y, 0.08, 1.1, 1.1, ['#ff5e5b', '#ffd23f', '#3bceac'][i - 1]); break;
      case 'lid': disc(ctx, cam, b.x, b.y, 0.08, 1.2, 1.2, '#b8c1c8'); disc(ctx, cam, b.x, b.y, 0.15, 0.35, 0.12, '#8a949b'); break;
      case 'pizza': poly(ctx, cam, sq(b.x, b.y, 1.1, Math.PI / 4), '#d9b382', INK, 1.2); break;
      case 'towel':
        poly(ctx, cam, sq(b.x, b.y, 1.2, 0.3), ['#ff6fa8', '#6fc3ff', '#ffd86f'][i - 1], INK, 1.2);
        poly(ctx, cam, sq(b.x, b.y, 0.6, 0.3), '#ffffff', null, 1);
        break;
      default: poly(ctx, cam, sq(b.x, b.y, 0.85, Math.PI / 4), '#ffffff', INK, 1.2);
    }
  }
}

const sq = (x: number, y: number, r: number, rot: number): P3[] =>
  [0, 1, 2, 3].map((i) => {
    const a = rot + (i * Math.PI) / 2;
    return [x + Math.cos(a) * r, y + Math.sin(a) * r, 0.06] as P3;
  });

// ─────────────────────────────────────────────────── fences and props

export function yardDrawables(cam: Camera, f: Field, t: number): Drawable[] {
  const out: Drawable[] = [];
  const y = f.yard;
  for (const seg of f.fence) {
    if (seg.kind === 'house') { out.push(houseDrawable(cam, y, seg)); continue; }
    const L = Math.hypot(seg.b[0] - seg.a[0], seg.b[1] - seg.a[1]);
    const n = Math.max(1, Math.ceil(L / 9));
    for (let i = 0; i < n; i++) {
      const u0 = i / n, u1 = (i + 1) / n;
      const ax = seg.a[0] + (seg.b[0] - seg.a[0]) * u0, ay = seg.a[1] + (seg.b[1] - seg.a[1]) * u0;
      const bx = seg.a[0] + (seg.b[0] - seg.a[0]) * u1, by = seg.a[1] + (seg.b[1] - seg.a[1]) * u1;
      const z = cam.depth((ax + bx) / 2, (ay + by) / 2, seg.height / 2);
      if (z < 0.5 && cam.depth(ax, ay, 0) < 0.5 && cam.depth(bx, by, 0) < 0.5) continue;
      out.push({ z, draw: (ctx) => fencePanel(ctx, cam, seg, ax, ay, bx, by, i, t) });
    }
  }
  for (const p of y.props) {
    const z = cam.depth(p.x, p.y, 2);
    if (z < 0.5) continue;
    out.push({ z, draw: (ctx) => drawProp(ctx, cam, p, t, y) });
  }
  // the neighborhood all around the yard
  const rng = new Rng(hashString(y.id) ^ 0x51);
  for (let i = 0; i < 30; i++) {
    const ang = (i * 12 + rng.range(-4, 4)) * (Math.PI / 180);
    const behind = Math.cos(ang) < -0.2;
    const r = behind ? rng.range(110, 170) : rng.range(250, 330);
    const x = Math.sin(ang) * r, yy = Math.cos(ang) * r;
    if (behind && Math.abs(x) < 70 && yy > -75) continue; // that's our own house
    const z = cam.depth(x, yy, 5);
    if (z < 1) continue;
    const kind = i % 4 === 1 ? 'house' : 'tree';
    const sc = rng.range(1.2, 1.8);
    const col = rng.pick(['#e8d5b5', '#c9d6e3', '#f2c6b4', '#d5e8c0', '#f1e3a6']);
    out.push({
      z,
      draw: (ctx) => {
        if (kind === 'tree') tree(ctx, cam, x, yy, sc, t, i);
        else {
          box(ctx, cam, x, yy, 16, 12, 0, 14, ang, { top: col, sideA: col, sideB: shade(col, -0.12) });
          gableRoof(ctx, cam, x, yy, 16, 12, 14, 7, ang, '#7a4b3a', shade(col, -0.05));
        }
      },
    });
  }
  return out;
}

function fencePanel(ctx: CanvasRenderingContext2D, cam: Camera, seg: FenceSeg, ax: number, ay: number, bx: number, by: number, idx: number, t: number) {
  const h = seg.height;
  const quad: P3[] = [[ax, ay, 0], [bx, by, 0], [bx, by, h], [ax, ay, h]];
  switch (seg.kind) {
    case 'picket': {
      poly(ctx, cam, [[ax, ay, h * 0.25], [bx, by, h * 0.25], [bx, by, h * 0.32], [ax, ay, h * 0.32]], '#f3efe6', INK, 1);
      poly(ctx, cam, [[ax, ay, h * 0.7], [bx, by, h * 0.7], [bx, by, h * 0.77], [ax, ay, h * 0.77]], '#f3efe6', INK, 1);
      const n = 7;
      for (let i = 0; i < n; i++) {
        const u = (i + 0.5) / n;
        const x = ax + (bx - ax) * u, y = ay + (by - ay) * u;
        const w = 0.28 / Math.max(1, Math.hypot(bx - ax, by - ay) / 9);
        const dx = (bx - ax) * w * 0.5 / n * 9, dy = (by - ay) * w * 0.5 / n * 9;
        poly(ctx, cam, [[x - dx, y - dy, 0], [x + dx, y + dy, 0], [x + dx, y + dy, h * 0.92], [x, y, h * 1.05], [x - dx, y - dy, h * 0.92]], '#ffffff', INK, 1);
      }
      break;
    }
    case 'wood': case 'garage': case 'barn': {
      const base = seg.kind === 'barn' ? '#b5352c' : seg.kind === 'garage' ? (seg.color ?? '#9aa0a6') : '#b9875a';
      poly(ctx, cam, quad, idx % 2 ? base : shade(base, -0.06), INK, 1.2);
      const n = seg.kind === 'wood' ? 5 : 3;
      for (let i = 1; i < n; i++) {
        const u = i / n;
        line3(ctx, cam, [ax + (bx - ax) * u, ay + (by - ay) * u, 0], [ax + (bx - ax) * u, ay + (by - ay) * u, h], shade(base, -0.3), 0.08);
      }
      if (seg.kind === 'barn') {
        line3(ctx, cam, [ax, ay, h * 0.98], [bx, by, h * 0.98], '#ffffff', 0.6);
        if (idx % 3 === 1) {
          // the big white X of a barn door
          line3(ctx, cam, [ax, ay, 0.2], [bx, by, h * 0.55], '#ffffff', 0.5);
          line3(ctx, cam, [bx, by, 0.2], [ax, ay, h * 0.55], '#ffffff', 0.5);
        }
      }
      if (seg.kind === 'garage' && idx % 2 === 0) {
        for (let k = 1; k < 4; k++) line3(ctx, cam, [ax, ay, h * k * 0.18], [bx, by, h * k * 0.18], shade(base, -0.18), 0.12);
      }
      break;
    }
    case 'chain': {
      poly(ctx, cam, quad, 'rgba(170,180,190,0.35)', null);
      for (let i = 0; i <= 4; i++) {
        const u = i / 4;
        line3(ctx, cam, [ax + (bx - ax) * u, ay + (by - ay) * u, 0], [bx - (bx - ax) * (1 - u) * 0, ay + (by - ay) * u, 0], '#9aa5ae', 0.03);
      }
      for (let i = 0; i < 6; i++) {
        const u = i / 6, v = (i + 1) / 6;
        line3(ctx, cam, [ax + (bx - ax) * u, ay + (by - ay) * u, 0], [ax + (bx - ax) * v, ay + (by - ay) * v, h], '#a7b1b9', 0.04);
        line3(ctx, cam, [ax + (bx - ax) * v, ay + (by - ay) * v, 0], [ax + (bx - ax) * u, ay + (by - ay) * u, h], '#a7b1b9', 0.04);
      }
      line3(ctx, cam, [ax, ay, h], [bx, by, h], '#7f8a93', 0.2);
      line3(ctx, cam, [ax, ay, 0], [ax, ay, h], '#6c757d', 0.25);
      break;
    }
    case 'hedge': {
      const g = '#3f7f2f';
      poly(ctx, cam, quad, idx % 2 ? g : shade(g, -0.08), INK, 1.2);
      for (let i = 0; i < 4; i++) {
        const u = (i + 0.5) / 4;
        const p = cam.project(ax + (bx - ax) * u, ay + (by - ay) * u, h);
        if (!p) continue;
        ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(2, 2.2 * p.s), Math.PI, 0);
        ctx.fillStyle = shade(g, 0.08); ctx.fill(); ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke();
      }
      break;
    }
    case 'sunflower': {
      poly(ctx, cam, [[ax, ay, 0], [bx, by, 0], [bx, by, h * 0.7], [ax, ay, h * 0.7]], '#4c8a2e', INK, 1);
      for (let i = 0; i < 3; i++) {
        const u = (i + 0.5) / 3;
        const sway = Math.sin(t * 1.5 + idx + i) * 0.3;
        const p = cam.project(ax + (bx - ax) * u + sway, ay + (by - ay) * u, h);
        if (!p) continue;
        const r = Math.max(2, 1.5 * p.s);
        ctx.fillStyle = '#ffcf33';
        for (let k = 0; k < 10; k++) {
          const a = (k / 10) * Math.PI * 2;
          ctx.beginPath(); ctx.ellipse(p.x + Math.cos(a) * r, p.y + Math.sin(a) * r, r * 0.55, r * 0.28, a, 0, Math.PI * 2); ctx.fill();
        }
        ctx.beginPath(); ctx.arc(p.x, p.y, r * 0.7, 0, Math.PI * 2); ctx.fillStyle = '#6b3e1e'; ctx.fill();
      }
      break;
    }
    case 'reeds': {
      for (let i = 0; i < 6; i++) {
        const u = (i + 0.5) / 6;
        const x = ax + (bx - ax) * u, yy = ay + (by - ay) * u;
        const sway = Math.sin(t * 2 + i + idx) * 0.4;
        line3(ctx, cam, [x, yy, 0], [x + sway, yy, h * 1.4], '#5d8a3a', 0.12);
        const p = cam.project(x + sway, yy, h * 1.3);
        if (p) { ctx.fillStyle = '#6b3e1e'; ctx.beginPath(); ctx.ellipse(p.x, p.y, Math.max(1, 0.18 * p.s), Math.max(2, 0.5 * p.s), 0, 0, Math.PI * 2); ctx.fill(); }
      }
      break;
    }
    default:
      poly(ctx, cam, quad, '#b9875a', INK, 1);
  }
}

function houseDrawable(cam: Camera, y: Yard, seg: FenceSeg): Drawable {
  const x0 = Math.min(seg.a[0], seg.b[0]), x1 = Math.max(seg.a[0], seg.b[0]);
  const yy = seg.a[1];
  const cx = (x0 + x1) / 2, hw = (x1 - x0) / 2 * 0.8, hd = 16;
  const cy = yy - hd;
  const h = seg.height;
  return {
    z: cam.depth(cx, yy, h / 2),
    draw: (ctx) => {
      const th = y.theme;
      box(ctx, cam, cx, cy, hw, hd, 0, h, 0, {
        top: th.houseWall, sideA: shade(th.houseWall, -0.1), sideB: th.houseWall,
        face: (c, cm, corners, which) => {
          if (which !== 2) return; // the side facing the field
          const [bl, br] = corners;
          for (let i = 0; i < 4; i++) {
            const u = 0.12 + i * 0.22;
            const wx = bl[0] + (br[0] - bl[0]) * u;
            poly(c, cm, [[wx, yy, h * 0.45], [wx + 5, yy, h * 0.45], [wx + 5, yy, h * 0.8], [wx, yy, h * 0.8]], '#9fd3f0', INK, 1.2);
            poly(c, cm, [[wx - 0.6, yy, h * 0.42], [wx + 5.6, yy, h * 0.42], [wx + 5.6, yy, h * 0.46], [wx - 0.6, yy, h * 0.46]], th.houseTrim, INK, 1);
          }
          // back door and a porch step
          poly(c, cm, [[cx - 2.5, yy, 0], [cx + 2.5, yy, 0], [cx + 2.5, yy, 7], [cx - 2.5, yy, 7]], shade(th.houseRoof, 0.1), INK, 1.2);
        },
      });
      gableRoof(ctx, cam, cx, cy, hw, hd, h, 9, 0, th.houseRoof, th.houseWall, 2);
      // deck
      box(ctx, cam, cx, yy + 3, hw * 0.6, 3, 0, 1.2, 0, { top: '#b07a4a', sideA: '#8f5f35', sideB: '#9c6a3e' });
    },
  };
}

// ─────────────────────────────────────────────────────────────── props

function tree(ctx: CanvasRenderingContext2D, cam: Camera, x: number, y: number, s: number, t: number, seed: number) {
  line3(ctx, cam, [x, y, 0], [x, y, 10 * s], INK, 2.8 * s);
  line3(ctx, cam, [x, y, 0], [x, y, 10 * s], '#7a4f2a', 2.3 * s);
  const greens = ['#3f8a37', '#4f9e3f', '#2f7a30', '#5cae48'];
  const rng = new Rng(seed * 31 + 7);
  const blobs: [number, number, number, number][] = [];
  for (let i = 0; i < 7; i++) blobs.push([rng.range(-6, 6) * s, rng.range(-3, 3) * s, (15 + rng.range(-3, 5)) * s, rng.range(4.5, 7) * s]);
  blobs.sort((a, b) => cam.depth(x + b[0], y + b[1], b[2]) - cam.depth(x + a[0], y + a[1], a[2]));
  for (const [dx, dy, dz, r] of blobs) {
    const p = cam.project(x + dx + Math.sin(t * 0.8 + dz) * 0.2, y + dy, dz);
    if (!p) continue;
    ctx.beginPath(); ctx.arc(p.x, p.y, r * p.s, 0, Math.PI * 2);
    ctx.fillStyle = greens[Math.floor(Math.abs(dx * 7 + dz)) % greens.length];
    ctx.fill(); ctx.strokeStyle = INK; ctx.lineWidth = 1.2; ctx.stroke();
  }
}

function drawProp(ctx: CanvasRenderingContext2D, cam: Camera, p: Prop, t: number, y: Yard) {
  const s = p.scale ?? 1;
  const rot = p.rot ?? 0;
  switch (p.kind) {
    case 'tree': tree(ctx, cam, p.x, p.y, s, t, Math.floor(p.x * 13 + p.y)); break;
    case 'treehouse': {
      tree(ctx, cam, p.x, p.y, s * 1.15, t, 99);
      box(ctx, cam, p.x, p.y - 2, 5 * s, 4 * s, 12 * s, 18 * s, 0.2, { top: '#b0703a', sideA: '#9a5f2e', sideB: '#c27f45' });
      gableRoof(ctx, cam, p.x, p.y - 2, 5 * s, 4 * s, 18 * s, 4 * s, 0.2, '#c0392b', '#b0703a');
      // rope ladder
      line3(ctx, cam, [p.x + 4, p.y + 2, 12 * s], [p.x + 4.5, p.y + 3, 0], '#d9b382', 0.15);
      break;
    }
    case 'pool': {
      const c = Math.cos(rot), sn = Math.sin(rot);
      const corner = (lx: number, ly: number): P3 => [p.x + lx * c - ly * sn, p.y + lx * sn + ly * c, 0.05];
      poly(ctx, cam, [corner(-24, -13), corner(24, -13), corner(24, 13), corner(-24, 13)], '#ffffff', INK, 1.5);
      poly(ctx, cam, [corner(-22, -11), corner(22, -11), corner(22, 11), corner(-22, 11)], '#4fb3e8', '#2c7fb8', 1.5);
      for (let i = 0; i < 5; i++) {
        const wob = Math.sin(t * 2 + i) * 2;
        line3(ctx, cam, corner(-18 + i * 8 + wob, -6), corner(-14 + i * 8 + wob, -4), 'rgba(255,255,255,0.7)', 0.4);
        line3(ctx, cam, corner(-16 + i * 8 - wob, 4), corner(-12 + i * 8 - wob, 6), 'rgba(255,255,255,0.7)', 0.4);
      }
      // a pool floatie
      disc(ctx, cam, corner(8 + Math.sin(t * 0.7) * 3, 2)[0], corner(8, 2)[1], 0.3, 2.2, 2.2, '#ff6fa8');
      disc(ctx, cam, corner(8 + Math.sin(t * 0.7) * 3, 2)[0], corner(8, 2)[1], 0.35, 1, 1, '#4fb3e8', null);
      break;
    }
    case 'shed':
      box(ctx, cam, p.x, p.y, 6 * s, 5 * s, 0, 9 * s, rot, { top: '#9b6b43', sideA: '#a9784c', sideB: '#8d5f3a' });
      gableRoof(ctx, cam, p.x, p.y, 6 * s, 5 * s, 9 * s, 3.5 * s, rot, '#5d6d7e', '#a9784c');
      break;
    case 'doghouse':
      box(ctx, cam, p.x, p.y, 2.6 * s, 3 * s, 0, 3 * s, rot, { top: '#c0392b', sideA: '#d35400', sideB: '#e67e22' });
      gableRoof(ctx, cam, p.x, p.y, 2.6 * s, 3 * s, 3 * s, 1.5 * s, rot + Math.PI / 2, '#922b21', '#e67e22', 0.3);
      // the dog, napping
      disc(ctx, cam, p.x + 3, p.y + 1, 0.6, 1.5, 0.9, '#c8a27a');
      disc(ctx, cam, p.x + 4.2, p.y + 1.4, 1.0, 0.7, 0.6, '#b08a62');
      break;
    case 'swingset': {
      const c = Math.cos(rot), sn = Math.sin(rot);
      const w = (lx: number, ly: number, lz: number): P3 => [p.x + lx * c - ly * sn, p.y + lx * sn + ly * c, lz];
      for (const lx of [-6, 6]) {
        line3(ctx, cam, w(lx, -2.5, 0), w(lx, 0, 8), '#c0392b', 0.35);
        line3(ctx, cam, w(lx, 2.5, 0), w(lx, 0, 8), '#c0392b', 0.35);
      }
      line3(ctx, cam, w(-6, 0, 8), w(6, 0, 8), '#e74c3c', 0.35);
      for (const lx of [-3, 2]) {
        const sw = Math.sin(t * 2 + lx) * 1.2;
        line3(ctx, cam, w(lx, 0, 8), w(lx, sw, 2), '#7f8c8d', 0.06);
        line3(ctx, cam, w(lx + 1.2, 0, 8), w(lx + 1.2, sw, 2), '#7f8c8d', 0.06);
        line3(ctx, cam, w(lx - 0.1, sw, 2), w(lx + 1.3, sw, 2), '#2c3e50', 0.3);
      }
      break;
    }
    case 'car':
      box(ctx, cam, p.x, p.y, 3.3 * s, 7.5 * s, 0.6, 3.4, rot, { top: '#8fb3a8', sideA: '#79998f', sideB: '#6b8a80' });
      box(ctx, cam, p.x, p.y + 0.5, 3 * s, 4 * s, 3.4, 5, rot, {
        top: '#8fb3a8', sideA: '#a9d6f0', sideB: '#9cc9e3',
      });
      for (const [lx, ly] of [[-3.3, -5], [3.3, -5], [-3.3, 5], [3.3, 5]]) {
        const c = Math.cos(rot), sn = Math.sin(rot);
        disc(ctx, cam, p.x + lx * c - ly * sn, p.y + lx * sn + ly * c, 0.4, 0.9, 0.9, '#2d2a2e');
      }
      break;
    case 'grill': {
      box(ctx, cam, p.x, p.y, 1.4, 1, 2.6, 3.4, 0, { top: '#2c3e50', sideA: '#34495e', sideB: '#2c3e50' });
      for (const lx of [-1, 1]) line3(ctx, cam, [p.x + lx, p.y, 0], [p.x + lx, p.y, 2.6], '#2c3e50', 0.15);
      for (let i = 0; i < 3; i++) {
        const q = cam.project(p.x + Math.sin(t * 2 + i) * 0.6, p.y, 4 + ((t * 2 + i * 1.3) % 4));
        if (q) { ctx.fillStyle = 'rgba(220,220,220,0.5)'; ctx.beginPath(); ctx.arc(q.x, q.y, Math.max(2, 0.8 * q.s), 0, Math.PI * 2); ctx.fill(); }
      }
      break;
    }
    case 'garden': {
      const c = Math.cos(rot), sn = Math.sin(rot);
      const hw = p.variant ? 12 : 16, hd = p.variant ? 6 : 7;
      box(ctx, cam, p.x, p.y, hw, hd, 0, 0.8, rot, { top: '#6b4a2b', sideA: '#8a5a2b', sideB: '#7a4f25' });
      for (let i = -2; i <= 2; i++) for (let j = -1; j <= 1; j++) {
        const lx = i * hw * 0.38, ly = j * hd * 0.55;
        const x = p.x + lx * c - ly * sn, yy = p.y + lx * sn + ly * c;
        line3(ctx, cam, [x, yy, 0.8], [x, yy, 3], '#3f8a37', 0.4);
        const q = cam.project(x, yy, 3);
        if (q) { ctx.fillStyle = (i + j) % 2 ? '#e74c3c' : '#4caf50'; ctx.beginPath(); ctx.arc(q.x, q.y, Math.max(2, 0.6 * q.s), 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke(); }
      }
      break;
    }
    case 'trampoline':
      for (const a of [0, 1.6, 3.2, 4.7]) line3(ctx, cam, [p.x + Math.cos(a) * 6, p.y + Math.sin(a) * 6, 0], [p.x + Math.cos(a) * 6, p.y + Math.sin(a) * 6, 3], '#7f8c8d', 0.25);
      disc(ctx, cam, p.x, p.y, 3, 7, 7, '#3a7bd5');
      disc(ctx, cam, p.x, p.y, 3.05, 6, 6, '#1d1d1d', null);
      break;
    case 'birdbath':
      line3(ctx, cam, [p.x, p.y, 0], [p.x, p.y, 3], '#bdc3c7', 0.8);
      disc(ctx, cam, p.x, p.y, 3.2, 1.8, 1.8, '#d5dbdf');
      disc(ctx, cam, p.x, p.y, 3.25, 1.3, 1.3, '#7fc8ef', null);
      break;
    case 'lawnchair': {
      const c = Math.cos(rot), sn = Math.sin(rot);
      const w = (lx: number, ly: number, lz: number): P3 => [p.x + lx * c - ly * sn, p.y + lx * sn + ly * c, lz];
      poly(ctx, cam, [w(-1.2, -1, 1.2), w(1.2, -1, 1.2), w(1.2, 1, 1.2), w(-1.2, 1, 1.2)], '#3bceac', INK, 1);
      poly(ctx, cam, [w(-1.2, 1, 1.2), w(1.2, 1, 1.2), w(1.2, 1.8, 3.4), w(-1.2, 1.8, 3.4)], '#ffd23f', INK, 1);
      break;
    }
    case 'flamingo': {
      line3(ctx, cam, [p.x, p.y, 0], [p.x, p.y, 2.2], '#2c3e50', 0.08);
      const q = cam.project(p.x, p.y, 2.8);
      if (q) {
        ctx.fillStyle = '#ff6fa8'; ctx.strokeStyle = INK; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.ellipse(q.x, q.y, 0.9 * q.s, 0.5 * q.s, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(q.x + 0.6 * q.s, q.y); ctx.quadraticCurveTo(q.x + 1.3 * q.s, q.y - 1.4 * q.s, q.x + 0.9 * q.s, q.y - 1.6 * q.s);
        ctx.lineWidth = Math.max(1, 0.25 * q.s); ctx.strokeStyle = '#ff6fa8'; ctx.stroke();
      }
      break;
    }
    case 'sprinkler': {
      disc(ctx, cam, p.x, p.y, 0.2, 0.5, 0.5, '#f1c40f');
      const a = Math.sin(t * 1.2) * 1.2;
      for (let i = 0; i < 12; i++) {
        const u = i / 12;
        const dx = Math.sin(a) * u * 14, dy = Math.cos(a) * u * 14;
        const q = cam.project(p.x + dx, p.y + dy, Math.sin(u * Math.PI) * 5);
        if (q) { ctx.fillStyle = 'rgba(150,210,255,0.7)'; ctx.beginPath(); ctx.arc(q.x, q.y, Math.max(1, 0.35 * q.s), 0, Math.PI * 2); ctx.fill(); }
      }
      break;
    }
    case 'gnome': {
      const q = cam.project(p.x, p.y, 0);
      if (!q) break;
      const k = q.s;
      ctx.lineWidth = 1; ctx.strokeStyle = INK;
      ctx.fillStyle = '#2e86de'; ctx.beginPath(); ctx.ellipse(q.x, q.y - 0.5 * k, 0.5 * k, 0.55 * k, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.moveTo(q.x - 0.4 * k, q.y - 0.9 * k); ctx.lineTo(q.x + 0.4 * k, q.y - 0.9 * k); ctx.lineTo(q.x, q.y - 0.3 * k); ctx.fill();
      ctx.fillStyle = '#f5c9a3'; ctx.beginPath(); ctx.arc(q.x, q.y - 1.05 * k, 0.28 * k, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#e74c3c'; ctx.beginPath(); ctx.moveTo(q.x - 0.35 * k, q.y - 1.2 * k); ctx.lineTo(q.x + 0.35 * k, q.y - 1.2 * k); ctx.lineTo(q.x + 0.1 * k, q.y - 2.1 * k); ctx.closePath(); ctx.fill(); ctx.stroke();
      break;
    }
    case 'wagon':
      box(ctx, cam, p.x, p.y, 1.4, 2.4, 0.8, 1.9, rot, { top: '#c0392b', sideA: '#e74c3c', sideB: '#c0392b' });
      break;
    case 'tire': disc(ctx, cam, p.x, p.y, 0.6, 1.4, 1.4, '#2d2a2e'); disc(ctx, cam, p.x, p.y, 0.65, 0.7, 0.7, shade(y.theme.grass, -0.1), null); break;
    case 'barn': {
      box(ctx, cam, p.x, p.y, 34, 14, 0, 22, 0, { top: '#b5352c', sideA: '#a12f27', sideB: '#b5352c' });
      gableRoof(ctx, cam, p.x, p.y, 34, 14, 22, 12, 0, '#5d4037', '#b5352c', 1.5);
      break;
    }
    case 'hay': box(ctx, cam, p.x, p.y, 2.6, 1.6, p.variant ? 2.6 : 0, p.variant ? 5.2 : 2.6, rot, { top: '#f2d16b', sideA: '#e3bd4f', sideB: '#d9b246' }); break;
    case 'scarecrow': {
      line3(ctx, cam, [p.x, p.y, 0], [p.x, p.y, 6], '#8a5a2b', 0.3);
      line3(ctx, cam, [p.x - 2.5, p.y, 4.5], [p.x + 2.5, p.y, 4.5], '#3a6ea5', 0.6);
      const q = cam.project(p.x, p.y, 6.4);
      if (q) {
        ctx.fillStyle = '#f2d16b'; ctx.strokeStyle = INK; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(q.x, q.y, 0.7 * q.s, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#8a5a2b'; ctx.beginPath(); ctx.ellipse(q.x, q.y - 0.6 * q.s, 1.3 * q.s, 0.3 * q.s, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      break;
    }
    case 'shrub': disc(ctx, cam, p.x, p.y, 0, 4, 3, '#3f7f2f'); { const q = cam.project(p.x, p.y, 2); if (q) { ctx.fillStyle = '#4f9e3f'; ctx.beginPath(); ctx.arc(q.x, q.y, 3 * q.s, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke(); } } break;
    case 'flowers': {
      for (let i = 0; i < 6; i++) {
        const q = cam.project(p.x + (i - 2.5) * 1.6, p.y, 1);
        if (q) { ctx.fillStyle = ['#ff7eb6', '#ffd23f', '#ffffff', '#c39bd3'][i % 4]; ctx.beginPath(); ctx.arc(q.x, q.y, Math.max(1.5, 0.5 * q.s), 0, Math.PI * 2); ctx.fill(); }
      }
      break;
    }
    case 'lemonade': {
      box(ctx, cam, p.x, p.y, 2.5, 1.2, 0, 2.8, 0, { top: '#f6e05e', sideA: '#ffffff', sideB: '#f2f2f2' });
      const q = cam.project(p.x, p.y + 1.3, 4.3);
      if (q && q.s > 2) {
        ctx.fillStyle = '#fffbe6'; ctx.strokeStyle = INK; ctx.lineWidth = 1;
        ctx.fillRect(q.x - 3 * q.s, q.y - 0.9 * q.s, 6 * q.s, 1.8 * q.s); ctx.strokeRect(q.x - 3 * q.s, q.y - 0.9 * q.s, 6 * q.s, 1.8 * q.s);
        ctx.fillStyle = '#e67e22'; ctx.font = `bold ${Math.max(6, 0.9 * q.s)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('LEMONADE 25¢', q.x, q.y);
      }
      break;
    }
    case 'bench': box(ctx, cam, p.x, p.y, 3, 0.8, 1.2, 1.6, 0, { top: '#a0703c', sideA: '#8a5a2b', sideB: '#7a4f25' }); break;
    case 'grownup': {
      const q = cam.project(p.x, p.y, 0);
      if (q) drawGrownup(ctx, q.x, q.y, q.s, p.variant ?? 0, t);
      break;
    }
    case 'tractor':
      box(ctx, cam, p.x, p.y, 2.6, 4.5, 1.2, 4.5, rot, { top: '#2e7d32', sideA: '#388e3c', sideB: '#1b5e20' });
      for (const [lx, ly, r] of [[-2.8, -3, 2.4], [2.8, -3, 2.4], [-2.6, 3, 1.3], [2.6, 3, 1.3]]) {
        const c = Math.cos(rot), sn = Math.sin(rot);
        const q = cam.project(p.x + lx * c - ly * sn, p.y + lx * sn + ly * c, r);
        if (q) { ctx.fillStyle = '#2d2a2e'; ctx.beginPath(); ctx.arc(q.x, q.y, r * q.s, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#f1c40f'; ctx.beginPath(); ctx.arc(q.x, q.y, r * 0.45 * q.s, 0, Math.PI * 2); ctx.fill(); }
      }
      break;
    case 'clothesline': {
      line3(ctx, cam, [p.x - 14, p.y, 0], [p.x - 14, p.y, 7], '#7f8c8d', 0.25);
      line3(ctx, cam, [p.x + 14, p.y, 0], [p.x + 14, p.y, 7], '#7f8c8d', 0.25);
      line3(ctx, cam, [p.x - 14, p.y, 6.8], [p.x + 14, p.y, 6.8], '#ecf0f1', 0.05);
      const cols = ['#e74c3c', '#ffffff', '#3498db', '#f1c40f', '#9b59b6'];
      for (let i = 0; i < 5; i++) {
        const x = p.x - 10 + i * 5, sw = Math.sin(t * 2 + i) * 0.4;
        poly(ctx, cam, [[x - 1.5, p.y, 6.8], [x + 1.5, p.y, 6.8], [x + 1.5 + sw, p.y, 4.4], [x - 1.5 + sw, p.y, 4.4]], cols[i], INK, 1);
      }
      break;
    }
    case 'cattails':
      for (let i = 0; i < 5; i++) {
        const x = p.x + (i - 2) * 1.5, sway = Math.sin(t * 2 + i) * 0.4;
        line3(ctx, cam, [x, p.y, 0], [x + sway, p.y, 5], '#5d8a3a', 0.12);
        const q = cam.project(x + sway, p.y, 4.6);
        if (q) { ctx.fillStyle = '#6b3e1e'; ctx.beginPath(); ctx.ellipse(q.x, q.y, Math.max(1, 0.2 * q.s), Math.max(2, 0.55 * q.s), 0, 0, Math.PI * 2); ctx.fill(); }
      }
      break;
    case 'lilypads':
      disc(ctx, cam, p.x, p.y, 0.02, 40, 70, '#4f9fc8', null);
      for (let i = 0; i < 8; i++) disc(ctx, cam, p.x - 20 + i * 6, p.y - 30 + (i % 3) * 22, 0.05, 2, 1.6, '#4caf50');
      break;
    case 'sandbox':
      // the whole infield is the sandbox here; add a bucket and shovel
      box(ctx, cam, p.x + 9, p.y + 4, 0.7, 0.7, 0, 1.3, 0.3, { top: '#e74c3c', sideA: '#e74c3c', sideB: '#c0392b' });
      line3(ctx, cam, [p.x - 8, p.y - 3, 0], [p.x - 6, p.y - 2, 1.4], '#3498db', 0.25);
      break;
    default:
      break;
  }
}
