import { Group, type Material } from 'three';
import { Batch, T, boxFt, cyl, gable, quad, type Adder } from '../gfx/build';
import { M } from '../gfx/materials';
import { WINDOW_VARIANTS, windowTex } from '../gfx/textures';

// The Mendozas' two-story house sits behind home plate. Built in three space:
// x = toward first base, z = toward the street (the yard is at -z),
// the back wall of the house is the plane z = FACADE.

export interface HouseStyle {
  wall: string;
  roof: string;
  trim: string;
  shutter: string;
  door: string;
  curtain: string;
}

export const FACADE = 28;

export interface WindowSpec { x: number; y: number; w: number; h: number; kind?: 'double' | 'slider' | 'picture'; shutters?: boolean }

/** Window or glass door on a wall facing -z at depth zFace: glass, casing, sill, optional shutters. */
export function addWindow(b: Adder, zFace: number, s: WindowSpec, style: HouseStyle, seed: number, facing: 1 | -1 = -1) {
  const trim = M.paint(style.trim, 0.55);
  const glassMat = M.tex(`win${s.kind ?? 'double'}${style.curtain}`, windowTex(s.kind ?? 'double', style.curtain), { roughness: 0.08, metalness: 0.15 });
  const ry = facing === -1 ? Math.PI : 0;
  const zf = zFace + facing * 0.03;
  // pick this window's cell of the atlas
  const glass = quad(s.w, s.h);
  const uv = glass.attributes.uv, cell = seed % WINDOW_VARIANTS;
  for (let i = 0; i < uv.count; i++) uv.setX(i, (uv.getX(i) + cell) / WINDOW_VARIANTS);
  b.add(glassMat, glass, T(s.x, s.y + s.h / 2, zf, ry), { castShadow: false });
  const t = 0.35, d = 0.28;
  const zc = zFace + facing * d / 2;
  b.add(trim, boxFt(s.w + t * 2, t, d), T(s.x, s.y + s.h + t / 2, zc));
  b.add(trim, boxFt(t, s.h, d), T(s.x - s.w / 2 - t / 2, s.y + s.h / 2, zc));
  b.add(trim, boxFt(t, s.h, d), T(s.x + s.w / 2 + t / 2, s.y + s.h / 2, zc));
  // head cap and a deeper sill
  b.add(trim, boxFt(s.w + t * 2 + 0.3, 0.12, d + 0.12), T(s.x, s.y + s.h + t + 0.06, zFace + facing * (d + 0.12) / 2));
  if (s.kind !== 'slider') b.add(trim, boxFt(s.w + t * 2 + 0.2, 0.22, 0.6), T(s.x, s.y - 0.11, zFace + facing * 0.3));
  if (s.shutters) {
    const sh = M.paint(style.shutter, 0.6);
    for (const side of [-1, 1]) {
      const sx = s.x + side * (s.w / 2 + t + 0.75);
      b.add(sh, boxFt(1.3, s.h + 0.3, 0.12), T(sx, s.y + s.h / 2, zFace + facing * 0.08));
      // louvers
      for (let k = 0; k < Math.floor(s.h / 0.35); k++) {
        b.add(sh, boxFt(1.0, 0.06, 0.08), T(sx, s.y + 0.3 + k * 0.35, zFace + facing * 0.17, 0, 0.5));
      }
    }
  }
}

function door(b: Batch, x: number, y: number, z: number, w: number, h: number, style: HouseStyle) {
  const trim = M.paint(style.trim, 0.55);
  const dm = M.paint(style.door, 0.45);
  b.add(dm, boxFt(w, h, 0.18), T(x, y + h / 2, z - 0.05));
  // raised panels
  for (const [px, py, pw, ph] of [[-0.62, 0.55, 0.9, 1.9], [0.62, 0.55, 0.9, 1.9], [-0.62, 0.66, 0.9, 1.5], [0.62, 0.66, 0.9, 1.5]] as const) {
    const yy = py < 0.6 ? y + 0.6 + ph / 2 : y + h - 0.5 - ph / 2;
    b.add(dm, boxFt(pw, ph, 0.08), T(x + px * (w / 3), yy, z - 0.18));
  }
  // small window in the top of the door
  b.add(M.glass(), quad(w * 0.6, 1.2), T(x, y + h - 1.3, z - 0.2, Math.PI), { castShadow: false });
  b.add(M.metal('#c9a227', 0.3), cyl(0.12, 0.12, 0.2, 10), T(x + w / 2 - 0.4, y + 3.2, z - 0.25, 0, Math.PI / 2));
  const t = 0.4;
  b.add(trim, boxFt(w + t * 2, t, 0.35), T(x, y + h + t / 2, z - 0.15));
  b.add(trim, boxFt(t, h, 0.35), T(x - w / 2 - t / 2, y + h / 2, z - 0.15));
  b.add(trim, boxFt(t, h, 0.35), T(x + w / 2 + t / 2, y + h / 2, z - 0.15));
}

/** A gable roof over [x0,x1] × [z0,z1] with the ridge along x, eaves at height `eave`. */
function roofAlongX(b: Batch, x0: number, x1: number, z0: number, z1: number, eave: number, pitch: number, over: number, rake: number, style: HouseStyle) {
  const sh = M.shingle(style.roof);
  const trim = M.paint(style.trim, 0.55);
  const half = (z1 - z0) / 2;
  const zc = (z0 + z1) / 2;
  const rise = half * pitch;
  const ang = Math.atan(pitch);
  const slope = (half + over) / Math.cos(ang);
  const len = x1 - x0 + rake * 2;
  const xc = (x0 + x1) / 2;
  const th = 0.45;
  for (const side of [-1, 1]) {
    // plane centre: halfway along the slope from ridge to eave
    const mid = (half + over) / 2;
    const yy = eave + rise - mid * pitch + th / 2;
    b.add(sh, boxFt(len, th, slope), T(xc, yy, zc + side * mid, 0, side * ang));
    // fascia + gutter along the eave
    const ez = zc + side * (half + over);
    const ey = eave - over * pitch;
    b.add(trim, boxFt(len, 0.7, 0.12), T(xc, ey + 0.05, ez + side * 0.04));
    b.add(M.paint('#e9e7e1', 0.4, 0.3), boxFt(len, 0.38, 0.42), T(xc, ey - 0.12, ez + side * 0.28));
  }
  // ridge cap
  b.add(M.shingle(style.roof), boxFt(len, 0.3, 1.1), T(xc, eave + rise + th * 0.9, zc));
  // gable ends (siding triangles) and rake boards
  const wall = M.siding(style.wall);
  for (const gx of [x0, x1]) {
    b.add(wall, gable(z1 - z0, rise, 0.4), T(gx + (gx === x0 ? 0.2 : -0.2), eave, zc, Math.PI / 2));
    for (const side of [-1, 1]) {
      b.add(trim, boxFt(0.15, 0.7, slope), T(gx + (gx === x0 ? -rake : rake), eave + rise - (half + over) * pitch / 2 + 0.25, zc + side * (half + over) / 2, 0, side * ang));
    }
  }
}

export function buildHouse(style: HouseStyle): Group {
  const b = new Batch();
  const wall = M.siding(style.wall);
  const trim = M.paint(style.trim, 0.55);
  const found = M.concrete('#b9b4ab');
  const F = FACADE;

  // ── main house: 60 ft wide, 34 deep, two stories
  const hx0 = -46, hx1 = 14, hz1 = F + 34;
  const base = 1.5, eave = 19.5;
  b.add(found, boxFt(hx1 - hx0 + 0.3, base, hz1 - F + 0.3), T((hx0 + hx1) / 2, base / 2, (F + hz1) / 2));
  b.add(wall, boxFt(hx1 - hx0, eave - base, hz1 - F), T((hx0 + hx1) / 2, base + (eave - base) / 2, (F + hz1) / 2));
  // floor band + corner boards
  b.add(trim, boxFt(hx1 - hx0 + 0.3, 0.7, 0.2), T((hx0 + hx1) / 2, 10.6, F - 0.1));
  b.add(trim, boxFt(hx1 - hx0 + 0.3, 0.9, 0.2), T((hx0 + hx1) / 2, eave - 0.45, F - 0.1));
  for (const cx of [hx0, hx1]) b.add(trim, boxFt(0.55, eave - base, 0.55), T(cx, base + (eave - base) / 2, F + 0.1));
  roofAlongX(b, hx0, hx1, F, hz1, eave, 0.5, 1.6, 1.1, style);

  // cross gable over the sliding door
  const gx0 = -35, gx1 = -19, gc = (gx0 + gx1) / 2, gh = (gx1 - gx0) / 2, gp = 0.78;
  b.add(wall, gable(gx1 - gx0, gh * gp, 0.4), T(gc, eave, F - 0.05));
  {
    const sh = M.shingle(style.roof);
    const ang = Math.atan(gp);
    const over = 1.4;
    const slope = (gh + over) / Math.cos(ang);
    const depth = 14;
    for (const side of [-1, 1]) {
      b.add(sh, boxFt(slope, 0.45, depth), T(gc + side * (gh + over) / 2, eave + (gh * gp) / 2 - (over * gp) / 2 + 0.25, F + depth / 2 - 1.6, 0, 0, -side * ang));
      b.add(trim, boxFt(slope, 0.7, 0.15), T(gc + side * (gh + over) / 2, eave + (gh * gp) / 2 - (over * gp) / 2, F - 1.65, 0, 0, -side * ang));
    }
    // round attic vent in the gable
    b.add(trim, cyl(1.25, 1.25, 0.25, 24), T(gc, eave + 2.6, F - 0.15, 0, Math.PI / 2));
    b.add(M.paint('#5b5f62', 0.7), cyl(1.0, 1.0, 0.3, 24), T(gc, eave + 2.6, F - 0.2, 0, Math.PI / 2));
    for (let k = -3; k <= 3; k++) b.add(trim, boxFt(Math.sqrt(1 - (k / 3.6) ** 2) * 2, 0.1, 0.12), T(gc, eave + 2.6 + k * 0.27, F - 0.36));
  }

  // windows: upstairs four with shutters, downstairs kitchen + one
  let seed = 1;
  for (const wx of [-41, -30.5, -23.5, -9, 5]) {
    addWindow(b, F, { x: wx, y: 12.4, w: 3, h: 4.6, shutters: wx !== -30.5 && wx !== -23.5 }, style, seed++);
  }
  addWindow(b, F, { x: -40.5, y: 4.6, w: 4.4, h: 3.6, kind: 'double', shutters: true }, style, seed++);
  addWindow(b, F, { x: 5, y: 3.6, w: 3, h: 5, shutters: true }, style, seed++);
  // sliding glass door onto the patio
  addWindow(b, F, { x: -27, y: base + 0.1, w: 8, h: 6.9, kind: 'slider' }, style, seed++);
  b.add(M.metal('#b8bcc0', 0.4), boxFt(8.6, 0.2, 0.5), T(-27, base, F - 0.2));
  // back door + storm-door frame, stoop and steps
  door(b, -8, base, F, 3, 6.8, style);
  b.add(M.concrete('#c4bfb6'), boxFt(6, base - 0.05, 4), T(-8, (base - 0.05) / 2, F - 2));
  b.add(M.concrete('#c4bfb6'), boxFt(6, 0.5, 1.2), T(-8, 0.25, F - 4.6));
  b.add(M.concrete('#c4bfb6'), boxFt(6, 0.95, 0.9), T(-8, 0.475, F - 4.1 + 0.1));
  // railing
  const rail = M.metal('#2b2b2b', 0.5);
  for (const sx of [-11, -5]) {
    b.add(rail, cyl(0.08, 0.08, 3, 6), T(sx, base + 1.5, F - 3.8));
    b.add(rail, cyl(0.08, 0.08, 3, 6), T(sx, base + 1.5, F - 0.4));
    b.add(rail, boxFt(0.12, 0.12, 3.6), T(sx, base + 3, F - 2.1));
  }
  // porch light by the door
  b.add(M.metal('#3a3a3a', 0.5), boxFt(0.6, 1.0, 0.5), T(-10.2, base + 6.2, F - 0.3));
  b.add(M.bulb('#fff2cc'), boxFt(0.4, 0.6, 0.3), T(-10.2, base + 6.1, F - 0.5), { castShadow: false });

  // downspouts
  const gutter = M.paint('#e9e7e1', 0.4, 0.3);
  for (const dx of [hx0 + 0.6, hx1 - 0.6]) {
    b.add(gutter, boxFt(0.3, eave - 0.6, 0.25), T(dx, eave / 2 + 0.2, F - 0.25));
    b.add(gutter, boxFt(0.3, 0.25, 1.2), T(dx, 0.3, F - 0.75));
  }

  // chimney on the 3B-side gable end
  const brick = M.brick();
  b.add(brick, boxFt(2.6, 32, 5), T(hx0 - 1.3, 16, F + 17));
  b.add(M.concrete('#9d978d'), boxFt(3.1, 0.5, 5.5), T(hx0 - 1.3, 32.2, F + 17));
  b.add(M.metal('#3b3b3b', 0.6), cyl(0.35, 0.35, 1.4, 10), T(hx0 - 1.3, 33.1, F + 16));

  // ── garage: one story, gable end facing the yard
  const gz1 = F + 24, gX0 = 14, gX1 = 40, gEave = 10.5;
  b.add(found, boxFt(gX1 - gX0, 0.6, gz1 - F + 0.2), T((gX0 + gX1) / 2, 0.3, (F + gz1) / 2));
  b.add(wall, boxFt(gX1 - gX0, gEave - 0.6, gz1 - F), T((gX0 + gX1) / 2, 0.6 + (gEave - 0.6) / 2, (F + gz1) / 2));
  b.add(trim, boxFt(0.55, gEave - 0.6, 0.55), T(gX1, 0.6 + (gEave - 0.6) / 2, F + 0.1));
  {
    const sh = M.shingle(style.roof);
    const half = (gX1 - gX0) / 2, pitch = 0.42, over = 1.3, ang = Math.atan(pitch);
    const slope = (half + over) / Math.cos(ang);
    const xc = (gX0 + gX1) / 2;
    const len = gz1 - F + 2.2;
    for (const side of [-1, 1]) {
      b.add(sh, boxFt(slope, 0.45, len), T(xc + side * (half + over) / 2, gEave + (half * pitch) / 2 - (over * pitch) / 2 + 0.25, (F + gz1) / 2 - 0.1, 0, 0, -side * ang));
      b.add(trim, boxFt(slope, 0.7, 0.15), T(xc + side * (half + over) / 2, gEave + (half * pitch) / 2 - (over * pitch) / 2, F - 1.2, 0, 0, -side * ang));
      b.add(trim, boxFt(0.12, 0.7, len), T(xc + side * (half + over), gEave - over * pitch + 0.05, (F + gz1) / 2 - 0.1));
      b.add(gutter, boxFt(0.42, 0.38, len), T(xc + side * (half + over + 0.25), gEave - over * pitch - 0.1, (F + gz1) / 2 - 0.1));
    }
    b.add(wall, gable(gX1 - gX0, half * pitch, 0.4), T(xc, gEave, F - 0.05));
    b.add(trim, boxFt(gX1 - gX0 + 0.3, 0.7, 0.2), T(xc, gEave - 0.35, F - 0.15));
    // little louvered vent in the gable
    b.add(trim, boxFt(2.2, 1.6, 0.2), T(xc, gEave + 2, F - 0.2));
    for (let k = 0; k < 5; k++) b.add(M.paint('#d8d5cc', 0.6), boxFt(1.8, 0.1, 0.12), T(xc, gEave + 1.45 + k * 0.28, F - 0.32, 0, 0.6));
  }
  addWindow(b, F, { x: 21, y: 3.8, w: 3.2, h: 3.8, shutters: true }, style, seed++);
  door(b, 33.5, 0.6, F, 3, 6.8, style);
  b.add(M.concrete('#c4bfb6'), boxFt(5, 0.55, 3.2), T(33.5, 0.27, F - 1.6));

  // ── privacy fence filling the gaps to the property lines, with a gate
  const cedar = M.wood('#a77a52', 6, [3, 6]);
  const post = M.woodSolid('#94694a');
  for (const [x0, x1] of [[-60, -46], [40, 60]] as const) {
    const L = x1 - x0;
    b.add(cedar, boxFt(L, 6, 0.12), T((x0 + x1) / 2, 3.15, F + 0.6));
    for (let px = x0; px <= x1 + 0.01; px += L / Math.ceil(L / 7)) b.add(post, boxFt(0.38, 6.4, 0.38), T(px, 3.2, F + 0.8));
    b.add(post, boxFt(L, 0.3, 0.2), T((x0 + x1) / 2, 1.2, F + 0.75));
    b.add(post, boxFt(L, 0.3, 0.2), T((x0 + x1) / 2, 5.1, F + 0.75));
  }
  // gate latch
  b.add(M.metal('#2b2b2b', 0.5), boxFt(0.2, 0.5, 0.15), T(50, 3.6, F + 0.5));

  // ── AC unit and its pad by the garage corner
  b.add(M.concrete('#aaa59c'), boxFt(3.6, 0.3, 3.6), T(11, 0.15, F - 2.2));
  b.add(M.paint('#c9c9c4', 0.5, 0.4), boxFt(3, 2.8, 3), T(11, 1.7, F - 2.2));
  b.add(M.metal('#4a4a4a', 0.6), cyl(1.2, 1.2, 0.1, 20), T(11, 3.15, F - 2.2));
  for (let k = -5; k <= 5; k++) b.add(M.metal('#6a6a6a', 0.6), boxFt(0.05, 2.4, 3.02), T(11 + k * 0.27, 1.7, F - 2.2));

  // ── flower bed along the back wall (soil strip; plants come from props)
  b.add(M.mulch(), boxFt(18, 0.35, 3), T(4, 0.1, F - 1.5), { castShadow: false });
  b.add(M.concrete('#a39d93'), boxFt(18.4, 0.5, 0.25), T(4, 0.2, F - 3.05));

  const g = b.build('house');
  return g;
}

export const MENDOZA_STYLE: HouseStyle = {
  wall: '#efd4b4', roof: '#7a4636', trim: '#f6f3ec', shutter: '#2f5d62', door: '#9c2f2a', curtain: '#efe1c0',
};

export type { Material };
