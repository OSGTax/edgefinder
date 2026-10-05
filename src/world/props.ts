import {
  CatmullRomCurve3, Color, CylinderGeometry, DoubleSide, Group, InstancedMesh, LatheGeometry, Matrix4,
  MeshStandardMaterial, Object3D, SphereGeometry, Vector2, Vector3,
} from 'three';
import { Batch, T, boxFt, boxUp, cyl, extrude, quad, sphere, torus, tube } from '../gfx/build';
import { M } from '../gfx/materials';
import { leafAtlas, paintTex } from '../gfx/textures';
import { mulberry } from '../gfx/noise';
import { W } from '../gfx/units';
import { leafCards } from './fences';

// Backyard props, each added into a shared Batch at a sim-space position.
// Local frames: x right, y up, z toward the viewer (front = +z).

const P = (x: number, y: number) => W(x, y, 0);

// ───────────────────────────────────────────────────────────── bases & mound

/** Home plate: a dusty rubber pentagon set flush in the dirt. */
export function homePlate(b: Batch) {
  const w = 1.42, s = w / 2;
  const g = extrude([[-s, 0], [s, 0], [s, s], [0, w], [-s, s]], 0.08);
  g.rotateX(-Math.PI / 2);
  b.add(M.paint('#f2efe6', 0.85), g, T(0, 0.03, 0), { castShadow: false });
  b.add(M.paint('#2a2a2a', 0.8), extrude([[-s - 0.06, -0.06], [s + 0.06, -0.06], [s + 0.06, s], [0, w + 0.08], [-s - 0.06, s]], 0.04).rotateX(-Math.PI / 2), T(0, 0.0, 0), { castShadow: false });
}

/** The homemade bases: a folded beach towel, a flattened pizza box and a doormat. */
export function bases(b: Batch, pts: { x: number; y: number }[]) {
  const towel = M.tex('towelBase', paintTex(128, 128, (c) => {
    const cols = ['#ff8a3d', '#ffffff', '#ffcf3d', '#ffffff'];
    for (let i = 0; i < 8; i++) { c.fillStyle = cols[i % 4]; c.fillRect(i * 16, 0, 16, 128); }
    c.fillStyle = 'rgba(120,80,40,0.25)'; c.fillRect(0, 96, 128, 32);
  }), { roughness: 0.95 });
  const pizza = M.tex('pizzaBase', paintTex(128, 128, (c) => {
    c.fillStyle = '#c9a06a'; c.fillRect(0, 0, 128, 128);
    c.strokeStyle = '#a7804e'; c.lineWidth = 3; c.strokeRect(6, 6, 116, 116);
    c.fillStyle = '#b3262a'; c.font = 'bold 26px sans-serif'; c.textAlign = 'center'; c.fillText('PIZZA', 64, 58);
    c.font = 'bold 13px sans-serif'; c.fillText('HOT & FRESH', 64, 80);
    c.fillStyle = 'rgba(90,60,30,0.3)'; c.beginPath(); c.arc(92, 100, 14, 0, 7); c.fill();
  }), { roughness: 0.95 });
  const mat = M.tex('matBase', paintTex(128, 128, (c) => {
    c.fillStyle = '#7a5b3a'; c.fillRect(0, 0, 128, 128);
    for (let i = 0; i < 700; i++) { c.fillStyle = `rgba(${60 + (i % 40)},${40 + (i % 30)},20,0.4)`; c.fillRect((i * 37) % 128, (i * 61) % 128, 2, 2); }
    c.fillStyle = '#3a2a1a'; c.font = 'bold 22px serif'; c.textAlign = 'center'; c.fillText('WELCOME', 64, 72);
  }), { roughness: 1 });
  const looks = [towel, pizza, mat];
  pts.forEach((p, i) => {
    const m = looks[i % 3];
    const th = i === 0 ? 0.22 : 0.08;
    const v = P(p.x, p.y);
    b.add(m, boxFt(1.5, th, 1.5), T(v.x, th / 2, v.z, Math.PI / 4 + (i - 1) * 0.06));
  });
}

/** Pitching "rubber": a scrap 2×4 pressed into the mound. */
export function rubber(b: Batch, x: number, y: number) {
  const v = P(x, y);
  b.add(M.woodSolid('#b7926a', 2), boxFt(2, 0.12, 0.33), T(v.x, 0.04, v.z, 0.03), { castShadow: false });
}

// ───────────────────────────────────────────────────────────── patio

export function patioSet(b: Batch, x: number, y: number, rot = 0) {
  const c = P(x, y);
  const base = T(c.x, 0.33, c.z, rot);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  const frame = M.metal('#2f3133', 0.55);
  // round glass-top table
  b.add(frame, torus(2.0, 0.07, 6, 32).rotateX(Math.PI / 2), at(0, 2.35, 0));
  b.add(M.paint('#9fc3c9', 0.05, 0.1), cyl(2.0, 2.0, 0.05, 32), at(0, 2.38, 0));
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + 0.4;
    b.add(frame, tube([[Math.cos(a) * 1.95, 2.35, Math.sin(a) * 1.95], [Math.cos(a) * 1.6, 1.2, Math.sin(a) * 1.6], [Math.cos(a) * 1.75, 0, Math.sin(a) * 1.75]], 0.06, 10, 5), at(0, 0, 0));
  }
  // umbrella: pole, canopy with scalloped valance, ribs
  b.add(M.paint('#e9e3d6', 0.4), cyl(0.09, 0.09, 8.2, 8), at(0, 4.1, 0));
  const canopy = new LatheGeometry([new Vector2(0.05, 1.6), new Vector2(1.6, 1.25), new Vector2(3.6, 0.55), new Vector2(4.6, 0)], 16);
  const stripe = M.stripes(['#d93a3a', '#fff6e6'], true);
  b.add(stripe, canopy, at(0, 6.6, 0));
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2;
    b.add(i % 2 ? M.paint('#fff6e6', 0.8) : M.paint('#d93a3a', 0.8), boxFt(1.75, 0.55, 0.03), at(Math.sin(a + Math.PI / 16) * 4.55, 6.35, Math.cos(a + Math.PI / 16) * 4.55, a + Math.PI / 16 + Math.PI / 2));
  }
  b.add(M.paint('#e9e3d6', 0.4), sphere(0.18, 10, 8), at(0, 8.3, 0));
  // chairs around it
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
    const cx = Math.sin(a) * 3.3, cz = Math.cos(a) * 3.3;
    const ry = a + Math.PI;
    const cm = (lx: number, ly: number, lz: number, rx = 0) => base.clone().multiply(T(cx, 0, cz, ry)).multiply(T(lx, ly, lz, 0, rx));
    b.add(M.paint('#4f6f52', 0.6), boxFt(1.6, 0.16, 1.6), cm(0, 1.5, 0));
    b.add(M.paint('#4f6f52', 0.6), boxFt(1.6, 1.6, 0.14), cm(0, 2.4, -0.82, -0.12));
    for (const [lx, lz] of [[-0.7, -0.7], [0.7, -0.7], [-0.7, 0.7], [0.7, 0.7]]) b.add(frame, cyl(0.05, 0.05, 1.5, 5), cm(lx, 0.75, lz));
    b.add(frame, boxFt(0.08, 0.08, 1.5), cm(-0.8, 2.15, 0));
    b.add(frame, boxFt(0.08, 0.08, 1.5), cm(0.8, 2.15, 0));
  }
  // lemonade pitcher + cups
  b.add(M.paint('#fff26b', 0.15), cyl(0.28, 0.3, 0.75, 14), at(0.6, 2.8, 0.3));
  for (let i = 0; i < 3; i++) b.add(M.paint(['#ff6b6b', '#4dabf7', '#69db7c'][i], 0.5), cyl(0.13, 0.1, 0.35, 10), at(-0.5 + i * 0.4, 2.58, -0.7 + (i % 2) * 0.3));
}

/** Mr. Mendoza's gas grill, with lid, side shelves, knobs and a propane tank. */
export function gasGrill(b: Batch, x: number, y: number, rot = 0): Vector3 {
  const c = P(x, y);
  const base = T(c.x, 0.33, c.z, rot);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  const black = M.paint('#232425', 0.45, 0.5);
  const steel = M.metal('#c4c8cc', 0.3);
  b.add(black, boxFt(2.6, 1.6, 1.8), at(0, 0.9, 0));
  b.add(steel, boxFt(2.5, 1.4, 0.04), at(0, 0.9, 0.92));
  b.add(black, boxFt(2.8, 0.8, 2.0), at(0, 2.1, 0));
  // domed lid with a thermometer and handle
  b.add(black, cyl(1.0, 1.0, 2.8, 16, false).rotateZ(Math.PI / 2), at(0, 2.5, -0.1, 0, 0, 0).multiply(new Matrix4().makeScale(1, 0.6, 0.95)));
  b.add(steel, tube([[-1.0, 2.75, 1.05], [-0.9, 2.95, 1.25], [0.9, 2.95, 1.25], [1.0, 2.75, 1.05]], 0.05, 12, 6), at(0, 0, 0));
  b.add(M.paint('#f3f3f3', 0.3), cyl(0.15, 0.15, 0.05, 12).rotateX(Math.PI / 2), at(0, 2.9, 0.92));
  for (let i = 0; i < 4; i++) b.add(M.paint('#111111', 0.4), cyl(0.09, 0.09, 0.14, 10).rotateX(Math.PI / 2), at(-0.9 + i * 0.6, 1.95, 1.06));
  for (const sx of [-1, 1]) {
    b.add(steel, boxFt(1.3, 0.08, 1.7), at(sx * 2.05, 2.25, 0));
    b.add(black, boxFt(0.08, 0.5, 1.6), at(sx * 2.65, 2.0, 0));
  }
  // propane tank peeking under the cart
  b.add(M.paint('#e8e8e2', 0.35), cyl(0.5, 0.5, 1.2, 14), at(0.6, 0.75, -1.1));
  b.add(M.paint('#e8e8e2', 0.35), sphere(0.5, 14, 8), at(0.6, 1.35, -1.1));
  for (const [lx, lz] of [[-1.2, -0.8], [1.2, -0.8], [-1.2, 0.8], [1.2, 0.8]]) b.add(M.paint('#111', 0.5), cyl(0.12, 0.12, 0.2, 8), at(lx, 0.1, lz));
  // spatula + tongs on the side shelf, and burgers on the grate (lid propped open)
  b.add(steel, boxFt(0.3, 0.02, 0.9), at(2.1, 2.31, 0.3, 0.3));
  b.add(M.paint('#5a3a22', 0.5), boxFt(0.12, 0.08, 0.7), at(2.0, 2.33, -0.5, 0.3));
  return new Vector3(c.x, 3.4, c.z);
}

export function cooler(b: Batch, x: number, y: number, rot = 0) {
  const c = P(x, y);
  const base = T(c.x, 0.33, c.z, rot);
  b.add(M.paint('#d9342b', 0.4), boxFt(2.4, 1.3, 1.4), base.clone().multiply(T(0, 0.65, 0)));
  b.add(M.paint('#f5f5f0', 0.35), boxFt(2.5, 0.3, 1.5), base.clone().multiply(T(0, 1.45, 0)));
  b.add(M.paint('#f5f5f0', 0.35), boxFt(1.0, 0.12, 0.2), base.clone().multiply(T(0, 1.65, 0)));
}

/** String lights: catenary strands between points (three space), with bulbs. */
export function stringLights(b: Batch, strands: [Vector3, Vector3][]) {
  const wire = M.paint('#1b1b1b', 0.6);
  const bulb = M.bulb('#ffe7a8');
  for (const [a, c] of strands) {
    const pts: Vector3[] = [];
    const n = 24;
    const L = a.distanceTo(c);
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      const p = a.clone().lerp(c, t);
      p.y -= Math.sin(t * Math.PI) * L * 0.06;
      pts.push(p);
    }
    const curve = new CatmullRomCurve3(pts);
    b.add(wire, tube(pts.map((p) => [p.x, p.y, p.z]), 0.025, 40, 3), undefined, { castShadow: false });
    const bulbs = Math.floor(L / 2);
    for (let i = 1; i < bulbs; i++) {
      const p = curve.getPoint(i / bulbs);
      b.add(bulb, sphere(0.14, 8, 6), T(p.x, p.y - 0.22, p.z, 0, 0, 0, [1, 1.3, 1]), { castShadow: false });
    }
  }
}

// ───────────────────────────────────────────────────────────── lawn things

export function lawnFlamingo(b: Batch, x: number, y: number, rot = 0, s = 1) {
  const c = P(x, y);
  const base = T(c.x, 0, c.z, rot, 0, 0, s);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0, sc: number | [number, number, number] = 1) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz, sc));
  const pink = M.paint('#ff6fae', 0.3);
  b.add(pink, sphere(0.55, 16, 10), at(0, 2.4, 0, 0, 0, -0.25, [1.25, 0.8, 0.7]));
  b.add(pink, sphere(0.35, 12, 8), at(-0.65, 2.55, 0, 0, 0, 0.5, [1.2, 0.6, 0.55]));
  b.add(pink, tube([[0.5, 2.5, 0], [0.75, 3.0, 0], [0.55, 3.5, 0], [0.7, 3.8, 0]], 0.09, 16, 6), at(0, 0, 0));
  b.add(pink, sphere(0.2, 12, 8), at(0.75, 3.85, 0));
  b.add(M.paint('#222', 0.4), cyl(0.07, 0.03, 0.35, 8), at(0.95, 3.7, 0, 0, 0, 0.9));
  b.add(M.paint('#111', 0.3), sphere(0.04, 6, 5), at(0.8, 3.92, 0.13));
  const wire = M.metal('#8a8f93', 0.4);
  b.add(wire, cyl(0.025, 0.025, 2.0, 4), at(-0.08, 1.0, 0.1));
  b.add(wire, cyl(0.025, 0.025, 2.0, 4), at(0.08, 1.0, -0.1));
}

/** Classic folding lawn chair with woven webbing. */
export function lawnChair(b: Batch, x: number, y: number, rot = 0, webbing: [string, string] = ['#2f9e66', '#ffffff']) {
  const c = P(x, y);
  const base = T(c.x, 0, c.z, rot);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  const al = M.metal('#d8dde1', 0.25);
  const web = M.stripes([webbing[0], webbing[1], webbing[0], webbing[1], webbing[0], webbing[1]], true);
  // seat + back webbing
  b.add(web, boxFt(1.8, 0.04, 1.6), at(0, 1.3, 0));
  b.add(web, boxFt(1.8, 2.0, 0.04), at(0, 2.3, -0.95, 0, 0.18));
  // aluminium tube frame
  for (const sx of [-0.92, 0.92]) {
    b.add(al, tube([[sx, 0, 0.8], [sx, 1.3, 0.8], [sx, 1.3, -0.8], [sx, 3.3, -1.15]], 0.045, 16, 5), at(0, 0, 0));
    b.add(al, tube([[sx, 0, -0.9], [sx, 1.3, 0.2], [sx, 1.95, 0.55], [sx, 1.95, -0.8]], 0.045, 16, 5), at(0, 0, 0));
  }
  b.add(M.paint('#f2f2ee', 0.5), boxFt(0.2, 0.08, 1.5), at(-0.92, 1.99, -0.1));
  b.add(M.paint('#f2f2ee', 0.5), boxFt(0.2, 0.08, 1.5), at(0.92, 1.99, -0.1));
}

export function gnome(b: Batch, x: number, y: number, rot = 0, s = 1) {
  const c = P(x, y);
  const base = T(c.x, 0, c.z, rot, 0, 0, s);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0, sc: number | [number, number, number] = 1) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz, sc));
  b.add(M.paint('#3b6fb6', 0.5), cyl(0.35, 0.5, 0.9, 14), at(0, 0.45, 0));
  b.add(M.paint('#5a3b22', 0.6), cyl(0.36, 0.36, 0.12, 14), at(0, 0.75, 0));
  b.add(M.paint('#f2c8a0', 0.5), sphere(0.3, 14, 10), at(0, 1.15, 0));
  b.add(M.paint('#f5f5f0', 0.8), sphere(0.32, 14, 10), at(0, 0.95, 0.12, 0, 0, 0, [1, 1.2, 0.8]));
  b.add(M.paint('#d33a2c', 0.45), cyl(0.0, 0.34, 0.9, 14), at(0, 1.65, -0.05, 0, -0.25));
  b.add(M.paint('#f2a58a', 0.5), sphere(0.08, 8, 6), at(0, 1.15, 0.3));
  b.add(M.concrete('#9a958b'), cyl(0.55, 0.6, 0.12, 16), at(0, 0.06, 0));
}

/** Flower bed plants: stems with colourful blossoms (instanced) and little shrubs (leaf cards). */
export function flowers(group: Group, spots: { x: number; y: number; r: number; n: number; palette?: string[] }[], seed = 9) {
  const rnd = mulberry(seed);
  const stems: Matrix4[] = [];
  const blooms: { m: Matrix4; c: Color }[] = [];
  const o = new Object3D();
  const col = new Color();
  for (const s of spots) {
    const pal = s.palette ?? ['#ff5e7e', '#ffd43b', '#ffffff', '#c084fc', '#ff922b'];
    const c = P(s.x, s.y);
    for (let i = 0; i < s.n; i++) {
      const a = rnd() * Math.PI * 2, r = Math.sqrt(rnd()) * s.r;
      const h = 0.8 + rnd() * 1.1;
      o.position.set(c.x + Math.cos(a) * r, h / 2, c.z + Math.sin(a) * r);
      o.rotation.set((rnd() - 0.5) * 0.25, 0, (rnd() - 0.5) * 0.25);
      o.scale.set(1, h, 1);
      o.updateMatrix();
      stems.push(o.matrix.clone());
      o.position.y = h;
      o.scale.setScalar(0.8 + rnd() * 0.5);
      o.rotation.set(rnd() * 0.5, rnd() * 6, 0);
      o.updateMatrix();
      blooms.push({ m: o.matrix.clone(), c: col.set(pal[Math.floor(rnd() * pal.length)]).clone() });
    }
  }
  const stemMesh = new InstancedMesh(new CylinderGeometry(0.025, 0.03, 1, 4), M.paint('#3f7a2e', 0.8), stems.length);
  stems.forEach((m, i) => stemMesh.setMatrixAt(i, m));
  const bloomGeo = new SphereGeometry(0.17, 7, 5);
  bloomGeo.scale(1, 0.55, 1);
  const bloomMesh = new InstancedMesh(bloomGeo, new MeshStandardMaterial({ roughness: 0.6 }), blooms.length);
  blooms.forEach((bl, i) => { bloomMesh.setMatrixAt(i, bl.m); bloomMesh.setColorAt(i, bl.c); });
  stemMesh.castShadow = bloomMesh.castShadow = true;
  stemMesh.receiveShadow = bloomMesh.receiveShadow = true;
  group.add(stemMesh, bloomMesh);
}

/** Round shrubs made of leaf cards. */
export function shrubs(group: Group, spots: { x: number; y: number; r: number }[], seed = 3, hue = 100) {
  const rnd = mulberry(seed);
  const mats: Matrix4[] = [];
  const nrm: number[] = [];
  const o = new Object3D();
  for (const s of spots) {
    const c = P(s.x, s.y);
    const n = Math.round(s.r * s.r * 26);
    for (let i = 0; i < n; i++) {
      const u = new Vector3(rnd() * 2 - 1, rnd(), rnd() * 2 - 1);
      if (u.lengthSq() > 1 || u.lengthSq() < 0.01) { i--; continue; }
      u.normalize();
      const rr = s.r * (0.7 + rnd() * 0.3);
      o.position.set(c.x + u.x * rr, u.y * rr * 0.85 + 0.2, c.z + u.z * rr);
      o.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
      o.scale.setScalar(0.6 + rnd() * 0.4);
      o.updateMatrix();
      mats.push(o.matrix.clone());
      nrm.push(u.x, u.y + 0.3, u.z);
    }
  }
  const inst = leafCards(mats, leafAtlas(256, hue), '#bcd6a0', 1.1, { wind: 0.4, normals: new Float32Array(nrm) });
  inst.name = 'shrubs';
  group.add(inst);
  return inst;
}

/** A kid's bike lying on its side in the grass. */
export function bike(b: Batch, x: number, y: number, rot = 0, color = '#2b8be0') {
  const c = P(x, y);
  const base = T(c.x, 0.12, c.z, rot, 0, Math.PI / 2 - 0.12);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  const tire = M.paint('#1d1d1d', 0.85);
  for (const lx of [-1.45, 1.45]) {
    b.add(tire, torus(0.85, 0.09, 8, 24), at(lx, 0, 0));
    b.add(M.metal('#c9cdd1', 0.3), cyl(0.07, 0.07, 0.25, 8).rotateX(Math.PI / 2), at(lx, 0, 0));
    for (let k = 0; k < 6; k++) b.add(M.metal('#d6dadd', 0.3), boxFt(1.6, 0.02, 0.02), at(lx, 0, 0, 0, 0, (k / 6) * Math.PI));
  }
  const fr = M.paint(color, 0.35, 0.3);
  b.add(fr, tube([[-1.45, 0, 0], [-0.3, 0.9, 0], [0.9, 0.95, 0], [1.45, 0, 0]], 0.07, 12, 6), at(0, 0, 0));
  b.add(fr, tube([[-1.45, 0, 0], [0.1, 0.05, 0], [0.9, 0.95, 0]], 0.06, 12, 6), at(0, 0, 0));
  b.add(fr, cyl(0.06, 0.06, 1.0, 6), at(-0.3, 1.2, 0, 0, 0, 0.3));
  b.add(M.paint('#222', 0.6), boxFt(0.8, 0.15, 0.4), at(-0.45, 1.68, 0));
  b.add(M.metal('#c9cdd1', 0.3), tube([[1.0, 1.4, -0.8], [1.15, 1.6, 0], [1.0, 1.4, 0.8]], 0.05, 10, 5), at(0, 0, 0));
  b.add(fr, cyl(0.06, 0.06, 0.6, 6), at(1.1, 1.2, 0, 0, 0, -0.3));
}

/** A wooden bench + blanket "dugout" with a hand-painted team sign and a bucket of bats. */
export function dugout(b: Batch, x: number, y: number, rot: number, team: { name: string; primary: string; secondary: string; accent: string }) {
  const c = P(x, y);
  const base = T(c.x, 0, c.z, rot);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  const wood = M.woodSolid('#a9825a', 3);
  // long bench
  b.add(wood, boxFt(12, 0.25, 1.4), at(0, 1.55, -1.5));
  for (const lx of [-5.5, 0, 5.5]) b.add(wood, boxFt(0.3, 1.5, 1.2), at(lx, 0.75, -1.5));
  // picnic blanket in team colours
  const blanket = M.tex(`blanket${team.name}`, paintTex(128, 128, (g) => {
    g.fillStyle = team.accent; g.fillRect(0, 0, 128, 128);
    g.fillStyle = team.primary;
    for (let i = 0; i < 8; i += 2) { g.globalAlpha = 0.8; g.fillRect(i * 16, 0, 16, 128); g.fillRect(0, i * 16, 128, 16); }
    g.globalAlpha = 1;
  }, true), { roughness: 1 });
  b.add(blanket, boxFt(10, 0.04, 6), at(0, 0.03, 2.4, 0.03), { castShadow: false });
  // sign on stakes
  const sign = M.tex(`sign${team.name}`, paintTex(512, 160, (g) => {
    g.fillStyle = '#f1e3c4'; g.fillRect(0, 0, 512, 160);
    g.strokeStyle = team.primary; g.lineWidth = 10; g.strokeRect(8, 8, 496, 144);
    g.fillStyle = team.primary; g.font = 'bold 64px "Trebuchet MS", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.save(); g.translate(256, 82); g.rotate(-0.03); g.fillText(team.name.toUpperCase(), 0, 0); g.restore();
    g.fillStyle = team.secondary; g.font = 'bold 22px sans-serif'; g.fillText('★ DUGOUT ★ NO GROWNUPS ★', 256, 136);
  }), { roughness: 0.9 });
  b.add(sign, quad(6, 1.9), at(0, 4.0, -2.35));
  b.add(wood, boxFt(6.3, 2.1, 0.12), at(0, 4.0, -2.43));
  for (const lx of [-2.8, 2.8]) b.add(wood, boxFt(0.25, 4.4, 0.25), at(lx, 2.2, -2.55));
  // bucket of bats + a water jug
  b.add(M.paint('#e6e6e0', 0.5), cyl(0.7, 0.6, 1.6, 16, true), at(6.6, 0.8, -0.6));
  for (let i = 0; i < 4; i++) b.add(M.woodSolid('#d9b27c', 2), cyl(0.12, 0.05, 2.7, 8), at(6.6 + Math.cos(i * 1.7) * 0.3, 1.7, -0.6 + Math.sin(i * 1.7) * 0.3, 0, (i - 1.5) * 0.12, Math.sin(i) * 0.15));
  b.add(M.paint(team.secondary, 0.4), cyl(0.55, 0.55, 1.3, 16), at(-6.6, 0.65, -1.2));
  b.add(M.paint('#f5f5f0', 0.4), cyl(0.45, 0.55, 0.3, 16), at(-6.6, 1.45, -1.2));
}

/** Garden hose: a green tube snaking from a wall spigot across the lawn. */
export function gardenHose(b: Batch, pts: [number, number][], start: Vector3) {
  const p3: [number, number, number][] = [[start.x, start.y, start.z]];
  for (const [x, y] of pts) { const v = P(x, y); p3.push([v.x, 0.08, v.z]); }
  b.add(M.paint('#2f8f3a', 0.45), tube(p3, 0.07, pts.length * 12, 6), undefined, { castShadow: true });
  // the reel by the wall
  b.add(M.paint('#2f8f3a', 0.45), torus(0.8, 0.12, 6, 20), T(start.x, 1.2, start.z - 0.4, 0));
  b.add(M.paint('#2f8f3a', 0.45), torus(0.6, 0.12, 6, 20), T(start.x, 1.2, start.z - 0.4, 0));
}

/** A simple wooden swing set. */
export function swingSet(b: Batch, x: number, y: number, rot = 0) {
  const c = P(x, y);
  const base = T(c.x, 0, c.z, rot);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  const wood = M.woodSolid('#9a7350', 3);
  b.add(wood, boxFt(14, 0.5, 0.5), at(0, 8, 0));
  for (const sx of [-6.8, 6.8]) for (const sz of [-1, 1]) b.add(wood, boxFt(0.45, 8.6, 0.45), at(sx, 4, sz * 2.0, 0, sz * -0.26));
  for (const [lx, col] of [[-3, '#e03131'], [1, '#1971c2']] as const) {
    for (const off of [-0.7, 0.7]) b.add(M.metal('#888', 0.5), cyl(0.02, 0.02, 6, 4), at(lx + off, 4.9, 0));
    b.add(M.paint(col, 0.5), boxFt(1.7, 0.12, 0.6), at(lx, 1.9, 0));
  }
  // slide
  b.add(M.paint('#f2c94c', 0.35), boxFt(1.8, 0.12, 9), at(5, 3, 4.2, 0, -0.62));
  b.add(wood, boxFt(2.4, 0.3, 2.4), at(5, 6.0, 0));
}

/** Round trampoline with a safety net. */
export function trampoline(b: Batch, x: number, y: number) {
  const c = P(x, y);
  const at = (lx: number, ly: number, lz: number) => T(c.x + lx, ly, c.z + lz);
  b.add(M.paint('#1d1d1d', 0.9), cyl(6, 6, 0.06, 32), at(0, 2.7, 0));
  b.add(M.paint('#2f6fd0', 0.5), torus(6.3, 0.35, 6, 32).rotateX(Math.PI / 2), at(0, 2.75, 0));
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    b.add(M.metal('#9aa0a6', 0.4), cyl(0.08, 0.08, 2.7, 5), at(Math.cos(a) * 6.3, 1.35, Math.sin(a) * 6.3));
    b.add(M.paint('#222', 0.6), cyl(0.06, 0.06, 5.5, 5), at(Math.cos(a) * 6.4, 5.4, Math.sin(a) * 6.4));
  }
  const net = new MeshStandardMaterial({ color: '#1d1d1d', transparent: true, opacity: 0.28, side: DoubleSide, roughness: 1, depthWrite: false });
  net.name = 'trampNet';
  b.add(net, cyl(6.4, 6.4, 5.4, 32, true), at(0, 5.4, 0), { castShadow: false });
}

/** Garden shed with a gambrel roof. */
export function shed(b: Batch, x: number, y: number, rot = 0, wall = '#b8452f') {
  const c = P(x, y);
  const base = T(c.x, 0, c.z, rot);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  b.add(M.wood(wall, 8, [4, 6]), boxUp(10, 7, 8), at(0, 0, 0));
  b.add(M.trim(), boxFt(3, 6, 0.12), at(0, 3, 4.03));
  b.add(M.trim(), boxFt(0.2, 6, 0.14), at(0, 3, 4.08));
  b.add(M.trim(), boxFt(3, 0.2, 0.14), at(0, 3, 4.1, 0, 0, 0.9));
  b.add(M.trim(), boxFt(3, 0.2, 0.14), at(0, 3, 4.1, 0, 0, -0.9));
  b.add(M.shingle('#4a4f55'), boxFt(5.4, 0.3, 9), at(-3.4, 8.3, 0, 0, 0, 0.95));
  b.add(M.shingle('#4a4f55'), boxFt(5.4, 0.3, 9), at(3.4, 8.3, 0, 0, 0, -0.95));
  b.add(M.shingle('#4a4f55'), boxFt(3.6, 0.3, 9), at(-1.5, 10.6, 0, 0, 0, 0.3));
  b.add(M.shingle('#4a4f55'), boxFt(3.6, 0.3, 9), at(1.5, 10.6, 0, 0, 0, -0.3));
  b.add(M.wood(wall, 8, [4, 6]), extrude([[-5, 0], [5, 0], [4.4, 2.6], [1.7, 4.1], [-1.7, 4.1], [-4.4, 2.6]], 8), at(0, 7, 0));
}

/** A boxy family car parked at the curb. */
export function car(b: Batch, x: number, y: number, rot = 0, color = '#5c7ea8') {
  const c = P(x, y);
  const base = T(c.x, 0, c.z, rot);
  const at = (lx: number, ly: number, lz: number, ry = 0, rx = 0, rz = 0) => base.clone().multiply(T(lx, ly, lz, ry, rx, rz));
  const paint = M.paint(color, 0.25, 0.4);
  b.add(paint, boxFt(6, 2.2, 15), at(0, 2.0, 0));
  b.add(paint, boxFt(5.6, 1.9, 8), at(0, 3.9, -0.5));
  b.add(M.glass(), boxFt(5.3, 1.6, 8.1), at(0, 4.0, -0.5));
  b.add(M.glass(), boxFt(5.65, 1.5, 7.2), at(0, 4.0, -0.5));
  b.add(M.chrome(), boxFt(6.1, 0.4, 0.3), at(0, 1.3, 7.55));
  b.add(M.chrome(), boxFt(6.1, 0.4, 0.3), at(0, 1.3, -7.55));
  b.add(M.bulb('#fff8e0'), boxFt(1, 0.5, 0.1), at(2.1, 2.2, 7.52), { castShadow: false });
  b.add(M.bulb('#fff8e0'), boxFt(1, 0.5, 0.1), at(-2.1, 2.2, 7.52), { castShadow: false });
  b.add(M.paint('#c22', 0.3), boxFt(1, 0.5, 0.1), at(2.1, 2.2, -7.52), { castShadow: false });
  b.add(M.paint('#c22', 0.3), boxFt(1, 0.5, 0.1), at(-2.1, 2.2, -7.52), { castShadow: false });
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    b.add(M.paint('#1b1b1b', 0.85), cyl(1.15, 1.15, 0.8, 18).rotateZ(Math.PI / 2), at(sx * 2.75, 1.15, sz * 4.8));
    b.add(M.chrome(), cyl(0.6, 0.6, 0.82, 14).rotateZ(Math.PI / 2), at(sx * 2.76, 1.15, sz * 4.8));
  }
}

