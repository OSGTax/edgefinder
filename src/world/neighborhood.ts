import { Group, Matrix4, Vector3 } from 'three';
import { Batch, T, boxFt, cyl, gable, tube } from '../gfx/build';
import { terrainHeight } from '../gfx/ground';
import { M } from '../gfx/materials';
import { paintTex } from '../gfx/textures';
import { mulberry } from '../gfx/noise';
import { W } from '../gfx/units';
import { addWindow, type HouseStyle } from './house';
import { car, shed, swingSet, trampoline } from './props';
import { buildFarTrees, type TreeSpec } from './trees';

// The neighbourhood around the Mendozas' yard: the houses next door, the
// street out front with parked cars, telephone poles, a ring of trees and a
// water tower on the hill.

interface HouseOpts {
  /** sim position of the house centre and the direction its back faces (radians, sim) */
  x: number; y: number; rot: number;
  w: number; d: number; stories: 1 | 2;
  style: HouseStyle;
  seed: number;
}

/** A tidy suburban house: box + gable roof, windows on front and back, a door, a chimney. */
function simpleHouse(batch: Batch, o: HouseOpts) {
  const c = W(o.x, o.y, 0);
  const b = batch.at(new Matrix4().copy(T(c.x, 0, c.z, o.rot)));
  const wall = M.siding(o.style.wall);
  const trim = M.paint(o.style.trim, 0.55);
  const eave = o.stories === 2 ? 19 : 10.5;
  const pitch = 0.55;
  b.add(M.concrete('#b9b4ab'), boxFt(o.w + 0.3, 1.4, o.d + 0.3), T(0, 0.7, 0));
  b.add(wall, boxFt(o.w, eave - 1.4, o.d), T(0, 1.4 + (eave - 1.4) / 2, 0));
  const half = o.d / 2, over = 1.4, ang = Math.atan(pitch), rise = half * pitch;
  const slope = (half + over) / Math.cos(ang);
  for (const side of [-1, 1]) {
    const mid = (half + over) / 2;
    b.add(M.shingle(o.style.roof), boxFt(o.w + 2, 0.45, slope), T(0, eave + rise - mid * pitch + 0.22, side * mid, 0, side * ang));
    b.add(trim, boxFt(o.w + 2, 0.6, 0.12), T(0, eave - over * pitch, side * (half + over)));
  }
  for (const sx of [-1, 1]) b.add(wall, gable(o.d, rise, 0.4), T(sx * (o.w / 2 - 0.2), eave, 0, Math.PI / 2));
  b.add(M.brick(), boxFt(3, eave + rise + 4, 3), T(-o.w / 2 + 6, (eave + rise + 4) / 2, -half + 6));
  // windows on the facade facing the yard (+z local) and the far side
  const rnd = mulberry(o.seed);
  const nWin = Math.max(2, Math.floor(o.w / 11));
  for (const face of [1, -1] as const) {
    for (let i = 0; i < nWin; i++) {
      const x = -o.w / 2 + ((i + 0.5) / nWin) * o.w;
      for (let st = 0; st < o.stories; st++) {
        if (st === 0 && face === 1 && i === Math.floor(nWin / 2)) continue; // the back door goes here
        const hh = 4.4;
        addWindow(b, face * half, { x, y: 3.6 + st * 9.5, w: 3, h: hh, shutters: rnd() > 0.4 }, o.style, o.seed + i + st * 7, face);
      }
    }
  }
  // back door with a little stoop
  const dx = -o.w / 2 + ((Math.floor(nWin / 2) + 0.5) / nWin) * o.w;
  b.add(M.paint(o.style.door, 0.45), boxFt(3, 6.8, 0.2), T(dx, 1.4 + 3.4, half + 0.05));
  b.add(trim, boxFt(3.8, 7.3, 0.12), T(dx, 1.4 + 3.55, half + 0.01));
  b.add(M.concrete('#c4bfb6'), boxFt(5, 1.3, 3), T(dx, 0.65, half + 1.5));
}


export interface Neighborhood { group: Group }

/** All the neighbours' windows share one curtain colour, so they share one window atlas (one draw call). */
const NEIGHBOR_CURTAIN = '#f0e8d8';

/** Real (leafy) trees in the neighbours' yards, close enough to deserve detail. */
export const NEIGHBOR_TREES: TreeSpec[] = [
  { kind: 'maple', x: -110, z: -290, crownY: 30, crownR: 16, crownRv: 14 },
  { kind: 'oak', x: 70, z: -305, crownY: 30, crownR: 20, crownRv: 13 },
  { kind: 'maple', x: 215, z: -180 },
  { kind: 'oak', x: -235, z: -200, crownY: 28, crownR: 18 },
  { kind: 'pine', x: 250, z: -40, height: 48 },
  { kind: 'maple', x: -262, z: -40, crownY: 30, crownR: 17, crownRv: 14 },
  { kind: 'oak', x: -170, z: 212, crownY: 26, crownR: 18 },
  { kind: 'maple', x: 40, z: 218 },
  { kind: 'pine', x: 150, z: 205, height: 44 },
];

export function buildNeighborhood(): Neighborhood {
  const group = new Group();
  group.name = 'neighborhood';
  const b = new Batch();

  const styles: HouseStyle[] = [
    { wall: '#b9d3e6', roof: '#4a4f55', trim: '#ffffff', shutter: '#1f3b5a', door: '#2b4a7a', curtain: NEIGHBOR_CURTAIN },
    { wall: '#f2e2a6', roof: '#6b4a3a', trim: '#ffffff', shutter: '#6b4a3a', door: '#3a6b4a', curtain: NEIGHBOR_CURTAIN },
    { wall: '#eeeae2', roof: '#3d4a3c', trim: '#ffffff', shutter: '#2f5d3a', door: '#7a2a2a', curtain: NEIGHBOR_CURTAIN },
    { wall: '#d9c0d6', roof: '#55504a', trim: '#f8f6f0', shutter: '#5a3a5a', door: '#3a3a3a', curtain: NEIGHBOR_CURTAIN },
    { wall: '#c9d9c0', roof: '#5a4636', trim: '#ffffff', shutter: '#3a5a3a', door: '#a33a2a', curtain: NEIGHBOR_CURTAIN },
    { wall: '#e8c9a8', roof: '#4a4f55', trim: '#ffffff', shutter: '#4a3a2a', door: '#2a4a6a', curtain: NEIGHBOR_CURTAIN },
  ];
  // rot: the house's local +z (its back door side) should face the Mendoza yard
  const face = (x: number, y: number, tx: number, ty: number) => Math.atan2(tx - x, -(ty - y));
  const houses: HouseOpts[] = [
    { x: 0, y: 275, rot: 0, w: 52, d: 30, stories: 2, style: styles[0], seed: 11 },
    { x: 196, y: 80, rot: 0, w: 58, d: 28, stories: 1, style: styles[1], seed: 23 },
    { x: -206, y: 96, rot: 0, w: 46, d: 30, stories: 2, style: styles[2], seed: 37 },
    { x: -150, y: 250, rot: 0, w: 44, d: 28, stories: 2, style: styles[3], seed: 41 },
    { x: 160, y: 240, rot: 0, w: 48, d: 28, stories: 1, style: styles[4], seed: 53 },
    // across the street, facing the street (and the back of the Mendozas' house)
    { x: -100, y: -170, rot: 0, w: 50, d: 30, stories: 2, style: styles[5], seed: 61 },
    { x: -10, y: -172, rot: 0, w: 46, d: 30, stories: 1, style: styles[1], seed: 67 },
    { x: 82, y: -168, rot: 0, w: 52, d: 30, stories: 2, style: styles[0], seed: 71 },
  ];
  for (const h of houses) {
    // north side houses face south toward us, street houses face north
    const ty = h.y < 0 ? 0 : 60;
    h.rot = face(h.x, h.y, h.y < 0 ? h.x : 0, ty);
    simpleHouse(b, h);
  }

  // ── the street out front (behind the Mendozas' house)
  const st = W(0, -100, 0);
  b.add(M.asphalt(), boxFt(1200, 0.1, 30), T(st.x, 0.03, st.z), { castShadow: false });
  for (const sz of [-1, 1]) {
    b.add(M.concrete('#bdb8ae'), boxFt(1200, 0.5, 0.6), T(st.x, 0.25, st.z + sz * 15.3));
    b.add(M.concrete('#cfcac0'), boxFt(1200, 0.18, 5), T(st.x, 0.09, st.z + sz * 19.5), { castShadow: false });
  }
  // Mendoza driveway from the garage to the street
  b.add(M.concrete('#c9c4ba'), boxFt(20, 0.14, 44), T(27, 0.07, 52 + 22), { castShadow: false });
  // driveways across the street
  for (const h of houses.filter((h) => h.y < 0)) {
    const v = W(h.x + h.w / 2 - 8, -136, 0);
    b.add(M.concrete('#c9c4ba'), boxFt(16, 0.14, 32), T(v.x, 0.07, v.z), { castShadow: false });
  }
  car(b, 44, -92, Math.PI / 2, '#b23a3a');
  car(b, -66, -108, -Math.PI / 2, '#d9d4c8');
  car(b, 27, -70, 0, '#4b6a8f');
  // mailbox at the curb
  const mb = W(-14, -83, 0);
  b.add(M.woodSolid('#8a6a4a'), boxFt(0.35, 3.6, 0.35), T(mb.x, 1.8, mb.z));
  b.add(M.paint('#2a2a2a', 0.4, 0.4), boxFt(0.8, 0.8, 1.7), T(mb.x, 3.9, mb.z));
  b.add(M.paint('#d33', 0.4), boxFt(0.08, 0.6, 0.12), T(mb.x + 0.45, 4.1, mb.z - 0.4));

  // ── neighbours' yards
  shed(b, -64, 214, 0.15, '#b8452f');
  swingSet(b, 40, 206, 0.1);
  trampoline(b, -14, 228);
  shed(b, 182, 136, -1.3, '#6f8f8a');
  // their back fences (stained wood) between yards
  const woodFence = M.wood('#9b7653', 6, [3, 6]);
  const fenceLine = (x0: number, y0: number, x1: number, y1: number) => {
    const a = W(x0, y0, 0), c = W(x1, y1, 0);
    const L = a.distanceTo(c);
    const yaw = Math.atan2(-(c.z - a.z), c.x - a.x);
    b.add(woodFence, boxFt(L, 6, 0.15), T((a.x + c.x) / 2, 3, (a.z + c.z) / 2, yaw));
    const n = Math.ceil(L / 8);
    for (let i = 0; i <= n; i++) {
      const p = a.clone().lerp(c, i / n);
      b.add(M.woodSolid('#8a6648'), boxFt(0.4, 6.4, 0.4), T(p.x, 3.2, p.z, yaw));
    }
  };
  fenceLine(100, 100, 260, 160);
  fenceLine(56, 150, 110, 320);
  fenceLine(-112, 112, -260, 170);
  fenceLine(-40, 172, -90, 330);
  fenceLine(-96, 34, -280, 20);
  fenceLine(90, 40, 280, 10);

  // ── telephone poles and sagging wires along the alley behind the north houses
  const poleX = [-360, -240, -120, 0, 120, 240, 360];
  const poleY = 320;
  const wood = M.woodSolid('#6e5640', 6);
  const wires: [Vector3, Vector3][] = [];
  poleX.forEach((x, i) => {
    const p = W(x, poleY + (i % 2) * 3, 0);
    p.y = terrainHeight(p.x, p.z);
    // far beyond the shadow map: no shadow casting
    const far = { castShadow: false };
    b.add(wood, cyl(0.45, 0.6, 40, 8), T(p.x, p.y + 19, p.z), far);
    b.add(wood, boxFt(8, 0.5, 0.5), T(p.x, p.y + 35, p.z), far);
    b.add(wood, boxFt(5, 0.45, 0.45), T(p.x, p.y + 32, p.z), far);
    b.add(M.paint('#5a6066', 0.5, 0.5), cyl(0.9, 0.9, 2.4, 10), T(p.x + 1.2, p.y + 29, p.z + 0.8), far);
    if (i > 0) {
      const prev = W(poleX[i - 1], poleY + ((i - 1) % 2) * 3, 0);
      prev.y = terrainHeight(prev.x, prev.z);
      for (const [dx, y] of [[-3.6, 35.4], [3.6, 35.4], [-2.2, 32.4], [2.2, 32.4]]) {
        wires.push([new Vector3(prev.x + dx, prev.y + y, prev.z), new Vector3(p.x + dx, p.y + y, p.z)]);
      }
    }
  });
  for (const [a, c] of wires) {
    const pts: [number, number, number][] = [];
    for (let i = 0; i <= 12; i++) {
      const t = i / 12;
      const p = a.clone().lerp(c, t);
      pts.push([p.x, p.y - Math.sin(t * Math.PI) * 3.2, p.z]);
    }
    b.add(M.paint('#1a1a1a', 0.7), tube(pts, 0.06, 16, 3), undefined, { castShadow: false });
  }

  // ── water tower on the hill
  const wt = W(-520, 980, 0);
  wt.y = terrainHeight(wt.x, wt.z) - 2;
  const steel = M.paint('#c9d3d8', 0.45, 0.4);
  const noShadow = { castShadow: false };
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
    b.add(steel, cyl(0.9, 0.9, 150, 6), T(wt.x + Math.cos(a) * 14, wt.y + 75, wt.z + Math.sin(a) * 14), noShadow);
  }
  b.add(steel, cyl(4, 4, 150, 10), T(wt.x, wt.y + 75, wt.z), noShadow);
  const tank = M.tex('waterTower', paintTex(1024, 256, (g) => {
    g.fillStyle = '#d8e2e6'; g.fillRect(0, 0, 1024, 256);
    g.fillStyle = '#2f5d8a'; g.font = 'bold 110px "Trebuchet MS", sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('MAPLE HOLLOW', 256, 128);
    g.fillText('MAPLE HOLLOW', 768, 128);
  }, true), { roughness: 0.5, metalness: 0.2 });
  b.add(tank, cyl(30, 30, 34, 32), T(wt.x, wt.y + 167, wt.z), noShadow);
  b.add(steel, cyl(4, 31, 14, 32), T(wt.x, wt.y + 191, wt.z), noShadow);
  b.add(steel, cyl(31, 18, 10, 32), T(wt.x, wt.y + 145, wt.z), noShadow);

  group.add(b.build('neighborhood'));

  // ── a ring of far trees on the hills (one cheap mesh)
  const rnd = mulberry(2024);
  const spots: { x: number; z: number; s: number; kind?: 'round' | 'pine' }[] = [];
  for (let i = 0; i < 260; i++) {
    const a = rnd() * Math.PI * 2;
    const r = 380 + Math.pow(rnd(), 0.7) * 900;
    const x = Math.sin(a) * r, z = -Math.cos(a) * r - 60;
    // keep the street corridor a bit clearer
    if (Math.abs(z - 100) < 30 && r < 500) continue;
    spots.push({ x, z, s: 0.9 + rnd() * 0.7, kind: rnd() > 0.78 ? 'pine' : 'round' });
  }
  const far = buildFarTrees(spots);
  // sit the far trees on the hills
  group.add(far);
  return { group };
}

