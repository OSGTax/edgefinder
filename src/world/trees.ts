import {
  Color, CylinderGeometry, Group, IcosahedronGeometry, Matrix4, Mesh, MeshLambertMaterial, Object3D, Quaternion, Vector3,
  Float32BufferAttribute, type BufferGeometry, type InstancedMesh,
} from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { Batch } from '../gfx/build';
import { M } from '../gfx/materials';
import { leafAtlas } from '../gfx/textures';
import { mulberry, Noise2 } from '../gfx/noise';
import type { Quality } from '../gfx/quality';
import { leafCards } from './fences';
import { terrainHeight } from '../gfx/ground';

// Procedural trees. Broadleaf trees are a crown of overlapping leafy clumps
// (leaf cards biased to each clump's surface, shaded with soft "volume"
// normals) held up by tapered limbs from a forked trunk. Pines are a cone of
// drooping needle tiers. Far trees are cheap lumpy blobs merged into one mesh.

export type TreeKind = 'oak' | 'maple' | 'pine' | 'birch';

export interface TreeSpec {
  kind: TreeKind;
  x: number; z: number;
  /** crown centre height, horizontal and vertical radius (ft) */
  crownY?: number; crownR?: number; crownRv?: number;
  /** pines: total height */
  height?: number;
  seed?: number;
}

const KIND = {
  oak: { y: 26, r: 18, rv: 12, hue: 96, tint: '#b9d49a', card: 2.0, clumps: 9, density: 1 },
  maple: { y: 26, r: 14, rv: 13, hue: 90, tint: '#c3dc93', card: 1.8, clumps: 8, density: 1.05 },
  birch: { y: 24, r: 9, rv: 11, hue: 76, tint: '#d4e6a0', card: 1.4, clumps: 6, density: 0.85 },
  pine: { y: 0, r: 0, rv: 0, hue: 122, tint: '#93ad84', card: 2.4, clumps: 0, density: 1 },
};

/** Tapered cylinder between two points with UVs in feet. */
function limb(a: Vector3, b: Vector3, r0: number, r1: number, seg: number) {
  const L = a.distanceTo(b);
  const g = new CylinderGeometry(r1, r0, L, seg, 1, true);
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * Math.PI * (r0 + r1), uv.getY(i) * L);
  const dir = b.clone().sub(a).normalize();
  const m = new Matrix4().compose(a.clone().add(b).multiplyScalar(0.5), new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), dir), new Vector3(1, 1, 1));
  return { g, m };
}

export interface Trees { group: Group; leaves: InstancedMesh[] }

export function buildTrees(specs: TreeSpec[], q: Quality): Trees {
  const group = new Group();
  group.name = 'trees';
  const bark = new Batch();
  const barkMat = M.bark();
  const birchBark = M.paint('#efeae2', 0.8);
  const leaves: InstancedMesh[] = [];
  const o = new Object3D();
  const budget = q.leaves / 1400;
  // leaf cards are pooled per kind (one instanced mesh, one draw call each);
  // trees far outside the yard skip the shadow pass — their shadows never reach it
  const pools = new Map<string, { kind: TreeKind; near: boolean; mats: Matrix4[]; nrm: number[] }>();
  for (const spec of specs) {
    const rnd = mulberry(spec.seed ?? Math.round(Math.abs(spec.x * 13 + spec.z * 7)) + 1);
    const k = KIND[spec.kind];
    const near = Math.hypot(spec.x, spec.z + 70) < 200;
    // trees behind the house get their own pool so the game cameras can cull them
    const key = `${spec.kind}${near}${spec.z > 30}`;
    let pool = pools.get(key);
    if (!pool) { pool = { kind: spec.kind, near, mats: [], nrm: [] }; pools.set(key, pool); }
    const mats = pool.mats, nrm = pool.nrm, first = mats.length;
    if (spec.kind === 'pine') {
      const H = spec.height ?? 40;
      const { g, m } = limb(new Vector3(spec.x, 0, spec.z), new Vector3(spec.x, H, spec.z), H * 0.025, 0.12, 9);
      bark.add(barkMat, g, m);
      const tiers = Math.round(H / 1.6);
      for (let t = 0; t < tiers; t++) {
        const f = t / tiers;
        const y = H * (0.15 + f * 0.83);
        const rad = (1 - f) * H * 0.2 + 1;
        // a few bare branches per tier
        for (let bI = 0; bI < 3; bI++) {
          const a = rnd() * Math.PI * 2;
          const tip = new Vector3(spec.x + Math.cos(a) * rad * 0.9, y - rad * 0.25, spec.z + Math.sin(a) * rad * 0.9);
          const lb = limb(new Vector3(spec.x, y + 0.3, spec.z), tip, 0.18, 0.04, 4);
          bark.add(barkMat, lb.g, lb.m);
        }
        const n = Math.round(rad * 7 * budget);
        for (let i = 0; i < n; i++) {
          const a = rnd() * Math.PI * 2, rr = (0.35 + Math.sqrt(rnd()) * 0.65) * rad;
          o.position.set(spec.x + Math.cos(a) * rr, y - rr * 0.3 + rnd() * 0.5, spec.z + Math.sin(a) * rr);
          o.rotation.set(-Math.PI / 2 + 0.5 + (rnd() - 0.5) * 0.6, -a + Math.PI / 2, (rnd() - 0.5) * 0.5, 'YXZ');
          o.scale.setScalar(1.5 + rnd() * 0.9);
          o.updateMatrix();
          mats.push(o.matrix.clone());
          const nv = new Vector3(Math.cos(a), 0.6 + (y / H) * 0.4, Math.sin(a)).normalize();
          nrm.push(nv.x, nv.y, nv.z);
        }
      }
      continue;
    }

    const cy = spec.crownY ?? k.y, R = spec.crownR ?? k.r, Rv = spec.crownRv ?? k.rv;
    const crown = new Vector3(spec.x, cy, spec.z);
    const fork = new Vector3(spec.x + (rnd() - 0.5) * 1.5, cy - Rv * 0.75, spec.z + (rnd() - 0.5) * 1.5);
    const trunkR = spec.kind === 'birch' ? 0.55 + R * 0.02 : 0.8 + R * 0.055;
    const bm = spec.kind === 'birch' ? birchBark : barkMat;
    // trunk with a root flare
    const base = new Vector3(spec.x, -0.2, spec.z);
    const t1 = limb(base, base.clone().lerp(fork, 0.12), trunkR * 1.5, trunkR * 1.05, 12);
    bark.add(bm, t1.g, t1.m);
    const t2 = limb(base.clone().lerp(fork, 0.12), fork, trunkR * 1.05, trunkR * 0.8, 12);
    bark.add(bm, t2.g, t2.m);
    if (spec.kind === 'birch') {
      for (let i = 0; i < 16; i++) {
        const p = base.clone().lerp(fork, 0.1 + rnd() * 0.85);
        bark.add(M.paint('#2a2724', 0.8), new CylinderGeometry(trunkR * 1.02, trunkR * 1.02, 0.12 + rnd() * 0.2, 12, 1, true, rnd() * 6, 0.8 + rnd()), new Matrix4().makeTranslation(p.x, p.y, p.z));
      }
    }
    // clumps arranged over the crown ellipsoid
    const clumps: { c: Vector3; r: number }[] = [{ c: crown.clone().add(new Vector3(0, Rv * 0.3, 0)), r: Math.min(R, Rv) * 0.62 }];
    for (let i = 0; i < k.clumps; i++) {
      const a = (i / k.clumps) * Math.PI * 2 + rnd() * 0.6;
      const up = (rnd() - 0.35) * 0.9;
      const dir = new Vector3(Math.cos(a) * Math.cos(up), Math.sin(up), Math.sin(a) * Math.cos(up));
      clumps.push({ c: crown.clone().add(new Vector3(dir.x * R * 0.55, dir.y * Rv * 0.55, dir.z * R * 0.55)), r: Math.min(R, Rv) * (0.45 + rnd() * 0.15) });
    }
    // limbs from the fork to every clump, bending up through a midpoint, plus twigs
    for (const cl of clumps) {
      const mid = fork.clone().lerp(cl.c, 0.5).add(new Vector3((rnd() - 0.5) * 2, 1.5, (rnd() - 0.5) * 2));
      const l1 = limb(fork, mid, trunkR * 0.55, trunkR * 0.38, 7);
      bark.add(bm, l1.g, l1.m);
      const l2 = limb(mid, cl.c, trunkR * 0.38, trunkR * 0.15, 6);
      bark.add(bm, l2.g, l2.m);
      for (let tw = 0; tw < 3; tw++) {
        const tip = cl.c.clone().add(new Vector3(rnd() - 0.5, rnd() * 0.6, rnd() - 0.5).normalize().multiplyScalar(cl.r * 0.8));
        const l3 = limb(cl.c, tip, trunkR * 0.14, 0.03, 4);
        bark.add(bm, l3.g, l3.m);
      }
    }
    // leaf cards: biased to each clump's outer shell, skipping the hidden core
    const area = clumps.reduce((s, c) => s + c.r * c.r, 0);
    const total = Math.round(area * 4.5 * budget * k.density);
    const u = new Vector3(), p = new Vector3(), n1 = new Vector3(), n2 = new Vector3();
    let guard = 0;
    while (mats.length - first < total && guard++ < total * 6) {
      const cl = clumps[Math.floor(rnd() * clumps.length)];
      u.set(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1);
      const l = u.length();
      if (l > 1 || l < 0.05) continue;
      u.divideScalar(l);
      const rr = cl.r * (0.72 + 0.28 * Math.pow(rnd(), 0.5));
      p.copy(cl.c).addScaledVector(u, rr);
      // hidden inside the crown? skip
      const dx = (p.x - crown.x) / R, dy = (p.y - crown.y) / Rv, dz = (p.z - crown.z) / R;
      if (dx * dx + dy * dy + dz * dz < 0.3) continue;
      if (p.y < cy - Rv * 0.95) continue;
      o.position.copy(p);
      o.rotation.set(rnd() * Math.PI, rnd() * Math.PI, rnd() * Math.PI);
      o.scale.setScalar(0.85 + rnd() * 0.5);
      o.updateMatrix();
      mats.push(o.matrix.clone());
      n1.set(dx, dy * 0.8, dz).normalize();
      n2.copy(u);
      n1.addScaledVector(n2, 0.8).add(new Vector3(0, 0.25, 0)).normalize();
      nrm.push(n1.x, n1.y, n1.z);
    }
  }
  for (const p of pools.values()) {
    const k = KIND[p.kind];
    const inst = leafCards(p.mats, leafAtlas(256, k.hue), k.tint, k.card, { wind: p.kind === 'pine' ? 0.4 : 1, normals: new Float32Array(p.nrm) });
    inst.name = p.kind === 'pine' ? 'pineNeedles' : `${p.kind}Leaves`;
    inst.castShadow = p.near;
    group.add(inst);
    leaves.push(inst);
  }
  group.add(bark.build('treeBark'));
  return { group, leaves };
}

export interface FarTreeSpot {
  x: number; z: number; s: number; kind?: 'round' | 'pine';
  /** canopy hue (0..1) and lightness; random when left out */
  hue?: number; light?: number;
}

/** Distant trees: lumpy vertex-coloured blobs merged into a single cheap mesh. */
export function buildFarTrees(spots: FarTreeSpot[], seed = 1): Mesh {
  const rnd = mulberry(seed);
  const noise = new Noise2(seed + 3);
  const b = new Batch();
  const mat = new MeshLambertMaterial({ vertexColors: true });
  mat.name = 'farTrees';
  // one welded unit icosahedron, scaled per blob (welding is the slow part)
  const unit = new IcosahedronGeometry(1, 1);
  unit.deleteAttribute('uv');
  unit.deleteAttribute('normal');
  const blob = mergeVertices(unit);
  unit.dispose();
  const col = new Color();
  for (const t of spots) {
    const hue = t.hue ?? 0.23 + rnd() * 0.07, light = t.light ?? 0.27 + rnd() * 0.1;
    const gy = terrainHeight(t.x, t.z) - 1;
    if (t.kind === 'pine') {
      const h = 40 * t.s;
      for (let k = 0; k < 3; k++) {
        const g = new CylinderGeometry(0.2, 8 * t.s * (1 - k * 0.22), h * 0.45, 8, 1);
        paintVerts(g, col.setHSL(0.3 + rnd() * 0.04, 0.33, 0.2 + k * 0.02), 0.12, rnd);
        b.add(mat, g, new Matrix4().makeTranslation(t.x, gy + h * 0.3 + k * h * 0.2, t.z));
      }
      continue;
    }
    const trunk = new CylinderGeometry(0.7 * t.s, 1.1 * t.s, 14 * t.s, 6);
    paintVerts(trunk, col.setHSL(0.07, 0.25, 0.2), 0, rnd);
    b.add(mat, trunk, new Matrix4().makeTranslation(t.x, gy + 7 * t.s, t.z));
    const blobs = 3 + Math.floor(rnd() * 3);
    for (let k = 0; k < blobs; k++) {
      const r = (9 + rnd() * 6) * t.s;
      const g = blob.clone().scale(r, r, r);
      const pos = g.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        const d = 1 + noise.value(x * 0.3 + t.x, y * 0.3 + z * 0.3 + t.z) * 0.3;
        pos.setXYZ(i, x * d, y * d * 0.85, z * d);
      }
      g.computeVertexNormals();
      paintVerts(g, col.setHSL(hue, 0.38, light), 0.08, rnd, true);
      const a = rnd() * Math.PI * 2, off = k === 0 ? 0 : r * 0.6;
      b.add(mat, g, new Matrix4().makeTranslation(t.x + Math.cos(a) * off, gy + (22 + rnd() * 6) * t.s + (k ? rnd() * 5 : 4), t.z + Math.sin(a) * off));
    }
  }
  blob.dispose();
  const g = b.build('farTrees', { castShadow: false, receiveShadow: false });
  return g.children[0] as Mesh;
}

function paintVerts(g: BufferGeometry, c: Color, jitter: number, rnd: () => number, topLight = false) {
  const n = g.attributes.position.count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const j = 1 + (rnd() - 0.5) * jitter * 2;
    const up = topLight ? 0.75 + Math.max(0, g.attributes.normal.getY(i)) * 0.4 : 1;
    arr[i * 3] = c.r * j * up; arr[i * 3 + 1] = c.g * j * up; arr[i * 3 + 2] = c.b * j * up;
  }
  g.setAttribute('color', new Float32BufferAttribute(arr, 3));
}
