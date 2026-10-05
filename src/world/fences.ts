import {
  BoxGeometry, BufferAttribute, Color, DoubleSide, Group, InstancedBufferAttribute, InstancedMesh, Matrix4, Mesh,
  MeshStandardMaterial, Object3D, PlaneGeometry, Vector3, type BufferGeometry, type Texture,
} from 'three';
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { Batch, T, boxFt, extrude } from '../gfx/build';
import { M } from '../gfx/materials';
import { hedgeTex, leafAtlas } from '../gfx/textures';
import { mulberry, Noise2 } from '../gfx/noise';
import type { Quality } from '../gfx/quality';
import type { FenceSeg } from '../data/types';

// Picket fences (instanced pickets on posts and rails) and clipped hedges
// (a lumpy displaced box with leafy cards on the surface for a soft edge).

function picketGeo(w: number, h: number, t: number): BufferGeometry {
  const g = extrude([[-w / 2, 0], [w / 2, 0], [w / 2, h - w * 0.55], [0, h], [-w / 2, h - w * 0.55]], t);
  const uv = g.attributes.uv, pos = g.attributes.position;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, pos.getX(i) + pos.getZ(i), pos.getY(i));
  return g;
}

/** Sim segment → three-space start, direction, length and yaw. */
function segFrame(s: FenceSeg) {
  const ax = s.a[0], az = -s.a[1], bx = s.b[0], bz = -s.b[1];
  const L = Math.hypot(bx - ax, bz - az);
  const ux = (bx - ax) / L, uz = (bz - az) / L;
  return { ax, az, L, ux, uz, yaw: Math.atan2(-uz, ux) };
}

/** A run of white picket fence along sim-space segments. */
export function buildPicketFence(segs: FenceSeg[], color = '#f3f0e8'): Group {
  const group = new Group();
  group.name = 'picketFence';
  const paint = M.woodSolid(color, [1.5, 4]);
  const postMat = M.paint(color, 0.6);
  const b = new Batch();
  const pickW = 0.3, gap = 0.17, pitch = pickW + gap;
  const rnd = mulberry(4242);
  const mats: Matrix4[] = [];
  const back = new Matrix4().makeTranslation(0, 0, 0.17);
  for (const s of segs) {
    const { ax, az, L, ux, uz, yaw } = segFrame(s);
    const h = s.height;
    const nPosts = Math.max(1, Math.round(L / 8));
    for (let i = 0; i <= nPosts; i++) {
      const t = (i / nPosts) * L;
      const px = ax + ux * t, pz = az + uz * t;
      b.add(postMat, boxFt(0.36, h + 0.35, 0.36), T(px, (h + 0.35) / 2, pz, yaw).multiply(back));
      b.add(postMat, boxFt(0.5, 0.12, 0.5), T(px, h + 0.41, pz, yaw).multiply(back));
      b.add(postMat, boxFt(0.26, 0.22, 0.26), T(px, h + 0.58, pz, yaw).multiply(back));
    }
    // two rails behind the pickets (the outer side)
    for (const ry of [0.9, h - 1.1]) b.add(paint, boxFt(L, 0.3, 0.14), T(ax + ux * L / 2, ry, az + uz * L / 2, yaw).multiply(back));
    const n = Math.floor(L / pitch);
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) * (L / n);
      mats.push(T(ax + ux * t, 0.12 + (rnd() - 0.5) * 0.05, az + uz * t, yaw + (rnd() - 0.5) * 0.02, 0, (rnd() - 0.5) * 0.025, [1, 1 + (rnd() - 0.5) * 0.03, 1]));
    }
  }
  group.add(b.build('picketPosts'));
  const inst = new InstancedMesh(picketGeo(pickW, segs[0]?.height ?? 4, 0.07), paint, mats.length);
  const col = new Color();
  mats.forEach((m, i) => {
    inst.setMatrixAt(i, m);
    const v = 0.88 + rnd() * 0.12;
    inst.setColorAt(i, col.setRGB(v, v, v * (0.96 + rnd() * 0.04)));
  });
  inst.castShadow = true;
  inst.receiveShadow = true;
  inst.name = 'pickets';
  group.add(inst);
  return group;
}

/** A box subdivided on every face, with shared vertices (so it can be displaced) and planar UVs in feet. */
function lumpBox(w: number, h: number, d: number, cell: number): BufferGeometry {
  const g = new BoxGeometry(w, h, d, Math.ceil(w / cell), Math.ceil(h / cell), Math.ceil(d / cell));
  g.deleteAttribute('uv');
  g.deleteAttribute('normal');
  const merged = mergeVertices(g, 1e-3);
  g.dispose();
  merged.computeVertexNormals();
  const pos = merged.attributes.position as BufferAttribute;
  const uv = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) { uv[i * 2] = pos.getX(i) + pos.getZ(i); uv[i * 2 + 1] = pos.getY(i) + pos.getZ(i) * 0.5; }
  merged.setAttribute('uv', new BufferAttribute(uv, 2));
  return merged;
}

/** A clipped hedge along sim-space segments. */
export function buildHedge(segs: FenceSeg[], q: Quality, thickness = 4.5): Group {
  const group = new Group();
  group.name = 'hedge';
  const noise = new Noise2(919);
  const ht = hedgeTex(q.texSize);
  const tile = (t: Texture) => { const c = t.clone(); c.repeat.set(1 / 3, 1 / 3); c.needsUpdate = true; return c; };
  const mat = new MeshStandardMaterial({ map: tile(ht.map), normalMap: tile(ht.normal!), roughness: 0.95, color: '#ffffff' });
  const leafMats: Matrix4[] = [];
  const rnd = mulberry(5150);
  const v = new Vector3(), n = new Vector3(), w = new Vector3();
  const hedgeGeos: BufferGeometry[] = [];
  for (const s of segs) {
    const { ax, az, L, ux, uz, yaw } = segFrame(s);
    const h = s.height;
    const len = L + thickness; // overlap at the corners
    const geo = lumpBox(len, h, thickness, 1.1);
    const m = T(ax + ux * L / 2, h / 2, az + uz * L / 2, yaw);
    const pos = geo.attributes.position as BufferAttribute;
    const nrm = geo.attributes.normal as BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      n.fromBufferAttribute(nrm, i);
      w.copy(v).applyMatrix4(m);
      const bulge = noise.fbm(w.x / 5, w.z / 5 + w.y / 4, 3) * 0.6 + noise.value(w.x * 0.9 + w.y, w.z * 0.9) * 0.2;
      if (v.y > -h / 2 + 0.01) v.addScaledVector(n, bulge);
      else v.y = -h / 2 - 0.05;
      // rounder top
      if (v.y > h / 2 - 0.2 && Math.abs(v.z) > thickness * 0.4) v.y -= 0.3;
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    hedgeGeos.push(geo.applyMatrix4(m));

    // leaf cards on the faces for a soft, leafy silhouette
    const cards = Math.round(L * (h * 2 + thickness) * 0.5 * (q.leaves / 1400));
    const o = new Object3D();
    for (let i = 0; i < cards; i++) {
      const f = rnd();
      const lx = (rnd() - 0.5) * len;
      let p: Vector3, out: Vector3;
      if (f < 0.4) { p = new Vector3(lx, rnd() * h - h / 2, thickness / 2); out = new Vector3(0, 0, 1); }
      else if (f < 0.8) { p = new Vector3(lx, rnd() * h - h / 2, -thickness / 2); out = new Vector3(0, 0, -1); }
      else { p = new Vector3(lx, h / 2 - 0.15, (rnd() - 0.5) * thickness); out = new Vector3(0, 1, 0); }
      w.copy(p).applyMatrix4(m);
      const bulge = noise.fbm(w.x / 5, w.z / 5 + w.y / 4, 3) * 0.6 + 0.1;
      out.transformDirection(m);
      o.position.copy(w).addScaledVector(out, bulge);
      o.rotation.set(rnd() * Math.PI, rnd() * Math.PI, rnd() * Math.PI);
      o.scale.setScalar(0.7 + rnd() * 0.6);
      o.updateMatrix();
      leafMats.push(o.matrix.clone());
    }
  }
  // every hedge run in one mesh
  if (hedgeGeos.length) {
    const mesh = new Mesh(mergeGeometries(hedgeGeos, false), mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.name = 'hedge';
    group.add(mesh);
    for (const g of hedgeGeos) g.dispose();
  }
  const leaves = leafCards(leafMats, leafAtlas(256, 98), '#c4dcae', 1.1);
  leaves.name = 'hedgeLeaves';
  group.add(leaves);
  return group;
}

export interface LeafOptions {
  /** wind sway amount (0 = still) */
  wind?: number;
  /** per-card world-space normals (3 floats each) so a clump shades as a soft volume */
  normals?: Float32Array;
}

/** Instanced alpha-tested leaf cards; each card shows one cell of the 2×2 leaf atlas. */
export function leafCards(mats: Matrix4[], atlas: Texture, tintHex = '#ffffff', size = 1.4, o: LeafOptions = {}): InstancedMesh {
  const g = new PlaneGeometry(size, size);
  const mat = new MeshStandardMaterial({ map: atlas, alphaTest: 0.45, side: DoubleSide, roughness: 0.85, color: tintHex });
  const inst = new InstancedMesh(g, mat, mats.length);
  const rnd = mulberry(mats.length + 7);
  const col = new Color();
  const cell = new Float32Array(mats.length * 2);
  mats.forEach((m, i) => {
    inst.setMatrixAt(i, m);
    const v = 0.78 + rnd() * 0.4;
    inst.setColorAt(i, col.setRGB(v, v * (0.97 + rnd() * 0.06), v * 0.95));
    const c = Math.floor(rnd() * 4);
    cell[i * 2] = (c % 2) * 0.5;
    cell[i * 2 + 1] = Math.floor(c / 2) * 0.5;
  });
  g.setAttribute('aCell', new InstancedBufferAttribute(cell, 2));
  const wind = o.wind ?? 0;
  if (o.normals) g.setAttribute('aN', new InstancedBufferAttribute(o.normals, 3));
  const uTime = { value: 0 };
  inst.userData.uTime = uTime;
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = uTime;
    let vs = sh.vertexShader
      .replace('#include <common>', `#include <common>
attribute vec2 aCell;
uniform float uTime;`)
      .replace('#include <uv_vertex>', '#include <uv_vertex>\nvMapUv = uv * 0.5 + aCell;');
    if (o.normals) {
      // volume normals: ignore the card's own orientation
      vs = vs
        .replace('#include <common>', '#include <common>\nattribute vec3 aN;')
        .replace('#include <defaultnormal_vertex>', 'vec3 transformedNormal = normalMatrix * aN;');
    }
    if (wind > 0) {
      vs = vs.replace('#include <begin_vertex>', `#include <begin_vertex>
vec4 iw0 = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
float sw = sin(uTime * 1.3 + iw0.x * 0.15 + iw0.y * 0.2) * 0.6 + sin(uTime * 2.7 + iw0.z * 0.3) * 0.25;
transformed += vec3(sw, sw * 0.3, sw * 0.6) * ${(wind * 0.08).toFixed(3)};`);
    }
    sh.vertexShader = vs;
    // both faces lit from the outside
    sh.fragmentShader = sh.fragmentShader.replace('#include <normal_fragment_begin>', 'vec3 normal = normalize(vNormal); vec3 nonPerturbedNormal = normal;');
  };
  mat.customProgramCacheKey = () => `leafCards${wind}${o.normals ? 'n' : ''}`;
  inst.castShadow = true;
  inst.receiveShadow = true;
  return inst;
}
