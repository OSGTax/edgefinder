import {
  BoxGeometry, BufferGeometry, CylinderGeometry, Euler, Float32BufferAttribute, FrontSide, Group, Material, Matrix4, Mesh,
  MeshStandardMaterial, Quaternion, Shape, ExtrudeGeometry, SphereGeometry, TorusGeometry, Vector3, CatmullRomCurve3,
  TubeGeometry, type Side,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

// Static scenery is built from simple primitives that are merged into one
// mesh per material, so a yard full of detail costs only a few draw calls.

/** Matrix from a position (three space), Euler rotation and scale. */
export function T(x: number, y: number, z: number, ry = 0, rx = 0, rz = 0, s: number | [number, number, number] = 1): Matrix4 {
  const sc = typeof s === 'number' ? new Vector3(s, s, s) : new Vector3(...s);
  return new Matrix4().compose(new Vector3(x, y, z), new Quaternion().setFromEuler(new Euler(rx, ry, rz, 'YXZ')), sc);
}

/** Box whose UVs are in feet, so tiled textures keep their real-world size. */
export function boxFt(w: number, h: number, d: number): BufferGeometry {
  const g = new BoxGeometry(w, h, d);
  const uv = g.attributes.uv;
  const dims = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  for (let f = 0; f < 6; f++) for (let k = 0; k < 4; k++) {
    const i = f * 4 + k;
    uv.setXY(i, uv.getX(i) * dims[f][0], uv.getY(i) * dims[f][1]);
  }
  return g;
}

/** A box resting on y=0 (its base at the origin). */
export function boxUp(w: number, h: number, d: number): BufferGeometry {
  return boxFt(w, h, d).translate(0, h / 2, 0);
}

export function cyl(rTop: number, rBot: number, h: number, seg = 12, open = false): BufferGeometry {
  const g = new CylinderGeometry(rTop, rBot, h, seg, 1, open);
  const uv = g.attributes.uv;
  const circ = Math.PI * (rTop + rBot);
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * circ, uv.getY(i) * h);
  return g;
}

export function sphere(r: number, ws = 16, hs = 10): BufferGeometry {
  return new SphereGeometry(r, ws, hs);
}

export function torus(r: number, tube: number, rs = 10, ts = 24, arc = Math.PI * 2): BufferGeometry {
  return new TorusGeometry(r, tube, rs, ts, arc);
}

/** A tube along points (three space). */
export function tube(points: [number, number, number][], r: number, seg = 24, radial = 6, closed = false): BufferGeometry {
  const curve = new CatmullRomCurve3(points.map((p) => new Vector3(...p)), closed, 'catmullrom', 0.2);
  return new TubeGeometry(curve, seg, r, radial, closed);
}

/** Extrude a 2D outline (x right, y up) by `depth` along +z, centered on z. */
export function extrude(pts: [number, number][], depth: number, bevel = 0): BufferGeometry {
  const sh = new Shape(pts.map(([x, y]) => ({ x, y }) as any));
  const g = new ExtrudeGeometry(sh, { depth, bevelEnabled: bevel > 0, bevelSize: bevel, bevelThickness: bevel, bevelSegments: 1, steps: 1 });
  g.translate(0, 0, -depth / 2);
  return g;
}

/** Triangular prism (a gable): base width w along x, height h, length d along z. */
export function gable(w: number, h: number, d: number): BufferGeometry {
  const g = extrude([[-w / 2, 0], [w / 2, 0], [0, h]], d);
  // world-scale UVs on the sloped faces: use position for planar mapping
  const pos = g.attributes.position, uv = g.attributes.uv;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) + pos.getZ(i), pos.getY(i));
  return g;
}

/** A flat quad in the XY plane, w×h, facing +z, UVs 0..1 (or in feet). */
export function quad(w: number, h: number, feetUV = false): BufferGeometry {
  const g = new BufferGeometry();
  const x = w / 2, y = h / 2;
  g.setAttribute('position', new Float32BufferAttribute([-x, -y, 0, x, -y, 0, x, y, 0, -x, -y, 0, x, y, 0, -x, y, 0], 3));
  g.setAttribute('normal', new Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3));
  const u = feetUV ? w : 1, v = feetUV ? h : 1;
  g.setAttribute('uv', new Float32BufferAttribute([0, 0, u, 0, u, v, 0, 0, u, v, 0, v], 2));
  return g;
}

function normalize(g: BufferGeometry): BufferGeometry {
  let out = g.index ? g.toNonIndexed() : g;
  if (out === g) out = g.clone();
  for (const name of Object.keys(out.attributes)) {
    if (name !== 'position' && name !== 'normal' && name !== 'uv' && name !== 'color') out.deleteAttribute(name);
  }
  if (!out.attributes.normal) out.computeVertexNormals();
  if (!out.attributes.uv) out.setAttribute('uv', new Float32BufferAttribute(new Float32Array(out.attributes.position.count * 2), 2));
  out.morphAttributes = {};
  out.clearGroups();
  return out;
}

export interface BatchOptions { castShadow?: boolean; receiveShadow?: boolean }

/** Anything geometry can be added to (a Batch, or a Batch seen through a transform). */
export interface Adder { add(mat: Material, geo: BufferGeometry, m?: Matrix4, o?: BatchOptions): unknown }

/**
 * A plain painted surface: a MeshStandardMaterial that is nothing but a colour,
 * roughness and metalness. Batches fold all of these into one shared material
 * (colour + roughness/metalness per vertex), so dozens of paint colours cost a
 * single draw call instead of one each.
 */
function isPlain(m: Material): m is MeshStandardMaterial {
  if (!(m instanceof MeshStandardMaterial) || m.type !== 'MeshStandardMaterial') return false;
  return !m.map && !m.normalMap && !m.roughnessMap && !m.metalnessMap && !m.emissiveMap && !m.alphaMap && !m.aoMap
    && !m.lightMap && !m.bumpMap && !m.displacementMap && !m.envMap && !m.vertexColors && !m.transparent && m.opacity === 1
    && m.alphaTest === 0 && m.emissive.getHex() === 0 && m.envMapIntensity === 1 && !m.flatShading && !m.wireframe
    && m.depthWrite && m.depthTest && !m.polygonOffset && m.visible && !m.userData.noMerge
    && m.onBeforeCompile === Material.prototype.onBeforeCompile;
}

const plainMats = new Map<Side, MeshStandardMaterial>();
/** The shared vertex-coloured material that plain paints are merged into. */
function plainMaterial(side: Side): MeshStandardMaterial {
  let m = plainMats.get(side);
  if (m) return m;
  m = new MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0, side });
  m.name = 'batchPaint';
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec2 aRM;\nvarying vec2 vRM;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvRM = aRM;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vRM;')
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = vRM.x;')
      .replace('#include <metalnessmap_fragment>', 'float metalnessFactor = vRM.y;');
  };
  m.customProgramCacheKey = () => 'batchPaint';
  plainMats.set(side, m);
  return m;
}

/** Bake a plain material's look into per-vertex colour + roughness/metalness. */
function bakePlain(g: BufferGeometry, m: MeshStandardMaterial) {
  const n = g.attributes.position.count;
  const c = new Float32Array(n * 3), rm = new Float32Array(n * 2);
  const { r, g: gr, b } = m.color;
  for (let i = 0; i < n; i++) {
    c[i * 3] = r; c[i * 3 + 1] = gr; c[i * 3 + 2] = b;
    rm[i * 2] = m.roughness; rm[i * 2 + 1] = m.metalness;
  }
  g.setAttribute('color', new Float32BufferAttribute(c, 3));
  g.setAttribute('aRM', new Float32BufferAttribute(rm, 2));
}

interface Bucket { mat: Material; list: BufferGeometry[]; opts?: BatchOptions }

/** Collects transformed geometry per material, then merges it. */
export class Batch {
  private buckets = new Map<string, Bucket>();

  add(mat: Material, geo: BufferGeometry, m?: Matrix4, o?: BatchOptions): this {
    const g = normalize(geo);
    if (m) g.applyMatrix4(m);
    let key = mat.uuid, target = mat;
    if (isPlain(mat)) {
      bakePlain(g, mat);
      target = plainMaterial(mat.side ?? FrontSide);
      key = `plain${mat.side}|${o?.castShadow ?? '-'}|${o?.receiveShadow ?? '-'}`;
    }
    let bucket = this.buckets.get(key);
    if (!bucket) { bucket = { mat: target, list: [] }; this.buckets.set(key, bucket); }
    bucket.list.push(g);
    if (o) bucket.opts = o;
    return this;
  }

  /** A view of this batch that places everything relative to `base`. */
  at(base: Matrix4): Adder {
    return { add: (mat, geo, m, o) => { this.add(mat, geo, m ? base.clone().multiply(m) : base, o); } };
  }

  build(name = 'batch', defaults: BatchOptions = { castShadow: true, receiveShadow: true }): Group {
    const group = new Group();
    group.name = name;
    for (const { mat, list, opts } of this.buckets.values()) {
      // keep vertex colors only if every part has them
      const withColor = list.every((g) => g.attributes.color);
      if (!withColor) for (const g of list) if (g.attributes.color) g.deleteAttribute('color');
      const merged = mergeGeometries(list, false);
      if (!merged) continue;
      merged.computeBoundingSphere();
      const mesh = new Mesh(merged, mat);
      const o = { ...defaults, ...opts };
      mesh.castShadow = !!o.castShadow;
      mesh.receiveShadow = !!o.receiveShadow;
      mesh.name = `${name}:${mat.name || mat.type}`;
      group.add(mesh);
      for (const g of list) g.dispose();
    }
    this.buckets.clear();
    return group;
  }
}

/** Paint a geometry's vertices one color (for vertex-colored batches). */
export function tint(g: BufferGeometry, r: number, gr: number, b: number): BufferGeometry {
  const n = g.attributes.position.count;
  const c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { c[i * 3] = r; c[i * 3 + 1] = gr; c[i * 3 + 2] = b; }
  g.setAttribute('color', new Float32BufferAttribute(c, 3));
  return g;
}

/**
 * Collapse a small rigid group of meshes (e.g. a glove) into one mesh per
 * material, in place. Looks identical; costs far fewer draw calls.
 */
export function mergeByMaterial(group: Group): Group {
  group.updateMatrix();
  const lists = new Map<Material, { geos: BufferGeometry[]; cast: boolean; receive: boolean }>();
  const meshes = group.children.filter((c): c is Mesh => (c as Mesh).isMesh && c.children.length === 0 && !Array.isArray((c as Mesh).material));
  for (const m of meshes) {
    m.updateMatrix();
    const mat = m.material as Material;
    let l = lists.get(mat);
    if (!l) { l = { geos: [], cast: false, receive: false }; lists.set(mat, l); }
    const g = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry.clone();
    l.geos.push(g.applyMatrix4(m.matrix));
    l.cast ||= m.castShadow;
    l.receive ||= m.receiveShadow;
  }
  for (const [mat, l] of lists) {
    if (l.geos.length < 2) continue;
    const merged = mergeGeometries(l.geos, false);
    if (!merged) continue;
    for (const m of meshes) if (m.material === mat) { group.remove(m); m.geometry.dispose(); }
    const mesh = new Mesh(merged, mat);
    mesh.castShadow = l.cast;
    mesh.receiveShadow = l.receive;
    group.add(mesh);
    for (const g of l.geos) g.dispose();
  }
  return group;
}
