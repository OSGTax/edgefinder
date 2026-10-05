import {
  BoxGeometry, BufferGeometry, CylinderGeometry, Euler, Float32BufferAttribute, Group, Matrix4, Mesh, Quaternion,
  Shape, ExtrudeGeometry, SphereGeometry, TorusGeometry, Vector3, type Material, CatmullRomCurve3, TubeGeometry,
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

/** Collects transformed geometry per material, then merges it. */
export class Batch {
  private parts = new Map<Material, BufferGeometry[]>();
  private opts = new Map<Material, BatchOptions>();

  add(mat: Material, geo: BufferGeometry, m?: Matrix4, o?: BatchOptions): this {
    const g = normalize(geo);
    if (m) g.applyMatrix4(m);
    let list = this.parts.get(mat);
    if (!list) { list = []; this.parts.set(mat, list); }
    list.push(g);
    if (o) this.opts.set(mat, o);
    return this;
  }

  /** A view of this batch that places everything relative to `base`. */
  at(base: Matrix4): Adder {
    return { add: (mat, geo, m, o) => { this.add(mat, geo, m ? base.clone().multiply(m) : base, o); } };
  }

  build(name = 'batch', defaults: BatchOptions = { castShadow: true, receiveShadow: true }): Group {
    const group = new Group();
    group.name = name;
    for (const [mat, list] of this.parts) {
      // keep vertex colors only if every part has them
      const withColor = list.every((g) => g.attributes.color);
      if (!withColor) for (const g of list) if (g.attributes.color) g.deleteAttribute('color');
      const merged = mergeGeometries(list, false);
      if (!merged) continue;
      merged.computeBoundingSphere();
      const mesh = new Mesh(merged, mat);
      const o = { ...defaults, ...this.opts.get(mat) };
      mesh.castShadow = !!o.castShadow;
      mesh.receiveShadow = !!o.receiveShadow;
      mesh.name = `${name}:${mat.name || mat.type}`;
      group.add(mesh);
      for (const g of list) g.dispose();
    }
    this.parts.clear();
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
