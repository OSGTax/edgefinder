import { BufferGeometry, Color, CylinderGeometry, Float32BufferAttribute, Matrix4, Quaternion, SphereGeometry, TorusGeometry, Uint16BufferAttribute, Vector3 } from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

// Geometry helpers for the procedural kids: lofted rings (torsos, sleeves),
// rounded limbs, and rigid / blended skin weights.

// ── detail level: kids are built at full detail for close-ups, and again with about a third
// of the triangles for kids far from the camera (KidModel.setDetail). Every round part asks
// segs() for its segment counts, and tiny parts (fingers, laces...) check isLite().
let detail = 1;
/** Build something at a detail level (1 = full). */
export function withDetail<T>(d: number, fn: () => T): T {
  const old = detail;
  detail = d;
  try { return fn(); } finally { detail = old; }
}
export const isLite = () => detail < 0.7;
/** A segment count scaled by the current detail level. */
export const segs = (n: number, min = 3) => Math.max(min, Math.round(n * detail));
export const sphere = (r: number, ws: number, hs: number, phiStart?: number, phiLength?: number, thetaStart?: number, thetaLength?: number) =>
  new SphereGeometry(r, segs(ws, 5), segs(hs, 3), phiStart, phiLength, thetaStart, thetaLength);
export const torus = (r: number, tube: number, radial: number, tubular: number, arc?: number) =>
  new TorusGeometry(r, tube, segs(radial, 3), segs(tubular, 4), arc);
export const cylinder = (rTop: number, rBottom: number, h: number, radial: number, hSeg = 1, open = false) =>
  new CylinderGeometry(rTop, rBottom, h, segs(radial, 5), hSeg, open);

export interface Ring { y: number; rx: number; rz: number; cz?: number; cx?: number }

/** Loft a tube through horizontal elliptical rings (bottom to top). UV: u around (0.5 = front), v up. */
export function loft(rings: Ring[], seg = 20, capBottom = false, capTop = false): BufferGeometry {
  seg = segs(seg, 4);
  const pos: number[] = [], nor: number[] = [], uv: number[] = [], idx: number[] = [];
  const y0 = rings[0].y, y1 = rings[rings.length - 1].y;
  for (let r = 0; r < rings.length; r++) {
    const R = rings[r];
    for (let i = 0; i <= seg; i++) {
      const u = i / seg;
      const a = (u - 0.5) * Math.PI * 2; // 0 = front (+z)
      const x = Math.sin(a) * R.rx + (R.cx ?? 0), z = Math.cos(a) * R.rz + (R.cz ?? 0);
      pos.push(x, R.y, z);
      // normal from the ellipse plus the slope between neighbouring rings
      const prev = rings[Math.max(0, r - 1)], next = rings[Math.min(rings.length - 1, r + 1)];
      const dy = next.y - prev.y || 1;
      const drx = (next.rx - prev.rx) / dy, drz = (next.rz - prev.rz) / dy;
      const n = new Vector3(Math.sin(a) / R.rx, 0, Math.cos(a) / R.rz).normalize();
      n.y = -(Math.abs(Math.sin(a)) * drx + Math.abs(Math.cos(a)) * drz) * 0.9;
      n.normalize();
      nor.push(n.x, n.y, n.z);
      uv.push(u, (R.y - y0) / (y1 - y0 || 1));
    }
  }
  for (let r = 0; r < rings.length - 1; r++) for (let i = 0; i < seg; i++) {
    const a = r * (seg + 1) + i, b = a + seg + 1;
    idx.push(a, a + 1, b, b, a + 1, b + 1);
  }
  const cap = (ri: number, up: boolean) => {
    const R = rings[ri];
    const c = pos.length / 3;
    pos.push(R.cx ?? 0, R.y, R.cz ?? 0);
    nor.push(0, up ? 1 : -1, 0);
    uv.push(0.5, up ? 1 : 0);
    for (let i = 0; i < seg; i++) {
      const a = ri * (seg + 1) + i;
      if (up) idx.push(c, a, a + 1); else idx.push(c, a + 1, a);
    }
  };
  if (capBottom) cap(0, false);
  if (capTop) cap(rings.length - 1, true);
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new Float32BufferAttribute(nor, 3));
  g.setAttribute('uv', new Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
}

/** Rings for a rounded capsule-like limb of length L from r0 (top) to r1 (bottom), hanging down -y from 0. */
export function limbRings(L: number, r0: number, r1: number, steps = 8, squashZ = 1): Ring[] {
  const rings: Ring[] = [];
  const capSteps = detail < 0.7 ? 2 : 3;
  steps = Math.max(1, Math.round(steps * Math.min(1, detail * 1.2)));
  // top cap
  for (let i = 0; i <= capSteps; i++) {
    const t = i / capSteps;
    const a = (1 - t) * Math.PI / 2;
    rings.push({ y: Math.sin(a) * r0 * 0.9, rx: Math.max(0.001, Math.cos(a) * r0), rz: Math.max(0.001, Math.cos(a) * r0 * squashZ) });
  }
  for (let i = 1; i < steps; i++) {
    const t = i / steps;
    const r = r0 + (r1 - r0) * t;
    rings.push({ y: -L * t, rx: r, rz: r * squashZ });
  }
  for (let i = 0; i <= capSteps; i++) {
    const t = i / capSteps;
    const a = t * Math.PI / 2;
    rings.push({ y: -L - Math.sin(a) * r1 * 0.9, rx: Math.max(0.001, Math.cos(a) * r1), rz: Math.max(0.001, Math.cos(a) * r1 * squashZ) });
  }
  return rings.reverse();
}

export function limb(L: number, r0: number, r1: number, seg = 12, squashZ = 1): BufferGeometry {
  return loft(limbRings(L, r0, r1, 6, squashZ), seg);
}

/** Matrix placing a -y-hanging part so it runs from a to b. */
export function alongMatrix(a: Vector3, b: Vector3): Matrix4 {
  const dir = b.clone().sub(a).normalize();
  const q = new Quaternion().setFromUnitVectors(new Vector3(0, -1, 0), dir);
  return new Matrix4().compose(a, q, new Vector3(1, 1, 1));
}

/** Rigid skin: every vertex follows one bone. */
export function rigid(g: BufferGeometry, bone: number): BufferGeometry {
  const n = g.attributes.position.count;
  const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) { si[i * 4] = bone; sw[i * 4] = 1; }
  g.setAttribute('skinIndex', new Uint16BufferAttribute(si, 4));
  g.setAttribute('skinWeight', new Float32BufferAttribute(sw, 4));
  return g;
}

/** Blended skin: a function returns up to two (bone, weight) pairs per vertex position. */
export function blended(g: BufferGeometry, fn: (p: Vector3) => [number, number, number, number]): BufferGeometry {
  const n = g.attributes.position.count;
  const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
  const p = new Vector3();
  for (let i = 0; i < n; i++) {
    p.fromBufferAttribute(g.attributes.position, i);
    const [a, wa, b, wb] = fn(p);
    si[i * 4] = a; sw[i * 4] = wa; si[i * 4 + 1] = b; sw[i * 4 + 1] = wb;
  }
  g.setAttribute('skinIndex', new Uint16BufferAttribute(si, 4));
  g.setAttribute('skinWeight', new Float32BufferAttribute(sw, 4));
  return g;
}

/** Weight that ramps 0→1 between y0 and y1 (smoothstep). */
export const ramp = (y: number, y0: number, y1: number) => {
  const t = Math.min(1, Math.max(0, (y - y0) / (y1 - y0)));
  return t * t * (3 - 2 * t);
};

export function paint(g: BufferGeometry, hex: string | Color): BufferGeometry {
  const c = typeof hex === 'string' ? new Color(hex) : hex;
  const n = g.attributes.position.count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b; }
  g.setAttribute('color', new Float32BufferAttribute(arr, 3));
  return g;
}

/** Paint per vertex from a function of position. */
export function paintFn(g: BufferGeometry, fn: (p: Vector3) => Color): BufferGeometry {
  const n = g.attributes.position.count;
  const arr = new Float32Array(n * 3);
  const p = new Vector3();
  for (let i = 0; i < n; i++) {
    p.fromBufferAttribute(g.attributes.position, i);
    const c = fn(p);
    arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b;
  }
  g.setAttribute('color', new Float32BufferAttribute(arr, 3));
  return g;
}

/** Collects skinned parts for one material and merges them. */
export class PartList {
  parts: BufferGeometry[] = [];
  /** parts added while this is set stay out of the ink outline (small or thin costume pieces) */
  noInk = false;
  add(g: BufferGeometry, m?: Matrix4): BufferGeometry {
    let out = g.index ? g.toNonIndexed() : g;
    if (out === g) out = g.clone();
    if (m) out.applyMatrix4(m);
    for (const name of Object.keys(out.attributes)) {
      if (!['position', 'normal', 'uv', 'color', 'skinIndex', 'skinWeight'].includes(name)) out.deleteAttribute(name);
    }
    if (!out.attributes.uv) out.setAttribute('uv', new Float32BufferAttribute(new Float32Array(out.attributes.position.count * 2), 2));
    if (this.noInk) out.userData.noInk = true;
    this.parts.push(out);
    return out;
  }
  /** Shapes for the ink outline (position, normal and skinning only), before merge() consumes the parts. */
  inkParts(): BufferGeometry[] {
    return this.parts.filter((p) => !p.userData.noInk).map((p) => {
      const g = new BufferGeometry();
      for (const a of ['position', 'normal', 'skinIndex', 'skinWeight']) g.setAttribute(a, p.getAttribute(a).clone());
      return g;
    });
  }
  merge(withColor: boolean): BufferGeometry | null {
    if (!this.parts.length) return null;
    for (const p of this.parts) {
      if (withColor && !p.attributes.color) paint(p, '#ffffff');
      if (!withColor && p.attributes.color) p.deleteAttribute('color');
      if (!p.attributes.skinIndex) throw new Error('unskinned part');
    }
    const g = mergeGeometries(this.parts, false);
    for (const p of this.parts) p.dispose();
    this.parts = [];
    g?.computeBoundingSphere();
    return g;
  }
}
