import { BufferGeometry, Color, CylinderGeometry, Euler, Float32BufferAttribute, Matrix4, Quaternion, SphereGeometry, TorusGeometry, Vector3 } from 'three';
import type { Kid, KidLook, Team } from '../data/types';
import { alongMatrix, blended, limb, loft, paint, paintFn, ramp, rigid } from './geom';
import { B, HEAD_SHAPE, type Proportions } from './rig';
import { CAP_LOGO_UV, type UniformColors } from './uniform';
import type { Lists } from './model';
import { headCentre, seeded, shapeHead } from './model';

// Hair styles, hats and the grown-up costume pieces that make each kid a
// mini adult. Everything is built around the head centre (head bone) or on
// the torso (chest/spine/hips bones).

const M4 = (x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, s: number | [number, number, number] = 1) =>
  new Matrix4().compose(new Vector3(x, y, z), new Quaternion().setFromEuler(new Euler(rx, ry, rz, 'YXZ')), typeof s === 'number' ? new Vector3(s, s, s) : new Vector3(...s));

/** Sphere cap around +y of angular size T, tilted back by alpha (front hairline higher than the back). */
function cap(r: number, T: number, alpha: number, ws = 36, hs = 18): BufferGeometry {
  const g = new SphereGeometry(r, ws, hs, 0, Math.PI * 2, 0, T);
  g.applyMatrix4(new Matrix4().makeRotationX(-alpha));
  return g;
}

/**
 * A scalp of hair with a natural hairline: high on the forehead, down at the temples into
 * short sideburns, up and around the ears, and down to the nape. (Angles are polar angles
 * from the top of the head, by azimuth from the front.)
 */
const HAIRLINE: [number, number][] = [[0, 0.78], [0.55, 0.92], [1.0, 1.42], [1.18, 1.68], [1.32, 1.62], [1.45, 1.28], [1.8, 1.32], [2.3, 1.85], [Math.PI, 2.05]];
function hairCap(r: number, front = 0.78): BufferGeometry {
  // (stop just short of the bottom pole so the last row is a clean edge, not a fan)
  const g = new SphereGeometry(r, 44, 16, 0, Math.PI * 2, 0, Math.PI * 0.999);
  const pos = g.attributes.position, uv = g.attributes.uv;
  const edgeAt = (a: number) => {
    const t = Math.abs(a);
    for (let i = 1; i < HAIRLINE.length; i++) {
      const [a1, e1] = HAIRLINE[i], [a0, e0] = HAIRLINE[i - 1];
      if (t <= a1) return (i === 1 ? front : e0) + ((i === 1 ? e1 : e1) - (i === 1 ? front : e0)) * ramp(t, a0, a1);
    }
    return HAIRLINE[HAIRLINE.length - 1][1];
  };
  for (let i = 0; i < pos.count; i++) {
    let a = uv.getX(i) * Math.PI * 2 - Math.PI / 2;
    if (a > Math.PI) a -= Math.PI * 2;
    const th = Math.acos(Math.max(-1, Math.min(1, pos.getY(i) / r))) / (Math.PI * 0.999) * edgeAt(a);
    pos.setXYZ(i, r * Math.sin(th) * Math.sin(a), r * Math.cos(th), r * Math.sin(th) * Math.cos(a));
  }
  g.computeVertexNormals();
  return g;
}

/** A band lying on a sphere of radius r around the head centre, between heights y0 ± h (a sphere zone). */
function hugBand(r: number, y0: number, h: number, seg = 32): BufferGeometry {
  const ring = (y: number) => ({ y, rx: Math.sqrt(Math.max(0, r * r - y * y)), rz: Math.sqrt(Math.max(0, r * r - y * y)) });
  return loft([ring(y0 - h), ring(y0), ring(y0 + h)], seg);
}

/** Lumpy displacement so hair masses aren't perfect spheres. */
function lumpy(g: BufferGeometry, amt: number, freq: number, seed: number): BufferGeometry {
  const pos = g.attributes.position;
  const v = new Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n = Math.sin(v.x * freq + seed) * Math.sin(v.y * freq * 1.3 + seed * 2) * Math.sin(v.z * freq * 0.9 + seed * 3);
    v.multiplyScalar(1 + n * amt);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}


/** Head-shape scale about the head centre, so head-hugging pieces fit wide/oval/square skulls. */
export function headShapeMatrix(p: Proportions, look: KidLook): Matrix4 {
  const [sx, sy, sz] = HEAD_SHAPE[look.head] ?? HEAD_SHAPE.round;
  const hc = headCentre(p);
  // square heads are a touch wider again over the temples
  const kx = look.head === 'square' ? 1.03 : 1;
  return new Matrix4().makeTranslation(hc.x, hc.y, hc.z)
    .multiply(new Matrix4().makeScale(sx * kx, sy, sz))
    .multiply(new Matrix4().makeTranslation(-hc.x, -hc.y, -hc.z));
}

/** Vertical grooves around the y axis so hair masses read as strands. */
function strands(g: BufferGeometry, amt: number, freq: number): BufferGeometry {
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    const a = Math.atan2(x, z);
    const k = 1 + amt * Math.abs(Math.sin(a * freq));
    pos.setX(i, x * k); pos.setZ(i, z * k);
  }
  g.computeVertexNormals();
  return g;
}

/** The volume a hat's crown takes up, in head-normalised space (see tuckUnderHat). */
interface Crown { y: number; tilt: number; yaw: number; T: number; r: number }

function crownOf(look: KidLook, R: number): Crown | null {
  const lift = look.hair === 'afro' ? R * 0.38 : 0;
  switch (look.hat) {
    case 'cap': case 'trucker': return { y: lift, tilt: -0.18, yaw: 0, T: 1.28, r: R * 1.04 };
    case 'capBack': return { y: lift, tilt: 0.18, yaw: Math.PI, T: 1.28, r: R * 1.04 };
    case 'bucket': return { y: 0, tilt: 0.1, yaw: 0, T: 1.22, r: R * 1.02 };
    case 'cowboy': return { y: 0, tilt: 0.12, yaw: 0, T: 1.2, r: R * 1.0 };
    case 'flatcap': return { y: R * 0.12, tilt: 0.25, yaw: 0, T: 1.1, r: R * 0.98 };
    case 'hardhat': return { y: 0, tilt: 0, yaw: 0, T: 1.3, r: R * 1.08 };
    default: return null;
  }
}

/**
 * Squash every hair vertex that would poke through the hat's crown back inside it,
 * so curls, spikes, bowls and fringes sit under the hat instead of through it.
 */
function tuckUnderHat(L: Lists, p: Proportions, look: KidLook) {
  const c = crownOf(look, p.headR);
  if (!c) return;
  const [sx, sy, sz] = HEAD_SHAPE[look.head] ?? HEAD_SHAPE.round;
  const kx = look.head === 'square' ? 1.03 : 1;
  const hc = headCentre(p);
  const inv = new Quaternion().setFromEuler(new Euler(c.tilt, c.yaw, 0, 'YXZ')).invert();
  const v = new Vector3();
  for (const g of L.hair.parts) {
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      v.set((pos.getX(i) - hc.x) / (sx * kx), (pos.getY(i) - hc.y) / sy - c.y, (pos.getZ(i) - hc.z) / sz);
      const d = v.length();
      if (d <= c.r) continue;
      const loc = v.clone().applyQuaternion(inv);
      const ang = Math.acos(Math.max(-1, Math.min(1, loc.y / d)));
      if (ang > c.T) continue;
      v.multiplyScalar(c.r / d);
      pos.setXYZ(i, hc.x + v.x * sx * kx, hc.y + (v.y + c.y) * sy, hc.z + v.z * sz);
    }
  }
}

// ─────────────────────────────────────────────────────────────── hair

export function addHair(L: Lists, p: Proportions, kid: Kid) {
  const look = kid.look;
  const R = p.headR;
  const hc = headCentre(p);
  const SH = headShapeMatrix(p, look);
  const at = (x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, s: number | [number, number, number] = 1) => M4(hc.x + x, hc.y + y, hc.z + z, rx, ry, rz, s);
  const H = (g: BufferGeometry, m: Matrix4) => L.hair.add(rigid(g, B.head), SH.clone().multiply(m));
  const rnd = seeded(kid.id + 'hair');
  const hatted = ['cap', 'capBack', 'trucker', 'bucket', 'cowboy', 'flatcap', 'hardhat'].includes(look.hat);
  // hair that falls below the head follows the chest
  const toChest = (g: BufferGeometry, m: Matrix4) => {
    g.applyMatrix4(SH.clone().multiply(m));
    L.hair.add(blended(g, (q) => { const w = ramp(q.y, p.neckY - 0.1, p.neckY + 0.25); return [B.chest, 1 - w, B.head, w]; }));
  };
  // a coloured hair tie
  const tie = (m: Matrix4, hex: string) => L.cloth.add(paint(rigid(new TorusGeometry(R * 0.085, R * 0.04, 6, 12), B.head), hex), SH.clone().multiply(m));
  const tieCol = ['#ff6fae', '#5fb3ff', '#ffd23f', '#8be07a'][Math.abs(kid.id.charCodeAt(0)) % 4];
  const along = (from: Vector3, dir: Vector3) => new Matrix4().compose(from, new Quaternion().setFromUnitVectors(new Vector3(0, -1, 0), dir.clone().normalize()), new Vector3(1, 1, 1));
  const scalp = (rk: number, front = 0.78) => H(hairCap(R * rk, front), at(0, 0, 0));
  switch (look.hair) {
    case 'buzz':
      H(lumpy(hairCap(R * 1.04, 0.86), 0.006, 40, 1), at(0, 0, 0));
      break;
    case 'sidepart':
      scalp(1.05);
      H(lumpy(new SphereGeometry(R * 0.55, 20, 12), 0.04, 12, 2), at(R * 0.25, R * 0.62, R * 0.22, 0.3, 0, -0.5, [1.25, 0.45, 0.9]));
      H(new SphereGeometry(R * 0.4, 16, 10), at(-R * 0.35, R * 0.7, R * 0.1, 0.2, 0, 0.4, [1.1, 0.4, 0.9]));
      break;
    case 'messy':
      scalp(1.06);
      for (let i = 0; i < 18; i++) {
        const a = rnd() * Math.PI * 2, el = hatted ? -0.05 + rnd() * 0.3 : 0.25 + rnd() * 0.9;
        const d = new Vector3(Math.cos(a) * Math.cos(el), Math.sin(el), Math.sin(a) * Math.cos(el) * 0.8 - 0.25);
        if (d.z > 0.55 && d.y < 0.6) continue;
        if (hatted && d.normalize().z > -0.35) continue; // under a hat only the back tufts show
        const g = new CylinderGeometry(0, R * 0.16, R * (hatted ? 0.3 : 0.45), 7);
        const q = new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), d.clone().normalize());
        H(g, new Matrix4().compose(hc.clone().addScaledVector(d.normalize(), R * 1.02), q, new Vector3(1, 1, 1)));
      }
      break;
    case 'spiky': {
      scalp(1.04);
      const n = hatted ? 7 : 12;
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 + rnd() * 0.4;
        const el = hatted ? -0.1 + rnd() * 0.2 : 0.45 + rnd() * 0.7;
        const d = new Vector3(Math.cos(a) * Math.cos(el), Math.sin(el), Math.sin(a) * Math.cos(el) * 0.9 - 0.15).normalize();
        if (d.z > 0.7 && d.y < 0.4) continue;
        if (hatted && d.z > -0.3) continue; // spikes stick out the back under a cap
        const len = R * (hatted ? 0.35 : 0.55 + rnd() * 0.25);
        const g = new CylinderGeometry(0, R * 0.17, len, 6);
        g.translate(0, len / 2, 0);
        H(g, new Matrix4().compose(hc.clone().addScaledVector(d, R * 0.9), new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), d), new Vector3(1, 1, 1)));
      }
      break;
    }
    case 'bowl':
      H(cap(R * 1.1, 1.08, 0.08, 40, 16), at(0, 0, 0));
      H(loft([{ y: -0.02, rx: R * 1.11, rz: R * 1.11 }, { y: 0.06, rx: R * 1.12, rz: R * 1.12 }], 40), at(0, R * Math.cos(1.08) * 1.1 - 0.02, 0));
      break;
    case 'curly': {
      scalp(1.06);
      for (let i = 0; i < (hatted ? 60 : 70); i++) {
        if (hatted) {
          // under a hat the curls fill a band around the sides and back, down to the nape
          const a = (rnd() * 2 - 1) * 2.2, el = -0.45 + rnd() * 0.6;
          if (Math.abs(a) > 1.55 && el < -0.12) continue; // not over the cheeks
          const u = new Vector3(Math.sin(a) * Math.cos(el), Math.sin(el), -Math.cos(a) * Math.cos(el));
          H(new SphereGeometry(R * (0.12 + rnd() * 0.05), 8, 6), at(u.x * R * 1.03, u.y * R * 1.03, u.z * R * 1.03));
          continue;
        }
        const u = new Vector3(rnd() * 2 - 1, rnd() * 1.2 - 0.2, rnd() * 2 - 1.2);
        if (u.lengthSq() > 1.4 || u.lengthSq() < 0.05) continue;
        u.normalize();
        if (u.z > 0.45 && u.y < 0.55) continue;
        if (u.y < -0.15) continue;
        H(new SphereGeometry(R * (0.15 + rnd() * 0.07), 9, 7), at(u.x * R * 1.08, u.y * R * 1.08, u.z * R * 1.08));
      }
      break;
    }
    case 'afro': {
      const g = lumpy(new SphereGeometry(R * 1.3, 30, 22), 0.05, 9, 3).applyMatrix4(at(0, R * 0.3, -R * 0.12, 0, 0, 0, [1, 0.9, 0.95]));
      // open the face: anything in front of the face below the hairline sinks inside the skull
      const pos = g.attributes.position, v = new Vector3();
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i).sub(hc);
        const hairline = R * (0.42 - 0.35 * (v.x / R) ** 2);
        if (v.z > R * 0.05 && v.y < hairline) {
          v.normalize().multiplyScalar(R * 0.9).add(hc);
          pos.setXYZ(i, v.x, v.y, v.z);
        }
      }
      g.computeVertexNormals();
      H(g, new Matrix4());
      break;
    }
    case 'ponytail':
    case 'bun':
      scalp(1.05);
      if (look.hair === 'bun') H(lumpy(new SphereGeometry(R * 0.38, 16, 12), 0.05, 14, 4), at(0, R * 0.95, -R * 0.45));
      else {
        tie(at(0, R * 0.35, -R * 1.02, 0.4), tieCol);
        toChest(lumpy(limb(R * 1.3, R * 0.2, R * 0.08, 10), 0.04, 20, 3), at(0, R * 0.35, -R * 1.08, -0.35));
      }
      break;
    case 'pigtails': {
      scalp(1.05);
      for (const sx of [-1, 1]) {
        // tied behind and above the ears, springing out and down
        const root = new Vector3(hc.x + sx * R * 0.82, hc.y + R * 0.12, hc.z - R * 0.52);
        const dir = new Vector3(sx * 0.32, -0.92, -0.25);
        tie(along(root, dir).multiply(new Matrix4().makeRotationX(Math.PI / 2)), tieCol);
        const tail = lumpy(limb(R * 0.8, R * 0.2, R * 0.1, 10), 0.05, 22, sx + 5);
        tail.scale(1, 1, 0.8);
        H(tail, along(root.clone().addScaledVector(dir.clone().normalize(), R * 0.08), dir));
      }
      break;
    }
    case 'braids': {
      scalp(1.05);
      for (const sx of [-1, 1]) {
        // plaited down behind the ears, over the back of the shoulders
        for (let k = 0; k < 9; k++) {
          const t = k / 8;
          const g = new SphereGeometry(R * (0.16 - t * 0.04), 10, 8);
          const wob = (k % 2 ? 1 : -1) * R * 0.035;
          toChest(g, at(sx * R * (0.72 + t * 0.05) + wob, -R * (0.05 + k * 0.24), -R * (0.72 - t * 0.12), 0, 0, (k % 2 ? 1 : -1) * 0.35, [1, 1.3, 0.95]));
        }
        tie(at(sx * R * 0.78, -R * 2.12, -R * 0.62, Math.PI / 2), tieCol);
        toChest(lumpy(new CylinderGeometry(R * 0.07, R * 0.12, R * 0.22, 8), 0.06, 30, sx), at(sx * R * 0.78, -R * 2.28, -R * 0.62));
      }
      break;
    }
    case 'bob':
    case 'long': {
      scalp(1.07, 0.72);
      const long = look.hair === 'long';
      // curtain of hair around the sides and back, open at the face
      const curtain = new SphereGeometry(R * 1.12, 34, 18, Math.PI / 2 + 0.95, Math.PI * 2 - 1.9, 0.25, long ? Math.PI * 0.62 : Math.PI * 0.6);
      H(strands(lumpy(curtain, 0.02, 18, 5), 0.025, 9), at(0, 0, -R * 0.02));
      if (long) {
        // the length falls down the back in a soft, rounded mass with strands
        const sheet = loft([
          { y: -R * 2.08, rx: R * 0.3, rz: R * 0.1, cz: -R * 0.6 },
          { y: -R * 1.98, rx: R * 0.62, rz: R * 0.2, cz: -R * 0.62 },
          { y: -R * 1.6, rx: R * 0.84, rz: R * 0.32, cz: -R * 0.64 },
          { y: -R * 1.05, rx: R * 0.94, rz: R * 0.45, cz: -R * 0.6 },
          { y: -R * 0.3, rx: R * 0.98, rz: R * 0.56, cz: -R * 0.5 },
        ], 28, true, false);
        // pointed locks at the ends
        const sp = sheet.attributes.position;
        for (let i = 0; i < sp.count; i++) {
          const y = sp.getY(i);
          if (y < -R * 1.9) sp.setY(i, y - Math.abs(Math.sin(Math.atan2(sp.getX(i), sp.getZ(i) + R * 0.6) * 4)) * R * 0.14);
        }
        toChest(strands(lumpy(sheet, 0.02, 12, 8), 0.07, 12), at(0, 0, 0));
      }
      // fringe
      H(lumpy(new SphereGeometry(R * 0.6, 18, 10), 0.03, 14, 6), at(0, R * 0.68, R * 0.42, 0.5, 0, 0, [1.3, 0.35, 0.8]));
      break;
    }
    case 'mohawk':
      H(cap(R * 1.015, 1.3, 0.7), at(0, 0, 0));
      for (let i = 0; i < 9; i++) {
        const a = -0.9 + (i / 8) * 2.4;
        const d = new Vector3(0, Math.cos(a), Math.sin(a));
        const g = new CylinderGeometry(0, R * 0.12, R * 0.6, 5);
        g.translate(0, R * 0.3, 0);
        g.scale(0.5, 1, 1.4);
        H(g, new Matrix4().compose(hc.clone().addScaledVector(d, R * 0.95), new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), d), new Vector3(1, 1, 1)));
      }
      break;
  }
}

// ─────────────────────────────────────────────────────────────── hats

export function addHat(L: Lists, p: Proportions, kid: Kid, _team: Team, col: UniformColors) {
  const look = kid.look;
  const R = p.headR;
  const hc = headCentre(p);
  tuckUnderHat(L, p, look);
  const SH = headShapeMatrix(p, look);
  const lift = look.hair === 'afro' ? R * 0.38 : 0;
  const at = (x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, s: number | [number, number, number] = 1) => SH.clone().multiply(M4(hc.x + x, hc.y + y + lift, hc.z + z, rx, ry, rz, s));
  const C = (g: BufferGeometry, hex: string | Color, m: Matrix4) => L.cloth.add(paint(rigid(g, B.head), hex), m);
  const brim = (w: number, d: number, curve: number) => {
    // a curved bill: half of a flat lens, thick at the crown and thinning to a rounded edge.
    // It starts a little inside the crown (z < 0) so there's never a gap where they meet.
    const g = new SphereGeometry(1, 22, 8, 0, Math.PI, 0, Math.PI);
    const back = 0.3;
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) * w, z = pos.getZ(i) * d * (1 + back) - d * back;
      const y = pos.getY(i) * R * 0.035 - x * x * curve - Math.max(0, z / d) ** 2 * R * 0.06;
      pos.setXYZ(i, x, y, z);
    }
    g.computeVertexNormals();
    return g;
  };
  switch (look.hat) {
    case 'cap':
    case 'capBack':
    case 'trucker': {
      const back = look.hat === 'capBack' ? Math.PI : 0;
      const trucker = look.hat === 'trucker';
      const RC = R * 1.09;
      // the crown comes down to just above the brows at the front, lower at the back
      // (worn backwards it still sits low at the back of the head, so the tilt flips)
      const CT = 1.26, CA = back ? -0.18 : 0.18;
      const crown = cap(RC, CT, CA, 36, 14);
      crown.scale(1, trucker ? 1.08 : 1.02, 1.03);
      if (trucker) {
        // foam front panel, mesh sides and back
        const foam = new Color('#f2efe6'), mesh = new Color('#2f5e34'), dark = new Color('#29532e');
        L.cloth.add(paintFn(rigid(crown, B.head), (q) => (Math.abs(Math.atan2(q.x, q.z)) < 0.78 ? foam : (Math.round(q.x * 40) + Math.round(q.y * 40)) % 2 ? mesh : dark)), at(0, 0.02, 0, 0, back));
      } else {
        L.cloth.add(paint(rigid(crown, B.head), col.cap), at(0, 0.02, 0, 0, back));
      }
      // button + bill
      C(new SphereGeometry(R * 0.08, 10, 8), trucker ? '#2f5e34' : col.cap, at(0, RC * (trucker ? 1.1 : 1.03), -Math.sign(CA) * R * 0.1, 0, back));
      // the bill grows out of the crown's front edge
      const edge = CT - CA;
      const bill = brim(R * 0.8, R * 0.9, 0.3 / R);
      const bm = at(0, 0.02 + Math.cos(edge) * RC * (trucker ? 1.08 : 1.02) - R * 0.02, 0, 0, back).multiply(M4(0, 0, Math.sin(edge) * RC * 1.03 - R * 0.04, 0.1));
      C(bill, trucker ? '#2f5e34' : col.brim, bm);
      if (!trucker) {
        // team logo on the front panel (uses the cap-logo corner of the jersey texture)
        const logo = new SphereGeometry(RC * 1.006, 12, 8, Math.PI / 2 - 0.42, 0.84, 0.55, 0.62);
        const uv = logo.attributes.uv;
        for (let i = 0; i < uv.count; i++) {
          uv.setXY(i, CAP_LOGO_UV.u0 + uv.getX(i) * (CAP_LOGO_UV.u1 - CAP_LOGO_UV.u0), CAP_LOGO_UV.v0 + uv.getY(i) * (CAP_LOGO_UV.v1 - CAP_LOGO_UV.v0));
        }
        logo.applyMatrix4(new Matrix4().makeRotationX(-CA));
        L.jersey.add(rigid(logo, B.head), at(0, 0.02, 0, 0, back, 0, [1, 1.02, 1.03]));
      } else {
        // a little pine-tree patch on the foam
        const patch = new SphereGeometry(RC * 1.012, 10, 6, Math.PI / 2 - 0.3, 0.6, 0.62, 0.36);
        patch.applyMatrix4(new Matrix4().makeRotationX(-0.18));
        L.cloth.add(paintFn(rigid(patch, B.head), (q) => new Color(Math.abs(q.x) < (RC * 0.8 - q.y) * 0.7 ? '#2f7d3a' : '#d9a441')), at(0, 0.02, 0, 0, 0, 0, [1, 1.08, 1.03]));
      }
      break;
    }
    case 'visor': {
      // a band hugging the head above the brows, the bill high enough to show the eyes
      C(hugBand(R * 1.09, R * 0.4, R * 0.13), '#f4f4f0', at(0, 0, 0, -0.2));
      C(brim(R * 0.8, R * 0.9, 0.3 / R), col.trim, at(0, 0, 0, -0.2).multiply(M4(0, R * 0.32, R * 0.98, 0.26)));
      break;
    }
    case 'bucket': {
      // sits on the crown of the head with the brim sloping down to just above the brows
      const crown = loft([{ y: 0, rx: R * 1.1, rz: R * 1.1 }, { y: R * 0.5, rx: R * 1.0, rz: R * 1.0 }, { y: R * 0.66, rx: R * 0.86, rz: R * 0.86 }, { y: R * 0.72, rx: R * 0.6, rz: R * 0.6 }], 32, false, true);
      C(crown, '#c9b98a', at(0, R * 0.38, -0.02, 0.1));
      C(loft([{ y: 0, rx: R * 1.105, rz: R * 1.105 }, { y: R * 0.1, rx: R * 1.09, rz: R * 1.09 }], 32), '#8a7a52', at(0, R * 0.4, -0.02, 0.1));
      const rim = loft([{ y: -R * 0.2, rx: R * 1.5, rz: R * 1.5 }, { y: -R * 0.16, rx: R * 1.42, rz: R * 1.42 }, { y: 0, rx: R * 1.1, rz: R * 1.1 }], 32);
      const rimBack = rim.clone();
      rimBack.index!.array.reverse();
      const rn = rimBack.attributes.normal;
      for (let i = 0; i < rn.count; i++) rn.setXYZ(i, -rn.getX(i), -rn.getY(i), -rn.getZ(i));
      C(rim, '#bfae7e', at(0, R * 0.4, -0.02, 0.1));
      C(rimBack, '#a8996c', at(0, R * 0.395, -0.02, 0.1));
      break;
    }
    case 'flatcap': {
      C(lumpy(new SphereGeometry(R * 1.12, 28, 12, 0, Math.PI * 2, 0, 1.3), 0.01, 10, 2), '#6b5a48', at(0, R * 0.12, -0.04, 0.25, 0, 0, [1.04, 0.62, 1.12]));
      C(brim(R * 0.7, R * 0.5, 0.1 / R), '#5e4f3f', at(0, R * 0.48, 0, 0.2).multiply(new Matrix4().makeTranslation(0, 0, R * 1.0)));
      break;
    }
    case 'cowboy': {
      const crown = loft([{ y: 0, rx: R * 1.08, rz: R * 1.12 }, { y: R * 0.65, rx: R * 0.95, rz: R * 1.0 }, { y: R * 0.8, rx: R * 0.8, rz: R * 0.92 }], 28, false, true);
      C(crown, '#c49a5a', at(0, R * 0.42, -0.03, 0.12));
      const rim = new TorusGeometry(R * 1.65, R * 0.45, 4, 40);
      rim.rotateX(Math.PI / 2);
      rim.scale(1, 0.12, 1);
      const pos = rim.attributes.position;
      for (let i = 0; i < pos.count; i++) { const x = pos.getX(i); pos.setY(i, pos.getY(i) + (x / (R * 2.1)) ** 2 * R * 0.6); }
      rim.computeVertexNormals();
      C(rim, '#b88c4e', at(0, R * 0.42, -0.03, 0.12));
      C(loft([{ y: 0, rx: R * 1.09, rz: R * 1.13 }, { y: R * 0.12, rx: R * 1.07, rz: R * 1.11 }], 28), '#5a3a22', at(0, R * 0.44, -0.03, 0.12));
      break;
    }
    case 'hardhat': {
      C(cap(R * 1.16, 1.45, 0.1, 30, 12), '#f2c94c', at(0, 0.04, 0));
      C(loft([{ y: 0, rx: R * 1.32, rz: R * 1.38, cz: R * 0.12 }, { y: R * 0.06, rx: R * 1.3, rz: R * 1.36, cz: R * 0.12 }], 32, true, true), '#f2c94c', at(0, R * 0.12, 0));
      break;
    }
  }
}

// ─────────────────────────────────────────────────────────────── costume

export function addCostume(L: Lists, p: Proportions, kid: Kid) {
  const look = kid.look;
  const R = p.headR, s = p.s, wf = p.wf;
  const hc = headCentre(p);
  const SH = headShapeMatrix(p, look);
  const [shx, shy, shz] = HEAD_SHAPE[look.head] ?? HEAD_SHAPE.round;
  const atH = (x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, sc: number | [number, number, number] = 1) => M4(hc.x + x, hc.y + y, hc.z + z, rx, ry, rz, sc);
  const head = (list: 'cloth' | 'shiny' | 'hair', g: BufferGeometry, hex: string | null, m: Matrix4) => {
    rigid(g, B.head);
    if (hex) paint(g, hex);
    L[list].add(g, m);
  };
  /** placed on the skull in head-normalised coordinates (follows the head shape) */
  const headS = (list: 'cloth' | 'shiny' | 'hair', g: BufferGeometry, hex: string | null, m: Matrix4) => head(list, g, hex, SH.clone().multiply(m));
  /** a point on the face surface (model space) for head-normalised x, y (in units of R) */
  const onFace = (x: number, y: number, out = 0) => {
    const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
    return shapeHead(new Vector3(x * R, y * R, (z + out) * R), R, look).add(hc);
  };
  const chestZ = 0.33 * wf * s + p.belly * 0.12 * s; // front of the jersey at chest height
  const onChest = (list: 'cloth' | 'shiny', g: BufferGeometry, hex: string, m: Matrix4) => {
    g.applyMatrix4(m);
    paint(g, hex);
    L[list].add(blended(g, (q) => { const w = ramp(q.y, p.waistY, p.chestY); return [B.spine, 1 - w, B.chest, w]; }));
  };

  // ── facial hair worn as stick-ons, sitting on the upper lip between nose and mouth
  if (look.face === 'walrus') {
    // a bushy brush of overlapping tufts that droops at the ends
    for (let i = 0; i < 6; i++) {
      const t = i / 5 * 2 - 1;
      const at = onFace(t * 0.38, -0.41 - t * t * 0.16, 0.03);
      const g = lumpy(new SphereGeometry(R * (0.19 - Math.abs(t) * 0.05), 12, 8), 0.07, 26, i);
      head('hair', g, null, new Matrix4().compose(at, new Quaternion().setFromEuler(new Euler(0.3, t * 0.5, -t * 0.7)), new Vector3(1.3, 0.72, 0.62)));
    }
  } else if (look.face === 'handlebar') {
    for (const sx of [-1, 1]) {
      // a waxed bar along the lip that curls up at the tip
      const a = onFace(sx * 0.03, -0.39, 0.01), b = onFace(sx * 0.36, -0.42, 0.0);
      head('hair', limb(a.distanceTo(b), R * 0.06, R * 0.035, 8), null, alongMatrix(a, b));
      head('hair', new TorusGeometry(R * 0.07, R * 0.025, 6, 12, Math.PI * 1.3), null,
        new Matrix4().compose(onFace(sx * 0.43, -0.35, -0.02), new Quaternion().setFromEuler(new Euler(0, sx * 0.6, sx > 0 ? -0.3 : Math.PI + 0.3, 'YXZ')), new Vector3(1, 1, 1)));
    }
  }
  if (look.face === 'beard') {
    // a cut-up yellow sponge on a string, hanging under the mouth
    const sponge = new SphereGeometry(R * 0.62, 18, 12, 0, Math.PI * 2, Math.PI * 0.45, Math.PI * 0.55);
    paintFn(lumpy(sponge, 0.05, 25, 7), (q) => new Color(Math.sin(q.x * 90) * Math.sin(q.y * 80) > 0.6 ? '#c7a92c' : '#e9cf4a'));
    L.cloth.add(rigid(sponge, B.head), SH.clone().multiply(atH(0, -R * 0.8, R * 0.32, -0.1, 0, 0, [1.0, 1.0, 0.95])));
    headS('cloth', loft([{ y: 0, rx: R * 1.03, rz: R * 1.03 }, { y: 0.02, rx: R * 1.03, rz: R * 1.03 }], 24), '#ddd6c2', atH(0, -R * 0.12, 0, 0.75));
  }

  // ── eyewear
  const eyeY = p.eyeY - hc.y, eyeX = p.eyeX, eyeZ = p.eyeZ - hc.z;
  const frame = (hex: string, lens: string | null, round: number, size: number) => {
    const rr = p.eyeR * size;
    const fz = eyeZ + p.eyeR * 1.0;
    for (const sx of [-1, 1]) {
      const ring = new TorusGeometry(rr, p.eyeR * 0.1, 6, 22);
      ring.scale(1, round, 1);
      head('shiny', ring, hex, atH(sx * eyeX, eyeY, fz));
      if (lens) {
        // a flat lens filling the whole frame
        const l = new CylinderGeometry(rr, rr, p.eyeR * 0.05, 22);
        l.rotateX(Math.PI / 2);
        l.scale(1, round, 1);
        head('shiny', l, lens, atH(sx * eyeX, eyeY, fz - p.eyeR * 0.02));
      }
      // arms bow around the side of the head to the ear
      const a = new Vector3(hc.x + sx * (eyeX + rr * 0.97), hc.y + eyeY + rr * round * 0.2, hc.z + fz - p.eyeR * 0.1);
      const ear = new Vector3(hc.x + sx * R * shx * 1.0, hc.y + eyeY - R * 0.04, hc.z - R * 0.12);
      const mid = a.clone().lerp(ear, 0.5);
      const off = mid.clone().sub(hc);
      off.x /= shx; off.z /= shz;
      const need = R * 1.04 - Math.hypot(off.x, off.z);
      if (need > 0) { const k = (Math.hypot(off.x, off.z) + need) / Math.hypot(off.x, off.z); mid.x = hc.x + off.x * k * shx; mid.z = hc.z + off.z * k * shz; }
      head('shiny', limb(a.distanceTo(mid), p.eyeR * 0.07, p.eyeR * 0.07, 5), hex, alongMatrix(a, mid));
      head('shiny', limb(mid.distanceTo(ear), p.eyeR * 0.07, p.eyeR * 0.07, 5), hex, alongMatrix(mid, ear));
    }
    // bridge
    head('shiny', limb(eyeX * 2 - rr * 2, p.eyeR * 0.08, p.eyeR * 0.08, 5), hex, atH(-(eyeX - rr), eyeY + rr * round * 0.25, fz + p.eyeR * 0.04, 0, 0, Math.PI / 2));
  };
  switch (look.eyewear) {
    case 'glasses': frame('#2a2a2a', null, 1, 1.25); break;
    case 'reading': frame('#7a3b2a', null, 0.7, 1.15);
      // chain
      // chain: from the arms behind the ears, looping down to the collar
      for (const sx of [-1, 1]) {
        const a = new Vector3(hc.x + sx * R * shx * 1.0, hc.y + eyeY - R * 0.1, hc.z - R * 0.1);
        const b = new Vector3(sx * 0.2 * s * wf, p.neckY + 0.03 * s, 0.06 * s);
        const mid = a.clone().lerp(b, 0.55).add(new Vector3(sx * R * 0.12, -R * 0.12, R * 0.08));
        for (const [u, v] of [[a, mid], [mid, b]]) {
          const g = limb(u.distanceTo(v), 0.012, 0.012, 4).applyMatrix4(alongMatrix(u, v));
          L.cloth.add(paint(blended(g, (q) => { const w = ramp(q.y, p.neckY, hc.y - R * 0.5); return [B.chest, 1 - w, B.head, w]; }), '#d4b44a'));
        }
      }
      break;
    case 'shades': frame('#151515', '#0d1014', 0.78, 1.34); break;
    case 'aviators': frame('#d4b44a', '#3a2a1a', 1.05, 1.3); break;
    case 'goggles': frame('#2f6fd0', '#9fd8ff', 1, 1.35); break;
    case 'monocle': head('shiny', new TorusGeometry(p.eyeR * 1.2, p.eyeR * 0.12, 6, 20), '#d4b44a', atH(eyeX, eyeY, eyeZ + p.eyeR * 0.95)); break;
  }

  // ── neckwear
  const neckY = p.neckY + 0.04 * s;
  switch (look.neck) {
    case 'tie': {
      onChest('cloth', new SphereGeometry(0.07 * s, 10, 8), '#b8312f', M4(0, neckY - 0.02, 0.2 * s * wf + 0.03));
      const tie = loft([{ y: -0.95 * s, rx: 0.01, rz: 0.01 }, { y: -0.85 * s, rx: 0.1 * s, rz: 0.02 }, { y: -0.1 * s, rx: 0.055 * s, rz: 0.02 }, { y: 0, rx: 0.045 * s, rz: 0.02 }], 6);
      onChest('cloth', tie, '#b8312f', M4(0, neckY - 0.06, chestZ + 0.02, -0.12));
      break;
    }
    case 'bowtie':
      for (const sx of [-1, 1]) onChest('cloth', new SphereGeometry(0.1 * s, 10, 8), '#c0392b', M4(sx * 0.1 * s, neckY - 0.04, 0.21 * s * wf + 0.02, 0, 0, sx * 0.4, [1.2, 0.7, 0.4]));
      onChest('cloth', new SphereGeometry(0.05 * s, 8, 6), '#962d22', M4(0, neckY - 0.04, 0.23 * s * wf + 0.02));
      break;
    case 'pearls':
      for (let i = 0; i < 18; i++) {
        const a = (i / 18) * Math.PI * 2;
        onChest('shiny', new SphereGeometry(0.04 * s, 8, 6), '#f6f1e6', M4(Math.sin(a) * 0.24 * s * wf, neckY - 0.08 - Math.max(0, Math.cos(a)) * 0.12 * s, Math.cos(a) * 0.23 * s * wf + 0.02));
      }
      break;
    case 'whistle':
    case 'lanyard':
    case 'medal': {
      const cord = loft([{ y: 0, rx: 0.235 * s * wf, rz: 0.225 * s * wf }, { y: 0.025, rx: 0.235 * s * wf, rz: 0.225 * s * wf }], 20);
      onChest('cloth', cord, look.neck === 'medal' ? '#2f6fd0' : '#d63a3a', M4(0, neckY - 0.1, 0.06, -0.55));
      if (look.neck === 'whistle') onChest('shiny', new CylinderGeometry(0.05 * s, 0.05 * s, 0.16 * s, 10), '#c9cdd1', M4(0, p.chestY + 0.02, chestZ + 0.05, 0, 0, Math.PI / 2));
      if (look.neck === 'lanyard') onChest('cloth', new CylinderGeometry(0.15 * s, 0.15 * s, 0.02, 4), '#f4f4f0', M4(0, p.chestY - 0.05, chestZ + 0.03, Math.PI / 2, Math.PI / 4));
      if (look.neck === 'medal') onChest('shiny', new CylinderGeometry(0.1 * s, 0.1 * s, 0.03, 16), '#d4b44a', M4(0, p.chestY, chestZ + 0.04, Math.PI / 2));
      break;
    }
    case 'scarf': {
      const sc = loft([{ y: -0.08 * s, rx: 0.26 * s * wf, rz: 0.25 * s * wf }, { y: 0.1 * s, rx: 0.25 * s * wf, rz: 0.24 * s * wf }, { y: 0.16 * s, rx: 0.22 * s * wf, rz: 0.21 * s * wf }], 20);
      const scol = look.hairColor % 2 ? '#e85d75' : '#7fb8e8';
      const stripe = (g: BufferGeometry, len: number) => {
        const a = new Color(scol), b = new Color(scol).lerp(new Color('#ffffff'), 0.45);
        return paintFn(g, (q) => (Math.floor((-q.y / len) * 6) % 2 ? b : a));
      };
      onChest('cloth', sc, scol, M4(0, neckY - 0.02, 0.01));
      // knotted at one side, two striped tails falling over the shoulder
      const kx = 0.17 * s * wf;
      onChest('cloth', new SphereGeometry(0.08 * s, 10, 8), scol, M4(kx, neckY - 0.06, 0.22 * s * wf, 0, 0, 0, [1, 0.9, 0.7]));
      for (const [dx, len, ang] of [[0, 0.62, 0.1], [0.07, 0.48, 0.32]] as const) {
        const tail = stripe(limb(len * s, 0.085 * s, 0.09 * s, 8, 0.32), len * s);
        tail.applyMatrix4(M4(kx + dx * s, neckY - 0.1, chestZ + 0.02 - dx * 0.2, -0.12, 0, ang));
        L.cloth.add(blended(tail, (q) => { const w = ramp(q.y, p.waistY, p.chestY); return [B.spine, 1 - w, B.chest, w]; }));
      }
      break;
    }
    case 'bandana': {
      const tri = new CylinderGeometry(0.0, 0.3 * s * wf, 0.32 * s, 3, 1);
      onChest('cloth', tri, '#c0392b', M4(0, neckY - 0.12, 0.2 * s * wf, Math.PI, 0, 0, [1, 1, 0.25]));
      onChest('cloth', loft([{ y: 0, rx: 0.235 * s * wf, rz: 0.225 * s * wf }, { y: 0.07, rx: 0.23 * s * wf, rz: 0.22 * s * wf }], 20), '#c0392b', M4(0, neckY - 0.04, 0.01));
      break;
    }
  }

  // ── extras on the head
  const earX = R * 0.98;
  switch (look.extra) {
    case 'headset':
      headS('cloth', new TorusGeometry(R * 1.1, R * 0.05, 6, 30, Math.PI), '#2a2a2a', atH(0, R * 0.05, -R * 0.05, 0, 0, 0));
      for (const sx of [-1, 1]) headS('cloth', new CylinderGeometry(R * 0.2, R * 0.2, R * 0.12, 14), '#2a2a2a', atH(sx * R * 1.05, 0, 0, 0, 0, Math.PI / 2));
      headS('cloth', limb(R * 0.8, R * 0.025, R * 0.025, 5), '#2a2a2a', atH(earX, -R * 0.05, R * 0.05, -1.25, 0, 0.35));
      headS('cloth', new SphereGeometry(R * 0.07, 8, 6), '#111', atH(R * 0.42, -R * 0.38, R * 0.82));
      break;
    case 'earpiece': {
      headS('cloth', new SphereGeometry(R * 0.08, 8, 6), '#e6e6e0', atH(-earX * 1.03, 0, R * 0.02));
      // coiled cord tucked down the back of the neck into the collar
      const a = new Vector3(hc.x - earX * 1.0 * shx, hc.y - R * 0.12 * shy, hc.z - R * 0.04);
      const b = new Vector3(-0.12 * s * wf, p.neckY + 0.06 * s, -0.17 * s * wf);
      const g = limb(a.distanceTo(b), R * 0.022, R * 0.022, 5).applyMatrix4(alongMatrix(a, b));
      L.cloth.add(paint(blended(g, (q) => { const w = ramp(q.y, p.neckY, hc.y - R * 0.3); return [B.chest, 1 - w, B.head, w]; }), '#e6e6e0'));
      break;
    }
    case 'pencil':
      headS('cloth', new CylinderGeometry(R * 0.04, R * 0.04, R * 0.8, 6), '#f2c94c', atH(earX * 1.02, R * 0.12, -R * 0.05, 0, 0, Math.PI / 2 - 0.5));
      headS('cloth', new CylinderGeometry(0.0, R * 0.04, R * 0.1, 6), '#e8c49a', atH(earX * 1.02 - R * 0.3, R * 0.3, -R * 0.05, 0, 0, Math.PI / 2 - 0.5 + Math.PI));
      break;
    case 'sweatband':
    case 'headband':
      // (a visor already has its own band)
      if (look.hat !== 'visor') headS('cloth', hugBand(R * 1.07, R * 0.42, R * 0.1), look.extra === 'sweatband' ? '#f4f4f0' : '#e85d75', atH(0, 0, 0, -0.2));
      if (look.extra === 'sweatband') {
        for (const sx of [1, -1]) {
          const bone = sx === 1 ? B.foreL : B.foreR;
          const wr = sx === 1 ? p.joints.handL : p.joints.handR;
          const el = sx === 1 ? p.joints.foreL : p.joints.foreR;
          const g = loft([{ y: -0.07 * s, rx: p.armR * 0.8, rz: p.armR * 0.8 }, { y: 0.07 * s, rx: p.armR * 0.84, rz: p.armR * 0.84 }], 14, true, true);
          const pos = el.clone().lerp(wr, 0.82);
          const q = new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), el.clone().sub(wr).normalize());
          L.cloth.add(paint(rigid(g, bone), '#f4f4f0'), new Matrix4().compose(pos, q, new Vector3(1, 1, 1)));
        }
      }
      break;
    case 'bow':
      for (const sx of [-1, 1]) headS('cloth', new SphereGeometry(R * 0.22, 10, 8), '#ff6fae', atH(sx * R * 0.24 + R * 0.35, R * 0.85, -R * 0.15, 0, 0, sx * 0.5, [1.3, 0.7, 0.45]));
      headS('cloth', new SphereGeometry(R * 0.1, 8, 6), '#e0559a', atH(R * 0.35, R * 0.85, -R * 0.12));
      break;
    case 'curlers':
      for (let i = 0; i < 6; i++) headS('cloth', new CylinderGeometry(R * 0.11, R * 0.11, R * 0.35, 10), ['#ff8fb1', '#8fd3ff', '#ffe38f'][i % 3], atH(-R * 0.5 + (i % 3) * R * 0.5, R * (0.85 - Math.floor(i / 3) * 0.3), -R * (0.1 + Math.floor(i / 3) * 0.5), 0, 0, Math.PI / 2));
      break;
    case 'earrings':
      for (const sx of [-1, 1]) headS('shiny', new TorusGeometry(R * 0.07, R * 0.015, 6, 14), '#d4b44a', atH(sx * earX, -R * 0.24, -R * 0.02));
      break;
    case 'flower':
      for (let i = 0; i < 5; i++) { const a = (i / 5) * Math.PI * 2; headS('cloth', new SphereGeometry(R * 0.08, 8, 6), '#ffffff', atH(R * 0.7 + Math.cos(a) * R * 0.08, R * 0.6 + Math.sin(a) * R * 0.08, R * 0.45)); }
      headS('cloth', new SphereGeometry(R * 0.06, 8, 6), '#f2c94c', atH(R * 0.7, R * 0.6, R * 0.5));
      break;
    case 'bandaid':
      headS('cloth', new SphereGeometry(R * 0.12, 10, 6), '#e8c09a', atH(-R * 0.45, R * 0.35, R * 0.86, 0, 0.3, 0.4, [1.5, 0.5, 0.25]));
      break;
  }

  // ── body pieces
  switch (look.body) {
    case 'suspenders':
      for (const sx of [-1, 1]) {
        for (const fz of [1, -1]) {
          const strap = loft([{ y: p.waistY, rx: 0.06 * s, rz: 0.012 }, { y: p.shoulderY + 0.1 * s, rx: 0.06 * s, rz: 0.012 }], 4);
          const tz = fz * (0.31 * wf * s + (fz > 0 ? p.belly * 0.13 * s : 0) + 0.02);
          onChest('cloth', strap, '#b8312f', M4(sx * 0.22 * wf * s, 0, tz, fz * 0.08 * -1, 0, 0));
        }
        onChest('shiny', new SphereGeometry(0.035 * s, 8, 6), '#d4b44a', M4(sx * 0.22 * wf * s, p.waistY + 0.04, 0.33 * wf * s + p.belly * 0.14 * s));
      }
      break;
    case 'apron': {
      // a cloth panel that wraps the front of the body from the chest to the knees
      const bz = p.belly * 0.13 * s;
      const rows = [
        { y: p.kneeY + 0.12 * s, rx: 0.5 * wf * s, rz: 0.34 * wf * s + bz * 0.4, cz: bz * 0.3 },
        { y: p.hipY - 0.1 * s, rx: 0.47 * wf * s, rz: 0.34 * wf * s + bz * 0.5, cz: bz * 0.35 },
        { y: p.waistY, rx: 0.47 * wf * s + p.belly * 0.04, rz: 0.34 * wf * s + bz, cz: bz * 0.7 },
        { y: (p.waistY + p.chestY) / 2, rx: 0.49 * wf * s + p.belly * 0.05, rz: 0.35 * wf * s + bz * 0.9, cz: bz * 0.6 },
        { y: p.chestY + 0.12 * s, rx: 0.5 * wf * s, rz: 0.35 * wf * s + bz * 0.3, cz: bz * 0.2 },
      ];
      const half = [0.95, 0.95, 0.9, 0.62, 0.55];
      const pos: number[] = [], idx: number[] = [];
      const seg = 10;
      rows.forEach((r, ri) => {
        for (let i = 0; i <= seg; i++) {
          const a = (i / seg * 2 - 1) * half[ri];
          pos.push(Math.sin(a) * r.rx, r.y, Math.cos(a) * r.rz + r.cz + 0.015);
        }
      });
      for (let r = 0; r < rows.length - 1; r++) for (let i = 0; i < seg; i++) {
        const a = r * (seg + 1) + i, b = a + seg + 1;
        idx.push(a, a + 1, b, b, a + 1, b + 1);
      }
      const cloth = new Color('#f6f1e6'), pocket = new Color('#e3dccb');
      for (const side of [1, -1]) {
        // front and (slightly inset, reversed) back faces, so it reads from any angle
        const ap = new BufferGeometry();
        ap.setAttribute('position', new Float32BufferAttribute(pos.map((v, i) => (i % 3 === 2 && side < 0 ? v - 0.008 : v)), 3));
        ap.setIndex(side > 0 ? idx : idx.map((_, i) => idx[i - (i % 3) + 2 - (i % 3)]));
        ap.computeVertexNormals();
        paintFn(ap, (q) => (side > 0 && q.y > p.hipY - 0.02 && q.y < p.waistY - 0.04 && Math.abs(q.x) < 0.2 * wf * s ? pocket : cloth));
        L.cloth.add(blended(ap, (q) => {
          if (q.y < p.waistY) { const w = ramp(q.y, p.hipY - 0.2 * s, p.waistY); return [B.hips, 1 - w, B.spine, w]; }
          const w = ramp(q.y, p.waistY, p.chestY); return [B.spine, 1 - w, B.chest, w];
        }));
      }
      // waist tie and the neck strap
      onChest('cloth', loft([{ y: 0, rx: 0.47 * wf * s + p.belly * 0.04 + 0.012, rz: 0.34 * wf * s + bz + 0.01, cz: bz * 0.7 }, { y: 0.05 * s, rx: 0.47 * wf * s + p.belly * 0.04 + 0.012, rz: 0.34 * wf * s + bz + 0.01, cz: bz * 0.7 }], 24), '#f6f1e6', M4(0, p.waistY - 0.02, 0));
      for (const sx of [-1, 1]) {
        const a = new Vector3(sx * 0.27 * wf * s, p.chestY + 0.1 * s, 0.33 * wf * s + bz * 0.25);
        const b = new Vector3(sx * 0.15 * s * wf, p.neckY + 0.04 * s, 0.02);
        onChest('cloth', limb(a.distanceTo(b), 0.022 * s, 0.022 * s, 5, 0.4), '#f6f1e6', alongMatrix(a, b));
      }
      break;
    }
    case 'pocketProtector':
      onChest('cloth', new CylinderGeometry(0.12 * s, 0.12 * s, 0.2 * s, 4), '#f4f4f0', M4(0.2 * s * wf, p.chestY + 0.08, chestZ + 0.01, 0, Math.PI / 4, 0, [1, 1, 0.15]));
      for (let k = 0; k < 3; k++) onChest('cloth', new CylinderGeometry(0.018 * s, 0.018 * s, 0.22 * s, 6), ['#2f6fd0', '#d63a3a', '#2a2a2a'][k], M4(0.15 * s * wf + k * 0.05 * s, p.chestY + 0.2, chestZ + 0.025));
      break;
    case 'badge': {
      const star = new CylinderGeometry(0.11 * s, 0.11 * s, 0.03, 6);
      onChest('shiny', star, '#d4b44a', M4(0.22 * s * wf, p.chestY + 0.05, chestZ + 0.025, Math.PI / 2));
      break;
    }
    case 'fannyPack':
      L.cloth.add(paint(rigid(new SphereGeometry(0.22 * s, 14, 10), B.hips), '#7a3fbf'), M4(0, p.waistY - 0.12 * s, 0.34 * wf * s + p.belly * 0.12 * s, 0, 0, 0, [1.4, 0.75, 0.6]));
      break;
    case 'toolbelt':
      for (const sx of [-1, 1]) L.cloth.add(paint(rigid(new CylinderGeometry(0.12 * s, 0.1 * s, 0.3 * s, 4), B.hips), '#8a5a2a'), M4(sx * 0.38 * wf * s, p.waistY - 0.2 * s, 0.18 * wf * s, 0, Math.PI / 4));
      break;
    case 'vest':
      for (const sx of [-1, 1]) {
        const panel = loft([{ y: p.waistY - 0.1 * s, rx: 0.2 * wf * s, rz: 0.03, cx: sx * 0.24 * wf * s, cz: 0.33 * wf * s + p.belly * 0.13 * s }, { y: p.shoulderY - 0.05 * s, rx: 0.17 * wf * s, rz: 0.03, cx: sx * 0.3 * wf * s, cz: 0.3 * wf * s }], 10);
        L.cloth.add(paint(blended(panel, (q) => { const w = ramp(q.y, p.waistY, p.chestY); return [B.spine, 1 - w, B.chest, w]; }), '#3a3f4a'));
      }
      break;
    case 'cape': {
      const cp = loft([{ y: p.hipY - 0.6 * s, rx: 0.65 * wf * s, rz: 0.05, cz: -0.42 * wf * s }, { y: p.shoulderY + 0.05 * s, rx: 0.42 * wf * s, rz: 0.04, cz: -0.3 * wf * s }], 14);
      L.cloth.add(paint(blended(cp, (q) => { const w = ramp(q.y, p.hipY, p.shoulderY); return [B.spine, 1 - w, B.chest, w]; }), '#c0392b'));
      break;
    }
    case 'overalls': {
      const bib = loft([{ y: p.waistY - 0.05 * s, rx: 0.26 * wf * s, rz: 0.02, cz: 0.34 * wf * s + p.belly * 0.13 * s }, { y: p.chestY + 0.15 * s, rx: 0.22 * wf * s, rz: 0.02, cz: 0.32 * wf * s }], 8);
      L.cloth.add(paint(blended(bib, (q) => { const w = ramp(q.y, p.waistY, p.chestY); return [B.spine, 1 - w, B.chest, w]; }), '#3a5f8f'));
      break;
    }
  }
}
