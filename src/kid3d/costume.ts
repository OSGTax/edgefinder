import { Color, CylinderGeometry, Euler, Matrix4, Quaternion, SphereGeometry, TorusGeometry, Vector3, type BufferGeometry } from 'three';
import type { Kid, Team } from '../data/types';
import { blended, limb, loft, paint, paintFn, ramp, rigid, type Ring } from './geom';
import { B, type Proportions } from './rig';
import { CAP_LOGO_UV, type UniformColors } from './uniform';
import type { Lists } from './model';
import { headCentre, seeded } from './model';

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

// ─────────────────────────────────────────────────────────────── hair

export function addHair(L: Lists, p: Proportions, kid: Kid) {
  const look = kid.look;
  const R = p.headR;
  const hc = headCentre(p);
  const at = (x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, s: number | [number, number, number] = 1) => M4(hc.x + x, hc.y + y, hc.z + z, rx, ry, rz, s);
  const H = (g: BufferGeometry, m: Matrix4) => L.hair.add(rigid(g, B.head), m);
  const rnd = seeded(kid.id + 'hair');
  const hatted = look.hat !== 'visor';
  // hair that falls below the head follows the chest
  const toChest = (g: BufferGeometry, m: Matrix4) => {
    g.applyMatrix4(m);
    L.hair.add(blended(g, (q) => { const w = ramp(q.y, p.neckY - 0.1, p.neckY + 0.25); return [B.chest, 1 - w, B.head, w]; }));
  };
  const scalp = (rk: number, T = 1.42, alpha = 0.68) => H(cap(R * rk, T, alpha), at(0, 0, 0));
  switch (look.hair) {
    case 'buzz':
      H(lumpy(cap(R * 1.025, 1.35, 0.7), 0.01, 40, 1), at(0, 0, 0));
      break;
    case 'sidepart':
      scalp(1.05);
      H(lumpy(new SphereGeometry(R * 0.55, 20, 12), 0.04, 12, 2), at(R * 0.25, R * 0.62, R * 0.22, 0.3, 0, -0.5, [1.25, 0.45, 0.9]));
      H(new SphereGeometry(R * 0.4, 16, 10), at(-R * 0.35, R * 0.7, R * 0.1, 0.2, 0, 0.4, [1.1, 0.4, 0.9]));
      break;
    case 'messy':
      scalp(1.06);
      for (let i = 0; i < 16; i++) {
        const a = rnd() * Math.PI * 2, el = hatted ? -0.1 + rnd() * 0.3 : 0.25 + rnd() * 0.9;
        const d = new Vector3(Math.cos(a) * Math.cos(el), Math.sin(el), Math.sin(a) * Math.cos(el) * 0.8 - 0.25);
        if (d.z > 0.55 && d.y < 0.6) continue;
        if (hatted && d.z > -0.1) continue; // under a cap only the back tufts show
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
        const el = hatted ? -0.15 + rnd() * 0.2 : 0.45 + rnd() * 0.7;
        const d = new Vector3(Math.cos(a) * Math.cos(el), Math.sin(el), Math.sin(a) * Math.cos(el) * 0.9 - 0.15).normalize();
        if (d.z > 0.7 && d.y < 0.4) continue;
        if (hatted && d.z > -0.25) continue; // spikes stick out the back under a cap
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
      for (let i = 0; i < 70; i++) {
        const u = new Vector3(rnd() * 2 - 1, rnd() * 1.2 - 0.2, rnd() * 2 - 1.2);
        if (u.lengthSq() > 1.4 || u.lengthSq() < 0.05) continue;
        u.normalize();
        if (u.z > 0.45 && u.y < 0.55) continue;
        if (u.y < -0.15) continue;
        H(new SphereGeometry(R * (0.15 + rnd() * 0.07), 9, 7), at(u.x * R * 1.08, u.y * R * 1.08, u.z * R * 1.08));
      }
      break;
    }
    case 'afro':
      H(lumpy(new SphereGeometry(R * 1.3, 30, 22), 0.05, 9, 3), at(0, R * 0.3, -R * 0.12, 0, 0, 0, [1, 0.9, 0.95]));
      break;
    case 'ponytail':
    case 'bun':
      scalp(1.05);
      if (look.hair === 'bun') H(lumpy(new SphereGeometry(R * 0.38, 16, 12), 0.05, 14, 4), at(0, R * 0.95, -R * 0.45));
      else {
        H(new TorusGeometry(R * 0.12, R * 0.05, 6, 12), at(0, R * 0.35, -R * 1.02, 0.4));
        toChest(limb(R * 1.3, R * 0.2, R * 0.08, 10), at(0, R * 0.35, -R * 1.08, -0.35));
      }
      break;
    case 'pigtails':
    case 'braids': {
      scalp(1.05);
      for (const sx of [-1, 1]) {
        H(new SphereGeometry(R * 0.11, 8, 6), at(sx * R * 0.92, -R * 0.05, -R * 0.2));
        if (look.hair === 'pigtails') {
          toChest(lumpy(limb(R * 0.85, R * 0.2, R * 0.1, 10), 0.04, 20, sx), at(sx * R * 0.98, -R * 0.12, -R * 0.3, 0.25, 0, sx * 0.22));
        } else {
          for (let k = 0; k < 7; k++) {
            const g = new SphereGeometry(R * (0.15 - k * 0.008), 10, 8);
            toChest(g, at(sx * R * (0.9 + k * 0.02), -R * (0.2 + k * 0.24), -R * (0.15 - k * 0.04), 0, 0, 0, [1, 1.25, 1]));
          }
          toChest(new SphereGeometry(R * 0.08, 8, 6), at(sx * R * 1.04, -R * 1.95, R * 0.12));
        }
      }
      break;
    }
    case 'bob':
    case 'long': {
      scalp(1.07, 1.45, 0.7);
      const long = look.hair === 'long';
      // curtain of hair around the sides and back, open at the face
      const curtain = new SphereGeometry(R * 1.12, 34, 18, Math.PI / 2 + 0.95, Math.PI * 2 - 1.9, 0.25, long ? Math.PI * 0.62 : Math.PI * 0.55);
      H(lumpy(curtain, 0.02, 18, 5), at(0, 0, -R * 0.02));
      if (long) {
        const sheet = loft([
          { y: -R * 2.1, rx: R * 0.75, rz: R * 0.25, cz: -R * 0.55 },
          { y: -R * 1.2, rx: R * 0.95, rz: R * 0.35, cz: -R * 0.65 },
          { y: -R * 0.4, rx: R * 1.08, rz: R * 0.55, cz: -R * 0.55 },
        ], 20, true, false);
        toChest(sheet, at(0, 0, 0));
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
  const lift = look.hair === 'afro' ? R * 0.38 : 0;
  const at = (x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, s: number | [number, number, number] = 1) => M4(hc.x + x, hc.y + y + lift, hc.z + z, rx, ry, rz, s);
  const C = (g: BufferGeometry, hex: string | Color, m: Matrix4) => L.cloth.add(paint(rigid(g, B.head), hex), m);
  const brim = (w: number, d: number, curve: number) => {
    // a curved bill: a lofted half-disc in front
    const rings: Ring[] = [];
    const g = new SphereGeometry(1, 24, 6, 0, Math.PI, Math.PI / 2 - 0.02, 0.04);
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      pos.setXYZ(i, x * w, y * 0.4 - (x * x) * curve, -z * d);
    }
    g.computeVertexNormals();
    void rings;
    return g;
  };
  switch (look.hat) {
    case 'cap':
    case 'capBack':
    case 'trucker': {
      const back = look.hat === 'capBack' ? Math.PI : 0;
      const crownCol = look.hat === 'trucker' ? '#f2efe6' : col.cap;
      const crown = cap(R * 1.12, 1.2, 0.18, 36, 14);
      crown.scale(1, look.hat === 'trucker' ? 1.18 : 1.02, 1.03);
      const cTint = new Color(crownCol), mesh = new Color(look.hat === 'trucker' ? '#3f6b3a' : col.cap);
      L.cloth.add(paintFn(rigid(crown, B.head), (q) => (look.hat === 'trucker' && q.z < R * 0.25 ? mesh : cTint)), at(0, 0.02, 0, 0, back));
      // button + bill
      C(new SphereGeometry(R * 0.08, 10, 8), col.cap, at(0, R * (look.hat === 'trucker' ? 1.33 : 1.15), -R * 0.1, 0, back));
      const bill = brim(R * 0.78, R * 0.95, 0.35 / R);
      const bm = at(0, R * 0.36, 0, 0.12, back).multiply(new Matrix4().makeTranslation(0, 0, R * 0.98));
      C(bill, col.brim, bm);
      if (look.hat !== 'trucker') {
        // team logo on the front panel (uses the cap-logo corner of the jersey texture)
        const logo = new SphereGeometry(R * 1.125, 12, 8, Math.PI / 2 - 0.42, 0.84, 0.55, 0.62);
        const uv = logo.attributes.uv;
        for (let i = 0; i < uv.count; i++) {
          uv.setXY(i, CAP_LOGO_UV.u0 + uv.getX(i) * (CAP_LOGO_UV.u1 - CAP_LOGO_UV.u0), CAP_LOGO_UV.v0 + (1 - uv.getY(i)) * (CAP_LOGO_UV.v1 - CAP_LOGO_UV.v0));
        }
        logo.applyMatrix4(new Matrix4().makeRotationX(-0.18));
        L.jersey.add(rigid(logo, B.head), at(0, 0.02, 0.003, 0, back, 0, [1, 1.02, 1.03]));
      } else {
        // trucker patch
        const patch = new SphereGeometry(R * 1.13, 10, 6, Math.PI / 2 - 0.35, 0.7, 0.55, 0.45);
        L.cloth.add(paint(rigid(patch, B.head), '#c0392b'), at(0, 0.02, 0, 0, 0, 0, [1, 1.18, 1.03]));
      }
      break;
    }
    case 'visor': {
      const band = loft([{ y: -0.12, rx: R * 1.06, rz: R * 1.06 }, { y: 0.12, rx: R * 1.07, rz: R * 1.07 }], 32);
      C(band, '#f4f4f0', at(0, R * 0.42, -0.02, 0.18));
      C(brim(R * 0.8, R * 0.95, 0.3 / R), col.trim, at(0, R * 0.36, 0, 0.18).multiply(new Matrix4().makeTranslation(0, 0, R * 1.0)));
      break;
    }
    case 'bucket': {
      const crown = loft([{ y: 0, rx: R * 1.13, rz: R * 1.13 }, { y: R * 0.55, rx: R * 1.02, rz: R * 1.02 }, { y: R * 0.75, rx: R * 0.85, rz: R * 0.85 }], 32, false, true);
      C(crown, '#c9b98a', at(0, R * 0.32, -0.02, 0.1));
      const rim = loft([{ y: -R * 0.32, rx: R * 1.65, rz: R * 1.65 }, { y: 0, rx: R * 1.14, rz: R * 1.14 }], 32);
      C(rim, '#bfae7e', at(0, R * 0.34, -0.02, 0.1));
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
  const atH = (x: number, y: number, z: number, rx = 0, ry = 0, rz = 0, sc: number | [number, number, number] = 1) => M4(hc.x + x, hc.y + y, hc.z + z, rx, ry, rz, sc);
  const head = (list: 'cloth' | 'shiny' | 'hair', g: BufferGeometry, hex: string | null, m: Matrix4) => {
    rigid(g, B.head);
    if (hex) paint(g, hex);
    L[list].add(g, m);
  };
  const chestZ = 0.33 * wf * s + p.belly * 0.12 * s; // front of the jersey at chest height
  const onChest = (list: 'cloth' | 'shiny', g: BufferGeometry, hex: string, m: Matrix4) => {
    g.applyMatrix4(m);
    paint(g, hex);
    L[list].add(blended(g, (q) => { const w = ramp(q.y, p.waistY, p.chestY); return [B.spine, 1 - w, B.chest, w]; }));
  };

  // ── facial hair worn as stick-ons
  if (look.face === 'walrus' || look.face === 'handlebar') {
    for (const sx of [-1, 1]) {
      if (look.face === 'walrus') {
        const g = lumpy(new SphereGeometry(R * 0.2, 14, 10), 0.08, 30, sx);
        head('hair', g, null, atH(sx * R * 0.17, -R * 0.33, R * 0.93, 0.2, 0, sx * 0.5, [1.35, 0.55, 0.55]));
        head('hair', lumpy(new SphereGeometry(R * 0.13, 10, 8), 0.1, 30, sx + 2), null, atH(sx * R * 0.33, -R * 0.45, R * 0.82, 0, 0, sx * 0.4, [0.8, 1.1, 0.6]));
      } else {
        const g = limb(R * 0.38, R * 0.06, R * 0.03, 8);
        head('hair', g, null, atH(sx * R * 0.04, -R * 0.3, R * 0.97, 0, 0, sx * -1.2));
        head('hair', new TorusGeometry(R * 0.07, R * 0.025, 6, 12, Math.PI * 1.3), null, atH(sx * R * 0.43, -R * 0.24, R * 0.86, 0, sx * 0.6, sx > 0 ? -0.3 : Math.PI + 0.3));
      }
    }
  }
  if (look.face === 'beard') {
    // a cut-up yellow sponge on a string
    const sponge = new SphereGeometry(R * 0.62, 18, 12, 0, Math.PI * 2, Math.PI * 0.45, Math.PI * 0.55);
    paintFn(lumpy(sponge, 0.05, 25, 7), (q) => new Color(Math.sin(q.x * 90) * Math.sin(q.y * 80) > 0.6 ? '#c7a92c' : '#e9cf4a'));
    L.cloth.add(rigid(sponge, B.head), atH(0, -R * 0.38, R * 0.42, -0.25, 0, 0, [1.25, 1.1, 0.95]));
    head('cloth', loft([{ y: 0, rx: R * 1.03, rz: R * 1.03 }, { y: 0.02, rx: R * 1.03, rz: R * 1.03 }], 24), '#ddd6c2', atH(0, -R * 0.05, 0, 0.6));
  }

  // ── eyewear
  const eyeY = p.eyeY - hc.y, eyeX = p.eyeX, eyeZ = p.eyeZ - hc.z;
  const frame = (hex: string, lens: string | null, round: number, size: number) => {
    for (const sx of [-1, 1]) {
      const ring = new TorusGeometry(p.eyeR * size, p.eyeR * 0.1, 6, 20);
      ring.scale(1, round, 1);
      head('shiny', ring, hex, atH(sx * eyeX, eyeY, eyeZ + p.eyeR * 0.95));
      if (lens) {
        const l = new SphereGeometry(p.eyeR * size, 16, 10, 0, Math.PI * 2, 0, 0.7);
        l.rotateX(Math.PI / 2);
        l.scale(1, round, 0.35);
        head('shiny', l, lens, atH(sx * eyeX, eyeY, eyeZ + p.eyeR * 0.7));
      }
      // arm back to the ear
      head('shiny', limb(R * 0.75, p.eyeR * 0.08, p.eyeR * 0.08, 5), hex, atH(sx * (eyeX + p.eyeR * size * 0.95), eyeY + p.eyeR * 0.1, eyeZ + p.eyeR * 0.8, Math.PI / 2 - 0.15, 0, 0));
    }
    head('shiny', limb(eyeX * 2 - p.eyeR * size * 2, p.eyeR * 0.08, p.eyeR * 0.08, 5), hex, atH(-(eyeX - p.eyeR * size), eyeY + p.eyeR * 0.15, eyeZ + p.eyeR * 1.0, 0, 0, Math.PI / 2));
  };
  switch (look.eyewear) {
    case 'glasses': frame('#2a2a2a', null, 1, 1.25); break;
    case 'reading': frame('#7a3b2a', null, 0.7, 1.15);
      // chain
      for (const sx of [-1, 1]) L.cloth.add(paint(blended(limb(R * 1.2, 0.012, 0.012, 4).applyMatrix4(atH(sx * R * 0.85, eyeY, eyeZ * 0.6, 0.3, 0, sx * 0.1)), (q) => { const w = ramp(q.y, p.neckY, hc.y - R * 0.3); return [B.chest, 1 - w, B.head, w]; }), '#d4b44a'));
      break;
    case 'shades': frame('#151515', '#101418', 0.82, 1.32); break;
    case 'aviators': frame('#d4b44a', '#3a2a1a', 1.1, 1.3); break;
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
      onChest('cloth', sc, scol, M4(0, neckY - 0.02, 0.01));
      onChest('cloth', limb(0.75 * s, 0.09 * s, 0.1 * s, 8, 0.35), scol, M4(0.12 * s, neckY - 0.08, chestZ + 0.03, -0.1, 0, 0.12));
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
      head('cloth', new TorusGeometry(R * 1.1, R * 0.05, 6, 30, Math.PI), '#2a2a2a', atH(0, R * 0.05, -R * 0.05, 0, 0, 0));
      for (const sx of [-1, 1]) head('cloth', new CylinderGeometry(R * 0.2, R * 0.2, R * 0.12, 14), '#2a2a2a', atH(sx * R * 1.05, 0, 0, 0, 0, Math.PI / 2));
      head('cloth', limb(R * 0.8, R * 0.025, R * 0.025, 5), '#2a2a2a', atH(earX, -R * 0.05, R * 0.05, -1.25, 0, 0.35));
      head('cloth', new SphereGeometry(R * 0.07, 8, 6), '#111', atH(R * 0.42, -R * 0.38, R * 0.82));
      break;
    case 'earpiece':
      head('cloth', new SphereGeometry(R * 0.07, 8, 6), '#e6e6e0', atH(-earX * 1.02, 0, R * 0.04));
      head('cloth', limb(R * 1.2, R * 0.015, R * 0.015, 4), '#e6e6e0', atH(-earX * 1.0, -R * 0.06, 0, 0.25, 0, 0));
      break;
    case 'pencil':
      head('cloth', new CylinderGeometry(R * 0.04, R * 0.04, R * 0.8, 6), '#f2c94c', atH(earX * 1.02, R * 0.12, -R * 0.05, 0, 0, Math.PI / 2 - 0.5));
      head('cloth', new CylinderGeometry(0.0, R * 0.04, R * 0.1, 6), '#e8c49a', atH(earX * 1.02 - R * 0.3, R * 0.3, -R * 0.05, 0, 0, Math.PI / 2 - 0.5 + Math.PI));
      break;
    case 'sweatband':
    case 'headband':
      head('cloth', loft([{ y: -R * 0.1, rx: R * 1.05, rz: R * 1.05 }, { y: R * 0.1, rx: R * 1.04, rz: R * 1.04 }], 32), look.extra === 'sweatband' ? '#f4f4f0' : '#e85d75', atH(0, R * 0.45, -R * 0.02, 0.2));
      if (look.extra === 'sweatband') {
        for (const sx of [1, -1]) {
          const bone = sx === 1 ? B.foreL : B.foreR;
          const wr = sx === 1 ? p.joints.handL : p.joints.handR;
          const el = sx === 1 ? p.joints.foreL : p.joints.foreR;
          const g = loft([{ y: -0.07 * s, rx: p.armR * 0.95, rz: p.armR * 0.95 }, { y: 0.07 * s, rx: p.armR, rz: p.armR }], 14);
          const pos = el.clone().lerp(wr, 0.82);
          const q = new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), el.clone().sub(wr).normalize());
          L.cloth.add(paint(rigid(g, bone), '#f4f4f0'), new Matrix4().compose(pos, q, new Vector3(1, 1, 1)));
        }
      }
      break;
    case 'bow':
      for (const sx of [-1, 1]) head('cloth', new SphereGeometry(R * 0.22, 10, 8), '#ff6fae', atH(sx * R * 0.24 + R * 0.35, R * 0.85, -R * 0.15, 0, 0, sx * 0.5, [1.3, 0.7, 0.45]));
      head('cloth', new SphereGeometry(R * 0.1, 8, 6), '#e0559a', atH(R * 0.35, R * 0.85, -R * 0.12));
      break;
    case 'curlers':
      for (let i = 0; i < 6; i++) head('cloth', new CylinderGeometry(R * 0.11, R * 0.11, R * 0.35, 10), ['#ff8fb1', '#8fd3ff', '#ffe38f'][i % 3], atH(-R * 0.5 + (i % 3) * R * 0.5, R * (0.85 - Math.floor(i / 3) * 0.3), -R * (0.1 + Math.floor(i / 3) * 0.5), 0, 0, Math.PI / 2));
      break;
    case 'earrings':
      for (const sx of [-1, 1]) head('shiny', new TorusGeometry(R * 0.07, R * 0.015, 6, 14), '#d4b44a', atH(sx * earX, -R * 0.24, -R * 0.02));
      break;
    case 'flower':
      for (let i = 0; i < 5; i++) { const a = (i / 5) * Math.PI * 2; head('cloth', new SphereGeometry(R * 0.08, 8, 6), '#ffffff', atH(R * 0.7 + Math.cos(a) * R * 0.08, R * 0.6 + Math.sin(a) * R * 0.08, R * 0.45)); }
      head('cloth', new SphereGeometry(R * 0.06, 8, 6), '#f2c94c', atH(R * 0.7, R * 0.6, R * 0.5));
      break;
    case 'bandaid':
      head('cloth', new SphereGeometry(R * 0.12, 10, 6), '#e8c09a', atH(-R * 0.45, R * 0.35, R * 0.86, 0, 0.3, 0.4, [1.5, 0.5, 0.25]));
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
      const ap = loft([{ y: p.hipY - 0.5 * s, rx: 0.42 * wf * s, rz: 0.05, cz: 0.36 * wf * s + p.belly * 0.12 * s }, { y: p.chestY + 0.15 * s, rx: 0.32 * wf * s, rz: 0.05, cz: 0.36 * wf * s + p.belly * 0.08 * s }], 12);
      ap.applyMatrix4(new Matrix4());
      L.cloth.add(paint(blended(ap, (q) => { const w = ramp(q.y, p.hipY, p.chestY); return [B.hips, 1 - w, B.chest, w]; }), '#f6f1e6'));
      onChest('cloth', loft([{ y: 0, rx: 0.46 * wf * s, rz: 0.34 * wf * s + p.belly * 0.12 * s }, { y: 0.06, rx: 0.46 * wf * s, rz: 0.34 * wf * s + p.belly * 0.12 * s }], 24), '#f6f1e6', M4(0, p.waistY + 0.1, 0));
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
