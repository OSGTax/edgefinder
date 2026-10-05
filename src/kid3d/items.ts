import {
  CanvasTexture, Group, LatheGeometry, Mesh, MeshStandardMaterial, SphereGeometry, SRGBColorSpace, TorusGeometry,
  Vector2, CylinderGeometry, type BufferGeometry, type Material,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { Holding } from '../data/types';
import { leatherTex } from '../gfx/textures';
import { paint } from './geom';

// Things kids hold: bats, gloves, the ball, and the persona props they carry
// around between plays (coffee mugs, clipboards, microphones...).

const mats = new Map<string, Material>();
const mat = (key: string, make: () => Material) => { let m = mats.get(key); if (!m) { m = make(); mats.set(key, m); } return m; };
const vc = () => mat('itemVC', () => new MeshStandardMaterial({ vertexColors: true, roughness: 0.5 }));

/** A wooden (or aluminium) kid's bat; its handle knob is at the origin, barrel along +y. */
export function makeBat(kind: 'wood' | 'metal' = 'wood', len = 2.55): Mesh {
  const pts: Vector2[] = [];
  const prof: [number, number][] = [[0, 0], [0.075, 0], [0.085, 0.03], [0.05, 0.07], [0.045, 0.25], [0.05, 0.7], [0.075, 1.15], [0.105, 1.6], [0.115, 2.1], [0.115, 2.42], [0.1, 2.52], [0, len]];
  for (const [r, y] of prof) pts.push(new Vector2(r, (y / 2.55) * len));
  const g = new LatheGeometry(pts, 18);
  const m = kind === 'wood'
    ? mat('batWood', () => new MeshStandardMaterial({ color: '#d9b27c', roughness: 0.45 }))
    : mat('batMetal', () => new MeshStandardMaterial({ color: '#c9ced4', roughness: 0.25, metalness: 0.9 }));
  const bat = new Mesh(g, m);
  // grip tape
  const tape = new Mesh(new CylinderGeometry(0.056, 0.052, 0.55, 14, 1, true).translate(0, 0.38, 0), mat('batTape', () => new MeshStandardMaterial({ color: '#2a2a2a', roughness: 0.8 })));
  bat.add(tape);
  bat.castShadow = true;
  tape.castShadow = true;
  return bat;
}

/** A leather mitt; the hand goes in at the origin, the pocket faces +z. */
export function makeGlove(s = 1): Group {
  const g = new Group();
  const lt = leatherTex(256);
  const leather = mat('glove', () => new MeshStandardMaterial({ map: lt.map, normalMap: lt.normal, roughness: 0.55, color: '#c98a4a' }));
  const dark = mat('gloveDark', () => new MeshStandardMaterial({ color: '#6b3d1e', roughness: 0.6 }));
  const palm = new SphereGeometry(0.33 * s, 18, 14);
  palm.scale(1, 1.25, 0.45);
  const pocket = new Mesh(palm, leather);
  pocket.position.set(0, -0.28 * s, 0.05 * s);
  g.add(pocket);
  // fingers (four fat stalls) and the thumb
  for (let i = 0; i < 4; i++) {
    const f = new Mesh(new SphereGeometry(0.1 * s, 10, 8).scale(1, 2.4, 0.9), leather);
    f.position.set((-0.2 + i * 0.13) * s, -0.62 * s, 0.03 * s);
    f.rotation.z = (i - 1.5) * 0.1;
    g.add(f);
  }
  const th = new Mesh(new SphereGeometry(0.1 * s, 10, 8).scale(1, 2.0, 0.9), leather);
  th.position.set(0.3 * s, -0.28 * s, 0.1 * s);
  th.rotation.z = -0.7;
  g.add(th);
  // webbing + lacing + wrist strap
  const web = new Mesh(new SphereGeometry(0.12 * s, 8, 6).scale(1, 1.4, 0.4), dark);
  web.position.set(0.22 * s, -0.52 * s, 0.08 * s);
  g.add(web);
  const strap = new Mesh(new TorusGeometry(0.13 * s, 0.035 * s, 6, 14), dark);
  strap.rotation.x = Math.PI / 2;
  strap.position.set(0, -0.02 * s, 0);
  g.add(strap);
  g.traverse((o) => { o.castShadow = true; });
  return g;
}

let ballTexCache: CanvasTexture | null = null;
function ballTexture(): CanvasTexture {
  if (ballTexCache) return ballTexCache;
  const c = document.createElement('canvas');
  c.width = 256; c.height = 128;
  const g = c.getContext('2d')!;
  g.fillStyle = '#f7f3ea';
  g.fillRect(0, 0, 256, 128);
  // grass-stained, scuffed backyard ball
  for (let i = 0; i < 14; i++) { g.fillStyle = `rgba(110,140,60,${0.06 + (i % 3) * 0.03})`; g.beginPath(); g.ellipse((i * 53) % 256, (i * 29) % 128, 18, 9, i, 0, 7); g.fill(); }
  g.strokeStyle = '#c8332b';
  g.lineWidth = 3;
  for (const off of [0, 128]) {
    g.beginPath();
    for (let x = 0; x <= 128; x += 2) {
      const y = 64 + Math.sin((x / 128) * Math.PI * 2) * 34;
      if (x === 0) g.moveTo(x + off, y); else g.lineTo(x + off, y);
    }
    g.stroke();
    for (let x = 4; x < 128; x += 8) {
      const y = 64 + Math.sin((x / 128) * Math.PI * 2) * 34;
      g.beginPath(); g.moveTo(x + off - 3, y - 5); g.lineTo(x + off + 3, y - 1); g.stroke();
      g.beginPath(); g.moveTo(x + off - 3, y + 5); g.lineTo(x + off + 3, y + 1); g.stroke();
    }
  }
  ballTexCache = new CanvasTexture(c);
  ballTexCache.colorSpace = SRGBColorSpace;
  return ballTexCache;
}

export function makeBall(r = 0.12): Mesh {
  const m = new Mesh(new SphereGeometry(r, 18, 12), mat('ball', () => new MeshStandardMaterial({ map: ballTexture(), roughness: 0.55 })));
  m.castShadow = true;
  return m;
}

/** Persona props, held in the right hand (grip at the origin). */
export function makeProp(kind: Holding): Group | null {
  const parts: BufferGeometry[] = [];
  const add = (g: BufferGeometry, hex: string, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0) => {
    g.rotateX(rx); g.rotateY(ry); g.rotateZ(rz); g.translate(x, y, z);
    parts.push(paint(g.index ? g.toNonIndexed() : g, hex));
  };
  const box = (w: number, h: number, d: number) => new CylinderGeometry(Math.SQRT1_2, Math.SQRT1_2, 1, 4, 1).rotateY(Math.PI / 4).scale(w, h, d);
  switch (kind) {
    case 'coffee':
      add(new CylinderGeometry(0.13, 0.11, 0.3, 14), '#f4f1e8', 0, 0.05, 0.12);
      add(new TorusGeometry(0.07, 0.02, 6, 12), '#f4f1e8', 0.14, 0.05, 0.12, 0, 0, 0);
      add(new CylinderGeometry(0.12, 0.12, 0.02, 14), '#4a2a14', 0, 0.19, 0.12);
      break;
    case 'clipboard':
      add(box(0.75, 1.0, 0.04), '#a8774a', 0, 0.1, 0.1, 0.3);
      add(box(0.62, 0.8, 0.045), '#fbfaf3', 0, 0.07, 0.115, 0.3);
      add(box(0.25, 0.1, 0.08), '#9aa0a6', 0, 0.55, 0.0, 0.3);
      break;
    case 'briefcase':
      add(box(1.1, 0.8, 0.3), '#4a2e1a', 0, -0.5, 0);
      add(new TorusGeometry(0.12, 0.03, 6, 12, Math.PI), '#2a1a10', 0, -0.07, 0);
      add(box(0.1, 0.08, 0.32), '#d4b44a', 0.3, -0.15, 0);
      add(box(0.1, 0.08, 0.32), '#d4b44a', -0.3, -0.15, 0);
      break;
    case 'microphone':
      add(new CylinderGeometry(0.05, 0.035, 0.45, 10), '#2a2a2a', 0, 0.1, 0.08);
      add(new SphereGeometry(0.1, 12, 10), '#b0b5ba', 0, 0.38, 0.08);
      break;
    case 'newspaper':
      add(box(0.6, 0.85, 0.05), '#ece8dc', 0, 0.2, 0.1, 0.2, 0, 0.2);
      break;
    case 'calculator':
      add(box(0.42, 0.6, 0.06), '#3a3f45', 0, 0.12, 0.12, 0.6);
      add(box(0.32, 0.14, 0.065), '#a8c79a', 0, 0.3, 0.14, 0.6);
      break;
    case 'binoculars':
      add(new CylinderGeometry(0.09, 0.1, 0.4, 12), '#2a2a2a', -0.1, 0.1, 0.12, Math.PI / 2);
      add(new CylinderGeometry(0.09, 0.1, 0.4, 12), '#2a2a2a', 0.1, 0.1, 0.12, Math.PI / 2);
      break;
    case 'gavel':
      add(new CylinderGeometry(0.03, 0.03, 0.8, 8), '#7a4a2a', 0, 0.25, 0.05);
      add(new CylinderGeometry(0.11, 0.11, 0.4, 12), '#6a3a1a', 0, 0.62, 0.05, 0, 0, Math.PI / 2);
      break;
    case 'rollingPin':
      add(new CylinderGeometry(0.11, 0.11, 1.2, 14), '#e2bd7a', 0, 0.3, 0.06, 0, 0, Math.PI / 2 - 0.3);
      add(new CylinderGeometry(0.04, 0.04, 1.6, 8), '#c99a5a', 0, 0.3, 0.06, 0, 0, Math.PI / 2 - 0.3);
      break;
    case 'wrench':
      add(box(0.08, 1.0, 0.04), '#9aa0a6', 0, 0.3, 0.05);
      add(new TorusGeometry(0.11, 0.04, 6, 12, Math.PI * 1.5), '#9aa0a6', 0, 0.85, 0.05);
      break;
    case 'juicebox':
      add(box(0.25, 0.38, 0.16), '#ff8a3d', 0, 0.1, 0.1);
      add(new CylinderGeometry(0.01, 0.01, 0.25, 4), '#ffffff', 0.05, 0.38, 0.1, 0, 0, 0.3);
      break;
    case 'phone':
      add(box(0.25, 0.48, 0.04), '#1b1b1b', 0, 0.12, 0.1, 0.3);
      break;
    default:
      return null;
  }
  const merged = mergeGeometries(parts, false);
  if (!merged) return null;
  const grp = new Group();
  const mesh = new Mesh(merged, vc());
  mesh.castShadow = true;
  grp.add(mesh);
  return grp;
}

