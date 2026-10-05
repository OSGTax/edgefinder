import { Color, Group, Mesh, MeshStandardMaterial, PlaneGeometry, Vector2, type IUniform } from 'three';
import { Batch, T, boxFt, cyl, sphere, torus, tube } from '../gfx/build';
import { M } from '../gfx/materials';
import { paintTex, waterNormal } from '../gfx/textures';
import { W } from '../gfx/units';

// The Mendozas' in-ground pool: coping, deck, tiled walls with a waterline
// band, steps, ladder, diving board, animated water with caustics, floaties.
// Built in a local frame (x along the length, z across, y up), then placed.

export interface PoolSpec { x: number; y: number; hw: number; hd: number; rot: number }

export const POOL_DEPTH_SHALLOW = 3.5;
export const POOL_DEPTH_DEEP = 8;
const WATER_Y = -0.55;

const CAUSTIC_GLSL = /* glsl */ `
uniform float uTime;
float pwave(vec2 p, float t) {
  return sin(p.x * 1.7 + sin(p.y * 1.3 + t) * 1.4 + t * 0.9) + sin(p.y * 1.9 + sin(p.x * 1.1 - t * 1.1) * 1.6 - t * 0.7);
}
float caustic(vec2 p, float t) {
  float a = 1.0 - abs(pwave(p, t)) * 0.5;
  float b = 1.0 - abs(pwave(p * 1.7 + 4.0, t * 1.3)) * 0.5;
  return pow(max(a, 0.0), 7.0) * 0.6 + pow(max(b, 0.0), 9.0) * 0.5;
}`;

/** Interior surfaces: tinted bluer with depth and lit by moving caustics. */
function interiorMaterial(base: MeshStandardMaterial, time: IUniform, localY0: number) {
  const m = base.clone();
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = time;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWP;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWP = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\nvarying vec3 vWP;\n${CAUSTIC_GLSL}`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
float depthBelow = clamp((${localY0.toFixed(2)} - vWP.y) / 8.0, 0.0, 1.0);
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.05, 0.36, 0.52), depthBelow * 0.65);
float cst = caustic(vWP.xz * 0.55, uTime * 1.2) * (1.0 - depthBelow * 0.5);
totalEmissiveRadiance += vec3(0.55, 0.85, 0.95) * cst * 0.55 * step(vWP.y, ${localY0.toFixed(2)});`);
  };
  m.customProgramCacheKey = () => `poolInterior${localY0}`;
  return m;
}

function waterMaterial(time: IUniform) {
  const n = waterNormal(256);
  const m = new MeshStandardMaterial({
    color: new Color('#3fb9d6'), roughness: 0.04, metalness: 0.05, transparent: true, opacity: 0.62,
    normalMap: n, normalScale: new Vector2(0.35, 0.35), envMapIntensity: 1.4, depthWrite: false,
  });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = time;
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uTime;')
      .replace('#include <normal_fragment_maps>', `
vec3 n1 = texture2D(normalMap, vNormalMapUv * 1.0 + vec2(uTime * 0.021, uTime * 0.013)).xyz * 2.0 - 1.0;
vec3 n2 = texture2D(normalMap, vNormalMapUv * 1.9 + vec2(-uTime * 0.017, uTime * 0.026)).xyz * 2.0 - 1.0;
vec3 mapN = normalize(n1 + n2);
mapN.xy *= normalScale;
normal = normalize(tbn * mapN);`)
      // more opaque and reflective at grazing angles
      .replace('#include <opaque_fragment>', `
float fres = pow(1.0 - clamp(dot(normalize(vViewPosition), -normal) * -1.0, 0.0, 1.0), 3.0);
diffuseColor.a = mix(diffuseColor.a, 0.95, fres);
#include <opaque_fragment>`);
  };
  m.customProgramCacheKey = () => 'poolWater';
  return m;
}

export interface Pool {
  group: Group;
  update(t: number): void;
  /** world-space (sim) position of the bobbing floats, for splashes/ball rest */
  floats: Group;
}

export function buildPool(s: PoolSpec): Pool {
  const time: IUniform = { value: 0 };
  const g = new Group();
  g.name = 'pool';
  g.position.copy(W(s.x, s.y, 0));
  g.rotation.y = s.rot;
  const { hw, hd } = s;
  const b = new Batch();

  // deck + coping (the lawn has a hole cut out under the deck)
  const deck = M.concrete('#d8cfbf');
  const coping = M.concrete('#efe6d6');
  const dw = 5, dEnd = 9; // deck widths: sides / the lounging end (−x)
  const deckY = 0.12;
  // four deck strips around the coping
  const cw = 1.4;
  b.add(deck, boxFt(hw * 2 + cw * 2 + dw + dEnd, deckY + 0.3, dw), T((-dEnd + dw) / 2, deckY / 2 - 0.15, hd + cw + dw / 2));
  b.add(deck, boxFt(hw * 2 + cw * 2 + dw + dEnd, deckY + 0.3, dw), T((-dEnd + dw) / 2, deckY / 2 - 0.15, -hd - cw - dw / 2));
  b.add(deck, boxFt(dEnd, deckY + 0.3, hd * 2 + cw * 2), T(-hw - cw - dEnd / 2, deckY / 2 - 0.15, 0));
  b.add(deck, boxFt(dw, deckY + 0.3, hd * 2 + cw * 2), T(hw + cw + dw / 2, deckY / 2 - 0.15, 0));
  // bull-nosed coping stones, slightly proud of the deck
  const cy = deckY + 0.08;
  for (const sz of [-1, 1]) {
    const n = Math.round((hw * 2 + cw * 2) / 2);
    for (let i = 0; i < n; i++) {
      const L = (hw * 2 + cw * 2) / n;
      b.add(coping, boxFt(L - 0.04, 0.32, cw), T(-hw - cw + L * (i + 0.5), cy - 0.08, sz * (hd + cw / 2)));
    }
  }
  for (const sx of [-1, 1]) {
    const n = Math.round((hd * 2) / 2);
    for (let i = 0; i < n; i++) {
      const L = (hd * 2) / n;
      b.add(coping, boxFt(cw, 0.32, L - 0.04), T(sx * (hw + cw / 2), cy - 0.08, -hd + L * (i + 0.5)));
    }
  }
  // expansion joints in the deck
  for (let x = -hw - cw - dEnd + 6; x < hw + cw + dw; x += 6) {
    for (const sz of [-1, 1]) b.add(M.paint('#8f897e', 0.95), boxFt(0.06, 0.02, dw), T(x, deckY + 0.01, sz * (hd + cw + dw / 2)), { castShadow: false });
  }

  // ── interior: walls and a floor that slopes from the shallow (+x) end to the deep (−x) end
  const plaster = interiorMaterial(new MeshStandardMaterial({ color: '#dff1f6', roughness: 0.55 }), time, WATER_Y);
  const tileMat = M.poolTile() as MeshStandardMaterial;
  const tileI = interiorMaterial(tileMat, time, WATER_Y);
  const depthAt = (x: number) => {
    const t = (x + hw) / (hw * 2); // 0 at deep end
    return t < 0.35 ? POOL_DEPTH_DEEP : t > 0.6 ? POOL_DEPTH_SHALLOW : POOL_DEPTH_DEEP + (POOL_DEPTH_SHALLOW - POOL_DEPTH_DEEP) * ((t - 0.35) / 0.25);
  };
  const segs = 24;
  for (let i = 0; i < segs; i++) {
    const x0 = -hw + (i / segs) * hw * 2, x1 = -hw + ((i + 1) / segs) * hw * 2;
    const d0 = depthAt(x0), d1 = depthAt(x1);
    const xm = (x0 + x1) / 2, dm = (d0 + d1) / 2;
    // floor tile
    const len = Math.hypot(x1 - x0, d1 - d0);
    b.add(plaster, boxFt(len + 0.02, 0.2, hd * 2), T(xm, -dm - 0.1, 0, 0, 0, Math.atan2(d0 - d1, x1 - x0)));
    // long walls
    for (const sz of [-1, 1]) b.add(plaster, boxFt(x1 - x0 + 0.01, Math.max(d0, d1) + 0.2, 0.2), T(xm, -Math.max(d0, d1) / 2 + 0.1, sz * (hd + 0.1)));
  }
  for (const sx of [-1, 1]) {
    const d = depthAt(sx * hw);
    b.add(plaster, boxFt(0.2, d + 0.2, hd * 2), T(sx * (hw + 0.1), -d / 2 + 0.1, 0));
  }
  // waterline tile band
  for (const sz of [-1, 1]) b.add(tileI, boxFt(hw * 2, 0.55, 0.05), T(0, -0.2, sz * (hd - 0.01)));
  for (const sx of [-1, 1]) b.add(tileI, boxFt(0.05, 0.55, hd * 2), T(sx * (hw - 0.01), -0.2, 0));
  // dark lane line on the floor
  b.add(interiorMaterial(new MeshStandardMaterial({ color: '#1f4f7a', roughness: 0.5 }), time, WATER_Y), boxFt(hw * 1.7, 0.02, 0.8), T(0, -POOL_DEPTH_SHALLOW + 0.02, 0, 0, 0, 0), { castShadow: false });
  // steps in the shallow end corner
  for (let k = 0; k < 3; k++) {
    const sw = 6 - k * 0;
    b.add(plaster, boxFt(sw, 1.1, 1.6 * (3 - k)), T(hw - sw / 2 - 0.1 + 0, -POOL_DEPTH_SHALLOW + 0.55 + k * 1.1, -hd + 0.8 * (3 - k)));
    b.add(tileI, boxFt(sw, 0.06, 0.25), T(hw - sw / 2 - 0.1, -POOL_DEPTH_SHALLOW + 1.13 + k * 1.1, -hd + 1.6 * (3 - k) - 0.12));
  }
  // drain grate
  b.add(M.paint('#e8eef0', 0.4), cyl(0.6, 0.6, 0.06, 16), T(-hw * 0.55, -POOL_DEPTH_DEEP + 0.03, 0), { castShadow: false });

  // ── ladder at the deep end side, handrail at the steps
  const chrome = M.chrome();
  for (const off of [-0.9, 0.9]) {
    b.add(chrome, tube([[-hw + 4 + off, -3.5, hd - 0.25], [-hw + 4 + off, -0.4, hd - 0.25], [-hw + 4 + off, 2.3, hd + 0.2], [-hw + 4 + off, 2.6, hd + 1.0], [-hw + 4 + off, 0, hd + 1.6]], 0.09, 28, 8));
  }
  for (let k = 0; k < 3; k++) b.add(M.paint('#e7eef0', 0.4), boxFt(1.7, 0.12, 0.45), T(-hw + 4, -0.9 - k * 0.95, hd - 0.4));
  b.add(chrome, tube([[hw - 3, -1.8, -hd + 3.2], [hw - 3, 0.6, -hd + 2.2], [hw - 3, 2.8, -hd + 0.4], [hw - 3, 2.9, -hd - 0.6], [hw - 3, 0, -hd - 1.4]], 0.08, 28, 8));

  // ── diving board at the deep end
  b.add(M.paint('#2c7fb8', 0.5), boxFt(2.2, 1.6, 2.4), T(-hw - cw - 1.8, deckY + 0.8, 0));
  b.add(M.paint('#f4f6f2', 0.35), boxFt(10, 0.22, 1.8), T(-hw - cw + 2.4, deckY + 1.75, 0, 0, 0, 0.025));
  b.add(M.paint('#c9d3d6', 0.6), boxFt(9.6, 0.04, 1.6), T(-hw - cw + 2.4, deckY + 1.87, 0, 0, 0, 0.025), { castShadow: false });

  // ── lounge chairs + towel + skimmer on the lounging end
  for (const [lz, col] of [[-hd + 3, '#f2a541'], [hd - 4.5, '#3fa7d6']] as const) {
    const frame = M.paint('#f2f2ee', 0.4);
    const lx = -hw - cw - dEnd / 2 - 0.5;
    // frame: seat + raised back
    b.add(frame, boxFt(4.4, 0.2, 2.2), T(lx + 0.6, deckY + 1.0, lz));
    b.add(frame, boxFt(2.4, 0.2, 2.2), T(lx - 2.3, deckY + 1.75, lz, 0, 0, -0.55));
    for (const [px, pz] of [[-1.2, -0.95], [-1.2, 0.95], [2.6, -0.95], [2.6, 0.95]]) {
      b.add(frame, cyl(0.08, 0.08, 1.0, 6), T(lx + px, deckY + 0.5, lz + pz));
    }
    const towel = M.stripes([col, '#ffffff', col, '#ffffff', col], false);
    b.add(towel, boxFt(4.2, 0.06, 1.9), T(lx + 0.6, deckY + 1.13, lz), { castShadow: false });
    b.add(towel, boxFt(2.2, 0.06, 1.9), T(lx - 2.25, deckY + 1.88, lz, 0, 0, -0.55), { castShadow: false });
  }
  b.add(M.metal('#c7ccd0', 0.35), cyl(0.06, 0.06, 14, 6), T(-hw + 2, deckY + 0.08, -hd - cw - 2.2, 0, 0, Math.PI / 2 + 0.02));
  b.add(M.paint('#3d7fb3', 0.6), boxFt(1.6, 0.1, 1.2), T(-hw + 9.2, deckY + 0.1, -hd - cw - 2.2));
  // a pump/skimmer lid
  b.add(M.paint('#e3e7e2', 0.5), cyl(0.5, 0.5, 0.05, 14), T(hw - 6, deckY + 0.02, hd + cw + 0.7), { castShadow: false });

  g.add(b.build('poolStatic'));

  // ── water
  const water = new Mesh(new PlaneGeometry(hw * 2, hd * 2, 1, 1).rotateX(-Math.PI / 2), waterMaterial(time));
  water.position.y = WATER_Y;
  // tile the normal map in feet
  const uv = water.geometry.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * (hw * 2) / 14, uv.getY(i) * (hd * 2) / 14);
  water.renderOrder = 2;
  water.name = 'water';
  g.add(water);

  // ── floaties: a pink flamingo ring, a donut and a beach ball
  const floats = new Group();
  const fb = new Batch();
  const pink = M.paint('#ff7eb6', 0.25);
  fb.add(pink, torus(1.5, 0.55, 12, 28).rotateX(Math.PI / 2), T(0, 0, 0));
  fb.add(pink, tube([[0, 0.3, 1.4], [0, 1.6, 1.7], [0, 2.8, 1.2], [0, 3.1, 0.6]], 0.28, 20, 8));
  fb.add(pink, sphere(0.45, 14, 10), T(0, 3.2, 0.5));
  fb.add(M.paint('#fff6ea', 0.4), cyl(0.2, 0.08, 0.6, 8), T(0, 3.05, 0.0, 0, Math.PI / 2 - 0.3));
  fb.add(M.paint('#222222', 0.4), cyl(0.09, 0.02, 0.35, 8), T(0, 2.98, -0.36, 0, Math.PI / 2 - 0.3));
  fb.add(M.paint('#111111', 0.3), sphere(0.07, 8, 6), T(0.38, 3.32, 0.6));
  fb.add(M.paint('#111111', 0.3), sphere(0.07, 8, 6), T(-0.38, 3.32, 0.6));
  const flamingo = fb.build('flamingoFloat');
  flamingo.position.set(-6, WATER_Y + 0.1, 3);
  floats.add(flamingo);
  const db = new Batch();
  const donutTex = paintTex(256, 64, (ctx) => {
    ctx.fillStyle = '#f5d28a'; ctx.fillRect(0, 0, 256, 64);
    ctx.fillStyle = '#ff9ec7'; ctx.fillRect(0, 0, 256, 36);
    const cols = ['#ffffff', '#5ad1e8', '#ffe066', '#7ee081'];
    for (let i = 0; i < 70; i++) { ctx.fillStyle = cols[i % 4]; ctx.fillRect((i * 37) % 256, (i * 13) % 30 + 3, 7, 2.5); }
  }, true);
  db.add(M.tex('donutFloat', donutTex, { roughness: 0.3 }), torus(1.35, 0.6, 12, 28).rotateX(Math.PI / 2), T(0, 0, 0));
  const donut = db.build('donutFloat');
  donut.position.set(9, WATER_Y + 0.15, -4);
  floats.add(donut);
  const ballTex = paintTex(256, 128, (ctx) => {
    const cols = ['#ff4d4d', '#ffffff', '#ffd23f', '#ffffff', '#3fa7ff', '#ffffff'];
    cols.forEach((c, i) => { ctx.fillStyle = c; ctx.fillRect((i * 256) / 6, 0, 256 / 6 + 1, 128); });
  });
  const beach = new Mesh(sphere(0.9, 20, 14), M.tex('beachball', ballTex, { roughness: 0.3 }));
  beach.castShadow = true;
  beach.position.set(2, WATER_Y + 0.75, 6);
  floats.add(beach);
  for (const f of floats.children) f.traverse((o) => { o.castShadow = true; });
  g.add(floats);

  const drift = floats.children.map((_, i) => ({ x: floats.children[i].position.x, z: floats.children[i].position.z, ph: i * 2.1 }));
  const update = (t: number) => {
    time.value = t;
    floats.children.forEach((f, i) => {
      const d = drift[i];
      f.position.x = d.x + Math.sin(t * 0.07 + d.ph) * 3.5;
      f.position.z = d.z + Math.cos(t * 0.05 + d.ph) * 2.2;
      f.position.y = (i === 2 ? WATER_Y + 0.75 : WATER_Y + 0.12) + Math.sin(t * 1.3 + d.ph) * 0.06;
      f.rotation.y = t * 0.04 + d.ph;
      f.rotation.x = Math.sin(t * 1.1 + d.ph) * 0.04;
      f.rotation.z = Math.cos(t * 0.9 + d.ph) * 0.04;
    });
  };
  return { group: g, update, floats };
}

/** The deck footprint (sim polygon) so the lawn can be cut out under it. */
export function poolDeckPoly(s: PoolSpec): [number, number][] {
  const cw = 1.4, dw = 5, dEnd = 9;
  const x0 = -s.hw - cw - dEnd, x1 = s.hw + cw + dw, z0 = -s.hd - cw - dw, z1 = s.hd + cw + dw;
  const c = Math.cos(s.rot), sn = Math.sin(s.rot);
  // local three (x, z) → sim (x, -z) then rotate by rot in the sim plane
  return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]].map(([lx, lz]) => {
    const sx = lx, sy = -lz;
    return [s.x + sx * c - sy * sn, s.y + sx * sn + sy * c] as [number, number];
  });
}
