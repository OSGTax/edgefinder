import {
  Bone, CanvasTexture, Color, DoubleSide, Group, SRGBColorSpace, Matrix4, MeshStandardMaterial, Object3D, Quaternion, SkinnedMesh, SphereGeometry,
  Vector2, Vector3, type BufferGeometry, type Material, type Skeleton, type Texture,
} from 'three';
import type { Kid, Team } from '../data/types';
import { HAIR, SKIN } from '../data/palette';
import { fabricNormal } from '../gfx/textures';
import { alongMatrix, blended, limb, loft, paint, paintFn, PartList, ramp, rigid, type Ring } from './geom';
import { B, HEAD_SHAPE, makeSkeleton, proportions, type BoneName, type Proportions } from './rig';
import { ATLAS_COLS, ATLAS_ROWS, EXPRESSIONS, FACE_PATCH, paintFaceAtlas, type Expression } from './face';
import { JERSEY_V0, paintJersey, uniformColors, type UniformColors } from './uniform';
import { addCostume, addHair, addHat } from './costume';

// A kid, built entirely in code: one skeleton shared by a handful of skinned
// meshes (skin, cloth, jersey, hair, eyes, face decal, shiny bits).

export interface Lists {
  skin: PartList; cloth: PartList; jersey: PartList; hair: PartList; eyes: PartList; face: PartList; shiny: PartList;
}

const shared = new Map<string, Material>();
function sharedMat<T extends Material>(key: string, make: () => T): T {
  let m = shared.get(key) as T | undefined;
  if (!m) { m = make(); m.name = key; shared.set(key, m); }
  return m;
}

function clothMaterial() {
  return sharedMat('kidCloth', () => {
    const n = fabricNormal(256).clone();
    n.repeat.set(6, 6);
    n.needsUpdate = true;
    return new MeshStandardMaterial({ vertexColors: true, roughness: 0.82, normalMap: n, normalScale: new Vector2(0.35, 0.35) });
  });
}

/** A non-uniform look (for grown-ups): a shirt texture in the jersey layout and colour overrides. */
export interface Outfit { shirt?: Texture; colors?: Partial<UniformColors> }

/** Everything the animator needs from a built kid. */
export class KidModel {
  readonly group = new Group();
  readonly bones: Record<BoneName, Bone>;
  readonly skeleton: Skeleton;
  readonly p: Proportions;
  readonly meshes: SkinnedMesh[] = [];
  readonly faceMat: MeshStandardMaterial;
  /** attach points for held things (follow the hands) */
  readonly gripL = new Object3D();
  readonly gripR = new Object3D();
  readonly hatTop = new Object3D();
  private expr: Expression = 'neutral';
  readonly colors: UniformColors;

  constructor(readonly kid: Kid, readonly team: Team, o: { faceCell?: number; jersey?: number; outfit?: Outfit } = {}) {
    const quality = { faceCell: o.faceCell ?? 256, jersey: o.jersey ?? 512 };
    this.p = proportions(kid.look);
    const { skeleton, bones } = makeSkeleton(this.p);
    this.skeleton = skeleton;
    this.bones = Object.fromEntries(bones.map((b) => [b.name, b])) as Record<BoneName, Bone>;
    this.colors = { ...uniformColors(team), ...o.outfit?.colors };
    const L: Lists = {
      skin: new PartList(), cloth: new PartList(), jersey: new PartList(), hair: new PartList(),
      eyes: new PartList(), face: new PartList(), shiny: new PartList(),
    };
    buildBody(L, this.p, kid, this.colors);
    addHair(L, this.p, kid);
    addHat(L, this.p, kid, team, this.colors);
    addCostume(L, this.p, kid);

    const skinHex = SKIN[kid.look.skin] ?? SKIN[1];
    const skinMat = sharedMat(`kidSkin${skinHex}`, () => new MeshStandardMaterial({ color: skinHex, roughness: 0.58 }));
    const hairMat = sharedMat(`kidHair${kid.look.hairColor}`, () => new MeshStandardMaterial({ color: HAIR[kid.look.hairColor] ?? HAIR[0], roughness: 0.5 }));
    const eyeHex = irisColor(kid);
    const eyeMat = sharedMat(`kidEyes${eyeHex}`, () => new MeshStandardMaterial({ map: eyeTexture(eyeHex), roughness: 0.08, envMapIntensity: 1.4 }));
    const shinyMat = sharedMat('kidShiny', () => new MeshStandardMaterial({ vertexColors: true, roughness: 0.22, metalness: 0.35 }));
    const jerseyMat = new MeshStandardMaterial({ map: o.outfit?.shirt ?? paintJersey(kid, team, quality.jersey), roughness: 0.8, normalMap: clothMaterial().normalMap, normalScale: new Vector2(0.3, 0.3) });
    jerseyMat.name = `jersey-${kid.id}`;
    const faceTex = paintFaceAtlas({ look: kid.look, eyePhi: 21, eyeTheta: 3, eyeSize: 12 }, quality.faceCell);
    faceTex.repeat.set(1 / ATLAS_COLS, 1 / ATLAS_ROWS);
    this.faceMat = new MeshStandardMaterial({
      map: faceTex, transparent: true, depthWrite: false, roughness: 0.6, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
    });
    this.faceMat.name = `face-${kid.id}`;

    const add = (list: PartList, mat: Material, color: boolean, shadow = true, order = 0) => {
      const g = list.merge(color);
      if (!g) return;
      const m = new SkinnedMesh(g, mat);
      m.castShadow = shadow;
      m.receiveShadow = true;
      m.frustumCulled = false;
      m.renderOrder = order;
      this.meshes.push(m);
    };
    add(L.skin, skinMat, false);
    add(L.cloth, clothMaterial(), true);
    add(L.jersey, jerseyMat, false);
    add(L.hair, hairMat, false);
    add(L.eyes, eyeMat, false, false);
    add(L.shiny, shinyMat, true);
    add(L.face, this.faceMat, false, false, 1);
    this.group.add(bones[0]);
    for (const m of this.meshes) {
      this.group.add(m);
      m.bind(skeleton, new Matrix4());
    }
    // grips: palm centres, holding things along the hand's local axes
    const hl = this.p.joints.handL, hr = this.p.joints.handR;
    this.gripL.position.set(0.02, -0.13 * this.p.s, 0.03);
    this.gripR.position.set(-0.02, -0.13 * this.p.s, 0.03);
    this.bones.handL.add(this.gripL);
    this.bones.handR.add(this.gripR);
    void hl; void hr;
    // eyes open at rest (the animator blinks them)
    this.bones.lidL.rotation.x = this.bones.lidR.rotation.x = -0.62;
    this.hatTop.position.set(0, this.p.headR * 1.95, 0.02);
    this.bones.head.add(this.hatTop);
    this.setExpression('neutral');
  }

  setExpression(e: Expression) {
    if (e === this.expr && this.faceMat.map!.offset.x + this.faceMat.map!.offset.y > -1) {
      // already showing (offset check keeps the first call honest)
    }
    this.expr = e;
    const i = EXPRESSIONS.indexOf(e);
    const col = i % ATLAS_COLS, row = Math.floor(i / ATLAS_COLS);
    // canvas row 0 is the top of the texture (v = 1)
    this.faceMat.map!.offset.set(col / ATLAS_COLS, 1 - (row + 1) / ATLAS_ROWS);
  }

  get expression() { return this.expr; }

  dispose() {
    for (const m of this.meshes) m.geometry.dispose();
    (this.faceMat.map)?.dispose();
    this.faceMat.dispose();
  }
}

// ─────────────────────────────────────────────────────────────── the body

const v3 = (x: number, y: number, z: number) => new Vector3(x, y, z);

function buildBody(L: Lists, p: Proportions, kid: Kid, col: UniformColors) {
  const j = p.joints, s = p.s, wf = p.wf;
  const look = kid.look;

  // ── head: a slightly egg-shaped sphere with a narrower jaw, plus ears and a button nose
  const shape = HEAD_SHAPE[look.head] ?? HEAD_SHAPE.round;
  const R = p.headR;
  const hc = headCentre(p);
  const head = new SphereGeometry(R, 34, 24);
  {
    const pos = head.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const t = -y / R;
      if (t > 0.15) { const k = 1 - 0.2 * ramp(t, 0.15, 1); x *= k; z *= 0.96 + 0.04 * k; }
      if (look.head === 'square' && y > 0.3 * R) { x *= 1.04; }
      // cheeks fill out a little at the front
      if (z > 0.3 * R && y < 0 && y > -0.6 * R) z *= 1.03;
      pos.setXYZ(i, x * shape[0], y * shape[1], z * shape[2]);
    }
    head.computeVertexNormals();
  }
  L.skin.add(rigid(head, B.head), new Matrix4().makeTranslation(hc.x, hc.y, hc.z));
  for (const sx of [-1, 1]) {
    const ear = new SphereGeometry(R * 0.2, 14, 10);
    ear.scale(0.55, 1, 0.8);
    L.skin.add(rigid(ear, B.head), new Matrix4().makeTranslation(hc.x + sx * R * 0.98 * shape[0], hc.y - R * 0.02, hc.z - R * 0.05));
  }
  const nose = new SphereGeometry(R * 0.13, 16, 12);
  nose.scale(1, 0.85, 0.9);
  L.skin.add(rigid(nose, B.head), new Matrix4().makeTranslation(hc.x, hc.y - R * 0.16, hc.z + R * 0.98 * shape[2]));

  // ── eyes: glossy whites with a coloured iris and pupil, and upper lids that blink
  for (const side of ['L', 'R'] as const) {
    const bi = side === 'L' ? B.eyeL : B.eyeR;
    const e = j[side === 'L' ? 'eyeL' : 'eyeR'];
    // the sphere's pole looks forward, so the iris and pupil are perfectly round bands of the texture
    const eye = new SphereGeometry(p.eyeR, 28, 20);
    eye.rotateX(Math.PI / 2);
    eye.scale(1, 1.08, 0.86);
    L.eyes.add(rigid(eye, bi), new Matrix4().makeTranslation(e.x, e.y, e.z));
    // upper lid: a skin-coloured shell cap; the lid bone rotates it closed
    const lid = new SphereGeometry(p.eyeR * 1.1, 24, 10, 0, Math.PI * 2, 0, Math.PI * 0.55);
    lid.scale(1.02, 1.08, 0.9);
    L.skin.add(rigid(lid, side === 'L' ? B.lidL : B.lidR), new Matrix4().makeTranslation(e.x, e.y, e.z));
    // eyelash line along the lid edge
    const lash = loft([0, 1].map((k) => ({ y: -0.01 + k * 0.02, rx: p.eyeR * 1.12, rz: p.eyeR * 1.0 })), 20);
    L.cloth.add(paint(rigid(lash, side === 'L' ? B.lidL : B.lidR), '#1a120d'), new Matrix4().makeTranslation(e.x, e.y + Math.cos(Math.PI * 0.55) * p.eyeR * 1.1 * 1.08, e.z).multiply(new Matrix4().makeRotationX(-0.25)));
  }

  // ── face decal: a thin patch over the front of the head for brows, mouth, cheeks
  {
    const segU = 28, segV = 20;
    const patch = new SphereGeometry(R * 1.006, segU, segV, Math.PI / 2 - FACE_PATCH.phi, FACE_PATCH.phi * 2, Math.PI / 2 - FACE_PATCH.thetaHi, FACE_PATCH.thetaHi - FACE_PATCH.thetaLo);
    // SphereGeometry measures phi from -x going around; rebuild UVs from angles so u runs viewer-left → right
    const pos = patch.attributes.position, uv = patch.attributes.uv;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const phi = Math.atan2(x, z), th = Math.asin(Math.max(-1, Math.min(1, y / (R * 1.006))));
      uv.setXY(i, (phi + FACE_PATCH.phi) / (2 * FACE_PATCH.phi), (th - FACE_PATCH.thetaLo) / (FACE_PATCH.thetaHi - FACE_PATCH.thetaLo));
      // follow the same head shaping as the skull
      let px = x, pz = z;
      const t = -y / R;
      if (t > 0.15) { const k = 1 - 0.2 * ramp(t, 0.15, 1); px *= k; pz *= 0.96 + 0.04 * k; }
      if (pz > 0.3 * R && y < 0 && y > -0.6 * R) pz *= 1.03;
      pos.setXYZ(i, px * shape[0], y * shape[1], pz * shape[2]);
    }
    patch.computeVertexNormals();
    L.face.add(rigid(patch, B.head), new Matrix4().makeTranslation(hc.x, hc.y, hc.z));
  }

  // ── neck
  L.skin.add(rigid(limb(0.32 * s, 0.17 * s * wf, 0.19 * s * wf, 14), B.neck), new Matrix4().makeTranslation(0, p.neckY + 0.2 * s, 0));

  // ── torso (jersey): lofted rings from the belt to the collar, blended between hips/spine/chest
  const bellyZ = p.belly * 0.14 * s;
  const torso: Ring[] = [
    { y: p.waistY - 0.16 * s, rx: 0.44 * wf * s, rz: 0.31 * wf * s + bellyZ * 0.6, cz: bellyZ * 0.4 },
    { y: p.waistY + 0.05 * s, rx: 0.45 * wf * s + p.belly * 0.04, rz: 0.32 * wf * s + bellyZ, cz: bellyZ * 0.7 },
    { y: (p.waistY + p.chestY) / 2, rx: 0.47 * wf * s + p.belly * 0.05, rz: 0.33 * wf * s + bellyZ * 0.9, cz: bellyZ * 0.6 },
    { y: p.chestY, rx: 0.5 * wf * s, rz: 0.33 * wf * s + bellyZ * 0.3, cz: bellyZ * 0.2 },
    { y: p.shoulderY - 0.1 * s, rx: 0.56 * wf * s, rz: 0.3 * wf * s },
    { y: p.shoulderY + 0.07 * s, rx: 0.5 * wf * s, rz: 0.26 * wf * s },
    { y: p.shoulderY + 0.14 * s, rx: 0.3 * wf * s, rz: 0.2 * wf * s },
    { y: p.neckY + 0.04 * s, rx: 0.2 * s * wf, rz: 0.19 * s * wf },
  ];
  const tg = loft(torso, 40);
  {
    const uv = tg.attributes.uv;
    for (let i = 0; i < uv.count; i++) uv.setY(i, JERSEY_V0 + uv.getY(i) * (1 - JERSEY_V0));
  }
  L.jersey.add(blended(tg, (q) => {
    if (q.y < p.waistY) { const w = ramp(q.y, p.waistY - 0.2 * s, p.waistY); return [B.hips, 1 - w, B.spine, w]; }
    const w = ramp(q.y, p.waistY + 0.1 * s, p.chestY);
    return [B.spine, 1 - w, B.chest, w];
  }));
  // collar trim
  const collar = loft([{ y: p.neckY - 0.01, rx: 0.215 * s * wf, rz: 0.205 * s * wf }, { y: p.neckY + 0.07 * s, rx: 0.2 * s * wf, rz: 0.19 * s * wf }], 24);
  L.cloth.add(paint(rigid(collar, B.chest), col.trim));

  // ── sleeves + arms
  for (const side of [1, -1] as const) {
    const S = side === 1 ? 'L' : 'R';
    const sh = j[`arm${S}` as BoneName], el = j[`fore${S}` as BoneName], wr = j[`hand${S}` as BoneName];
    const armBone = side === 1 ? B.armL : B.armR, foreBone = side === 1 ? B.foreL : B.foreR, handBone = side === 1 ? B.handL : B.handR;
    const ua = el.clone().sub(sh).length();
    // shoulder ball (jersey) so the sleeve joins the torso
    const ball = new SphereGeometry(p.armR * 1.45, 18, 12);
    L.cloth.add(paint(rigid(ball, armBone), col.jersey), new Matrix4().makeTranslation(sh.x - side * 0.03, sh.y - 0.04, sh.z));
    const sleeve = loft([
      { y: -ua * 0.62, rx: p.armR * 1.36, rz: p.armR * 1.3 },
      { y: -ua * 0.6, rx: p.armR * 1.4, rz: p.armR * 1.34 },
      { y: -ua * 0.545, rx: p.armR * 1.42, rz: p.armR * 1.36 },
      { y: -ua * 0.535, rx: p.armR * 1.42, rz: p.armR * 1.36 },
      { y: -ua * 0.2, rx: p.armR * 1.48, rz: p.armR * 1.41 },
      { y: 0.02, rx: p.armR * 1.5, rz: p.armR * 1.44 },
    ], 18);
    L.cloth.add(paintFn(rigid(sleeve, armBone), (q) => new Color(q.y < -ua * 0.54 ? col.trim : col.jersey)), alongMatrix(sh, el));
    L.skin.add(rigid(limb(ua * 0.98, p.armR, p.armR * 0.88, 14), armBone), alongMatrix(sh, el));
    const fa = wr.clone().sub(el).length();
    L.skin.add(rigid(limb(fa * 0.95, p.armR * 0.9, p.armR * 0.72, 14), foreBone), alongMatrix(el, wr));
    addHand(L, p, wr, el, side, handBone);
  }

  // ── pants: pelvis, thighs, knees, upper shins; belt
  const pants = new Color(col.pants);
  const stain = new Color('#7d8a46');
  const pelvis = loft([
    { y: p.hipY - 0.2 * s, rx: 0.36 * wf * s, rz: 0.27 * wf * s },
    { y: p.hipY + 0.05 * s, rx: 0.44 * wf * s, rz: 0.31 * wf * s },
    { y: p.waistY - 0.08 * s, rx: 0.45 * wf * s, rz: 0.31 * wf * s + bellyZ * 0.5, cz: bellyZ * 0.3 },
    { y: p.waistY + 0.02 * s, rx: 0.455 * wf * s, rz: 0.32 * wf * s + bellyZ * 0.6, cz: bellyZ * 0.4 },
  ], 32, true, false);
  L.cloth.add(paint(rigid(pelvis, B.hips), pants));
  const belt = loft([{ y: p.waistY - 0.06 * s, rx: 0.462 * wf * s, rz: 0.325 * wf * s + bellyZ * 0.6, cz: bellyZ * 0.38 }, { y: p.waistY + 0.04 * s, rx: 0.462 * wf * s, rz: 0.327 * wf * s + bellyZ * 0.6, cz: bellyZ * 0.4 }], 32);
  L.cloth.add(paint(blended(belt, () => [B.hips, 0.5, B.spine, 0.5]), '#2b2420'));
  const buckle = new SphereGeometry(0.06 * s, 10, 8);
  buckle.scale(1.3, 1, 0.4);
  L.shiny.add(paint(rigid(buckle, B.hips), '#d8c27a'), new Matrix4().makeTranslation(0, p.waistY - 0.01 * s, 0.33 * wf * s + bellyZ));
  const rnd = seeded(kid.id);
  for (const side of [1, -1] as const) {
    const S = side === 1 ? 'L' : 'R';
    const hip = j[`thigh${S}` as BoneName], knee = j[`shin${S}` as BoneName], ank = j[`foot${S}` as BoneName];
    const thighBone = side === 1 ? B.thighL : B.thighR, shinBone = side === 1 ? B.shinL : B.shinR, footBone = side === 1 ? B.footL : B.footR;
    const tl = knee.clone().sub(hip).length();
    const thigh = limb(tl, p.legR * 1.25, p.legR * 1.02, 16);
    const kneeStain = rnd() > 0.35;
    L.cloth.add(paintFn(rigid(thigh, thighBone), (q) => (kneeStain && q.y < -tl * 0.82 && q.z > 0 ? pants.clone().lerp(stain, 0.55) : pants)), alongMatrix(hip, knee));
    const sl = ank.clone().sub(knee).length();
    // baggy pant leg to mid-shin, then a sock with stirrup stripes
    const shinPants = loft([
      { y: -sl * 0.52, rx: p.legR * 0.98, rz: p.legR * 0.98 },
      { y: -sl * 0.45, rx: p.legR * 1.06, rz: p.legR * 1.06 },
      { y: 0, rx: p.legR * 1.06, rz: p.legR * 1.06 },
      { y: p.legR * 0.6, rx: p.legR * 0.8, rz: p.legR * 0.8 },
    ], 16, false, true);
    L.cloth.add(paintFn(rigid(shinPants, shinBone), (q) => (kneeStain && q.y > -sl * 0.15 && q.z > 0 ? pants.clone().lerp(stain, 0.55) : pants)), alongMatrix(knee, ank));
    const sock = limb(sl * 0.98, p.legR * 0.86, p.legR * 0.7, 14);
    L.cloth.add(paint(rigid(sock, shinBone), col.socks), alongMatrix(knee, ank));
    for (const [t0, t1] of [[0.6, 0.66], [0.7, 0.74]]) {
      const r0 = p.legR * (0.86 + (0.7 - 0.86) * t0) + 0.006, r1 = p.legR * (0.86 + (0.7 - 0.86) * t1) + 0.006;
      const band = loft([{ y: -sl * t1, rx: r1, rz: r1 }, { y: -sl * t0, rx: r0, rz: r0 }], 14);
      L.cloth.add(paint(rigid(band, shinBone), col.sockStripe), alongMatrix(knee, ank));
    }
    addShoe(L, p, ank, footBone, kid);
  }
}

export function headCentre(p: Proportions): Vector3 {
  return p.joints.head.clone().add(new Vector3(0, p.headR * 0.92, 0.02));
}

function addHand(L: Lists, p: Proportions, wr: Vector3, el: Vector3, side: 1 | -1, bone: number) {
  const s = p.s;
  const dir = wr.clone().sub(el).normalize();
  const m = alongMatrix(wr, wr.clone().add(dir));
  // palm: a squashed ball; fingers: a rounded block curling forward; thumb in front
  const palm = new SphereGeometry(0.15 * s, 16, 12);
  palm.scale(0.75, 1.05, 1.05);
  L.skin.add(rigid(palm, bone), m.clone().multiply(new Matrix4().makeTranslation(0, -0.1 * s, 0.01)));
  const fingers = limb(0.15 * s, 0.09 * s, 0.08 * s, 12, 1.5);
  L.skin.add(rigid(fingers, bone), m.clone().multiply(new Matrix4().makeTranslation(0, -0.2 * s, 0.03)).multiply(new Matrix4().makeRotationX(0.5)));
  const thumb = limb(0.12 * s, 0.055 * s, 0.05 * s, 10);
  L.skin.add(rigid(thumb, bone), m.clone().multiply(new Matrix4().makeTranslation(side * -0.03 * s, -0.08 * s, 0.09 * s)).multiply(new Matrix4().makeRotationX(0.9)).multiply(new Matrix4().makeRotationZ(side * 0.4)));
}

function addShoe(L: Lists, p: Proportions, ank: Vector3, bone: number, kid: Kid) {
  const s = p.s;
  const len = p.footLen, w = 0.3 * s;
  const palette = [['#1f1f22', '#f4f4f0'], ['#f4f4f0', '#d63a3a'], ['#2a5bd7', '#f4f4f0'], ['#f4f4f0', '#1f1f22'], ['#d63a3a', '#f4f4f0'], ['#3a3a3a', '#f2c94c']];
  const [upper, accent] = palette[Math.abs(hash(kid.id)) % palette.length];
  const shoe = new SphereGeometry(0.5, 16, 10);
  const pos = shoe.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    if (y < -0.1) y = -0.1 - (y + 0.1) * 0.2; // flat sole
    const toe = z > 0 ? 1 + z * 0.25 : 1;
    x *= toe;
    pos.setXYZ(i, x * w, y * 0.55 * 0.6 * s, z * len);
  }
  shoe.computeVertexNormals();
  const up = new Color(upper), acc = new Color(accent), sole = new Color('#f2f0ea');
  paintFn(shoe, (q) => (q.y < -0.025 * s ? sole : Math.abs(q.x) > w * 0.42 && q.y > -0.01 && q.z < len * 0.1 && q.z > -len * 0.35 ? acc : up));
  L.cloth.add(rigid(shoe, bone), new Matrix4().makeTranslation(ank.x, 0.11 * s, ank.z + len * 0.22));
  // laces
  for (let k = 0; k < 3; k++) {
    const lace = limb(w * 0.55, 0.012 * s, 0.012 * s, 5);
    L.cloth.add(paint(rigid(lace, bone), '#ffffff'), new Matrix4().makeTranslation(ank.x + w * 0.27, 0.2 * s + k * 0.01, ank.z + len * (0.28 + k * 0.08)).multiply(new Matrix4().makeRotationZ(Math.PI / 2)));
  }
}

const eyeTex = new Map<string, CanvasTexture>();
/** Eye texture: v = 1 is the front of the eye (pupil), bands outward: iris with streaks, rim, white. */
function eyeTexture(irisHex: string): CanvasTexture {
  let t = eyeTex.get(irisHex);
  if (t) return t;
  const c = document.createElement('canvas');
  c.width = 64; c.height = 256;
  const g = c.getContext('2d')!;
  const H = 256;
  g.fillStyle = '#fbfbf7';
  g.fillRect(0, 0, 64, H);
  // a hint of shadow toward the back of the eyeball
  const sh = g.createLinearGradient(0, H * 0.25, 0, H);
  sh.addColorStop(0, 'rgba(0,0,0,0)');
  sh.addColorStop(1, 'rgba(120,110,120,0.35)');
  g.fillStyle = sh;
  g.fillRect(0, H * 0.25, 64, H * 0.75);
  const irisEnd = H * (0.5 / Math.PI) * 1.05, pupilEnd = H * (0.24 / Math.PI);
  const ig = g.createLinearGradient(0, 0, 0, irisEnd);
  ig.addColorStop(0, shadeHex(irisHex, 1.25));
  ig.addColorStop(0.7, irisHex);
  ig.addColorStop(1, shadeHex(irisHex, 0.45));
  g.fillStyle = ig;
  g.fillRect(0, 0, 64, irisEnd);
  for (let i = 0; i < 64; i += 3) { g.fillStyle = `rgba(255,255,255,${0.05 + (i % 7) * 0.012})`; g.fillRect(i, pupilEnd, 1, irisEnd - pupilEnd); }
  g.fillStyle = shadeHex(irisHex, 0.35);
  g.fillRect(0, irisEnd - 3, 64, 3);
  g.fillStyle = '#0b0807';
  g.fillRect(0, 0, 64, pupilEnd);
  t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  eyeTex.set(irisHex, t);
  return t;
}

function shadeHex(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16);
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)));
  return `rgb(${f((n >> 16) & 255)},${f((n >> 8) & 255)},${f(n & 255)})`;
}

function irisColor(kid: Kid): string {
  const opts = ['#5a3a1f', '#3b2a1a', '#6b4a2a', '#3f6f9f', '#4f7f4f', '#7a5a2a'];
  const pick = Math.abs(hash(kid.id + 'eye')) % opts.length;
  // darker skin tones favour brown eyes
  return kid.look.skin >= 3 ? opts[pick % 3] : opts[pick];
}

export function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h | 0;
}

export function seeded(s: string): () => number {
  let a = hash(s) >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export { v3, DoubleSide, Quaternion, type BufferGeometry };
