import {
  Bone, CanvasTexture, Color, DoubleSide, Group, SphereGeometry, SRGBColorSpace, Matrix4, MeshStandardMaterial, Object3D, Quaternion, SkinnedMesh, 
  Vector2, Vector3, type BufferGeometry, type Material, type Skeleton, type Texture,
} from 'three';
import type { Kid, Team } from '../data/types';
import { HAIR, SKIN } from '../data/palette';
import { fabricNormal } from '../gfx/textures';
import { alongMatrix, blended, isLite, limb, limbRings, loft, paint, paintFn, PartList, ramp, rigid, sphere, torus, withDetail, type Ring } from './geom';
import { B, HEAD_SHAPE, makeSkeleton, proportions, type BoneName, type Proportions } from './rig';
import { ATLAS_COLS, ATLAS_ROWS, EXPRESSIONS, FACE_PATCH, paintFaceAtlas, type Expression } from './face';
import { EYE_SHAPES, faceRecipe, type FaceRecipe } from './face-recipes';
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
  /** the full-detail meshes (see setDetail for the lite set) */
  readonly meshes: SkinnedMesh[];
  private liteMeshes: SkinnedMesh[] | null = null;
  private detailLevel: 'full' | 'lite' = 'full';
  private mats: Record<keyof Lists, Material>;
  readonly faceMat: MeshStandardMaterial;
  /** attach points for held things (follow the hands) */
  readonly gripL = new Object3D();
  readonly gripR = new Object3D();
  readonly hatTop = new Object3D();
  private expr: Expression = 'neutral';
  private lidClose = 0;
  readonly colors: UniformColors;
  /** this kid's hand-picked face (eyes, brows, mouth...) */
  readonly recipe: FaceRecipe;

  constructor(readonly kid: Kid, readonly team: Team, o: { faceCell?: number; jersey?: number; outfit?: Outfit } = {}) {
    const quality = { faceCell: o.faceCell ?? 256, jersey: o.jersey ?? 512 };
    this.recipe = faceRecipe(kid);
    this.p = proportions(kid.look, this.recipe);
    const { skeleton, bones } = makeSkeleton(this.p);
    this.skeleton = skeleton;
    this.bones = Object.fromEntries(bones.map((b) => [b.name, b])) as Record<BoneName, Bone>;
    this.colors = { ...uniformColors(team), ...o.outfit?.colors };
    const skinHex = SKIN[kid.look.skin] ?? SKIN[1];
    const skinMat = sharedMat(`kidSkin${skinHex}`, () => skinMaterial(skinHex));
    const hairMat = sharedMat(`kidHair${kid.look.hairColor}`, () => new MeshStandardMaterial({ color: HAIR[kid.look.hairColor] ?? HAIR[0], roughness: 0.5 }));
    const eyeHex = this.recipe.iris;
    const eyeMat = sharedMat(`kidEyes${eyeHex}`, () => {
      const t = eyeTexture(eyeHex);
      // a little self-light so eyes never go dead-dark under a cap brim
      return new MeshStandardMaterial({ map: t, emissiveMap: t, emissive: '#ffffff', emissiveIntensity: 0.3, roughness: 0.42, envMapIntensity: 0.25 });
    });
    const shinyMat = sharedMat('kidShiny', () => new MeshStandardMaterial({ vertexColors: true, roughness: 0.22, metalness: 0.35 }));
    const jerseyMat = new MeshStandardMaterial({ map: o.outfit?.shirt ?? paintJersey(kid, team, quality.jersey), roughness: 0.8, normalMap: clothMaterial().normalMap, normalScale: new Vector2(0.3, 0.3) });
    jerseyMat.name = `jersey-${kid.id}`;
    const faceTex = paintFaceAtlas(faceSpec(this.p, kid, this.recipe), quality.faceCell);
    faceTex.repeat.set(1 / ATLAS_COLS, 1 / ATLAS_ROWS);
    this.faceMat = new MeshStandardMaterial({
      map: faceTex, transparent: true, depthWrite: false, roughness: 0.6, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
    });
    this.faceMat.name = `face-${kid.id}`;
    this.mats = { skin: skinMat, cloth: clothMaterial(), jersey: jerseyMat, hair: hairMat, eyes: eyeMat, shiny: shinyMat, face: this.faceMat };

    this.group.add(bones[0]);
    this.meshes = this.buildMeshes(FULL);
    // grips: palm centres, holding things along the hand's local axes
    const hl = this.p.joints.handL, hr = this.p.joints.handR;
    this.gripL.position.set(0.02, -0.13 * this.p.s, 0.03);
    this.gripR.position.set(-0.02, -0.13 * this.p.s, 0.03);
    this.bones.handL.add(this.gripL);
    this.bones.handR.add(this.gripR);
    void hl; void hr;
    this.hatTop.position.set(0, this.p.headR * 1.95, 0.02);
    this.bones.head.add(this.hatTop);
    this.setExpression('neutral');
  }

  setExpression(e: Expression) {
    if (e === this.expr && this.faceMat.map!.offset.x + this.faceMat.map!.offset.y > -1) {
      // already showing (offset check keeps the first call honest)
    }
    this.expr = e;
    this.setLids(this.lidClose);
    const i = EXPRESSIONS.indexOf(e);
    const col = i % ATLAS_COLS, row = Math.floor(i / ATLAS_COLS);
    // canvas row 0 is the top of the texture (v = 1)
    this.faceMat.map!.offset.set(col / ATLAS_COLS, 1 - (row + 1) / ATLAS_ROWS);
  }

  get expression() { return this.expr; }

  /**
   * Close the upper lids by `close` (0 = as open as the expression wants, 1 = shut).
   * The animator calls this to blink; the expression decides the resting opening.
   */
  setLids(close: number) {
    this.lidClose = close;
    const open = lidOpening(this.recipe, this.expr);
    const a = open + (LID_SHUT - open) * Math.max(0, Math.min(1, close));
    this.bones.lidL.rotation.set(a, 0, 0);
    this.bones.lidR.rotation.set(a, 0, 0);
  }

  /**
   * Switch between the full model and a lite one with about a third of the triangles (same
   * skeleton, materials and face; it looks the same from 60+ ft). The lite meshes are built
   * the first time they're asked for — call setDetail('lite') during loading to pay for it early.
   */
  setDetail(d: 'full' | 'lite') {
    if (d === this.detailLevel) return;
    if (d === 'lite') this.prepareLite();
    this.detailLevel = d;
    for (const m of this.meshes) m.visible = d === 'full';
    for (const m of this.liteMeshes ?? []) m.visible = d === 'lite';
  }

  get detail() { return this.detailLevel; }

  /** Build the lite meshes now (e.g. during loading) so the first switch to lite doesn't hitch. */
  prepareLite() {
    if (this.liteMeshes) return;
    this.liteMeshes = this.buildMeshes(LITE);
    for (const m of this.liteMeshes) {
      m.visible = this.detailLevel === 'lite';
      // shadow casting follows the matching full mesh, so code that budgets shadows on
      // `meshes` drives both sets
      const twin = this.meshes.find((f) => f.material === m.material);
      if (twin) Object.defineProperty(m, 'castShadow', { get: () => twin.castShadow, set: () => {}, configurable: true });
    }
  }

  /** The meshes currently drawn (full or lite). */
  get activeMeshes(): readonly SkinnedMesh[] { return this.detailLevel === 'lite' ? this.liteMeshes! : this.meshes; }

  /** Build every part at a detail level and bind it to the shared skeleton. */
  private buildMeshes(detail: number): SkinnedMesh[] {
    const kid = this.kid;
    const L: Lists = {
      skin: new PartList(), cloth: new PartList(), jersey: new PartList(), hair: new PartList(),
      eyes: new PartList(), face: new PartList(), shiny: new PartList(),
    };
    withDetail(detail, () => {
      buildBody(L, this.p, kid, this.colors, this.recipe);
      addHair(L, this.p, kid);
      addHat(L, this.p, kid, this.team, this.colors);
      addCostume(L, this.p, kid);
    });
    const out: SkinnedMesh[] = [];
    const add = (key: keyof Lists, color: boolean, shadow = true, order = 0) => {
      const g = L[key].merge(color);
      if (!g) return;
      const m = new SkinnedMesh(g, this.mats[key]);
      m.castShadow = shadow;
      m.receiveShadow = true;
      m.frustumCulled = false;
      m.renderOrder = order;
      out.push(m);
      return m;
    };
    add('skin', false);
    add('cloth', true);
    add('jersey', false);
    add('hair', false);
    // eyes don't take shadows: a cap brim or the lid must never black them out
    const eyes = add('eyes', false, false);
    if (eyes) eyes.receiveShadow = false;
    add('shiny', true);
    add('face', false, false, 1);
    for (const m of out) {
      this.group.add(m);
      m.bind(this.skeleton, new Matrix4());
    }
    return out;
  }

  dispose() {
    for (const m of [...this.meshes, ...(this.liteMeshes ?? [])]) m.geometry.dispose();
    (this.faceMat.map)?.dispose();
    this.faceMat.dispose();
  }
}

// ─────────────────────────────────────────────────────────────── the body

const v3 = (x: number, y: number, z: number) => new Vector3(x, y, z);

function buildBody(L: Lists, p: Proportions, kid: Kid, col: UniformColors, fr: FaceRecipe) {
  const j = p.joints, s = p.s, wf = p.wf;
  const look = kid.look;

  // ── head: round, with full cheeks and a small soft chin; ears and a nose from the recipe
  const shape = HEAD_SHAPE[look.head] ?? HEAD_SHAPE.round;
  const R = p.headR;
  const hc = headCentre(p);
  // the skull and face patch keep a fixed resolution on the full model (close-ups); lite halves it
  const hd = isLite() ? 0.55 : 1;
  const head = new SphereGeometry(R, Math.round(34 * hd), Math.round(24 * hd));
  {
    const pos = head.attributes.position;
    const v = new Vector3();
    for (let i = 0; i < pos.count; i++) {
      shapeHead(v.fromBufferAttribute(pos, i), R, look);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    head.computeVertexNormals();
  }
  L.skin.add(rigid(head, B.head), new Matrix4().makeTranslation(hc.x, hc.y, hc.z));
  for (const sx of [-1, 1]) {
    const ear = sphere(R * 0.2 * fr.ears, 14, 10);
    ear.scale(0.5, 1, 0.78);
    // a little inner fold so ears read as ears
    const pos = ear.attributes.position;
    for (let i = 0; i < pos.count; i++) if (pos.getX(i) * sx > 0) pos.setX(i, pos.getX(i) * 0.75);
    ear.computeVertexNormals();
    L.skin.add(rigid(ear, B.head), new Matrix4().makeTranslation(hc.x + sx * R * 0.97 * shape[0], hc.y - R * 0.12, hc.z - R * 0.04).multiply(new Matrix4().makeRotationY(sx * 0.35)));
  }
  {
    const n = NOSES[fr.nose];
    const nose = sphere(R * n.r, 16, 12);
    nose.scale(n.sx, n.sy, n.sz);
    if (n.tilt) nose.rotateX(n.tilt);
    const ny = R * NOSE_Y;
    const nz = Math.sqrt(Math.max(0, 1 - NOSE_Y * NOSE_Y)) * R * shape[2] * 0.985;
    L.skin.add(rigid(nose, B.head), new Matrix4().makeTranslation(hc.x, hc.y + ny * shape[1], hc.z + nz));
  }

  // ── eyes: mostly dark iris with big catchlights; upper lids that rest high and blink
  const es = EYE_SHAPES[fr.eye];
  for (const side of ['L', 'R'] as const) {
    const bi = side === 'L' ? B.eyeL : B.eyeR;
    const e = j[side === 'L' ? 'eyeL' : 'eyeR'];
    // the sphere's pole looks forward, so the iris and pupil are perfectly round bands of the texture;
    // height == depth so the lid (which rotates about x) hugs the eyeball whatever its angle
    const eye = sphere(p.eyeR, 24, 16);
    eye.rotateX(Math.PI / 2);
    eye.scale(es.w, es.h, es.h);
    // almond eyes lift a touch at the outer corner
    if (fr.eye === 'almond') {
      const pos = eye.attributes.position, out = side === 'L' ? 1 : -1;
      for (let i = 0; i < pos.count; i++) { const x = pos.getX(i) * out; if (x > 0) pos.setY(i, pos.getY(i) + x * 0.12); }
      eye.computeVertexNormals();
    }
    L.eyes.add(rigid(eye, bi), new Matrix4().makeTranslation(e.x, e.y, e.z));
    // upper lid: a skin-coloured shell just outside the eyeball; the lid bone rotates it closed
    const lidR = p.eyeR * 1.08, lidT = Math.PI * 0.5;
    const lid = sphere(lidR, 18, 7, 0, Math.PI * 2, 0, lidT);
    lid.scale(es.w * 1.04, es.h, es.h);
    const lidBone = side === 'L' ? B.lidL : B.lidR;
    L.skin.add(rigid(lid, lidBone), new Matrix4().makeTranslation(e.x, e.y, e.z));
    // lash line on the lid's front edge: hidden in the head while the eye is open, a soft dark
    // line when it blinks (too small to see on the lite model)
    if (!isLite()) {
      const arc = Math.PI * 1.1;
      const lash = torus(1, 0.05, 4, 14, arc);
      lash.rotateZ(Math.PI * 1.5 - arc / 2);
      lash.rotateX(-Math.PI / 2);
      lash.scale(lidR * es.w * 1.04, lidR * 0.5, lidR * es.h);
      L.cloth.add(paint(rigid(lash, lidBone), '#3a2117'), new Matrix4().makeTranslation(e.x, e.y + 0.002, e.z));
    }
  }

  // ── face decal: a thin patch over the front of the head for brows, mouth, cheeks
  {
    const segU = 28, segV = 20;
    const patch = new SphereGeometry(R * 1.006, Math.round(segU * hd), Math.round(segV * hd), Math.PI / 2 - FACE_PATCH.phi, FACE_PATCH.phi * 2, Math.PI / 2 - FACE_PATCH.thetaHi, FACE_PATCH.thetaHi - FACE_PATCH.thetaLo);
    // SphereGeometry measures phi from -x going around; rebuild UVs from angles so u runs viewer-left → right
    const pos = patch.attributes.position, uv = patch.attributes.uv;
    const v = new Vector3();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const phi = Math.atan2(x, z), th = Math.asin(Math.max(-1, Math.min(1, y / (R * 1.006))));
      uv.setXY(i, (phi + FACE_PATCH.phi) / (2 * FACE_PATCH.phi), (th - FACE_PATCH.thetaLo) / (FACE_PATCH.thetaHi - FACE_PATCH.thetaLo));
      // follow the same head shaping as the skull
      shapeHead(v.set(x, y, z), R, look);
      pos.setXYZ(i, v.x, v.y, v.z);
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
    { y: p.chestY, rx: 0.49 * wf * s, rz: 0.33 * wf * s + bellyZ * 0.3, cz: bellyZ * 0.2 },
    // soft sloping kid shoulders (no shoulder pads)
    { y: p.shoulderY - 0.1 * s, rx: 0.5 * wf * s, rz: 0.29 * wf * s },
    { y: p.shoulderY + 0.03 * s, rx: 0.43 * wf * s, rz: 0.25 * wf * s },
    { y: p.shoulderY + 0.12 * s, rx: 0.3 * wf * s, rz: 0.2 * wf * s },
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
    // one sleeve with a domed top that rounds into the shoulder; the dome follows the
    // shoulder bone and the rest the arm, so it bends smoothly when the arm swings up
    const a = p.armR * 1.14;
    const sleeve = loft([
      { y: -ua * 0.6, rx: a * 0.93, rz: a * 0.9 },
      { y: -ua * 0.585, rx: a * 0.98, rz: a * 0.95 },
      { y: -ua * 0.5, rx: a * 0.99, rz: a * 0.96 },
      { y: -ua * 0.49, rx: a * 0.985, rz: a * 0.955 },
      { y: -ua * 0.25, rx: a * 0.99, rz: a * 0.96 },
      { y: 0, rx: a, rz: a * 0.97 },
      { y: a * 0.3, rx: a * 0.92, rz: a * 0.9 },
      { y: a * 0.55, rx: a * 0.66, rz: a * 0.64 },
      { y: a * 0.66, rx: a * 0.24, rz: a * 0.22 },
    ], 18, false, true);
    const shBone = side === 1 ? B.shoulderL : B.shoulderR;
    paintFn(sleeve, (q) => new Color(q.y < -ua * 0.495 ? col.trim : col.jersey));
    L.cloth.add(blended(sleeve, (q) => { const w = ramp(-q.y, -a * 0.2, ua * 0.3); return [shBone, 1 - w, armBone, w]; }), alongMatrix(sh, el));
    L.skin.add(rigid(limb(ua * 0.98, p.armR, p.armR * 0.88, 14), armBone), alongMatrix(sh, el));
    const fa = wr.clone().sub(el).length();
    L.skin.add(rigid(limb(fa * 0.95, p.armR * 0.9, p.armR * 0.72, 14), foreBone), alongMatrix(el, wr));
    addHand(L, p, wr, el, side, handBone);
  }

  // ── pants: pelvis, thighs, knees, upper shins; belt
  const pants = new Color(col.pants);
  const stain = new Color('#7d8a46');
  const pelvis = loft([
    { y: p.hipY - 0.3 * s, rx: 0.12 * wf * s, rz: 0.12 * wf * s },
    { y: p.hipY - 0.24 * s, rx: 0.28 * wf * s, rz: 0.22 * wf * s },
    { y: p.hipY - 0.12 * s, rx: 0.4 * wf * s, rz: 0.29 * wf * s },
    { y: p.hipY + 0.05 * s, rx: 0.44 * wf * s, rz: 0.31 * wf * s },
    { y: p.waistY - 0.08 * s, rx: 0.45 * wf * s, rz: 0.31 * wf * s + bellyZ * 0.5, cz: bellyZ * 0.3 },
    { y: p.waistY + 0.02 * s, rx: 0.455 * wf * s, rz: 0.32 * wf * s + bellyZ * 0.6, cz: bellyZ * 0.4 },
  ], 32, true, false);
  L.cloth.add(paint(rigid(pelvis, B.hips), pants));
  const belt = loft([{ y: p.waistY - 0.06 * s, rx: 0.462 * wf * s, rz: 0.325 * wf * s + bellyZ * 0.6, cz: bellyZ * 0.38 }, { y: p.waistY + 0.04 * s, rx: 0.462 * wf * s, rz: 0.327 * wf * s + bellyZ * 0.6, cz: bellyZ * 0.4 }], 32);
  L.cloth.add(paint(blended(belt, () => [B.hips, 0.5, B.spine, 0.5]), '#2b2420'));
  if (!isLite()) {
  const buckle = sphere(0.06 * s, 10, 8);
  buckle.scale(1.3, 1, 0.4);
  L.shiny.add(paint(rigid(buckle, B.hips), '#d8c27a'), new Matrix4().makeTranslation(0, p.waistY - 0.01 * s, 0.33 * wf * s + bellyZ));
  }
  const rnd = seeded(kid.id);
  for (const side of [1, -1] as const) {
    const S = side === 1 ? 'L' : 'R';
    const hip = j[`thigh${S}` as BoneName], knee = j[`shin${S}` as BoneName], ank = j[`foot${S}` as BoneName];
    const thighBone = side === 1 ? B.thighL : B.thighR, shinBone = side === 1 ? B.shinL : B.shinR, footBone = side === 1 ? B.footL : B.footR;
    const tl = knee.clone().sub(hip).length();
    const kneeStain = rnd() > 0.35;
    const sl = ank.clone().sub(knee).length();
    // one baggy pant leg from the hip to mid-shin (the leg is straight down at bind pose),
    // skinned thigh → shin across the knee so there's no seam; a sock with stirrup stripes below
    const legR = p.legR, lz = (y: number) => (y > knee.y ? knee.z + (hip.z - knee.z) * (y - knee.y) / (hip.y - knee.y) : knee.z + (ank.z - knee.z) * (knee.y - y) / (knee.y - ank.y));
    const legRings: Ring[] = ([
      [knee.y - sl * 0.54, 0.9], [knee.y - sl * 0.5, 1.0], [knee.y - sl * 0.42, 1.1], [knee.y - sl * 0.2, 1.12], [knee.y, 1.11],
      [knee.y + tl * 0.25, 1.13], [knee.y + tl * 0.6, 1.18], [hip.y, 1.24], [hip.y + legR * 0.5, 1.12],
    ] as const).map(([y, r]) => ({ y, rx: legR * r, rz: legR * r * 0.97, cx: hip.x, cz: lz(y) }));
    const leg = loft(legRings, 16);
    paintFn(leg, (q) => (kneeStain && Math.abs(q.y - knee.y) < tl * 0.18 && q.z > lz(q.y) ? pants.clone().lerp(stain, 0.55) : pants));
    L.cloth.add(blended(leg, (q) => { const w = ramp(q.y, knee.y - 0.1 * s, knee.y + 0.1 * s); return [shinBone, 1 - w, thighBone, w]; }));
    // a knee filler (hidden inside the leg until the knee bends)
    L.cloth.add(paint(rigid(sphere(legR * 0.98, 12, 8), shinBone), pants), new Matrix4().makeTranslation(knee.x, knee.y, knee.z));
    const sock = limb(sl * 0.98, p.legR * 0.86, p.legR * 0.7, 14);
    L.cloth.add(paint(rigid(sock, shinBone), col.socks), alongMatrix(knee, ank));
    if (!isLite()) for (const [t0, t1] of [[0.6, 0.66], [0.7, 0.74]]) {
      const r0 = p.legR * (0.86 + (0.7 - 0.86) * t0) + 0.006, r1 = p.legR * (0.86 + (0.7 - 0.86) * t1) + 0.006;
      const band = loft([{ y: -sl * t1, rx: r1, rz: r1 }, { y: -sl * t0, rx: r0, rz: r0 }], 14);
      L.cloth.add(paint(rigid(band, shinBone), col.sockStripe), alongMatrix(knee, ank));
    }
    addShoe(L, p, ank, footBone, kid);
  }
}

/** Detail factor for the lite model (segment counts scale by it). */
const LITE = 0.5;
/** The full model is a touch under 1 so it fits the triangle budget; the head keeps its own resolution. */
const FULL = 0.85;

/** Heights on the head (fractions of the head radius, before shaping) of the nose and mouth. */
const NOSE_Y = -0.27, MOUTH_Y = -0.48;

const NOSES: Record<FaceRecipe['nose'], { r: number; sx: number; sy: number; sz: number; tilt?: number }> = {
  button: { r: 0.12, sx: 1, sy: 0.85, sz: 0.8 },
  round: { r: 0.15, sx: 1.05, sy: 0.9, sz: 0.85 },
  small: { r: 0.1, sx: 1, sy: 0.85, sz: 0.8 },
  long: { r: 0.12, sx: 0.9, sy: 1.15, sz: 1.0, tilt: -0.25 },
  snub: { r: 0.11, sx: 1.05, sy: 0.8, sz: 0.85, tilt: 0.35 },
  broad: { r: 0.13, sx: 1.35, sy: 0.8, sz: 0.78 },
};

/**
 * Head shaping (in place, about the head centre): a round skull, a shorter lower face with full
 * cheeks and a small soft chin, then the head-shape scale. The face decal uses it too.
 */
export function shapeHead(v: Vector3, R: number, look: Kid['look']): Vector3 {
  const shape = HEAD_SHAPE[look.head] ?? HEAD_SHAPE.round;
  let { x, y, z } = v;
  const t = -y / R;
  if (t > 0) {
    // full cheeks bulge out and forward around the mouth line
    const cheek = Math.exp(-(((t - 0.42) / 0.26) ** 2));
    x *= 1 + 0.06 * cheek;
    if (z > 0) z *= 1 + 0.05 * cheek;
    // a small chin: narrow only near the bottom, and a slightly shorter lower face
    const k = 1 - 0.16 * ramp(t, 0.55, 1);
    x *= k; z *= 0.97 + 0.03 * k;
    y *= 1 - 0.07 * t;
  }
  if (look.head === 'square' && y > 0.3 * R) x *= 1.04;
  return v.set(x * shape[0], y * shape[1], z * shape[2]);
}

/** Where the 3D eyes, nose and mouth land on the painted face patch (face degrees). */
function faceSpec(p: Proportions, kid: Kid, fr: FaceRecipe) {
  const [sx, sy] = HEAD_SHAPE[kid.look.head] ?? HEAD_SHAPE.round;
  const R = p.headR, hc = headCentre(p);
  const D = 180 / Math.PI;
  const es = EYE_SHAPES[fr.eye];
  // the eyeball is sunk 0.6 of its radius, so the visible opening is 0.8 of it
  const vis = 0.78 * p.eyeR;
  return {
    look: kid.look, recipe: fr,
    eyePhi: Math.asin(Math.min(1, p.eyeX / (R * sx))) * D,
    eyeTheta: Math.asin((p.eyeY - hc.y) / (R * sy)) * D,
    eyeW: Math.asin(Math.min(1, (vis * es.w) / R)) * D,
    eyeH: Math.asin(Math.min(1, (vis * es.h) / R)) * D,
    noseTheta: Math.asin(NOSE_Y) * D,
    mouthTheta: Math.asin(MOUTH_Y) * D,
  };
}

/** Lid angles (radians about the eye's x axis): negative opens, LID_SHUT closes. */
const LID_SHUT = 0.95;
const LID_BY_EXPR: Record<Expression, number> = {
  neutral: 0, happy: 0.12, focus: 0.2, surprised: -0.2, sad: 0.2, yell: 0.14, smug: 0.26, oops: -0.06,
};
function lidOpening(fr: FaceRecipe, e: Expression): number {
  const open = EYE_SHAPES[fr.eye].lidOpen;
  return Math.max(-1.55, open + (LID_SHUT - open) * LID_BY_EXPR[e]);
}

/**
 * Kid skin: warm, a little self-lit (light passing through skin) so faces never go
 * dead-dark under a cap brim, with a soft rim of sky light around the edges.
 */
function skinMaterial(hex: string): MeshStandardMaterial {
  // a touch warmer than the palette swatch, most of all for the palest skin
  const base = new Color(hex);
  base.lerp(new Color('#e39a76'), 0.06 + 0.1 * Math.max(0, base.getHSL({ h: 0, s: 0, l: 0 }).l - 0.75) / 0.2);
  const m = new MeshStandardMaterial({ color: base, roughness: 0.62 });
  m.emissive = base.clone().multiply(new Color('#ff9d7a')).multiplyScalar(0.11);
  m.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
      float kidRim = pow(1.0 - clamp(dot(normalize(normal), normalize(vViewPosition)), 0.0, 1.0), 2.6);
      totalEmissiveRadiance += vec3(1.0, 0.86, 0.74) * kidRim * 0.18;`);
  };
  m.customProgramCacheKey = () => 'kidSkinRim';
  return m;
}

export function headCentre(p: Proportions): Vector3 {
  return p.joints.head.clone().add(new Vector3(0, p.headR * 0.92, 0.02));
}

function addHand(L: Lists, p: Proportions, wr: Vector3, el: Vector3, side: 1 | -1, bone: number) {
  const s = p.s;
  const dir = wr.clone().sub(el).normalize();
  const m = alongMatrix(wr, wr.clone().add(dir));
  // palm: a squashed ball, flat side toward the body; four short fingers curling in; thumb in front
  // chunky cartoon hands: a round palm and short, thick fingers
  const palm = sphere(0.155 * s, 14, 10);
  palm.scale(0.74, 1.0, 1.08);
  L.skin.add(rigid(palm, bone), m.clone().multiply(new Matrix4().makeTranslation(0, -0.1 * s, 0.01)));
  // (the lite model wears mittens: palm and thumb only)
  if (!isLite()) for (let k = 0; k < 4; k++) {
    const fl = (0.115 - Math.abs(k - 1.2) * 0.012) * s;
    const finger = loft(limbRings(fl, 0.046 * s, 0.041 * s, 2), 7);
    L.skin.add(rigid(finger, bone), m.clone()
      .multiply(new Matrix4().makeTranslation(0, -0.2 * s, (0.075 - k * 0.048) * s))
      .multiply(new Matrix4().makeRotationZ(-side * 0.5))
      .multiply(new Matrix4().makeRotationX((k - 1.5) * 0.06)));
  }
  const thumb = limb(0.1 * s, 0.056 * s, 0.048 * s, 8);
  L.skin.add(rigid(thumb, bone), m.clone().multiply(new Matrix4().makeTranslation(side * -0.03 * s, -0.08 * s, 0.09 * s)).multiply(new Matrix4().makeRotationX(0.9)).multiply(new Matrix4().makeRotationZ(side * 0.4)));
}

function addShoe(L: Lists, p: Proportions, ank: Vector3, bone: number, kid: Kid) {
  const s = p.s;
  const len = p.footLen, w = 0.3 * s, hgt = 0.3 * s;
  const palette = [['#1f1f22', '#f4f4f0'], ['#f4f4f0', '#d63a3a'], ['#2a5bd7', '#f4f4f0'], ['#f4f4f0', '#1f1f22'], ['#d63a3a', '#f4f4f0'], ['#3a3a3a', '#f2c94c']];
  const [upper, accent] = palette[Math.abs(hash(kid.id)) % palette.length];
  // a sneaker: flat sole, rounded toe box lower than the heel, a little wider at the toes
  const FLAT = -0.3;
  const top = (z: number) => (z > 0.05 ? 1 - (z - 0.05) * 0.95 : 1);
  const shoe = sphere(0.5, 18, 10);
  const pos = shoe.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = Math.max(pos.getY(i), FLAT), z = pos.getZ(i);
    pos.setXYZ(i, x * w * (z > 0 ? 1 + z * 0.3 : 1), (y - FLAT) * (hgt / (0.5 - FLAT)) * top(z), z * len);
  }
  shoe.computeVertexNormals();
  const up = new Color(upper), acc = new Color(accent), sole = new Color('#f2f0ea'), toe = new Color(upper).lerp(new Color('#ffffff'), 0.25);
  paintFn(shoe, (q) => (q.y < 0.05 * s ? sole
    : q.z > len * 0.3 && q.y < 0.12 * s ? toe
      : Math.abs(q.x) > w * 0.4 && q.y > 0.08 * s && q.y < 0.17 * s && q.z < len * 0.15 && q.z > -len * 0.3 ? acc : up));
  const at = new Vector3(ank.x, 0.012, ank.z + len * 0.22);
  L.cloth.add(rigid(shoe, bone), new Matrix4().makeTranslation(at.x, at.y, at.z));
  // laces across the top of the instep
  if (!isLite()) for (let k = 0; k < 3; k++) {
    const zn = 0.02 + k * 0.08;
    const y = (Math.sqrt(0.25 - zn * zn) - FLAT) * (hgt / (0.5 - FLAT)) * top(zn) + 0.004;
    const lace = loft([{ y: -w * 0.42, rx: 0.014 * s, rz: 0.014 * s }, { y: 0, rx: 0.014 * s, rz: 0.014 * s }], 4);
    L.cloth.add(paint(rigid(lace, bone), '#ffffff'), new Matrix4().makeTranslation(at.x - w * 0.21, at.y + y, at.z + zn * len).multiply(new Matrix4().makeRotationZ(Math.PI / 2)));
  }
}

const eyeTex = new Map<string, CanvasTexture>();
/**
 * Eye texture, painted per texel from directions on the eyeball (canvas row 0 is the front pole;
 * u = 0.75 is straight up, u = 1 toward the viewer's left). A big pupil and a big dark iris that
 * fill most of the opening, a lighter lower iris, and two round catchlights.
 */
function eyeTexture(irisHex: string): CanvasTexture {
  let t = eyeTex.get(irisHex);
  if (t) return t;
  const Wd = 64, H = 256;
  const c = document.createElement('canvas');
  c.width = Wd; c.height = H;
  const g = c.getContext('2d')!;
  const img = g.createImageData(Wd, H);
  const n = parseInt(irisHex.slice(1), 16);
  const ir = (n >> 16) & 255, ig = (n >> 8) & 255, ib = n & 255;
  const PUPIL = 0.42, IRIS = 0.93;
  const dir = (th: number, al: number): [number, number, number] => [Math.sin(th) * Math.cos(al), Math.sin(th) * Math.sin(al), Math.cos(th)];
  const c1 = dir(0.36, Math.PI * 1.75), c2 = dir(0.5, Math.PI * 0.75);
  const ang = (a: number[], b: number[]) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
  const smooth = (e0: number, e1: number, x: number) => { const k = Math.max(0, Math.min(1, (x - e0) / (e1 - e0))); return k * k * (3 - 2 * k); };
  for (let y = 0; y < H; y++) {
    const th = ((y + 0.5) / H) * Math.PI;
    for (let x = 0; x < Wd; x++) {
      const al = ((x + 0.5) / Wd) * Math.PI * 2;
      const d = dir(th, al);
      const up = -Math.sin(al);           // +1 at the top of the eye, −1 at the bottom
      let r: number, gg: number, b: number;
      if (th < IRIS + 0.03) {
        // iris: lighter toward the bottom, darker toward the rim, with faint streaks
        const low = smooth(-0.2, -0.9, up * Math.min(1, th / 0.6));
        const rim = smooth(IRIS - 0.16, IRIS, th);
        const streak = 1 + 0.06 * Math.sin(al * 23) * smooth(PUPIL, IRIS, th);
        const k = (0.62 + 0.75 * low) * (1 - 0.5 * rim) * streak;
        r = ir * k; gg = ig * k; b = ib * k;
        // pupil, with a soft edge
        const pp = smooth(PUPIL + 0.03, PUPIL - 0.03, th);
        r += (14 - r) * pp; gg += (10 - gg) * pp; b += (9 - b) * pp;
        // antialias the iris edge into the white
        const wv = smooth(IRIS - 0.02, IRIS + 0.03, th);
        r += (246 - r) * wv; gg += (242 - gg) * wv; b += (236 - b) * wv;
      } else {
        // the white, a little shadowed toward the back and under the lid
        const back = smooth(1.0, 2.0, th) * 0.35 + smooth(0.2, 1, up) * 0.08;
        r = 248 * (1 - back); gg = 244 * (1 - back); b = 238 * (1 - back * 0.8);
      }
      // catchlights
      const k1 = smooth(0.15, 0.11, ang(d, c1)), k2 = smooth(0.075, 0.05, ang(d, c2)) * 0.85;
      const k = Math.max(k1, k2);
      r += (255 - r) * k; gg += (255 - gg) * k; b += (255 - b) * k;
      const o = (y * Wd + x) * 4;
      img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.anisotropy = 4;
  eyeTex.set(irisHex, t);
  return t;
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
