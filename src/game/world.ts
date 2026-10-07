import {
  BufferGeometry, CylinderGeometry, DoubleSide, MathUtils, MeshStandardMaterial, SphereGeometry, TorusGeometry, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial,
  PerspectiveCamera, PlaneGeometry, RingGeometry, Scene, Vector3, type WebGLRenderer,
} from 'three';
import type { Kid, Team } from '../data/types';
import { kid as kidById } from '../data/kids';
import { createRenderer } from '../gfx/renderer';
import { mergeByMaterial } from '../gfx/build';
import {
  autoCeiling, deviceInfo, forcedTier, getQuality, gfxPrefs, onGfxPrefs, pixelRatioFor, qualitySetting, rememberTier, TIER_LABEL, TIER_ORDER,
  TIERS, type Quality, type QualityName,
} from '../gfx/quality';
import { TierGovernor } from '../gfx/governor';
import { PerfReadout } from '../gfx/perf';
import { ContactShadows } from '../gfx/contact';
import { W, yawOf } from '../gfx/units';
import { surfaceAt, type Field } from '../sim/field';
import { WINDUP, type Match } from '../sim/match';
import { FIELD_ORDER, type LivePlay } from '../sim/play';
import type { FielderAnim, PlayResult, RunnerAnim } from '../sim/types';
import { Stadium, LAYOUT } from '../world/stadium';
import { KidModel } from '../kid3d/model';
import { Animator, type AnimInput, type Mode } from '../kid3d/anim';
import { makeBall, makeBat, makeGlove, makeProp } from '../kid3d/items';
import { Effects } from './fx';
import { Emotes } from './emotes';
import type { Steps } from '../engine/steps';
import { hawaiianShirt } from '../kid3d/outfits';
import { GROWNUP_SCALE, MR_MENDOZA } from '../world/grownups';

// The 3D side of a game: the yard, all eighteen kids, the ball, overlays.
// Each frame `sync` reads the Match and tells every kid where to be and what
// to do; kids not involved in the play walk to their team's dugout.

const FIELDER_MODE: Record<FielderAnim, Mode> = {
  ready: 'ready', run: 'run', catch: 'catch', throw: 'throw', dive: 'dive', jump: 'jump',
  stumble: 'stumble', cheer: 'cheer', pitch: 'follow', crouch: 'crouch',
};
const RUNNER_MODE: Record<RunnerAnim, Mode> = { run: 'run', stand: 'ready', slide: 'slide', trot: 'trot', out: 'sad', cheer: 'cheer' };

interface Want {
  x: number; y: number;
  /** sim facing; null = face the ball */
  facing: number | null;
  mode: Mode;
  /** time in the mode; −1 = since this kid started doing it */
  t: number;
  /** follow the sim exactly (no walking there) */
  exact: boolean;
  speed?: number;
  lift?: number;
  reach?: Vector3 | null;
  glove?: boolean;
  prop?: boolean;
  seat?: number;
  look?: Vector3 | null;
  /** cheer/groan variety */
  variant?: number;
  /** walking there: top speed (ft/s) and how (e.g. 'walk' for a stroll) */
  maxSpeed?: number;
  moveMode?: Mode;
  /** where the off hand holds on (Mr. Mendoza's skimmer pole) */
  offHand?: Vector3 | null;
}

const _goal = new Vector3(), _dir = new Vector3();

type Moment = 'none' | 'hr' | 'run' | 'hit' | 'out' | 'k' | 'splash' | 'over';

/** Mr. Mendoza's way round to the pool: behind home plate and up the first-base side, never across the diamond. */
const MZ_ROUTE = [{ x: -10, y: -22 }, { x: 10, y: -24 }, { x: 40, y: -18 }, { x: 70, y: 20 }, { x: 85, y: 60 }, { x: 92, y: 85 }];

/** A small stable number per kid (for picking variants). */
function hashId(id: string): number {
  let h = 7;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return Math.abs(h);
}

class Actor {
  readonly model: KidModel;
  readonly anim: Animator;
  readonly glove: Group;
  readonly bat;
  readonly ballInHand;
  readonly prop: Group | null;
  readonly pos = new Vector3();
  yaw = Math.PI;
  private placed = false;
  lefty: boolean;
  /** current speed when walking/jogging somewhere on its own */
  private jogV = 0;
  /** measured ground speed (for sim-driven kids), smoothed */
  private groundV = 0;
  private propK = 0;
  private clockMode: Mode | null = null;
  private clock = 0;
  /** never show the persona prop (Mr. Mendoza on a pool trip) */
  propHidden = false;
  private inp: AnimInput = { mode: 'stand', t: 0 };

  constructor(readonly kid: Kid, readonly team: Team, scene: Scene, outfit?: ConstructorParameters<typeof KidModel>[2]) {
    this.model = new KidModel(kid, team, outfit);
    this.anim = new Animator(this.model);
    this.lefty = kid.throws === 'L';
    this.glove = mergeByMaterial(makeGlove(this.model.p.s));
    (this.lefty ? this.model.bones.handR : this.model.bones.handL).add(this.glove);
    this.glove.rotation.y = this.lefty ? Math.PI / 2 : -Math.PI / 2;
    this.glove.position.y = -0.05;
    this.bat = makeBat(kid.traits.power >= 7 ? 'metal' : 'wood', 2.2 + this.model.p.s * 0.45);
    this.bat.visible = false;
    scene.add(this.bat);
    this.ballInHand = makeBall();
    this.ballInHand.visible = false;
    (this.lefty ? this.model.gripL : this.model.gripR).add(this.ballInHand);
    this.prop = makeProp(kid.look.holding);
    if (this.prop) {
      this.model.gripR.add(this.prop);
      this.prop.visible = false;
    }
    scene.add(this.model.group);
  }

  apply(w: Want, dt: number, ballPos: Vector3 | null, batSideLefty: boolean) {
    const goal = _goal.set(w.x, 0, -w.y);
    let mode = w.mode, speed = w.speed ?? 0, facing = w.facing;
    if (!this.placed) { this.pos.copy(goal); this.placed = true; }
    if (w.exact) {
      // the sim moves this kid: measure how fast the feet must go
      const d = this.pos.distanceTo(goal);
      const v = dt > 0 && d < 6 ? d / dt : this.groundV;
      this.groundV += (v - this.groundV) * Math.min(1, dt * 12);
      this.pos.copy(goal);
      this.jogV = 0;
      if (mode === 'run' || mode === 'trot' || mode === 'homer') speed = this.groundV;
    } else {
      const d = this.pos.distanceTo(goal);
      if (d > 0.3 || this.jogV > 0.5) {
        // walk/jog there: speed up, ease off on arrival (no snapping into place)
        const top = w.maxSpeed ?? (d > 25 ? 15 : 9);
        const want = Math.min(top, Math.sqrt(2 * 14 * d));
        this.jogV += MathUtils.clamp(want - this.jogV, -20 * dt, 16 * dt);
        const step = Math.min(d, this.jogV * dt);
        _dir.copy(goal).sub(this.pos);
        if (d > 1e-4) {
          _dir.multiplyScalar(1 / d);
          this.pos.addScaledVector(_dir, step);
          if (d > 0.6) facing = Math.atan2(_dir.x, -_dir.z);
        }
        if (this.jogV > 1) {
          mode = w.moveMode ?? (this.jogV > 11 ? 'run' : this.jogV > 5.5 ? 'trot' : 'walk');
          speed = this.jogV;
        }
        if (d <= 0.3 && this.jogV < 2) { this.pos.copy(goal); this.jogV = 0; }
      } else {
        this.pos.copy(goal);
      }
    }
    // turning
    let wantYaw = this.yaw;
    if (facing !== null && facing !== undefined) wantYaw = yawOf(facing);
    else if (ballPos) wantYaw = Math.atan2(ballPos.x - this.pos.x, ballPos.z - this.pos.z);
    const dy = ((wantYaw - this.yaw + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    const turnRate = mode === 'run' || mode === 'trot' ? 12 : mode === 'swing' || mode === 'windup' || mode === 'bat' ? 30 : 7;
    const turn = dy * Math.min(1, dt * turnRate);
    this.yaw += turn;
    this.model.group.position.copy(this.pos);
    this.model.group.rotation.y = this.yaw;
    this.model.group.updateMatrixWorld(true);
    // a clock for "how long have I been doing this"
    if (mode !== this.clockMode) { this.clockMode = mode; this.clock = 0; } else this.clock += dt;
    const batting = mode === 'bat' || mode === 'swing' || mode === 'bunt';
    this.glove.visible = !!w.glove && !batting;
    const inp = this.inp;
    inp.mode = mode; inp.t = w.t < 0 ? this.clock : w.t;
    inp.speed = speed / this.model.group.scale.x; inp.lift = w.lift; inp.reach = w.reach ?? null; inp.windup = WINDUP;
    inp.lefty = batting ? batSideLefty : this.lefty; inp.seat = w.seat;
    inp.lookAt = w.look ?? ballPos; inp.turn = dt > 0 ? turn / dt : 0; inp.variant = w.variant ?? 0;
    this.anim.update(dt, inp);
    if (w.offHand) this.anim.holdWith('L', w.offHand);
    // the persona prop comes out of a pocket (scales up) rather than popping in
    if (this.prop) {
      const show = (!!w.prop || this.anim.propOut) && !batting && !this.propHidden;
      this.propK = MathUtils.clamp(this.propK + (show ? dt : -dt) * 7, 0, 1);
      this.prop.visible = this.propK > 0.02;
      if (this.prop.visible) this.prop.scale.setScalar(this.propK);
    }
    this.bat.visible = this.anim.batActive;
    if (this.anim.batActive) {
      this.bat.position.copy(this.anim.batHandle);
      this.bat.quaternion.setFromUnitVectors(UP, this.anim.batDir);
    }
  }
}

const UP = new Vector3(0, 1, 0);

/** Mr. Mendoza's pool skimmer: a long aluminium pole with a net on the end. */
function makeSkimmer(): Group {
  const g = new Group();
  const metal = new MeshStandardMaterial({ color: '#c9ced3', metalness: 0.6, roughness: 0.35 });
  const blue = new MeshStandardMaterial({ color: '#2d6fb3', roughness: 0.6 });
  const net = new MeshStandardMaterial({ color: '#e9f2f6', roughness: 0.9, transparent: true, opacity: 0.75, side: DoubleSide });
  const pole = new Mesh(new CylinderGeometry(0.06, 0.06, SKIM_LEN, 8).translate(0, SKIM_LEN / 2, 0), metal);
  const grip = new Mesh(new CylinderGeometry(0.075, 0.075, 1.2, 8).translate(0, 0.6, 0), blue);
  const rim = new Mesh(new TorusGeometry(0.75, 0.05, 6, 20), blue);
  rim.position.set(0, SKIM_LEN + 0.7, 0);
  const bag = new Mesh(new SphereGeometry(0.72, 12, 6, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), net);
  bag.position.copy(rim.position);
  bag.scale.set(1, 0.6, 1);
  // the hoop lies flat to the pole's sweep; the bag hangs below it
  rim.rotation.x = Math.PI / 2;
  bag.rotation.x = 0;
  for (const m of [pole, grip, rim, bag]) { m.castShadow = true; g.add(m); }
  g.visible = false;
  return g;
}
const SKIM_LEN = 11;
const byDist = (p: { dist: number }, q: { dist: number }) => p.dist - q.dist;

export interface Overlay {
  aim: { x: number; z: number } | null;
  aimColor: string;
  aimRadius: number;
  pitchAim: { x: number; z: number } | null;
  showZone: boolean;
  bases: { selected: number | null } | null;
}

export class World {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly stadium: Stadium;
  readonly fx: Effects;
  readonly q: Quality;
  readonly actors = new Map<string, Actor>();
  readonly ball = makeBall(0.13);
  private zoneLines: LineSegments;
  private zoneFill: Mesh;
  private reticle: Mesh;
  private mitt: Mesh;
  private baseRings: Mesh[] = [];
  private smokeT = 0;
  /** the bat a batter drops when they take off for first */
  private looseBat = makeBat('wood', 2.6);
  private looseT = -1;
  private looseFrom = { pos: new Vector3(), dir: new Vector3() };
  private batWasUp: string | null = null;
  /** the tier in use right now (`q` is the tier the yard was built at) */
  tier: QualityName;
  private gov: TierGovernor;
  /** tiers chosen automatically (setting Auto, no `q=` in the URL) */
  private autoTiers: boolean;
  /** off when a graphics tier is forced in the URL (screenshots, testing) */
  private adaptive = !forcedTier();
  private readout: PerfReadout | null = null;
  private contact: ContactShadows;
  /** Mr. Mendoza, at the grill */
  mendoza!: Actor;
  time = 0;

  /** Cheap setup only: run `build()` (all at once or paced) before using the World. */
  constructor(canvas: HTMLCanvasElement, readonly field: Field, readonly teams: [Team, Team]) {
    this.q = getQuality();
    this.tier = this.q.name;
    const ti = TIER_ORDER.indexOf(this.q.name);
    this.autoTiers = !forcedTier() && qualitySetting() === 'auto';
    this.gov = new TierGovernor({ tier: ti, min: 0, max: this.autoTiers ? Math.max(ti, TIER_ORDER.indexOf(autoCeiling())) : ti, tiers: this.autoTiers });
    this.renderer = createRenderer(canvas, this.q);
    this.watchContext(canvas);
    this.contact = new ContactShadows(this.scene, 24);
    this.applyPrefs();
    onGfxPrefs(() => this.applyPrefs());
    this.camera = new PerspectiveCamera(45, 16 / 9, 0.3, 12000);
    this.stadium = new Stadium(this.scene, this.renderer, field, this.q);
    this.fx = new Effects(this.scene, this.q.pixelRatio);
    this.scene.add(this.ball);
    this.looseBat.visible = false;
    this.scene.add(this.looseBat);

    // strike zone + aim overlays (drawn in the plate plane)
    const zg = new BufferGeometry();
    zg.setAttribute('position', new Float32BufferAttribute(new Float32Array(16 * 3), 3));
    this.zoneLines = new LineSegments(zg, new LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.7, depthTest: false }));
    this.zoneLines.renderOrder = 10;
    this.zoneFill = new Mesh(new PlaneGeometry(1, 1), new MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.07, depthTest: false, side: DoubleSide }));
    this.zoneFill.renderOrder = 9;
    this.reticle = new Mesh(new RingGeometry(0.82, 1, 40), new MeshBasicMaterial({ color: '#ffe14d', transparent: true, opacity: 0.9, depthTest: false, side: DoubleSide }));
    this.reticle.renderOrder = 11;
    const dot = new Mesh(new RingGeometry(0, 0.12, 16), (this.reticle.material as MeshBasicMaterial));
    this.reticle.add(dot);
    this.mitt = new Mesh(new RingGeometry(0.0, 0.3, 24), new MeshBasicMaterial({ color: '#8b5a2b', transparent: true, opacity: 0.85, depthTest: false, side: DoubleSide }));
    const mittRing = new Mesh(new RingGeometry(0.34, 0.42, 24), new MeshBasicMaterial({ color: '#ffe14d', depthTest: false, transparent: true, side: DoubleSide }));
    this.mitt.add(mittRing);
    this.mitt.renderOrder = 11;
    this.scene.add(this.zoneLines, this.zoneFill, this.reticle, this.mitt);
    for (let b = 0; b < 4; b++) {
      const ring = new Mesh(new RingGeometry(2.2, 2.9, 40).rotateX(-Math.PI / 2), new MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.6, depthTest: false }));
      ring.renderOrder = 8;
      const bp = field.bases[b];
      ring.position.copy(W(bp.x, bp.y, 0.08));
      this.baseRings.push(ring);
      this.scene.add(ring);
    }
  }

  /** The yard, then the kids, one team at a time. */
  *build(): Steps {
    yield* this.stadium.build(0, 0.62);
    const tk = performance.now();
    for (const [i, t] of this.teams.entries()) {
      yield { done: 0.62 + i * 0.15, msg: `Rounding up the ${t.name}` };
      for (const id of t.roster) this.actors.set(id, new Actor(kidById(id), t, this.scene));
    }
    yield { done: 0.92, msg: 'Handing Mr. Mendoza his spatula' };
    this.mendoza = new Actor(MR_MENDOZA, this.teams[1], this.scene, {
      outfit: { shirt: hawaiianShirt('#1f8a8a'), colors: { pants: '#c8b48a', trim: '#1f8a8a', jersey: '#1f8a8a', socks: '#f4f4f0', sockStripe: '#f4f4f0' } },
    });
    this.mendoza.model.group.scale.setScalar(GROWNUP_SCALE);
    this.setTier(this.tier);
    if (import.meta.env?.DEV) {
      console.debug(`[world] kids ${Math.round(performance.now() - tk)} ms`);
      (window as unknown as { __world: World }).__world = this;
    }
  }

  /**
   * Compile every shader now, in parallel where the browser can, instead of
   * one at a time during the first frames. Never waits more than a few seconds.
   */
  warmUp(): Promise<void> {
    const done = this.renderer.compileAsync(this.scene, this.camera).then(() => {}, () => {});
    return Promise.race([done, new Promise<void>((ok) => setTimeout(ok, 5000))]);
  }

  private size = { w: 1280, h: 720 };

  resize(w: number, h: number) {
    this.size = { w, h };
    this.applyPixelRatio();
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private applyPixelRatio() {
    const pr = pixelRatioFor(TIERS[this.tier], this.size.w, this.size.h, this.gov.scale);
    if (Math.abs(pr - this.renderer.getPixelRatio()) > 0.01) {
      this.renderer.setPixelRatio(pr);
      this.renderer.setSize(this.size.w, this.size.h, false);
    }
  }

  /**
   * Called once per rendered frame by the game and title loops. Frame times
   * are measured between real renders (see `render`), so the argument is
   * only kept for the callers.
   */
  adapt(_realDt?: number) {
    const dt = this.lastFrameMs;
    if (dt <= 0) return;
    this.lastFrameMs = 0;
    this.readout?.frame(dt, this.renderer, TIER_LABEL[this.tier], this.gov.scale);
    if (!this.adaptive) return;
    const act = this.gov.frame(dt);
    if (!act) return;
    if (import.meta.env?.DEV) console.debug(`[gfx] ${act} → ${this.gov.tier} @${this.gov.scale}`);
    if (act === 'down' || act === 'up') {
      this.setTier(TIER_ORDER[this.gov.tier]);
      if (this.autoTiers) rememberTier(this.tier, this.gov.tooSlow === null ? null : TIER_ORDER[this.gov.tooSlow]);
    } else this.applyPixelRatio();
  }

  /**
   * Switch the live tier: resolution, shadow map, grass and kid budgets
   * change now; textures, leaves and antialiasing are fixed when the yard is
   * built, so a tier above the build tier shows fully on the next visit.
   */
  setTier(t: QualityName) {
    this.tier = t;
    const spec = TIERS[t];
    this.stadium.env?.setShadowMapSize(spec.shadowMap);
    const grass = this.stadium.ground?.grass;
    if (grass) {
      const built = grass.userData.built as number ?? grass.count;
      grass.userData.built = built;
      grass.count = Math.min(built, spec.grassBlades);
      grass.visible = grass.count > 0;
    }
    this.applyPixelRatio();
  }

  /** frames per second the battery saver allows (0 = no cap) */
  private fpsCap = 0;
  private lastRender = 0;
  private lastFrameMs = 0;

  private applyPrefs() {
    const p = gfxPrefs();
    this.fpsCap = p.cap30 ? 30 : 0;
    this.gov.setTarget(this.fpsCap ? 1000 / this.fpsCap : 1000 / 60);
    if (p.readout && !this.readout && typeof document !== 'undefined') this.readout = new PerfReadout(deviceInfo().gpu);
    if (this.readout) this.readout.visible = p.readout;
  }

  /**
   * Real shadows for the kids nearest the camera (as many as the tier
   * allows), soft contact shadows for the rest; far kids use the lite model.
   */
  private budgetKids() {
    const spec = TIERS[this.tier];
    const cam = this.camera.position;
    const list = this.kidOrder;
    for (const a of list) a.dist = a.actor.model.group.position.distanceToSquared(cam);
    list.sort(byDist);
    const lite2 = spec.liteDist * spec.liteDist;
    this.contact.begin();
    for (let i = 0; i < list.length; i++) {
      const k = list[i];
      const m = k.actor.model;
      const visible = m.group.visible;
      const real = i < spec.kidShadows && k.dist < 160 * 160;
      if (k.shadow !== real) {
        k.shadow = real;
        for (let j = 0; j < m.meshes.length; j++) m.meshes[j].castShadow = real && k.casts[j];
      }
      if (!real && visible) this.contact.add(m.group.position, 2.4 * m.p.s * m.group.scale.x);
      // the lite model (Faces helper: KidModel.setDetail), when it exists
      const detail = k.dist > lite2 ? 'lite' : 'full';
      if (detail !== k.detail) {
        k.detail = detail;
        (m as unknown as { setDetail?: (d: 'lite' | 'full') => void }).setDetail?.(detail);
      }
    }
    this.contact.end();
  }

  private kidOrder: { actor: Actor; dist: number; shadow: boolean; detail: string; casts: boolean[] }[] = [];

  /** where a kid's throwing hand is, in three space */
  handOf(id: string): Vector3 | null {
    const a = this.actors.get(id);
    if (!a) return null;
    return (a.lefty ? a.model.gripL : a.model.gripR).getWorldPosition(new Vector3());
  }

  gloveOf(id: string): Vector3 | null {
    const a = this.actors.get(id);
    if (!a) return null;
    return a.glove.getWorldPosition(new Vector3()).add(new Vector3(0, -0.25, 0));
  }

  // ───────────────────────────────────────────────────────────── sync

  private wants = new Map<string, Want>();
  private wantPool = new Map<string, Want>();
  /** the latest moment worth reacting to: who's happy, who isn't, and since when */
  private mo = { kind: 'none' as Moment, side: 0 as 0 | 1, t: -99, kid: '', fielder: -1, caught: false, last: false };
  private seenPlay: LivePlay | null = null;
  private seenResult: PlayResult | null = null;
  private hrSeen = false;
  private splashSeen = false;
  private lastPhase = '';
  private lastRuns = -1;
  /** kids walking back to the dugout with their heads down, until this time */
  private moods = new Map<string, number>();
  /** Mr. Mendoza's errands */
  private mz = { state: 'grill' as 'grill' | 'out' | 'skim' | 'back', path: [] as { x: number; y: number }[], seg: 0, t: 0, ball: { x: 0, y: 0 }, edge: { x: 0, y: 0 }, has: false };
  private skimmer = makeSkimmer();
  private poolBall = makeBall(0.13);
  private _look = new Vector3();
  private _net = new Vector3();
  private _grip = new Vector3();
  private _hold = new Vector3();
  private emotes: Emotes | null = null;
  private jamKey = -1;
  private bangPlay: LivePlay | null = null;
  private notesMo = -1;
  private humT = 20;

  /** A reusable Want for this kid this frame. */
  private put(id: string, x: number, y: number, facing: number | null, mode: Mode, t: number, exact: boolean): Want {
    let w = this.wantPool.get(id);
    if (!w) { w = { x, y, facing, mode, t, exact }; this.wantPool.set(id, w); }
    w.x = x; w.y = y; w.facing = facing; w.mode = mode; w.t = t; w.exact = exact;
    w.speed = undefined; w.lift = undefined; w.reach = null; w.glove = false; w.prop = false; w.seat = undefined;
    w.look = null; w.variant = undefined; w.maxSpeed = undefined; w.moveMode = undefined; w.offHand = null;
    this.wants.set(id, w);
    return w;
  }

  private moment(kind: Moment, side: 0 | 1, kid = '', fielder = -1, caught = false) {
    const mo = this.mo;
    mo.kind = kind; mo.side = side; mo.t = this.time; mo.kid = kid; mo.fielder = fielder; mo.caught = caught; mo.last = false;
  }

  /** Spot the moments the kids react to, from the match state (events are the screen's). */
  private detect(m: Match, play: LivePlay | null) {
    const phase = m.phase;
    if (play && play !== this.seenPlay) { this.seenPlay = play; this.hrSeen = false; this.splashSeen = false; }
    if (play && play.deadKind === 'hr' && !this.hrSeen) { this.hrSeen = true; this.moment('hr', m.battingSide, m.batter.id); }
    if (play && phase === 'live' && !this.splashSeen && play.ball.p.z < 1 && surfaceAt(this.field, play.ball.p.x, play.ball.p.y) === 'water') {
      this.splashSeen = true;
      this.moment('splash', m.battingSide, m.batter.id);
      this.startSkim(play.ball.p.x, play.ball.p.y);
    }
    const r = m.lastResult;
    if (r && r !== this.seenResult) {
      this.seenResult = r;
      if (!r.homeRun && r.outs.length) { this.moment('out', m.battingSide, r.outs[0].kidId, r.outs[0].fielderIdx, r.caughtFly); this.mo.last = m.outs >= 3; }
      else if (!r.homeRun && !r.foul && !r.groundRule && r.hitBases >= 1) this.moment('hit', m.battingSide, m.batter.id);
    }
    if (phase === 'result' && this.lastPhase !== 'result' && m.strikes >= 3 && (m.lastCall === 'strike' || m.lastCall === 'swinging')) {
      this.moment('k', m.battingSide, m.batter.id);
      this.mo.last = m.outs >= 3;
      this.moods.set(m.batter.id, this.time + 7);
    }
    if (phase === 'over' && this.lastPhase !== 'over') this.moment('over', (m.winner === 1 ? 1 : 0));
    const runs = m.score[0] + m.score[1];
    if (this.lastRuns >= 0 && runs > this.lastRuns && phase !== 'over' && !(this.mo.kind === 'hr' && this.time - this.mo.t < 5)) this.moment('run', m.battingSide);
    this.lastRuns = runs;
    this.lastPhase = phase;
  }

  /** How a team's bench feels about the latest moment. */
  private benchMood(side: 0 | 1): 'party' | 'cheer' | 'clap' | 'groan' | null {
    const mo = this.mo, age = this.time - mo.t, ours = mo.side === side;
    switch (mo.kind) {
      case 'hr': return age < 4.5 ? (ours ? 'party' : 'groan') : null;
      case 'run': return age < 3.2 ? (ours ? 'party' : age < 1.8 ? 'groan' : null) : null;
      case 'splash': return age < 3 && ours ? 'cheer' : null;
      case 'hit': return age < 2.2 && ours ? 'cheer' : null;
      case 'out': case 'k': return age < (mo.last ? 2.6 : 1.8) ? (ours ? 'groan' : 'clap') : null;
      case 'over': return ours ? 'party' : 'groan';
    }
    return null;
  }

  sync(m: Match, dt: number, ov: Overlay) {
    this.time += dt;
    const f = this.field;
    this.wants.clear();
    const want = this.wants;
    const play: LivePlay | null = m.play ?? (m.phase === 'result' ? m.lastPlay : null);
    const phase = m.phase;
    const defense = m.defenseKids();
    const batTeamSide = m.battingSide;
    let ballSim: { x: number; y: number; z: number } | null = null;
    let ballHolder: string | null = null;
    let tossT = -1;
    this.detect(m, play);
    const mo = this.mo, moAge = this.time - mo.t;

    if (play && phase !== 'halfOver' && phase !== 'over') {
      const live = play.mode !== 'held' ? W(play.ball.p.x, play.ball.p.y, play.ball.p.z) : null;
      // after a home run the fielders stop and watch it go (the closest one longest)
      let watcher = -1;
      if (play.deadKind === 'hr') {
        let best = 1e9;
        for (const fl of play.fielders) {
          const d = Math.hypot(play.ball.p.x - fl.p.x, play.ball.p.y - fl.p.y);
          if (d < best) { best = d; watcher = fl.idx; }
        }
      }
      for (const fl of play.fielders) {
        // reach for a ball that's arriving
        let reach: Vector3 | null = null;
        if (live && !fl.hasBall) {
          const d = Math.hypot(play.ball.p.x - fl.p.x, play.ball.p.y - fl.p.y);
          if (d < 7 && play.ball.p.z < 9) reach = live;
        }
        let mode = FIELDER_MODE[fl.anim], t = fl.animT, variant = fl.idx;
        const idle = mode === 'ready' || mode === 'run' && play.deadKind === 'hr';
        if (play.deadKind === 'hr' && mo.kind === 'hr' && idle) {
          if (fl.idx === watcher || moAge < 0.5) mode = 'watch';
          else { mode = 'groan'; t = moAge - 0.5; }
        } else if (phase === 'result' && mo.kind === 'out' && moAge < 2.6) {
          if (fl.idx === mo.fielder) { mode = mo.caught ? 'catchJoy' : 'pump'; t = moAge; }
          else if (mo.last && idle && moAge > 0.25 + fl.idx * 0.06) { mode = 'pump'; t = moAge - 0.25 - fl.idx * 0.06; }
        }
        const w = this.put(fl.kid.id, fl.p.x, fl.p.y, fl.anim === 'run' || fl.anim === 'throw' ? fl.facing : null, mode, t, true);
        w.speed = fl.speed; w.lift = fl.lift; w.glove = true; w.reach = reach; w.variant = variant;
      }
      for (const r of play.runners) {
        if ((r.scored || r.out) && r.animT > 1.4) continue; // heading back to the dugout
        const p = play.runnerPos(r);
        const nb = play.basePos(r.base + 1), pb = play.basePos(r.base);
        const dx = r.dir >= 0 ? nb.x - pb.x : pb.x - nb.x, dy = r.dir >= 0 ? nb.y - pb.y : pb.y - nb.y;
        let mode: Mode = r.dir === 0 && !r.out && !r.scored ? 'ready' : RUNNER_MODE[r.anim];
        let t = r.animT;
        const homer = r.isBatter && play.deadKind === 'hr';
        // the home-run trot is the batter's own; touching home sets off their signature celebration
        if (homer && !r.scored && mode === 'trot') { mode = 'homer'; t = moAge; }
        else if (homer && r.scored) mode = 'celebrate';
        if (r.out) this.moods.set(r.kid.id, this.time + 6);
        const w = this.put(r.kid.id, p.x, p.y, r.dir === 0 ? null : Math.atan2(dx, dy), mode, t, true);
        w.speed = r.speed; w.variant = hashId(r.kid.id);
      }
      if (play.mode === 'held' && play.holder >= 0) ballHolder = play.fielders[play.holder].kid.id;
      else ballSim = { ...play.ball.p };
    } else if (phase !== 'halfOver' && phase !== 'over') {
      defense.forEach((k, i) => {
        const pos = FIELD_ORDER[i];
        if (pos === 'P' || pos === 'C') return;
        const s = f.defaultSpots[pos];
        const w = this.put(k.id, s.x, s.y, Math.atan2(-s.x, -s.y), phase === 'prePitch' ? 'stand' : 'ready', 0, false);
        w.glove = true;
        // a strikeout: the fielders punch their gloves
        if (mo.kind === 'k' && moAge < 1.4 && phase === 'result') { w.mode = 'pump'; w.t = moAge - i * 0.05; w.variant = i; }
      });
      // pitcher on the mound
      const p = m.pitcher;
      let pm: Mode = 'stand', pt = 0;
      if (phase === 'windup') { pm = 'windup'; pt = m.phaseT; }
      else if (phase === 'pitch' || phase === 'result') { pm = 'follow'; pt = phase === 'pitch' ? m.pitchT : m.pitchT + m.phaseT; }
      const pw = this.put(p.id, 0, f.mound.y - 0.6, Math.PI, pm, pt, phase !== 'prePitch');
      pw.glove = true; pw.look = W(0, 0, 3);
      // catcher
      const c = defense[1];
      const pb = m.pitchBallPos();
      const mittAt = m.pitch ? W(m.pitch.arrival.x, -1.6, Math.max(0.6, m.pitch.arrival.z)) : null;
      const toss = phase === 'prePitch' && !m.lastPlay && !!m.pitch && m.phaseT < 0.9;
      if (toss) tossT = m.phaseT;
      const cw = this.put(c.id, 0, -5.3, 0, toss && m.phaseT < 0.7 ? 'throw' : 'crouch', m.phaseT, phase !== 'prePitch');
      cw.glove = true;
      cw.reach = phase === 'pitch' || phase === 'result' ? mittAt : W(ov.pitchAim?.x ?? 0, -1.6, ov.pitchAim?.z ?? 2.2);
      // batter in the box
      const b = m.batter;
      const side = m.batterSide;
      let bm: Mode = 'bat', bt = 0;
      const sw = m.swingIn;
      if (sw && (phase === 'pitch' || phase === 'result')) {
        if (sw.kind === 'bunt') bm = 'bunt';
        else { bm = 'swing'; bt = Math.max(0, (phase === 'pitch' ? m.pitchT : m.pitchT + m.phaseT) - sw.tSwing); }
      } else if (ov.aimColor === '#6fc3ff' && m.humanBatting) bm = 'bunt';
      // strike three: this kid's own reaction (after the follow-through)
      const struckOut = phase === 'result' && m.strikes >= 3 && (m.lastCall === 'strike' || m.lastCall === 'swinging');
      if (struckOut && (bm !== 'swing' || bt > 0.55)) { bm = 'strikeout'; bt = -1; }
      if (struckOut) {
        const kw = this.put(p.id, 0, f.mound.y - 0.6, Math.PI, m.phaseT > 0.35 ? 'catchJoy' : 'follow', m.phaseT > 0.35 ? m.phaseT - 0.35 : m.phaseT, true);
        kw.glove = true;
      }
      const bw = this.put(b.id, side === 'R' ? -2.55 : 2.55, 0.1, side === 'R' ? Math.PI / 2 : -Math.PI / 2, bm, bt, phase !== 'prePitch');
      bw.look = bm === 'strikeout' ? null : W(0, f.mound.y, 4.5);
      // runners leading off
      m.bases.forEach((id, i) => {
        if (!id) return;
        const a = f.bases[i + 1], nb = f.bases[(i + 2) % 4];
        const u = (phase === 'windup' || phase === 'pitch' ? 7 : 3.5) / f.base;
        this.put(id, a.x + (nb.x - a.x) * u, a.y + (nb.y - a.y) * u, Math.atan2(-a.x, -a.y), 'ready', 0, false);
      });
      if (phase === 'pitch' && pb && pb.y > -3) ballSim = pb;
      else if (phase === 'pitch' || phase === 'result') ballHolder = c.id;
      else ballHolder = p.id;
    }

    // celebrations and the walk-off between halves
    if (phase === 'halfOver' || phase === 'over') {
      const winSide = phase === 'over' && m.winner !== null && m.winner >= 0 ? m.winner : -1;
      if (winSide >= 0) {
        const team = m.side(winSide as 0 | 1);
        team.lineup.order.forEach((id, i) => {
          const a = (i / 9) * Math.PI * 2;
          // a ring on the mound, everyone doing their own thing
          const w = this.put(id, Math.cos(a) * 9, f.mound.y - 8 + Math.sin(a) * 7, null, 'celebrate', this.time + i * 0.37, false);
          w.look = W(0, f.mound.y - 8, 3);
        });
      }
    }

    // everyone else: the dugouts (bench sitters + a couple of kids at the fence)
    const look = ballSim ? this._look.copy(W(ballSim.x, ballSim.y, ballSim.z)) : ballHolder ? this.handOf(ballHolder) : null;
    for (const side of [0, 1] as const) {
      const team = m.side(side).team;
      const di = team.id === this.teams[0].id ? 0 : 1;
      const d = LAYOUT.dugouts[di];
      const mood = phase === 'over' ? (m.winner === side ? null : 'groan') : this.benchMood(side);
      const excited = (phase === 'live' || phase === 'result') && side === batTeamSide;
      const c = Math.cos(d.rot), s = Math.sin(d.rot);
      const tx = W(d.x, d.y, 0);
      const benchFacing = Math.PI - d.rot; // inverse of yawOf: yaw = d.rot
      const ids = m.side(side).lineup.order;
      let count = 0;
      for (const id of ids) if (!want.has(id)) count++;
      let seat = 0;
      for (const id of ids) {
        if (want.has(id)) continue;
        const i = seat++;
        // seats along the bench (local x), facing the field (local +z); in a party everyone's up
        const fence = i % 3 === 2;
        const party = mood === 'party';
        const lx = -4.6 + i * 1.32, lz = fence ? 1.6 : party ? 0.6 : -0.95;
        const standing = fence || party;
        const wx = tx.x + lx * c + lz * s, wz = tx.z - lx * s + lz * c;
        let mode: Mode, t = this.time, facing: number | null = benchFacing;
        const variant = hashId(id);
        if (!standing) mode = mood === 'cheer' || mood === 'clap' ? 'sitCheer' : mood === 'groan' ? 'sitGroan' : 'sit';
        else if (party) {
          // pairs high-five first, then everybody whoops
          const partner = fence ? -1 : i % 3 === 0 ? i + 1 : i - 1;
          if (partner >= 0 && partner < count && moAge < 1.6 && phase !== 'over') {
            mode = 'highfive'; t = Math.max(0, moAge - (i >> 1) * 0.1);
            facing = partner > i ? benchFacing - Math.PI / 2 : benchFacing + Math.PI / 2;
          } else { mode = 'cheer'; facing = null; }
        } else if (mood === 'cheer') mode = variant % 2 ? 'clap' : 'pump', t = moAge;
        else if (mood === 'clap') mode = 'clap';
        else if (mood === 'groan') { mode = phase === 'over' && moAge > 3 ? 'sad' : 'groan'; t = moAge; }
        else mode = excited && phase === 'live' ? (variant % 3 === 0 ? 'cheer' : 'clap') : 'stand';
        const w = this.put(id, wx, -wz, facing, mode, t, false);
        w.prop = !standing; w.seat = 1.67; w.look = look; w.variant = variant;
        // a long face on the walk back after striking out / being thrown out
        const sad = this.moods.get(id);
        if (sad !== undefined && sad > this.time) { w.moveMode = 'mope'; w.maxSpeed = 4.2; }
      }
    }

    // the catcher lobs the ball back to the pitcher after a pitch
    if (tossT >= 0) {
      const c = defense[1];
      const from = this.handOf(c.id), to = this.handOf(m.pitcher.id);
      const u = Math.min(1, Math.max(0, (tossT - 0.22) / 0.6));
      if (from && to && u > 0 && u < 1) {
        const p = from.clone().lerp(to, u);
        p.y += Math.sin(u * Math.PI) * 7;
        ballSim = { x: p.x, y: -p.z, z: p.y };
        ballHolder = null;
      } else if (u <= 0) ballHolder = c.id;
    }
    // ball position (three space)
    let ballPos: Vector3 | null = null;
    if (ballSim) ballPos = W(ballSim.x, ballSim.y, ballSim.z);
    else if (ballHolder) ballPos = this.handOf(ballHolder);

    this.syncMendoza(m, dt, play, ballPos);
    for (const [id, a] of this.actors) {
      const w = want.get(id);
      if (!w) continue;
      const batSideL = m.batter.id === id ? m.batterSide === 'L' : a.lefty;
      a.apply(w, dt, ballPos, batSideL);
      a.ballInHand.visible = false;
    }
    this.cartoonSymbols(m, play, dt);
    // the batter drops the bat on contact; it tumbles into the dirt and stays until the next batter
    const batter = this.actors.get(m.batter.id);
    if (batter?.anim.batActive) {
      this.batWasUp = m.batter.id;
      this.looseFrom.pos.copy(batter.anim.batHandle);
      this.looseFrom.dir.copy(batter.anim.batDir);
      if (phase === 'prePitch') { this.looseT = -1; this.looseBat.visible = false; }
    } else if (this.batWasUp && phase === 'live') {
      this.batWasUp = null;
      this.looseT = 0;
      this.looseBat.material = this.actors.get(m.batter.id)?.bat.material ?? this.looseBat.material;
    }
    if (this.looseT >= 0) {
      this.looseT += dt;
      const u = Math.min(1, this.looseT / 0.45);
      const flat = new Vector3(this.looseFrom.dir.x, 0, this.looseFrom.dir.z).normalize();
      if (flat.lengthSq() < 0.1) flat.set(1, 0, 0);
      const dir = this.looseFrom.dir.clone().lerp(flat, u * u).normalize();
      const p = this.looseFrom.pos.clone().addScaledVector(flat, u * 1.2);
      p.y = Math.max(0.1, this.looseFrom.pos.y * (1 - u * u) + Math.sin(u * Math.PI) * 0.4);
      this.looseBat.visible = true;
      this.looseBat.position.copy(p);
      this.looseBat.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), dir);
      if (phase === 'prePitch' && m.phaseT > 0.5) { this.looseT = -1; this.looseBat.visible = false; }
    }
    // the ball: in a hand, in flight, or on the ground
    let wet = false;
    if (ballSim) {
      this.ball.visible = true;
      // a ball in the pool bobs on the water instead of resting on the lawn height
      wet = ballSim.z < 1 && surfaceAt(this.field, ballSim.x, ballSim.y) === 'water';
      this.ball.position.copy(W(ballSim.x, ballSim.y, wet ? -0.48 + Math.sin(this.time * 3) * 0.04 : Math.max(0.12, ballSim.z)));
      this.ball.rotation.x += dt * 25;
    } else if (ballHolder) {
      this.ball.visible = false;
      const a = this.actors.get(ballHolder);
      if (a) {
        if (phase === 'pitch' || phase === 'result' && !play) {
          // in the catcher's mitt
          a.ballInHand.visible = false;
          this.ball.visible = true;
          const g = this.gloveOf(ballHolder);
          if (g) this.ball.position.copy(g);
        } else a.ballInHand.visible = true;
      }
    } else this.ball.visible = false;
    // the splashed ball keeps floating until Mr. Mendoza nets it
    this.poolBall.visible = this.mz.has && !wet && (this.mz.state !== 'back' || this.skimmer.visible);
    const flying = ballSim && (phase === 'pitch' || phase === 'live');
    this.fx.trailTo(flying ? this.ball.position : null, this.camera, phase === 'pitch' ? 0.8 : 1);
    this.fx.ballShadow(this.ball.visible && ballSim && this.ball.position.y > -0.2 ? this.ball.position : null);

    this.updateOverlay(m, ov);
    // grill smoke
    this.smokeT -= dt;
    if (this.smokeT <= 0) { this.fx.smoke(this.stadium.grillTop); this.smokeT = 0.18; }
    this.fx.update(dt);
    this.stadium.update(this.time, dt);
  }

  /** Stars, sweat drops, "!" and notes over heads, at real moments only. */
  private cartoonSymbols(m: Match, play: LivePlay | null, dt: number) {
    if (!this.emotes) this.emotes = new Emotes(this.scene);
    const em = this.emotes;
    const head = (id: string) => this.actors.get(id)?.model.bones.head;
    if (play && m.phase === 'live') {
      for (const fl of play.fielders) {
        // a bobble: seeing stars
        if (fl.anim === 'stumble' && fl.animT < 0.1) { const h = head(fl.kid.id); if (h) em.show('stars', h, 1.8); }
      }
      // a high fly: "!" over the kid going to get it (once a play)
      if (this.bangPlay !== play && play.mode === 'batted' && play.ball.p.z > 14 && play.ball.v.z < 0) {
        let best: (typeof play.fielders)[number] | null = null, bd = 1e9;
        for (const fl of play.fielders) {
          if (fl.task !== 'chase') continue;
          const d = Math.hypot(fl.target.x - play.ball.p.x, fl.target.y - play.ball.p.y);
          if (d < bd) { bd = d; best = fl; }
        }
        if (best) { this.bangPlay = play; const h = head(best.kid.id); if (h) em.show('bang', h, 0.9); }
      }
    }
    // the pitcher in a jam: bases loaded, or three balls
    if (m.phase === 'prePitch' && m.phaseT > 0.2) {
      const loaded = m.bases.every((b) => !!b);
      const key = loaded || (m.balls === 3 && m.strikes < 2) ? m.inning * 1000 + m.batterIdx[m.battingSide] * 10 + m.balls : -1;
      if (key >= 0 && key !== this.jamKey) { this.jamKey = key; const h = head(m.pitcher.id); if (h) em.show('sweat', h, 1.8); }
    }
    // a happy bench hums along (one kid, once a celebration)
    if ((this.mo.kind === 'hr' || this.mo.kind === 'run' || this.mo.kind === 'over') && this.notesMo !== this.mo.t && this.time - this.mo.t > 1.6) {
      this.notesMo = this.mo.t;
      const side = this.mo.side;
      const id = m.side(side).lineup.order.find((k) => !this.wants.get(k)?.exact && this.wants.get(k)?.mode === 'cheer');
      const h = id && head(id);
      if (h) em.show('notes', h, 2.2);
    }
    // Mr. Mendoza hums at the grill now and then
    this.humT -= dt;
    if (this.humT <= 0) {
      this.humT = 25 + Math.random() * 20;
      if (this.mz.state === 'grill') em.show('notes', this.mendoza.model.bones.head, 2.5, GROWNUP_SCALE);
    }
    em.update(dt);
  }

  // ───────────────────────────────────────────────────────────── Mr. Mendoza

  /** A Splash Double: Mr. Mendoza puts down the spatula and goes for the skimmer. */
  private startSkim(x: number, y: number) {
    const z = this.mz;
    z.ball.x = x; z.ball.y = y; z.has = true;
    // stand on the deck at the nearest edge he can reach without wading in
    const from = MZ_ROUTE[MZ_ROUTE.length - 1];
    let best = 1e9;
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * Math.PI * 2, ux = Math.sin(a), uy = Math.cos(a);
      let d = 0;
      while (d < 30 && surfaceAt(this.field, x + ux * d, y + uy * d) === 'water') d += 0.5;
      if (d >= 30) continue;
      const ex = x + ux * (d + 1.8), ey = y + uy * (d + 1.8);
      // the walk from the route to this spot mustn't cross the water
      let dry = true;
      for (let u = 0.05; u < 1 && dry; u += 0.05) if (surfaceAt(this.field, from.x + (ex - from.x) * u, from.y + (ey - from.y) * u) === 'water') dry = false;
      const score = d + (dry ? 0 : 1000);
      if (score < best) { best = score; z.edge.x = ex; z.edge.y = ey; }
    }
    if (z.state === 'grill') { z.state = 'out'; z.seg = 0; z.path = [...MZ_ROUTE, z.edge]; }
    else if (z.state === 'back') { z.state = 'out'; z.path = [z.edge]; z.seg = 0; }
    else if (z.state === 'out') z.path[z.path.length - 1] = z.edge;
    else z.t = Math.min(z.t, 1);
  }

  private syncMendoza(m: Match, dt: number, play: LivePlay | null, ballPos: Vector3 | null) {
    const z = this.mz, a = this.mendoza, g = LAYOUT.grill;
    if (!this.skimmer.parent) { this.scene.add(this.skimmer); this.scene.add(this.poolBall); this.poolBall.visible = false; }
    const home = { x: g.x + 5.2, y: g.y + 1.6 };
    const mo = this.mo, moAge = this.time - mo.t;
    a.propHidden = z.state !== 'grill';
    this.skimmer.visible = z.state === 'skim' || z.state === 'back';
    if (z.state === 'grill') {
      // works the grill, turns to watch anything exciting, and cheers his boy's home runs
      const live = m.phase === 'live' && ballPos && play && !play.deadKind;
      let mode: Mode = 'grill', facing: number | null = live ? null : Math.atan2(-5.2, -1.6);
      if (mo.kind === 'hr' && moAge < 4) { mode = mo.kid === 'kai' ? 'cheer' : 'clap'; facing = null; }
      else if (live) mode = 'watch';
      const w = this.put(MR_MENDOZA.id, home.x, home.y, facing, mode, this.time, true);
      w.prop = mode === 'grill'; w.look = live || mode !== 'grill' ? ballPos : W(g.x, g.y, 3);
      a.apply(w, dt, live ? ballPos : null, false);
      return;
    }
    if (z.state === 'out' || z.state === 'back') {
      const tgt = z.path[Math.min(z.seg, z.path.length - 1)];
      const d = Math.hypot(a.pos.x - tgt.x, -a.pos.z - tgt.y);
      if (d < 1.2) {
        z.seg++;
        if (z.seg >= z.path.length) {
          if (z.state === 'out') { z.state = 'skim'; z.t = 0; }
          else { z.state = 'grill'; z.has = false; }
        }
      }
    }
    if (z.state === 'skim') {
      z.t += dt;
      const w = this.put(MR_MENDOZA.id, z.edge.x, z.edge.y, Math.atan2(z.ball.x - z.edge.x, z.ball.y - z.edge.y), 'skim', z.t, false);
      // the net sweeps toward the ball, scoops it at 2.6 s and lifts it out
      const bx = z.ball.x, by = z.ball.y;
      const ox = z.edge.x - bx, oy = z.edge.y - by, ol = Math.hypot(ox, oy) || 1;
      const sweep = Math.sin(z.t * 1.6) * 1.6 * (1 - Math.min(1, z.t / 2.6));
      const lift = MathUtils.smoothstep(z.t, 2.6, 3.5);
      const nx = bx + (-oy / ol) * sweep + (ox / ol) * lift * 4, ny = by + (ox / ol) * sweep + (oy / ol) * lift * 4;
      this._net.set(nx, -0.45 + lift * 4.5, -ny);
      w.look = this._net;
      a.apply(w, dt, null, false);
      this.placeSkimmer(this._net);
      if (z.t > 2.6) this.poolBall.position.copy(this._net).y -= 0.25;
      else this.poolBall.position.set(bx, -0.48 + Math.sin(this.time * 3) * 0.04, -by);
      if (z.t > 4.2) { z.state = 'back'; z.seg = 0; z.path = [...MZ_ROUTE].reverse().concat([home]); }
      return;
    }
    const tgt = z.path[Math.min(z.seg, z.path.length - 1)];
    const w = this.put(MR_MENDOZA.id, tgt.x, tgt.y, null, 'walk', -1, false);
    w.maxSpeed = 6.5; w.moveMode = 'walk';
    a.apply(w, dt, null, false);
    if (z.state === 'back') {
      // carrying the skimmer upright, ball in the net, like a flag
      a.model.gripR.getWorldPosition(this._grip);
      _dir.set(Math.sin(a.yaw) * 0.35, 1, Math.cos(a.yaw) * 0.35).normalize();
      this._net.copy(this._grip).addScaledVector(_dir, SKIM_LEN - 0.6);
      this.placeSkimmer(this._net);
      this.poolBall.position.copy(this._net).y -= 0.25;
    } else if (z.has) {
      this.poolBall.position.set(z.ball.x, -0.48 + Math.sin(this.time * 3) * 0.04, -z.ball.y);
    }
  }

  /** Lay the skimmer from Mr. Mendoza's hands to the net at `net` (three space). */
  private placeSkimmer(net: Vector3) {
    const a = this.mendoza;
    a.model.gripR.getWorldPosition(this._grip);
    _dir.copy(net).sub(this._grip).normalize();
    // the pole slides through his hands: the net lands on the spot
    this.skimmer.position.copy(net).addScaledVector(_dir, -(SKIM_LEN + 0.7));
    this.skimmer.quaternion.setFromUnitVectors(UP, _dir);
    this._hold.copy(this._grip).addScaledVector(_dir, 2.2);
    a.anim.holdWith('L', this._hold);
  }

  private updateOverlay(m: Match, ov: Overlay) {
    const z = m.zone;
    const showZone = ov.showZone;
    this.zoneLines.visible = this.zoneFill.visible = showZone;
    if (showZone) {
      const pos = this.zoneLines.geometry.attributes.position as Float32BufferAttribute;
      const x0 = -z.half, x1 = z.half, y0 = z.bottom, y1 = z.top, zz = 0.05;
      const segs = [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0], [x0 + (x1 - x0) / 3, y0, x0 + (x1 - x0) / 3, y1], [x0 + (2 * (x1 - x0)) / 3, y0, x0 + (2 * (x1 - x0)) / 3, y1], [x0, y0 + (y1 - y0) / 3, x1, y0 + (y1 - y0) / 3], [x0, y0 + (2 * (y1 - y0)) / 3, x1, y0 + (2 * (y1 - y0)) / 3]];
      segs.forEach(([a, b, c, d], i) => { pos.setXYZ(i * 2, a, b, zz); pos.setXYZ(i * 2 + 1, c, d, zz); });
      pos.needsUpdate = true;
      this.zoneFill.position.set(0, (y0 + y1) / 2, zz);
      this.zoneFill.scale.set(x1 - x0, y1 - y0, 1);
      (this.zoneLines.material as LineBasicMaterial).opacity = ov.pitchAim ? 0.9 : 0.55;
    }
    this.reticle.visible = !!ov.aim;
    if (ov.aim) {
      this.reticle.position.set(ov.aim.x, ov.aim.z, 0.1);
      this.reticle.scale.setScalar(ov.aimRadius);
      (this.reticle.material as MeshBasicMaterial).color.set(ov.aimColor);
      this.reticle.scale.y = ov.aimColor === '#6fc3ff' ? ov.aimRadius * 0.35 : ov.aimRadius;
      this.reticle.scale.x = ov.aimColor === '#6fc3ff' ? ov.aimRadius * 1.6 : ov.aimRadius;
    }
    this.mitt.visible = !!ov.pitchAim;
    if (ov.pitchAim) this.mitt.position.set(ov.pitchAim.x, ov.pitchAim.z, 0.12);
    this.baseRings.forEach((r, i) => {
      r.visible = !!ov.bases;
      if (!ov.bases) return;
      const b = i === 0 ? 4 : i;
      const sel = ov.bases.selected === b;
      (r.material as MeshBasicMaterial).color.set(sel ? '#ffe14d' : '#ffffff');
      (r.material as MeshBasicMaterial).opacity = sel ? 0.95 : 0.55 + Math.sin(this.time * 6) * 0.15;
      r.scale.setScalar(1 + Math.sin(this.time * 6 + i) * 0.05);
    });
  }

  /** Draw a frame (skipped when the battery saver says it's too soon). */
  render() {
    const now = performance.now();
    if (this.fpsCap && this.lastRender && now - this.lastRender < 1000 / this.fpsCap - 3) return;
    if (this.lost) return;
    if (this.lastRender) this.lastFrameMs = now - this.lastRender;
    this.lastRender = now;
    if (this.kidOrder.length !== this.actors.size + 1 && this.mendoza) {
      this.kidOrder = [...this.actors.values(), this.mendoza].map((actor) => ({
        actor, dist: 0, shadow: true, detail: 'full', casts: actor.model.meshes.map((m) => m.castShadow),
      }));
    }
    this.budgetKids();
    this.renderer.render(this.scene, this.camera);
  }

  // ───────────────────────────────────────────────────── lost context

  /** the browser took the GPU away (phone backgrounded, driver reset); nothing draws until it's back */
  lost = false;

  private watchContext(canvas: HTMLCanvasElement) {
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault(); // ask for it back
      this.lost = true;
      this.onContextChange?.(true);
    });
    canvas.addEventListener('webglcontextrestored', () => {
      // three re-uploads geometry and textures by itself; rebuild what was rendered on the GPU
      this.stadium.env?.buildEnvMap(this.scene, this.renderer);
      this.renderer.shadowMap.needsUpdate = true;
      this.lost = false;
      this.lastRender = 0;
      this.gov.reset();
      this.onContextChange?.(false);
    });
  }

  /** the app shows a "graphics are waking up" note while the context is lost */
  onContextChange: ((lost: boolean) => void) | null = null;

  dispose() {
    this.readout?.dispose();
    this.renderer.dispose();
    for (const a of this.actors.values()) a.model.dispose();
  }
}

