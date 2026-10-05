import {
  BufferGeometry, DoubleSide, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial,
  PerspectiveCamera, PlaneGeometry, RingGeometry, Scene, Vector3, type WebGLRenderer,
} from 'three';
import type { Kid, Team } from '../data/types';
import { kid as kidById } from '../data/kids';
import { createRenderer } from '../gfx/renderer';
import { mergeByMaterial } from '../gfx/build';
import { getQuality, type Quality } from '../gfx/quality';
import { W, yawOf } from '../gfx/units';
import { surfaceAt, type Field } from '../sim/field';
import { WINDUP, type Match } from '../sim/match';
import { FIELD_ORDER, type LivePlay } from '../sim/play';
import type { FielderAnim, RunnerAnim } from '../sim/types';
import { Stadium, LAYOUT } from '../world/stadium';
import { KidModel } from '../kid3d/model';
import { Animator, type AnimInput, type Mode } from '../kid3d/anim';
import { makeBall, makeBat, makeGlove, makeProp } from '../kid3d/items';
import { Effects } from './fx';
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
  private walkT = 0;
  private placed = false;
  lefty: boolean;

  constructor(readonly kid: Kid, readonly team: Team, scene: Scene, outfit?: ConstructorParameters<typeof KidModel>[2]) {
    this.model = new KidModel(kid, team, outfit);
    this.anim = new Animator(this.model);
    this.lefty = kid.throws === 'L';
    this.glove = mergeByMaterial(makeGlove(this.model.p.s));
    (this.lefty ? this.model.bones.handR : this.model.bones.handL).add(this.glove);
    this.glove.rotation.y = this.lefty ? Math.PI / 2 : -Math.PI / 2;
    this.glove.position.y = -0.05;
    this.bat = makeBat(kid.traits.hitting >= 7 ? 'metal' : 'wood', 2.2 + this.model.p.s * 0.45);
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
    const goal = W(w.x, w.y, 0);
    let mode = w.mode, speed = w.speed ?? 0, facing = w.facing;
    if (!this.placed) { this.pos.copy(goal); this.placed = true; }
    if (w.exact) {
      this.pos.copy(goal);
    } else {
      const d = this.pos.distanceTo(goal);
      if (d > 0.4) {
        // jog there
        const v = Math.min(d / dt, d > 25 ? 15 : 9);
        const dir = goal.clone().sub(this.pos).normalize();
        this.pos.addScaledVector(dir, v * dt);
        mode = v > 11 ? 'run' : 'trot';
        speed = v;
        facing = Math.atan2(dir.x, -dir.z);
        this.walkT += dt;
      } else {
        this.pos.copy(goal);
      }
    }
    // turning
    let wantYaw = this.yaw;
    if (facing !== null && facing !== undefined) wantYaw = yawOf(facing);
    else if (ballPos) wantYaw = Math.atan2(ballPos.x - this.pos.x, ballPos.z - this.pos.z);
    let dy = ((wantYaw - this.yaw + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    const turnRate = mode === 'run' || mode === 'trot' ? 12 : mode === 'swing' || mode === 'windup' || mode === 'bat' ? 30 : 7;
    this.yaw += dy * Math.min(1, dt * turnRate);
    this.model.group.position.copy(this.pos);
    this.model.group.rotation.y = this.yaw;
    this.model.group.updateMatrixWorld(true);
    const batting = mode === 'bat' || mode === 'swing' || mode === 'bunt';
    this.glove.visible = !!w.glove && !batting;
    if (this.prop) this.prop.visible = !!w.prop && !batting;
    const inp: AnimInput = {
      mode, t: w.t, speed, lift: w.lift, reach: w.reach ?? null, windup: WINDUP,
      lefty: batting ? batSideLefty : this.lefty, seat: w.seat,
      lookAt: w.look ?? ballPos,
    };
    this.anim.update(dt, inp);
    this.bat.visible = this.anim.batActive;
    if (this.anim.batActive) {
      this.bat.position.copy(this.anim.batHandle);
      this.bat.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), this.anim.batDir);
    }
  }
}

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
  /** Mr. Mendoza, at the grill */
  readonly mendoza: Actor;
  time = 0;

  constructor(canvas: HTMLCanvasElement, readonly field: Field, readonly teams: [Team, Team]) {
    this.q = getQuality();
    this.renderer = createRenderer(canvas, this.q);
    this.camera = new PerspectiveCamera(45, 16 / 9, 0.3, 12000);
    this.stadium = new Stadium(this.scene, this.renderer, field, this.q);
    const tk = performance.now();
    this.fx = new Effects(this.scene, this.q.pixelRatio);
    this.scene.add(this.ball);
    this.looseBat.visible = false;
    this.scene.add(this.looseBat);
    for (const t of teams) for (const id of t.roster) this.actors.set(id, new Actor(kidById(id), t, this.scene));
    this.mendoza = new Actor(MR_MENDOZA, teams[1], this.scene, {
      outfit: { shirt: hawaiianShirt('#1f8a8a'), colors: { pants: '#c8b48a', trim: '#1f8a8a', jersey: '#1f8a8a', socks: '#f4f4f0', sockStripe: '#f4f4f0' } },
    });
    this.mendoza.model.group.scale.setScalar(GROWNUP_SCALE);
    if (import.meta.env?.DEV) {
      console.debug(`[world] kids + fx ${Math.round(performance.now() - tk)} ms`);
      (window as unknown as { __world: World }).__world = this;
    }

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
    // start compiling every shader now, in parallel where the browser can,
    // instead of one at a time during the first frames
    this.renderer.compileAsync(this.scene, this.camera).catch(() => {});
  }

  private size = { w: 1280, h: 720 };
  private frameAvg = 16;
  private slowT = 0;
  private fastT = 0;
  private scale = 1;

  resize(w: number, h: number) {
    this.size = { w, h };
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Adaptive resolution: if frames are slow for a while, render fewer
   * pixels; if there's headroom, creep back up to the tier's sharpness.
   */
  /** off when a graphics tier is forced in the URL (screenshots, testing) */
  private adaptive = typeof location === 'undefined' || !new URLSearchParams(location.hash.slice(1)).has('q');

  adapt(realDt: number) {
    if (!this.adaptive || realDt <= 0 || realDt > 0.25) return;
    this.frameAvg += (realDt * 1000 - this.frameAvg) * 0.05;
    if (this.frameAvg > 24) { this.slowT += realDt; this.fastT = 0; }
    else if (this.frameAvg < 13) { this.fastT += realDt; this.slowT = 0; }
    else { this.slowT = 0; this.fastT = 0; }
    let next = this.scale;
    if (this.slowT > 1.5 && this.scale > 0.55) next = Math.max(0.55, this.scale - 0.15);
    if (this.fastT > 4 && this.scale < 1) next = Math.min(1, this.scale + 0.1);
    if (next !== this.scale) {
      this.scale = next;
      this.slowT = this.fastT = 0;
      this.renderer.setPixelRatio(Math.max(0.5, this.q.pixelRatio * this.scale));
      this.renderer.setSize(this.size.w, this.size.h, false);
    }
  }

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

  sync(m: Match, dt: number, ov: Overlay) {
    this.time += dt;
    const f = this.field;
    const want = new Map<string, Want>();
    const play: LivePlay | null = m.play ?? (m.phase === 'result' ? m.lastPlay : null);
    const phase = m.phase;
    const defense = m.defenseKids();
    const batTeamSide = m.battingSide;
    let ballSim: { x: number; y: number; z: number } | null = null;
    let ballHolder: string | null = null;
    let tossT = -1;

    if (play && phase !== 'halfOver' && phase !== 'over') {
      const live = play.mode !== 'held' ? W(play.ball.p.x, play.ball.p.y, play.ball.p.z) : null;
      for (const fl of play.fielders) {
        // reach for a ball that's arriving
        let reach: Vector3 | null = null;
        if (live && !fl.hasBall) {
          const d = Math.hypot(play.ball.p.x - fl.p.x, play.ball.p.y - fl.p.y);
          if (d < 7 && play.ball.p.z < 9) reach = live;
        }
        want.set(fl.kid.id, {
          x: fl.p.x, y: fl.p.y, facing: fl.anim === 'run' || fl.anim === 'throw' ? fl.facing : null,
          mode: FIELDER_MODE[fl.anim], t: fl.animT, exact: true, speed: fl.speed, lift: fl.lift, glove: true, reach,
        });
      }
      for (const r of play.runners) {
        if ((r.scored || r.out) && r.animT > 1.4) continue; // heading back to the dugout
        const p = play.runnerPos(r);
        const nb = play.basePos(r.base + 1), pb = play.basePos(r.base);
        const dx = r.dir >= 0 ? nb.x - pb.x : pb.x - nb.x, dy = r.dir >= 0 ? nb.y - pb.y : pb.y - nb.y;
        const mode: Mode = r.dir === 0 && !r.out && !r.scored ? 'ready' : RUNNER_MODE[r.anim];
        want.set(r.kid.id, {
          x: p.x, y: p.y, facing: r.dir === 0 ? null : Math.atan2(dx, dy), mode, t: r.animT, exact: true, speed: r.speed,
        });
      }
      if (play.mode === 'held' && play.holder >= 0) ballHolder = play.fielders[play.holder].kid.id;
      else ballSim = { ...play.ball.p };
    } else if (phase !== 'halfOver' && phase !== 'over') {
      defense.forEach((k, i) => {
        const pos = FIELD_ORDER[i];
        if (pos === 'P' || pos === 'C') return;
        const s = f.defaultSpots[pos];
        want.set(k.id, { x: s.x, y: s.y, facing: Math.atan2(-s.x, -s.y), mode: phase === 'prePitch' ? 'stand' : 'ready', t: 0, exact: false, glove: true });
      });
      // pitcher on the mound
      const p = m.pitcher;
      let pm: Mode = 'stand', pt = 0;
      if (phase === 'windup') { pm = 'windup'; pt = m.phaseT; }
      else if (phase === 'pitch' || phase === 'result') { pm = 'follow'; pt = phase === 'pitch' ? m.pitchT : m.pitchT + m.phaseT; }
      want.set(p.id, { x: 0, y: f.mound.y - 0.6, facing: Math.PI, mode: pm, t: pt, exact: phase !== 'prePitch', glove: true, look: W(0, 0, 3) });
      // catcher
      const c = defense[1];
      const pb = m.pitchBallPos();
      const mittAt = m.pitch ? W(m.pitch.arrival.x, -1.6, Math.max(0.6, m.pitch.arrival.z)) : null;
      const toss = phase === 'prePitch' && !m.lastPlay && !!m.pitch && m.phaseT < 0.9;
      if (toss) tossT = m.phaseT;
      want.set(c.id, { x: 0, y: -5.3, facing: 0, mode: toss && m.phaseT < 0.7 ? 'throw' : 'crouch', t: m.phaseT, exact: phase !== 'prePitch', glove: true, reach: phase === 'pitch' || phase === 'result' ? mittAt : W(ov.pitchAim?.x ?? 0, -1.6, ov.pitchAim?.z ?? 2.2) });
      // batter in the box
      const b = m.batter;
      const side = m.batterSide;
      let bm: Mode = 'bat', bt = 0;
      const sw = m.swingIn;
      if (sw && (phase === 'pitch' || phase === 'result')) {
        if (sw.kind === 'bunt') bm = 'bunt';
        else { bm = 'swing'; bt = Math.max(0, (phase === 'pitch' ? m.pitchT : m.pitchT + m.phaseT) - sw.tSwing); }
      } else if (ov.aimColor === '#6fc3ff' && m.humanBatting) bm = 'bunt';
      // strike three: shoulders slump (after the follow-through)
      const struckOut = phase === 'result' && m.strikes >= 3 && (m.lastCall === 'strike' || m.lastCall === 'swinging');
      if (struckOut && (bm !== 'swing' || bt > 0.55)) { bm = 'sad'; bt = m.phaseT; }
      if (struckOut) want.set(p.id, { x: 0, y: f.mound.y - 0.6, facing: Math.PI, mode: m.phaseT > 0.35 ? 'cheer' : 'follow', t: m.phaseT, exact: true, glove: true });
      want.set(b.id, { x: side === 'R' ? -2.55 : 2.55, y: 0.1, facing: side === 'R' ? Math.PI / 2 : -Math.PI / 2, mode: bm, t: bt, exact: phase !== 'prePitch', look: W(0, f.mound.y, 4.5) });
      // runners leading off
      m.bases.forEach((id, i) => {
        if (!id) return;
        const a = f.bases[i + 1], nb = f.bases[(i + 2) % 4];
        const u = (phase === 'windup' || phase === 'pitch' ? 7 : 3.5) / f.base;
        want.set(id, { x: a.x + (nb.x - a.x) * u, y: a.y + (nb.y - a.y) * u, facing: Math.atan2(-a.x, -a.y), mode: 'ready', t: 0, exact: false });
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
          want.set(id, { x: Math.cos(a) * 9, y: f.mound.y - 8 + Math.sin(a) * 7, facing: null, mode: 'cheer', t: this.time + i, exact: false, look: W(0, f.mound.y - 8, 3) });
        });
      }
    }

    // everyone else: the dugouts (bench sitters + a couple of kids at the fence)
    const look = ballSim ? W(ballSim.x, ballSim.y, ballSim.z) : ballHolder ? this.handOf(ballHolder) : null;
    for (const side of [0, 1] as const) {
      const team = m.side(side).team;
      const di = team.id === this.teams[0].id ? 0 : 1;
      const d = LAYOUT.dugouts[di];
      let seat = 0;
      const excited = (phase === 'live' || phase === 'result') && side === batTeamSide;
      for (const id of m.side(side).lineup.order) {
        if (want.has(id)) continue;
        const i = seat++;
        // seats along the bench (local x), facing the field (local +z)
        const lx = -4.6 + i * 1.32, lz = i % 3 === 2 ? 1.6 : -0.95;
        const c = Math.cos(d.rot), s = Math.sin(d.rot);
        // local (x, z) in three around the dugout origin → sim
        const tx = W(d.x, d.y, 0);
        const wx = tx.x + lx * c + lz * s, wz = tx.z - lx * s + lz * c;
        const standing = lz > 0;
        const facing = Math.PI - d.rot; // inverse of yawOf: yaw = d.rot
        want.set(id, {
          x: wx, y: -wz, facing, exact: false, prop: !standing, seat: 1.67,
          mode: standing ? (excited && m.phase === 'live' ? 'cheer' : (i + Math.floor(this.time / 4)) % 4 === 0 ? 'clap' : 'stand') : 'sit',
          t: this.time, look,
        });
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

    {
      // Mr. Mendoza works the grill, and turns to watch anything exciting
      const live = phase === 'live' && ballPos && play && !play.deadKind;
      const g = LAYOUT.grill;
      this.mendoza.apply({
        x: g.x + 5.2, y: g.y + 1.6, facing: live ? null : Math.atan2(-5.2, -1.6), mode: 'grill', t: this.time, exact: true,
        prop: true, look: live ? ballPos : W(g.x, g.y, 3),
      }, dt, live ? ballPos : null, false);
    }
    for (const [id, a] of this.actors) {
      const w = want.get(id);
      if (!w) continue;
      const batSideL = m.batter.id === id ? m.batterSide === 'L' : a.lefty;
      a.apply(w, dt, ballPos, batSideL);
      a.ballInHand.visible = false;
    }
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
    if (ballSim) {
      this.ball.visible = true;
      // a ball in the pool bobs on the water instead of resting on the lawn height
      const wet = ballSim.z < 1 && surfaceAt(this.field, ballSim.x, ballSim.y) === 'water';
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

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.renderer.dispose();
    for (const a of this.actors.values()) a.model.dispose();
  }
}

