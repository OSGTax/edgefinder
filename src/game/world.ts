import {
  BufferGeometry, DoubleSide, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial,
  PerspectiveCamera, PlaneGeometry, RingGeometry, Scene, Vector3, type WebGLRenderer,
} from 'three';
import type { Kid, Team } from '../data/types';
import { kid as kidById } from '../data/kids';
import { createRenderer } from '../gfx/renderer';
import { getQuality, type Quality } from '../gfx/quality';
import { W, yawOf } from '../gfx/units';
import type { Field } from '../sim/field';
import { WINDUP, type Match } from '../sim/match';
import { FIELD_ORDER, type LivePlay } from '../sim/play';
import type { FielderAnim, RunnerAnim } from '../sim/types';
import { Stadium, LAYOUT } from '../world/stadium';
import { KidModel } from '../kid3d/model';
import { Animator, type AnimInput, type Mode } from '../kid3d/anim';
import { makeBall, makeBat, makeGlove, makeProp } from '../kid3d/items';
import { Effects } from './fx';

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

  constructor(readonly kid: Kid, readonly team: Team, scene: Scene) {
    this.model = new KidModel(kid, team);
    this.anim = new Animator(this.model);
    this.lefty = kid.throws === 'L';
    this.glove = makeGlove(this.model.p.s);
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
  time = 0;

  constructor(canvas: HTMLCanvasElement, readonly field: Field, readonly teams: [Team, Team]) {
    this.q = getQuality();
    this.renderer = createRenderer(canvas, this.q);
    this.camera = new PerspectiveCamera(45, 16 / 9, 0.3, 12000);
    this.stadium = new Stadium(this.scene, this.renderer, field, this.q);
    this.fx = new Effects(this.scene, this.q.pixelRatio);
    this.scene.add(this.ball);
    for (const t of teams) for (const id of t.roster) this.actors.set(id, new Actor(kidById(id), t, this.scene));

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

  resize(w: number, h: number) {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
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
      want.set(c.id, { x: 0, y: -5.3, facing: 0, mode: 'crouch', t: 0, exact: phase !== 'prePitch', glove: true, reach: phase === 'pitch' || phase === 'result' ? mittAt : W(ov.pitchAim?.x ?? 0, -1.6, ov.pitchAim?.z ?? 2.2) });
      // batter in the box
      const b = m.batter;
      const side = m.batterSide;
      let bm: Mode = 'bat', bt = 0;
      const sw = m.swingIn;
      if (sw && (phase === 'pitch' || phase === 'result')) {
        if (sw.kind === 'bunt') bm = 'bunt';
        else { bm = 'swing'; bt = Math.max(0, (phase === 'pitch' ? m.pitchT : m.pitchT + m.phaseT) - sw.tSwing); }
      } else if (ov.aimColor === '#6fc3ff' && m.humanBatting) bm = 'bunt';
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

    // ball position (three space)
    let ballPos: Vector3 | null = null;
    if (ballSim) ballPos = W(ballSim.x, ballSim.y, ballSim.z);
    else if (ballHolder) ballPos = this.handOf(ballHolder);

    for (const [id, a] of this.actors) {
      const w = want.get(id);
      if (!w) continue;
      const batSideL = m.batter.id === id ? m.batterSide === 'L' : a.lefty;
      a.apply(w, dt, ballPos, batSideL);
      a.ballInHand.visible = false;
    }
    // the ball: in a hand, in flight, or on the ground
    if (ballSim) {
      this.ball.visible = true;
      this.ball.position.copy(W(ballSim.x, ballSim.y, Math.max(0.12, ballSim.z)));
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
    this.fx.ballShadow(this.ball.visible && ballSim ? this.ball.position : null);

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

