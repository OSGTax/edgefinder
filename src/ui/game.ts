import { clamp, type Vec3 } from '../engine/math';
import { audio } from '../audio';
import { kid } from '../data/kids';
import { SPECIAL_INFO, type Kid, type PitchType, type Team } from '../data/types';
import { contactWindow, SWING_TIME, type SwingKind } from '../sim/batting';
import { kidHeightFt } from '../sim/field';
import { Match, WINDUP, type MatchConfig } from '../sim/match';
import { PITCHES } from '../sim/pitching';
import { FIELD_ORDER, type LivePlay } from '../sim/play';
import type { FielderAnim, RunnerAnim } from '../sim/types';
import { lerpPose, type CamPose } from '../render/camera';
import { batCam, followCam, OVERVIEW_CAM } from '../render/cameras';
import { SceneRenderer, type KidSprite, type SceneFrame } from '../render/scene';
import type { KidAnim } from '../art/kid';
import { INK } from '../data/palette';
import { Booth, ordinal, type Line } from './commentary';
import { clear, h } from './dom';
import { portraitCanvas } from './portraits';
import { logoCanvas } from '../art/logo';
import { settings } from './settings';

export interface GameOptions {
  cfg: MatchConfig;
  /** called when the player leaves; match is null if they quit early */
  onExit: (m: Match | null) => void;
  label?: string;
}

const FIELDER_ANIM: Record<FielderAnim, KidAnim> = {
  ready: 'ready', run: 'run', catch: 'catch', throw: 'throw', dive: 'dive', jump: 'jump',
  stumble: 'stumble', cheer: 'cheer', pitch: 'pitch', crouch: 'crouch',
};
const RUNNER_ANIM: Record<RunnerAnim, KidAnim> = { run: 'run', stand: 'ready', slide: 'run', trot: 'trot', out: 'out', cheer: 'cheer' };

type CamMode = 'bat' | 'field' | 'overview';

export class GameScreen {
  readonly match: Match;
  private opts: GameOptions;
  private root: HTMLElement;
  private renderer: SceneRenderer;
  private booth: Booth;
  private raf = 0;
  private last = 0;
  private time = 0;
  private paused = false;
  private destroyed = false;
  private camPose: CamPose;
  private camMode: CamMode = 'bat';
  private camT = 0;
  private ballCam: Vec3 = { x: 0, y: 60, z: 0 };

  // batting input
  private aim = { x: 0, z: 2.2 };
  private swingKind: SwingKind = 'normal';
  private armed = false;
  private lastAimInput = -9;
  // pitching input
  private pitchType: PitchType = 'fastball';
  private pitchAim = { x: 0, z: 2 };
  // pointer
  private dragging: { id: number; x: number; y: number; moved: number; touch: boolean } | null = null;

  // dom
  private hud!: HTMLElement;
  private sbEl!: HTMLElement;
  private cardsEl!: HTMLElement;
  private tickerEl!: HTMLElement;
  private bubbleEl!: HTMLElement;
  private controlsEl!: HTMLElement;
  private bannerEl!: HTMLElement;
  private hintEl!: HTMLElement;
  private controlsKey = '';
  private cardsKey = '';
  private tickerQueue: Line[] = [];
  private tickerT = 0;
  private bubbleT = 0;
  private bounceSfxT = 0;
  private lastFoulBack = 0;
  /** which side the player controls (fixed at the start, even if they sim the rest) */
  private humanSide: -1 | 0 | 1;

  constructor(container: HTMLElement, opts: GameOptions) {
    this.opts = opts;
    this.match = new Match({ ...opts.cfg, autoThrowDelay: settings.autoThrow });
    this.humanSide = opts.cfg.home.human ? 1 : opts.cfg.away.human ? 0 : -1;
    this.booth = new Booth(opts.cfg.seed);
    this.root = h('div', { class: 'game' });
    container.appendChild(this.root);
    const canvas = h('canvas', { class: 'scene' });
    this.root.appendChild(canvas);
    this.renderer = new SceneRenderer(canvas);
    this.renderer.field = this.match.field;
    this.camPose = OVERVIEW_CAM;
    this.camMode = 'overview';
    this.buildHud();
    this.resize();
    window.addEventListener('resize', this.onResize);
    window.addEventListener('keydown', this.onKey);
    canvas.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('pointercancel', this.onPointerUp);
    audio.playMusic('game');
    audio.setAmbience(true);
    this.showIntro();
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.frame);
  }

  destroy() {
    this.destroyed = true;
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.onResize);
    window.removeEventListener('keydown', this.onKey);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('pointercancel', this.onPointerUp);
    audio.setAmbience(false);
    window.speechSynthesis?.cancel?.();
    this.root.remove();
  }

  private onResize = () => this.resize();

  private resize() {
    const r = this.root.getBoundingClientRect();
    this.renderer.resize(Math.max(320, r.width || window.innerWidth), Math.max(200, r.height || window.innerHeight));
  }

  // ─────────────────────────────────────────────────────────── main loop

  private frame = (now: number) => {
    if (this.destroyed) return;
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    if (!this.paused) {
      this.time += dt;
      this.updateAimAssist(dt);
      if (this.introT <= 0) this.match.update(dt);
      this.handleEvents();
      this.updateCamera(dt);
      this.updateHud(dt);
    }
    this.renderer.cam.pose = this.camPose;
    this.renderer.draw(this.buildFrame(), this.paused ? 0 : dt);
    this.raf = requestAnimationFrame(this.frame);
  };

  // ─────────────────────────────────────────────────────────── camera

  private updateCamera(dt: number) {
    const m = this.match;
    let mode: CamMode = 'bat';
    if (m.phase === 'live' || (m.phase === 'result' && m.lastPlay)) mode = 'field';
    if (m.phase === 'halfOver' || m.phase === 'over') mode = 'overview';
    if (this.introT > 0) mode = 'overview';
    if (mode !== this.camMode) { this.camMode = mode; this.camT = 0; }
    this.camT += dt;
    let target: CamPose;
    if (mode === 'bat') target = batCam(m.batterSide);
    else if (mode === 'field') {
      const play = m.play ?? m.lastPlay;
      if (play) {
        const b = play.ball.p;
        const k = 1 - Math.exp(-dt * 3);
        this.ballCam = { x: this.ballCam.x + (b.x - this.ballCam.x) * k, y: this.ballCam.y + (b.y - this.ballCam.y) * k, z: this.ballCam.z + (b.z - this.ballCam.z) * k };
      }
      target = followCam(this.ballCam.x, this.ballCam.y, this.ballCam.z);
    } else {
      // sway around the outfield side so we never fly through the house
      const a = Math.sin(this.time * 0.12) * 1.1;
      target = { ...OVERVIEW_CAM, pos: { x: Math.sin(a) * 190, y: 60 + Math.cos(a) * 190, z: 75 }, target: { x: 0, y: 45, z: 0 } };
    }
    const speed = mode === 'bat' ? (this.camT < 0.6 ? 5 : 12) : mode === 'field' ? 4 : 2;
    const k = 1 - Math.exp(-dt * speed);
    this.camPose = lerpPose(this.camPose, target, k);
    if (mode === 'bat' && this.camT > 1.2) this.camPose = target;
    if (mode !== 'field') this.ballCam = { x: 0, y: 40, z: 0 };
  }

  // ───────────────────────────────────────────────────── scene building

  private teamOf(side: 0 | 1): Team {
    return this.match.side(side).team;
  }

  private buildFrame(): SceneFrame {
    const m = this.match;
    const f = m.field;
    const kids: KidSprite[] = [];
    const defTeam = this.teamOf(m.fieldingSide);
    const batTeam = this.teamOf(m.battingSide);
    let ball: Vec3 | null = null;
    const play: LivePlay | null = m.play ?? (m.phase === 'result' ? m.lastPlay : null);

    if (play && m.phase !== 'halfOver') {
      for (const fl of play.fielders) {
        kids.push({
          kid: fl.kid, team: defTeam, x: fl.p.x, y: fl.p.y, facing: fl.facing,
          pose: { anim: FIELDER_ANIM[fl.anim], t: fl.animT + fl.idx * 0.37, ball: fl.hasBall, lift: fl.lift },
          ring: fl.hasBall && m.humanPitching ? '#ffe14d' : undefined,
        });
      }
      for (const r of play.runners) {
        if (r.scored && r.animT > 1.6) continue;
        if (r.out && r.animT > 1.6) continue;
        const p = play.runnerPos(r);
        const nb = play.basePos(r.base + 1), pb = play.basePos(r.base);
        const dirx = (r.dir >= 0 ? nb.x - pb.x : pb.x - nb.x), diry = (r.dir >= 0 ? nb.y - pb.y : pb.y - nb.y);
        const facing = r.dir === 0 ? Math.PI : Math.atan2(dirx, diry);
        kids.push({
          kid: r.kid, team: batTeam, x: p.x + (r.scored ? -3 : 0), y: p.y, facing,
          pose: { anim: r.dir === 0 && !r.out && !r.scored ? 'ready' : RUNNER_ANIM[r.anim], t: r.animT + this.time, lean: Math.sign(dirx) },
        });
      }
      if (play.mode === 'held' && play.holder >= 0) {
        const fl = play.fielders[play.holder];
        ball = { x: fl.p.x + 0.6, y: fl.p.y, z: kidHeightFt(fl.kid.look.height) * 0.62 + fl.lift };
      } else if (!(play.deadKind === 'hr' && play.ball.dead && play.t > 8)) {
        ball = { ...play.ball.p };
      }
    } else if (m.phase !== 'halfOver' && m.phase !== 'over') {
      const def = m.defenseKids();
      def.forEach((k, i) => {
        const pos = FIELD_ORDER[i];
        if (pos === 'P' || pos === 'C') return;
        const s = f.defaultSpots[pos];
        kids.push({ kid: k, team: defTeam, x: s.x, y: s.y, facing: Math.PI, pose: { anim: 'ready', t: this.time + i * 0.4 } });
      });
      // pitcher
      const p = m.pitcher;
      let windup = 0;
      if (m.phase === 'windup') windup = m.phaseT / WINDUP;
      else if (m.phase === 'pitch' || m.phase === 'result') windup = 1 + Math.min(1, m.pitchT * 2);
      kids.push({ kid: p, team: defTeam, x: 0, y: f.mound.y - 0.5, facing: Math.PI, pose: { anim: 'pitch', t: this.time, windup } });
      // catcher
      kids.push({ kid: def[1], team: defTeam, x: 0, y: -4.2, facing: 0, pose: { anim: 'crouch', t: this.time } });
      // batter
      const b = m.batter;
      const side = m.batterSide;
      let swing = 0;
      const s = m.swingIn;
      if (s && (m.phase === 'pitch' || m.phase === 'result')) {
        const st = s.kind === 'bunt' ? 0 : (m.pitchT - s.tSwing) / SWING_TIME[s.kind];
        swing = s.kind === 'bunt' ? 0 : clamp(st, 0, 2);
      }
      kids.push({
        kid: b, team: batTeam, x: side === 'R' ? -2.4 : 2.4, y: 0.3, facing: side === 'R' ? Math.PI / 2 : -Math.PI / 2,
        pose: { anim: s?.kind === 'bunt' || (this.swingKind === 'bunt' && m.humanBatting) ? 'bunt' : 'bat', t: this.time, swing, view: 'back', batLeft: side === 'L' },
      });
      // runners leading off
      m.bases.forEach((id, i) => {
        if (!id) return;
        const a = f.bases[i + 1], nb = f.bases[(i + 2) % 4];
        const u = 4 / f.base;
        kids.push({ kid: kid(id), team: batTeam, x: a.x + (nb.x - a.x) * u, y: a.y + (nb.y - a.y) * u, facing: Math.PI, pose: { anim: 'ready', t: this.time + i } });
      });
      if (m.phase === 'pitch' || m.phase === 'result') {
        const bp = m.pitchBallPos();
        if (bp && bp.y > -3.5) ball = bp;
      } else if (m.phase === 'windup' && m.phaseT > WINDUP * 0.6) {
        ball = { x: m.pitcher.throws === 'R' ? -1.5 : 1.5, y: f.mound.y - 1.5, z: 5.5 };
      }
    }
    if (m.phase === 'halfOver' || m.phase === 'over') {
      // everyone mills around the infield
      const roster = m.phase === 'over' && m.winner !== null && m.winner >= 0 ? m.side(m.winner as 0 | 1) : m.side(m.fieldingSide);
      roster.lineup.order.forEach((id, i) => {
        const a = (i / 9) * Math.PI * 2 + this.time * 0.2;
        kids.push({ kid: kid(id), team: roster.team, x: Math.cos(a) * 14, y: f.mound.y + Math.sin(a) * 10, facing: a + Math.PI / 2, pose: { anim: m.phase === 'over' ? 'cheer' : 'run', t: this.time + i, lean: 1 } });
      });
    }
    return { kids, ball, t: this.time, overlay: (ctx, cam) => this.drawOverlay(ctx, cam) };
  }

  private drawOverlay(ctx: CanvasRenderingContext2D, cam: SceneRenderer['cam']) {
    const m = this.match;
    if (this.camMode !== 'bat') {
      this.drawBaseTargets(ctx, cam);
      return;
    }
    const z = m.zone;
    const pz = (x: number, zz: number) => cam.project(x, 0.2, zz);
    const tl = pz(-z.half, z.top), br = pz(z.half, z.bottom);
    const pitchingNow = m.humanPitching && m.phase === 'prePitch';
    if (tl && br && (settings.showZone || pitchingNow)) {
      ctx.save();
      ctx.strokeStyle = pitchingNow ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.55)';
      ctx.lineWidth = 2;
      ctx.setLineDash(pitchingNow ? [] : [6, 5]);
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      if (pitchingNow) {
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.fillRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        for (let i = 1; i < 3; i++) {
          const x = tl.x + ((br.x - tl.x) * i) / 3, y = tl.y + ((br.y - tl.y) * i) / 3;
          ctx.beginPath(); ctx.moveTo(x, tl.y); ctx.lineTo(x, br.y); ctx.moveTo(tl.x, y); ctx.lineTo(br.x, y); ctx.stroke();
        }
      }
      ctx.restore();
    }
    // pitch target (the catcher's mitt)
    if (m.humanPitching && (m.phase === 'prePitch' || m.phase === 'windup')) {
      const p = pz(this.pitchAim.x, this.pitchAim.z);
      if (p) {
        ctx.save();
        ctx.fillStyle = 'rgba(139,90,43,0.85)';
        ctx.strokeStyle = INK;
        ctx.lineWidth = 2;
        const r = Math.max(10, 0.32 * p.s);
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.strokeStyle = '#ffe14d';
        ctx.beginPath(); ctx.moveTo(p.x - r * 1.5, p.y); ctx.lineTo(p.x + r * 1.5, p.y); ctx.moveTo(p.x, p.y - r * 1.5); ctx.lineTo(p.x, p.y + r * 1.5); ctx.stroke();
        ctx.restore();
      }
    }
    // the bat's sweet spot
    if (m.humanBatting && (m.phase === 'prePitch' || m.phase === 'windup' || m.phase === 'pitch')) {
      const p = pz(this.aim.x, this.aim.z);
      if (p) {
        const tune = { window: 1, radius: m.cfg.difficulty === 'rookie' ? 1.3 : m.cfg.difficulty === 'pro' ? 1.1 : 1 };
        const { r } = contactWindow(m.batter, { aimX: 0, aimZ: 0, tSwing: 0, kind: this.swingKind, special: this.armed }, tune);
        const rad = Math.max(8, r * p.s);
        ctx.save();
        ctx.lineWidth = 3;
        ctx.strokeStyle = this.swingKind === 'power' ? '#ff5e5b' : this.swingKind === 'bunt' ? '#6fc3ff' : '#ffe14d';
        ctx.fillStyle = 'rgba(255,225,77,0.12)';
        ctx.beginPath();
        if (this.swingKind === 'bunt') ctx.ellipse(p.x, p.y, rad * 1.6, rad * 0.5, 0, 0, Math.PI * 2);
        else ctx.arc(p.x, p.y, rad, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fillStyle = ctx.strokeStyle; ctx.fill();
        ctx.restore();
      }
    }
  }

  private drawBaseTargets(ctx: CanvasRenderingContext2D, cam: SceneRenderer['cam']) {
    const m = this.match;
    const play = m.play;
    if (!play || !m.humanPitching || play.deadKind) return;
    const f = m.field;
    ctx.save();
    for (let b = 1; b <= 4; b++) {
      const bp = f.bases[b % 4];
      const p = cam.project(bp.x, bp.y, 0.2);
      if (!p) continue;
      const sel = play.throwRequest === b;
      const pulse = 1 + Math.sin(this.time * 8) * 0.08;
      const r = Math.max(16, 2.6 * p.s) * pulse;
      ctx.beginPath(); ctx.ellipse(p.x, p.y, r, r * 0.55, 0, 0, Math.PI * 2);
      ctx.fillStyle = sel ? 'rgba(255,225,77,0.55)' : 'rgba(255,255,255,0.22)';
      ctx.fill();
      ctx.lineWidth = 3; ctx.strokeStyle = sel ? '#ffe14d' : 'rgba(255,255,255,0.85)';
      ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.strokeStyle = INK; ctx.lineWidth = 3;
      ctx.font = '900 14px "Trebuchet MS", sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const lab = b === 4 ? 'H' : String(b);
      ctx.strokeText(lab, p.x, p.y - r * 0.9); ctx.fillText(lab, p.x, p.y - r * 0.9);
    }
    ctx.restore();
  }

  // ─────────────────────────────────────────────────────────── input

  private updateAimAssist(dt: number) {
    const m = this.match;
    if (!m.humanBatting) return;
    if (m.phase === 'prePitch') {
      // drift back toward the middle of the zone between pitches
      const z = m.zone;
      if (this.time - this.lastAimInput > 0.8) {
        this.aim.x += (0 - this.aim.x) * Math.min(1, dt * 2);
        this.aim.z += ((z.top + z.bottom) / 2 - this.aim.z) * Math.min(1, dt * 2);
      }
      return;
    }
    if (m.phase !== 'pitch' || !m.pitch || m.swingIn) return;
    const mode = settings.aimAssist;
    const strength = mode === 'off' ? 0 : mode === 'on' ? 0.85 : m.cfg.difficulty === 'rookie' ? 0.9 : m.cfg.difficulty === 'pro' ? 0.45 : 0;
    if (strength <= 0) return;
    const arr = m.pitch.arrival;
    const k = Math.min(1, dt * 9 * strength);
    this.aim.x += (arr.x - this.aim.x) * k;
    this.aim.z += (arr.z - this.aim.z) * k;
  }

  private doSwing() {
    const m = this.match;
    if (!m.humanBatting || m.swingIn) return;
    if (m.phase !== 'windup' && m.phase !== 'pitch') return;
    audio.unlock();
    m.swing(this.aim.x, this.aim.z, this.swingKind, this.armed);
    if (this.armed) this.armed = false;
  }

  private doPitch() {
    const m = this.match;
    if (!m.humanPitching || m.phase !== 'prePitch') return;
    audio.unlock();
    m.selectPitch(this.pitchType, { ...this.pitchAim }, this.armed);
    this.armed = false;
  }

  private screenToZone(sx: number, sy: number) {
    const hit = this.renderer.cam.rayToPlaneY(sx, sy, 0.2);
    if (!hit) return null;
    return { x: clamp(hit.x, -2.4, 2.4), z: clamp(hit.z, 0, 5.5) };
  }

  private onPointerDown = (e: PointerEvent) => {
    audio.unlock();
    const m = this.match;
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const touch = e.pointerType !== 'mouse';
    this.dragging = { id: e.pointerId, x: sx, y: sy, moved: 0, touch };
    if (this.introT > 0) { this.introT = 0; return; }
    if (m.phase === 'live' && m.humanPitching && m.play) {
      // tap near a base to throw there
      const f = m.field;
      let best = -1, bestD = Infinity;
      for (let b = 1; b <= 4; b++) {
        const bp = f.bases[b % 4];
        const p = this.renderer.cam.project(bp.x, bp.y, 0);
        if (!p) continue;
        const d = Math.hypot(p.x - sx, p.y - sy);
        if (d < bestD) { bestD = d; best = b; }
      }
      if (best > 0 && bestD < 70) this.throwTo(best);
      return;
    }
    if (m.humanPitching && m.phase === 'prePitch' && this.camMode === 'bat') {
      const z = this.screenToZone(sx, sy);
      if (z) {
        this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
        if (!touch) this.doPitch();
      }
      return;
    }
    if (m.humanBatting && this.camMode === 'bat') {
      if (!touch) {
        const z = this.screenToZone(sx, sy);
        if (z) { this.aim = z; this.lastAimInput = this.time; }
        this.doSwing();
      }
    }
  };

  private onPointerMove = (e: PointerEvent) => {
    const m = this.match;
    const canvas = this.renderer.canvas;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const d = this.dragging;
    if (e.pointerType === 'mouse' && !d) {
      if (m.humanBatting && this.camMode === 'bat' && (m.phase !== 'pitch' || settings.aimAssist === 'off' || m.cfg.difficulty === 'allstar')) {
        const z = this.screenToZone(sx, sy);
        if (z) { this.aim = z; this.lastAimInput = this.time; }
      }
      if (m.humanPitching && m.phase === 'prePitch') {
        const z = this.screenToZone(sx, sy);
        if (z) this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
      }
      return;
    }
    if (!d || d.id !== e.pointerId) return;
    const dx = sx - d.x, dy = sy - d.y;
    d.moved += Math.hypot(dx, dy);
    d.x = sx; d.y = sy;
    if (!d.touch) return;
    // touch drags nudge the aim (relative, like a trackpad)
    const p = this.renderer.cam.project(0, 0.2, 2);
    const ppf = p ? p.s : 40;
    if (m.humanBatting && this.camMode === 'bat') {
      this.aim.x = clamp(this.aim.x + (dx / ppf) * 0.9, -2.4, 2.4);
      this.aim.z = clamp(this.aim.z - (dy / ppf) * 0.9, 0, 5.5);
      this.lastAimInput = this.time;
    } else if (m.humanPitching && m.phase === 'prePitch') {
      const z = this.screenToZone(sx, sy);
      if (z) this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
    }
  };

  private onPointerUp = (e: PointerEvent) => {
    const d = this.dragging;
    if (!d || d.id !== e.pointerId) return;
    this.dragging = null;
  };

  private onKey = (e: KeyboardEvent) => {
    const m = this.match;
    const k = e.key;
    if (k === 'Escape' || k === 'p' && !m.humanBatting) { this.togglePause(); return; }
    if (this.paused) return;
    audio.unlock();
    if (this.introT > 0) { this.introT = 0; return; }
    const step = 0.2;
    if (m.humanBatting && this.camMode === 'bat') {
      if (k === ' ' || k === 'Enter') { e.preventDefault(); this.doSwing(); }
      else if (k === 'ArrowLeft') this.nudgeAim(-step, 0);
      else if (k === 'ArrowRight') this.nudgeAim(step, 0);
      else if (k === 'ArrowUp') this.nudgeAim(0, step);
      else if (k === 'ArrowDown') this.nudgeAim(0, -step);
      else if (k === 'p' || k === 'P') this.setKind(this.swingKind === 'power' ? 'normal' : 'power');
      else if (k === 'b' || k === 'B') this.setKind(this.swingKind === 'bunt' ? 'normal' : 'bunt');
      else if (k === 's' || k === 'S') this.toggleSpecial();
    }
    if (m.humanPitching && m.phase === 'prePitch') {
      const pitches = m.pitcher.pitches;
      const n = Number(k);
      if (n >= 1 && n <= pitches.length) this.pitchType = pitches[n - 1];
      else if (k === 'ArrowLeft') this.pitchAim.x = clamp(this.pitchAim.x - step, -1.6, 1.6);
      else if (k === 'ArrowRight') this.pitchAim.x = clamp(this.pitchAim.x + step, -1.6, 1.6);
      else if (k === 'ArrowUp') this.pitchAim.z = clamp(this.pitchAim.z + step, 0.4, 4.2);
      else if (k === 'ArrowDown') this.pitchAim.z = clamp(this.pitchAim.z - step, 0.4, 4.2);
      else if (k === ' ' || k === 'Enter') { e.preventDefault(); this.doPitch(); }
      else if (k === 's' || k === 'S') this.toggleSpecial();
    }
    if (m.phase === 'live' && m.play) {
      if (m.humanPitching) {
        const map: Record<string, number> = { '1': 1, '2': 2, '3': 3, '4': 4, ArrowRight: 1, ArrowUp: 2, ArrowLeft: 3, ArrowDown: 4 };
        if (map[k]) { e.preventDefault(); this.throwTo(map[k]); }
      } else if (m.humanBatting) {
        if (k === 'r' || k === 'R' || k === 'ArrowRight') m.runners('advance');
        if (k === 'f' || k === 'F' || k === 'ArrowLeft') m.runners('retreat');
      }
    }
  };

  private nudgeAim(dx: number, dz: number) {
    this.aim.x = clamp(this.aim.x + dx, -2.4, 2.4);
    this.aim.z = clamp(this.aim.z + dz, 0, 5.5);
    this.lastAimInput = this.time;
  }

  private setKind(kind: SwingKind) {
    this.swingKind = kind;
    audio.play('uiTap');
    this.controlsKey = '';
  }

  private toggleSpecial() {
    const m = this.match;
    const side = m.humanBatting ? m.battingSide : m.fieldingSide;
    const who = m.humanBatting ? m.batter : m.pitcher;
    if (!m.canSpecial(side, who)) return;
    this.armed = !this.armed;
    audio.play(this.armed ? 'special' : 'uiBack');
    this.controlsKey = '';
  }

  private throwTo(base: number) {
    this.match.throwTo(base);
    audio.play('uiTap');
  }

  // ─────────────────────────────────────────────────────────── events

  private handleEvents() {
    const m = this.match;
    const r = this.renderer;
    const humanSide = m.cfg.home.human ? 1 : m.cfg.away.human ? 0 : -1;
    const good = (battingGood: boolean) => (humanSide < 0 ? true : (m.battingSide === humanSide) === battingGood);
    for (const e of m.events) {
      this.say(this.booth.react(e, m));
      switch (e.type) {
        case 'pitch':
          if (e.special) { r.popup(SPECIAL_INFO[e.special].label.toUpperCase() + '!', '#c39bff', 0.9, 1.3); audio.play('special'); }
          else audio.play('throw', { intensity: 0.4 });
          break;
        case 'special':
          audio.play('special');
          break;
        case 'contact': {
          const strong = e.quality > 0.55 && e.ev > 55;
          audio.play(strong ? 'batCrack' : 'batTink', { intensity: clamp((e.ev - 30) / 60, 0, 1) });
          break;
        }
        case 'whiff':
          audio.play('whiff');
          break;
        case 'call':
          if (e.call === 'ball') { r.popup('BALL', '#9fd3ff', 0.6, 0.8); audio.play('mittPop', { intensity: 0.5 }); }
          else if (e.call === 'strike' || e.call === 'swinging') {
            if (m.strikes < 3) r.popup('STRIKE!', '#ffe14d', 0.8, 0.9);
            audio.play('mittPop', { intensity: 0.8 });
            audio.play('strike');
          } else if (e.call === 'foul') {
            if (this.time - this.lastFoulBack > 0.3) r.popup('FOUL!', '#ffffff', 0.7, 0.9);
            this.lastFoulBack = this.time;
          }
          break;
        case 'strikeout':
          r.popup(e.looking ? 'STRIKE THREE!' : 'STRUCK OUT!', '#ffe14d', 1, 1.4);
          audio.play(good(false) ? 'cheer' : 'aww');
          break;
        case 'walk':
          r.popup(e.hbp ? 'OUCH!' : 'BALL FOUR', '#9fd3ff', 0.85, 1.2);
          break;
        case 'catch':
          audio.play('catch', { intensity: e.hard ? 1 : 0.6 });
          if (e.fly && e.hard) r.popup('WHAT A GRAB!', '#7dff9a', 0.9, 1.4, 0.5, 0.28);
          break;
        case 'bobble':
          audio.play('aww');
          r.popup('BOBBLE!', '#ffb36b', 0.7, 1);
          break;
        case 'throw':
          audio.play('throw', { intensity: 0.7 });
          break;
        case 'out':
          r.popup('OUT!', '#ff6b6b', 0.9, 1);
          audio.play('out');
          break;
        case 'safe':
          break;
        case 'run':
          r.popup('RUN SCORES!', '#7dff9a', 0.75, 1.2, 0.5, 0.22);
          audio.play('safe');
          break;
        case 'hit': {
          const label = e.bases >= 3 ? 'TRIPLE!' : e.bases === 2 ? 'DOUBLE!' : 'BASE HIT!';
          r.popup(label, '#7dff9a', 0.95, 1.3);
          audio.play(good(true) ? 'cheer' : 'aww');
          break;
        }
        case 'homeRun':
          r.popup('HOME RUN!', '#ffe14d', 1.3, 2.2);
          audio.play('homeRun');
          audio.play(good(true) ? 'bigCheer' : 'aww');
          break;
        case 'groundRule':
          r.popup(e.why === 'splash' ? 'SPLASH DOUBLE!' : 'GROUND-RULE DOUBLE', '#6fc3ff', 1, 1.6);
          break;
        case 'splash':
          audio.play('splash');
          break;
        case 'error':
          r.popup('E! OOPS!', '#ffb36b', 0.8, 1.2, 0.5, 0.5);
          break;
        case 'bounce':
          if (this.time - this.bounceSfxT > 0.12) { audio.play('bounce', { intensity: clamp(e.speed / 40, 0, 1) }); this.bounceSfxT = this.time; }
          if (e.speed > 6) r.puff(e.x, e.y, 0, 1 + e.speed * 0.04, e.surface === 'grass' ? 'rgba(120,180,90,0.6)' : 'rgba(190,150,100,0.6)');
          break;
        case 'fence':
          audio.play('fence', { intensity: 0.8 });
          break;
        case 'tree':
          audio.play('leaves');
          break;
        case 'dog':
          audio.play('dogBark');
          r.popup('WOOF!', '#ffffff', 0.8, 1, 0.3, 0.4);
          break;
        case 'quip':
          this.showBubble(kid(e.kid), e.text);
          break;
        case 'halfOver':
          audio.play('whistle');
          this.showBanner(`${e.half === 0 ? 'Middle' : 'End'} of the ${ordinal(e.inning)}`, `${m.cfg.away.team.abbr} ${m.score[0]} — ${m.cfg.home.team.abbr} ${m.score[1]}`);
          break;
        case 'gameOver':
          this.onGameOver();
          break;
        default:
          break;
      }
    }
    m.events.length = 0;
  }

  private say(lines: Line[]) {
    if (!lines.length) return;
    this.tickerQueue.push(...lines);
    if (this.tickerQueue.length > 4) this.tickerQueue.splice(0, this.tickerQueue.length - 4);
  }

  // ─────────────────────────────────────────────────────────── HUD

  private buildHud() {
    this.sbEl = h('div', { class: 'sb' });
    this.cardsEl = h('div', { class: 'cards' });
    this.tickerEl = h('div', { class: 'ticker' });
    this.bubbleEl = h('div', { class: 'bubble hidden' });
    this.controlsEl = h('div', { class: 'controls' });
    this.bannerEl = h('div', { class: 'banner hidden', onpointerdown: () => { audio.unlock(); if (this.introT > 0) { this.introT = 0; this.bannerEl.classList.add('hidden'); } } });
    this.hintEl = h('div', { class: 'hint' });
    const pause = h('button', { class: 'btn icon pause', 'aria-label': 'Pause', onclick: () => this.togglePause() }, '❚❚');
    this.hud = h('div', { class: 'hud' }, this.sbEl, pause, this.cardsEl, this.tickerEl, this.bubbleEl, this.hintEl, this.controlsEl, this.bannerEl);
    this.root.appendChild(this.hud);
  }

  private updateHud(dt: number) {
    const m = this.match;
    this.renderScoreboard();
    // matchup cards (rebuilt only when the batter/pitcher changes)
    const key = `${m.batter.id}|${m.pitcher.id}|${m.half}`;
    if (key !== this.cardsKey && m.phase !== 'halfOver') {
      this.cardsKey = key;
      clear(this.cardsEl);
      const bl = m.box[m.batter.id]?.bat;
      this.cardsEl.append(
        this.card(m.batter, this.teamOf(m.battingSide), 'AT BAT', bl && bl.ab ? `${bl.h}-for-${bl.ab} today` : m.batter.persona),
        this.card(m.pitcher, this.teamOf(m.fieldingSide), 'PITCHING', m.pitcher.pitches.map((p) => PITCHES[p].short).join(' · ')),
      );
    }
    // ticker
    this.tickerT -= dt;
    if (this.tickerT <= 0 && this.tickerQueue.length) {
      const line = this.tickerQueue.shift()!;
      clear(this.tickerEl);
      this.tickerEl.append(h('b', { class: line.who === 'Chet' ? 'chet' : 'dottie' }, line.who === 'Chet' ? 'CHET: ' : 'DOTTIE: '), line.text);
      this.tickerT = Math.max(2.2, line.text.length * 0.055);
      if (settings.voice) speak(line);
    }
    // bubble
    if (this.bubbleT > 0) {
      this.bubbleT -= dt;
      if (this.bubbleT <= 0) this.bubbleEl.classList.add('hidden');
    }
    if (this.bannerT > 0) {
      this.bannerT -= dt;
      if (this.bannerT <= 0 && m.phase !== 'over') this.bannerEl.classList.add('hidden');
    }
    if (this.introT > 0) {
      this.introT -= dt;
      if (this.introT <= 0) this.bannerEl.classList.add('hidden');
    } else if (!this.bannerEl.classList.contains('hidden') && this.bannerT <= 0 && this.match.phase !== 'over') {
      this.bannerEl.classList.add('hidden');
    }
    this.renderControls();
  }

  private card(k: Kid, team: Team, label: string, sub: string) {
    return h('div', { class: 'card', style: `--team:${team.colors.primary};--team2:${team.colors.secondary}` },
      portraitCanvas(k, team, 54, 54),
      h('div', { class: 'card-txt' },
        h('div', { class: 'card-label' }, label),
        h('div', { class: 'card-name' }, k.nick),
        h('div', { class: 'card-sub' }, sub)));
  }

  private sbKey = '';
  private renderScoreboard() {
    const m = this.match;
    const key = [m.score.join(), m.inning, m.half, m.outs, m.balls, m.strikes, m.bases.map((b) => (b ? 1 : 0)).join(''), Math.floor(m.hype[0] / 10), Math.floor(m.hype[1] / 10), m.phase === 'live'].join('|');
    if (key === this.sbKey) return;
    this.sbKey = key;
    clear(this.sbEl);
    const row = (side: 0 | 1) => {
      const t = this.teamOf(side);
      const bat = m.battingSide === side;
      return h('div', { class: `sb-row${bat ? ' bat' : ''}`, style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
        logoCanvas(t, 22),
        h('span', { class: 'sb-abbr' }, t.abbr),
        h('span', { class: 'sb-hype', title: 'Hype' }, h('i', { style: `width:${m.hype[side]}%` })),
        h('span', { class: 'sb-runs' }, String(m.score[side])));
    };
    const dots = (n: number, of: number, cls: string) => h('span', { class: `dots ${cls}` }, ...Array.from({ length: of }, (_, i) => h('i', { class: i < n ? 'on' : '' })));
    const diamond = h('div', { class: 'diamond' },
      ...[1, 2, 3].map((b) => h('i', { class: `b${b}${m.bases[b - 1] ? ' on' : ''}` })));
    this.sbEl.append(
      h('div', { class: 'sb-teams' }, row(0), row(1)),
      h('div', { class: 'sb-state' },
        h('div', { class: 'sb-inning' }, `${m.half === 0 ? '▲' : '▼'} ${m.inning}`),
        diamond,
        h('div', { class: 'sb-count' }, h('span', null, 'B'), dots(m.balls, 3, 'balls'), h('span', null, 'S'), dots(m.strikes, 2, 'strikes')),
        h('div', { class: 'sb-count' }, h('span', null, 'O'), dots(m.outs, 2, 'outs'))));
  }

  private renderControls() {
    const m = this.match;
    let mode = 'none';
    if (this.introT > 0) mode = 'intro';
    else if (m.phase === 'live' && m.play && !m.play.deadKind) mode = m.humanPitching ? 'field' : m.humanBatting ? 'run' : 'none';
    else if (m.humanBatting && ['prePitch', 'windup', 'pitch'].includes(m.phase)) mode = 'bat';
    else if (m.humanPitching && m.phase === 'prePitch') mode = 'pitch';
    const canSp = mode === 'bat' ? m.canSpecial(m.battingSide, m.batter) : mode === 'pitch' ? m.canSpecial(m.fieldingSide, m.pitcher) : false;
    const key = `${mode}|${this.swingKind}|${this.pitchType}|${this.armed}|${canSp}|${m.pitcher.id}|${m.batter.id}`;
    if (key === this.controlsKey) return;
    this.controlsKey = key;
    clear(this.controlsEl);
    const touch = matchMedia('(pointer: coarse)').matches;
    const hint = (s: string) => { this.hintEl.textContent = s; };
    const special = (k: Kid) => canSp ? h('button', { class: `btn special${this.armed ? ' armed' : ''}`, onpointerdown: (e: Event) => { e.preventDefault(); this.toggleSpecial(); } }, `⚡ ${SPECIAL_INFO[k.special].label}`) : null;
    switch (mode) {
      case 'bat': {
        hint(touch ? 'Drag to aim · tap SWING when the ball arrives' : 'Aim with the mouse · click (or Space) to swing');
        const kindBtn = (k: SwingKind, label: string) => h('button', {
          class: `btn small${this.swingKind === k ? ' on' : ''}`,
          onpointerdown: (e: Event) => { e.preventDefault(); this.setKind(this.swingKind === k ? 'normal' : k); },
        }, label);
        this.controlsEl.append(h('div', { class: 'ctl-col' }, special(m.batter), kindBtn('power', 'POWER'), kindBtn('bunt', 'BUNT')));
        if (touch) this.controlsEl.append(h('button', { class: 'btn swing', onpointerdown: (e: Event) => { e.preventDefault(); this.doSwing(); } }, 'SWING!'));
        break;
      }
      case 'pitch': {
        hint(touch ? 'Pick a pitch · tap the zone to aim · THROW!' : 'Pick a pitch (1-3) · click the zone to throw');
        const pitches = m.pitcher.pitches;
        if (!pitches.includes(this.pitchType)) this.pitchType = pitches[0];
        this.controlsEl.append(
          h('div', { class: 'ctl-col' }, special(m.pitcher),
            ...pitches.map((p, i) => h('button', {
              class: `btn small${this.pitchType === p ? ' on' : ''}`,
              onpointerdown: (e: Event) => { e.preventDefault(); this.pitchType = p; audio.play('uiTap'); this.controlsKey = ''; },
            }, `${i + 1} ${PITCHES[p].label}`))),
        );
        if (touch) this.controlsEl.append(h('button', { class: 'btn swing', onpointerdown: (e: Event) => { e.preventDefault(); this.doPitch(); } }, 'THROW!'));
        break;
      }
      case 'field': {
        hint('Tap a base to throw there (or let your kid decide)');
        const base = (b: number, label: string) => h('button', { class: 'btn small', onpointerdown: (e: Event) => { e.preventDefault(); this.throwTo(b); } }, label);
        this.controlsEl.append(h('div', { class: 'ctl-bases' }, base(2, '2nd'), h('div', null, base(3, '3rd'), base(1, '1st')), base(4, 'Home')));
        break;
      }
      case 'run':
        hint('Runners run on their own — or take charge!');
        this.controlsEl.append(h('div', { class: 'ctl-col' },
          h('button', { class: 'btn small', onpointerdown: (e: Event) => { e.preventDefault(); m.runners('advance'); audio.play('uiTap'); } }, 'RUN! ▶'),
          h('button', { class: 'btn small', onpointerdown: (e: Event) => { e.preventDefault(); m.runners('retreat'); audio.play('uiTap'); } }, '◀ BACK!')));
        break;
      case 'intro':
        hint('Tap to play ball!');
        break;
      default:
        hint(m.phase === 'live' ? '' : m.humanBatting || m.humanPitching ? '' : 'Watching the CPU play');
    }
  }

  private showBubble(k: Kid, text: string) {
    const team = this.match.box[k.id] ? this.teamOf(this.match.box[k.id].side) : null;
    clear(this.bubbleEl);
    this.bubbleEl.append(portraitCanvas(k, team, 64, 64), h('div', { class: 'bubble-txt' }, h('b', null, k.nick), h('span', null, `"${text}"`)));
    this.bubbleEl.classList.remove('hidden');
    this.bubbleT = 2.8;
  }

  private bannerT = 0;
  private introT = 0;

  private showBanner(title: string, sub: string, hold = 2.2) {
    clear(this.bannerEl);
    this.bannerEl.append(h('div', { class: 'banner-title' }, title), h('div', { class: 'banner-sub' }, sub));
    this.bannerEl.classList.remove('hidden');
    this.bannerT = hold;
  }

  private showIntro() {
    const m = this.match;
    const y = m.field.yard;
    clear(this.bannerEl);
    const grownup = y.props.find((p) => p.kind === 'grownup');
    this.bannerEl.append(
      h('div', { class: 'banner-vs' },
        logoCanvas(m.cfg.away.team, 64), h('span', null, 'at'), logoCanvas(m.cfg.home.team, 64)),
      h('div', { class: 'banner-title' }, y.name),
      h('div', { class: 'banner-sub' }, `${y.owner}. ${y.blurb}`),
      h('ul', { class: 'banner-rules' }, ...y.rules.map((r) => h('li', null, r)), grownup?.label ? h('li', null, `Watching: ${grownup.label.replace(/\.$/, '')}.`) : null),
      h('div', { class: 'banner-tap' }, 'Tap to play ball!'));
    this.bannerEl.classList.remove('hidden');
    this.introT = 6;
  }

  private togglePause() {
    if ((this.match.phase as string) === 'over') return;
    this.paused = !this.paused;
    audio.play(this.paused ? 'uiBack' : 'uiTap');
    if (this.paused) this.showPauseMenu();
    else this.root.querySelector('.pausemenu')?.remove();
  }

  private showPauseMenu() {
    const menu = h('div', { class: 'pausemenu overlay' },
      h('div', { class: 'panel' },
        h('h2', null, 'Time Out!'),
        h('button', { class: 'btn', onclick: () => this.togglePause() }, 'Resume'),
        h('button', { class: 'btn ghost', onclick: () => this.simToEnd() }, 'Sim the rest of the game'),
        h('label', { class: 'toggle' }, h('input', { type: 'checkbox', checked: settings.showZone, onchange: (e: Event) => { settings.showZone = (e.target as HTMLInputElement).checked; } }), ' Show strike zone'),
        h('label', { class: 'toggle' }, h('input', { type: 'checkbox', checked: settings.voice, onchange: (e: Event) => { settings.voice = (e.target as HTMLInputElement).checked; } }), ' Announcer voice'),
        h('button', { class: 'btn ghost danger', onclick: () => { this.destroy(); this.opts.onExit(null); } }, 'Quit game')));
    this.root.appendChild(menu);
  }

  private simToEnd() {
    const m = this.match;
    m.cfg.away.human = false;
    m.cfg.home.human = false;
    (m.cfg as { fast?: boolean }).fast = true;
    let guard = 0;
    while (m.phase !== 'over' && guard++ < 400000) {
      m.update(1 / 30);
      if ((m.phase as string) !== 'over') m.events.length = 0;
    }
    this.paused = false;
    this.root.querySelector('.pausemenu')?.remove();
    this.handleEvents();
  }

  private onGameOver() {
    const m = this.match;
    const humanSide = this.humanSide;
    const won = humanSide >= 0 && m.winner === humanSide;
    audio.playMusic('victory');
    if (won) audio.play('bigCheer');
    // player of the game: most bases + runs + RBI, pitcher strikeouts count too
    let star: string | null = null, best = -1;
    for (const [id, l] of Object.entries(m.box)) {
      const tb = l.bat.h + l.bat.d + l.bat.t * 2 + l.bat.hr * 3;
      const score = tb * 2 + l.bat.rbi * 1.5 + l.bat.r + l.pitch.so * 0.8 + (l.side === m.winner ? 1 : 0);
      if (score > best) { best = score; star = id; }
    }
    const line = (side: 0 | 1) => {
      const t = this.teamOf(side);
      const cells = Array.from({ length: Math.max(m.cfg.innings, m.inning) }, (_, i) => h('td', null, m.line[side][i] ?? (i < m.inning ? 'x' : '')));
      return h('tr', null, h('th', null, t.abbr), ...cells, h('td', { class: 'tot' }, String(m.score[side])), h('td', null, String(m.hits[side])), h('td', null, String(m.errors[side])));
    };
    const header = h('tr', null, h('th', null, ''), ...Array.from({ length: Math.max(m.cfg.innings, m.inning) }, (_, i) => h('th', null, String(i + 1))), h('th', null, 'R'), h('th', null, 'H'), h('th', null, 'E'));
    const sk = star ? kid(star) : null;
    const starLine = sk ? (() => {
      const b = m.box[sk.id].bat, p = m.box[sk.id].pitch;
      const parts = [`${b.h}-for-${b.ab}`];
      if (b.hr) parts.push(`${b.hr} HR`);
      if (b.rbi) parts.push(`${b.rbi} RBI`);
      if (p.so) parts.push(`${p.so} K`);
      return parts.join(', ');
    })() : '';
    const title = m.winner === -1 ? 'It\'s a tie!' : humanSide < 0 ? `${this.teamOf(m.winner as 0 | 1).name} win!` : won ? 'You win!' : 'Tough loss!';
    const panel = h('div', { class: 'overlay final' },
      h('div', { class: 'panel wide' },
        h('h2', null, title),
        h('table', { class: 'linescore' }, header, line(0), line(1)),
        sk ? h('div', { class: 'star' }, portraitCanvas(sk, this.teamOf(m.box[sk.id].side), 84, 84),
          h('div', null, h('div', { class: 'card-label' }, 'PLAYER OF THE GAME'), h('div', { class: 'card-name' }, `${sk.first} "${sk.nick}" ${sk.last}`), h('div', { class: 'card-sub' }, starLine), h('div', { class: 'quote' }, `"${sk.quips[0]}"`))) : null,
        h('button', { class: 'btn', onclick: () => { this.destroy(); this.opts.onExit(m); } }, 'Continue')));
    setTimeout(() => { if (!this.destroyed) this.root.appendChild(panel); }, 1800);
  }
}

function speak(line: Line) {
  try {
    const s = window.speechSynthesis;
    if (!s) return;
    const u = new SpeechSynthesisUtterance(line.text.replace(/["“”]/g, ''));
    u.rate = line.who === 'Chet' ? 1.15 : 1.0;
    u.pitch = line.who === 'Chet' ? 1.5 : 1.7;
    u.volume = settings.sfx;
    s.speak(u);
  } catch {
    /* no voice available */
  }
}

