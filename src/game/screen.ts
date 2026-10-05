import { Plane, Raycaster, Vector2, Vector3 } from 'three';
import { clamp } from '../engine/math';
import { audio } from '../audio';
import { kid } from '../data/kids';
import { SPECIAL_INFO, type Kid, type PitchType, type Team } from '../data/types';
import { contactWindow, type SwingKind } from '../sim/batting';
import { Match, type MatchConfig } from '../sim/match';
import { PITCHES } from '../sim/pitching';
import { W } from '../gfx/units';
import { Booth, ordinal, type Line } from '../ui/commentary';
import { clear, h } from '../ui/dom';
import { settings } from '../ui/settings';
import { World } from './world';
import { Director } from './director';
import { PortraitStudio } from './portraits';
import type { Expression } from '../kid3d/face';

export interface GameOptions {
  cfg: MatchConfig;
  /** called when the player leaves; match is null if they quit early */
  onExit: (m: Match | null, again?: boolean) => void;
}

/** A small round team badge. */
export function teamBadge(t: Team, size = 28): HTMLCanvasElement {
  const c = document.createElement('canvas');
  const pr = Math.min(2, window.devicePixelRatio || 1);
  c.width = c.height = size * pr;
  c.style.width = c.style.height = `${size}px`;
  c.className = 'logo';
  const g = c.getContext('2d')!;
  g.scale(pr, pr);
  const r = size / 2;
  g.fillStyle = t.colors.primary;
  g.beginPath(); g.arc(r, r, r - 1.5, 0, Math.PI * 2); g.fill();
  g.lineWidth = Math.max(2, size * 0.08);
  g.strokeStyle = '#2b1d14';
  g.stroke();
  g.strokeStyle = t.colors.secondary;
  g.lineWidth = Math.max(1.5, size * 0.06);
  g.beginPath(); g.arc(r, r, r - size * 0.16, 0, Math.PI * 2); g.stroke();
  g.fillStyle = t.colors.secondary;
  g.font = `900 ${Math.round(size * 0.52)}px Georgia, serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(t.name[0], r, r + size * 0.04);
  return c;
}

export class GameScreen {
  readonly match: Match;
  private director: Director;
  private booth: Booth;
  private root: HTMLElement;
  private canvas: HTMLCanvasElement;
  private raf = 0;
  private last = 0;
  private time = 0;
  private paused = false;
  private destroyed = false;
  private ready = false;

  // batting input
  private aim = { x: 0, z: 2.2 };
  private swingKind: SwingKind = 'normal';
  private armed = false;
  private lastAimInput = -9;
  // pitching input
  private pitchType: PitchType = 'fastball';
  private pitchAim = { x: 0, z: 2 };
  private dragging: { id: number; x: number; y: number; touch: boolean } | null = null;
  private ray = new Raycaster();
  private plate = new Plane(new Vector3(0, 0, 1), 0);

  // dom
  private hud!: HTMLElement;
  private sbEl!: HTMLElement;
  private cardsEl!: HTMLElement;
  private tickerEl!: HTMLElement;
  private bubbleEl!: HTMLElement;
  private controlsEl!: HTMLElement;
  private bannerEl!: HTMLElement;
  private hintEl!: HTMLElement;
  private popEl!: HTMLElement;
  private controlsKey = '';
  private cardsKey = '';
  private sbKey = '';
  private tickerQueue: Line[] = [];
  private tickerT = 0;
  private bubbleT = 0;
  private bannerT = 0;
  private introT = 0;
  private bounceSfxT = 0;
  private lastFoulBack = 0;
  private humanSide: -1 | 0 | 1;

  constructor(container: HTMLElement, private world: World, private studio: PortraitStudio, readonly opts: GameOptions) {
    this.match = new Match({ ...opts.cfg, autoThrowDelay: settings.autoThrow });
    this.humanSide = opts.cfg.home.human ? 1 : opts.cfg.away.human ? 0 : -1;
    this.booth = new Booth(opts.cfg.seed);
    this.canvas = world.renderer.domElement;
    this.root = h('div', { class: 'game' });
    container.appendChild(this.root);
    this.director = new Director(world.camera);
    this.director.adopt(world.camera);
    this.buildHud();
    window.addEventListener('keydown', this.onKey);
    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('pointercancel', this.onPointerUp);
    audio.playMusic('game');
    audio.setAmbience(true);
    this.ready = true;
    if (import.meta.env.DEV) {
      (window as unknown as { __game: GameScreen }).__game = this;
      this.ff = Number(new URLSearchParams(location.hash.slice(1)).get('ff') ?? 0);
    }
    this.showIntro();
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.frame);
  }

  destroy() {
    this.destroyed = true;
    cancelAnimationFrame(this.raf);
    window.removeEventListener('keydown', this.onKey);
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('pointercancel', this.onPointerUp);
    audio.setAmbience(false);
    window.speechSynthesis?.cancel?.();
    this.root.remove();
  }

  // ─────────────────────────────────────────────────────────── main loop

  private slowmo = 0;
  /** dev only: simulate this many fixed steps per rendered frame (slow headless browsers) */
  ff = 0;

  private frame = (now: number) => {
    if (this.destroyed) return;
    const real = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const steps = this.ff > 0 ? this.ff : 1;
    // a crushed ball gets a moment of slow motion
    this.slowmo = Math.max(0, this.slowmo - real);
    const scale = this.slowmo > 0 ? 0.3 : 1;
    for (let i = 0; i < steps && !this.paused && this.ready; i++) {
      const dt = (this.ff > 0 ? 1 / 30 : real) * scale;
      this.time += dt;
      this.updateAimAssist(dt);
      if (this.introT <= 0) this.match.update(dt);
      this.handleEvents();
      this.world.sync(this.match, dt, this.overlay());
      this.director.forced = this.introT > 0 ? 'intro' : null;
      this.director.update(this.match, dt, this.world.ball.visible ? this.world.ball.position : null);
      this.updateHud(dt);
    }
    if (this.ready) {
      this.studio.update();
      this.world.render();
      this.world.adapt(real);
    }
    this.raf = requestAnimationFrame(this.frame);
  };

  private overlay() {
    const m = this.match;
    const batView = this.director?.shot === 'bat';
    const humanBat = m.humanBatting && batView && (m.phase === 'prePitch' || m.phase === 'windup' || m.phase === 'pitch');
    const humanPitch = m.humanPitching && batView && (m.phase === 'prePitch' || m.phase === 'windup');
    let radius = 0.5;
    if (humanBat) {
      const tune = { window: 1, radius: m.cfg.difficulty === 'rookie' ? 1.3 : m.cfg.difficulty === 'pro' ? 1.1 : 1 };
      radius = contactWindow(m.batter, { aimX: 0, aimZ: 0, tSwing: 0, kind: this.swingKind, special: this.armed }, tune).r;
    }
    const fielding = m.phase === 'live' && m.humanPitching && m.play && !m.play.deadKind;
    return {
      aim: humanBat ? this.aim : null,
      aimColor: this.swingKind === 'power' ? '#ff5e5b' : this.swingKind === 'bunt' ? '#6fc3ff' : '#ffe14d',
      aimRadius: radius,
      pitchAim: humanPitch ? this.pitchAim : null,
      showZone: batView && (settings.showZone || humanPitch) && m.phase !== 'live',
      bases: fielding ? { selected: m.play!.throwRequest } : null,
    };
  }

  // ─────────────────────────────────────────────────────────── input

  private updateAimAssist(dt: number) {
    const m = this.match;
    if (!m.humanBatting) return;
    if (m.phase === 'prePitch') {
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

  private screenToZone(cx: number, cy: number) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new Vector2(((cx - rect.left) / rect.width) * 2 - 1, -(((cy - rect.top) / rect.height) * 2 - 1));
    this.ray.setFromCamera(ndc, this.world.camera);
    const hit = this.ray.ray.intersectPlane(this.plate, new Vector3());
    if (!hit) return null;
    return { x: clamp(hit.x, -2.4, 2.4), z: clamp(hit.y, 0, 5.5) };
  }

  private project(v: Vector3) {
    const rect = this.canvas.getBoundingClientRect();
    const p = v.clone().project(this.world.camera);
    if (p.z > 1) return null;
    return { x: rect.left + (p.x * 0.5 + 0.5) * rect.width, y: rect.top + (-p.y * 0.5 + 0.5) * rect.height };
  }

  private onPointerDown = (e: PointerEvent) => {
    audio.unlock();
    const m = this.match;
    const touch = e.pointerType !== 'mouse';
    this.dragging = { id: e.pointerId, x: e.clientX, y: e.clientY, touch };
    if (this.introT > 0) { this.endIntro(); return; }
    const batView = this.director.shot === 'bat';
    if (m.phase === 'live' && m.humanPitching && m.play) {
      let best = -1, bestD = Infinity;
      for (let b = 1; b <= 4; b++) {
        const bp = m.field.bases[b % 4];
        const p = this.project(W(bp.x, bp.y, 0));
        if (!p) continue;
        const d = Math.hypot(p.x - e.clientX, p.y - e.clientY);
        if (d < bestD) { bestD = d; best = b; }
      }
      if (best > 0 && bestD < 90) this.throwTo(best);
      return;
    }
    if (m.humanPitching && m.phase === 'prePitch' && batView) {
      const z = this.screenToZone(e.clientX, e.clientY);
      if (z) {
        this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
        if (!touch) this.doPitch();
      }
      return;
    }
    if (m.humanBatting && batView && !touch) {
      const z = this.screenToZone(e.clientX, e.clientY);
      if (z) { this.aim = z; this.lastAimInput = this.time; }
      this.doSwing();
    }
  };

  private onPointerMove = (e: PointerEvent) => {
    if (!this.ready) return;
    const m = this.match;
    const d = this.dragging;
    const batView = this.director.shot === 'bat';
    if (e.pointerType === 'mouse' && !d) {
      if (m.humanBatting && batView && (m.phase !== 'pitch' || settings.aimAssist === 'off' || m.cfg.difficulty === 'allstar')) {
        const z = this.screenToZone(e.clientX, e.clientY);
        if (z) { this.aim = z; this.lastAimInput = this.time; }
      }
      if (m.humanPitching && m.phase === 'prePitch') {
        const z = this.screenToZone(e.clientX, e.clientY);
        if (z) this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
      }
      return;
    }
    if (!d || d.id !== e.pointerId) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    d.x = e.clientX; d.y = e.clientY;
    if (!d.touch) return;
    // touch drags nudge the aim like a trackpad
    const a = this.project(W(0, 0, 2)), b = this.project(W(1, 0, 2));
    const ppf = a && b ? Math.max(20, Math.abs(b.x - a.x)) : 60;
    if (m.humanBatting && batView) {
      this.aim.x = clamp(this.aim.x + (dx / ppf) * 0.9, -2.4, 2.4);
      this.aim.z = clamp(this.aim.z - (dy / ppf) * 0.9, 0, 5.5);
      this.lastAimInput = this.time;
    } else if (m.humanPitching && m.phase === 'prePitch') {
      const z = this.screenToZone(e.clientX, e.clientY);
      if (z) this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
    }
  };

  private onPointerUp = (e: PointerEvent) => {
    if (this.dragging?.id === e.pointerId) this.dragging = null;
  };

  private onKey = (e: KeyboardEvent) => {
    if (!this.ready) return;
    const m = this.match;
    const k = e.key;
    if (k === 'Escape') { this.togglePause(); return; }
    if (this.paused) return;
    audio.unlock();
    if (this.introT > 0) { this.endIntro(); return; }
    const step = 0.2;
    const batView = this.director.shot === 'bat';
    if (m.humanBatting && batView) {
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
      if (n >= 1 && n <= pitches.length) { this.pitchType = pitches[n - 1]; this.controlsKey = ''; }
      else if (k === 'ArrowLeft') this.pitchAim.x = clamp(this.pitchAim.x - step, -1.6, 1.6);
      else if (k === 'ArrowRight') this.pitchAim.x = clamp(this.pitchAim.x + step, -1.6, 1.6);
      else if (k === 'ArrowUp') this.pitchAim.z = clamp(this.pitchAim.z + step, 0.4, 4.2);
      else if (k === 'ArrowDown') this.pitchAim.z = clamp(this.pitchAim.z - step, 0.4, 4.2);
      else if (k === ' ' || k === 'Enter') { e.preventDefault(); this.doPitch(); }
      else if (k === 's' || k === 'S') this.toggleSpecial();
    }
    if (m.phase === 'live' && m.play) {
      if (m.humanPitching) {
        const map: Record<string, number> = { '1': 1, '2': 2, '3': 3, '4': 4, h: 4, H: 4, ArrowRight: 1, ArrowUp: 2, ArrowLeft: 3, ArrowDown: 4 };
        if (map[k]) { e.preventDefault(); this.throwTo(map[k]); }
      } else if (m.humanBatting) {
        if (k === 'r' || k === 'R' || k === 'ArrowRight' || k === ' ') { e.preventDefault(); m.runners('advance'); audio.play('uiTap'); }
        if (k === 'f' || k === 'F' || k === 'ArrowLeft') { m.runners('retreat'); audio.play('uiTap'); }
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
    const fx = this.world.fx;
    const good = (battingGood: boolean) => (this.humanSide < 0 ? true : (m.battingSide === this.humanSide) === battingGood);
    for (const e of m.events) {
      this.say(this.booth.react(e, m));
      switch (e.type) {
        case 'pitch':
          if (e.special) { this.popup(SPECIAL_INFO[e.special].label.toUpperCase() + '!', '#c39bff', 1); audio.play('special'); }
          else audio.play('throw', { intensity: 0.4 });
          this.radar(`${PITCHES[e.pitch as PitchType]?.label ?? e.pitch} · ${Math.round(e.mph)} mph`);
          break;
        case 'special':
          audio.play('special');
          fx.sparkle(this.world.handOf(e.kid) ?? W(0, 0, 3), '#c39bff');
          break;
        case 'contact': {
          const strong = e.quality > 0.55 && e.ev > 55;
          audio.play(strong ? 'batCrack' : 'batTink', { intensity: clamp((e.ev - 30) / 60, 0, 1) });
          if (strong) fx.sparkle(this.world.ball.position.clone(), '#fff6c4');
          if (strong && e.ev > 62 && e.la > 14 && e.la < 40) { this.slowmo = 0.45; this.director.shake(0.5); }
          break;
        }
        case 'whiff':
          audio.play('whiff');
          break;
        case 'call':
          if (e.call === 'ball') { this.popup('BALL', '#9fd3ff', 0.7); audio.play('mittPop', { intensity: 0.5 }); }
          else if (e.call === 'strike' || e.call === 'swinging') {
            if (m.strikes < 3) this.popup('STRIKE!', '#ffe14d', 0.9);
            audio.play('mittPop', { intensity: 0.8 });
            audio.play('strike');
          } else if (e.call === 'foul') {
            if (this.time - this.lastFoulBack > 0.3) this.popup('FOUL!', '#ffffff', 0.8);
            this.lastFoulBack = this.time;
          }
          break;
        case 'strikeout':
          this.popup(e.looking ? 'STRIKE THREE!' : 'STRUCK OUT!', '#ffe14d', 1.1);
          audio.play(good(false) ? 'cheer' : 'aww');
          break;
        case 'walk':
          this.popup(e.hbp ? 'OUCH!' : 'BALL FOUR', '#9fd3ff', 0.9);
          break;
        case 'catch':
          audio.play('catch', { intensity: e.hard ? 1 : 0.6 });
          if (e.fly && e.hard) this.popup('WHAT A GRAB!', '#7dff9a', 1);
          if (e.hard) fx.sparkle(this.world.gloveOf(e.fielder) ?? this.world.ball.position.clone());
          break;
        case 'bobble':
          audio.play('aww');
          this.popup('BOBBLE!', '#ffb36b', 0.8);
          break;
        case 'throw':
          audio.play('throw', { intensity: 0.7 });
          break;
        case 'out':
          this.popup('OUT!', '#ff6b6b', 1);
          audio.play('out');
          break;
        case 'run':
          this.popup('RUN SCORES!', '#7dff9a', 0.85);
          audio.play('safe');
          break;
        case 'hit': {
          const label = e.bases >= 3 ? 'TRIPLE!' : e.bases === 2 ? 'DOUBLE!' : 'BASE HIT!';
          this.popup(label, '#7dff9a', 1.05);
          audio.play(good(true) ? 'cheer' : 'aww');
          break;
        }
        case 'homeRun':
          this.popup('HOME RUN!', '#ffe14d', 1.5);
          audio.play('homeRun');
          audio.play(good(true) ? 'bigCheer' : 'aww');
          for (const d of [W(-47, 15, 3), W(47, 15, 3)]) fx.confetti(d, 90);
          break;
        case 'groundRule':
          this.popup(e.why === 'splash' ? 'SPLASH DOUBLE!' : 'GROUND-RULE DOUBLE', '#6fc3ff', 1.1);
          break;
        case 'splash':
          audio.play('splash');
          fx.splash(this.world.ball.position.clone().setY(-0.5));
          break;
        case 'error':
          this.popup('E! OOPS!', '#ffb36b', 0.9);
          break;
        case 'bounce':
          if (this.time - this.bounceSfxT > 0.12) { audio.play('bounce', { intensity: clamp(e.speed / 40, 0, 1) }); this.bounceSfxT = this.time; }
          if (e.speed > 5) {
            const p = W(e.x, e.y, 0);
            if (e.surface === 'grass') fx.grass(p, clamp(e.speed / 30, 0.3, 1.2));
            else fx.dust(p, clamp(e.speed / 30, 0.3, 1.2), e.surface === 'patio' ? '#d8d0c4' : '#c8a77a');
          }
          break;
        case 'fence':
          audio.play('fence', { intensity: 0.8 });
          break;
        case 'tree':
          audio.play('leaves');
          break;
        case 'dog':
          audio.play('dogBark');
          break;
        case 'quip':
          this.showBubble(kid(e.kid), e.text);
          break;
        case 'pitchingChange':
          this.popup('PITCHING CHANGE', '#ffffff', 0.8);
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
      // dust when a runner slides into a base
    }
    m.events.length = 0;
    const play = m.play;
    if (play) for (const r of play.runners) {
      if (r.anim === 'slide' && r.animT < 0.05) fx.dust(W(play.runnerPos(r).x, play.runnerPos(r).y, 0), 1.2);
    }
  }

  private say(lines: Line[]) {
    if (!lines.length) return;
    this.tickerQueue.push(...lines);
    if (this.tickerQueue.length > 4) this.tickerQueue.splice(0, this.tickerQueue.length - 4);
  }

  // ─────────────────────────────────────────────────────────── HUD

  private portrait(k: Kid, expr: Expression = 'happy', size = 120): HTMLImageElement {
    const a = this.world.actors.get(k.id);
    const team = this.teamOfKid(k);
    const img = h('img', { class: 'portrait', width: Math.round(size / 2), height: Math.round(size / 2), alt: k.nick });
    if (a && team) this.studio.into(img, a.model, team, expr, size);
    return img;
  }

  private teamOfKid(k: Kid): Team | null {
    const m = this.match;
    if (m.cfg.away.team.roster.includes(k.id)) return m.cfg.away.team;
    if (m.cfg.home.team.roster.includes(k.id)) return m.cfg.home.team;
    return null;
  }

  private teamOf(side: 0 | 1): Team { return this.match.side(side).team; }

  private buildHud() {
    this.sbEl = h('div', { class: 'sb' });
    this.cardsEl = h('div', { class: 'cards' });
    this.tickerEl = h('div', { class: 'ticker' });
    this.bubbleEl = h('div', { class: 'bubble hidden' });
    this.controlsEl = h('div', { class: 'controls' });
    this.popEl = h('div', { class: 'pops' });
    this.bannerEl = h('div', { class: 'banner hidden', onpointerdown: () => { audio.unlock(); if (this.introT > 0) this.endIntro(); } });
    this.hintEl = h('div', { class: 'hint' });
    const pause = h('button', { class: 'btn icon pause', 'aria-label': 'Pause', onclick: () => this.togglePause() }, '❚❚');
    const rotate = h('div', { class: 'rotate-hint' }, '📱↻ Turn your phone sideways for the best view');
    this.hud = h('div', { class: 'hud' }, this.sbEl, pause, this.cardsEl, this.tickerEl, this.bubbleEl, this.hintEl, this.controlsEl, this.popEl, this.bannerEl, rotate);
    this.root.appendChild(this.hud);
  }

  private updateHud(dt: number) {
    const m = this.match;
    this.renderScoreboard();
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
    this.tickerT -= dt;
    if (this.tickerT <= 0 && this.tickerQueue.length) {
      const line = this.tickerQueue.shift()!;
      clear(this.tickerEl);
      this.tickerEl.append(h('b', { class: line.who === 'Chet' ? 'chet' : 'dottie' }, line.who === 'Chet' ? 'CHET: ' : 'DOTTIE: '), line.text);
      this.tickerT = Math.max(2.2, line.text.length * 0.055);
      if (settings.voice) speak(line);
    }
    if (this.bubbleT > 0) { this.bubbleT -= dt; if (this.bubbleT <= 0) this.bubbleEl.classList.add('hidden'); }
    if (this.bannerT > 0) { this.bannerT -= dt; if (this.bannerT <= 0 && m.phase !== 'over') this.bannerEl.classList.add('hidden'); }
    if (this.introT > 0) { this.introT -= dt; if (this.introT <= 0) this.endIntro(); }
    this.renderControls();
  }

  private card(k: Kid, team: Team, label: string, sub: string) {
    const t = k.traits;
    const bar = (n: string, v: number) => h('div', { class: 'bar' }, h('span', null, n), h('i', null, h('b', { style: `width:${v * 10}%` })));
    return h('div', { class: 'card', style: `--team:${team.colors.primary};--team2:${team.colors.secondary}` },
      this.portrait(k, label === 'AT BAT' ? 'focus' : 'smug', 112),
      h('div', { class: 'card-txt' },
        h('div', { class: 'card-label' }, label),
        h('div', { class: 'card-name' }, k.nick),
        h('div', { class: 'card-sub' }, sub),
        h('div', { class: 'card-bars' }, bar('HIT', t.hitting), bar('SPD', t.speed), bar('FLD', t.fielding), bar('PIT', t.pitching))));
  }

  private renderScoreboard() {
    const m = this.match;
    const key = [m.score.join(), m.inning, m.half, m.outs, m.balls, m.strikes, m.bases.map((b) => (b ? 1 : 0)).join(''), Math.floor(m.hype[0] / 10), Math.floor(m.hype[1] / 10)].join('|');
    if (key === this.sbKey) return;
    this.sbKey = key;
    clear(this.sbEl);
    const row = (side: 0 | 1) => {
      const t = this.teamOf(side);
      return h('div', { class: `sb-row${m.battingSide === side ? ' bat' : ''}`, style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
        teamBadge(t, 22),
        h('span', { class: 'sb-abbr' }, t.abbr),
        h('span', { class: 'sb-hype', title: 'Hype' }, h('i', { style: `width:${m.hype[side]}%` })),
        h('span', { class: 'sb-runs' }, String(m.score[side])));
    };
    const dots = (n: number, of: number, cls: string) => h('span', { class: `dots ${cls}` }, ...Array.from({ length: of }, (_, i) => h('i', { class: i < n ? 'on' : '' })));
    const diamond = h('div', { class: 'diamond' }, ...[1, 2, 3].map((b) => h('i', { class: `b${b}${m.bases[b - 1] ? ' on' : ''}` })));
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
    const batView = this.director.shot === 'bat';
    if (this.introT > 0) mode = 'intro';
    else if (m.phase === 'live' && m.play && !m.play.deadKind) mode = m.humanPitching ? 'field' : m.humanBatting ? 'run' : 'none';
    else if (batView && m.humanBatting && ['prePitch', 'windup', 'pitch'].includes(m.phase)) mode = 'bat';
    else if (batView && m.humanPitching && m.phase === 'prePitch') mode = 'pitch';
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
        hint(touch ? 'Drag to aim · tap SWING when the ball arrives' : 'Aim with the mouse · click or Space to swing · P power · B bunt');
        const kindBtn = (k: SwingKind, label: string) => h('button', { class: `btn small${this.swingKind === k ? ' on' : ''}`, onpointerdown: (e: Event) => { e.preventDefault(); this.setKind(this.swingKind === k ? 'normal' : k); } }, label);
        this.controlsEl.append(h('div', { class: 'ctl-col' }, special(m.batter), kindBtn('power', 'POWER'), kindBtn('bunt', 'BUNT')));
        if (touch) this.controlsEl.append(h('button', { class: 'btn swing', onpointerdown: (e: Event) => { e.preventDefault(); this.doSwing(); } }, 'SWING!'));
        break;
      }
      case 'pitch': {
        hint(touch ? 'Pick a pitch · tap the zone to aim · THROW!' : 'Pick a pitch (1-3) · click in the zone to throw');
        const pitches = m.pitcher.pitches;
        if (!pitches.includes(this.pitchType)) this.pitchType = pitches[0];
        this.controlsEl.append(h('div', { class: 'ctl-col' }, special(m.pitcher),
          ...pitches.map((p, i) => h('button', { class: `btn small${this.pitchType === p ? ' on' : ''}`, onpointerdown: (e: Event) => { e.preventDefault(); this.pitchType = p; audio.play('uiTap'); this.controlsKey = ''; } }, `${i + 1} ${PITCHES[p].label}`))));
        if (touch) this.controlsEl.append(h('button', { class: 'btn swing', onpointerdown: (e: Event) => { e.preventDefault(); this.doPitch(); } }, 'THROW!'));
        break;
      }
      case 'field': {
        hint('Tap a base to throw there — or let your kid decide');
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
        hint(m.humanBatting || m.humanPitching ? '' : 'Watching the kids play');
    }
  }

  private radarEl: HTMLElement | null = null;
  /** the radar-gun readout after each pitch */
  private radar(text: string) {
    if (!this.radarEl) { this.radarEl = h('div', { class: 'radar' }); this.hud.appendChild(this.radarEl); }
    this.radarEl.textContent = text;
    this.radarEl.classList.remove('show');
    void this.radarEl.offsetWidth;
    this.radarEl.classList.add('show');
  }

  private popup(text: string, color: string, scale = 1) {
    const el = h('div', { class: 'pop', style: `--c:${color};--s:${scale}` }, text);
    clear(this.popEl);
    this.popEl.appendChild(el);
    setTimeout(() => el.remove(), 1500);
  }

  private showBubble(k: Kid, text: string) {
    clear(this.bubbleEl);
    this.bubbleEl.append(this.portrait(k, 'yell', 128), h('div', { class: 'bubble-txt' }, h('b', null, k.nick), h('span', null, `"${text}"`)));
    this.bubbleEl.classList.remove('hidden');
    this.bubbleT = 2.8;
  }

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
      h('div', { class: 'banner-vs' }, teamBadge(m.cfg.away.team, 64), h('span', null, 'at'), teamBadge(m.cfg.home.team, 64)),
      h('div', { class: 'banner-title' }, y.name),
      h('div', { class: 'banner-sub' }, `${y.owner}. ${y.blurb}`),
      h('ul', { class: 'banner-rules' }, ...y.rules.map((r) => h('li', null, r)), grownup?.label ? h('li', null, `Watching: ${grownup.label.replace(/\.$/, '')}.`) : null),
      h('div', { class: 'banner-tap' }, 'Tap to play ball!'));
    this.bannerEl.classList.remove('hidden');
    this.introT = 9;
  }

  private endIntro() {
    this.introT = 0;
    this.bannerEl.classList.add('hidden');
    this.director.cut();
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
        h('button', { class: 'btn ghost danger', onclick: () => { this.destroy(); this.opts.onExit(null); } }, 'Quit to menu')));
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
    const won = this.humanSide >= 0 && m.winner === this.humanSide;
    audio.playMusic('victory');
    if (won) audio.play('bigCheer');
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
    const title = m.winner === -1 ? 'It\'s a tie!' : this.humanSide < 0 ? `${this.teamOf(m.winner as 0 | 1).name} win!` : won ? 'You win!' : 'Tough loss!';
    const panel = h('div', { class: 'overlay final' },
      h('div', { class: 'panel wide' },
        h('h2', null, title),
        h('table', { class: 'linescore' }, header, line(0), line(1)),
        sk ? h('div', { class: 'star' }, this.portrait(sk, 'happy', 168),
          h('div', null, h('div', { class: 'card-label' }, 'PLAYER OF THE GAME'), h('div', { class: 'card-name' }, `${sk.first} "${sk.nick}" ${sk.last}`), h('div', { class: 'card-sub' }, starLine), h('div', { class: 'quote' }, `"${sk.quips[0]}"`))) : null,
        h('div', { class: 'cta' },
          h('button', { class: 'btn', onclick: () => { this.destroy(); this.opts.onExit(m, true); } }, 'Play again'),
          h('button', { class: 'btn ghost', onclick: () => { this.destroy(); this.opts.onExit(m); } }, 'Main menu'))));
    setTimeout(() => { if (!this.destroyed) this.root.appendChild(panel); }, 2500);
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
