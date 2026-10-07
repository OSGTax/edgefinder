import { Plane, Raycaster, Vector2, Vector3 } from 'three';
import { clamp } from '../engine/math';
import { audio, GameSound } from '../audio';
import { kid } from '../data/kids';
import { SPECIAL_INFO, type Kid, type PitchType, type Team } from '../data/types';
import { contactWindow, type SwingKind, type SwingRead } from '../sim/batting';
import { Match, type MatchConfig } from '../sim/match';
import { PITCHES } from '../sim/pitching';
import { W } from '../gfx/units';
import { Booth, ordinal, type Line } from '../ui/commentary';
import { clear, h } from '../ui/dom';
import { icon, lettering, lowerThird, panel, rotateHint, teamPatch, tradingCard } from '../ui/look';
import { saveSettings, settings } from '../ui/settings';
import { World } from './world';
import { Director } from './director';
import { PortraitStudio } from './portraits';
import { ComicPops } from './comic';
import type { Expression } from '../kid3d/face';

export interface GameOptions {
  cfg: MatchConfig;
  /** called when the player leaves; match is null if they quit early */
  onExit: (m: Match | null, again?: boolean) => void;
}

// ─────────────────────────────────────────────────────────── first-game coach

type CoachTip = 'swing' | 'pitch' | 'throw' | 'run';
const COACH_TIPS: CoachTip[] = ['swing', 'pitch', 'throw', 'run'];

/**
 * Show the first-game coach again from the start (How to Play calls this).
 * The tips appear in the next game, each the first time it applies.
 */
export function replayCoach() {
  settings.coachDone = false;
  settings.coachSeen = [];
  saveSettings();
}

/** the same tips for a mouse and keyboard */
const COACH_DESKTOP: Partial<Record<CoachTip, string[]>> = {
  swing: [
    'Click — or press Space — just as the ball gets to the plate.',
    'The circle aims itself on Rookie. Move the mouse (or the arrow keys) to steer it yourself.',
  ],
  pitch: [
    'Pick a pitch (1, 2, 3) and move the mouse to place the mitt.',
    'Click or press Space to start the meter, then again when the needle is in the green.',
  ],
  throw: ['Click a base (or press 1, 2, 3, H) to throw there. Wait, and your kid picks for you.'],
  run: ['Runners go on their own. R (or the right arrow) sends everybody, F sends them back.'],
};

const COACH_TEXT: Record<CoachTip, { title: string; body: string[] }> = {
  swing: {
    title: 'You\'re up!',
    body: [
      'Tap SWING — or anywhere on the right side — just as the ball gets to the plate.',
      'The circle aims itself on Rookie. Drag on the left side if you want to steer it.',
    ],
  },
  pitch: {
    title: 'You\'re pitching',
    body: [
      'Pick a pitch on the left. Drag anywhere to move the mitt.',
      'Tap THROW to start the meter, then tap again when the needle is in the green.',
    ],
  },
  throw: {
    title: 'Your kid has the ball',
    body: [
      'Tap a base to throw there. Wait, and your kid picks for you.',
    ],
  },
  run: {
    title: 'Ball in play — run!',
    body: [
      'Runners go on their own. GO sends everybody, BACK sends them back.',
    ],
  },
};

// ─────────────────────────────────────────────────────────── pitch meter

/** Where the needle wants to stop, as a fraction of the meter. */
const METER_SWEET = 0.78;

interface Meter { t: number; locked: number | null; acc: number; shownT: number }

export class GameScreen {
  readonly match: Match;
  private director: Director;
  private booth: Booth;
  /** sound: crowd, kids' barks, the booth's voices, surfaces, inning jingle (src/audio/cues.ts) */
  private sound = new GameSound();
  private root: HTMLElement;
  private canvas: HTMLCanvasElement;
  private raf = 0;
  private last = 0;
  private time = 0;
  private paused = false;
  private destroyed = false;
  ready = false;

  // batting input
  private aim = { x: 0, z: 2.2 };
  private swingKind: SwingKind = 'normal';
  private armed = false;
  private lastAimInput = -9;
  // pitching input
  private pitchType: PitchType = 'fastball';
  private pitchAim = { x: 0, z: 2 };
  private meter: Meter | null = null;
  private dragging: { id: number; x: number; y: number; touch: boolean; mode: 'aim' | 'pitch' | 'none' } | null = null;
  private ray = new Raycaster();
  private plate = new Plane(new Vector3(0, 0, 1), 0);

  // feel
  private hitStop = 0;
  private slowmo = 0;
  private coach: CoachTip | null = null;

  // dom
  private hud!: HTMLElement;
  private sbEl!: HTMLElement;
  private cardsEl!: HTMLElement;
  private tickerEl!: HTMLElement;
  private bubbleEl!: HTMLElement;
  private controlsLeft!: HTMLElement;
  private controlsRight!: HTMLElement;
  private bannerEl!: HTMLElement;
  private hintEl!: HTMLElement;
  private popEl!: HTMLElement;
  private readEl!: HTMLElement;
  private meterEl!: HTMLElement;
  private holderEl!: HTMLElement;
  private ballMarkEl!: HTMLElement;
  private coachEl!: HTMLElement;
  private radarEl!: HTMLElement;
  private controlsKey = '';
  private sbKey = '';
  private tickerQueue: Line[] = [];
  private tickerT = 0;
  private bubbleT = 0;
  private bannerT = 0;
  private readT = 0;
  private introT = 0;
  private outsThisPlay = 0;
  private aimedByDrag = false;
  private humanPitches = 0;
  private comic!: ComicPops;
  private cardT = 0;
  private humanSide: -1 | 0 | 1;
  private wake: WakeLockSentinel | null = null;

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
    document.addEventListener('visibilitychange', this.onVisibility);
    window.addEventListener('pagehide', this.onHide);
    this.canvas.addEventListener('contextmenu', this.noMenu);
    audio.playMusic('game');
    audio.setAmbience(true);
    this.keepAwake();
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
    document.removeEventListener('visibilitychange', this.onVisibility);
    window.removeEventListener('pagehide', this.onHide);
    this.canvas.removeEventListener('contextmenu', this.noMenu);
    this.releaseWake();
    audio.setAmbience(false);
    this.root.remove();
  }

  // ─────────────────────────────────────────────────────────── main loop

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
    const held = this.paused || !!this.coach;
    for (let i = 0; i < steps && !held && this.ready; i++) {
      const dt = (this.ff > 0 ? 1 / 30 : real) * scale;
      // hit-stop: the world holds still for a beat when bat meets ball
      if (this.hitStop > 0) { this.hitStop -= this.ff > 0 ? 1 / 30 : real; this.world.sync(this.match, 0, this.overlay()); continue; }
      this.time += dt;
      this.sound.tick(dt);
      this.updateAimAssist(dt);
      this.updateMeter(this.ff > 0 ? 1 / 30 : real);
      // the first time you bat, the pitcher waits for the coach to say its piece
      const waitCoach = !settings.coachDone && !settings.coachSeen.includes('swing') && this.match.humanBatting && this.match.phase === 'prePitch';
      if (this.introT <= 0 && !waitCoach) this.match.update(dt);
      this.handleEvents();
      this.world.sync(this.match, dt, this.overlay());
      this.director.forced = this.introT > 0 ? 'intro' : null;
      this.director.update(this.match, dt, this.world.ball.visible ? this.world.ball.position : null);
      this.updateHud(dt);
    }
    if (this.ready) {
      this.updateMarkers();
      const m = this.match;
      this.world.fx.ballHalo(this.world.ball.visible && (m.phase === 'live' || m.phase === 'pitch') ? this.world.ball.position : null, this.world.camera);
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

  // ─────────────────────────────────────────────────────────── feel

  /** A short buzz on phones that support it (Android browsers; iOS ignores it). */
  private buzz(pattern: number | number[]) {
    if (!settings.haptics || this.ff > 0) return;
    try { navigator.vibrate?.(pattern); } catch { /* not allowed */ }
  }

  private async keepAwake() {
    try {
      if (document.visibilityState !== 'visible' || this.wake) return;
      this.wake = await navigator.wakeLock?.request('screen') ?? null;
      this.wake?.addEventListener('release', () => { this.wake = null; });
    } catch { /* not supported or refused */ }
  }

  private releaseWake() {
    this.wake?.release().catch(() => {});
    this.wake = null;
  }

  private onVisibility = () => {
    if (document.visibilityState === 'hidden') this.onHide();
    else if (!this.destroyed) {
      this.keepAwake();
      this.last = performance.now();
    }
  };

  /** App switch, phone call, screen off: stop the game and wait for the player. */
  private onHide = () => {
    if (this.destroyed || this.paused || this.match.phase === 'over') return;
    this.dragging = null;
    this.togglePause(true);
  };

  private noMenu = (e: Event) => e.preventDefault();

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
    const strength = this.assist();
    if (strength <= 0) return;
    const arr = m.pitch.arrival;
    const k = Math.min(1, dt * 9 * strength);
    this.aim.x += (arr.x - this.aim.x) * k;
    this.aim.z += (arr.z - this.aim.z) * k;
  }

  private assist() {
    const mode = settings.aimAssist;
    const d = this.match.cfg.difficulty;
    return mode === 'off' ? 0 : mode === 'on' ? 0.85 : d === 'rookie' ? 0.9 : d === 'pro' ? 0.45 : 0;
  }

  private doSwing() {
    const m = this.match;
    if (!m.humanBatting || m.swingIn) return;
    if (m.phase !== 'windup' && m.phase !== 'pitch') return;
    audio.unlock();
    m.swing(this.aim.x, this.aim.z, this.swingKind, this.armed);
    this.buzz(8);
    if (this.armed) this.armed = false;
  }

  /** THROW: first press starts the meter, the second stops the needle and lets it go. */
  private pitchPress() {
    const m = this.match;
    if (!m.humanPitching || m.phase !== 'prePitch' || this.coach) return;
    audio.unlock();
    if (!this.meter) {
      this.meter = { t: 0, locked: null, acc: 0, shownT: 0 };
      audio.play('uiTap');
      this.controlsKey = '';
      return;
    }
    if (this.meter.locked === null) this.lockMeter();
  }

  private meterPeriod() {
    const d = this.match.cfg.difficulty;
    return d === 'rookie' ? 1.25 : d === 'pro' ? 1 : 0.82;
  }

  /** half-width of the green, from the pitcher's Control */
  private meterSweet() {
    const d = this.match.cfg.difficulty;
    const base = 0.045 + this.match.pitcher.traits.control * 0.008;
    return base * (d === 'rookie' ? 1.45 : d === 'pro' ? 1.15 : 1);
  }

  private meterU(t: number) { return Math.min(1, t / this.meterPeriod()); }

  private lockMeter() {
    const mt = this.meter!;
    const u = this.meterU(mt.t);
    const off = Math.max(0, Math.abs(u - METER_SWEET) - this.meterSweet());
    mt.locked = u;
    mt.acc = clamp(1 - off / 0.2, 0, 1);
    this.buzz(mt.acc > 0.95 ? [10, 30, 10] : 12);
    const label = mt.acc > 0.95 ? 'PAINTED IT!' : mt.acc > 0.6 ? 'Good release' : mt.acc > 0.25 ? 'A little wild' : 'Wild!';
    this.meterEl.dataset.read = label;
    this.match.selectPitch(this.pitchType, { ...this.pitchAim }, this.armed, mt.acc);
    this.armed = false;
    this.controlsKey = '';
  }

  private updateMeter(dt: number) {
    const mt = this.meter;
    if (!mt) { this.meterEl.classList.add('hidden'); return; }
    this.meterEl.classList.remove('hidden');
    if (mt.locked === null) {
      mt.t += dt;
      if (this.meterU(mt.t) >= 1) this.lockMeter(); // ran out: it goes wherever it goes
    } else {
      mt.shownT += dt;
      if (mt.shownT > 1.1) { this.meter = null; this.meterEl.classList.add('hidden'); return; }
    }
    const u = mt.locked ?? this.meterU(mt.t);
    const sw = this.meterSweet();
    this.meterEl.style.setProperty('--u', String(u));
    this.meterEl.style.setProperty('--s0', String(METER_SWEET - sw));
    this.meterEl.style.setProperty('--s1', String(METER_SWEET + sw));
    this.meterEl.classList.toggle('locked', mt.locked !== null);
    this.meterEl.classList.toggle('good', mt.locked !== null && mt.acc > 0.6);
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

  /** screen pixels per foot at the plate (for trackpad-style drags) */
  private plateScale() {
    const a = this.project(W(0, 0, 2)), b = this.project(W(1, 0, 2));
    return a && b ? Math.max(20, Math.abs(b.x - a.x)) : 60;
  }

  private onPointerDown = (e: PointerEvent) => {
    audio.unlock();
    if (this.paused) return;
    if (this.coach) { this.dismissCoach(); return; } // any tap gets on with it
    const m = this.match;
    const touch = e.pointerType !== 'mouse';
    this.dragging = { id: e.pointerId, x: e.clientX, y: e.clientY, touch, mode: 'none' };
    if (this.introT > 0) { this.endIntro(); return; }
    // tap through the pauses between pitches and innings
    if (m.phase === 'result' || m.phase === 'halfOver') { m.skip(); return; }
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
      if (best > 0 && bestD < 70) this.throwTo(best);
      return;
    }
    if (m.humanPitching && m.phase === 'prePitch' && batView) {
      this.dragging.mode = 'pitch';
      const z = this.screenToZone(e.clientX, e.clientY);
      const zone = m.zone;
      // a tap on (or right by) the zone puts the mitt there; elsewhere it's a nudge pad
      if (z && (!touch || (Math.abs(z.x) < zone.half + 0.8 && z.z > zone.bottom - 0.8 && z.z < zone.top + 0.8))) {
        this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
      }
      if (!touch) this.pitchPress();
      return;
    }
    if (m.humanBatting && batView) {
      if (!touch) {
        const z = this.screenToZone(e.clientX, e.clientY);
        if (z) { this.aim = z; this.lastAimInput = this.time; }
        this.doSwing();
        return;
      }
      // phones: the right side of the screen is one big swing button, the left side steers
      const rect = this.canvas.getBoundingClientRect();
      if (e.clientX > rect.left + rect.width * 0.55) this.doSwing();
      else this.dragging.mode = 'aim';
    }
  };

  private onPointerMove = (e: PointerEvent) => {
    if (!this.ready || this.paused) return;
    const m = this.match;
    const d = this.dragging;
    const batView = this.director.shot === 'bat';
    if (e.pointerType === 'mouse' && !d) {
      if (m.humanBatting && batView && (m.phase !== 'pitch' || this.assist() === 0)) {
        const z = this.screenToZone(e.clientX, e.clientY);
        if (z) { this.aim = z; this.lastAimInput = this.time; }
      }
      if (m.humanPitching && m.phase === 'prePitch' && !this.meter) {
        const z = this.screenToZone(e.clientX, e.clientY);
        if (z) this.pitchAim = { x: clamp(z.x, -1.6, 1.6), z: clamp(z.z, 0.4, 4.2) };
      }
      return;
    }
    if (!d || d.id !== e.pointerId) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    d.x = e.clientX; d.y = e.clientY;
    if (!d.touch) return;
    // touch drags nudge the aim like a trackpad, so the thumb never covers the target
    const ppf = this.plateScale();
    if (d.mode === 'aim' && m.humanBatting && batView) {
      this.aim.x = clamp(this.aim.x + (dx / ppf) * 1.1, -2.4, 2.4);
      this.aim.z = clamp(this.aim.z - (dy / ppf) * 1.1, 0, 5.5);
      this.lastAimInput = this.time;
      if (!this.aimedByDrag && Math.hypot(dx, dy) > 2) { this.aimedByDrag = true; this.controlsKey = ''; }
    } else if (d.mode === 'pitch' && m.humanPitching && m.phase === 'prePitch' && !this.meter) {
      this.pitchAim.x = clamp(this.pitchAim.x + (dx / ppf) * 1.1, -1.6, 1.6);
      this.pitchAim.z = clamp(this.pitchAim.z - (dy / ppf) * 1.1, 0.4, 4.2);
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
    if (this.coach) { if (k === ' ' || k === 'Enter') { e.preventDefault(); this.dismissCoach(); } return; }
    if (this.introT > 0) { this.endIntro(); return; }
    if ((m.phase === 'result' || m.phase === 'halfOver') && (k === ' ' || k === 'Enter')) { e.preventDefault(); m.skip(); return; }
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
      if (n >= 1 && n <= pitches.length && !this.meter) this.pickPitch(pitches[n - 1]);
      else if (this.meter) { if (k === ' ' || k === 'Enter') { e.preventDefault(); this.pitchPress(); } }
      else if (k === 'ArrowLeft') this.pitchAim.x = clamp(this.pitchAim.x - step, -1.6, 1.6);
      else if (k === 'ArrowRight') this.pitchAim.x = clamp(this.pitchAim.x + step, -1.6, 1.6);
      else if (k === 'ArrowUp') this.pitchAim.z = clamp(this.pitchAim.z + step, 0.4, 4.2);
      else if (k === 'ArrowDown') this.pitchAim.z = clamp(this.pitchAim.z - step, 0.4, 4.2);
      else if (k === ' ' || k === 'Enter') { e.preventDefault(); this.pitchPress(); }
      else if (k === 's' || k === 'S') this.toggleSpecial();
    }
    if (m.phase === 'live' && m.play) {
      if (m.humanPitching) {
        const map: Record<string, number> = { '1': 1, '2': 2, '3': 3, '4': 4, h: 4, H: 4, ArrowRight: 1, ArrowUp: 2, ArrowLeft: 3, ArrowDown: 4 };
        if (map[k]) { e.preventDefault(); this.throwTo(map[k]); }
      } else if (m.humanBatting) {
        if (k === 'r' || k === 'R' || k === 'ArrowRight' || k === ' ') { e.preventDefault(); this.sendRunners('advance'); }
        if (k === 'f' || k === 'F' || k === 'ArrowLeft') this.sendRunners('retreat');
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

  private pickPitch(p: PitchType) {
    if (this.meter) return;
    this.pitchType = p;
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
    this.buzz(10);
    this.controlsKey = '';
  }

  private sendRunners(cmd: 'advance' | 'retreat') {
    this.match.runners(cmd);
    audio.play('uiTap');
    this.buzz(10);
    this.runCmd = cmd;
    this.controlsKey = '';
  }
  private runCmd: 'advance' | 'retreat' | null = null;

  // ─────────────────────────────────────────────────────────── events

  private handleEvents() {
    const m = this.match;
    const fx = this.world.fx;
    this.sound.events(m.events, m, this.humanSide); // sound: see GameSound
    const mine = this.humanSide >= 0 && m.battingSide === this.humanSide;
    for (const e of m.events) {
      this.say(this.booth.react(e, m));
      switch (e.type) {
        case 'batterUp':
          this.runCmd = null;
          this.outsThisPlay = 0;
          this.showCard(m.batter, m.battingSide, 'Up to bat');
          break;
        case 'pitch':
          this.outsThisPlay = 0;
          if (this.humanSide >= 0 && ++this.humanPitches === 4) this.controlsKey = '';
          if (e.special) { this.comic.show('special', this.time, SPECIAL_INFO[e.special].label.toUpperCase() + '!'); audio.play('special'); }
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
          // weight: a frame or three of stillness, a kick of the camera, slow-mo on a crush
          this.hitStop = 0.035 + e.quality * 0.075;
          this.director.kick(0.3 + e.quality * 0.9);
          if (strong && e.ev > 62 && e.la > 14 && e.la < 40) { this.slowmo = 0.45; this.director.shake(0.5); }
          if (e.quality > 0.85 && e.ev > 58) this.comic.show('crush', this.time);
          if (mine) this.buzz(strong ? [0, 35, 25, 20] : 18);
          if (mine && e.read) this.showRead(e.read, 'contact', e.quality, e.la);
          break;
        }
        case 'whiff':
          audio.play('whiff');
          if (mine && e.read) this.showRead(e.read, 'miss');
          break;
        case 'foulTip':
          if (mine) this.showRead(e.read, 'tip');
          break;
        case 'call':
          // routine calls get no pop-up: the count on the score bug flips instead
          if (e.call === 'ball') audio.play('mittPop', { intensity: 0.5 });
          else if (e.call === 'strike' || e.call === 'swinging') {
            audio.play('mittPop', { intensity: 0.8 });
            audio.play('strike');
            if (e.call === 'strike' && mine) this.showTake();
          }
          break;
        case 'strikeout':
          this.comic.show(e.looking ? 'kLooking' : 'kSwinging', this.time);
          if (!mine && this.humanSide >= 0) this.buzz([0, 20, 40, 20]);
          break;
        case 'walk':
          if (e.hbp) this.comic.show('hbp', this.time);
          break;
        case 'catch':
          audio.play('catch', { intensity: e.hard ? 1 : 0.6 });
          if (e.fly && e.hard) this.comic.show('snag', this.time);
          if (e.hard) { fx.sparkle(this.world.gloveOf(e.fielder) ?? this.world.ball.position.clone()); this.director.kick(0.35); }
          if (this.humanSide >= 0 && !mine) this.buzz(e.hard ? 25 : 12);
          break;
        case 'bobble':
          this.comic.show('oops', this.time);
          break;
        case 'throw':
          audio.play('throw', { intensity: 0.7 });
          if (this.humanSide >= 0 && !mine) this.buzz(8);
          break;
        case 'out':
          audio.play('out');
          if (++this.outsThisPlay === 2) this.comic.show('doublePlay', this.time);
          this.director.kick(0.25);
          if (this.humanSide >= 0) this.buzz(mine ? 12 : 30);
          break;
        case 'run':
          if (m.phase === 'live') this.comic.show('scores', this.time);
          audio.play('safe');
          if (mine) this.buzz([0, 20, 30, 20]);
          break;
        case 'hit': {
          if (e.bases >= 2) this.comic.show(e.bases >= 3 ? 'triple' : 'double', this.time);
          break;
        }
        case 'homeRun':
          this.comic.show('homer', this.time);
          audio.play('homeRun');
          for (const d of [W(-47, 15, 3), W(47, 15, 3)]) fx.confetti(d, 90);
          if (mine) this.buzz([0, 40, 50, 40, 50, 80]);
          break;
        case 'groundRule':
          if (e.why !== 'splash') this.comic.show('double', this.time, 'GROUND RULE!');
          break;
        case 'splash':
          audio.play('splash');
          this.comic.show('splash', this.time);
          fx.splash(this.world.ball.position.clone().setY(-0.5));
          break;
        case 'error':
          this.comic.show('oops', this.time);
          break;
        case 'bounce':
          if (e.speed > 5) {
            const p = W(e.x, e.y, 0);
            if (e.surface === 'grass') fx.grass(p, clamp(e.speed / 30, 0.3, 1.2));
            else fx.dust(p, clamp(e.speed / 30, 0.3, 1.2), e.surface === 'patio' ? '#d8d0c4' : '#c8a77a');
          }
          break;
        case 'fence':
          if (!e.cleared) this.comic.show('fence', this.time);
          break;
        case 'tree':
          audio.play('leaves');
          break;
        case 'dog':
          audio.play('dogBark');
          if (Math.random() < 0.5) this.comic.show('dog', this.time);
          break;
        case 'quip':
          this.showBubble(kid(e.kid), e.text);
          break;
        case 'pitchingChange':
          this.showCard(kid(e.to), m.fieldingSide, 'New pitcher');
          break;
        case 'halfOver':
          audio.play('whistle');
          this.showBanner(`${e.half === 0 ? 'Middle' : 'End'} of the ${ordinal(e.inning)}`, `${m.cfg.away.team.abbr} ${m.score[0]} — ${m.cfg.home.team.abbr} ${m.score[1]}`, 2.4, 'Tap to skip');
          break;
        case 'gameOver':
          this.onGameOver();
          break;
        default:
          break;
      }
    }
    m.events.length = 0;
    // dust when a runner slides into a base
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

  // ─────────────────────────────────────────────────────────── swing feedback

  /**
   * After every swing: a little timing ruler (where the bat arrived against the
   * kid's window) and a word or two about how it went.
   */
  private showRead(r: SwingRead, how: 'contact' | 'miss' | 'tip', quality = 0, la = 0) {
    const t = r.timing;
    const bunt = this.swingKind === 'bunt';
    let word: string, sub: string, tone: 'good' | 'ok' | 'bad';
    const when = Math.abs(t) <= 0.35 ? 'On time' : t < 0 ? (t < -1 ? 'Way early' : 'Early') : (t > 1 ? 'Way late' : 'Late');
    if (how === 'contact') {
      tone = quality > 0.75 ? 'good' : quality > 0.45 ? 'ok' : 'bad';
      word = bunt ? 'Bunted' : quality > 0.85 ? 'CRUSHED' : quality > 0.65 ? 'Squared up' : quality > 0.45 ? 'Solid' : la < 0 ? 'Topped it' : la > 45 ? 'Got under it' : t > 0.5 ? 'Jammed' : 'Off the end';
      sub = bunt ? 'Laid down' : when;
    } else if (how === 'tip') {
      tone = 'ok';
      word = 'Just ticked it';
      sub = Math.abs(t) > 1 ? when : r.under > 0 ? `${when} · a hair under` : `${when} · a hair over`;
    } else {
      tone = 'bad';
      if (Math.abs(t) > 1.05 && t < 9) { word = when; sub = r.aim > 1.2 ? (r.under > 0 ? 'and under it' : 'and over it') : 'right height'; }
      else if (t >= 9) { word = 'Too late to bunt'; sub = 'Square around sooner'; }
      else { word = r.under > 0 ? 'Swung under' : 'Swung over'; sub = when; }
    }
    this.renderRead(word, sub, tone, t);
  }

  /** A called strike the player watched go by. */
  private showTake() {
    if (this.match.humanBatting && !this.match.swingIn) this.renderRead('Watched it', 'Called strike', 'bad', null);
  }

  private renderRead(word: string, sub: string, tone: string, timing: number | null) {
    clear(this.readEl);
    this.readEl.className = `swingread ${tone}`;
    const ruler = timing === null || Math.abs(timing) > 5 ? null : h('div', { class: 'sr-ruler' },
      h('span', { class: 'sr-lbl' }, 'early'), h('i', { class: 'sr-win' }),
      h('b', { class: 'sr-tick', style: `left:${50 + clamp(timing, -1.6, 1.6) * 28}%` }), h('span', { class: 'sr-lbl r' }, 'late'));
    this.readEl.append(h('div', { class: 'sr-word' }, word), h('div', { class: 'sr-sub' }, sub));
    if (ruler) this.readEl.append(ruler);
    this.readT = 1.9;
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
    this.controlsLeft = h('div', { class: 'controls left' });
    this.controlsRight = h('div', { class: 'controls right' });
    this.popEl = h('div', { class: 'pops' });
    this.comic = new ComicPops(this.popEl);
    this.readEl = h('div', { class: 'swingread hidden' });
    this.meterEl = h('div', { class: 'pmeter hidden' }, h('div', { class: 'pm-bar' }, h('i', { class: 'pm-sweet' }), h('b', { class: 'pm-needle' })));
    this.holderEl = h('div', { class: 'holder hidden' });
    this.ballMarkEl = h('div', { class: 'ballmark hidden' }, h('i'), h('span'));
    this.coachEl = h('div', { class: 'coach hidden', onpointerdown: (e: Event) => { e.preventDefault(); e.stopPropagation(); this.dismissCoach(); } });
    this.radarEl = h('div', { class: 'radar' });
    this.bannerEl = h('div', { class: 'banner hidden', onpointerdown: () => { audio.unlock(); if (this.introT > 0) this.endIntro(); else this.match.skip(); } });
    this.hintEl = h('div', { class: 'hint' });
    const pause = h('button', { class: 'btn icon pause', 'aria-label': 'Pause', onclick: () => this.togglePause() }, icon('pause', { title: 'Pause' }));
    const rotate = h('div', { class: 'rotate-hint' }, rotateHint());
    this.hud = h('div', { class: 'hud' }, this.holderEl, this.ballMarkEl, this.sbEl, pause, this.cardsEl, this.tickerEl, this.radarEl, this.bubbleEl, this.hintEl,
      this.readEl, this.controlsLeft, this.controlsRight, this.meterEl, this.popEl, this.bannerEl, this.coachEl, rotate);
    this.root.appendChild(this.hud);
  }

  private updateHud(dt: number) {
    const m = this.match;
    this.renderScoreboard();
    if (this.cardT > 0) { this.cardT -= dt; if (this.cardT <= 0) this.cardsEl.classList.remove('show'); }
    // captions: one line at a time, then the screen is clear again
    this.tickerT -= dt;
    if (this.tickerT <= 0 && this.tickerQueue.length) {
      const line = this.tickerQueue.shift()!;
      clear(this.tickerEl);
      const chet = line.who === 'Chet';
      this.tickerEl.append(lowerThird({ who: chet ? 'Chet' : 'Dottie', role: chet ? 'play-by-play' : 'color', text: line.text, tone: chet ? 'chet' : 'dottie' }));
      this.tickerEl.classList.add('show');
      this.tickerT = Math.max(2.2, line.text.length * 0.055);
      this.sound.caption(line.who, line.text); // sound: the booth talks under its caption
    } else if (this.tickerT <= -0.4) this.tickerEl.classList.remove('show');
    if (this.bubbleT > 0) { this.bubbleT -= dt; if (this.bubbleT <= 0) this.bubbleEl.classList.add('hidden'); }
    if (this.bannerT > 0) { this.bannerT -= dt; if (this.bannerT <= 0 && m.phase !== 'over') this.bannerEl.classList.add('hidden'); }
    if (m.phase === 'prePitch' && this.bannerT > 0 && this.introT <= 0) { this.bannerT = 0; this.bannerEl.classList.add('hidden'); }
    if (this.introT > 0) { this.introT -= dt; if (this.introT <= 0) this.endIntro(); }
    if (this.readT > 0) { this.readT -= dt; this.readEl.classList.toggle('hidden', this.readT <= 0); }
    this.renderControls();
  }

  /** The name card that slides in when a kid steps up (or comes in to pitch), then gets out of the way. */
  private showCard(k: Kid, side: 0 | 1, label: string) {
    if (this.introT > 0) return;
    const team = this.teamOf(side);
    const t = k.traits;
    const pitching = label === 'New pitcher';
    const stats: [string, number][] = pitching ? [['Pitching', t.pitching], ['Control', t.control]] : [['Contact', t.contact], ['Power', t.power]];
    const bl = this.match.box[k.id]?.bat;
    const sub = pitching ? k.pitches.map((x) => PITCHES[x].label).join(', ') : bl && bl.ab ? `${bl.h} for ${bl.ab} today` : k.persona;
    clear(this.cardsEl);
    const tc = tradingCard({
      photo: this.portrait(k, pitching ? 'smug' : 'focus', 160), name: k.nick, persona: sub, team,
      stats: stats.map(([n, v]) => [n, v] as [string, number]), class: 'hud-card', seed: k.id,
    });
    this.cardsEl.append(h('div', { class: 'card-tag' }, label), tc);
    this.cardsEl.classList.remove('show');
    void this.cardsEl.offsetWidth;
    this.cardsEl.classList.add('show');
    this.cardT = 3;
  }

  /** The score bug: both scores, the inning, the bases and the count, in one small strip. */
  private renderScoreboard() {
    const m = this.match;
    const key = [m.score.join(), m.inning, m.half, m.outs, m.balls, m.strikes, m.bases.map((b) => (b ? 1 : 0)).join('')].join('|');
    if (key === this.sbKey) return;
    const flip = this.sbKey !== '' && this.sbKey.split('|')[0] !== m.score.join();
    this.sbKey = key;
    clear(this.sbEl);
    const team = (side: 0 | 1) => {
      const t = this.teamOf(side);
      return h('div', { class: `sb-team${m.battingSide === side ? ' bat' : ''}`, style: `--team:${t.colors.primary};--team2:${t.colors.secondary}` },
        teamPatch(t, 22),
        h('span', { class: 'sb-runs' }, lettering(String(m.score[side]), { style: 'comic', size: 17, color: 'var(--poster)', seed: `run${side}` })));
    };
    const dots = (n: number, of: number, cls: string) => h('span', { class: `dots ${cls}` }, ...Array.from({ length: of }, (_, i) => h('i', { class: i < n ? 'on' : '' })));
    const diamond = h('div', { class: 'diamond' }, ...[1, 2, 3].map((b) => h('i', { class: `b${b}${m.bases[b - 1] ? ' on' : ''}` })));
    this.sbEl.append(
      team(0), team(1),
      h('div', { class: 'sb-inning', 'aria-label': `${m.half === 0 ? 'Top' : 'Bottom'} ${m.inning}` }, icon(m.half === 0 ? 'up' : 'down'), lettering(String(m.inning), { style: 'comic', size: 14, color: 'var(--sunshine)', seed: 'inn' })),
      diamond,
      h('div', { class: 'sb-count' },
        h('span', null, 'B'), dots(m.balls, 3, 'balls'),
        h('span', null, 'S'), dots(m.strikes, 2, 'strikes'),
        h('span', null, 'O'), dots(m.outs, 2, 'outs')));
    // the caption sits just right of the bug, however wide it is
    this.hud.style.setProperty('--sbw', `${this.sbEl.offsetWidth}px`);
    if (flip) { this.sbEl.classList.remove('flip'); void this.sbEl.offsetWidth; this.sbEl.classList.add('flip'); }
  }

  private controlMode() {
    const m = this.match;
    const batView = this.director.shot === 'bat';
    if (this.introT > 0) return 'intro';
    if (m.phase === 'live' && m.play && !m.play.deadKind) return m.humanPitching ? 'field' : m.humanBatting ? 'run' : 'none';
    if (batView && m.humanBatting && ['prePitch', 'windup', 'pitch'].includes(m.phase)) return 'bat';
    if (batView && m.humanPitching && m.phase === 'prePitch') return 'pitch';
    return 'none';
  }

  private renderControls() {
    const m = this.match;
    const mode = this.controlMode();
    this.maybeCoach(mode);
    const canSp = mode === 'bat' ? m.canSpecial(m.battingSide, m.batter) : mode === 'pitch' ? m.canSpecial(m.fieldingSide, m.pitcher) : false;
    const holding = mode === 'field' && m.play!.holder >= 0;
    const req = mode === 'field' ? m.play!.throwRequest : null;
    const key = `${mode}|${this.swingKind}|${this.pitchType}|${this.armed}|${canSp}|${m.pitcher.id}|${m.batter.id}|${!!this.meter}|${holding}|${req}|${this.runCmd}`;
    if (key === this.controlsKey) return;
    this.controlsKey = key;
    clear(this.controlsLeft);
    clear(this.controlsRight);
    this.hud.dataset.mode = mode;
    const touch = matchMedia('(pointer: coarse)').matches;
    // the desktop key reminder fades once you've seen a few pitches
    const hint = (s: string) => { this.hintEl.textContent = this.humanPitches < 4 ? s : ''; };
    // every control fires on pointerdown: no 300 ms wait, no missed swings
    const tap = (fn: () => void) => (e: Event) => { e.preventDefault(); e.stopPropagation(); if (this.coach) this.dismissCoach(); else fn(); };
    const special = (k: Kid) => canSp ? h('button', { class: `btn ctl special${this.armed ? ' armed' : ''}`, onpointerdown: tap(() => this.toggleSpecial()) }, icon('bolt'), h('span', null, SPECIAL_INFO[k.special].label)) : null;
    switch (mode) {
      case 'bat': {
        hint(touch ? '' : 'Aim with the mouse · click or Space to swing · P power · B bunt');
        const kindBtn = (k: SwingKind, label: string) => h('button', { class: `btn ctl kind ${k}${this.swingKind === k ? ' on' : ''}`, onpointerdown: tap(() => this.setKind(this.swingKind === k ? 'normal' : k)) }, label);
        this.controlsRight.append(
          h('div', { class: 'ctl-col' }, special(m.batter), kindBtn('power', 'Power'), kindBtn('bunt', 'Bunt')),
          h('button', { class: `btn big-round swing ${this.swingKind}`, 'aria-label': 'Swing', onpointerdown: tap(() => this.doSwing()) },
            icon('bat'), lettering(this.swingKind === 'bunt' ? 'BUNT' : 'SWING', { style: 'comic', size: 17, color: 'var(--poster)', seed: 'swing' })));
        // the aim pad marking only shows where aiming matters and until you've used it once
        if (touch && this.assist() < 0.9 && !this.aimedByDrag) this.controlsLeft.append(h('div', { class: 'aimpad' }, h('span', null, 'Drag here to aim')));
        break;
      }
      case 'pitch': {
        hint(touch ? '' : 'Pick a pitch (1-3) · move the mitt · Space or click twice to throw');
        const pitches = m.pitcher.pitches;
        if (!pitches.includes(this.pitchType)) this.pitchType = pitches[0];
        this.controlsLeft.append(h('div', { class: 'ctl-col pitches' }, special(m.pitcher),
          ...pitches.map((p, i) => h('button', { class: `btn ctl pitchpick${this.pitchType === p ? ' on' : ''}`, disabled: !!this.meter, onpointerdown: tap(() => this.pickPitch(p)) },
            h('small', null, touch ? PITCHES[p].short : `${i + 1}`), PITCHES[p].label))));
        this.controlsRight.append(h('button', { class: `btn big-round throw${this.meter ? ' metering' : ''}`, 'aria-label': 'Throw', onpointerdown: tap(() => this.pitchPress()) },
          icon('ball'), lettering(this.meter ? 'NOW!' : 'THROW', { style: 'comic', size: 17, color: 'var(--poster)', seed: 'throw' })));
        break;
      }
      case 'field': {
        hint(holding ? '' : 'Your kids are chasing it');
        const base = (b: number, label: string) => h('button', { class: `btn ctl base b${b}${req === b ? ' on' : ''}`, onpointerdown: tap(() => this.throwTo(b)) }, label);
        this.controlsRight.append(h('div', { class: `ctl-diamond${holding ? ' ready' : ''}` }, h('span', { class: 'cd-label' }, holding ? 'Throw to' : 'Throw to…'),
          base(2, '2nd'), base(3, '3rd'), base(1, '1st'), base(4, 'Home')));
        break;
      }
      case 'run':
        hint('');
        this.controlsRight.append(h('div', { class: 'ctl-run' },
          h('button', { class: `btn ctl run go${this.runCmd === 'advance' ? ' on' : ''}`, onpointerdown: tap(() => this.sendRunners('advance')) }, h('span', null, 'Go!'), icon('forward')),
          h('button', { class: `btn ctl run back${this.runCmd === 'retreat' ? ' on' : ''}`, onpointerdown: tap(() => this.sendRunners('retreat')) }, icon('back'), h('span', null, 'Back'))));
        break;
      case 'intro':
        hint('');
        break;
      default:
        hint(m.humanBatting || m.humanPitching || m.phase === 'over' ? '' : 'CPU vs CPU');
    }
  }

  /** the "has the ball" arrow over your fielder, and an edge marker for a ball off the screen */
  private updateMarkers() {
    const m = this.match;
    const play = m.play;
    const rect = this.canvas.getBoundingClientRect();
    // who has it
    let showHolder = false;
    if (play && m.phase === 'live' && m.humanPitching && !play.deadKind && play.holder >= 0) {
      const fl = play.fielders[play.holder];
      const p = this.project(W(fl.p.x, fl.p.y, 6.2));
      if (p) {
        showHolder = true;
        if (this.holderEl.dataset.kid !== fl.kid.id) { this.holderEl.dataset.kid = fl.kid.id; this.holderEl.textContent = fl.kid.nick; }
        this.holderEl.style.transform = `translate(${Math.round(clamp(p.x, 40, rect.width - 40))}px, ${Math.round(clamp(p.y, 30, rect.height - 10))}px) translate(-50%, -100%)`;
      }
    }
    this.holderEl.classList.toggle('hidden', !showHolder);
    // a high fly or a ball out of frame: point at it from the edge
    let showBall = false;
    if (m.phase === 'live' && this.world.ball.visible && this.director.shot === 'live') {
      const b = this.world.ball.position;
      const p = b.clone().project(this.world.camera);
      const behind = p.z > 1;
      const sx = (p.x * 0.5 + 0.5) * rect.width, sy = (-p.y * 0.5 + 0.5) * rect.height;
      const off = behind || sx < 0 || sx > rect.width || sy < 0 || sy > rect.height;
      if (off) {
        showBall = true;
        const x = clamp(behind ? rect.width - sx : sx, 28, rect.width - 28);
        const y = clamp(behind ? 0 : sy, 28, rect.height - 28);
        const ang = Math.atan2((behind ? -1 : sy) - y, (behind ? rect.width - sx : sx) - x);
        this.ballMarkEl.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px) translate(-50%, -50%)`;
        (this.ballMarkEl.firstChild as HTMLElement).style.transform = `rotate(${ang}rad)`;
        (this.ballMarkEl.lastChild as HTMLElement).textContent = `${Math.round(Math.max(0, b.y))} ft`;
      }
    }
    this.ballMarkEl.classList.toggle('hidden', !showBall);
  }

  /** the radar-gun readout after each pitch */
  private radar(text: string) {
    this.radarEl.textContent = text;
    this.radarEl.classList.remove('show');
    void this.radarEl.offsetWidth;
    this.radarEl.classList.add('show');
  }


  private showBubble(k: Kid, text: string) {
    clear(this.bubbleEl);
    this.bubbleEl.append(this.portrait(k, 'yell', 128), h('div', { class: 'bubble-txt' }, h('b', null, k.nick), h('span', null, `"${text}"`)));
    this.bubbleEl.classList.remove('hidden');
    this.bubbleT = 2.8;
  }

  private showBanner(title: string, sub: string, hold = 2.2, tapNote?: string) {
    clear(this.bannerEl);
    this.bannerEl.append(h('div', { class: 'banner-title' }, lettering(title.toUpperCase(), { style: 'comic', size: 22, color: 'var(--sunshine)', seed: title })), h('div', { class: 'banner-sub' }, sub));
    if (tapNote) this.bannerEl.append(h('div', { class: 'banner-tap' }, tapNote));
    this.bannerEl.classList.remove('hidden');
    this.bannerT = hold;
  }

  private showIntro() {
    const m = this.match;
    const y = m.field.yard;
    clear(this.bannerEl);
    const grownup = y.props.find((p) => p.kind === 'grownup');
    this.bannerEl.append(
      h('div', { class: 'banner-vs' }, teamPatch(m.cfg.away.team, 52), h('span', null, 'at'), teamPatch(m.cfg.home.team, 52)),
      h('div', { class: 'banner-title' }, lettering(y.name.toUpperCase(), { style: 'comic', size: 24, color: 'var(--sunshine)', seed: 'yard' })),
      h('div', { class: 'banner-sub' }, `${y.owner}. ${y.blurb}`),
      h('ul', { class: 'banner-rules' }, ...y.rules.map((r) => h('li', null, r)), grownup?.label ? h('li', null, `Watching: ${grownup.label.replace(/\.$/, '')}.`) : null),
      h('div', { class: 'banner-tap' }, 'Tap to play ball'));
    this.bannerEl.classList.add('intro');
    this.bannerEl.classList.remove('hidden');
    this.introT = 9;
  }

  endIntro() {
    this.introT = 0;
    this.bannerEl.classList.add('hidden');
    this.bannerEl.classList.remove('intro');
    this.director.cut();
    this.showCard(this.match.batter, this.match.battingSide, 'Up to bat');
  }

  // ─────────────────────────────────────────────────────────── coach

  private maybeCoach(mode: string) {
    if (settings.coachDone || this.coach || this.paused) return;
    const m = this.match;
    if ((mode === 'bat' || mode === 'pitch') && this.director.shotT < 0.6) return; // let the camera settle first
    let tip: CoachTip | null = null;
    if (mode === 'bat' && m.phase === 'prePitch') tip = 'swing';
    else if (mode === 'pitch' && !this.meter) tip = 'pitch';
    else if (mode === 'field' && m.play!.holder >= 0) tip = 'throw';
    else if (mode === 'run' && m.play!.t > 0.5) tip = 'run';
    if (!tip || settings.coachSeen.includes(tip)) return;
    this.coach = tip;
    const t = COACH_TEXT[tip];
    const body = matchMedia('(pointer: coarse)').matches ? t.body : COACH_DESKTOP[tip] ?? t.body;
    clear(this.coachEl);
    this.coachEl.className = `coach at-${tip}`;
    this.coachEl.append(
      h('div', { class: 'coach-title' }, t.title),
      ...body.map((b) => h('p', null, b)),
      h('button', { class: 'btn ctl', onpointerdown: (e: Event) => { e.preventDefault(); e.stopPropagation(); this.dismissCoach(); } }, 'Got it'));
  }

  private dismissCoach() {
    const tip = this.coach;
    if (!tip) return;
    audio.play('uiTap');
    this.coach = null;
    this.coachEl.classList.add('hidden');
    settings.coachSeen = [...settings.coachSeen.filter((x) => x !== tip), tip];
    settings.coachDone = COACH_TIPS.every((x) => settings.coachSeen.includes(x));
    saveSettings();
    this.last = performance.now();
  }

  // ─────────────────────────────────────────────────────────── pause / end

  private togglePause(force?: boolean) {
    if ((this.match.phase as string) === 'over') return;
    const next = force ?? !this.paused;
    if (next === this.paused) return;
    this.paused = next;
    audio.play(this.paused ? 'uiBack' : 'uiTap');
    if (this.paused) {
      this.showPauseMenu();
    } else {
      this.root.querySelector('.pausemenu')?.remove();
      audio.unlock(); // a phone call can leave audio suspended
      this.keepAwake();
      this.last = performance.now();
    }
  }

  private showPauseMenu() {
    const m = this.match;
    const toggle = (label: string, on: boolean, set: (v: boolean) => void) => h('label', { class: 'toggle' },
      h('input', { type: 'checkbox', checked: on, onchange: (e: Event) => { set((e.target as HTMLInputElement).checked); saveSettings(); } }), ` ${label}`);
    const quit = h('button', { class: 'btn ghost danger', onclick: () => {
      if (!quit.classList.contains('armed')) { quit.classList.add('armed'); quit.textContent = matchMedia('(pointer: coarse)').matches ? 'Tap again to quit' : 'Click again to quit'; return; }
      this.destroy(); this.opts.onExit(null);
    } }, 'Quit to menu');
    const menu = h('div', { class: 'pausemenu overlay', onpointerdown: (e: Event) => e.stopPropagation() },
      panel([h('div', { class: 'pause-panel' },
        h('div', { class: 'pp-main' },
          h('div', { class: 'pp-score' }, `${m.cfg.away.team.abbr} ${m.score[0]} — ${m.cfg.home.team.abbr} ${m.score[1]} · ${m.half === 0 ? 'Top' : 'Bottom'} of the ${ordinal(m.inning)}`),
          h('button', { class: 'btn big go resume', onclick: () => this.togglePause(false) }, icon('play'), h('span', null, 'Back to the game'))),
        h('div', { class: 'pp-side' },
          toggle('Show strike zone', settings.showZone, (v) => { settings.showZone = v; }),
          toggle('Voices', settings.voices > 0, (v) => { settings.voices = v ? 0.8 : 0; audio.setVoiceVolume(settings.voices); }),
          toggle('Buzz on big plays', settings.haptics, (v) => { settings.haptics = v; }),
          h('button', { class: 'btn ghost', onclick: () => this.simToEnd() }, 'Sim to the end'),
          quit))], { title: 'TIME OUT!' }));
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
    this.meter = null;
    this.root.querySelector('.pausemenu')?.remove();
    this.handleEvents();
  }

  private onGameOver() {
    const m = this.match;
    const won = this.humanSide >= 0 && m.winner === this.humanSide;
    this.releaseWake();
    // clear the stage for the final screen
    this.cardT = 0;
    this.cardsEl.classList.remove('show');
    this.tickerQueue.length = 0;
    this.tickerEl.classList.remove('show');
    audio.playMusic(this.humanSide >= 0 && m.winner !== -1 && !won ? 'defeat' : 'victory');
    if (won) { audio.play('bigCheer'); this.buzz([0, 40, 60, 40, 60, 90]); }
    let star: string | null = null, best = -1;
    for (const [id, l] of Object.entries(m.box)) {
      const tb = l.bat.h + l.bat.d + l.bat.t * 2 + l.bat.hr * 3;
      const score = tb * 2 + l.bat.rbi * 1.5 + l.bat.r + l.pitch.so * 0.8 + (l.side === m.winner ? 1 : 0);
      if (score > best) { best = score; star = id; }
    }
    const innings = Math.max(m.cfg.innings, m.inning);
    const line = (side: 0 | 1) => {
      const t = this.teamOf(side);
      const cells = Array.from({ length: innings }, (_, i) => h('td', null, m.line[side][i] ?? (i < m.inning ? 'x' : '')));
      return h('tr', null, h('th', null, t.abbr), ...cells, h('td', { class: 'tot' }, String(m.score[side])), h('td', null, String(m.hits[side])), h('td', null, String(m.errors[side])));
    };
    const header = h('tr', null, h('th', null, ''), ...Array.from({ length: innings }, (_, i) => h('th', null, String(i + 1))), h('th', null, 'R'), h('th', null, 'H'), h('th', null, 'E'));
    const sk = star ? kid(star) : null;
    const starLine = sk ? (() => {
      const b = m.box[sk.id].bat, p = m.box[sk.id].pitch;
      const parts = [`${b.h} for ${b.ab}`];
      if (b.hr) parts.push(`${b.hr} HR`);
      if (b.rbi) parts.push(`${b.rbi} RBI`);
      if (p.so) parts.push(`${p.so} K`);
      return parts.join(', ');
    })() : '';
    const title = m.winner === -1 ? 'It\'s a tie!' : this.humanSide < 0 ? `${this.teamOf(m.winner as 0 | 1).name} win!` : won ? 'You win!' : 'Tough loss!';
    const starCard = sk ? tradingCard({
      photo: this.portrait(sk, 'happy', 200), name: sk.nick, persona: starLine, team: this.teamOfKid(sk) ?? this.teamOf(0),
      stats: [['Hits', m.box[sk.id].bat.h], ['RBI', m.box[sk.id].bat.rbi], ['K', m.box[sk.id].pitch.so]], class: 'final-card', seed: sk.id,
    }) : null;
    const final = h('div', { class: 'overlay final' },
      panel([h('div', { class: 'final-panel' },
        h('div', { class: 'fp-main' },
          h('table', { class: 'linescore' }, header, line(0), line(1)),
          h('div', { class: 'cta' },
            h('button', { class: 'btn go', onclick: () => { this.destroy(); this.opts.onExit(m, true); } }, icon('replay'), h('span', null, 'Play again')),
            h('button', { class: 'btn ghost', onclick: () => { this.destroy(); this.opts.onExit(m); } }, icon('home'), h('span', null, 'Main menu')))),
        sk && starCard ? h('div', { class: 'fp-star' }, h('div', { class: 'card-tag' }, 'Player of the game'), starCard, h('div', { class: 'quote' }, `"${sk.quips[0]}"`)) : null)],
      { title: title.toUpperCase() }));
    setTimeout(() => { if (!this.destroyed) this.root.appendChild(final); }, 2500);
  }
}
