import { approach2, clamp, dist2, GRAVITY, MPH, type Vec2 } from '../engine/math';
import type { Rng } from '../engine/rng';
import type { Kid, Position } from '../data/types';
import { isFair, kidHeightFt, pastBases, type Field } from './field';
import { predict, stepBall, type Ball, type BallEvent, type PathSample } from './physics';
import type {
  Difficulty, FielderState, MatchEvent, OutKind, OutRecord, PlayResult, RunnerState,
} from './types';

export const FIELD_ORDER: Position[] = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF'];

export const fielderSpeed = (k: Kid) => 11 + k.traits.speed * 1.15;
export const runnerSpeed = (k: Kid) => 13.5 + k.traits.speed * 1.3 + (k.special === 'zoomies' ? 3.5 : 0);
export const throwSpeed = (k: Kid) => (28 + k.traits.arm * 2.7 + (k.special === 'rocketArm' ? 14 : 0)) * MPH;
/** Kids rainbow their long throws: effective speed drops past ~70 ft. */
export const throwSpeedAt = (k: Kid, d: number) => throwSpeed(k) / (1 + Math.max(0, d - 70) / 160);
const reachOf = (k: Kid) => 1.7 + k.traits.fielding * 0.09 + (k.special === 'flypaper' ? 0.9 : 0);
const catchHeight = (k: Kid) => kidHeightFt(k.look.height) * 1.15 + 0.6 + (k.special === 'springs' ? 4 : 0);
const reactionOf = (k: Kid) => 0.42 - k.traits.fielding * 0.018;
/** seconds to get the ball out of the glove and set to throw */
const gatherOf = (k: Kid) => 0.9 - k.traits.fielding * 0.05;
const isOF = (p: Position) => p === 'LF' || p === 'CF' || p === 'RF';

export interface PlaySetup {
  field: Field;
  rng: Rng;
  defense: Kid[]; // in FIELD_ORDER
  batter: Kid;
  runners: (Kid | null)[]; // on 1st, 2nd, 3rd
  outs: number;
  ball: Ball;
  humanDefense: boolean;
  humanOffense: boolean;
  autoThrowDelay: number;
  difficulty: Difficulty;
  /** the defense is CPU-controlled and gets the difficulty bump */
  cpuDefense: boolean;
}

type BallMode = 'batted' | 'loose' | 'thrown' | 'held' | 'dead';

interface ThrowInfo { from: number; base: number; receiver: number; wild: boolean }

export class LivePlay {
  readonly f: Field;
  private rng: Rng;
  readonly ball: Ball;
  mode: BallMode = 'batted';
  holder = -1;
  throwInfo: ThrowInfo | null = null;
  judged: 'pending' | 'fair' | 'foul' = 'pending';
  readonly fielders: FielderState[];
  readonly runners: RunnerState[];
  outs: number;
  t = 0;
  over = false;
  events: MatchEvent[] = [];
  deadKind: null | 'hr' | 'foul' | 'groundRule' = null;
  /** human pre-selected throw target (base 1..4) */
  throwRequest: number | null = null;
  private setup: PlaySetup;
  private plan = { chaser: -1, backup: -1, t: 0, point: { x: 0, y: 0 } as Vec2, fly: false, next: 0 };
  private path: PathSample[] = [];
  private stableT = 0;
  private outsList: OutRecord[] = [];
  private scored: string[] = [];
  private errors: number[] = [];
  private batterBaseAtError = -1;
  private firstFielder = -1;
  private caughtFly = false;
  private cancelledRuns = 0;
  private decisionT = 0;

  constructor(s: PlaySetup) {
    this.setup = s;
    this.f = s.field;
    this.rng = s.rng;
    this.ball = s.ball;
    this.outs = s.outs;
    this.fielders = s.defense.map((kid, idx) => {
      const pos = FIELD_ORDER[idx];
      const home = { ...s.field.defaultSpots[pos] };
      const p = pos === 'P' ? { x: 0.8, y: s.field.mound.y - 3 } : { ...home };
      return {
        idx, pos, kid, p, home, facing: 0,
        speed: fielderSpeed(kid) * (s.cpuDefense ? diffBoost(s.difficulty) : 1),
        task: 'idle', target: { ...p }, coverBase: -1, hasBall: false,
        reaction: reactionOf(kid) + (pos === 'P' ? 0.15 : 0), catchCd: 0, holdT: 0,
        anim: 'ready', animT: 0, lift: 0,
        misread: { x: s.rng.gauss() * (11 - kid.traits.fielding) * 1.1, y: s.rng.gauss() * (11 - kid.traits.fielding) * 2 },
      } satisfies FielderState;
    });

    const occupied = [true, !!s.runners[0], !!s.runners[1], !!s.runners[2]];
    const forcedAt = (b: number) => {
      for (let i = 0; i <= b; i++) if (!occupied[i]) return false;
      return true;
    };
    const mk = (kid: Kid, base: number, isBatter: boolean): RunnerState => ({
      kid, base, d: 0, dir: 0, forced: forcedAt(base), mustTag: false, isBatter, startBase: base,
      out: false, scored: false, goal: null, holdAt: null, manual: false,
      speed: runnerSpeed(kid), anim: 'stand', animT: 0, reevalT: 0,
    });
    this.runners = [mk(s.batter, 0, true)];
    for (let b = 1; b <= 3; b++) {
      const k = s.runners[b - 1];
      if (k) this.runners.push(mk(k, b, false));
    }
    this.replan();
    // batter takes off; other runners make their first read
    const batter = this.runners[0];
    batter.dir = 0;
    batter.anim = 'stand';
    this.batterDelay = 0.28;
    for (const r of this.runners) if (!r.isBatter) this.initialRead(r);
  }

  get L() { return this.f.base; }

  basePos(b: number): Vec2 {
    return this.f.bases[((b % 4) + 4) % 4];
  }

  runnerPos(r: RunnerState): Vec2 {
    const a = this.basePos(r.base);
    const b = this.basePos(r.base + 1);
    const u = r.d / this.L;
    return { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u };
  }

  // ───────────────────────────────────────────────────────────── update

  update(dt: number) {
    if (this.over) return;
    this.t += dt;
    const sub = Math.max(1, Math.ceil(dt / (1 / 120)));
    const h = dt / sub;
    for (let i = 0; i < sub && !this.over; i++) this.tick(h);
  }

  private tick(dt: number) {
    if (this.mode !== 'held' && this.mode !== 'dead') {
      const evs: BallEvent[] = [];
      stepBall(this.ball, this.f, dt, this.rng, evs);
      for (const e of evs) this.onBallEvent(e);
      if (this.over) return;
      const mode = this.mode as BallMode; // event handlers may have changed it
      if (this.judged === 'pending' && mode !== 'dead') this.judgeRolling();
      if (mode !== 'dead' && this.ball.dead && !this.deadKind) {
        // left the yard some other way (rare): treat as foul if never judged
        this.makeFoul();
        return;
      }
      if (this.t >= this.plan.next && mode !== 'dead') this.replan();
    }
    this.updateFielders(dt);
    if (this.over) return;
    this.updateRunners(dt);
    if (this.over) return;
    this.checkOuts();
    this.checkEnd(dt);
  }

  // ───────────────────────────────────────────────────────── ball events

  private onBallEvent(e: BallEvent) {
    switch (e.type) {
      case 'bounce':
        this.events.push({ type: 'bounce', x: e.x, y: e.y, speed: e.speed, surface: e.surface });
        if (this.judged === 'pending' && pastBases(this.f, e.x, e.y)) {
          if (isFair(e.x, e.y)) this.judged = 'fair';
          else return this.makeFoul();
        }
        if (this.mode === 'thrown') this.overthrow();
        this.onDrop();
        this.plan.next = 0;
        break;
      case 'fence': {
        this.events.push({ type: 'fence', kind: e.seg.kind, cleared: e.cleared });
        const fair = this.judged === 'fair' || (this.judged === 'pending' && isFair(e.x, e.y));
        if (e.seg.kind === 'house') return this.makeFoul();
        if (e.cleared) {
          if (!fair) return this.makeFoul();
          if (this.mode === 'thrown' || this.mode === 'loose') return this.makeGroundRule('bounce');
          if (this.ball.touched) return this.makeGroundRule('bounce');
          return this.makeHomeRun();
        }
        if (!fair) return this.makeFoul();
        this.judged = 'fair';
        if (e.seg.splash) {
          this.events.push({ type: 'splash' });
          return this.makeGroundRule('splash');
        }
        if (this.mode === 'thrown') this.overthrow();
        this.onDrop();
        this.plan.next = 0;
        break;
      }
      case 'water': {
        const fair = this.judged === 'fair' || isFair(e.x, e.y);
        this.events.push({ type: 'splash' });
        if (!fair) return this.makeFoul();
        this.judged = 'fair';
        return this.makeGroundRule('splash');
      }
      case 'canopy':
        this.events.push({ type: 'tree' });
        this.plan.next = 0;
        break;
      case 'obstacle':
        if (e.ob.effect === 'dog') this.events.push({ type: 'dog' });
        if (this.judged === 'pending' && isFair(e.x, e.y) && pastBases(this.f, e.x, e.y)) this.judged = 'fair';
        if (this.mode === 'thrown') this.overthrow();
        this.onDrop();
        this.plan.next = 0;
        break;
      case 'rest':
        if (this.judged === 'pending') {
          if (isFair(e.x, e.y)) this.judged = 'fair';
          else return this.makeFoul();
        }
        this.plan.next = 0;
        break;
    }
  }

  /** A throw got past everybody and hit the ground. */
  private overthrow() {
    const ti = this.throwInfo;
    this.mode = 'loose';
    this.ball.drag = true;
    this.throwInfo = null;
    if (ti?.wild) this.chargeError(ti.from);
    for (const r of this.runners) if (!r.manual) r.reevalT = 0;
  }

  private judgeRolling() {
    const b = this.ball.p;
    if (!this.ball.touched) return;
    if (b.y < -1.5) return this.makeFoul();
    if (pastBases(this.f, b.x, b.y)) {
      if (isFair(b.x, b.y)) this.judged = 'fair';
      else this.makeFoul();
    }
  }

  private makeFoul() {
    this.judged = 'foul';
    this.deadKind = 'foul';
    this.mode = 'dead';
    this.events.push({ type: 'call', call: 'foul' });
    this.over = true;
  }

  private makeHomeRun() {
    this.judged = 'fair';
    this.deadKind = 'hr';
    this.mode = 'dead';
    for (const r of this.runners) {
      if (r.out || r.scored) continue;
      r.goal = 4; r.dir = 1; r.holdAt = null; r.mustTag = false; r.anim = 'trot';
      r.speed = Math.min(r.speed, 15);
    }
    for (const fl of this.fielders) { fl.task = 'idle'; fl.target = { ...fl.p }; }
  }

  private makeGroundRule(why: 'splash' | 'bounce') {
    this.deadKind = 'groundRule';
    this.mode = 'dead';
    this.events.push({ type: 'groundRule', batter: this.runners[0].kid.id, why });
    for (const r of this.runners) {
      if (r.out || r.scored) continue;
      r.goal = Math.min(4, r.startBase + 2);
      if (r.base >= r.goal) { r.goal = r.base; r.dir = 0; r.d = 0; continue; }
      r.dir = 1; r.holdAt = null; r.mustTag = false; r.anim = 'trot';
    }
    for (const fl of this.fielders) { fl.task = 'idle'; fl.target = { ...fl.p }; }
  }

  // ─────────────────────────────────────────────────────────── planning

  private replan() {
    this.plan.next = this.t + 0.35;
    if (this.mode === 'held' || this.mode === 'dead') return;
    this.path = predict(this.ball, this.f, 8, 1 / 40);
    const path = this.path;
    let best = -1, bestT = Infinity, bestPt: PathSample = path[path.length - 1];
    let second = -1, secondT = Infinity;
    let bestFly = false;
    const receiver = this.mode === 'thrown' && this.throwInfo ? this.throwInfo.receiver : -1;

    for (const fl of this.fielders) {
      const ch = catchHeight(fl.kid), reach = reachOf(fl.kid);
      let tHit = Infinity, pt = path[path.length - 1], fly = false;
      for (let i = 1; i < path.length; i++) {
        const s = path[i];
        if (s.z > ch) continue;
        const d = Math.hypot(s.x - fl.p.x, s.y - fl.p.y) - reach;
        const need = Math.max(0, fl.reaction) + Math.max(0, d) / fl.speed;
        if (need <= s.t) { tHit = s.t; pt = s; fly = !s.touched; break; }
      }
      if (tHit === Infinity) {
        const last = path[path.length - 1];
        tHit = Math.max(0, fl.reaction) + Math.hypot(last.x - fl.p.x, last.y - fl.p.y) / fl.speed + 0.3;
        pt = last;
      }
      // pitchers and catchers let others take balls they can share
      let biased = tHit + (fl.pos === 'C' ? 0.7 : fl.pos === 'P' ? 0.4 : 0);
      if (fl.idx === receiver) biased -= 0.5;
      if (biased < bestT) {
        second = best; secondT = bestT;
        best = fl.idx; bestT = biased; bestPt = pt; bestFly = fly;
      } else if (biased < secondT) {
        second = fl.idx; secondT = biased;
      }
    }

    this.plan.chaser = best;
    this.plan.t = this.t + bestT;
    this.plan.point = { x: bestPt.x, y: bestPt.y };
    this.plan.fly = bestFly && !this.ball.touched && this.mode === 'batted';
    const deep = Math.abs(bestPt.x) + bestPt.y > 2 * this.f.s + 10;
    this.plan.backup = deep ? second : -1;
    this.assignCovers();
  }

  private assignCovers() {
    const chaser = this.plan.chaser;
    const taken = new Set<number>([chaser]);
    if (this.plan.backup >= 0) taken.add(this.plan.backup);
    if (this.holder >= 0) taken.add(this.holder);
    const idxOf = (p: Position) => FIELD_ORDER.indexOf(p);
    const rightSide = this.plan.point.x > 0;
    const prefs: [number, Position[]][] = [
      [0, ['C', 'P']],
      [1, ['1B', 'P', '2B']],
      [2, rightSide ? ['SS', '2B', 'CF'] : ['2B', 'SS', 'CF']],
      [3, ['3B', 'SS', 'LF']],
    ];
    for (const fl of this.fielders) if (!taken.has(fl.idx)) fl.coverBase = -1;
    for (const [base, list] of prefs) {
      for (const pos of list) {
        const i = idxOf(pos);
        if (taken.has(i)) continue;
        taken.add(i);
        const fl = this.fielders[i];
        fl.coverBase = base;
        fl.task = 'cover';
        const bp = this.basePos(base);
        fl.target = { x: bp.x, y: bp.y };
        break;
      }
    }
    for (const fl of this.fielders) {
      if (fl.idx === chaser && this.holder < 0) {
        fl.task = this.mode === 'thrown' && this.throwInfo?.receiver === fl.idx ? 'receive' : 'chase';
        // kids misjudge the ball at first and correct as it gets close
        const err = this.mode === 'thrown' ? 0 : clamp((this.plan.t - this.t - 0.25) / 1.1, 0, 1);
        fl.target = { x: this.plan.point.x + fl.misread.x * err, y: this.plan.point.y + fl.misread.y * err };
        fl.coverBase = -1;
      } else if (fl.idx === this.plan.backup) {
        fl.task = 'backup';
        const p = this.plan.point;
        fl.target = { x: p.x * 1.08, y: p.y * 1.08 };
        fl.coverBase = -1;
      } else if (!taken.has(fl.idx)) {
        fl.task = 'idle';
        // drift a step toward the action
        fl.target = { x: fl.home.x + (this.ball.p.x - fl.home.x) * 0.15, y: fl.home.y + (this.ball.p.y - fl.home.y) * 0.15 };
      }
    }
  }

  // ─────────────────────────────────────────────────────────── fielders

  private updateFielders(dt: number) {
    for (const fl of this.fielders) {
      fl.animT += dt;
      fl.catchCd -= dt;
      if (fl.reaction > 0) { fl.reaction -= dt; continue; }
      if (this.deadKind === 'hr' || this.deadKind === 'groundRule') {
        if (fl.anim === 'run') fl.anim = 'ready';
        continue;
      }
      const before = fl.p;
      let speed = fl.speed;
      if (fl.hasBall && fl.task !== 'carry' && fl.task !== 'tag') speed = 0;
      if (fl.task === 'cover' || fl.task === 'idle') speed *= 0.9;
      fl.p = approach2(fl.p, fl.target, speed * dt);
      const moved = dist2(before, fl.p);
      if (moved > 1e-3) fl.facing = Math.atan2(fl.p.x - before.x, fl.p.y - before.y);
      if (fl.anim !== 'catch' && fl.anim !== 'throw' && fl.anim !== 'dive' && fl.anim !== 'stumble' && fl.anim !== 'jump' || fl.animT > 0.45) {
        const next = moved / dt > 2 ? 'run' : 'ready';
        if (fl.anim !== next) { fl.anim = next; fl.animT = 0; }
      }
      fl.lift = Math.max(0, fl.lift - dt * 8);
    }
    if (this.mode === 'held') this.updateHolder(dt);
    else if (this.mode !== 'dead') this.tryCatches();
  }

  private tryCatches() {
    const b = this.ball;
    const thrown = this.mode === 'thrown';
    let bestI = -1, bestD = Infinity;
    for (const fl of this.fielders) {
      if (fl.catchCd > 0) continue;
      const d = Math.hypot(b.p.x - fl.p.x, b.p.y - fl.p.y);
      const isRecv = thrown && this.throwInfo?.receiver === fl.idx;
      const isThrower = thrown && this.throwInfo?.from === fl.idx;
      if (isThrower && this.t < 0.6 + (this.throwT ?? 0)) continue;
      const reach = reachOf(fl.kid) + (isRecv ? 1.2 : 0);
      if (d > reach || b.p.z > catchHeight(fl.kid) || b.p.z < -0.1) continue;
      if (d < bestD) { bestD = d; bestI = fl.idx; }
    }
    if (bestI < 0) return;
    const fl = this.fielders[bestI];
    const reach = reachOf(fl.kid);
    const sp = Math.hypot(b.v.x, b.v.y, b.v.z);
    const edge = bestD / reach;
    const kidH = kidHeightFt(fl.kid.look.height);
    const high = b.p.z > kidH * 1.15;
    let p: number;
    if (thrown) {
      p = 0.945 + fl.kid.traits.fielding * 0.005 - (this.throwInfo?.wild ? 0.35 : 0);
    } else {
      const hot = clamp((sp - (b.touched ? 55 : 42)) / (b.touched ? 70 : 45), 0, 1);
      const groundHop = b.touched && b.p.z > 0.6 && b.p.z < 3 ? 0.03 : 0;
      // catching on the dead run is hard when you're ten (flies only — rollers are easy)
      const inAir = !b.touched;
      const onTheRun = inAir && dist2(fl.p, fl.target) > 2.5 && fl.task === 'chase' ? 0.14 : 0;
      const slow = sp < 18;
      p = 0.97 - 0.2 * hot - (slow ? 0.05 : 0.25) * Math.max(0, edge - 0.5) - (10 - fl.kid.traits.fielding) * 0.012 - groundHop - (high ? 0.1 : 0) - onTheRun;
      if (slow && b.touched) p = Math.max(p, 0.95);
      if (fl.kid.special === 'flypaper') p = 1 - (1 - p) * 0.35;
      if (this.setup.cpuDefense) p = 1 - (1 - p) * diffCatch(this.setup.difficulty);
    }
    if (fl.reaction > 0) p *= 0.7; // reflex grab before they could even move
    p = clamp(p, 0.2, 0.998);
    const fielding = this.judged === 'pending' && b.touched;
    if (fielding) {
      if (isFair(b.p.x, b.p.y)) this.judged = 'fair';
      else return this.makeFoul();
    }
    if (this.firstFielder < 0 && !thrown) this.firstFielder = fl.idx;

    if (this.rng.chance(p)) {
      const flyOut = !b.touched && this.mode === 'batted' || (!b.touched && this.mode === 'loose' && this.flyLive);
      fl.hasBall = true;
      fl.holdT = 0;
      fl.anim = edge > 0.85 && !thrown ? 'dive' : high ? 'jump' : 'catch';
      fl.animT = 0;
      fl.lift = high ? b.p.z - kidH : 0;
      this.scooped = !thrown && b.touched;
      this.holder = fl.idx;
      this.mode = 'held';
      this.throwInfo = null;
      b.v = { x: 0, y: 0, z: 0 };
      this.events.push({ type: 'catch', fielder: fl.kid.id, fly: flyOut, hard: edge > 0.85 || high });
      this.decisionT = 0;
      if (flyOut) {
        this.judged = 'fair';
        this.caughtFly = true;
        this.recordOut(this.runners[0], 'fly', fl.idx);
        if (this.over) return;
        this.onFlyCaught();
      } else {
        this.onFielded();
      }
    } else {
      // bobble — the comedy portion of the program
      const wasThrow = thrown;
      const ti = this.throwInfo;
      fl.catchCd = 0.7;
      fl.anim = edge > 0.85 ? 'dive' : 'stumble';
      fl.animT = 0;
      this.flyLive = !b.touched && this.mode === 'batted';
      b.v = { x: b.v.x * -0.18 + this.rng.range(-6, 6), y: b.v.y * -0.18 + this.rng.range(-6, 6), z: Math.abs(b.v.z) * 0.2 + 4 };
      b.grounded = false;
      b.drag = true;
      this.mode = 'loose';
      this.throwInfo = null;
      const routine = p > 0.86;
      if (routine || wasThrow) this.chargeError(wasThrow && ti?.wild ? ti.from : fl.idx);
      this.events.push({ type: 'bobble', fielder: fl.kid.id });
      if (this.plan.fly && !this.flyLive) this.onDrop();
      this.replan();
    }
  }

  private flyLive = false;
  private batterDelay = 0;
  /** the holder picked a rolling ball off the grass */
  private scooped = false;
  private throwT: number | null = null;

  private chargeError(idx: number) {
    this.errors.push(idx);
    if (this.batterBaseAtError < 0) {
      const bat = this.runners[0];
      this.batterBaseAtError = bat.out ? 0 : bat.base;
    }
    this.events.push({ type: 'error', fielder: this.fielders[idx].kid.id });
  }

  private updateHolder(dt: number) {
    const fl = this.fielders[this.holder];
    fl.holdT += dt;
    this.decisionT -= dt;
    if (fl.task === 'carry') {
      if (dist2(fl.p, fl.target) < 0.8) { fl.task = 'hold'; }
      else return;
    }
    if (fl.task === 'tag') {
      const r = this.runners.find((x) => !x.out && !x.scored && x === this.tagTarget);
      if (!r || r.d <= 0.01 || r.dir === 0 && r.d === 0) { fl.task = 'hold'; }
      else { fl.target = this.runnerPos(r); return; }
    }
    const human = this.setup.humanDefense;
    const gather = gatherOf(fl.kid) + (isOF(fl.pos) && this.scooped ? 0.55 : 0);
    if (fl.holdT < gather) return; // still getting the ball out of the glove
    if (human && this.throwRequest !== null) {
      const b = this.throwRequest;
      this.throwRequest = null;
      this.executeTo(fl, b);
      return;
    }
    const aiReady = human ? fl.holdT >= Math.max(gather, this.setup.autoThrowDelay) : true;
    if (!aiReady || this.decisionT > 0) return;
    const choice = this.chooseThrow(fl);
    if (choice === null) {
      this.decisionT = 0.25;
      fl.task = 'hold';
      return;
    }
    this.executeTo(fl, choice);
  }

  private tagTarget: RunnerState | null = null;

  /** Throw (or run) the ball to base b (1..3, 4 = home). */
  private executeTo(fl: FielderState, b: number) {
    const bp = this.basePos(b);
    const d = dist2(fl.p, bp);
    if (d < 1.6) { fl.task = 'hold'; this.decisionT = 0.2; return; }
    const ts = throwSpeedAt(fl.kid, d);
    let receiver = this.fielders.find((x) => x.coverBase === (b % 4) && x.idx !== fl.idx);
    if (!receiver) {
      let bestD = Infinity;
      for (const x of this.fielders) {
        if (x.idx === fl.idx) continue;
        const dd = dist2(x.p, bp);
        if (dd < bestD) { bestD = dd; receiver = x; }
      }
      if (receiver) { receiver.coverBase = b % 4; receiver.task = 'cover'; receiver.target = { ...bp }; }
    }
    const runTime = d / fl.speed;
    const throwTime = 0.15 + d / ts;
    const recvEta = receiver ? dist2(receiver.p, bp) / receiver.speed : Infinity;
    // close enough to just step on it yourself, or nobody's there to catch it yet
    if (d < 30 && (runTime < throwTime + 0.35 || recvEta > throwTime + 0.15)) {
      fl.task = 'carry';
      fl.target = { ...bp };
      return;
    }
    if (!receiver) return;
    const wild = this.rng.chance((0.022 + (10 - fl.kid.traits.fielding) * 0.005 + (10 - fl.kid.traits.arm) * 0.003) * (isOF(fl.pos) ? 1.6 : 1));
    const sigma = 0.6 + (10 - fl.kid.traits.fielding) * 0.12;
    const tx = bp.x + (wild ? this.rng.range(-1, 1) * 9 : this.rng.gauss() * sigma);
    const ty = bp.y + (wild ? this.rng.range(-1, 1) * 9 : this.rng.gauss() * sigma);
    const tz = 3.6 + (wild ? this.rng.range(-1.5, 4) : this.rng.gauss() * sigma * 0.5);
    const z0 = kidHeightFt(fl.kid.look.height) * 0.85;
    const D = Math.hypot(tx - fl.p.x, ty - fl.p.y);
    const T = Math.max(0.25, D / ts);
    this.ball.p = { x: fl.p.x, y: fl.p.y, z: z0 };
    this.ball.v = {
      x: (tx - fl.p.x) / T,
      y: (ty - fl.p.y) / T,
      z: (tz - z0 + 0.5 * GRAVITY * T * T) / T,
    };
    this.ball.drag = false;
    this.ball.grounded = false;
    this.ball.resting = false;
    this.ball.touched = false;
    this.ball.canopy = -1;
    fl.hasBall = false;
    fl.task = 'idle';
    fl.anim = 'throw';
    fl.animT = 0;
    fl.target = { ...fl.p };
    this.holder = -1;
    this.mode = 'thrown';
    this.throwT = this.t;
    this.throwInfo = { from: fl.idx, base: b, receiver: receiver.idx, wild };
    receiver.task = 'receive';
    this.events.push({ type: 'throw', fielder: fl.kid.id, base: b });
    this.replan();
    // receiver heads for the base, then adjusts to the throw
    const pathEnd = this.plan.point;
    receiver.target = dist2(pathEnd, bp) < 8 ? { ...pathEnd } : { ...bp };
    for (const r of this.runners) if (!r.manual) r.reevalT = 0;
  }

  /** AI: where should the ball go? null = hold it. */
  private chooseThrow(fl: FielderState): number | null {
    let best: number | null = null;
    let bestScore = -Infinity;
    const live = this.runners.filter((r) => !r.out && !r.scored && r.goal === null);
    // chase down a runner caught off base nearby
    for (const r of live) {
      if (r.d > 0.5 && r.d < this.L - 0.5) {
        const rp = this.runnerPos(r);
        if (dist2(rp, fl.p) < 9) {
          this.tagTarget = r;
          fl.task = 'tag';
          fl.target = rp;
          return null;
        }
      }
    }
    for (const r of live) {
      let base: number;
      if (r.mustTag) base = r.base;
      else if (r.dir > 0) base = r.base + 1;
      else if (r.dir < 0 && r.d > 0) base = r.base;
      else continue;
      const remaining = r.mustTag || r.dir < 0 ? r.d : this.L - r.d;
      const tRunner = remaining / r.speed;
      const bp = this.basePos(base);
      const d = dist2(fl.p, bp);
      const ts = throwSpeedAt(fl.kid, d);
      const tBall = d < 22 ? Math.min(d / fl.speed, 0.3 + d / ts) : 0.3 + d / ts;
      const recv = this.fielders.find((x) => x.coverBase === base % 4);
      const tRecv = recv ? dist2(recv.p, bp) / recv.speed : 3;
      const tArrive = Math.max(tBall, d < 22 ? 0 : tRecv);
      const forceLike = (r.forced && r.dir > 0) || r.mustTag;
      const margin = tRunner - tArrive - (forceLike ? 0 : 0.25);
      const lead = base === 4 ? 4 : base;
      const score = margin > 0.05 ? 10 + lead * 2 + margin : -20 + lead + margin;
      if (score > bestScore) { bestScore = score; best = base; }
    }
    if (best !== null && bestScore > 0) return best;
    // nobody to get: keep the lead runner honest, else get it back to the infield
    const movers = live.filter((r) => r.dir > 0);
    if (movers.length) {
      const lead = movers.reduce((a, b) => (a.base > b.base ? a : b));
      const target = lead.base + 1;
      if (dist2(fl.p, this.basePos(target)) > 3) return target;
      return null;
    }
    const isOutfielder = fl.pos === 'LF' || fl.pos === 'CF' || fl.pos === 'RF';
    if (isOutfielder && dist2(fl.p, this.basePos(2)) > 30) return 2;
    return null;
  }

  // ───────────────────────────────────────────────────── runner brains

  private initialRead(r: RunnerState) {
    if (this.outs >= 2) { r.dir = 1; r.anim = 'run'; return; }
    if (this.plan.fly) {
      // catchable fly: go partway (or tag up from third) and see what happens;
      // deep flies get a bigger lead than liners an infielder might snag
      if (r.base === 3) { r.dir = 0; return; }
      const deep = Math.hypot(this.plan.point.x, this.plan.point.y) > 95;
      r.holdAt = this.L * (deep ? 0.5 : 0.25);
      r.dir = 1;
      r.anim = 'run';
      return;
    }
    if (r.forced) { r.dir = 1; r.anim = 'run'; return; }
    const pt = this.plan.point;
    const outfield = Math.abs(pt.x) + pt.y > 2 * this.f.s + 12;
    if (r.base === 2 && (pt.x > 6 || outfield)) { r.dir = 1; r.anim = 'run'; return; }
    if (r.base === 3 && outfield) { r.dir = 1; r.anim = 'run'; return; }
    r.dir = 0;
  }

  /** Seconds until the defense could have the ball at base b. */
  private threatTime(b: number): number {
    const bp = this.basePos(b);
    if (this.mode === 'held' && this.holder >= 0) {
      const fl = this.fielders[this.holder];
      const d = dist2(fl.p, bp);
      const g = gatherOf(fl.kid) + (isOF(fl.pos) && this.scooped ? 0.55 : 0);
      return Math.min(d / fl.speed, 0.15 + d / throwSpeedAt(fl.kid, d)) + Math.max(0, g - fl.holdT);
    }
    if (this.mode === 'thrown' && this.throwInfo) {
      const ti = this.throwInfo;
      const tgt = this.basePos(ti.base);
      const flightLeft = Math.max(0, dist2({ x: this.ball.p.x, y: this.ball.p.y }, tgt) / Math.max(30, Math.hypot(this.ball.v.x, this.ball.v.y)));
      if (ti.base % 4 === b % 4) return flightLeft;
      const recv = this.fielders[ti.receiver];
      return flightLeft + 0.35 + dist2(tgt, bp) / throwSpeedAt(recv.kid, dist2(tgt, bp));
    }
    if (this.plan.chaser < 0) return 9;
    const ch = this.fielders[this.plan.chaser];
    const tReach = Math.max(0, this.plan.t - this.t) + 0.25;
    const scoop = isOF(ch.pos) && this.ball.touched ? 0.55 : 0;
    const dd = dist2(this.plan.point, bp);
    return tReach + gatherOf(ch.kid) + scoop + 0.15 + dd / throwSpeedAt(ch.kid, dd);
  }

  private aheadBlocked(r: RunnerState, toBase: number): boolean {
    if (toBase >= 4) return false;
    return this.runners.some((o) => o !== r && !o.out && !o.scored && o.base === toBase && o.d === 0 && o.dir === 0 && !o.mustTag);
  }

  private wantsExtra(r: RunnerState): boolean {
    const next = r.base + 1;
    if (this.aheadBlocked(r, next)) return false;
    if (this.plan.fly && this.mode === 'batted' && this.outs < 2) return false;
    const tRun = this.L / r.speed;
    const margin = this.threatTime(next) - tRun;
    // kid coaches wave everybody around: close plays at the plate are the fun part
    let need = next === 4 ? -0.25 : -0.05;
    if (r.kid.special === 'zoomies') need -= 0.2;
    if (this.outs >= 2) need -= 0.15;
    return margin > need;
  }

  private onReachBase(r: RunnerState) {
    if (r.base >= 4) {
      r.scored = true;
      r.dir = 0;
      r.anim = 'cheer';
      this.scored.push(r.kid.id);
      this.events.push({ type: 'run', runner: r.kid.id });
      return;
    }
    r.forced = false;
    if (r.goal !== null) {
      if (r.base >= r.goal) { r.dir = 0; r.anim = 'stand'; }
      return;
    }
    if (r.manual) { r.manual = false; r.dir = 0; r.anim = 'stand'; return; }
    if (this.wantsExtra(r)) { r.dir = 1; r.anim = 'run'; }
    else { r.dir = 0; r.anim = 'stand'; if (r.isBatter || r.base > r.startBase) this.events.push({ type: 'safe', runner: r.kid.id, base: r.base }); }
  }

  private onFlyCaught() {
    for (const r of this.runners) {
      r.forced = false;
      if (r.isBatter || r.out || r.scored) continue;
      r.holdAt = null;
      if (r.d > 0.01 || r.base !== r.startBase) {
        if (r.base > r.startBase) { r.base = r.startBase; r.d = this.L - 0.01; }
        r.mustTag = true;
        r.dir = -1;
        r.anim = 'run';
      } else {
        this.tagUpRead(r);
      }
    }
  }

  private tagUpRead(r: RunnerState) {
    if (this.outs >= 3) return;
    const margin = this.threatTime(r.base + 1) - this.L / r.speed;
    if (!this.aheadBlocked(r, r.base + 1) && margin > 0.6) { r.dir = 1; r.anim = 'run'; }
  }

  /** A catchable fly fell in: everyone who was waiting reads it now. */
  private onDrop() {
    this.plan.fly = false;
    for (const r of this.runners) {
      if (r.isBatter || r.out || r.scored || r.manual) continue;
      if (r.holdAt === null && !(r.base === 3 && r.dir === 0 && r.d === 0)) continue;
      r.holdAt = null;
      if (r.forced) { r.dir = 1; r.anim = 'run'; continue; }
      if (this.wantsExtraFrom(r)) { r.dir = 1; r.anim = 'run'; }
      else if (r.d > 0) { r.dir = -1; r.anim = 'run'; }
    }
  }

  private wantsExtraFrom(r: RunnerState): boolean {
    const next = r.base + 1;
    if (this.aheadBlocked(r, next)) return false;
    const margin = this.threatTime(next) - (this.L - r.d) / r.speed;
    return margin > (next === 4 ? -0.1 : 0.05);
  }

  private onFielded() {
    for (const r of this.runners) {
      if (r.isBatter || r.out || r.scored || r.manual || r.goal !== null) continue;
      if (r.holdAt !== null) {
        r.holdAt = null;
        if (r.forced) { r.dir = 1; continue; }
        r.dir = r.d > this.L * 0.5 && this.wantsExtraFrom(r) ? 1 : -1;
      }
      r.reevalT = 0;
    }
  }

  private updateRunners(dt: number) {
    const L = this.L;
    if (this.batterDelay > 0) {
      this.batterDelay -= dt;
      const b = this.runners[0];
      if (this.batterDelay <= 0 && !b.out && b.d === 0 && b.base === 0 && b.goal === null) { b.dir = 1; b.anim = 'run'; }
    }
    for (const r of this.runners) {
      if (r.out || r.scored) continue;
      r.animT += dt;
      r.reevalT -= dt;
      if (r.dir !== 0) {
        r.d += r.dir * r.speed * dt;
        if (r.holdAt !== null && r.dir > 0 && r.d >= r.holdAt) {
          r.d = r.holdAt; r.dir = 0; r.anim = 'stand';
        }
        if (r.d >= L) {
          r.base += 1; r.d = 0;
          this.onReachBase(r);
          if (this.over) return;
        } else if (r.d <= 0 && r.dir < 0) {
          r.d = 0; r.dir = 0; r.anim = 'stand'; r.manual = false;
          if (r.mustTag) { r.mustTag = false; this.tagUpRead(r); }
        }
      }
      // mid-path re-read: bail out if it's hopeless
      if (r.dir > 0 && !r.forced && !r.manual && r.goal === null && r.reevalT <= 0 && r.d < L * 0.5) {
        r.reevalT = 0.2;
        const margin = this.threatTime(r.base + 1) - (L - r.d) / r.speed;
        if (margin < -0.15 && !this.plan.fly) { r.dir = -1; }
      }
      // standing on a base: take the extra base on a throw, an overthrow or a bobble
      if (r.dir === 0 && r.d === 0 && !r.mustTag && r.goal === null && !r.manual && r.reevalT <= 0
        && this.mode !== 'dead' && !(this.plan.fly && this.mode === 'batted')) {
        r.reevalT = 0.3;
        if (this.wantsExtra(r)) { r.dir = 1; r.anim = 'run'; }
      }
      // waiting halfway and the ball's been dealt with
      if (r.dir === 0 && r.d > 0 && r.holdAt === null && !r.manual && r.goal === null) {
        r.dir = this.wantsExtraFrom(r) ? 1 : -1;
        r.anim = 'run';
      }
    }
    // no two runners on the same base: trailing runner stops short
    for (const r of this.runners) {
      if (r.out || r.scored || r.dir <= 0) continue;
      const next = r.base + 1;
      if (next < 4 && this.aheadBlocked(r, next) && !r.forced && r.goal === null && r.d > L - 4) {
        r.dir = -1;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────── outs

  private checkOuts() {
    if (this.mode !== 'held' || this.holder < 0) return;
    if (this.deadKind) return;
    const fl = this.fielders[this.holder];
    for (const r of this.runners) {
      if (r.out || r.scored || r.goal !== null) continue;
      // force play or a runner who still has to tag up
      if (r.forced && r.dir > 0) {
        const bp = this.basePos(r.base + 1);
        if (dist2(fl.p, bp) <= 2.2) { this.recordOut(r, 'force', fl.idx); if (this.over) return; continue; }
      }
      if (r.mustTag) {
        const bp = this.basePos(r.base);
        if (dist2(fl.p, bp) <= 2.2) { this.recordOut(r, 'doubledOff', fl.idx); if (this.over) return; continue; }
      }
      const offBase = r.d > 0.35 && r.d < this.L - 0.35;
      if (offBase) {
        const rp = this.runnerPos(r);
        if (dist2(rp, fl.p) <= 2.5) { this.recordOut(r, 'tag', fl.idx); if (this.over) return; }
      }
    }
  }

  private recordOut(r: RunnerState, kind: OutKind, fielderIdx: number) {
    r.out = true;
    r.dir = 0;
    r.anim = 'out';
    r.animT = 0;
    this.outs++;
    this.outsList.push({ kind, kidId: r.kid.id, fielderIdx });
    this.events.push({ type: 'out', kind, runner: r.kid.id, fielder: this.fielders[fielderIdx].kid.id });
    if (r.isBatter) for (const o of this.runners) o.forced = false;
    if (this.outs >= 3) {
      const noRuns = kind === 'force' || kind === 'doubledOff' || (r.isBatter && r.base === 0);
      if (noRuns) this.cancelledRuns = this.scored.length;
      this.over = true;
      return;
    }
    this.decisionT = 0.12;
    if (this.tagTarget === r) this.tagTarget = null;
    const holder = this.fielders[fielderIdx];
    if (holder.task === 'tag') holder.task = 'hold';
  }

  private checkEnd(dt: number) {
    if (this.over) return;
    const live = this.runners.filter((r) => !r.out && !r.scored);
    if (this.deadKind === 'hr' || this.deadKind === 'groundRule') {
      if (live.every((r) => r.dir === 0 && r.d === 0)) this.over = true;
      return;
    }
    const settled = live.every((r) => r.dir === 0 && r.d === 0 && !r.mustTag);
    if (this.mode === 'held' && settled && this.fielders[this.holder].task !== 'carry') {
      this.stableT += dt;
      if (this.stableT > 0.55) this.over = true;
    } else {
      this.stableT = 0;
    }
    if (this.t > 40) {
      for (const r of live) { if (r.d > this.L / 2) { r.base++; } r.d = 0; r.dir = 0; if (r.base >= 4) { r.scored = true; this.scored.push(r.kid.id); } }
      this.over = true;
    }
  }

  // ─────────────────────────────────────────────────────────── controls

  /** Human defense taps a base: throw there as soon as we have the ball. */
  requestThrow(base: number) {
    this.throwRequest = base;
  }

  /** Human offense: send everyone, or send everyone back. */
  commandRunners(cmd: 'advance' | 'retreat') {
    for (const r of this.runners) {
      if (r.out || r.scored || r.goal !== null) continue;
      if (cmd === 'advance') {
        if (r.dir <= 0 && r.d === 0 && this.aheadBlocked(r, r.base + 1)) continue;
        r.dir = 1; r.holdAt = null; r.manual = true; r.anim = 'run';
      } else if (!r.forced && r.d > 0) {
        r.dir = -1; r.holdAt = null; r.manual = true; r.anim = 'run';
      }
    }
  }

  // ───────────────────────────────────────────────────────────── result

  result(): PlayResult {
    const bases: (string | null)[] = [null, null, null];
    const batter = this.runners[0];
    for (const r of this.runners) {
      if (r.out || r.scored) continue;
      const b = r.d > this.L / 2 ? r.base + 1 : r.base;
      if (b >= 1 && b <= 3) bases[b - 1] = r.kid.id;
    }
    const foul = this.deadKind === 'foul';
    let batterBase = batter.out ? 0 : batter.scored ? 4 : batter.base;
    if (this.deadKind === 'hr') batterBase = 4;
    const error = this.errors.length > 0;
    let hitBases = 0;
    if (!batter.out && !foul) {
      hitBases = error ? Math.max(0, this.batterBaseAtError) : batterBase;
      const otherForce = this.outsList.some((o) => o.kind === 'force' && o.kidId !== batter.kid.id);
      if (otherForce && batterBase === 1) hitBases = 0; // fielder's choice
    }
    if (this.deadKind === 'groundRule') hitBases = 2;
    return {
      foul,
      homeRun: this.deadKind === 'hr',
      groundRule: this.deadKind === 'groundRule',
      outs: this.outsList,
      scored: this.scored.slice(0, this.scored.length - this.cancelledRuns),
      batterBase,
      bases,
      error,
      errorBy: this.errors,
      hitBases: Math.min(4, hitBases),
      caughtFly: this.caughtFly,
      firstFielder: this.firstFielder,
      cancelledRuns: this.cancelledRuns,
    };
  }
}

function diffBoost(d: Difficulty) {
  return d === 'rookie' ? 0.92 : d === 'pro' ? 1 : 1.06;
}
function diffCatch(d: Difficulty) {
  return d === 'rookie' ? 1.35 : d === 'pro' ? 1 : 0.7;
}
