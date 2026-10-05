import { Rng } from '../engine/rng';
import { SPECIAL_INFO, type Kid, type PitchType, type Team, type Yard } from '../data/types';
import { kid } from '../data/kids';
import { cpuChoosePitch, cpuDecideSwing } from './ai';
import { batArrival, batSide, resolveSwing, type SwingInput, type SwingKind, type SwingTuning } from './batting';
import { buildField, kidHeightFt, strikeZone, type Field } from './field';
import { newBall } from './physics';
import { isStrike, makePitch, pitchPos, type ActivePitch } from './pitching';
import { LivePlay } from './play';
import { emptyBat, emptyPitch, type BoxScore } from './stats';
import type { Lineup } from './lineup';
import type { Difficulty, MatchEvent, PlayResult } from './types';

export interface SideConfig {
  team: Team;
  lineup: Lineup;
  human: boolean;
}

export interface MatchConfig {
  away: SideConfig;
  home: SideConfig;
  yard: Yard;
  innings: number;
  seed: number;
  difficulty: Difficulty;
  /** run lead that ends the game early once half the innings are done (0 = off) */
  mercy?: number;
  /** seconds a human gets to pick a throw before the CPU does it */
  autoThrowDelay?: number;
  /** headless sim: skip the cosmetic pauses */
  fast?: boolean;
}

export type Phase = 'prePitch' | 'windup' | 'pitch' | 'live' | 'result' | 'halfOver' | 'over';

const HUMAN_TUNE: Record<Difficulty, SwingTuning> = {
  rookie: { window: 1.35, radius: 1.3 },
  pro: { window: 1.1, radius: 1.1 },
  allstar: { window: 1, radius: 1 },
};
const CPU_TUNE: SwingTuning = { window: 1, radius: 1 };

export const WINDUP = 0.8;

export class Match {
  readonly cfg: MatchConfig;
  readonly field: Field;
  readonly rng: Rng;
  inning = 1;
  half: 0 | 1 = 0;
  outs = 0;
  balls = 0;
  strikes = 0;
  bases: (string | null)[] = [null, null, null];
  score: [number, number] = [0, 0];
  line: number[][] = [[], []];
  hits: [number, number] = [0, 0];
  errors: [number, number] = [0, 0];
  batterIdx: [number, number] = [0, 0];
  pitchCount: [number, number] = [0, 0];
  hype: [number, number] = [0, 0];
  phase: Phase = 'prePitch';
  phaseT = 0;
  pitch: ActivePitch | null = null;
  pitchT = 0;
  swingIn: SwingInput | null = null;
  private swingDone = false;
  whiffed = false;
  play: LivePlay | null = null;
  /** the play that just ended (for drawing the aftermath) */
  lastPlay: LivePlay | null = null;
  lastResult: PlayResult | null = null;
  lastCall: string | null = null;
  events: MatchEvent[] = [];
  box: BoxScore = {};
  winner: 0 | 1 | -1 | null = null;
  /** human-picked pitch waiting for the windup */
  private queuedPitch: { type: PitchType; aim: { x: number; z: number }; special: boolean } | null = null;
  private batterDone = false;
  private playStartOuts = 0;

  constructor(cfg: MatchConfig) {
    this.cfg = cfg;
    this.field = buildField(cfg.yard);
    this.rng = new Rng(cfg.seed);
    for (const [side, sc] of [[0, cfg.away], [1, cfg.home]] as const) {
      for (const id of sc.lineup.order) this.box[id] = { bat: emptyBat(), pitch: emptyPitch(), e: 0, side };
      this.line[side] = [];
    }
    this.line[0][0] = 0;
    this.emitBatterUp();
  }

  // ───────────────────────────────────────────────────────── accessors

  side(i: 0 | 1) { return i === 0 ? this.cfg.away : this.cfg.home; }
  get battingSide(): 0 | 1 { return this.half; }
  get fieldingSide(): 0 | 1 { return (1 - this.half) as 0 | 1; }
  get batter(): Kid { const s = this.side(this.battingSide); return kid(s.lineup.order[this.batterIdx[this.battingSide]]); }
  get pitcher(): Kid { return kid(this.side(this.fieldingSide).lineup.defense[0]); }
  defenseKids(): Kid[] { return this.side(this.fieldingSide).lineup.defense.map(kid); }
  onDeck(): Kid { const s = this.side(this.battingSide); return kid(s.lineup.order[(this.batterIdx[this.battingSide] + 1) % 9]); }
  get humanBatting() { return this.side(this.battingSide).human; }
  get humanPitching() { return this.side(this.fieldingSide).human; }
  get zone() { return strikeZone(kidHeightFt(this.batter.look.height)); }
  get batterSide() { return batSide(this.batter.bats, this.pitcher.throws); }
  hypeFull(side: 0 | 1) { return this.hype[side] >= 100; }
  get fatigue() { return Math.max(0, (this.pitchCount[this.fieldingSide] - 55) / 60); }

  // ───────────────────────────────────────────────────── human controls

  selectPitch(type: PitchType, aim: { x: number; z: number }, special = false) {
    if (this.phase !== 'prePitch' || !this.humanPitching) return;
    this.beginPitch(type, aim, special);
  }

  swing(aimX: number, aimZ: number, kind: SwingKind, special = false) {
    if (!this.humanBatting || this.swingIn) return;
    if (this.phase === 'windup') {
      this.swingIn = { aimX, aimZ, tSwing: this.phaseT - WINDUP, kind, special: special && this.canSpecial(this.battingSide, this.batter) };
    } else if (this.phase === 'pitch') {
      this.swingIn = { aimX, aimZ, tSwing: this.pitchT, kind, special: special && this.canSpecial(this.battingSide, this.batter) };
    } else return;
    if (this.swingIn.special) this.useSpecial(this.battingSide, this.batter);
  }

  throwTo(base: number) { this.play?.requestThrow(base); }
  runners(cmd: 'advance' | 'retreat') { this.play?.commandRunners(cmd); }

  canSpecial(side: 0 | 1, k: Kid) {
    const kind = SPECIAL_INFO[k.special].kind;
    return this.hypeFull(side) && (side === this.battingSide ? kind === 'bat' : kind === 'pitch');
  }

  private useSpecial(side: 0 | 1, k: Kid) {
    this.hype[side] = 0;
    this.events.push({ type: 'special', kid: k.id, special: k.special });
  }

  // ─────────────────────────────────────────────────────────── the loop

  update(dt: number) {
    if (this.phase === 'over') return;
    this.phaseT += dt;
    const fast = !!this.cfg.fast;
    switch (this.phase) {
      case 'prePitch':
        if (!this.humanPitching && this.phaseT >= (fast ? 0 : 1.0)) {
          const plan = cpuChoosePitch(this.pitcher, this.countObj(), this.zone, this.hypeFull(this.fieldingSide), this.cfg.difficulty, this.rng);
          this.beginPitch(plan.type, plan.aim, plan.special);
        }
        break;
      case 'windup':
        if (this.phaseT >= WINDUP) this.release();
        break;
      case 'pitch':
        this.pitchT += dt;
        this.updatePitch();
        break;
      case 'live':
        this.play!.update(dt);
        this.drainPlay();
        if (this.play!.over) this.finishPlay();
        break;
      case 'result':
        if (this.phaseT >= (fast ? 0 : this.resultHold)) this.afterResult();
        break;
      case 'halfOver':
        if (this.phaseT >= (fast ? 0 : 2.4)) this.startHalf();
        break;
    }
  }

  private resultHold = 1;

  private countObj() { return { balls: this.balls, strikes: this.strikes, outs: this.outs }; }

  private setPhase(p: Phase) { this.phase = p; this.phaseT = 0; }

  private beginPitch(type: PitchType, aim: { x: number; z: number }, special: boolean) {
    const p = this.pitcher;
    this.lastPlay = null;
    const useSp = special && this.canSpecial(this.fieldingSide, p);
    if (useSp) this.useSpecial(this.fieldingSide, p);
    this.queuedPitch = { type, aim, special: useSp };
    this.swingIn = null;
    this.swingDone = false;
    this.whiffed = false;
    this.lastCall = null;
    this.setPhase('windup');
  }

  private release() {
    const q = this.queuedPitch!;
    const p = this.pitcher;
    this.pitch = makePitch(p, q.type, q.aim, q.special ? p.special : null, this.field.mound.y, this.rng, this.fatigue);
    this.pitchT = 0;
    this.pitchCount[this.fieldingSide]++;
    this.box[p.id].pitch.pitches++;
    this.events.push({ type: 'pitch', pitcher: p.id, pitch: q.type, special: this.pitch.special, mph: this.pitch.mph });
    this.setPhase('pitch');
    if (!this.humanBatting) {
      const s = cpuDecideSwing(this.batter, this.pitch, this.countObj(), this.zone, this.hypeFull(this.battingSide), this.cfg.difficulty, this.rng);
      if (s) {
        if (s.special && this.canSpecial(this.battingSide, this.batter)) this.useSpecial(this.battingSide, this.batter);
        else s.special = false;
        this.swingIn = s;
      }
    }
  }

  private updatePitch() {
    const pitch = this.pitch!;
    const s = this.swingIn;
    if (s && !this.swingDone) {
      const tHit = s.kind === 'bunt' ? pitch.Treal : batArrival(s);
      if (this.pitchT >= tHit) {
        this.swingDone = true;
        const tune = this.humanBatting ? HUMAN_TUNE[this.cfg.difficulty] : CPU_TUNE;
        const out = resolveSwing(this.batter, this.batterSide, pitch, s, tune, this.rng);
        if (out.kind === 'contact') {
          this.events.push({ type: 'contact', batter: this.batter.id, ev: out.ev, la: out.la, spray: out.spray, quality: out.quality });
          this.startLive(out.contactPoint, out.v);
          return;
        }
        if (out.kind === 'foulTip') {
          this.events.push({ type: 'call', call: 'foul' });
          this.callFoul();
          return;
        }
        this.whiffed = true;
        this.events.push({ type: 'whiff', batter: this.batter.id });
      }
    }
    if (this.pitchT >= pitch.Treal + 0.2) {
      if (s) return this.callStrike(false); // any swing that didn't connect
      const arr = pitch.arrival;
      const side = this.batterSide;
      const bx = side === 'R' ? -2.25 : 2.25;
      if (Math.abs(arr.x - bx) < 0.75 && arr.z < kidHeightFt(this.batter.look.height) * 0.9 && arr.z > 0.4) return this.callWalk(true);
      if (isStrike(arr, this.zone)) this.callStrike(true);
      else this.callBall();
    }
  }

  /** Ball position during the pitch for rendering. */
  pitchBallPos() {
    if (!this.pitch) return null;
    return pitchPos(this.pitch, this.pitchT);
  }

  private startLive(cp: { x: number; y: number; z: number }, v: { x: number; y: number; z: number }) {
    const fielding = this.side(this.fieldingSide);
    const batting = this.side(this.battingSide);
    this.playStartOuts = this.outs;
    this.play = new LivePlay({
      field: this.field,
      rng: this.rng,
      defense: this.defenseKids(),
      batter: this.batter,
      runners: this.bases.map((id) => (id ? kid(id) : null)),
      outs: this.outs,
      ball: newBall(cp, v, true),
      humanDefense: fielding.human,
      humanOffense: batting.human,
      autoThrowDelay: this.cfg.autoThrowDelay ?? 1.6,
      difficulty: this.cfg.difficulty,
      cpuDefense: !fielding.human && batting.human,
    });
    this.setPhase('live');
  }

  private drainPlay() {
    const p = this.play!;
    if (p.events.length) {
      this.events.push(...p.events);
      p.events.length = 0;
    }
  }

  // ───────────────────────────────────────────────────────── outcomes

  private callBall() {
    this.balls++;
    this.lastCall = 'ball';
    this.events.push({ type: 'call', call: 'ball' });
    if (this.balls >= 4) return this.callWalk(false);
    this.toResult(0.7);
  }

  private callStrike(looking: boolean) {
    this.strikes++;
    this.lastCall = looking ? 'strike' : 'swinging';
    this.events.push({ type: 'call', call: looking ? 'strike' : 'swinging' });
    if (this.strikes >= 3) {
      const b = this.batter;
      this.box[b.id].bat.pa++;
      this.box[b.id].bat.ab++;
      this.box[b.id].bat.so++;
      this.box[this.pitcher.id].pitch.so++;
      this.box[this.pitcher.id].pitch.outs++;
      this.outs++;
      this.addHype(this.fieldingSide, 8);
      this.events.push({ type: 'strikeout', batter: b.id, looking });
      this.maybeQuip(this.pitcher, 0.45);
      this.batterDone = true;
      return this.toResult(1.3);
    }
    this.toResult(0.7);
  }

  private callFoul() {
    if (this.strikes < 2) this.strikes++;
    this.lastCall = 'foul';
    this.toResult(0.8);
  }

  private callWalk(hbp: boolean) {
    const b = this.batter;
    this.lastCall = hbp ? 'hbp' : 'walk';
    this.box[b.id].bat.pa++;
    this.box[b.id].bat.bb++;
    this.box[this.pitcher.id].pitch.bb++;
    this.events.push({ type: 'walk', batter: b.id, hbp });
    const runs = this.forceAdvance(b.id);
    if (runs) this.box[b.id].bat.rbi += runs;
    this.addHype(this.battingSide, 4);
    this.batterDone = true;
    this.toResult(1.2);
  }

  /** Walk: push forced runners up one base. Returns runs scored. */
  private forceAdvance(batterId: string): number {
    const [f, s, t] = this.bases;
    let runs = 0;
    if (f && s && t) { this.scoreRun(t); runs++; }
    if (f && s) this.bases[2] = s;
    if (f) this.bases[1] = f;
    this.bases[0] = batterId;
    return runs;
  }

  private scoreRun(id: string) {
    const side = this.battingSide;
    this.score[side]++;
    this.line[side][this.inning - 1] = (this.line[side][this.inning - 1] ?? 0) + 1;
    this.box[id].bat.r++;
    this.box[this.pitcher.id].pitch.r++;
    this.events.push({ type: 'run', runner: id });
  }

  private finishPlay() {
    const play = this.play!;
    this.lastPlay = play;
    const res = play.result();
    this.lastResult = res;
    const b = this.batter;
    const bl = this.box[b.id].bat;
    const pl = this.box[this.pitcher.id].pitch;
    if (res.foul) {
      // a caught foul pop is an out (handled as a fly by the play)
      if (res.outs.length) {
        this.outs += res.outs.length;
        pl.outs += res.outs.length;
        bl.pa++; bl.ab++;
        this.batterDone = true;
        this.lastCall = 'out';
        this.play = null;
        return this.toResult(1.4);
      }
      this.callFoul();
      this.play = null;
      return;
    }
    const outsMade = res.outs.length;
    this.outs = Math.min(3, this.playStartOuts + outsMade);
    pl.outs += outsMade;
    const side = this.battingSide;
    for (const id of res.scored) {
      this.score[side]++;
      this.line[side][this.inning - 1] = (this.line[side][this.inning - 1] ?? 0) + 1;
      this.box[id].bat.r++;
      pl.r++;
    }
    const runs = res.scored.length;
    const doublePlay = outsMade >= 2;
    bl.pa++;
    const sacFly = res.caughtFly && runs > 0 && this.outs < 3;
    if (!sacFly) bl.ab++;
    if (!res.error && !doublePlay) bl.rbi += runs;
    else if (res.error) bl.rbi += Math.max(0, runs - 1);
    if (res.hitBases >= 1 && res.batterBase >= 1) {
      bl.h++;
      this.hits[side]++;
      pl.h++;
      if (res.hitBases === 2) bl.d++;
      if (res.hitBases === 3) bl.t++;
      if (res.hitBases >= 4) { bl.hr++; pl.hr++; }
      if (res.homeRun) this.events.push({ type: 'homeRun', batter: b.id, runs });
      else this.events.push({ type: 'hit', batter: b.id, bases: res.hitBases });
      this.addHype(side, res.hitBases >= 4 ? 35 : res.hitBases >= 2 ? 20 : 11);
      this.maybeQuip(b, res.hitBases >= 2 ? 0.7 : 0.3);
    }
    this.addHype(side, runs * 6);
    this.addHype(this.fieldingSide, outsMade * 6 + (doublePlay ? 14 : 0));
    if (res.caughtFly && outsMade) {
      const fl = play.fielders[res.outs[0].fielderIdx];
      this.maybeQuip(fl.kid, 0.25);
    }
    for (const i of res.errorBy) {
      const k = play.fielders[i].kid;
      this.box[k.id].e++;
      this.errors[this.fieldingSide]++;
    }
    this.bases = this.outs >= 3 ? [null, null, null] : res.bases;
    this.batterDone = true;
    this.lastCall = res.homeRun ? 'homeRun' : res.hitBases >= 1 ? 'hit' : outsMade ? 'out' : 'safe';
    this.play = null;
    this.toResult(res.homeRun ? 2.2 : 1.5);
  }

  private addHype(side: 0 | 1, n: number) {
    this.hype[side] = Math.min(100, this.hype[side] + n);
  }

  private maybeQuip(k: Kid, p: number) {
    if (!k.quips.length || !this.rng.chance(p)) return;
    this.events.push({ type: 'quip', kid: k.id, text: this.rng.pick(k.quips) });
  }

  private toResult(hold: number) {
    this.resultHold = hold;
    this.setPhase('result');
  }

  private afterResult() {
    this.pitch = null;
    if (this.checkWalkOff()) return;
    if (this.outs >= 3) return this.endHalf();
    if (this.batterDone) this.nextBatter();
    this.setPhase('prePitch');
  }

  private nextBatter() {
    const side = this.battingSide;
    this.batterIdx[side] = (this.batterIdx[side] + 1) % 9;
    this.balls = 0;
    this.strikes = 0;
    this.batterDone = false;
    this.emitBatterUp();
  }

  private emitBatterUp() {
    this.events.push({ type: 'batterUp', batter: this.batter.id, pitcher: this.pitcher.id });
  }

  private checkWalkOff(): boolean {
    if (this.half === 1 && this.inning >= this.cfg.innings && this.score[1] > this.score[0]) {
      this.endGame();
      return true;
    }
    return false;
  }

  private endHalf() {
    const inn = this.inning, half = this.half;
    this.events.push({ type: 'halfOver', inning: inn, half });
    const n = this.cfg.innings;
    const [a, h] = this.score;
    // home is already ahead after the top of the last inning: no need to bat
    if (half === 0 && inn >= n && h > a) return this.endGame();
    if (half === 1 && inn >= n && a !== h) return this.endGame();
    if (half === 1 && inn >= n + 3) return this.endGame(); // kids have to go home for dinner
    const mercy = this.cfg.mercy ?? 0;
    if (mercy > 0 && inn >= Math.ceil(n / 2) && half === 1 && Math.abs(a - h) >= mercy) return this.endGame();
    this.nextBatter();
    this.setPhase('halfOver');
  }

  private startHalf() {
    this.outs = 0;
    this.balls = 0;
    this.strikes = 0;
    this.bases = [null, null, null];
    if (this.half === 0) this.half = 1;
    else { this.half = 0; this.inning++; }
    this.line[this.half][this.inning - 1] = this.line[this.half][this.inning - 1] ?? 0;
    this.batterDone = false;
    this.emitBatterUp();
    this.setPhase('prePitch');
  }

  private endGame() {
    const [a, h] = this.score;
    this.winner = a > h ? 0 : h > a ? 1 : -1;
    this.events.push({ type: 'gameOver', winner: this.winner });
    this.setPhase('over');
  }
}

/** Run a whole CPU-vs-CPU game as fast as possible. */
export function simulateMatch(cfg: MatchConfig, maxSteps = 400000): Match {
  const m = new Match({ ...cfg, fast: true, away: { ...cfg.away, human: false }, home: { ...cfg.home, human: false } });
  let steps = 0;
  while (m.phase !== 'over' && steps++ < maxSteps) {
    m.update(m.phase === 'live' ? 1 / 30 : 1 / 30);
    m.events.length = 0;
  }
  return m;
}
