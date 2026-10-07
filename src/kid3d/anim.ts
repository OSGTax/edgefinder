import { Euler, MathUtils, Quaternion, Vector3, type Bone } from 'three';
import type { KidModel } from './model';
import { B, BONES } from './rig';
import { orientBone, pointBone, twoBoneIK } from './ik';
import type { Expression } from './face';
import {
  add, blendPose, compile, copyPose, HX, HY, HZ, mirrorPose, newPose, set, smooth, Track, fullTrack, UPPER,
  type Frame, type Pose, type PoseDef,
} from './pose';
import { personaOf, type Anchor, type Gesture, type HandGoal, type Personality } from './personality';

// Procedural animation for the kids. Every frame the game says what a kid
// is doing (a Mode plus a little context) and the animator builds a target
// pose from hand-tuned key poses and cycles, eases each bone toward it, then
// layers on head/eye tracking, blinking, breathing and IK (bat grips, glove
// reaches, hands to the face for persona fidgets). Baseball actions are
// authored for right-handers and mirrored; persona gestures are not (props
// live in the right hand). Each kid's quirks come from `personality.ts`.
// Nothing here allocates per frame: poses are flat arrays (`pose.ts`).

export type Mode =
  | 'stand' | 'ready' | 'crouch' | 'run' | 'trot' | 'walk' | 'catch' | 'throw' | 'dive' | 'jump' | 'stumble'
  | 'slide' | 'cheer' | 'sad' | 'sit' | 'clap' | 'bat' | 'swing' | 'bunt' | 'windup' | 'follow' | 'out' | 'wave' | 'grill'
  // reactions and life around the game
  | 'celebrate' | 'strikeout' | 'catchJoy' | 'groan' | 'highfive' | 'pump' | 'sitCheer' | 'sitGroan' | 'watch' | 'skim'
  | 'homer' | 'mope';

export interface AnimInput {
  mode: Mode;
  /** seconds in this mode (swing: since the swing started; windup: 0..WINDUP) */
  t: number;
  /** ground speed, ft/s (in the kid's own size: divide by any model scale) */
  speed?: number;
  /** world-space point to look at */
  lookAt?: Vector3 | null;
  /** world-space point the glove hand should reach for */
  reach?: Vector3 | null;
  /** jump height (ft) for leaps */
  lift?: number;
  /** batting: aim height of the swing (ft) and swing kind */
  aimZ?: number;
  power?: boolean;
  /** windup: total duration */
  windup?: number;
  /** the kid bats/throws left-handed: mirror everything */
  lefty?: boolean;
  /** seat height when sitting */
  seat?: number;
  /** how fast the kid is turning on the spot (rad/s, + = to their left) */
  turn?: number;
  /** picks between generic variants (cheers, groans) so neighbours don't move in unison */
  variant?: number;
}

const lerp = MathUtils.lerp;
const clamp = MathUtils.clamp;
const TAU = Math.PI * 2;
const env = (t: number, a: number, b: number, fade: number) => smooth((t - a) / fade) * (1 - smooth((t - (b - fade)) / fade));

// ─────────────────────────────────────────────────────────────── key poses
// (right-handed; +x rotation swings hanging limbs back / leans the torso
// forward; armL +z raises the left arm sideways, armR −z the right)

const STAND_DEF: PoseDef = {
  armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0],
  thighL: [0, 0, 0.03], thighR: [0, 0, -0.03],
};
const STAND = compile(STAND_DEF);

const READY_DEF: PoseDef = {
  hipY: -0.32, hips: [0.35, 0, 0], spine: [0.15, 0, 0], chest: [0.05, 0, 0], neck: [-0.2, 0, 0], head: [-0.35, 0, 0],
  thighL: [-0.75, 0, 0.18], thighR: [-0.75, 0, -0.18], shinL: [1.0, 0, 0], shinR: [1.0, 0, 0], footL: [-0.25, 0, 0], footR: [-0.25, 0, 0],
  armL: [-0.95, 0, 0.12], foreL: [-0.55, 0, 0], armR: [-0.8, 0, -0.12], foreR: [-0.7, 0, 0],
};
const READY = compile(READY_DEF);

const CROUCH = compile({ // catcher
  hipY: -1.05, hips: [0.15, 0, 0], spine: [0.2, 0, 0], neck: [-0.1, 0, 0], head: [-0.25, 0, 0],
  thighL: [-1.75, 0, 0.42], thighR: [-1.75, 0, -0.42], shinL: [2.25, 0, 0], shinR: [2.25, 0, 0], footL: [-0.5, 0, 0], footR: [-0.5, 0, 0],
  armL: [-1.3, 0, 0.2], foreL: [-0.4, 0, 0], armR: [-0.6, 0, -0.3], foreR: [-1.2, 0, 0],
});

const SAD_DEF: PoseDef = {
  ...STAND_DEF, spine: [0.25, 0, 0], chest: [0.15, 0, 0], neck: [0.25, 0, 0], head: [0.45, 0, 0],
  shoulderL: [0, 0, -0.15], shoulderR: [0, 0, 0.15], armL: [-0.15, 0, 0.05], armR: [-0.15, 0, -0.05],
};
const SAD = compile(SAD_DEF);

const SIT = compile({
  thighL: [-1.55, 0, 0.12], thighR: [-1.5, 0, -0.12], shinL: [1.45, 0, 0], shinR: [1.6, 0, 0],
  spine: [0.1, 0, 0], armL: [-0.55, 0, 0.18], armR: [-0.55, 0, -0.18], foreL: [-0.8, 0, 0], foreR: [-0.8, 0, 0],
});

const BAT_DEF: PoseDef = {
  hipY: -0.22, hips: [0.12, -0.25, 0], spine: [0.2, -0.1, 0], chest: [0, -0.15, 0], neck: [0, 0.5, 0], head: [0.08, 0.8, 0],
  thighL: [-0.3, 0, 0.32], thighR: [-0.25, 0, -0.32], shinL: [0.55, 0, 0], shinR: [0.5, 0, 0], footL: [-0.1, 0.4, 0], footR: [-0.1, -0.3, 0],
  armL: [-0.9, 0, -0.3], foreL: [-1.2, 0, 0], armR: [-0.5, 0, -0.9], foreR: [-1.6, 0, 0],
};
const BAT_STANCE = compile(BAT_DEF);
const BUNT = compile({ ...BAT_DEF, hips: [0.1, 0.9, 0], chest: [0.05, 0.3, 0], head: [0.1, 0.1, 0], thighL: [-0.4, 0, 0.3], shinL: [0.6, 0, 0], thighR: [-0.4, 0, -0.3], shinR: [0.6, 0, 0], hipY: -0.35 });

// ─────────────────────────────────────────────────────────────── actions
// Timings are hand-tuned: a beat of anticipation before each big move, the
// fast part eased in so it snaps, then a slower settle ('out') afterwards.

const THROW = fullTrack([
  [0, { ...READY_DEF, hipY: -0.2 }],
  // gather: crow-hop weight onto the back leg, glove arm points at the target
  [0.13, { hipY: -0.1, hips: [0.05, -0.75, 0], spine: [-0.02, -0.3, 0], chest: [-0.12, -0.38, 0], head: [0, 0.95, 0],
    armR: [0.45, 0, -1.55], foreR: [-1.75, 0, 0], armL: [-1.35, 0, 0.55], foreL: [-0.3, 0, 0],
    thighL: [-0.75, 0, 0.1], shinL: [0.95, 0, 0], thighR: [0.1, 0, -0.15], shinR: [0.35, 0, 0] }],
  // max external rotation: hips already opening, the arm lags behind (overlap)
  [0.2, { hipY: -0.2, hips: [0.1, -0.15, 0], spine: [0.1, -0.15, 0], chest: [-0.05, -0.25, 0], head: [-0.1, 0.5, 0],
    armR: [0.55, 0, -1.7], foreR: [-1.9, 0.2, 0], armL: [-0.9, 0, 0.7], foreL: [-0.9, 0, 0],
    thighL: [-0.8, 0, 0.1], shinL: [0.55, 0, 0], thighR: [0.25, 0, -0.12], shinR: [0.6, 0, 0] }, 'io'],
  // release: whip through, glove tucks
  [0.26, { hipY: -0.28, hips: [0.18, 0.3, 0], spine: [0.35, 0.3, 0], chest: [0.22, 0.38, 0], head: [-0.25, 0.1, 0],
    armR: [-2.4, 0, -0.5], foreR: [-0.35, 0, 0], armL: [-0.25, 0, 0.6], foreL: [-1.4, 0, 0],
    thighL: [-0.8, 0, 0.1], shinL: [0.4, 0, 0], thighR: [0.45, 0, -0.1], shinR: [0.9, 0, 0] }, 'in'],
  // follow-through: arm across the body, back leg swings round
  [0.44, { hipY: -0.32, hips: [0.28, 0.5, 0], spine: [0.5, 0.25, 0], chest: [0.2, 0.2, 0], head: [-0.4, -0.1, 0],
    armR: [-0.6, 0, 0.6], foreR: [-0.55, 0, 0], armL: [0.25, 0, 0.4], foreL: [-1.0, 0, 0],
    thighL: [-0.6, 0, 0.1], shinL: [0.6, 0, 0], thighR: [0.65, 0, -0.1], shinR: [1.25, 0, 0], footR: [0.6, 0, 0] }, 'out'],
  [0.8, READY_DEF],
]);

const JUMP = fullTrack([
  [0, { ...READY_DEF, hipY: -0.5 }],
  [0.12, { hipY: 0, spine: [-0.1, 0, 0], head: [-0.6, 0, 0], neck: [-0.3, 0, 0], armL: [-2.6, 0, 0.5], foreL: [-0.2, 0, 0], armR: [-0.6, 0, -0.9], foreR: [-0.6, 0, 0], thighL: [-0.9, 0, 0.1], shinL: [1.5, 0, 0], thighR: [0.2, 0, -0.1], shinR: [1.1, 0, 0], footL: [0.6, 0, 0], footR: [0.7, 0, 0] }, 'out'],
  [0.5, { hipY: 0, spine: [0, 0, 0], head: [-0.4, 0, 0], armL: [-2.8, 0, 0.3], armR: [-0.4, 0, -0.8], thighL: [-0.5, 0, 0.1], shinL: [0.8, 0, 0], thighR: [-0.2, 0, -0.1], shinR: [0.6, 0, 0] }],
  // landing: absorb deep, then stand
  [0.72, { ...READY_DEF, hipY: -0.6, hips: [0.5, 0, 0], shinL: [1.3, 0, 0], shinR: [1.3, 0, 0] }, 'in'],
  [1.1, READY_DEF, 'out'],
]);

const STUMBLE = fullTrack([
  [0, READY_DEF],
  [0.12, { hipY: -0.15, hips: [-0.15, 0.3, 0.1], spine: [-0.35, 0.2, 0.15], head: [0.2, 0.4, 0.2], armL: [-1.8, 0, 1.3], armR: [-1.6, 0, -1.5], foreL: [-0.6, 0, 0], foreR: [-0.6, 0, 0], thighL: [-0.6, 0, 0.3], shinL: [0.5, 0, 0], thighR: [0.3, 0, -0.2], shinR: [0.4, 0, 0] }, 'out'],
  [0.3, { hipY: -0.2, hips: [0.1, 0.1, -0.05], spine: [0.1, 0.0, -0.1], head: [0.1, -0.2, -0.1], armL: [-1.2, 0, 1.0], armR: [-1.0, 0, -1.2], foreL: [-0.6, 0, 0], foreR: [-0.6, 0, 0], thighL: [0.2, 0, 0.2], shinL: [0.4, 0, 0], thighR: [-0.6, 0, -0.2], shinR: [0.8, 0, 0] }],
  [0.5, { hipY: -0.28, hips: [0.35, -0.2, -0.1], spine: [0.4, -0.2, 0], head: [0.2, -0.3, 0], armL: [-0.4, 0, 1.0], armR: [-0.5, 0, -1.0], thighL: [-0.3, 0, 0.2], shinL: [0.7, 0, 0], thighR: [-0.5, 0, -0.2], shinR: [0.9, 0, 0] }],
  [0.85, READY_DEF],
]);

/** A right-handed swing. t = seconds since the swing started; contact at `tc`. */
function swingTrack(tc: number) {
  return fullTrack([
    // load: hands back, hips coil, front knee tucks in
    [0, { ...BAT_DEF, hips: [0.12, -0.38, 0], chest: [0, -0.32, 0], thighL: [-0.25, 0, 0.5], shinL: [0.4, 0, 0] }],
    // stride + hips fire first; the hands are still back (the hips lead the bat)
    [tc * 0.55, { ...BAT_DEF, hipY: -0.24, hips: [0.15, 0.3, 0], spine: [0.25, 0.05, 0], chest: [0.05, 0.0, 0], head: [0.15, 0.6, 0], neck: [0, 0.3, 0], thighL: [-0.2, 0, 0.42], shinL: [0.25, 0, 0], thighR: [-0.35, 0, -0.3], shinR: [0.75, 0, 0], footR: [0.1, 0.5, 0] }],
    [tc, { ...BAT_DEF, hipY: -0.27, hips: [0.15, 0.85, 0], spine: [0.25, 0.25, 0], chest: [0.05, 0.28, 0], head: [0.25, -0.0, 0], neck: [0.1, -0.15, 0], thighL: [-0.15, 0, 0.35], shinL: [0.08, 0, 0], thighR: [-0.45, 0.4, -0.25], shinR: [0.95, 0, 0], footR: [0.25, 0.9, 0] }, 'in'],
    [tc + 0.12, { ...BAT_DEF, hipY: -0.2, hips: [0.1, 1.3, 0], spine: [0.15, 0.38, 0], chest: [0, 0.32, 0], head: [0.1, -0.5, 0], neck: [0, -0.4, 0], thighL: [-0.1, 0, 0.3], shinL: [0.05, 0, 0], thighR: [-0.5, 0.6, -0.15], shinR: [1.0, 0, 0], footR: [0.5, 1.1, 0] }, 'out'],
    [tc + 0.45, { ...BAT_DEF, hipY: -0.12, hips: [0.05, 1.45, 0], spine: [0.05, 0.3, 0], chest: [-0.05, 0.3, 0], head: [0.05, -0.8, 0], neck: [0, -0.4, 0], thighL: [-0.05, 0, 0.28], shinL: [0.05, 0, 0], thighR: [-0.4, 0.6, -0.1], shinR: [0.8, 0, 0], footR: [0.6, 1.2, 0] }, 'out'],
  ]);
}
const SWING = swingTrack(0.15), SWING_POWER = swingTrack(0.18);

/** A right-handed windup + delivery. u = 0..1 over the windup (release at 1). */
const WINDUP_DEFS: Frame[] = [
  [0, { ...STAND_DEF, head: [0.05, 0, 0], armL: [-0.75, 0, 0.35], foreL: [-1.3, -0.4, 0], armR: [-0.75, 0, -0.35], foreR: [-1.3, 0.4, 0] }],
  // rocker step back
  [0.16, { hipY: -0.06, hips: [-0.04, -0.45, 0], spine: [-0.08, -0.2, 0], head: [0, 0.6, 0], armL: [-1.05, 0, 0.3], foreL: [-1.5, -0.4, 0], armR: [-1.05, 0, -0.3], foreR: [-1.5, 0.4, 0], thighR: [0.05, 0, -0.05], thighL: [0.2, 0, 0.05], shinL: [0.3, 0, 0] }],
  // leg lift, balanced over the rubber: the pause before the go
  [0.42, { hipY: 0.03, hips: [-0.05, -1.25, 0], spine: [-0.12, -0.2, 0], chest: [0, -0.1, 0], head: [0.05, 1.2, 0], neck: [0, 0.2, 0],
    armL: [-1.1, 0, 0.4], foreL: [-1.6, -0.5, 0], armR: [-1.0, 0, -0.4], foreR: [-1.6, 0.5, 0],
    thighL: [-1.6, 0, 0.1], shinL: [1.55, 0, 0], footL: [0.5, 0, 0], thighR: [0.05, 0, -0.05], shinR: [0.25, 0, 0] }, 'out'],
  [0.5, { hipY: 0.02, hips: [-0.04, -1.2, 0], spine: [-0.1, -0.2, 0], chest: [0, -0.1, 0], head: [0.05, 1.15, 0], neck: [0, 0.2, 0],
    armL: [-1.1, 0, 0.4], foreL: [-1.6, -0.5, 0], armR: [-1.0, 0, -0.4], foreR: [-1.6, 0.5, 0],
    thighL: [-1.5, 0, 0.12], shinL: [1.5, 0, 0], footL: [0.45, 0, 0], thighR: [0.05, 0, -0.05], shinR: [0.28, 0, 0] }],
  // stride: hands break, arms spread like wings
  [0.74, { hipY: -0.34, hips: [0.1, -0.7, 0], spine: [0.0, -0.35, 0], chest: [-0.15, -0.42, 0], head: [0, 0.95, 0], neck: [0, 0.3, 0],
    armL: [-1.5, 0, 0.35], foreL: [-0.3, 0, 0], armR: [0.6, 0, -1.55], foreR: [-1.65, 0, 0],
    thighL: [-0.9, 0, 0.25], shinL: [0.6, 0, 0], footL: [0.1, 0, 0], thighR: [0.15, 0, -0.25], shinR: [0.6, 0, 0] }],
  [0.9, { hipY: -0.42, hips: [0.2, 0.05, 0], spine: [0.25, 0.25, 0], chest: [0.1, 0.25, 0], head: [-0.15, 0.2, 0],
    armL: [-0.7, 0, 0.6], foreL: [-1.2, 0, 0], armR: [0.3, 0, -1.6], foreR: [-1.8, 0.3, 0],
    thighL: [-0.85, 0, 0.2], shinL: [0.55, 0, 0], thighR: [0.45, 0, -0.2], shinR: [0.8, 0, 0] }],
  [1.0, { hipY: -0.45, hips: [0.3, 0.35, 0], spine: [0.4, 0.3, 0], chest: [0.15, 0.25, 0], head: [-0.3, 0, 0],
    armL: [-0.2, 0, 0.6], foreL: [-1.4, 0, 0], armR: [-2.5, 0, -0.45], foreR: [-0.35, 0, 0],
    thighL: [-0.8, 0, 0.2], shinL: [0.45, 0, 0], thighR: [0.55, 0, -0.15], shinR: [0.9, 0, 0], footR: [0.4, 0, 0] }, 'in'],
];
const WINDUP = fullTrack(WINDUP_DEFS);

const FOLLOW = fullTrack([
  [0, WINDUP_DEFS[WINDUP_DEFS.length - 1][1]],
  [0.14, { hipY: -0.5, hips: [0.45, 0.5, 0], spine: [0.55, 0.25, 0], chest: [0.2, 0.2, 0], head: [-0.45, -0.1, 0],
    armL: [0.3, 0, 0.5], foreL: [-1.4, 0, 0], armR: [-0.9, 0, 0.55], foreR: [-0.4, 0, 0],
    thighL: [-0.75, 0, 0.2], shinL: [0.6, 0, 0], thighR: [0.85, 0, -0.1], shinR: [1.1, 0, 0], footR: [0.5, 0, 0] }, 'out'],
  // fielding position
  [0.5, { ...READY_DEF, hipY: -0.3 }],
]);

/** After the ball is in the glove: give with it, then bring it to the throwing hand. */
const CATCH = fullTrack([
  [0, READY_DEF],
  [0.1, { ...READY_DEF, hipY: -0.42, hips: [0.42, 0, 0], spine: [0.2, 0, 0], armL: [-1.05, 0, 0.1], foreL: [-0.9, 0, 0] }, 'out'],
  [0.32, { ...READY_DEF, hipY: -0.3, spine: [0.1, 0, 0], armL: [-0.7, -0.3, 0.05], foreL: [-1.6, 0, 0], armR: [-0.7, 0.3, -0.05], foreR: [-1.6, 0, 0] }],
]);

const HIGHFIVE = new Track([
  [0, STAND_DEF],
  [0.28, { ...STAND_DEF, hipY: -0.12, spine: [0.05, 0, 0], armR: [-0.5, 0, -2.6], foreR: [-0.6, 0, 0], armL: [0.2, 0, 0.3] }, 'out'],
  [0.4, { ...STAND_DEF, hipY: 0.18, spine: [-0.15, 0.15, 0], armR: [-2.2, 0, -0.9], foreR: [-0.1, 0, 0], armL: [0.3, 0, 0.4], thighL: [-0.2, 0, 0.05], shinL: [0.4, 0, 0] }, 'in'],
  [0.62, { ...STAND_DEF, hipY: 0, spine: [-0.05, 0.1, 0], armR: [-1.6, 0, -0.9], foreR: [-0.9, 0, 0], armL: [0.1, 0, 0.3] }, 'out'],
  [1.1, STAND_DEF],
]);

// ─────────────────────────────────────────────────────────────── the animator

const _e = new Euler(), _q = new Quaternion();
const _v = new Vector3(), _w = new Vector3(), _hp = new Vector3(), _a = new Vector3(), _b = new Vector3(), _c = new Vector3();
const _pole = new Vector3();
const UP = new Vector3(0, 1, 0), DOWN_Y = new Vector3(0, -1, 0);

/** Modes that are baseball actions (mirrored for lefties); everything else is persona-driven. */
const MIRRORED: Partial<Record<Mode, true>> = {
  ready: true, crouch: true, catch: true, throw: true, dive: true, jump: true, stumble: true, slide: true,
  bat: true, swing: true, bunt: true, windup: true, follow: true, catchJoy: true,
};

export class Animator {
  private cur: Record<string, Quaternion> = {};
  private hipOff = new Vector3();
  private blinkT = 2 + Math.random() * 3;
  private blinkPhase = -1;
  private seed = Math.random() * 10;
  private time = 0;
  private lastMode: Mode | null = null;
  private modeAge = 0;
  /** run-cycle phase, advanced by ground speed so feet stay planted */
  private runPh = Math.random() * TAU;
  private stepPh = 0;
  private tgt = newPose();
  private tmp = newPose();
  private tmp2 = newPose();
  private dive: Track;
  private slide: Track;
  // idle fidgets
  private fidget: Gesture | null = null;
  private fidgetT = 0;
  private fidgetWait = 2 + Math.random() * 5;
  private lastFidget = -1;
  // hand goals this frame (from gestures and idle styles)
  private goalG: HandGoal[] = [];
  private goalW = new Float32Array(12);
  private goalN = 0;
  private lookMul = 1;
  private gestureExpr: Expression | null = null;
  readonly persona: Personality;
  private sitFidgets: Gesture[];
  /** set by the game: an expression override; otherwise picked from the mode */
  expression: Expression | null = null;
  /** the persona prop should be out (the game shows it in the right hand) */
  propOut = false;
  /** world-space bat transform for the batting renderer (handle point + barrel direction) */
  readonly batHandle = new Vector3();
  readonly batDir = new Vector3(0, 1, 0);
  batActive = false;
  // bat path scratch (kid-local)
  private bh = [0, 1, 2, 3, 4].map(() => new Vector3());
  private bd = [0, 1, 2, 3, 4].map(() => new Vector3());

  constructor(readonly kid: KidModel) {
    for (const n of BONES) this.cur[n] = new Quaternion();
    this.persona = personaOf(kid.kid.id);
    this.sitFidgets = this.persona.fidgets.filter((g) => g.sit !== false);
    // dive and slide lie the kid on the ground: hip heights depend on their size
    const lay = -kid.p.hipY + 0.55, low = -kid.p.hipY + 0.55;
    this.dive = fullTrack([
      [0, READY_DEF],
      // load: drop and lean into it
      [0.14, { hipY: -0.6, hips: [0.9, 0, 0], spine: [0.2, 0, 0], head: [-0.6, 0, 0], armL: [-2.4, 0, 0.2], armR: [-2.2, 0, -0.3], thighL: [-0.6, 0, 0.1], shinL: [0.9, 0, 0], thighR: [0.1, 0, -0.1], shinR: [0.8, 0, 0] }, 'out'],
      // full extension: superman
      [0.32, { hipY: lay + 0.3, hipZ: 0.5, hips: [1.4, 0, 0], spine: [0.0, 0, 0], neck: [-0.4, 0, 0], head: [-0.7, 0, 0], armL: [-3.05, 0, 0.15], foreL: [-0.05, 0, 0], armR: [-2.85, 0, -0.3], foreR: [-0.3, 0, 0], thighL: [0.25, 0, 0.12], thighR: [0.3, 0, -0.12], shinL: [0.3, 0, 0], shinR: [0.6, 0, 0], footL: [0.8, 0, 0], footR: [0.8, 0, 0] }, 'in'],
      // belly flop: legs kick up on impact
      [0.42, { hipY: lay, hipZ: 0.65, hips: [1.48, 0, 0], spine: [0.05, 0, 0], neck: [-0.45, 0, 0], head: [-0.7, 0, 0], armL: [-3.0, 0, 0.15], foreL: [-0.1, 0, 0], armR: [-2.8, 0, -0.3], foreR: [-0.3, 0, 0], thighL: [0.1, 0, 0.12], thighR: [0.2, 0, -0.12], shinL: [1.2, 0, 0], shinR: [1.5, 0, 0], footL: [0.8, 0, 0], footR: [0.8, 0, 0] }, 'out'],
      [0.95, { hipY: lay, hipZ: 0.6, hips: [1.4, 0, 0.05], spine: [0, 0, 0], neck: [-0.5, 0, 0], head: [-0.7, 0.2, 0], armL: [-2.9, 0, 0.3], armR: [-2.0, 0, -0.6], thighL: [0.2, 0, 0.12], thighR: [0.2, 0, -0.12], shinL: [0.8, 0, 0], shinR: [0.7, 0, 0], footL: [0.8, 0, 0], footR: [0.8, 0, 0] }],
      // push up to a knee
      [1.35, { hipY: -0.7, hips: [0.6, 0, 0], spine: [0.3, 0, 0], head: [-0.4, 0, 0], thighL: [-1.5, 0, 0.2], thighR: [-0.2, 0, -0.2], shinL: [2.0, 0, 0], shinR: [1.6, 0, 0], armL: [-1.2, 0, 0.3], armR: [-0.6, 0, -0.4] }],
      [1.7, READY_DEF, 'out'],
    ]);
    this.slide = fullTrack([
      [0, { hipY: -0.05, hips: [0.15, 0, 0], spine: [0.25, 0, 0], thighL: [-0.8, 0, 0.04], shinL: [0.9, 0, 0], thighR: [0.6, 0, -0.04], shinR: [1.2, 0, 0], armL: [0.8, 0, 0.18], armR: [-0.8, 0, -0.18], foreL: [-1.3, 0, 0], foreR: [-1.3, 0, 0] }],
      // drop: lead leg shoots out, arms fly up for balance
      [0.15, { hipY: low * 0.5, hips: [-0.6, 0, 0], spine: [0.15, 0, 0], armL: [-0.6, 0, 1.4], armR: [-0.4, 0, -1.6], thighL: [-1.25, 0, 0.1], shinL: [0.15, 0, 0], thighR: [-0.5, 0, -0.2], shinR: [1.6, 0, 0] }, 'out'],
      [0.35, { hipY: low, hips: [-1.05, 0, 0], spine: [0.38, 0, 0], neck: [0.3, 0, 0], head: [0.5, 0, 0], armL: [-2.6, 0, 0.7], foreL: [-0.4, 0, 0], armR: [-2.4, 0, -0.9], foreR: [-0.5, 0, 0], thighL: [-0.5, 0, 0.12], shinL: [0.05, 0, 0], footL: [0.5, 0, 0], thighR: [-0.1, 0, -0.25], shinR: [1.8, 0, 0], footR: [0.3, 0, 0] }],
      [0.9, { hipY: low, hips: [-1.0, 0, 0], spine: [0.42, 0, 0], neck: [0.2, 0, 0], head: [0.4, 0, 0], armL: [-2.2, 0, 0.6], armR: [-0.6, 0, -0.9], thighL: [-0.5, 0, 0.12], shinL: [0.05, 0, 0], thighR: [-0.1, 0, -0.25], shinR: [1.8, 0, 0] }],
    ]);
  }

  /** Start fidget i right now (dev previews; normally fidgets pick themselves). */
  playFidget(i: number) {
    this.fidget = this.persona.fidgets[i % this.persona.fidgets.length] ?? null;
    this.fidgetT = 0;
  }

  update(dt: number, inp: AnimInput) {
    this.time += dt;
    this.dt = dt;
    const k = this.kid;
    const P = this.persona;
    const t = inp.t;
    const o = this.tgt;
    let rate = 14;
    let expr: Expression = 'neutral';
    this.batActive = false;
    this.propOut = false;
    this.lookMul = 1;
    this.goalN = 0;
    this.gestureExpr = null;
    if (inp.mode !== this.lastMode) { this.lastMode = inp.mode; this.modeAge = 0; } else this.modeAge += dt;
    const v = inp.speed ?? 0;
    // the run phase only moves with the feet
    if (inp.mode === 'run' || inp.mode === 'trot' || inp.mode === 'walk' || inp.mode === 'homer' || inp.mode === 'mope') this.advanceGait(dt, v, inp.mode === 'run');

    switch (inp.mode) {
      case 'stand': this.idle(o, P, inp); rate = 6; break;
      case 'watch': this.idle(o, P, inp, false); rate = 6; break;
      case 'ready': copyPose(o, READY); this.readyBounce(o); rate = 10; expr = 'focus'; break;
      case 'crouch': copyPose(o, CROUCH); rate = 10; expr = 'focus'; break;
      case 'run': this.gait(o, v, false); rate = 18; expr = 'focus'; break;
      case 'trot': this.gait(o, v || 9, true); rate = 14; expr = 'happy'; break;
      case 'walk': this.gait(o, v || 4, true); rate = 10; break;
      case 'mope': {
        // head down, shoulders forward, arms barely swinging
        this.gait(o, v || 3.5, true);
        add(o, B.spine, 0.22, 0, 0); add(o, B.chest, 0.1, 0, 0); add(o, B.neck, 0.2, 0, 0); add(o, B.head, 0.35, 0, 0);
        add(o, B.shoulderL, 0, 0, -0.12); add(o, B.shoulderR, 0, 0, 0.12);
        o[B.armL * 3] *= 0.4; o[B.armR * 3] *= 0.4;
        rate = 10; expr = 'sad'; this.lookMul = 0.15;
        break;
      }
      case 'homer': {
        // the home-run trot: this kid's own trot, with their signature arm business
        this.gait(o, v || 10, true);
        const g = P.trot;
        if (g) this.layer(g, t % g.dur, o, 0.85 * smooth(t / 0.4));
        rate = 14; expr = 'happy';
        break;
      }
      case 'catch': CATCH.sample(t, o); rate = 20; expr = 'focus'; break;
      case 'throw': THROW.sample(t, o); rate = 32; expr = 'focus'; break;
      case 'dive': this.dive.sample(t, o); rate = 24; expr = t < 1 ? 'yell' : 'oops'; break;
      case 'jump': JUMP.sample(t, o); rate = 22; expr = 'yell'; break;
      case 'stumble': STUMBLE.sample(t, o); rate = 20; expr = 'oops'; break;
      case 'slide': this.slide.sample(t, o); rate = 20; expr = 'yell'; break;
      case 'cheer': this.cheer(o, inp.variant ?? 0); rate = 14; expr = 'yell'; break;
      case 'pump': this.pump(o, t, inp.variant ?? 0); rate = 16; expr = 'happy'; break;
      case 'clap': this.clap(o); rate = 14; expr = 'happy'; break;
      case 'wave': copyPose(o, STAND); set(o, B.armR, -0.2, 0, -2.6); set(o, B.foreR, 0, 0, -0.4 + Math.sin(this.time * 9) * 0.5); rate = 12; expr = 'happy'; break;
      case 'grill': this.grill(o); rate = 8; expr = 'happy'; break;
      case 'skim': this.skim(o, t); rate = 8; expr = 'focus'; break;
      case 'highfive': HIGHFIVE.sample(t, o); rate = 22; expr = t < 1.2 ? 'yell' : 'happy'; break;
      case 'groan': this.groan(o, t, inp.variant ?? 0); rate = 9; expr = 'oops'; break;
      case 'sad': case 'out': copyPose(o, SAD); this.breatheSad(o); rate = 6; expr = 'sad'; break;
      case 'celebrate': this.play(P.celebrate, t, o, STAND); rate = 16; expr = 'yell'; break;
      case 'strikeout': this.play(P.strikeout, t, o, STAND); rate = 12; expr = 'sad'; break;
      case 'catchJoy': this.play(P.catchJoy, t, o, READY); rate = 14; expr = 'happy'; break;
      case 'sit': case 'sitCheer': case 'sitGroan': {
        this.sit(o, inp);
        if (inp.mode === 'sitCheer') {
          const j = Math.abs(Math.sin(this.time * 7 + this.seed));
          set(o, B.armL, -0.4, 0, 2.3 + j * 0.3); set(o, B.armR, -0.4, 0, -2.3 - j * 0.3);
          set(o, B.foreL, -0.5, 0, 0); set(o, B.foreR, -0.5, 0, 0);
          o[B.spine * 3] = -0.1; o[HY] += j * 0.08; expr = 'yell';
        } else if (inp.mode === 'sitGroan') {
          set(o, B.spine, 0.45, 0, 0); set(o, B.chest, 0.2, 0, 0); set(o, B.neck, 0.2, 0, 0); set(o, B.head, 0.35, 0, 0);
          this.goal(GOAL_FACE_L, 1); this.goal(GOAL_FACE_R, 1);
          this.lookMul = 0; expr = 'sad';
        } else this.fidgets(dt, o, P, true);
        rate = 6;
        break;
      }
      case 'bat': this.stance(o, P); rate = 10; expr = 'focus'; this.batActive = true; break;
      case 'swing': (inp.power ? SWING_POWER : SWING).sample(t, o); rate = 45; expr = t < 0.3 ? 'yell' : 'focus'; this.batActive = true; break;
      case 'bunt': copyPose(o, BUNT); rate = 16; expr = 'focus'; this.batActive = true; break;
      case 'windup': WINDUP.sample(t / (inp.windup ?? 0.8), o); rate = 40; expr = 'focus'; break;
      case 'follow': FOLLOW.sample(t, o); rate = 30; expr = 'focus'; break;
      default: copyPose(o, STAND);
    }
    // stepping round when turning on the spot (feet don't swivel on the grass)
    const turning = Math.abs(inp.turn ?? 0);
    if (turning > 1.2 && (inp.mode === 'stand' || inp.mode === 'ready' || inp.mode === 'watch' || inp.mode === 'clap')) {
      this.stepPh += dt * 9;
      const st = Math.sin(this.stepPh), w = clamp((turning - 1.2) / 2, 0, 1);
      add(o, B.thighL, -Math.max(0, st) * 0.45 * w, 0, 0); add(o, B.shinL, Math.max(0, st) * 0.8 * w, 0, 0);
      add(o, B.thighR, -Math.max(0, -st) * 0.45 * w, 0, 0); add(o, B.shinR, Math.max(0, -st) * 0.8 * w, 0, 0);
    }
    let pose = o;
    const mirrored = !!inp.lefty && !!MIRRORED[inp.mode];
    if (mirrored) pose = mirrorPose(this.tmp2, o);

    // ease bones toward the target pose; a fresh mode eases in more gently so
    // nothing pops, fast actions (swing, throw, windup) keep their snap
    const ramp = rate >= 25 ? 1 : 0.45 + 0.55 * smooth(this.modeAge / 0.3);
    const a = 1 - Math.exp(-dt * rate * ramp);
    for (let i = 0; i < BONES.length; i++) {
      const n = BONES[i];
      const j = i * 3;
      _q.setFromEuler(_e.set(pose[j], pose[j + 1], pose[j + 2], 'YXZ'));
      this.cur[n].slerp(_q, a);
      k.bones[n].quaternion.copy(this.cur[n]);
    }
    // hips offset (crouch / bob / jump / lie down)
    const s = k.p.s;
    const abs = inp.mode === 'sit' || inp.mode === 'sitCheer' || inp.mode === 'sitGroan' || inp.mode === 'dive' || inp.mode === 'slide';
    _hp.set(pose[HX] * s, pose[HY] * (abs ? 1 : s) + (inp.lift ?? 0), pose[HZ] * s);
    this.hipOff.lerp(_hp, Math.min(1, a * 1.2));
    k.bones.hips.position.copy(k.p.joints.hips).add(this.hipOff);

    // breathing (faster and deeper after running)
    const br = Math.sin(this.time * 2.1 + this.seed) * 0.025;
    k.bones.chest.quaternion.multiply(_q.setFromEuler(_e.set(br, 0, 0)));
    k.group.updateMatrixWorld(true);

    // head + eyes track the look target
    if (inp.lookAt && this.lookMul > 0) this.look(inp.lookAt, inp.mode);

    // bat grip / glove reach / hands to the face
    if (this.batActive) this.solveBat(inp);
    if (inp.reach && (inp.mode === 'catch' || inp.mode === 'ready' || inp.mode === 'run' || inp.mode === 'jump' || inp.mode === 'dive')) {
      const glove = inp.lefty ? 'R' : 'L';
      this.reachFor(glove, inp.reach, inp.mode === 'catch' ? clamp(1 - (t - 0.08) / 0.25, 0, 1) : 0.85);
    }
    if (inp.mode === 'windup' && t / (inp.windup ?? 0.8) < 0.62) {
      // hands together at the chest (ball in the glove)
      const cw = k.bones.chest.localToWorld(_w.set(0, 0.05 * s, 0.55 * s));
      this.reachFor('L', cw, 0.8);
      this.reachFor('R', _c.copy(cw).add(_v.set(0, 0.03, 0)), 0.8);
    }
    for (let i = 0; i < this.goalN; i++) this.handTo(this.goalG[i], this.goalW[i]);

    this.blink(dt, this.expression ?? this.gestureExpr ?? expr);
    const want = this.expression ?? this.gestureExpr ?? expr;
    if (want !== k.expression) k.setExpression(want);
  }

  // ───────────────────────────────────────────────── cycles

  private advanceGait(dt: number, v: number, run: boolean) {
    const G = this.persona.gait;
    // cadence rises with speed; the stride is whatever keeps the planted foot
    // moving backward exactly as fast as the ground goes by
    const f = (run ? 1.45 + v * 0.045 : 1.3 + v * 0.05) * G.cadence;
    this.runPh += TAU * f * dt;
    if (this.runPh > TAU * 1000) this.runPh -= TAU * 1000;
    this.cadence = f;
  }
  private cadence = 2;

  /** Run / trot / walk, matched to ground speed v (ft/s). */
  private gait(o: Pose, v: number, easy: boolean) {
    const G = this.persona.gait;
    const L = this.kid.p.hipY * 0.95;
    const f = this.cadence;
    // planted foot speed = L·amp·2πf → amp
    const amp = clamp(v / (TAU * L * f), 0.06, 1.1);
    const a = clamp(v / 20, 0.12, 1.05);
    const runK = smooth((v - 7) / 6) * (easy ? 0.75 : 1);
    const ph = this.runPh;
    const s = Math.sin(ph), c = Math.cos(ph);
    // the arms trail the legs a touch (overlap)
    const sa = Math.sin(ph - 0.25);
    const knee = (x: number) => 0.3 + Math.max(0, x) * (0.8 + 0.9 * runK);
    const bounceRun = Math.abs(s) * 0.17 * a - 0.08 * a;     // up in the flight phase
    const bounceWalk = Math.abs(c) * 0.06 - 0.06;            // up over the planted leg
    const lean = (0.08 + 0.2 * a * runK) * G.lean;
    o.fill(0);
    o[HY] = lerp(bounceWalk, bounceRun, runK) * G.bounce;
    o[HX] = -s * 0.05 * (1 - runK);
    set(o, B.hips, lean * 0.6, s * 0.2 * amp, c * 0.05 * (1 - runK));
    set(o, B.spine, lean, -s * 0.14 * amp, 0);
    set(o, B.chest, 0.04, -s * 0.12 * amp, 0);
    // the head stays level: counter the torso
    set(o, B.neck, -lean * 0.6, s * 0.1 * amp, 0);
    set(o, B.head, -lean * 0.5, s * 0.05 * amp, 0);
    set(o, B.thighL, -s * amp - 0.1 * a, 0, 0.04);
    set(o, B.thighR, s * amp - 0.1 * a, 0, -0.04);
    set(o, B.shinL, knee(c) * Math.min(1, amp * 1.4 + 0.2), 0, 0);
    set(o, B.shinR, knee(-c) * Math.min(1, amp * 1.4 + 0.2), 0, 0);
    // toe-off and heel-strike
    set(o, B.footL, -0.15 * a + Math.max(0, -s) * 0.35 * amp, 0, 0);
    set(o, B.footR, -0.15 * a + Math.max(0, s) * 0.35 * amp, 0, 0);
    const arm = (0.35 + 0.65 * runK) * G.arms;
    const swingA = Math.min(1.0, amp * 1.1 + 0.15);
    set(o, B.armL, sa * swingA * arm, 0, 0.16);
    set(o, B.armR, -sa * swingA * arm, 0, -0.16);
    const elbow = lerp(0.35, 1.3, runK);
    set(o, B.foreL, -elbow - Math.max(0, -sa) * 0.3 * runK, 0, 0);
    set(o, B.foreR, -elbow - Math.max(0, sa) * 0.3 * runK, 0, 0);
  }

  /** Standing around: weight shifts, this kid's posture, now and then a fidget. */
  private idle(o: Pose, P: Personality, inp: AnimInput, fidget = true) {
    const I = P.idle;
    const t = this.time;
    const w = Math.sin(t * 0.7 * I.sway + this.seed) * 0.5 + 0.5; // weight shift
    copyPose(o, STAND);
    if (I.posture) blendPose(o, I.posture.poses[0], 1, I.posture.touched);
    o[HX] += (w - 0.5) * 0.1;
    add(o, B.hips, 0, 0, (w - 0.5) * 0.09);
    add(o, B.spine, 0.02, Math.sin(t * 0.33 + this.seed) * 0.08, -(w - 0.5) * 0.07);
    add(o, B.thighL, 0, 0, (0.5 - w) * 0.07); add(o, B.thighR, 0, 0, (0.5 - w) * 0.07);
    // the unweighted knee relaxes
    add(o, B.shinL, w < 0.5 ? 0.16 : 0.02, 0, 0); add(o, B.shinR, w > 0.5 ? 0.16 : 0.02, 0, 0);
    if (I.hands) for (const g of I.hands) this.goal(g, 1);
    if (I.prop) this.propOut = true;
    if (fidget && inp.mode === 'stand') this.fidgets(this.dt, o, P, false);
  }
  private dt = 1 / 60;

  private fidgets(dt: number, o: Pose, P: Personality, sitting: boolean) {
    const list = sitting ? this.sitFidgets : P.fidgets;
    if (!list.length) return;
    if (!this.fidget) {
      this.fidgetWait -= dt;
      if (this.fidgetWait > 0) return;
      let i = Math.floor(Math.random() * list.length);
      if (list.length > 1 && i === this.lastFidget) i = (i + 1) % list.length;
      this.lastFidget = i;
      this.fidget = list[i];
      this.fidgetT = 0;
    }
    const g = this.fidget;
    this.fidgetT += dt;
    if (this.fidgetT >= g.dur) {
      this.fidget = null;
      this.fidgetWait = 3 + Math.random() * 6;
      return;
    }
    const w = env(this.fidgetT, 0, g.dur, Math.min(0.35, g.dur * 0.25));
    this.layer(g, this.fidgetT, o, w, sitting ? UPPER : undefined);
  }

  /** Lay a gesture over the pose `o` at weight w (only the bones it names). */
  private layer(g: Gesture, t: number, o: Pose, w: number, limit?: Uint8Array) {
    g.track.sample(t, this.tmp);
    if (limit) {
      const m = g.track.touched;
      for (let b = 0; b < m.length; b++) _mask[b] = m[b] & limit[b];
      blendPose(o, this.tmp, w, _mask);
    } else blendPose(o, this.tmp, w, g.mask ?? g.track.touched);
    if (g.hands) for (const h of g.hands) {
      const hw = h.from !== undefined ? env(t, h.from, h.to ?? g.dur, 0.18) : 1;
      if (hw * w > 0.01) this.goal(h, hw * w * (h.w ?? 1));
    }
    if (g.prop && w > 0.2) this.propOut = true;
    if (g.look !== undefined) this.lookMul = lerp(this.lookMul, g.look, w);
    if (g.expr && w > 0.5) this.gestureExpr = g.expr;
  }

  /** A whole-body persona moment (celebration, strikeout, catch) over a base pose. */
  private play(g: Gesture, t: number, o: Pose, base: Pose) {
    copyPose(o, base);
    this.layer(g, g.loop ? t % g.dur : Math.min(t, g.dur), o, 1);
  }

  private readyBounce(o: Pose) {
    // infielders sway on the balls of their feet
    const b = Math.sin(this.time * 3.2 + this.seed);
    o[HY] += b * 0.025;
    add(o, B.hips, 0, b * 0.03, 0);
  }

  private breatheSad(o: Pose) {
    o[B.head * 3] += Math.sin(this.time * 0.9 + this.seed) * 0.05;
  }

  private sit(o: Pose, inp: AnimInput) {
    copyPose(o, SIT);
    o[HY] = (inp.seat ?? 1.6) + 0.12 - this.kid.p.hipY - 0.05;
    // swinging legs, out of step with each other
    const sw = Math.sin(this.time * 2.2 + this.seed) * 0.25;
    set(o, B.shinL, 1.45 + sw, 0, 0);
    set(o, B.shinR, 1.6 - Math.sin(this.time * 2.2 * 1.13 + this.seed + 1.7) * 0.22, 0, 0);
  }

  private cheer(o: Pose, variant: number) {
    const t = this.time + this.seed;
    copyPose(o, STAND);
    switch (variant % 4) {
      case 0: { // jumping with both arms up, hands waving out of step
        const j = Math.abs(Math.sin(t * 6.5));
        o[HY] = j * 0.5 - 0.05;
        set(o, B.spine, -0.15, 0, 0); set(o, B.neck, -0.2, 0, 0); set(o, B.head, -0.3, 0, 0);
        set(o, B.armL, -0.3, 0, 2.5 + Math.sin(t * 13) * 0.25); set(o, B.armR, -0.3, 0, -2.5 - Math.sin(t * 13 + 1) * 0.25);
        set(o, B.foreL, -0.4, 0, 0); set(o, B.foreR, -0.4, 0, 0);
        set(o, B.thighL, -0.3 * j, 0, 0.1); set(o, B.thighR, -0.3 * j, 0, -0.1); set(o, B.shinL, 0.6 * j, 0, 0); set(o, B.shinR, 0.6 * j, 0, 0);
        set(o, B.footL, 0.5 * j, 0, 0); set(o, B.footR, 0.5 * j, 0, 0);
        break;
      }
      case 1: { // one fist pumping up, the other on the knee, bouncing
        const p = Math.max(0, Math.sin(t * 7));
        o[HY] = -0.15 + p * 0.1;
        set(o, B.hips, 0.2, 0, 0); set(o, B.spine, 0.1 - p * 0.15, 0, 0); set(o, B.head, -0.3, 0, 0);
        set(o, B.armR, -0.6 - p * 1.2, 0, -1.2 - p * 0.8); set(o, B.foreR, -1.6 + p * 1.0, 0, 0);
        set(o, B.armL, -0.6, 0, 0.3); set(o, B.foreL, -0.4, 0, 0);
        set(o, B.thighL, -0.4, 0, 0.12); set(o, B.thighR, -0.4, 0, -0.12); set(o, B.shinL, 0.6, 0, 0); set(o, B.shinR, 0.6, 0, 0);
        break;
      }
      case 2: { // hopping from foot to foot, arms windmilling
        const s = Math.sin(t * 8);
        o[HY] = Math.abs(s) * 0.25;
        set(o, B.thighL, -Math.max(0, s) * 0.8, 0, 0.05); set(o, B.shinL, Math.max(0, s) * 1.2, 0, 0);
        set(o, B.thighR, -Math.max(0, -s) * 0.8, 0, -0.05); set(o, B.shinR, Math.max(0, -s) * 1.2, 0, 0);
        set(o, B.armL, -t * 9 % TAU, 0, 0.4); set(o, B.armR, -(t * 9 + Math.PI) % TAU, 0, -0.4);
        set(o, B.head, -0.25, 0, 0);
        break;
      }
      default: { // clapping over the head
        const k = Math.sin(t * 12) * 0.5 + 0.5;
        o[HY] = Math.abs(Math.sin(t * 6)) * 0.12;
        set(o, B.armL, -0.4, 0, 2.6 - k * 0.3); set(o, B.armR, -0.4, 0, -2.6 + k * 0.3);
        set(o, B.foreL, -0.3, 0, 0.4); set(o, B.foreR, -0.3, 0, -0.4);
        set(o, B.head, -0.35, 0, 0); set(o, B.spine, -0.1, 0, 0);
      }
    }
  }

  /** A quick fist pump (fielders after an out). */
  private pump(o: Pose, t: number, variant: number) {
    copyPose(o, STAND);
    const side = variant % 2 === 0 ? 1 : -1; // which arm
    const p = t < 0.25 ? smooth(t / 0.25) : t < 0.55 ? 1 - smooth((t - 0.25) / 0.3) * 0.6 : 0.4 + Math.max(0, Math.sin((t - 0.55) * 9)) * 0.4;
    const arm = side > 0 ? B.armR : B.armL, fore = side > 0 ? B.foreR : B.foreL;
    set(o, arm, -0.5 - p * 0.3, 0, -side * (0.4 + p * 0.5));
    set(o, fore, -1.0 - p * 1.1, 0, 0);
    set(o, B.spine, 0.05 + p * 0.12, 0, side * 0.05);
    set(o, B.hips, 0.1, 0, 0);
    o[HY] = -0.1 * p;
    set(o, B.thighL, -0.2 * p, 0, 0.05); set(o, B.thighR, -0.2 * p, 0, -0.05); set(o, B.shinL, 0.35 * p, 0, 0); set(o, B.shinR, 0.35 * p, 0, 0);
  }

  private clap(o: Pose) {
    const t = this.time + this.seed;
    const k = Math.sin(t * 14) * 0.5 + 0.5;
    copyPose(o, STAND);
    set(o, B.spine, 0.05, 0, 0);
    set(o, B.armL, -0.95, 0, 0.45 - k * 0.3); set(o, B.armR, -0.95, 0, -0.45 + k * 0.3);
    set(o, B.foreL, -1.0, -0.6, 0); set(o, B.foreR, -1.0, 0.6, 0);
    o[HY] = Math.abs(Math.sin(t * 7)) * 0.06;
  }

  /** Disappointment: hands on head / a face-palm / hands on knees, by variant. */
  private groan(o: Pose, t: number, variant: number) {
    copyPose(o, STAND);
    const k = smooth(t / 0.4);
    switch (variant % 3) {
      case 0:
        set(o, B.spine, -0.15 * k, 0, 0); set(o, B.head, -0.3 * k, 0, 0); set(o, B.neck, -0.15 * k, 0, 0);
        this.goal(GOAL_TOP_L, k); this.goal(GOAL_TOP_R, k);
        break;
      case 1:
        set(o, B.spine, 0.2 * k, 0, 0); set(o, B.head, 0.35 * k, -0.2 * k, 0);
        this.goal(GOAL_BROW_R, k);
        break;
      default:
        o[HY] = -0.25 * k;
        set(o, B.hips, 0.6 * k, 0, 0); set(o, B.spine, 0.35 * k, 0, 0); set(o, B.head, 0.1 * k, 0, 0);
        set(o, B.thighL, -0.55 * k, 0, 0.12); set(o, B.thighR, -0.55 * k, 0, -0.12); set(o, B.shinL, 0.7 * k, 0, 0); set(o, B.shinR, 0.7 * k, 0, 0);
        this.goal(GOAL_KNEE_L, k); this.goal(GOAL_KNEE_R, k);
    }
    this.lookMul = 0.3;
  }

  /** A grown-up at the grill: spatula hand flipping, the other on the hip, a sip now and then. */
  private grill(o: Pose) {
    const t = this.time;
    const flip = Math.max(0, Math.sin(t * 1.3)) ** 6;
    copyPose(o, STAND);
    set(o, B.spine, 0.12, 0, 0); set(o, B.neck, 0.15, 0, 0); set(o, B.head, 0.2, 0, 0);
    set(o, B.armR, -0.9 - flip * 0.5, 0, -0.2); set(o, B.foreR, -0.8 + flip * 0.6, flip * 0.8, 0);
    this.goal(GOAL_HIP_L, 1);
    // shifts his weight, hums along
    o[HX] = Math.sin(t * 0.6) * 0.06;
    add(o, B.hips, 0, 0, Math.sin(t * 0.6) * 0.05);
    this.propOut = true;
  }

  /** Fishing a ball out of the pool with the long skimmer (held in both hands). */
  private skim(o: Pose, t: number) {
    copyPose(o, STAND);
    // reach out over the water, sweep, then lift
    const sweep = Math.sin(t * 1.6) * 0.35;
    const lift = smooth((t - 2.6) / 0.8);
    o[HY] = -0.12 + lift * 0.1;
    set(o, B.hips, 0.2 - lift * 0.15, sweep * 0.3, 0);
    set(o, B.spine, 0.25 - lift * 0.2, sweep * 0.4, 0);
    set(o, B.head, 0.25 - lift * 0.3, -sweep * 0.5, 0);
    set(o, B.armR, -0.75 - lift * 0.5, 0, -0.35); set(o, B.foreR, -0.9, 0, 0);
    set(o, B.armL, -1.1 - lift * 0.5, 0, 0.1); set(o, B.foreL, -0.6, 0, 0);
    set(o, B.thighL, -0.35, 0, 0.12); set(o, B.shinL, 0.4, 0, 0); set(o, B.thighR, 0.15, 0, -0.1); set(o, B.shinR, 0.2, 0, 0);
  }

  /** At the plate: this kid's stance and waggle. */
  private stance(o: Pose, P: Personality) {
    const S = P.stance;
    copyPose(o, BAT_STANCE);
    const t = this.time * S.waggleSpeed;
    o[HY] += -S.crouch * 0.3;
    add(o, B.hips, S.crouch * 0.25 + S.lean * 0.2, -S.open * 0.3, 0);
    add(o, B.spine, S.crouch * 0.2 + S.lean * 0.25, 0, 0);
    add(o, B.head, 0, S.open * 0.25, 0);
    add(o, B.thighL, -S.crouch * 0.35, -S.open * 0.2, S.wide * 0.18); add(o, B.thighR, -S.crouch * 0.35, 0, -S.wide * 0.18);
    add(o, B.shinL, S.crouch * 0.6, 0, 0); add(o, B.shinR, S.crouch * 0.6, 0, 0);
    add(o, B.footL, 0, S.open * 0.4, 0);
    // rhythm: weight rocks on the back leg in time with the waggle
    const r = Math.sin(t * 2.4);
    if (S.waggle === 'bob') { o[HY] += r * 0.03; add(o, B.shinL, r * 0.06, 0, 0); add(o, B.shinR, r * 0.06, 0, 0); }
    else if (S.waggle === 'pump') { add(o, B.chest, 0, -Math.max(0, r) * 0.08, 0); }
    else if (S.waggle === 'twitch') { const k = Math.max(0, Math.sin(t * 5)) ** 8; add(o, B.thighL, -k * 0.4, 0, 0); add(o, B.shinL, k * 0.6, 0, 0); }
    add(o, B.chest, 0, Math.sin(t * 3) * 0.04, 0);
  }

  // ───────────────────────────────────────────────── IK

  private goal(g: HandGoal, w: number) {
    if (this.goalN >= this.goalW.length) return;
    this.goalG[this.goalN] = g; this.goalW[this.goalN++] = w;
  }

  /** After update(): put a hand on a world point (e.g. the off hand on a pole). */
  holdWith(side: 'L' | 'R', target: Vector3) {
    this.reachFor(side, target, 1);
  }

  private reachFor(side: 'L' | 'R', target: Vector3, blend: number, poleLocal?: Vector3) {
    const k = this.kid;
    const arm = side === 'L' ? k.bones.armL : k.bones.armR;
    const fore = side === 'L' ? k.bones.foreL : k.bones.foreR;
    const hand = side === 'L' ? k.bones.handL : k.bones.handR;
    // elbows bend down and out
    if (poleLocal) k.bones.chest.localToWorld(_pole.copy(poleLocal));
    else k.bones.chest.localToWorld(_pole.set(side === 'L' ? 1.4 : -1.4, -1.2, -0.4));
    twoBoneIK(arm, fore, hand, target, _pole, blend);
  }

  /** Put a hand on a body landmark (mouth, eyes, hips...) for gestures. */
  private handTo(g: HandGoal, w: number) {
    const k = this.kid;
    const p = k.p, s = p.s, hr = p.headR;
    const sides: ('L' | 'R')[] = g.side === 'B' ? ['L', 'R'] : [g.side];
    for (const side of sides) {
      const m = side === 'L' ? 1 : -1;
      const { bone, x, y, z, pole } = anchor(g.at, p, hr, s);
      const off = g.off;
      _a.set(x * m + (off ? off[0] * m : 0), y + (off ? off[1] : 0), z + (off ? off[2] : 0));
      bone(k).localToWorld(_a);
      _b.set(pole[0] * m, pole[1], pole[2]);
      this.reachFor(side, _a, clamp(w, 0, 1), _b);
      const hand = side === 'L' ? k.bones.handL : k.bones.handR;
      if (!g.aim && (g.at === 'hip' || g.at === 'back' || g.at === 'knee')) {
        // the wrist bends so the palm rests flat: fingers down (and a little forward on the hips)
        k.bones.hips.updateMatrixWorld(true);
        const hq = k.bones.hips.getWorldQuaternion(_q);
        _c.set(-0.25 * m, -1, g.at === 'hip' ? 0.45 : g.at === 'knee' ? 0.6 : -0.2).applyQuaternion(hq);
        pointBone(hand, DOWN_Y, _c, Math.min(1, w) * 0.9);
      }
      if (g.aim && w > 0.05) {
        // hold the prop the right way: toward the face (mic, mug) or level to the eyes (binoculars)
        k.bones.head.updateMatrixWorld(true);
        const hq = k.bones.head.getWorldQuaternion(_q);
        const fwd = _v.set(0, 0, 1).applyQuaternion(hq), up = _w.set(0, 1, 0).applyQuaternion(hq);
        if (g.aim === 'bino') orientBone(hand, up, fwd, Math.min(1, w));
        else if (g.aim === 'mic') { hand.getWorldPosition(_c); k.bones.head.localToWorld(_b.set(0, hr * 0.5, hr * 0.9)); pointBone(hand, UP, _b.sub(_c), Math.min(1, w)); }
        else if (g.aim === 'cup') pointBone(hand, DOWN_Y, _c.copy(up).multiplyScalar(-1).addScaledVector(fwd, -0.9), Math.min(1, w));
        else if (g.aim === 'flat') orientBone(hand, _c.copy(fwd), up.negate(), Math.min(1, w));
      }
    }
  }

  /** Place the bat from the swing/stance and wrap both hands around the handle. */
  private solveBat(inp: AnimInput) {
    const k = this.kid;
    const s = k.p.s;
    const lefty = !!inp.lefty;
    const m = lefty ? -1 : 1;
    const S = this.persona.stance;
    // in the kid's local frame (facing the plate, pitcher toward +x for a righty)
    const handle = _a, dir = _b;
    const sh = k.p.shoulderY;
    if (inp.mode === 'bunt') {
      handle.set(-0.35, sh - 0.35 * s, 0.75 * s);
      dir.set(0.2, 0.12, 1).normalize();
    } else if (inp.mode === 'swing') {
      const tc = inp.power ? 0.18 : 0.15;
      const u = inp.t;
      const aimY = clamp(((inp.aimZ ?? 2.2) - 2.2) * 0.35, -0.5, 0.6);
      const H = this.bh, D = this.bd;
      const T0 = 0, T1 = tc * 0.55, T2 = tc, T3 = tc + 0.12, T4 = tc + 0.45;
      H[0].set(-0.45, sh + (0.05 + S.hands * 0.12) * s, 0.05); D[0].set(-0.6 + S.tilt * 0.3, 0.75, -0.35);
      H[1].set(-0.3, sh - 0.25 * s, 0.35 * s); D[1].set(-0.9, 0.15 + aimY * 0.4, 0.2);
      H[2].set(0.12, sh - 0.6 * s + aimY * 0.3, 0.8 * s); D[2].set(0.2, -0.1 + aimY, 1);
      H[3].set(0.45, sh - 0.35 * s, 0.4 * s); D[3].set(0.95, 0.25, 0.1);
      H[4].set(0.45, sh + 0.05 * s, -0.15); D[4].set(0.1, 0.65, -0.85);
      const T = [T0, T1, T2, T3, T4];
      let i = 0;
      while (i < 3 && u > T[i + 1]) i++;
      const f = i === 1 ? (clamp((u - T[i]) / (T[i + 1] - T[i]), 0, 1)) ** 2 : smooth((u - T[i]) / (T[i + 1] - T[i]));
      handle.copy(H[i]).lerp(H[i + 1], f);
      dir.copy(D[i]).lerp(D[i + 1], f).normalize();
    } else {
      const t = this.time * S.waggleSpeed;
      let wag = Math.sin(t * 3) * 0.06 * S.waggleAmt, wy = 0;
      if (S.waggle === 'circle') { wag = Math.sin(t * 3) * 0.12 * S.waggleAmt; wy = Math.cos(t * 3) * 0.1 * S.waggleAmt; }
      else if (S.waggle === 'still') wag *= 0.25;
      handle.set(-0.42, sh + (0.08 + S.hands * 0.15 - S.crouch * 0.12) * s, 0.05);
      dir.set(-0.55 + wag + S.tilt * 0.5, 0.8, -0.3 + wy).normalize();
    }
    handle.x *= m; dir.x *= m;
    // to world
    k.group.localToWorld(this.batHandle.copy(handle));
    this.batDir.copy(dir).transformDirection(k.group.matrixWorld);
    // bottom hand (glove-side) at the knob end, top hand just above it
    const bottom = lefty ? 'R' : 'L', top = lefty ? 'L' : 'R';
    this.reachFor(bottom, _c.copy(this.batHandle).addScaledVector(this.batDir, 0.05), 1);
    this.reachFor(top, _c.copy(this.batHandle).addScaledVector(this.batDir, 0.32 * s), 1);
    // hands point along the bat
    pointBone(k.bones.handL, _v.set(0, -1, 0), this.batDir, 0.7);
    pointBone(k.bones.handR, _v.set(0, -1, 0), this.batDir, 0.7);
  }

  private look(target: Vector3, mode: Mode) {
    const k = this.kid;
    const head = k.bones.head;
    // turn the head (limited) toward the target, on top of the animated pose
    head.updateMatrixWorld(true);
    const local = head.parent!.worldToLocal(_v.copy(target));
    const hp = head.position;
    const dx = local.x - hp.x, dy = local.y - hp.y - k.p.headR, dz = local.z - hp.z;
    const yaw = Math.atan2(dx, dz), pitch = -Math.atan2(dy, Math.hypot(dx, dz));
    const busy = mode === 'swing' || mode === 'windup' || mode === 'throw' || mode === 'dive' || mode === 'slide' || mode === 'follow';
    const amt = (busy ? 0.35 : 0.85) * this.lookMul;
    const curE = _e.setFromQuaternion(head.quaternion, 'YXZ');
    const ty = clamp(yaw, -1.3, 1.3), tp = clamp(pitch, -0.7, 0.6);
    curE.y = lerp(curE.y, ty, amt);
    curE.x = lerp(curE.x, tp, amt * 0.7);
    head.quaternion.setFromEuler(curE);
    head.updateMatrixWorld(true);
    // eyes do the rest
    for (const eb of [k.bones.eyeL, k.bones.eyeR]) {
      const lp = eb.parent!.worldToLocal(_w.copy(target)).sub(eb.position);
      const ey = clamp(Math.atan2(lp.x, lp.z), -0.55, 0.55), ex = clamp(-Math.atan2(lp.y, Math.hypot(lp.x, lp.z)), -0.4, 0.4);
      eb.quaternion.setFromEuler(_e.set(ex, ey, 0, 'YXZ'));
    }
  }

  private blink(dt: number, expr: Expression) {
    const k = this.kid;
    this.blinkT -= dt;
    if (this.blinkT <= 0 && this.blinkPhase < 0) { this.blinkPhase = 0; }
    let close = 0;
    if (this.blinkPhase >= 0) {
      this.blinkPhase += dt / 0.16;
      close = Math.sin(Math.min(1, this.blinkPhase) * Math.PI);
      if (this.blinkPhase >= 1) { this.blinkPhase = -1; this.blinkT = 1.5 + Math.random() * 3.5; }
    }
    const open = { neutral: -0.62, happy: -0.42, focus: -0.38, surprised: -0.9, sad: -0.3, yell: -0.5, smug: -0.32, oops: -0.7 }[expr];
    const ang = lerp(open, 0.62, close);
    k.bones.lidL.quaternion.setFromEuler(_e.set(ang, 0, 0));
    k.bones.lidR.quaternion.setFromEuler(_e.set(ang, 0, 0));
  }
}

const _mask = new Uint8Array(BONES.length + 1);

// ─────────────────────────────────────────────────────────────── hand anchors

type AnchorPt = { bone: (k: KidModel) => Bone; x: number; y: number; z: number; pole: [number, number, number] };
const headB = (k: KidModel) => k.bones.head, chestB = (k: KidModel) => k.bones.chest, hipsB = (k: KidModel) => k.bones.hips;
const _anc: AnchorPt = { bone: headB, x: 0, y: 0, z: 0, pole: [0, 0, 0] };
const pl = (a: AnchorPt, x: number, y: number, z: number) => { a.pole[0] = x; a.pole[1] = y; a.pole[2] = z; };

/**
 * Where a wrist goes for each landmark, for the LEFT hand (x is mirrored for
 * the right). Head points are in head space (centre ≈ 0.92·headR up), the
 * rest in chest/hip space. The wrist sits a hand's length short of the spot.
 */
function anchor(at: Anchor, p: KidModel['p'], hr: number, s: number): AnchorPt {
  const c = hr * 0.92;
  const a = _anc;
  a.bone = headB;
  switch (at) {
    case 'mouth': a.x = 0.12; a.y = c - hr * 0.85; a.z = hr * 1.25; pl(a, 1.2, -1.6, 0.2); break;
    case 'eyes': a.x = 0.2; a.y = c - hr * 0.45; a.z = hr * 1.35; pl(a, 1.3, -1.6, 0.4); break;
    case 'brow': a.x = 0.1; a.y = c + hr * 0.15; a.z = hr * 1.15; pl(a, 1.3, -1.0, 0.6); break;
    case 'ear': a.x = hr * 1.05; a.y = c - hr * 0.45; a.z = hr * 0.15; pl(a, 1.6, -1.2, -0.3); break;
    case 'cheek': a.x = hr * 0.55; a.y = c - hr * 0.9; a.z = hr * 0.95; pl(a, 1.0, -1.6, 0.6); break;
    case 'top': a.x = hr * 0.35; a.y = c + hr * 0.75; a.z = hr * 0.1; pl(a, 1.8, 0.4, -0.2); break;
    case 'chin': a.x = 0.05; a.y = c - hr * 1.15; a.z = hr * 0.9; pl(a, 1.0, -1.6, 0.3); break;
    case 'neck': a.x = 0.08; a.y = -hr * 0.35; a.z = hr * 0.7; pl(a, 1.2, -1.6, 0.2); break;
    case 'front': a.bone = chestB; a.x = 0.12; a.y = -0.05 * s; a.z = 0.75 * s; pl(a, 1.4, -1.4, -0.2); break;
    case 'belly': a.bone = chestB; a.x = 0.15; a.y = -0.5 * s; a.z = 0.55 * s + p.belly * 0.2; pl(a, 1.4, -1.0, -0.4); break;
    case 'hip': a.bone = hipsB; a.x = p.hipX + 0.2 * p.wf; a.y = 0.5 * s; a.z = 0.0; pl(a, 2.2, 0.3, -0.9); break;
    case 'back': a.bone = hipsB; a.x = 0.18; a.y = 0.25 * s; a.z = -0.45 * s - p.belly * 0.05; pl(a, 1.5, -0.3, -1.2); break;
    case 'knee': a.bone = hipsB; a.x = p.hipX + 0.08; a.y = -0.7 * s; a.z = 0.55 * s; pl(a, 1.5, 0.2, -0.4); break;
    case 'sky': a.bone = chestB; a.x = 0.45; a.y = 1.6 * s; a.z = 0.35; pl(a, 1.6, 0.2, -0.6); break;
  }
  return a;
}

const GOAL_FACE_L: HandGoal = { side: 'L', at: 'cheek' }, GOAL_FACE_R: HandGoal = { side: 'R', at: 'cheek' };
const GOAL_TOP_L: HandGoal = { side: 'L', at: 'top' }, GOAL_TOP_R: HandGoal = { side: 'R', at: 'top' };
const GOAL_BROW_R: HandGoal = { side: 'R', at: 'brow' };
const GOAL_KNEE_L: HandGoal = { side: 'L', at: 'knee' }, GOAL_KNEE_R: HandGoal = { side: 'R', at: 'knee' };
const GOAL_HIP_L: HandGoal = { side: 'L', at: 'hip' };
