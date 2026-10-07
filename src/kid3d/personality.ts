import type { Expression } from './face';
import { Track, type Frame, type PoseDef } from './pose';

// How each kid moves. The game is kids playing at being grown-ups, so every
// kid carries their grown-up job into the way they stand, bat, trot and
// celebrate: Toby blows his whistle, Wren raises her binoculars, Priya taps
// her calculator, Jun sells to a camera nobody else can see, Hank scans the
// yard like it's the food court. Gestures are keyframed by hand (see
// `pose.ts` for the conventions) and only name the bones they move; hands
// that must land on the body (mouth, eyes, hips) use IK anchors.

/** Body landmarks a hand can go to (see `anchor()` in anim.ts). */
export type Anchor = 'mouth' | 'eyes' | 'brow' | 'ear' | 'cheek' | 'top' | 'chin' | 'neck' | 'front' | 'belly' | 'hip' | 'back' | 'knee' | 'sky';

export interface HandGoal {
  side: 'L' | 'R' | 'B';
  at: Anchor;
  /** active between these gesture times (fading in/out), else the whole gesture */
  from?: number; to?: number;
  w?: number;
  /** offset from the anchor (left hand; mirrored for the right) */
  off?: [number, number, number];
  /** hold the prop the right way: a mic or mug toward the mouth, binoculars level, a flat hand */
  aim?: 'mic' | 'bino' | 'cup' | 'flat';
}

export interface Gesture {
  name: string;
  track: Track;
  /** seconds (one-shot length, or loop length) */
  dur: number;
  loop?: boolean;
  hands?: HandGoal[];
  /** show the persona prop (right hand) */
  prop?: boolean;
  expr?: Expression;
  /** how much the head still follows the ball (0 = the gesture owns the head) */
  look?: number;
  mask?: Uint8Array;
  /** false: don't play this fidget sitting on the bench */
  sit?: boolean;
}

export interface Gait {
  /** hip bounce multiplier */
  bounce: number;
  /** arm swing multiplier */
  arms: number;
  /** forward lean multiplier */
  lean: number;
  /** step-rate multiplier (short quick steps > 1, long loping < 1) */
  cadence: number;
}

export interface Stance {
  /** 0 upright .. 1 deep crouch */
  crouch: number;
  /** −1 closed .. 1 open (front foot pulled back, chest toward the pitcher) */
  open: number;
  lean: number;
  /** feet apart, 0..1 */
  wide: number;
  /** hands high (+) or low (−) */
  hands: number;
  /** bat laid back flat (−) or straight up (+) */
  tilt: number;
  waggle: 'bob' | 'circle' | 'twitch' | 'still' | 'pump';
  waggleAmt: number;
  waggleSpeed: number;
}

export interface Idle {
  /** weight-shift speed */
  sway: number;
  /** a one-key track laid over the standing pose (only the bones it names) */
  posture?: Track;
  hands?: HandGoal[];
  prop?: boolean;
}

export interface Personality {
  idle: Idle;
  fidgets: Gesture[];
  stance: Stance;
  gait: Gait;
  /** home-run trot: arm business on top of the trot (loops) */
  trot?: Gesture;
  /** big celebration (loops): home runs, scoring, winning */
  celebrate: Gesture;
  /** after striking out (one-shot, holds the last key) */
  strikeout: Gesture;
  /** after making a catch for an out (one-shot) */
  catchJoy: Gesture;
}

const G = (name: string, dur: number, frames: Frame[], o: Partial<Gesture> = {}): Gesture => ({ name, dur, track: new Track(frames), ...o });
const loop = (name: string, dur: number, frames: Frame[], o: Partial<Gesture> = {}) => G(name, dur, frames, { loop: true, ...o });
const posture = (p: PoseDef) => new Track([[0, p]]);

// ─────────────────────────────────────────────────────────────── shared pieces

const ARMS_DOWN: PoseDef = { armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0] };
const V_UP: PoseDef = { armL: [-0.25, 0, 2.6], armR: [-0.25, 0, -2.6], foreL: [-0.25, 0, 0], foreR: [-0.25, 0, 0] };
const FLEX: PoseDef = { armL: [-0.1, 0, 1.45], foreL: [0, 0, 1.75], armR: [-0.1, 0, -1.45], foreR: [0, 0, -1.75] };
const HOP_UP: PoseDef = { hipY: 0.35, thighL: [-0.35, 0, 0.06], thighR: [-0.25, 0, -0.06], shinL: [0.7, 0, 0], shinR: [0.6, 0, 0], footL: [0.5, 0, 0], footR: [0.5, 0, 0] };
const HOP_DOWN: PoseDef = { hipY: -0.15, thighL: [-0.3, 0, 0.06], thighR: [-0.3, 0, -0.06], shinL: [0.5, 0, 0], shinR: [0.5, 0, 0], footL: [-0.1, 0, 0], footR: [-0.1, 0, 0] };
const LEGS: PoseDef = { hipY: 0, thighL: [0, 0, 0.04], thighR: [0, 0, -0.04], shinL: [0, 0, 0], shinR: [0, 0, 0], footL: [0, 0, 0], footR: [0, 0, 0] };
const SLUMP: PoseDef = { spine: [0.3, 0, 0], chest: [0.15, 0, 0], neck: [0.25, 0, 0], head: [0.4, 0, 0], shoulderL: [0, 0, -0.15], shoulderR: [0, 0, 0.15] };
const TALL: PoseDef = { spine: [-0.08, 0, 0], chest: [-0.06, 0, 0], neck: [-0.05, 0, 0], head: [-0.12, 0, 0] };

/** A generic fist pump into a hop, for kids without a fancier catch move. */
const gloveUp = (name = 'glove up') => G(name, 1.2, [
  [0, { armL: [-0.9, 0, 0.1], foreL: [-0.6, 0, 0] }],
  [0.25, { armL: [-0.3, 0, 2.4], foreL: [-0.2, 0, 0], armR: [-0.5, 0, -0.6], foreR: [-1.4, 0, 0], ...HOP_UP, spine: [-0.1, 0, 0] }, 'out'],
  [0.55, { armL: [-0.3, 0, 2.5], foreL: [-0.2, 0, 0], armR: [-0.4, 0, -0.5], foreR: [-1.3, 0, 0], ...LEGS, spine: [0, 0, 0] }],
  [1.2, { armL: [-0.9, 0, 0.3], foreL: [-0.8, 0, 0], armR: [0, 0, -0.15], foreR: [-0.3, 0, 0] }],
], { expr: 'happy' });

/** Punch the glove (right fist into left glove) a couple of times. */
const gloveSmack = () => G('glove smack', 1.1, [
  [0, { armL: [-0.9, 0, 0.1], foreL: [-1.0, -0.5, 0], armR: [-0.4, 0, -0.3], foreR: [-1.3, 0, 0] }],
  [0.2, { armL: [-0.9, 0, 0.1], foreL: [-1.1, -0.6, 0], armR: [-0.9, 0.2, -0.1], foreR: [-1.2, 0.5, 0], spine: [0.1, 0, 0] }, 'in'],
  [0.35, { armR: [-0.6, 0, -0.4], foreR: [-1.5, 0, 0], spine: [0, 0, 0] }, 'out'],
  [0.5, { armR: [-0.9, 0.2, -0.1], foreR: [-1.2, 0.5, 0], spine: [0.1, 0, 0] }, 'in'],
  [1.1, { armL: [-0.7, 0, 0.15], foreL: [-0.9, 0, 0], armR: [-0.2, 0, -0.2], foreR: [-0.6, 0, 0], spine: [0, 0, 0] }, 'out'],
], { expr: 'smug' });

const stretch = () => G('stretch', 2.6, [
  [0, ARMS_DOWN],
  [0.6, { ...V_UP, armL: [-0.1, 0, 2.9], armR: [-0.1, 0, -2.9], spine: [-0.15, 0, 0], head: [-0.3, 0, 0], hipY: 0.05 }, 'out'],
  [1.4, { armL: [-0.1, 0, 2.7], armR: [-0.1, 0, -2.95], spine: [-0.12, 0, 0.18], head: [-0.25, 0, 0.1], hipY: 0.05 }],
  [2.0, { armL: [-0.1, 0, 2.95], armR: [-0.1, 0, -2.7], spine: [-0.12, 0, -0.18], head: [-0.25, 0, -0.1], hipY: 0.05 }],
  [2.6, { ...ARMS_DOWN, spine: [0, 0, 0], head: [0, 0, 0], hipY: 0 }],
], { look: 0.2, expr: 'happy' });

const kickDirt = () => G('kick dirt', 1.6, [
  [0, { thighR: [0, 0, -0.04], shinR: [0, 0, 0], head: [0, 0, 0] }],
  [0.3, { thighR: [-0.25, 0, -0.04], shinR: [0.5, 0, 0], head: [0.45, 0, 0], spine: [0.1, 0, 0] }],
  [0.5, { thighR: [0.25, 0, -0.04], shinR: [0.2, 0, 0], footR: [0.3, 0, 0] }, 'in'],
  [0.75, { thighR: [-0.2, 0, -0.04], shinR: [0.45, 0, 0], footR: [0, 0, 0] }],
  [0.95, { thighR: [0.2, 0, -0.04], shinR: [0.15, 0, 0] }, 'in'],
  [1.6, { thighR: [0, 0, -0.04], shinR: [0, 0, 0], head: [0.1, 0, 0], spine: [0, 0, 0] }],
], { look: 0.1 });

const scratchHead = () => G('scratch head', 1.8, [
  [0, { head: [0, 0, 0] }],
  [0.4, { head: [-0.05, 0.2, 0.15] }],
  [1.4, { head: [-0.05, 0.25, 0.2] }],
  [1.8, { head: [0, 0, 0] }],
], { hands: [{ side: 'R', at: 'top', from: 0.1, to: 1.6, off: [0.05, 0.05, 0.1] }], look: 0.3 });

const adjustCap = () => G('cap tug', 1.0, [[0, { head: [0, 0, 0] }], [0.35, { head: [0.2, 0, 0] }], [1.0, { head: [0, 0, 0] }]],
  { hands: [{ side: 'R', at: 'brow', from: 0.1, to: 0.85, off: [0, 0.12, 0.05] }], look: 0.4 });

// ─────────────────────────────────────────────────────────────── the kids

const DEFAULT_GAIT: Gait = { bounce: 1, arms: 1, lean: 1, cadence: 1 };
const DEFAULT_STANCE: Stance = { crouch: 0.2, open: 0, lean: 0, wide: 0.3, hands: 0, tilt: 0, waggle: 'bob', waggleAmt: 1, waggleSpeed: 1 };

const PERSONAS: Record<string, Personality> = {
  // ── Bo, the retired plumber: bad back, slow everything, a wrench in his pocket
  bo: {
    idle: { sway: 0.6, posture: posture({ spine: [0.12, 0, 0], neck: [0.05, 0, 0], head: [-0.05, 0, 0] }), hands: [{ side: 'R', at: 'back' }] },
    fidgets: [
      G('back stretch', 2.4, [
        [0, { spine: [0.12, 0, 0], head: [0, 0, 0] }],
        [0.8, { spine: [-0.3, 0, 0], chest: [-0.15, 0, 0], head: [-0.35, 0, 0], hipX: 0.08 }, 'out'],
        [1.6, { spine: [-0.32, 0, 0], chest: [-0.15, 0, 0], head: [-0.3, 0, 0] }],
        [2.4, { spine: [0.12, 0, 0], chest: [0, 0, 0], head: [0, 0, 0], hipX: 0 }],
      ], { hands: [{ side: 'B', at: 'back' }], look: 0.1, expr: 'oops' }),
      G('wrench tap', 2.2, [
        [0, { armR: [0, 0, -0.1], foreR: [-0.3, 0, 0], head: [0, 0, 0] }],
        [0.4, { armR: [-0.5, 0.3, -0.2], foreR: [-1.4, 0, 0], armL: [-0.5, -0.3, 0.2], foreL: [-1.3, 0, 0], head: [0.35, 0, 0] }],
        [0.6, { foreR: [-1.0, 0, 0] }], [0.75, { foreR: [-1.5, 0, 0] }, 'in'], [0.95, { foreR: [-1.0, 0, 0] }], [1.1, { foreR: [-1.5, 0, 0] }, 'in'],
        [1.3, { foreR: [-1.0, 0, 0] }], [1.45, { foreR: [-1.5, 0, 0] }, 'in'],
        [2.2, { armR: [0, 0, -0.1], foreR: [-0.3, 0, 0], armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], head: [0, 0, 0] }],
      ], { prop: true, look: 0.2 }),
      scratchHead(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.05, open: -0.2, wide: 0.5, hands: -0.3, tilt: -0.3, waggle: 'still', waggleAmt: 0.5, waggleSpeed: 0.6 },
    gait: { bounce: 0.5, arms: 0.6, lean: 1.3, cadence: 0.85 },
    trot: loop('hand on back', 1, [[0, { spine: [0.1, 0, 0] }]], { hands: [{ side: 'R', at: 'back' }] }),
    celebrate: loop('raise the roof, slowly', 1.6, [
      [0, { armL: [-0.3, 0, 2.2], armR: [-0.3, 0, -2.2], foreL: [0, 0, 1.2], foreR: [0, 0, -1.2], spine: [0, 0, 0] }],
      [0.8, { armL: [-0.2, 0, 2.6], armR: [-0.2, 0, -2.6], foreL: [0, 0, 0.6], foreR: [0, 0, -0.6], spine: [-0.08, 0, 0], hipY: 0.05 }],
      [1.6, { armL: [-0.3, 0, 2.2], armR: [-0.3, 0, -2.2], foreL: [0, 0, 1.2], foreR: [0, 0, -1.2], spine: [0, 0, 0], hipY: 0 }],
    ], { expr: 'happy' }),
    strikeout: G('oh my back', 2.0, [
      [0, { spine: [0, 0, 0] }],
      [0.5, { spine: [0.35, 0, 0.1], head: [0.1, 0, 0], hipY: -0.1, thighL: [-0.2, 0, 0.05], shinL: [0.3, 0, 0] }, 'out'],
      [2.0, { spine: [0.3, 0, 0.08], head: [0.25, 0.2, 0], hipY: -0.08, thighL: [-0.15, 0, 0.05], shinL: [0.25, 0, 0] }],
    ], { hands: [{ side: 'B', at: 'back', from: 0.2 }], expr: 'oops', look: 0.2 }),
    catchJoy: G('slow nod', 1.4, [[0, { head: [0, 0, 0] }], [0.4, { head: [0.3, 0, 0] }], [0.8, { head: [-0.1, 0, 0] }], [1.4, { head: [0, 0, 0] }]], { expr: 'smug' }),
  },

  // ── Ines, the soap-opera star: everything is a season finale
  ines: {
    idle: { sway: 0.8, posture: posture({ ...TALL, hips: [0, 0.15, 0], head: [-0.15, -0.2, 0.08] }), hands: [{ side: 'L', at: 'hip' }] },
    fidgets: [
      G('hair flip', 1.4, [
        [0, { head: [-0.15, -0.2, 0.08] }],
        [0.35, { head: [-0.05, 0.25, -0.2] }],
        [0.6, { head: [-0.35, -0.3, 0.2] }, 'in'],
        [1.4, { head: [-0.15, -0.2, 0.08] }],
      ], { hands: [{ side: 'R', at: 'ear', from: 0.05, to: 0.75, off: [0.05, 0.2, -0.1] }], look: 0, expr: 'smug' }),
      G('swoon', 2.2, [
        [0, { spine: [0, 0, 0], head: [0, 0, 0] }],
        [0.5, { spine: [-0.3, 0, 0.12], chest: [-0.12, 0, 0], head: [-0.45, 0.3, 0.2], hipY: -0.05, armL: [-0.2, 0, 1.2], foreL: [0, 0, 0.6] }, 'out'],
        [1.6, { spine: [-0.32, 0, 0.14], chest: [-0.12, 0, 0], head: [-0.5, 0.3, 0.25], hipY: -0.05, armL: [-0.2, 0, 1.3], foreL: [0, 0, 0.7] }],
        [2.2, { spine: [0, 0, 0], chest: [0, 0, 0], head: [0, 0, 0], hipY: 0, armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0] }],
      ], { hands: [{ side: 'R', at: 'brow', from: 0.2, to: 1.9, off: [0.05, 0.05, 0.05] }], look: 0, expr: 'surprised' }),
      G('to the mic', 2.4, [[0, { head: [0, 0, 0] }], [0.5, { head: [-0.1, 0.1, 0], spine: [-0.05, 0, 0] }], [2.4, { head: [0, 0, 0], spine: [0, 0, 0] }]],
        { prop: true, hands: [{ side: 'R', at: 'mouth', from: 0.2, to: 2.2, aim: 'mic' }], expr: 'yell', look: 0.4 }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.1, open: 0.3, wide: 0.2, hands: 0.4, tilt: 0.3, waggle: 'circle', waggleAmt: 0.8, waggleSpeed: 0.8 },
    gait: { bounce: 0.8, arms: 0.7, lean: 0.6, cadence: 1.05 },
    trot: loop('blowing kisses', 1.6, [
      [0, { armR: [-0.4, 0, -0.3], foreR: [-1.8, 0, 0], head: [-0.15, 0, 0] }],
      [0.5, { armR: [-0.4, 0, -0.3], foreR: [-1.8, 0, 0] }],
      [0.8, { armR: [-1.6, 0, -1.2], foreR: [-0.1, 0, 0] }, 'out'],
      [1.6, { armR: [-0.4, 0, -0.3], foreR: [-1.8, 0, 0] }],
    ], { hands: [{ side: 'L', at: 'hip' }, { side: 'R', at: 'mouth', from: 0.1, to: 0.65 }], expr: 'smug' }),
    celebrate: loop('curtain call', 2.6, [
      [0, { ...ARMS_DOWN, spine: [0, 0, 0], head: [0, 0, 0] }],
      [0.6, { armL: [-0.4, 0, 1.6], armR: [-0.4, 0, -1.6], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0], spine: [-0.15, 0, 0], head: [-0.35, 0, 0] }, 'out'],
      [1.3, { armL: [0.3, 0, 0.9], armR: [-1.5, 0, -0.6], foreL: [-0.2, 0, 0], foreR: [-0.4, 0, 0], spine: [0.75, 0, 0], head: [0.3, 0, 0], thighR: [0.3, 0, -0.05], shinR: [0.6, 0, 0] }],
      [2.0, { armL: [0.3, 0, 0.9], armR: [-1.5, 0, -0.6], spine: [0.7, 0, 0], head: [0.25, 0, 0], thighR: [0.3, 0, -0.05], shinR: [0.6, 0, 0] }],
      [2.6, { ...ARMS_DOWN, spine: [0, 0, 0], head: [0, 0, 0], thighR: [0, 0, -0.04], shinR: [0, 0, 0] }],
    ], { expr: 'happy' }),
    strikeout: G('dramatic faint', 2.2, [
      [0, { spine: [0, 0, 0], head: [0, 0, 0], hipY: 0 }],
      [0.4, { spine: [-0.25, 0.1, 0.1], head: [-0.4, 0.2, 0.2], armL: [-0.3, 0, 1.5], foreL: [0, 0, 0.5] }, 'out'],
      [1.2, { spine: [-0.35, 0.1, 0.2], chest: [-0.1, 0, 0], head: [-0.55, 0.3, 0.3], armL: [-0.3, 0, 1.8], foreL: [0, 0, 0.5], hipY: -0.35, thighL: [-0.6, 0, 0.1], shinL: [1.1, 0, 0], thighR: [-0.5, 0, -0.1], shinR: [1.0, 0, 0] }],
      [2.2, { spine: [-0.35, 0.1, 0.2], chest: [-0.1, 0, 0], head: [-0.55, 0.35, 0.3], armL: [-0.3, 0, 1.9], foreL: [0, 0, 0.5], hipY: -0.4, thighL: [-0.65, 0, 0.1], shinL: [1.2, 0, 0], thighR: [-0.55, 0, -0.1], shinR: [1.1, 0, 0] }],
    ], { hands: [{ side: 'R', at: 'brow', from: 0.15 }], look: 0, expr: 'surprised' }),
    catchJoy: G('pose for the camera', 1.4, [
      [0, { head: [0, 0, 0] }],
      [0.35, { hips: [0, 0.3, 0], head: [-0.2, -0.3, 0.15], armL: [-0.3, 0, 2.3], foreL: [-0.2, 0, 0] }, 'out'],
      [1.4, { hips: [0, 0.3, 0], head: [-0.2, -0.3, 0.15], armL: [-0.3, 0, 2.2], foreL: [-0.2, 0, 0] }],
    ], { hands: [{ side: 'R', at: 'hip' }], expr: 'smug' }),
  },

  // ── Toby, the gym teacher: whistle, clipboard, hands on hips
  toby: {
    idle: { sway: 1.2, posture: posture({ ...TALL }), hands: [{ side: 'B', at: 'hip' }] },
    fidgets: [
      G('whistle', 1.8, [
        [0, { spine: [0, 0, 0], head: [0, 0, 0] }],
        [0.3, { spine: [-0.1, 0, 0], head: [-0.2, 0, 0] }],
        [0.5, { spine: [0.05, 0, 0], head: [-0.1, 0, 0] }, 'in'],
        [0.75, { spine: [-0.1, 0, 0], head: [-0.2, 0, 0] }], [0.95, { spine: [0.05, 0, 0], head: [-0.1, 0, 0] }, 'in'],
        [1.8, { spine: [0, 0, 0], head: [0, 0, 0] }],
      ], { hands: [{ side: 'R', at: 'mouth', from: 0.1, to: 1.4 }, { side: 'L', at: 'hip' }], expr: 'yell', look: 0.3 }),
      G('laps!', 2.0, [
        [0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], head: [0, 0, 0] }],
        [0.35, { armR: [-1.5, -0.6, -0.2], foreR: [-0.05, 0, 0], head: [-0.1, -0.5, 0], spine: [0, -0.2, 0] }, 'out'],
        [0.9, { armR: [-1.5, 0.2, -0.3], foreR: [-0.05, 0, 0], head: [-0.1, 0.2, 0], spine: [0, 0.15, 0] }],
        [1.4, { armR: [-1.5, -0.3, -0.25], foreR: [-0.05, 0, 0], head: [-0.1, -0.3, 0], spine: [0, -0.1, 0] }],
        [2.0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], head: [0, 0, 0], spine: [0, 0, 0] }],
      ], { hands: [{ side: 'L', at: 'hip' }], expr: 'yell', look: 0 }),
      G('clipboard', 2.6, [
        [0, { head: [0, 0, 0] }],
        [0.5, { head: [0.45, 0.1, 0], spine: [0.08, 0, 0] }],
        [1.4, { head: [0.45, -0.05, 0] }], [1.8, { head: [0.1, 0, 0] }], [2.1, { head: [0.45, 0, 0] }],
        [2.6, { head: [0, 0, 0], spine: [0, 0, 0] }],
      ], { prop: true, hands: [{ side: 'R', at: 'front', from: 0.1, to: 2.4, off: [0.05, -0.1, 0.0], aim: 'flat' }], look: 0 }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.4, wide: 0.4, waggle: 'bob', waggleAmt: 1.2, waggleSpeed: 1.5 },
    gait: { bounce: 1.3, arms: 1.2, lean: 1.1, cadence: 1.12 },
    trot: loop('whistling round the bases', 1.2, [[0, { head: [-0.2, 0, 0] }], [0.6, { head: [-0.1, 0, 0] }], [1.2, { head: [-0.2, 0, 0] }]],
      { hands: [{ side: 'R', at: 'mouth' }], expr: 'yell' }),
    celebrate: loop('hustle windmill', 1.0, [
      // one full turn of the arm (−3.0 and −9.28 are the same angle, so the loop is seamless)
      [0, { armR: [-3.0, 0, -0.3], armL: [-0.2, 0, 0.3], spine: [0, 0, 0], hipY: 0 }],
      [0.25, { armR: [-6.14, 0, -0.3], spine: [0.05, 0, 0], hipY: 0.15 }, 'lin'],
      [0.5, { armR: [-9.28, 0, -0.3], hipY: 0 }, 'lin'],
      [1.0, { armR: [-9.28, 0, -0.3], spine: [0, 0, 0] }],
    ], { hands: [{ side: 'L', at: 'mouth' }], expr: 'yell' }),
    strikeout: G('whistle on himself', 1.8, [
      [0, { head: [0, 0, 0] }],
      [0.5, { head: [0.25, 0, 0], spine: [0.1, 0, 0] }],
      [0.9, { armR: [-0.9, 0.6, 0], foreR: [-1.6, 0, 0], head: [0.3, 0, 0] }],
      [1.8, { armR: [-0.9, 0.6, 0], foreR: [-1.7, 0, 0], head: [0.35, 0.1, 0], spine: [0.12, 0, 0] }],
    ], { hands: [{ side: 'L', at: 'mouth', from: 0.1, to: 0.8 }], expr: 'sad', look: 0 }),
    catchJoy: G('point and whistle', 1.4, [
      [0, { armR: [0, 0, -0.1] }],
      [0.3, { armR: [-1.6, 0, -0.3], foreR: [-0.05, 0, 0], head: [-0.1, 0, 0] }, 'out'],
      [1.4, { armR: [-1.5, 0, -0.3], foreR: [-0.1, 0, 0], head: [-0.1, 0, 0] }],
    ], { expr: 'yell' }),
  },

  // ── Wren, the bird watcher: binoculars up, little flappy hops
  wren: {
    idle: { sway: 1.1, posture: posture({ head: [-0.1, 0, 0.05], armL: [-0.1, 0.2, 0.08], armR: [-0.1, -0.2, -0.08], foreL: [-0.9, -0.3, 0], foreR: [-0.9, 0.3, 0] }) },
    fidgets: [
      G('binoculars', 3.4, [
        [0, { head: [0, 0, 0], spine: [0, 0, 0] }],
        [0.5, { head: [-0.45, 0.25, 0], spine: [-0.1, 0.1, 0] }],
        [1.6, { head: [-0.5, -0.25, 0], spine: [-0.1, -0.1, 0] }],
        [2.4, { head: [-0.4, 0.05, 0], spine: [-0.08, 0, 0] }],
        [3.4, { head: [0, 0, 0], spine: [0, 0, 0] }],
      ], { prop: true, hands: [{ side: 'R', at: 'eyes', from: 0.15, to: 3.1, aim: 'bino' }, { side: 'L', at: 'eyes', from: 0.2, to: 3.05 }], look: 0, expr: 'surprised' }),
      G('flap', 1.5, [
        [0, { armL: [0, 0, 0.3], armR: [0, 0, -0.3], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0] }],
        [0.18, { armL: [0, 0, 1.3], armR: [0, 0, -1.3], hipY: 0.1 }, 'out'], [0.36, { armL: [0, 0, 0.5], armR: [0, 0, -0.5], hipY: 0 }],
        [0.54, { armL: [0, 0, 1.3], armR: [0, 0, -1.3], hipY: 0.1 }, 'out'], [0.72, { armL: [0, 0, 0.5], armR: [0, 0, -0.5], hipY: 0 }],
        [0.9, { armL: [0, 0, 1.3], armR: [0, 0, -1.3], hipY: 0.1 }, 'out'],
        [1.5, { armL: [0, 0, 0.3], armR: [0, 0, -0.3], hipY: 0 }],
      ], { expr: 'happy' }),
      G('notes it down', 2.2, [[0, { head: [0, 0, 0] }], [0.4, { head: [0.5, 0, 0] }], [1.8, { head: [0.5, 0.1, 0] }], [2.2, { head: [0, 0, 0] }]],
        { hands: [{ side: 'L', at: 'front', from: 0.1, to: 2.0, off: [-0.05, -0.15, 0], aim: 'flat' }, { side: 'R', at: 'front', from: 0.3, to: 1.9, off: [0.02, -0.05, 0.05] }], look: 0 }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.55, open: 0.2, wide: 0.2, hands: -0.2, waggle: 'twitch', waggleAmt: 0.8, waggleSpeed: 1.3 },
    gait: { bounce: 1.5, arms: 0.9, lean: 0.8, cadence: 1.2 },
    trot: loop('flapping round', 0.5, [
      [0, { armL: [0, 0, 1.2], armR: [0, 0, -1.2], foreL: [0, 0, -0.2], foreR: [0, 0, 0.2] }],
      [0.25, { armL: [0, 0, 0.6], armR: [0, 0, -0.6] }],
      [0.5, { armL: [0, 0, 1.2], armR: [0, 0, -1.2] }],
    ], { expr: 'happy' }),
    celebrate: loop('bird hops', 0.8, [
      [0, { armL: [0, 0, 0.5], armR: [0, 0, -0.5], ...HOP_DOWN }],
      [0.2, { armL: [0, 0, 1.5], armR: [0, 0, -1.5], ...HOP_UP }, 'out'],
      [0.4, { armL: [0, 0, 0.5], armR: [0, 0, -0.5], ...HOP_DOWN }, 'in'],
      [0.6, { armL: [0, 0, 1.4], armR: [0, 0, -1.4], ...LEGS }],
      [0.8, { armL: [0, 0, 0.5], armR: [0, 0, -0.5], ...HOP_DOWN }],
    ], { expr: 'yell' }),
    strikeout: G('shrinking', 1.6, [
      [0, { spine: [0, 0, 0] }],
      [0.6, { ...SLUMP, armL: [-0.4, 0.4, 0.1], armR: [-0.4, -0.4, -0.1], foreL: [-1.2, -0.4, 0], foreR: [-1.2, 0.4, 0], hipY: -0.05 }, 'out'],
      [1.6, { ...SLUMP, head: [0.5, 0.1, 0], armL: [-0.4, 0.4, 0.1], armR: [-0.4, -0.4, -0.1], foreL: [-1.2, -0.4, 0], foreR: [-1.2, 0.4, 0], hipY: -0.05 }],
    ], { expr: 'sad', look: 0 }),
    catchJoy: G('cupped like a bird', 1.4, [
      [0, { head: [0, 0, 0] }],
      [0.4, { armL: [-1.2, -0.3, 0.1], foreL: [-1.0, 0, 0], head: [0.4, 0, 0], spine: [0.1, 0, 0] }, 'out'],
      [1.4, { armL: [-1.1, -0.3, 0.1], foreL: [-1.0, 0, 0], head: [0.35, 0, 0], spine: [0.1, 0, 0] }],
    ], { hands: [{ side: 'R', at: 'front', off: [0.1, -0.05, 0.1] }], expr: 'happy', look: 0 }),
  },

  // ── Dez, the late-night jazz DJ: smooth, slow, always grooving
  dez: {
    idle: { sway: 1.4, posture: posture({ hips: [0, -0.15, 0], spine: [-0.06, 0.1, 0], head: [0.05, 0, -0.08] }), hands: [{ side: 'L', at: 'hip' }] },
    fidgets: [
      G('snaps', 2.4, [
        [0, { armR: [-0.3, 0, -0.3], foreR: [-1.3, 0, 0], head: [0, 0, 0], hipX: 0 }],
        [0.6, { foreR: [-1.5, 0, 0], head: [0.15, 0, 0.1], hipX: 0.08 }, 'in'],
        [1.2, { foreR: [-1.3, 0, 0], head: [-0.05, 0, -0.1], hipX: -0.08 }],
        [1.8, { foreR: [-1.5, 0, 0], head: [0.15, 0, 0.1], hipX: 0.08 }, 'in'],
        [2.4, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], head: [0, 0, 0], hipX: 0 }],
      ], { expr: 'smug', look: 0.4 }),
      G('on the air', 2.6, [[0, { spine: [0, 0, 0], head: [0, 0, 0] }], [0.6, { spine: [0.15, 0, 0], head: [0.1, 0, 0] }], [2.6, { spine: [0, 0, 0], head: [0, 0, 0] }]],
        { prop: true, hands: [{ side: 'R', at: 'mouth', from: 0.2, to: 2.3, aim: 'mic' }], expr: 'smug', look: 0.3 }),
      adjustCap(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.15, open: -0.3, wide: 0.35, hands: 0.1, tilt: -0.1, waggle: 'still', waggleAmt: 0.6, waggleSpeed: 0.6 },
    gait: { bounce: 0.6, arms: 0.7, lean: 0.4, cadence: 0.9 },
    trot: loop('point to the sky', 2, [
      [0, { armR: [-0.3, 0, -2.8], foreR: [-0.1, 0, 0], armL: [-0.3, 0, 2.8], foreL: [-0.1, 0, 0] }],
      [1, { armR: [-0.3, 0, -2.6], foreR: [-0.1, 0, 0], armL: [-0.3, 0, 2.6], foreL: [-0.1, 0, 0], head: [-0.2, 0, 0] }],
      [2, { armR: [-0.3, 0, -2.8], foreR: [-0.1, 0, 0], armL: [-0.3, 0, 2.8], foreL: [-0.1, 0, 0], head: [0, 0, 0] }],
    ], { expr: 'smug' }),
    celebrate: loop('smooth groove', 1.6, [
      [0, { hipX: 0.12, hips: [0, -0.2, 0.06], spine: [-0.1, 0.15, -0.08], armR: [-1.4, -0.3, -0.4], foreR: [-0.15, 0, 0], armL: [-0.2, 0, 0.6], foreL: [-1.0, 0, 0], head: [0.1, 0.1, -0.1] }],
      [0.8, { hipX: -0.12, hips: [0, 0.2, -0.06], spine: [-0.1, -0.15, 0.08], armR: [-1.3, 0.2, -0.3], foreR: [-0.15, 0, 0], armL: [-0.3, 0, 0.5], foreL: [-1.1, 0, 0], head: [0.1, -0.1, 0.1] }],
      [1.6, { hipX: 0.12, hips: [0, -0.2, 0.06], spine: [-0.1, 0.15, -0.08], armR: [-1.4, -0.3, -0.4], foreR: [-0.15, 0, 0], armL: [-0.2, 0, 0.6], foreL: [-1.0, 0, 0], head: [0.1, 0.1, -0.1] }],
    ], { expr: 'smug' }),
    strikeout: G('cool head shake', 1.8, [
      [0, { head: [0, 0, 0] }], [0.4, { head: [0.2, 0.3, 0] }], [0.7, { head: [0.2, -0.3, 0] }], [1.0, { head: [0.2, 0.25, 0] }], [1.8, { head: [0.25, 0, 0], spine: [0.05, 0, 0] }],
    ], { hands: [{ side: 'B', at: 'hip', from: 0.3 }], expr: 'smug', look: 0 }),
    catchJoy: G('finger guns', 1.2, [
      [0, { armR: [0, 0, -0.1] }],
      [0.3, { armR: [-1.5, 0, -0.4], foreR: [-0.2, 0, 0], armL: [-0.3, 0, 0.3], head: [0, 0, -0.1] }, 'out'],
      [0.45, { armR: [-1.8, 0, -0.4] }, 'out'], [1.2, { armR: [-1.5, 0, -0.4], foreR: [-0.2, 0, 0] }],
    ], { expr: 'smug' }),
  },

  // ── Priya, the CPA: calculator, ledger, glasses up the nose
  priya: {
    idle: { sway: 0.9, posture: posture({ ...TALL, head: [0.05, 0, 0] }) },
    fidgets: [
      G('calculator', 3.0, [
        [0, { head: [0, 0, 0] }], [0.4, { head: [0.5, 0, 0], spine: [0.06, 0, 0] }], [2.6, { head: [0.5, 0.05, 0] }], [3.0, { head: [0, 0, 0], spine: [0, 0, 0] }],
      ], {
        prop: true, look: 0,
        hands: [
          { side: 'R', at: 'front', from: 0.1, to: 2.8, off: [-0.05, -0.1, 0.0], aim: 'flat' },
          { side: 'L', at: 'front', from: 0.4, to: 0.7, off: [-0.08, 0.05, 0.08] }, { side: 'L', at: 'front', from: 0.75, to: 1.0, off: [-0.14, -0.02, 0.06] },
          { side: 'L', at: 'front', from: 1.05, to: 1.35, off: [-0.1, 0.06, 0.08] }, { side: 'L', at: 'front', from: 1.45, to: 1.8, off: [-0.16, 0.02, 0.06] },
          { side: 'L', at: 'front', from: 1.9, to: 2.4, off: [-0.1, 0.04, 0.08] },
        ],
      }),
      G('glasses push', 1.0, [[0, { head: [0, 0, 0] }], [0.3, { head: [-0.1, 0, 0] }], [1.0, { head: [0, 0, 0] }]],
        { hands: [{ side: 'L', at: 'eyes', from: 0.05, to: 0.75, off: [-0.18, 0.1, -0.05] }], expr: 'smug', look: 0.3 }),
      G('checks the numbers', 2.0, [[0, { head: [0, 0, 0] }], [0.4, { head: [0.1, 0.5, 0] }], [1.2, { head: [0.1, -0.4, 0] }], [2.0, { head: [0, 0, 0] }]],
        { hands: [{ side: 'L', at: 'chin', from: 0.1, to: 1.8 }], look: 0 }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.3, open: 0, wide: 0.25, hands: 0.2, tilt: 0.4, waggle: 'still', waggleAmt: 0.4, waggleSpeed: 1 },
    gait: { bounce: 0.7, arms: 0.75, lean: 0.9, cadence: 1.15 },
    trot: loop('brisk and tidy', 1.4, [[0, { head: [-0.05, 0, 0] }], [0.7, { head: [0, 0, 0] }], [1.4, { head: [-0.05, 0, 0] }]],
      { hands: [{ side: 'L', at: 'eyes', from: 0.1, to: 0.6, off: [-0.18, 0.1, -0.05] }], expr: 'smug' }),
    celebrate: loop('balanced books', 1.4, [
      [0, { armR: [-0.4, 0, -0.4], foreR: [-1.8, 0, 0], ...LEGS, head: [0, 0, 0] }],
      [0.25, { armR: [-0.7, 0, -0.5], foreR: [-2.2, 0, 0], ...HOP_UP, hipY: 0.15, head: [-0.2, 0, 0] }, 'out'],
      [0.5, { armR: [-0.4, 0, -0.4], foreR: [-1.8, 0, 0], ...LEGS }, 'in'],
      [1.4, { armR: [-0.4, 0, -0.4], foreR: [-1.8, 0, 0], head: [0.1, 0.1, 0] }],
    ], { hands: [{ side: 'L', at: 'eyes', from: 0.6, to: 1.3, off: [-0.18, 0.1, -0.05] }], expr: 'happy' }),
    strikeout: G('writes it off', 2.0, [
      [0, { head: [0, 0, 0] }], [0.5, { head: [0.5, 0, 0], spine: [0.08, 0, 0] }], [1.4, { head: [0.5, 0.25, 0] }], [2.0, { head: [0.45, -0.2, 0] }],
    ], { hands: [{ side: 'R', at: 'front', from: 0.2, off: [-0.05, -0.1, 0], aim: 'flat' }, { side: 'L', at: 'front', from: 0.4, off: [-0.12, 0.02, 0.06] }], prop: true, expr: 'sad', look: 0 }),
    catchJoy: G('noted', 1.2, [[0, { head: [0, 0, 0] }], [0.3, { head: [0.25, 0, 0] }], [1.2, { head: [0, 0, 0] }]],
      { hands: [{ side: 'R', at: 'eyes', from: 0.2, to: 0.9, off: [-0.18, 0.1, -0.05] }], expr: 'smug' }),
  },

  // ── Gus, the lumberjack: wide stance, arms crossed, chops everything
  gus: {
    idle: { sway: 0.6, posture: posture({ thighL: [0, 0, 0.12], thighR: [0, 0, -0.12], ...TALL }), hands: [{ side: 'L', at: 'front', off: [-0.3, -0.1, -0.12] }, { side: 'R', at: 'front', off: [-0.3, 0.02, -0.08] }] },
    fidgets: [
      G('chop', 2.0, [
        [0, { armL: [-0.6, 0, 0.1], armR: [-0.6, 0, -0.1], foreL: [-0.8, 0, 0], foreR: [-0.8, 0, 0], spine: [0, 0, 0] }],
        [0.5, { armL: [-3.0, -0.3, 0.1], armR: [-3.0, 0.3, -0.1], foreL: [-0.6, 0, 0], foreR: [-0.6, 0, 0], spine: [-0.25, 0, 0], head: [-0.2, 0, 0] }, 'out'],
        [0.68, { armL: [-0.9, -0.3, 0], armR: [-0.9, 0.3, 0], foreL: [-0.1, 0, 0], foreR: [-0.1, 0, 0], spine: [0.55, 0, 0], head: [0.2, 0, 0], hipY: -0.12 }, 'in'],
        [1.2, { armL: [-0.9, -0.3, 0], armR: [-0.9, 0.3, 0], foreL: [-0.1, 0, 0], foreR: [-0.1, 0, 0], spine: [0.5, 0, 0], head: [0.25, 0, 0], hipY: -0.12 }],
        [2.0, { armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0], spine: [0, 0, 0], head: [0, 0, 0], hipY: 0 }],
      ], { expr: 'yell', look: 0.2 }),
      G('flex', 1.8, [[0, ARMS_DOWN], [0.4, { ...FLEX, spine: [-0.1, 0, 0] }, 'out'], [1.3, { ...FLEX, spine: [-0.1, 0, 0] }], [1.8, { ...ARMS_DOWN, spine: [0, 0, 0] }]], { expr: 'smug' }),
      stretch(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.15, open: -0.1, wide: 0.7, hands: 0.3, tilt: 0.2, waggle: 'pump', waggleAmt: 1.2, waggleSpeed: 0.8 },
    gait: { bounce: 0.9, arms: 1.25, lean: 0.9, cadence: 0.9 },
    trot: loop('flexing round', 1.2, [[0, FLEX], [0.6, { ...FLEX, armL: [-0.1, 0, 1.6], armR: [-0.1, 0, -1.6] }], [1.2, FLEX]], { expr: 'yell' }),
    celebrate: loop('timber!', 1.4, [
      [0, { armL: [-3.0, -0.3, 0.1], armR: [-3.0, 0.3, -0.1], foreL: [-0.6, 0, 0], foreR: [-0.6, 0, 0], spine: [-0.25, 0, 0], head: [-0.25, 0, 0], hipY: 0 }],
      [0.5, { armL: [-3.0, -0.3, 0.1], armR: [-3.0, 0.3, -0.1], spine: [-0.28, 0, 0], hipY: 0.05 }],
      [0.68, { armL: [-0.9, -0.3, 0], armR: [-0.9, 0.3, 0], foreL: [-0.1, 0, 0], foreR: [-0.1, 0, 0], spine: [0.55, 0, 0], head: [0.1, 0, 0], hipY: -0.15 }, 'in'],
      [1.0, { ...FLEX, spine: [-0.15, 0, 0], head: [-0.3, 0, 0], hipY: 0 }, 'out'],
      [1.4, { armL: [-3.0, -0.3, 0.1], armR: [-3.0, 0.3, -0.1], foreL: [-0.6, 0, 0], foreR: [-0.6, 0, 0], spine: [-0.25, 0, 0], head: [-0.25, 0, 0] }],
    ], { expr: 'yell' }),
    strikeout: G('stomp', 1.6, [
      [0, { thighR: [0, 0, -0.04], shinR: [0, 0, 0] }],
      [0.3, { thighR: [-0.9, 0, -0.04], shinR: [1.2, 0, 0], spine: [-0.1, 0, 0], armL: [-0.3, 0, 0.5], armR: [-0.3, 0, -0.5] }, 'out'],
      [0.42, { thighR: [0, 0, -0.04], shinR: [0, 0, 0], spine: [0.2, 0, 0], hipY: -0.1 }, 'in'],
      [1.6, { ...SLUMP, hipY: -0.05 }],
    ], { hands: [{ side: 'B', at: 'knee', from: 0.6 }], expr: 'yell', look: 0 }),
    catchJoy: gloveSmack(),
  },

  // ── Molly, the deli counter lady: pencil behind the ear, order up!
  molly: {
    idle: { sway: 1.3, posture: posture({ spine: [0.06, 0, 0], head: [0, 0.1, 0.1] }), hands: [{ side: 'B', at: 'belly', off: [-0.1, 0, 0.05] }] },
    fidgets: [
      G('takes an order', 2.8, [
        [0, { head: [0, 0, 0] }], [0.5, { head: [0.1, 0.2, 0.15] }], [0.9, { head: [0.45, 0, 0] }], [2.4, { head: [0.45, 0.1, 0] }], [2.8, { head: [0, 0, 0] }],
      ], {
        look: 0,
        hands: [{ side: 'R', at: 'ear', from: 0.05, to: 0.8 }, { side: 'R', at: 'front', from: 0.8, to: 2.6, off: [-0.02, 0.0, 0.08] }, { side: 'L', at: 'front', from: 0.6, to: 2.6, off: [-0.05, -0.1, 0], aim: 'flat' }],
      }),
      G('order up!', 1.8, [
        [0, { spine: [0, 0, 0], head: [0, 0, 0] }], [0.3, { spine: [0.15, 0, 0], head: [-0.25, 0, 0] }], [1.4, { spine: [0.15, 0, 0], head: [-0.25, 0, 0] }], [1.8, { spine: [0, 0, 0], head: [0, 0, 0] }],
      ], { hands: [{ side: 'R', at: 'mouth', from: 0.1, to: 1.5, off: [-0.1, 0.05, 0.05] }], expr: 'yell', look: 0.3 }),
      kickDirt(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.35, open: 0.1, wide: 0.3, waggle: 'bob', waggleAmt: 1.4, waggleSpeed: 1.6 },
    gait: { bounce: 1.4, arms: 1, lean: 0.9, cadence: 1.15 },
    trot: loop('ta-da skip', 1.0, [[0, { armL: [-0.3, 0, 1.2], armR: [-0.3, 0, -1.2], foreL: [0, 0, 0.4], foreR: [0, 0, -0.4] }], [0.5, { armL: [-0.3, 0, 1.5], armR: [-0.3, 0, -1.5] }], [1.0, { armL: [-0.3, 0, 1.2], armR: [-0.3, 0, -1.2] }]], { expr: 'happy' }),
    celebrate: loop('ding ding ding', 0.9, [
      [0, { armR: [-1.1, 0, -0.2], foreR: [-0.9, 0, 0], ...LEGS, spine: [0.05, 0, 0] }],
      [0.15, { armR: [-1.1, 0, -0.2], foreR: [-0.3, 0, 0], ...HOP_DOWN }, 'in'],
      [0.3, { armR: [-1.1, 0, -0.2], foreR: [-0.9, 0, 0], ...HOP_UP }, 'out'],
      [0.45, { armR: [-1.1, 0, -0.2], foreR: [-0.3, 0, 0], ...HOP_DOWN }, 'in'],
      [0.9, { armR: [-1.1, 0, -0.2], foreR: [-0.9, 0, 0], ...LEGS }],
    ], { hands: [{ side: 'L', at: 'mouth', off: [-0.1, 0.05, 0.05] }], expr: 'yell' }),
    strikeout: G('wipes hands on apron', 1.8, [[0, { head: [0, 0, 0] }], [0.5, { head: [0.3, 0, 0] }], [1.8, { head: [0.2, 0.3, 0.1], spine: [0.05, 0, 0] }]],
      { hands: [{ side: 'L', at: 'belly', from: 0.1, to: 0.9, off: [0, -0.1, 0.05] }, { side: 'R', at: 'belly', from: 0.1, to: 0.9, off: [0, 0.05, 0.08] }, { side: 'B', at: 'hip', from: 1.0 }], expr: 'oops', look: 0 }),
    catchJoy: gloveUp('pickle on the side'),
  },

  // ── Jun, the infomercial host: always selling to a camera nobody else sees
  jun: {
    idle: { sway: 1.2, posture: posture({ ...TALL, head: [-0.1, 0.15, 0] }) },
    fidgets: [
      G('but wait!', 2.6, [
        [0, { hips: [0, 0, 0], spine: [0, 0, 0], head: [0, 0, 0], armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0] }],
        [0.4, { hips: [0, 0.5, 0], spine: [0, 0.3, 0], head: [-0.1, 0.4, 0] }],
        [0.8, { armR: [-1.2, 0.6, -0.9], foreR: [-0.1, 0, 0], armL: [-0.3, 0, 0.3], foreL: [-0.6, 0, 0] }, 'out'],
        [1.4, { armR: [-1.3, -0.2, -0.5], foreR: [-0.1, 0, 0], head: [-0.1, 0.45, -0.1] }],
        [1.7, { armR: [-1.6, 0, -0.3], foreR: [-0.05, 0, 0], armL: [-0.3, 0, 0.3] }, 'out'],
        [2.2, { armR: [-1.6, 0, -0.3], foreR: [-0.05, 0, 0] }],
        [2.6, { hips: [0, 0, 0], spine: [0, 0, 0], head: [0, 0, 0], armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0] }],
      ], { expr: 'yell', look: 0 }),
      G('thumbs to camera', 1.6, [
        [0, { armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0], head: [0, 0, 0] }],
        [0.35, { armL: [-0.6, -0.3, 0.3], armR: [-0.6, 0.3, -0.3], foreL: [-1.6, 0, 0], foreR: [-1.6, 0, 0], head: [-0.1, 0.4, 0.1], hips: [0, 0.4, 0] }, 'out'],
        [1.2, { armL: [-0.6, -0.3, 0.3], armR: [-0.6, 0.3, -0.3], foreL: [-1.7, 0, 0], foreR: [-1.7, 0, 0] }],
        [1.6, { armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0], head: [0, 0, 0], hips: [0, 0, 0] }],
      ], { expr: 'happy', look: 0 }),
      G('headset check', 1.4, [[0, { head: [0, 0, 0] }], [0.4, { head: [0, -0.2, 0.15] }], [1.4, { head: [0, 0, 0] }]],
        { hands: [{ side: 'L', at: 'ear', from: 0.1, to: 1.2, off: [0, 0.1, 0.1] }], look: 0.3 }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.2, open: 0.4, wide: 0.3, hands: 0.3, tilt: 0.1, waggle: 'circle', waggleAmt: 1.0, waggleSpeed: 1.2 },
    gait: { bounce: 1.2, arms: 1.1, lean: 0.8, cadence: 1.05 },
    trot: loop('waving at the camera', 0.7, [
      [0, { armR: [-0.3, 0, -2.5], foreR: [0, 0, -0.6] }], [0.35, { armR: [-0.3, 0, -2.5], foreR: [0, 0, 0.2] }], [0.7, { armR: [-0.3, 0, -2.5], foreR: [0, 0, -0.6] }],
    ], { expr: 'yell' }),
    celebrate: loop('ta-da, act now', 2.0, [
      [0, { armL: [-0.6, 0, 1.0], armR: [-0.6, 0, -1.0], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0], spine: [-0.05, 0, 0], hipY: 0 }],
      [0.3, { armL: [-0.5, 0, 1.9], armR: [-0.5, 0, -1.9], foreL: [0, 0, 0.3], foreR: [0, 0, -0.3], spine: [-0.15, 0, 0], head: [-0.25, 0, 0], hipY: 0.15 }, 'out'],
      [0.9, { armL: [-0.5, 0, 1.9], armR: [-0.5, 0, -1.9], spine: [-0.15, 0, 0], hipY: 0 }],
      [1.2, { armR: [-1.6, 0, -0.2], foreR: [-0.05, 0, 0], armL: [-0.2, 0, 0.4], foreL: [-0.8, 0, 0], spine: [0.1, 0, 0], head: [-0.1, 0, 0] }, 'out'],
      [2.0, { armR: [-1.6, 0, -0.2], foreR: [-0.05, 0, 0], armL: [-0.6, 0, 1.0], foreL: [-0.2, 0, 0], spine: [0.05, 0, 0] }],
    ], { expr: 'yell' }),
    strikeout: G('but wait, there\'s more', 1.8, [
      [0, { armR: [0, 0, -0.1], head: [0, 0, 0] }],
      [0.6, { head: [0.3, 0, 0], spine: [0.1, 0, 0] }],
      [1.0, { armR: [-0.4, 0, -2.6], foreR: [0, 0, 0.3], head: [-0.2, 0, 0], spine: [-0.1, 0, 0] }, 'out'],
      [1.8, { armR: [-0.4, 0, -2.6], foreR: [0, 0, 0.3], head: [-0.2, 0.3, 0], spine: [-0.1, 0, 0] }],
    ], { expr: 'surprised', look: 0 }),
    catchJoy: G('present the glove', 1.4, [
      [0, { armL: [-0.9, 0, 0.1] }],
      [0.35, { armL: [-1.4, 0.5, 0.6], foreL: [-0.2, 0, 0], armR: [-0.5, 0, -0.9], foreR: [-0.3, 0, 0], head: [-0.1, 0.4, 0] }, 'out'],
      [1.4, { armL: [-1.4, 0.5, 0.6], foreL: [-0.2, 0, 0], armR: [-0.5, 0, -0.9], foreR: [-0.3, 0, 0] }],
    ], { expr: 'yell', look: 0 }),
  },

  // ── Kai, the monster-truck announcer: maximum volume, maximum flex
  kai: {
    idle: { sway: 1.1, posture: posture({ ...TALL, chest: [-0.12, 0, 0], thighL: [0, 0, 0.1], thighR: [0, 0, -0.1], armL: [0.05, 0, 0.3], armR: [0.05, 0, -0.3] }) },
    fidgets: [
      G('SUNDAY!', 2.2, [
        [0, { spine: [0, 0, 0], head: [0, 0, 0], armL: [0.05, 0, 0.3], foreL: [-0.2, 0, 0] }],
        [0.4, { spine: [0.2, 0, 0], head: [-0.3, 0, 0], armL: [-0.6, 0, 0.6], foreL: [-1.6, 0, 0] }, 'out'],
        [0.7, { armL: [-1.8, 0, 0.6], foreL: [-1.0, 0, 0] }, 'in'], [1.0, { armL: [-0.6, 0, 0.6], foreL: [-1.6, 0, 0] }],
        [1.3, { armL: [-1.8, 0, 0.6], foreL: [-1.0, 0, 0] }, 'in'],
        [2.2, { spine: [0, 0, 0], head: [0, 0, 0], armL: [0.05, 0, 0.3], foreL: [-0.2, 0, 0] }],
      ], { prop: true, hands: [{ side: 'R', at: 'mouth', from: 0.1, to: 1.9, aim: 'mic' }], expr: 'yell', look: 0 }),
      G('double flex', 1.8, [[0, ARMS_DOWN], [0.35, { ...FLEX, spine: [-0.15, 0, 0], head: [-0.2, 0, 0] }, 'out'], [1.3, { ...FLEX, spine: [-0.15, 0, 0], head: [-0.2, 0, 0] }], [1.8, { ...ARMS_DOWN, spine: [0, 0, 0], head: [0, 0, 0] }]], { expr: 'yell' }),
      stretch(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.15, open: -0.15, wide: 0.6, hands: 0.5, tilt: 0.3, waggle: 'pump', waggleAmt: 1.4, waggleSpeed: 1.1 },
    gait: { bounce: 1.1, arms: 1.3, lean: 1, cadence: 0.95 },
    trot: loop('MAXIMUM POWER', 0.8, [[0, { ...V_UP, foreL: [0, 0, 0.6], foreR: [0, 0, -0.6] }], [0.4, { ...V_UP, armL: [-0.25, 0, 2.3], armR: [-0.25, 0, -2.3], foreL: [0, 0, 1.0], foreR: [0, 0, -1.0] }, 'in'], [0.8, { ...V_UP, foreL: [0, 0, 0.6], foreR: [0, 0, -0.6] }]], { expr: 'yell' }),
    celebrate: loop('roar', 1.2, [
      [0, { ...FLEX, spine: [0.25, 0, 0], head: [-0.3, 0, 0], hipY: -0.2, thighL: [-0.4, 0, 0.18], thighR: [-0.4, 0, -0.18], shinL: [0.7, 0, 0], shinR: [0.7, 0, 0] }],
      [0.6, { ...FLEX, armL: [-0.1, 0, 1.25], armR: [-0.1, 0, -1.25], foreL: [0, 0, 2.0], foreR: [0, 0, -2.0], spine: [0.35, 0, 0], head: [-0.45, 0, 0], hipY: -0.25 }],
      [1.2, { ...FLEX, spine: [0.25, 0, 0], head: [-0.3, 0, 0], hipY: -0.2 }],
    ], { expr: 'yell' }),
    strikeout: G('fists down', 1.4, [
      [0, { spine: [0, 0, 0] }],
      [0.25, { armL: [-0.8, 0, 0.6], armR: [-0.8, 0, -0.6], foreL: [-1.8, 0, 0], foreR: [-1.8, 0, 0], spine: [-0.15, 0, 0], head: [-0.4, 0, 0] }, 'out'],
      [0.45, { armL: [0.2, 0, 0.3], armR: [0.2, 0, -0.3], foreL: [-0.5, 0, 0], foreR: [-0.5, 0, 0], spine: [0.35, 0, 0], head: [0.3, 0, 0], hipY: -0.2, thighL: [-0.3, 0, 0.15], shinL: [0.5, 0, 0], thighR: [-0.3, 0, -0.15], shinR: [0.5, 0, 0] }, 'in'],
      [1.4, { armL: [0.15, 0, 0.25], armR: [0.15, 0, -0.25], foreL: [-0.4, 0, 0], foreR: [-0.4, 0, 0], spine: [0.3, 0, 0], head: [0.35, 0, 0], hipY: -0.1, thighL: [-0.15, 0, 0.1], shinL: [0.25, 0, 0], thighR: [-0.15, 0, -0.1], shinR: [0.25, 0, 0] }],
    ], { expr: 'yell', look: 0 }),
    catchJoy: G('one-arm flex', 1.3, [[0, { armR: [0, 0, -0.1] }], [0.3, { armR: [-0.1, 0, -1.45], foreR: [0, 0, -1.8], head: [-0.2, -0.2, 0] }, 'out'], [1.3, { armR: [-0.1, 0, -1.45], foreR: [0, 0, -1.85] }]], { expr: 'yell' }),
  },

  // ── Ruby, the morning-show host: coffee in hand, waving to everyone
  ruby: {
    idle: { sway: 1.3, posture: posture({ ...TALL, head: [-0.1, 0, 0.1] }) },
    fidgets: [
      G('sip', 2.2, [[0, { head: [0, 0, 0] }], [0.6, { head: [-0.25, 0, 0] }], [1.5, { head: [-0.3, 0, 0] }], [2.2, { head: [0, 0, 0] }]],
        { prop: true, hands: [{ side: 'R', at: 'front', from: 0.05, to: 0.5, off: [0, -0.05, 0.05] }, { side: 'R', at: 'mouth', from: 0.45, to: 1.7, off: [0, -0.05, 0.08], aim: 'cup' }, { side: 'R', at: 'front', from: 1.65, to: 2.1, off: [0, -0.05, 0.05] }], expr: 'happy', look: 0.2 }),
      G('big wave', 1.8, [
        [0, { armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], head: [0, 0, 0] }],
        [0.3, { armL: [-0.3, 0, 2.5], foreL: [0, 0, 0.5], head: [-0.15, -0.3, 0] }, 'out'],
        [0.55, { foreL: [0, 0, -0.3] }], [0.8, { foreL: [0, 0, 0.5] }], [1.05, { foreL: [0, 0, -0.3] }], [1.3, { foreL: [0, 0, 0.5] }],
        [1.8, { armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], head: [0, 0, 0] }],
      ], { expr: 'happy', look: 0 }),
      G('traffic report', 2.0, [[0, { head: [0, 0, 0] }], [0.5, { head: [-0.2, 0.5, 0] }], [1.3, { head: [-0.2, -0.5, 0] }], [2.0, { head: [0, 0, 0] }]],
        { hands: [{ side: 'L', at: 'brow', from: 0.1, to: 1.8, off: [0, 0.05, 0.1], aim: 'flat' }], look: 0 }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.25, open: 0.15, wide: 0.25, hands: 0.1, waggle: 'bob', waggleAmt: 1.1, waggleSpeed: 1.3 },
    gait: { bounce: 1.3, arms: 1.05, lean: 0.8, cadence: 1.1 },
    trot: loop('waving both hands', 1.0, [
      [0, { armL: [-0.3, 0, 2.4], armR: [-0.3, 0, -1.4], foreL: [0, 0, 0.4], foreR: [-0.4, 0, 0] }],
      [0.5, { armL: [-0.3, 0, 1.4], armR: [-0.3, 0, -2.4], foreL: [-0.4, 0, 0], foreR: [0, 0, -0.4] }],
      [1.0, { armL: [-0.3, 0, 2.4], armR: [-0.3, 0, -1.4], foreL: [0, 0, 0.4], foreR: [-0.4, 0, 0] }],
    ], { expr: 'yell' }),
    celebrate: loop('good morning!', 0.8, [
      [0, { armL: [-0.3, 0, 2.5], armR: [-0.3, 0, -2.5], foreL: [0, 0, 0.5], foreR: [0, 0, -0.5], ...HOP_DOWN }],
      [0.2, { foreL: [0, 0, -0.3], foreR: [0, 0, 0.3], ...HOP_UP }, 'out'],
      [0.4, { foreL: [0, 0, 0.5], foreR: [0, 0, -0.5], ...HOP_DOWN }, 'in'],
      [0.6, { foreL: [0, 0, -0.3], foreR: [0, 0, 0.3], ...LEGS }],
      [0.8, { foreL: [0, 0, 0.5], foreR: [0, 0, -0.5], ...HOP_DOWN }],
    ], { expr: 'yell' }),
    strikeout: G('brave face', 2.0, [[0, { head: [0, 0, 0] }], [0.5, { head: [0.3, 0, 0], spine: [0.1, 0, 0] }], [1.2, { head: [-0.1, 0.2, 0.1], spine: [-0.05, 0, 0] }], [2.0, { head: [-0.1, 0.25, 0.12] }]],
      { prop: true, hands: [{ side: 'R', at: 'mouth', from: 1.0, off: [0, -0.05, 0.08], aim: 'cup' }], expr: 'oops', look: 0 }),
    catchJoy: G('thumbs up', 1.2, [[0, { armR: [0, 0, -0.1] }], [0.3, { armR: [-0.7, 0.3, -0.3], foreR: [-1.6, 0, 0], head: [-0.1, 0, 0.1] }, 'out'], [1.2, { armR: [-0.7, 0.3, -0.3], foreR: [-1.7, 0, 0] }]], { expr: 'happy' }),
  },

  // ── Ezra, the insurance salesman: checks his watch, straightens his tie
  ezra: {
    idle: { sway: 0.8, posture: posture({ ...TALL, spine: [-0.02, 0, 0] }), hands: [{ side: 'B', at: 'front', off: [-0.1, -0.35, -0.1] }] },
    fidgets: [
      G('checks watch', 1.6, [[0, { head: [0, 0, 0] }], [0.4, { head: [0.45, 0.2, 0] }], [1.2, { head: [0.45, 0.2, 0] }], [1.6, { head: [0, 0, 0] }]],
        { hands: [{ side: 'L', at: 'front', from: 0.1, to: 1.4, off: [0.02, 0.1, 0.0] }], look: 0 }),
      G('tie', 1.4, [[0, { head: [0, 0, 0] }], [0.4, { head: [-0.2, 0, 0.1] }], [1.4, { head: [0, 0, 0] }]],
        { hands: [{ side: 'R', at: 'neck', from: 0.1, to: 1.2 }, { side: 'L', at: 'neck', from: 0.15, to: 1.1, off: [0, -0.15, 0] }], expr: 'smug', look: 0.3 }),
      G('briefcase', 2.0, [[0, { armR: [0.05, 0, -0.15] }], [2.0, { armR: [0.05, 0, -0.15] }]], { prop: true, sit: false }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.25, open: -0.2, wide: 0.3, hands: 0, tilt: 0, waggle: 'still', waggleAmt: 0.5, waggleSpeed: 0.9 },
    gait: { bounce: 0.6, arms: 0.6, lean: 0.7, cadence: 1.1 },
    trot: loop('handshakes all round', 1.2, [[0, { armR: [-1.2, 0.3, -0.2], foreR: [-0.3, 0, 0] }], [0.3, { foreR: [-0.5, 0, 0] }], [0.6, { foreR: [-0.3, 0, 0] }], [1.2, { armR: [-1.2, 0.3, -0.2], foreR: [-0.3, 0, 0] }]], { expr: 'happy' }),
    celebrate: loop('fully covered', 1.6, [
      [0, { armR: [-0.7, 0.3, -0.3], foreR: [-1.6, 0, 0], armL: [-0.7, -0.3, 0.3], foreL: [-1.6, 0, 0], ...LEGS }],
      [0.3, { armR: [-0.7, 0.3, -0.3], foreR: [-1.7, 0, 0], armL: [-0.7, -0.3, 0.3], foreL: [-1.7, 0, 0], ...HOP_UP, hipY: 0.15 }, 'out'],
      [0.6, { ...LEGS }, 'in'],
      [1.6, { armR: [-0.7, 0.3, -0.3], foreR: [-1.6, 0, 0], armL: [-0.7, -0.3, 0.3], foreL: [-1.6, 0, 0] }],
    ], { expr: 'happy' }),
    strikeout: G('act of nature', 1.8, [
      [0, { armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12] }],
      [0.5, { armL: [-0.6, 0, 0.7], armR: [-0.6, 0, -0.7], foreL: [-0.9, -0.8, 0], foreR: [-0.9, 0.8, 0], head: [0.1, 0, 0.25], shoulderL: [0, 0, 0.2], shoulderR: [0, 0, -0.2] }, 'out'],
      [1.8, { armL: [-0.55, 0, 0.65], armR: [-0.55, 0, -0.65], foreL: [-0.9, -0.8, 0], foreR: [-0.9, 0.8, 0], head: [0.15, 0, 0.2], shoulderL: [0, 0, 0.1], shoulderR: [0, 0, -0.1] }],
    ], { expr: 'oops', look: 0 }),
    catchJoy: G('thumbs up', 1.2, [[0, { armR: [0, 0, -0.1] }], [0.3, { armR: [-0.7, 0.3, -0.3], foreR: [-1.6, 0, 0] }, 'out'], [1.2, { armR: [-0.7, 0.3, -0.3], foreR: [-1.7, 0, 0] }]], { expr: 'smug' }),
  },

  // ── Maya, the mail carrier: always on her toes, salutes, delivers the paper
  maya: {
    idle: { sway: 1.6, posture: posture({ head: [-0.05, 0, 0], hipY: 0.03, footL: [-0.15, 0, 0], footR: [-0.15, 0, 0] }) },
    fidgets: [
      G('delivers the paper', 1.6, [
        [0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], hips: [0, 0, 0], spine: [0, 0, 0] }],
        [0.45, { armR: [-0.4, -0.8, -1.2], foreR: [-1.2, 0, 0], hips: [0, -0.4, 0], spine: [0, -0.3, 0], head: [0, 0.5, 0] }, 'out'],
        [0.65, { armR: [-1.4, 0.4, -0.9], foreR: [-0.1, 0, 0], hips: [0, 0.3, 0], spine: [0.1, 0.25, 0], head: [0, 0.3, 0] }, 'in'],
        [1.6, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], hips: [0, 0, 0], spine: [0, 0, 0], head: [0, 0, 0] }],
      ], { prop: true, look: 0.2 }),
      G('stretches her calves', 2.2, [
        [0, { ...LEGS }], [0.5, { thighL: [-0.4, 0, 0.05], shinL: [0.6, 0, 0], thighR: [0.3, 0, -0.04], hipY: -0.1, spine: [0.15, 0, 0] }],
        [1.6, { thighL: [-0.45, 0, 0.05], shinL: [0.65, 0, 0], thighR: [0.32, 0, -0.04], hipY: -0.12, spine: [0.18, 0, 0] }], [2.2, { ...LEGS, spine: [0, 0, 0] }],
      ], { hands: [{ side: 'B', at: 'hip', from: 0.2, to: 2.0 }], sit: false }),
      adjustCap(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.45, open: 0.25, wide: 0.25, hands: -0.1, waggle: 'twitch', waggleAmt: 1, waggleSpeed: 1.5 },
    gait: { bounce: 1.1, arms: 1.15, lean: 1.2, cadence: 1.15 },
    trot: loop('salute', 1.2, [[0, { head: [-0.05, 0, 0] }], [1.2, { head: [-0.05, 0, 0] }]], { hands: [{ side: 'R', at: 'brow', off: [0.05, 0.05, 0.05], aim: 'flat' }], expr: 'happy' }),
    celebrate: loop('special delivery', 1.0, [
      [0, { ...LEGS, armL: [0.5, 0, 0.2], foreL: [-1.3, 0, 0] }],
      [0.25, { thighL: [-0.8, 0, 0.05], shinL: [1.2, 0, 0], thighR: [0.2, 0, -0.04], shinR: [0.3, 0, 0], hipY: 0.1, armL: [-0.6, 0, 0.2], foreL: [-1.3, 0, 0] }],
      [0.5, { ...LEGS, armL: [0.5, 0, 0.2], foreL: [-1.3, 0, 0] }],
      [0.75, { thighR: [-0.8, 0, -0.05], shinR: [1.2, 0, 0], thighL: [0.2, 0, 0.04], shinL: [0.3, 0, 0], hipY: 0.1, armL: [0.5, 0, 0.2] }],
      [1.0, { ...LEGS, armL: [0.5, 0, 0.2], foreL: [-1.3, 0, 0] }],
    ], { hands: [{ side: 'R', at: 'brow', off: [0.05, 0.05, 0.05], aim: 'flat' }], expr: 'yell' }),
    strikeout: kickDirt(),
    catchJoy: G('salute', 1.3, [[0, { head: [0, 0, 0] }], [1.3, { head: [-0.1, 0, 0] }]], { hands: [{ side: 'R', at: 'brow', from: 0.1, to: 1.2, off: [0.05, 0.05, 0.05], aim: 'flat' }], expr: 'happy' }),
  },

  // ── Leo, the "French" chef: chef's kiss, twirls the mustache
  leo: {
    idle: { sway: 0.9, posture: posture({ ...TALL, chest: [-0.1, 0, 0], head: [-0.15, 0, 0] }), hands: [{ side: 'L', at: 'back', off: [0.05, 0.05, 0] }] },
    fidgets: [
      G('chef\'s kiss', 1.8, [
        [0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], head: [-0.15, 0, 0] }],
        [0.5, { head: [-0.25, 0, 0] }],
        [0.75, { armR: [-1.4, 0.3, -1.0], foreR: [-0.2, 0, 0], head: [-0.3, 0, 0] }, 'out'],
        [1.3, { armR: [-1.3, 0.3, -1.1], foreR: [-0.2, 0, 0] }],
        [1.8, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], head: [-0.15, 0, 0] }],
      ], { hands: [{ side: 'R', at: 'mouth', from: 0.1, to: 0.65 }], expr: 'smug', look: 0 }),
      G('mustache twirl', 1.8, [[0, { head: [-0.15, 0, 0] }], [0.5, { head: [-0.15, 0.2, 0.1] }], [1.8, { head: [-0.15, 0, 0] }]],
        { hands: [{ side: 'R', at: 'cheek', from: 0.1, to: 1.6, off: [-0.12, 0.05, 0.1] }], expr: 'smug', look: 0.2 }),
      G('rolling pin', 2.4, [
        [0, { head: [-0.15, 0, 0] }], [0.5, { head: [0.3, 0, 0], spine: [0.15, 0, 0] }], [2.0, { head: [0.3, 0, 0], spine: [0.15, 0, 0] }], [2.4, { head: [-0.15, 0, 0], spine: [0, 0, 0] }],
      ], { prop: true, hands: [{ side: 'R', at: 'belly', from: 0.3, to: 0.9, off: [-0.1, 0.1, 0.25] }, { side: 'R', at: 'belly', from: 0.9, to: 1.5, off: [-0.1, 0.1, 0.45] }, { side: 'R', at: 'belly', from: 1.5, to: 2.1, off: [-0.1, 0.1, 0.25] }], look: 0 }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.1, open: 0, wide: 0.2, hands: 0.3, tilt: 0.5, waggle: 'circle', waggleAmt: 0.6, waggleSpeed: 0.7 },
    gait: { bounce: 1.0, arms: 0.6, lean: 0.5, cadence: 1.2 },
    trot: loop('prancing', 0.9, [[0, { head: [-0.25, 0, 0], armL: [-0.3, 0, 0.8], armR: [-0.3, 0, -0.8], foreL: [-0.4, 0, 0], foreR: [-0.4, 0, 0] }], [0.9, { head: [-0.25, 0, 0] }]], { expr: 'smug' }),
    celebrate: loop('magnifique!', 2.2, [
      [0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], armL: [0.05, 0, 0.12], head: [-0.15, 0, 0] }],
      [0.6, { armR: [-1.4, 0.3, -1.0], foreR: [-0.2, 0, 0], head: [-0.3, 0, 0] }, 'out'],
      [1.2, { armR: [-0.4, 0, -1.9], armL: [-0.4, 0, 1.9], foreR: [0, 0, -0.3], foreL: [0, 0, 0.3], spine: [-0.15, 0, 0], head: [-0.4, 0, 0], hipY: 0.1 }, 'out'],
      [2.2, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], spine: [0, 0, 0], head: [-0.15, 0, 0], hipY: 0 }],
    ], { hands: [{ side: 'R', at: 'mouth', from: 0.05, to: 0.5 }], expr: 'happy' }),
    strikeout: G('zut alors', 1.8, [
      [0, { head: [0, 0, 0], armL: [0.05, 0, 0.12], armR: [0.05, 0, -0.12] }],
      [0.4, { armL: [-0.5, 0, 1.4], armR: [-0.5, 0, -1.4], foreL: [0, 0, 1.0], foreR: [0, 0, -1.0], head: [-0.3, 0, 0], spine: [-0.1, 0, 0] }, 'out'],
      [0.9, { armL: [0.05, 0, 0.3], armR: [0.05, 0, -0.3], foreL: [-0.3, 0, 0], foreR: [-0.3, 0, 0], head: [0.1, -0.8, 0], hips: [0, -0.5, 0], spine: [0, -0.2, 0] }, 'io'],
      [1.8, { head: [0.15, -0.8, 0], hips: [0, -0.6, 0], spine: [0, -0.2, 0] }],
    ], { expr: 'oops', look: 0 }),
    catchJoy: G('chef\'s kiss', 1.4, [[0, { armR: [0, 0, -0.1] }], [0.55, { armR: [-1.4, 0.3, -1.0], foreR: [-0.2, 0, 0] }, 'out'], [1.4, { armR: [-1.3, 0.3, -1.1], foreR: [-0.2, 0, 0] }]],
      { hands: [{ side: 'R', at: 'mouth', from: 0.05, to: 0.45 }], expr: 'smug' }),
  },

  // ── Anya, the TV weather lady: presents the forecast with sweeping arms
  anya: {
    idle: { sway: 1, posture: posture({ ...TALL, hips: [0, 0.1, 0], head: [-0.08, -0.1, 0.06] }), hands: [{ side: 'B', at: 'front', off: [-0.08, -0.1, 0.0] }] },
    fidgets: [
      G('five-day forecast', 3.0, [
        [0, { armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], head: [0, 0, 0], hips: [0, 0, 0] }],
        [0.5, { armL: [-0.5, 0, 1.4], foreL: [-0.1, 0, 0], head: [-0.05, 0.6, 0], hips: [0, 0.3, 0] }, 'out'],
        [1.4, { armL: [-1.4, -0.4, 0.5], foreL: [-0.1, 0, 0], head: [-0.05, 0.1, 0] }],
        [2.2, { armL: [-1.2, -0.9, 0.2], foreL: [-0.1, 0, 0], head: [-0.05, -0.2, 0] }],
        [3.0, { armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], head: [0, 0, 0], hips: [0, 0, 0] }],
      ], { hands: [{ side: 'R', at: 'hip', from: 0.2, to: 2.8 }], expr: 'happy', look: 0 }),
      G('checks the sky', 2.0, [[0, { head: [0, 0, 0] }], [0.5, { head: [-0.6, 0.2, 0] }], [1.5, { head: [-0.6, -0.2, 0] }], [2.0, { head: [0, 0, 0] }]],
        { hands: [{ side: 'R', at: 'brow', from: 0.1, to: 1.8, off: [0, 0.05, 0.1], aim: 'flat' }], look: 0 }),
      scratchHead(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.2, open: 0.1, wide: 0.45, hands: 0.2, tilt: 0.2, waggle: 'bob', waggleAmt: 0.9, waggleSpeed: 1 },
    gait: { bounce: 0.9, arms: 1.0, lean: 0.8, cadence: 1 },
    trot: loop('sunny all week', 2.0, [
      [0, { armL: [-0.5, 0, 0.6], armR: [-0.5, 0, -0.6], foreL: [-0.1, 0, 0], foreR: [-0.1, 0, 0] }],
      [1.0, { armL: [-0.4, 0, 2.6], armR: [-0.4, 0, -2.6] }],
      [2.0, { armL: [-0.5, 0, 0.6], armR: [-0.5, 0, -0.6] }],
    ], { expr: 'happy' }),
    celebrate: loop('sunshine sweep', 1.8, [
      [0, { armL: [-0.4, 0, 0.5], armR: [-0.4, 0, -0.5], foreL: [-0.1, 0, 0], foreR: [-0.1, 0, 0], ...LEGS }],
      [0.6, { armL: [-0.3, 0, 2.7], armR: [-0.3, 0, -2.7], ...HOP_UP, hipY: 0.2 }, 'out'],
      [0.9, { ...LEGS }, 'in'],
      [1.8, { armL: [-0.4, 0, 0.5], armR: [-0.4, 0, -0.5] }],
    ], { expr: 'yell' }),
    strikeout: G('cold front', 1.8, [[0, { head: [0, 0, 0] }], [0.4, { head: [-0.1, 0, 0], spine: [-0.08, 0, 0] }], [1.8, { head: [0.05, 0, 0.1], spine: [-0.05, 0, 0] }]],
      { hands: [{ side: 'B', at: 'cheek', from: 0.1 }], expr: 'surprised', look: 0 }),
    catchJoy: G('presenting', 1.3, [[0, { armR: [0, 0, -0.1] }], [0.35, { armR: [-0.6, 0, -1.4], foreR: [-0.1, 0, 0], head: [-0.1, -0.2, 0] }, 'out'], [1.3, { armR: [-0.6, 0, -1.4], foreR: [-0.1, 0, 0] }]], { expr: 'happy' }),
  },

  // ── Darius, the big-shot lawyer: objection!
  darius: {
    idle: { sway: 0.8, posture: posture({ ...TALL, chest: [-0.08, 0, 0] }), hands: [{ side: 'L', at: 'front', off: [-0.3, -0.1, -0.12] }, { side: 'R', at: 'front', off: [-0.3, 0.02, -0.08] }] },
    fidgets: [
      G('objection!', 1.8, [
        [0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], spine: [0, 0, 0] }],
        [0.3, { armR: [-0.6, 0, -0.4], foreR: [-1.6, 0, 0], spine: [-0.08, 0, 0] }],
        [0.45, { armR: [-1.6, 0.2, -0.2], foreR: [-0.05, 0, 0], spine: [0.12, 0.1, 0], head: [-0.1, 0.1, 0] }, 'in'],
        [1.3, { armR: [-1.6, 0.2, -0.2], foreR: [-0.05, 0, 0], spine: [0.12, 0.1, 0], head: [-0.1, 0.1, 0] }],
        [1.8, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], spine: [0, 0, 0], head: [0, 0, 0] }],
      ], { hands: [{ side: 'L', at: 'hip' }], expr: 'yell', look: 0.2 }),
      G('tie', 1.4, [[0, { head: [0, 0, 0] }], [0.4, { head: [-0.2, 0, 0.1] }], [1.4, { head: [0, 0, 0] }]],
        { hands: [{ side: 'R', at: 'neck', from: 0.1, to: 1.2 }, { side: 'L', at: 'neck', from: 0.15, to: 1.1, off: [0, -0.15, 0] }], expr: 'smug', look: 0.3 }),
      G('briefcase', 2.0, [[0, { armR: [0.05, 0, -0.15] }], [2.0, { armR: [0.05, 0, -0.15] }]], { prop: true, sit: false }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.2, open: -0.25, wide: 0.35, hands: 0.25, tilt: 0.1, waggle: 'pump', waggleAmt: 0.8, waggleSpeed: 1 },
    gait: { bounce: 0.8, arms: 0.85, lean: 0.8, cadence: 1 },
    trot: loop('I rest my case', 1.6, [[0, { armL: [-0.5, 0, 1.0], armR: [-0.5, 0, -1.0], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0] }], [0.8, { armL: [-0.5, 0, 1.2], armR: [-0.5, 0, -1.2] }], [1.6, { armL: [-0.5, 0, 1.0], armR: [-0.5, 0, -1.0] }]], { expr: 'smug' }),
    celebrate: loop('case closed', 2.0, [
      [0, { armL: [-0.5, 0, 1.1], armR: [-0.5, 0, -1.1], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0], spine: [-0.1, 0, 0], head: [-0.2, 0, 0] }],
      [0.9, { armL: [-0.5, 0, 1.3], armR: [-0.5, 0, -1.3], spine: [-0.12, 0, 0], head: [-0.25, 0, 0] }],
      [1.2, { armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], armR: [-0.4, 0.6, 0], foreR: [-1.8, 0, 0], spine: [0, 0, 0], head: [0.1, 0.3, 0] }, 'out'],
      [1.45, { armR: [-0.6, 0.8, 0.1], foreR: [-1.5, 0, 0] }], [1.6, { armR: [-0.4, 0.6, 0], foreR: [-1.8, 0, 0] }],
      [2.0, { armL: [-0.5, 0, 1.1], armR: [-0.5, 0, -1.1], foreL: [-0.2, 0, 0], foreR: [-0.2, 0, 0], spine: [-0.1, 0, 0], head: [-0.2, 0, 0] }],
    ], { expr: 'smug' }),
    strikeout: G('objection, your honor', 2.0, [
      [0, { armR: [0, 0, -0.1], spine: [0, 0, 0] }],
      [0.35, { armR: [-1.7, 0.4, -0.2], foreR: [-0.05, 0, 0], spine: [0.15, 0.2, 0], head: [-0.15, 0.3, 0], hipY: -0.05 }, 'out'],
      [0.8, { armR: [-1.6, 0.4, -0.2], spine: [0.15, 0.2, 0] }],
      [1.1, { armR: [-1.9, 0.4, -0.3], spine: [0.2, 0.2, 0] }, 'in'],
      [2.0, { armR: [-1.6, 0.4, -0.2], foreR: [-0.05, 0, 0], spine: [0.15, 0.2, 0], head: [-0.15, 0.3, 0] }],
    ], { hands: [{ side: 'L', at: 'hip' }], expr: 'yell', look: 0 }),
    catchJoy: G('point', 1.2, [[0, { armR: [0, 0, -0.1] }], [0.3, { armR: [-1.6, 0.2, -0.2], foreR: [-0.05, 0, 0] }, 'out'], [1.2, { armR: [-1.6, 0.2, -0.2], foreR: [-0.05, 0, 0] }]], { expr: 'smug' }),
  },

  // ── Pepper, the auctioneer: gavel, fast talk, SOLD!
  pepper: {
    idle: { sway: 1.8, posture: posture({ spine: [0.05, 0, 0], head: [-0.05, 0, 0] }) },
    fidgets: [
      G('going once…', 2.0, [
        [0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], head: [0, 0, 0] }],
        [0.3, { armR: [-1.2, 0, -0.5], foreR: [-1.6, 0, 0], head: [-0.1, 0.3, 0] }, 'out'],
        [0.5, { armR: [-1.0, 0, -0.4], foreR: [-0.6, 0, 0], head: [0, 0.2, 0], spine: [0.1, 0, 0] }, 'in'],
        [0.75, { armR: [-1.2, 0, -0.5], foreR: [-1.6, 0, 0], head: [-0.1, -0.3, 0], spine: [0, 0, 0] }],
        [0.95, { armR: [-1.0, 0, -0.4], foreR: [-0.6, 0, 0], head: [0, -0.2, 0], spine: [0.1, 0, 0] }, 'in'],
        [1.3, { armR: [-1.4, 0, -0.5], foreR: [-1.9, 0, 0], head: [-0.15, 0, 0], spine: [-0.05, 0, 0] }, 'out'],
        [1.45, { armR: [-0.9, 0, -0.4], foreR: [-0.4, 0, 0], head: [0.05, 0, 0], spine: [0.15, 0, 0] }, 'in'],
        [2.0, { armR: [0.05, 0, -0.12], foreR: [-0.2, 0, 0], head: [0, 0, 0], spine: [0, 0, 0] }],
      ], { prop: true, expr: 'yell', look: 0 }),
      G('fast talk', 1.8, [
        [0, { armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], head: [0, 0, 0] }],
        [0.3, { armL: [-1.3, -0.4, 0.3], foreL: [-0.1, 0, 0], head: [0, 0.4, 0] }, 'out'],
        [0.7, { armL: [-1.3, 0.3, 0.6], head: [0, -0.2, 0] }],
        [1.1, { armL: [-1.3, -0.6, 0.2], head: [0, 0.5, 0] }],
        [1.8, { armL: [0.05, 0, 0.12], foreL: [-0.2, 0, 0], head: [0, 0, 0] }],
      ], { expr: 'yell', look: 0 }),
      adjustCap(),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.35, open: 0.2, wide: 0.2, hands: -0.1, tilt: 0.2, waggle: 'twitch', waggleAmt: 1.2, waggleSpeed: 2 },
    gait: { bounce: 1.2, arms: 1.1, lean: 1.0, cadence: 1.3 },
    trot: loop('gavel high', 0.6, [[0, { armR: [-0.4, 0, -2.4], foreR: [0, 0, -0.8] }], [0.3, { armR: [-0.4, 0, -2.4], foreR: [0, 0, 0.1] }, 'in'], [0.6, { armR: [-0.4, 0, -2.4], foreR: [0, 0, -0.8] }]], { prop: true, expr: 'yell' }),
    celebrate: loop('SOLD!', 1.2, [
      [0, { armR: [-1.5, 0, -0.5], foreR: [-2.0, 0, 0], ...LEGS, spine: [-0.05, 0, 0] }],
      [0.18, { armR: [-1.0, 0, -0.4], foreR: [-0.3, 0, 0], spine: [0.2, 0, 0], ...HOP_DOWN }, 'in'],
      [0.45, { armR: [-0.4, 0, -2.5], foreR: [0, 0, -0.4], ...HOP_UP, spine: [-0.15, 0, 0] }, 'out'],
      [0.7, { ...LEGS }, 'in'],
      [1.2, { armR: [-1.5, 0, -0.5], foreR: [-2.0, 0, 0], spine: [-0.05, 0, 0] }],
    ], { prop: true, expr: 'yell' }),
    strikeout: G('stamps her foot', 1.6, [
      [0, { thighR: [0, 0, -0.04], shinR: [0, 0, 0] }],
      [0.25, { thighR: [-0.7, 0, -0.04], shinR: [1.0, 0, 0] }, 'out'], [0.38, { thighR: [0, 0, -0.04], shinR: [0, 0, 0], hipY: -0.06 }, 'in'],
      [0.6, { thighR: [-0.6, 0, -0.04], shinR: [0.9, 0, 0], hipY: 0 }, 'out'], [0.72, { thighR: [0, 0, -0.04], shinR: [0, 0, 0], hipY: -0.06 }, 'in'],
      [1.6, { hipY: 0, head: [0.2, 0.2, 0] }],
    ], { hands: [{ side: 'B', at: 'hip', from: 0.1 }], expr: 'yell', look: 0 }),
    catchJoy: gloveUp('sold to the kid in the glove'),
  },

  // ── Hank, the mall security guard: hands behind his back, scanning the yard
  hank: {
    idle: { sway: 0.5, posture: posture({ ...TALL, thighL: [0, 0, 0.1], thighR: [0, 0, -0.1] }), hands: [{ side: 'B', at: 'back' }] },
    fidgets: [
      G('security scan', 4.0, [
        [0, { head: [0, 0, 0], spine: [0, 0, 0] }],
        [1.2, { head: [0, 0.8, 0], spine: [0, 0.15, 0] }],
        [1.8, { head: [0, 0.8, 0], spine: [0, 0.15, 0] }],
        [3.2, { head: [0, -0.8, 0], spine: [0, -0.15, 0] }],
        [4.0, { head: [0, 0, 0], spine: [0, 0, 0] }],
      ], { look: 0, expr: 'focus' }),
      G('earpiece', 1.8, [[0, { head: [0, 0, 0] }], [0.4, { head: [0.05, 0, -0.2] }], [1.8, { head: [0, 0, 0] }]],
        { hands: [{ side: 'R', at: 'ear', from: 0.1, to: 1.6, off: [0, 0.05, 0.05] }], look: 0.4, expr: 'focus' }),
      G('belt hitch', 1.2, [[0, { hipY: 0 }], [0.4, { hipY: 0.06, spine: [-0.05, 0, 0] }], [0.7, { hipY: -0.02, spine: [0, 0, 0] }], [1.2, { hipY: 0 }]],
        { hands: [{ side: 'B', at: 'hip', from: 0.05, to: 1.0, off: [-0.1, -0.1, 0.15] }], sit: false }),
    ],
    stance: { ...DEFAULT_STANCE, crouch: 0.05, open: -0.1, wide: 0.55, hands: -0.2, tilt: -0.2, waggle: 'still', waggleAmt: 0.4, waggleSpeed: 0.5 },
    gait: { bounce: 0.6, arms: 0.8, lean: 0.6, cadence: 0.82 },
    trot: loop('polite wave', 2.0, [[0, { armR: [-0.4, 0, -1.6], foreR: [0, 0, -0.5] }], [1.0, { armR: [-0.4, 0, -1.6], foreR: [0, 0, -0.2] }], [2.0, { armR: [-0.4, 0, -1.6], foreR: [0, 0, -0.5] }]], { expr: 'happy' }),
    celebrate: loop('polite applause', 1.0, [
      [0, { armL: [-0.95, 0, 0.4], armR: [-0.95, 0, -0.4], foreL: [-1.0, -0.6, 0], foreR: [-1.0, 0.6, 0], head: [0.1, 0, 0] }],
      [0.25, { armL: [-0.95, 0, 0.2], armR: [-0.95, 0, -0.2], head: [0.15, 0, 0] }, 'in'],
      [0.5, { armL: [-0.95, 0, 0.4], armR: [-0.95, 0, -0.4], head: [0.1, 0, 0] }],
      [0.75, { armL: [-0.95, 0, 0.2], armR: [-0.95, 0, -0.2], head: [0.05, 0, 0] }, 'in'],
      [1.0, { armL: [-0.95, 0, 0.4], armR: [-0.95, 0, -0.4], head: [0.1, 0, 0] }],
    ], { expr: 'happy' }),
    strikeout: G('move along', 1.8, [[0, { head: [0, 0, 0] }], [0.5, { head: [0.2, 0, 0], shoulderL: [0, 0, 0.15], shoulderR: [0, 0, -0.15] }, 'out'], [1.0, { head: [0.1, 0, 0], shoulderL: [0, 0, 0], shoulderR: [0, 0, 0] }], [1.8, { head: [0.1, 0.3, 0] }]],
      { hands: [{ side: 'B', at: 'hip', from: 0.8, off: [-0.1, -0.1, 0.15] }], expr: 'neutral', look: 0 }),
    catchJoy: G('thumbs up', 1.3, [[0, { armR: [0, 0, -0.1] }], [0.35, { armR: [-0.7, 0.3, -0.3], foreR: [-1.6, 0, 0], head: [0.15, 0, 0] }, 'out'], [1.3, { armR: [-0.7, 0.3, -0.3], foreR: [-1.7, 0, 0], head: [0, 0, 0] }]], { expr: 'happy' }),
  },
};

const DEFAULT: Personality = {
  idle: { sway: 1 },
  fidgets: [stretch(), scratchHead(), kickDirt(), adjustCap()],
  stance: DEFAULT_STANCE,
  gait: DEFAULT_GAIT,
  celebrate: loop('jump and pump', 0.8, [[0, { ...V_UP, ...HOP_DOWN }], [0.3, { ...V_UP, ...HOP_UP }, 'out'], [0.6, { ...V_UP, ...LEGS }, 'in'], [0.8, { ...V_UP, ...HOP_DOWN }]], { expr: 'yell' }),
  strikeout: G('slump', 1.4, [[0, { spine: [0, 0, 0] }], [0.7, SLUMP, 'out'], [1.4, SLUMP]], { expr: 'sad', look: 0 }),
  catchJoy: gloveUp(),
};

/** This kid's personality (grown-ups and unknown ids get a plain one). */
export function personaOf(id: string): Personality {
  return PERSONAS[id] ?? DEFAULT;
}

export const PERSONA_IDS = Object.keys(PERSONAS);
