import type { Vec2 } from '../engine/math';
import type { Kid, Position, Special } from '../data/types';

export type Difficulty = 'rookie' | 'pro' | 'allstar';

export type FielderTask = 'idle' | 'chase' | 'cover' | 'backup' | 'carry' | 'hold' | 'receive' | 'tag';
export type FielderAnim = 'ready' | 'run' | 'catch' | 'throw' | 'dive' | 'jump' | 'stumble' | 'cheer' | 'pitch' | 'crouch';

export interface FielderState {
  idx: number;
  pos: Position;
  kid: Kid;
  p: Vec2;
  home: Vec2;
  facing: number;
  speed: number;
  task: FielderTask;
  target: Vec2;
  coverBase: number; // 0 home .. 3 third, -1 none
  hasBall: boolean;
  reaction: number;
  catchCd: number;
  holdT: number;
  anim: FielderAnim;
  animT: number;
  /** height of the glove for drawing jumps/dives */
  lift: number;
  /** this play's misjudgment of where the ball is going (shrinks as it arrives) */
  misread: Vec2;
}

export type RunnerAnim = 'run' | 'stand' | 'slide' | 'trot' | 'out' | 'cheer';

export interface RunnerState {
  kid: Kid;
  /** last base safely touched: 0 = home (batter), 1..3 */
  base: number;
  /** feet travelled from `base` toward base+1 */
  d: number;
  dir: -1 | 0 | 1;
  forced: boolean;
  mustTag: boolean;
  isBatter: boolean;
  startBase: number;
  out: boolean;
  scored: boolean;
  /** if set the runner auto-advances to this base (ground-rule awards, HR trot) */
  goal: number | null;
  /** stop leading off at this many feet (fly ball "halfway") */
  holdAt: number | null;
  /** human pressed advance/retreat: stick to it until the next base */
  manual: boolean;
  speed: number;
  anim: RunnerAnim;
  animT: number;
  reevalT: number;
}

export type OutKind = 'fly' | 'force' | 'tag' | 'doubledOff' | 'strikeout';

export interface OutRecord {
  kind: OutKind;
  kidId: string;
  fielderIdx: number;
}

export interface PlayResult {
  foul: boolean;
  homeRun: boolean;
  groundRule: boolean;
  outs: OutRecord[];
  scored: string[];
  /** batter's final base (0 = out / not reached, 4 = scored) */
  batterBase: number;
  /** final occupants of 1st..3rd */
  bases: (string | null)[];
  error: boolean;
  errorBy: number[];
  /** bases the batter earned cleanly before any error */
  hitBases: number;
  caughtFly: boolean;
  firstFielder: number;
  /** runs to cancel because the third out was a force */
  cancelledRuns: number;
}

export type MatchEvent =
  | { type: 'pitch'; pitcher: string; pitch: string; special: Special | null; mph: number }
  | { type: 'call'; call: 'ball' | 'strike' | 'foul' | 'swinging' }
  | { type: 'contact'; batter: string; ev: number; la: number; spray: number; quality: number }
  | { type: 'whiff'; batter: string }
  | { type: 'catch'; fielder: string; fly: boolean; hard: boolean }
  | { type: 'bobble'; fielder: string }
  | { type: 'throw'; fielder: string; base: number }
  | { type: 'out'; kind: OutKind; runner: string; fielder: string }
  | { type: 'safe'; runner: string; base: number }
  | { type: 'run'; runner: string }
  | { type: 'hit'; batter: string; bases: number }
  | { type: 'homeRun'; batter: string; runs: number }
  | { type: 'groundRule'; batter: string; why: 'splash' | 'bounce' }
  | { type: 'walk'; batter: string; hbp: boolean }
  | { type: 'strikeout'; batter: string; looking: boolean }
  | { type: 'error'; fielder: string }
  | { type: 'bounce'; x: number; y: number; speed: number; surface: string }
  | { type: 'fence'; kind: string; cleared: boolean }
  | { type: 'tree' }
  | { type: 'dog' }
  | { type: 'splash' }
  | { type: 'special'; kid: string; special: Special }
  | { type: 'quip'; kid: string; text: string }
  | { type: 'halfOver'; inning: number; half: 0 | 1 }
  | { type: 'gameOver'; winner: 0 | 1 | -1 }
  | { type: 'batterUp'; batter: string; pitcher: string };
