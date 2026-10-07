// Watches real frame times and decides when to change graphics tier.
// Pure logic (no DOM, no Three) so it can be tested with made-up frame times.
//
// Frames are judged in 2-second windows by their median and 90th percentile
// (single hitches — a shader compiling, a garbage collection — don't count).
// Two slow windows in a row step the whole tier down; at the bottom tier the
// resolution shrinks instead. Five smooth windows in a row grow the
// resolution back, then (where allowed) try the next tier up; a tier that
// turns out too slow is marked and never retried automatically.

export type GovernorAction = 'down' | 'up' | 'shrink' | 'grow';

export interface GovernorOptions {
  /** current tier index (0 = Fast) */
  tier: number;
  /** lowest and highest tiers it may pick */
  min: number;
  max: number;
  /** may change tiers at all (false when the player chose a tier: resolution only) */
  tiers: boolean;
}

export const WINDOW_S = 2;
/** seconds ignored after start and after every change (shaders compile, shadow maps reallocate) */
export const SETTLE_S = 2;
export const SCALES = [1, 0.85, 0.72, 0.6];

export class TierGovernor {
  tier: number;
  scaleIdx = 0;
  /** lowest tier index found too slow (null = none) */
  tooSlow: number | null = null;
  private frames: number[] = [];
  private winT = 0;
  private settle = SETTLE_S;
  private bad = 0;
  private good = 0;
  /** frame budget in ms (16.7 at 60 fps, 33.3 with the battery saver) */
  targetMs = 1000 / 60;

  constructor(private o: GovernorOptions) {
    this.tier = o.tier;
  }

  get scale() { return SCALES[this.scaleIdx]; }

  /** Change the frame target (battery saver on/off) and start measuring afresh. */
  setTarget(ms: number) {
    this.targetMs = ms;
    this.reset();
  }

  reset() {
    this.frames.length = 0;
    this.winT = 0;
    this.settle = SETTLE_S;
    this.bad = this.good = 0;
  }

  /** Feed one rendered frame's interval; returns an action when one is due. */
  frame(dtMs: number): GovernorAction | null {
    if (!(dtMs > 0) || dtMs > 250) return null; // tab hidden, breakpoint, loading hitch
    const dt = dtMs / 1000;
    if (this.settle > 0) { this.settle -= dt; return null; }
    this.frames.push(dtMs);
    this.winT += dt;
    if (this.winT < WINDOW_S) return null;
    const sorted = this.frames.slice().sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length * 0.5)];
    const p90 = sorted[Math.floor(sorted.length * 0.9)];
    this.frames.length = 0;
    this.winT = 0;
    const t = this.targetMs;
    if (med > t * 1.35) { this.bad++; this.good = 0; }
    else if (med < t * 1.1 && p90 < t * 1.35) { this.good++; this.bad = 0; }
    else { this.bad = 0; this.good = 0; }

    if (this.bad >= 2) {
      // a tier we just climbed to is too slow: go back and remember
      if (this.o.tiers && this.tier > this.o.min && this.scaleIdx === 0) {
        this.tooSlow = this.tooSlow === null ? this.tier : Math.min(this.tooSlow, this.tier);
        return this.act('down');
      }
      if (this.scaleIdx < SCALES.length - 1) return this.act('shrink');
      this.bad = 0;
      return null;
    }
    if (this.good >= 5) {
      if (this.scaleIdx > this.growFloor) return this.act('grow');
      const next = this.tier + 1;
      // under the battery saver every frame waits for the cap, so headroom can't be seen: never climb
      if (this.o.tiers && this.targetMs < 20 && next <= this.o.max && (this.tooSlow === null || next < this.tooSlow)) return this.act('up');
      this.good = 0;
    }
    return null;
  }

  /** a sharper resolution that proved too slow is not retried */
  private growFloor = 0;
  private lastAct: GovernorAction | null = null;

  private act(a: GovernorAction): GovernorAction {
    if (a === 'shrink' && this.lastAct === 'grow') this.growFloor = this.scaleIdx + 1;
    this.lastAct = a;
    if (a === 'down') this.tier--;
    if (a === 'up') this.tier++;
    if (a === 'shrink') this.scaleIdx++;
    if (a === 'grow') this.scaleIdx--;
    this.reset();
    return a;
  }
}
