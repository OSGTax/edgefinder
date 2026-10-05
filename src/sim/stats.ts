export interface BatLine { pa: number; ab: number; h: number; d: number; t: number; hr: number; rbi: number; r: number; bb: number; so: number }
export interface PitchLine { outs: number; h: number; r: number; bb: number; so: number; hr: number; pitches: number }
export interface KidLine { bat: BatLine; pitch: PitchLine; e: number; side: 0 | 1 }

export const emptyBat = (): BatLine => ({ pa: 0, ab: 0, h: 0, d: 0, t: 0, hr: 0, rbi: 0, r: 0, bb: 0, so: 0 });
export const emptyPitch = (): PitchLine => ({ outs: 0, h: 0, r: 0, bb: 0, so: 0, hr: 0, pitches: 0 });

export type BoxScore = Record<string, KidLine>;

export function addBat(a: BatLine, b: BatLine) {
  for (const k of Object.keys(b) as (keyof BatLine)[]) a[k] += b[k];
}
export function addPitch(a: PitchLine, b: PitchLine) {
  for (const k of Object.keys(b) as (keyof PitchLine)[]) a[k] += b[k];
}

export const ipString = (outs: number) => `${Math.floor(outs / 3)}.${outs % 3}`;
export const era = (p: PitchLine, innings = 6) => (p.outs === 0 ? (p.r > 0 ? 99.99 : 0) : (p.r * innings * 3) / p.outs);
