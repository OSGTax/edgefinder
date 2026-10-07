/**
 * Compact text notation for the step sequencer. One step = one 16th note.
 *
 *   notes('C5/2 E5/2 ./4 C4+E4+G4/8 | …')   pitch(es)/length, '.' = rest
 *   beat('x...X...o...x...')                x hit, X accent, o ghost, . rest
 *   bassLine('G G | Em Em', [[0, 3], [7, 2], [12, 3]])   pattern per chord slot
 *   comp('G G | Em Em', '..x...x.')         chord stabs per chord slot
 *
 * '|' marks bar lines; every bar must add up to 16 steps (checked at load).
 */
import type { Inst } from './instruments';

export interface Ev {
  step: number;
  midis: number[];
  len: number;
  vel: number;
}

export interface Seq {
  len: number;
  evs: Ev[];
}

export interface Part {
  inst: Inst;
  gain: number;
  len: number;
  /** Events indexed by step within the part (parts loop independently). */
  steps: Array<Ev[] | undefined>;
}

export interface Song {
  stepDur: number;
  /** Overall track level (0..1), so background tracks can sit lower than menu tracks. */
  level: number;
  /** Fraction of a step to delay off-beat 8ths by. */
  swing: number;
  loop: boolean;
  /** One-shots only: hand back to the loop that was playing when this ends. */
  then: boolean;
  /** 0..1: how loosely the melodic parts are played (0 = on the grid). */
  feel: number;
  length: number;
  parts: Part[];
}

const PC: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const accidental = (a: string): number => (a === '#' ? 1 : a === 'b' ? -1 : 0);

export function midi(name: string): number {
  const m = /^([A-G])([#b]?)(\d)$/.exec(name);
  if (!m) throw new Error(`audio: bad note "${name}"`);
  return PC[m[1]] + accidental(m[2]) + (Number(m[3]) + 1) * 12;
}

function checkBar(src: string, bar: string, steps: number): void {
  if (src.includes('|') && steps !== 16) {
    throw new Error(`audio: bar "${bar.trim()}" has ${steps} steps, expected 16`);
  }
}

export function notes(src: string, vel = 1): Seq {
  const evs: Ev[] = [];
  let step = 0;
  for (const bar of src.split('|')) {
    const start = step;
    for (const tok of bar.trim().split(/\s+/).filter(Boolean)) {
      const [pitch, lenStr] = tok.split('/');
      const len = Number(lenStr);
      if (!pitch || !Number.isInteger(len) || len <= 0) throw new Error(`audio: bad token "${tok}"`);
      if (pitch !== '.') evs.push({ step, midis: pitch.split('+').map((n) => midi(n)), len, vel });
      step += len;
    }
    checkBar(src, bar, step - start);
  }
  return { len: step, evs };
}

const HIT: Record<string, number> = { X: 1, x: 0.7, o: 0.35, '.': 0 };

export function beat(src: string): Seq {
  const evs: Ev[] = [];
  let step = 0;
  for (const bar of src.split('|')) {
    const cells = bar.replace(/\s+/g, '');
    for (const c of cells) {
      const vel = HIT[c];
      if (vel === undefined) throw new Error(`audio: bad drum cell "${c}"`);
      if (vel > 0) evs.push({ step, midis: [], len: 1, vel });
      step++;
    }
    checkBar(src, bar, cells.length);
  }
  return { len: step, evs };
}

const QUALITY: Record<string, number[]> = {
  '': [0, 4, 7],
  m: [0, 3, 7],
  '7': [0, 4, 7, 10],
  m7: [0, 3, 7, 10],
  maj7: [0, 4, 7, 11],
};

interface Chord {
  pc: number;
  intervals: number[];
}

/** One chord symbol per slot (half a bar by default); '|' is just for readability. */
function progression(src: string): Chord[] {
  return src
    .split(/[\s|]+/)
    .filter(Boolean)
    .map((sym) => {
      const m = /^([A-G])([#b]?)(.*)$/.exec(sym);
      const intervals = m ? QUALITY[m[3]] : undefined;
      if (!m || !intervals) throw new Error(`audio: bad chord "${sym}"`);
      return { pc: (PC[m[1]] + accidental(m[2]) + 12) % 12, intervals };
    });
}

/** Bass: pattern entries are [semitones above the root | null for rest, length]. Roots sit in F2..E3. */
export function bassLine(
  chords: string,
  pattern: ReadonlyArray<readonly [number | null, number]>,
  slot = 8,
): Seq {
  const evs: Ev[] = [];
  let step = 0;
  for (const c of progression(chords)) {
    const root = 41 + ((c.pc - 5 + 12) % 12);
    let s = 0;
    for (const [iv, len] of pattern) {
      if (iv !== null) evs.push({ step: step + s, midis: [root + iv], len, vel: 1 });
      s += len;
    }
    if (s !== slot) throw new Error(`audio: bass pattern spans ${s} steps, slot is ${slot}`);
    step += slot;
  }
  return { len: step, evs };
}

/** A chord voiced inside the octave starting at `low` (default F3). */
function voice(c: Chord, low: number): number[] {
  return c.intervals.map((iv) => low + ((c.pc + iv - (low % 12) + 24) % 12)).sort((a, b) => a - b);
}

/**
 * Chord stabs voiced inside the octave from `low` (default F3..E4), struck
 * wherever `rhythm` has an x (X = accent, o = ghost; rhythm = one slot).
 */
export function comp(chords: string, rhythm: string, len = 2, low = 53): Seq {
  const evs: Ev[] = [];
  let step = 0;
  for (const c of progression(chords)) {
    const voicing = voice(c, low);
    for (let i = 0; i < rhythm.length; i++) {
      const vel = HIT[rhythm[i]] ?? 0;
      if (vel > 0) evs.push({ step: step + i, midis: voicing, len, vel: rhythm[i] === 'x' ? 1 : vel });
    }
    step += rhythm.length;
  }
  return { len: step, evs };
}

/**
 * Picked arpeggios: `pattern` is one slot long; digits pick chord tones from
 * the bottom of the voicing (4+ wrap up an octave), '.' rests.
 */
export function arp(chords: string, pattern: string, len = 2, low = 60): Seq {
  const evs: Ev[] = [];
  let step = 0;
  for (const c of progression(chords)) {
    const v = voice(c, low);
    for (let i = 0; i < pattern.length; i++) {
      const d = pattern[i];
      if (d === '.') continue;
      const k = Number(d);
      if (!Number.isInteger(k)) throw new Error(`audio: bad arp cell "${d}"`);
      evs.push({ step: step + i, midis: [v[k % v.length] + 12 * Math.floor(k / v.length)], len, vel: 1 });
    }
    step += pattern.length;
  }
  return { len: step, evs };
}

/** Lay sequences end to end (sections of a tune). */
export function chain(...seqs: Seq[]): Seq {
  const evs: Ev[] = [];
  let step = 0;
  for (const s of seqs) {
    for (const e of s.evs) evs.push({ ...e, step: e.step + step });
    step += s.len;
  }
  return { len: step, evs };
}

/** n bars of silence. */
export const rest = (bars: number): Seq => ({ len: bars * 16, evs: [] });

export interface SongSpec {
  bpm: number;
  level?: number;
  swing?: number;
  loop: boolean;
  then?: boolean;
  feel?: number;
  parts: Array<[Inst, number, Seq]>;
}

export function song(spec: SongSpec): Song {
  const parts = spec.parts.map(([inst, gain, seq]): Part => {
    const steps: Array<Ev[] | undefined> = new Array<Ev[] | undefined>(seq.len);
    for (const ev of seq.evs) (steps[ev.step] ??= []).push(ev);
    return { inst, gain, len: seq.len, steps };
  });
  return {
    stepDur: 60 / spec.bpm / 4,
    level: spec.level ?? 1,
    swing: spec.swing ?? 0,
    loop: spec.loop,
    then: !spec.loop && !!spec.then,
    feel: spec.feel ?? 0.5,
    length: Math.max(...parts.map((p) => p.len)),
    parts,
  };
}
