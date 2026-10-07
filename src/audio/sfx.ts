/**
 * Every sound effect, synthesized on demand.
 *
 * The sounds are built from what's physically going on: a wooden bat rings
 * at a handful of modes and the house behind home plate throws a little of
 * the crack back; the ball sounds different on grass, dirt and the pool
 * deck; the crowd is kids on the bench plus the neighbours in lawn chairs.
 * The stingers are played on the same toy xylophone and glockenspiel the
 * kids use in the music, so nothing sounds like a stock game "blip".
 */
import { crack } from './bake';
import { getBus, isMuted, isReady } from './context';
import {
  FLOOR,
  Patch,
  clamp,
  env,
  glide,
  hits,
  mtof,
  path,
  pick,
  rand,
  softClip,
  spread,
  wire,
} from './dsp';
import { INSTRUMENTS, type Inst } from './instruments';
import { beat, notes, type Seq } from './notation';
import type { PlayOpts, SfxName } from './types';

export interface Hit {
  p: Patch;
  out: AudioNode;
  t: number;
  /** intensity 0..1 */
  i: number;
  pan: number;
  panner: StereoPannerNode | null;
}

export type SfxFn = (h: Hit) => void;

/** Drift the sound across the stereo field around its base pan. */
function panSweep(h: Hit, from: number, to: number, dur: number): void {
  if (!h.panner) return;
  h.panner.pan.setValueAtTime(clamp(h.pan + from, -1, 1), h.t);
  h.panner.pan.linearRampToValueAtTime(clamp(h.pan + to, -1, 1), h.t + dur);
}

/** Damped partials: [frequency, relative level, decay seconds]. */
function modes(p: Patch, out: AudioNode, t: number, a: number, list: ReadonlyArray<readonly [number, number, number]>): void {
  for (const [f, amp, dec] of list) p.tone(out, { t, freq: f, to: f * 0.985, peak: a * amp, decay: dec });
}

const { xylo, glock, kazoo } = INSTRUMENTS;

// ── contact ──────────────────────────────────────────────────────────────────

let crackN = 0;
const batCrack: SfxFn = ({ p, out, t, i }) => {
  const a = 0.45 + 0.5 * i;
  // the crack itself is baked (see bake.ts) so every swing peaks the same; the hit's
  // strength decides how bright it is, and a crush gets weight and air on top
  const lp = p.filter('lowpass', 2600 + 8000 * i * i, 0);
  const g = p.gain(a);
  wire(p.play(crack(p.ctx, crackN++), t, rand(0.97, 1.03)), lp, g, out);
  if (i > 0.55) {
    const k = (i - 0.55) / 0.45;
    p.tone(out, { t, freq: 175, to: 70, glide: 0.09, peak: 0.36 * k, decay: 0.14 });
    p.burst(out, { t: t + 0.008, attack: 0.015, peak: 0.12 * k, decay: 0.3 + 0.4 * k, freq: 3000, to: 1100, q: 0.8 });
  }
};

const batTink: SfxFn = ({ p, out, t, i }) => {
  const a = 0.3 + 0.35 * i;
  // off the end of the bat or the handle: dull, with a stinging buzz
  p.tone(out, { t, type: 'triangle', freq: 330, to: 170, glide: 0.05, peak: a, decay: 0.08 });
  p.burst(out, { t, peak: a * 0.45, decay: 0.016, freq: 2800, q: 2.5 });
  modes(p, out, t, a * 0.5, [[480, 0.4, 0.05], [905, 0.3, 0.03]]);
  const g = p.gain(0);
  const end = env(g.gain, t + 0.004, a * 0.07, 0.002, 0.07);
  wire(p.osc('square', 172, t, end + 0.02), p.filter('bandpass', 1300, 3), g, out); // hands buzzing
};

const whiff: SfxFn = (h) => {
  const { p, out, t, i } = h;
  const f = p.burst(out, { t, attack: 0.1, peak: 0.3 + 0.25 * i, decay: 0.2, q: 1.3 });
  path(f.frequency, [[t, 420], [t + 0.11, 1900], [t + 0.32, 650]]);
  panSweep(h, -0.3, 0.3, 0.3);
};

const throwWhoosh: SfxFn = (h) => {
  const { p, out, t, i } = h;
  const f = p.burst(out, { t, attack: 0.05, peak: 0.25 + 0.35 * i, decay: 0.15, q: 1.8 });
  path(f.frequency, [[t, 700], [t + 0.06, 2900], [t + 0.2, 1300]]);
  p.burst(out, { t, attack: 0.04, peak: 0.08 + 0.08 * i, decay: 0.09, filter: 'highpass', freq: 5000 });
  panSweep(h, -0.45, 0.45, 0.2);
};

const mittPop: SfxFn = ({ p, out, t, i }) => {
  const a = 0.5 + 0.45 * i;
  const v = rand(0.95, 1.05);
  const thump = p.tone(out, { t, freq: 200 * v, peak: a * 0.6, decay: 0.13 });
  path(thump.frequency, [[t, 200 * v], [t + 0.025, 118], [t + 0.13, 68]]);
  // the "pok" of the pocket: mid-range, so it still lands on phone speakers
  p.tone(out, { t, type: 'triangle', freq: 560 * v, to: 260, glide: 0.04, peak: a * 0.85, decay: 0.06 });
  p.burst(out, { t, peak: a * 0.9, decay: 0.045, filter: 'lowpass', freq: 1500, q: 0.7 }); // leather slap
  p.burst(out, { t, peak: a * 0.7, decay: 0.02, freq: 850 * v, q: 1.5 }); // pop
  p.burst(out, { t: t + 0.002, peak: a * 0.15, decay: 0.01, filter: 'highpass', freq: 3500 }); // laces
};

const glovePop: SfxFn = ({ p, out, t, i }) => {
  const a = 0.3 + 0.35 * i;
  const v = rand(0.93, 1.07);
  const thump = p.tone(out, { t, freq: 240 * v, peak: a * 0.5, decay: 0.09 });
  path(thump.frequency, [[t, 240 * v], [t + 0.02, 150], [t + 0.09, 100]]);
  p.tone(out, { t, type: 'triangle', freq: 440 * v, to: 240, glide: 0.035, peak: a * 0.9, decay: 0.05 });
  p.burst(out, { t, peak: a, decay: 0.035, filter: 'lowpass', freq: 1200, q: 0.7 });
};

// ── the ball on the ground ───────────────────────────────────────────────────

const bounce: SfxFn = ({ p, out, t, i }) => {
  // on the lawn: a soft "thup" and the blades brushing
  const a = 0.15 + 0.6 * i;
  p.tone(out, { t, freq: 90 + 50 * i, to: 48, glide: 0.08, peak: a * 0.45, decay: 0.1 });
  p.tone(out, { t, type: 'triangle', freq: 330 + 80 * i, to: 170, glide: 0.05, peak: a * 0.9, decay: 0.07 });
  p.burst(out, { t, peak: a * 0.8, decay: 0.03, freq: 700, q: 0.8 });
  p.burst(out, { t: t + 0.004, attack: 0.01, peak: a * 0.14, decay: 0.07, filter: 'highpass', freq: 3500 });
};

const bounceDirt: SfxFn = ({ p, out, t, i }) => {
  // on the base paths: a duller thud with grit kicked up
  const a = 0.15 + 0.6 * i;
  p.tone(out, { t, type: 'triangle', freq: 260 + 60 * i, to: 120, glide: 0.04, peak: a * 0.9, decay: 0.06 });
  p.burst(out, { t, peak: a * 0.7, decay: 0.025, freq: 900, q: 0.9 });
  const grit = spread(t + 0.005, t + 0.06 + 0.06 * i, 6 + Math.round(6 * i), 0.006, 1.6).map(
    (x): [number, number] => [x, a * rand(0.1, 0.35)],
  );
  const g = p.gain(0);
  const end = hits(g.gain, grit, 0.001, 0.008);
  wire(p.noise(t, end + 0.02), p.filter('bandpass', 2600, 1.2), g, out);
};

const bouncePatio: SfxFn = ({ p, out, t, i }) => {
  // on the pool deck: a hard, bright "tock" with a little ring off the concrete
  const a = 0.18 + 0.6 * i;
  const v = rand(0.95, 1.05);
  p.burst(out, { t, peak: a * 0.8, decay: 0.004, filter: 'highpass', freq: 2500 });
  p.tone(out, { t, type: 'triangle', freq: 980 * v, to: 760, glide: 0.02, peak: a * 0.8, decay: 0.035 });
  modes(p, out, t, a, [[1850 * v, 0.25, 0.04], [2950 * v, 0.12, 0.03]]);
  p.tone(out, { t, freq: 210, to: 120, glide: 0.03, peak: a * 0.35, decay: 0.05 });
};

// ── the yard ─────────────────────────────────────────────────────────────────

/** Loose boards rattling after a knock, slowing and fading. */
function rattle(p: Patch, out: AudioNode, t: number, a: number, n: number, freq: number, gap0 = 0.032): void {
  const list: Array<[number, number]> = [];
  let x = t;
  let gap = gap0;
  let pk = a;
  for (let k = 0; k < n; k++) {
    list.push([x, pk * rand(0.6, 1)]);
    x += gap * rand(0.8, 1.25);
    gap *= 1.08;
    pk *= 0.8;
  }
  const g = p.gain(0);
  const end = hits(g.gain, list, 0.002, 0.025);
  wire(p.noise(t, end + 0.02), p.filter('bandpass', freq, 2.2), g, out);
}

const fence: SfxFn = ({ p, out, t, i }) => {
  // a solid board fence: hollow knock, two damped modes, a short rattle
  const a = 0.4 + 0.45 * i;
  p.tone(out, { t, type: 'triangle', freq: 235, to: 200, peak: a * 0.8, decay: 0.17 });
  p.tone(out, { t, freq: 560, to: 520, peak: a * 0.4, decay: 0.09 });
  p.burst(out, { t, peak: a * 0.5, decay: 0.02, freq: 1700 });
  rattle(p, out, t + 0.03, a * 0.4, 7 + Math.round(3 * i), 1250);
};

const picket: SfxFn = ({ p, out, t, i }) => {
  // the short white picket fence in right: a lighter, higher clack and every slat chattering
  const a = 0.4 + 0.45 * i;
  const v = rand(0.95, 1.05);
  p.tone(out, { t, type: 'triangle', freq: 410 * v, to: 360, peak: a * 0.7, decay: 0.09 });
  modes(p, out, t, a, [[930 * v, 0.35, 0.05], [1620 * v, 0.2, 0.03]]);
  p.burst(out, { t, peak: a * 0.55, decay: 0.012, freq: 2400, q: 1.2 });
  rattle(p, out, t + 0.02, a * 0.45, 10 + Math.round(4 * i), 1900, 0.024);
};

const hedge: SfxFn = ({ p, out, t, i }) => {
  // into the hedge: no knock at all, just a soft thump swallowed by leaves and twigs
  const a = 0.35 + 0.45 * i;
  p.tone(out, { t, type: 'triangle', freq: 210, to: 120, glide: 0.06, peak: a * 0.45, decay: 0.08 });
  p.burst(out, { t, attack: 0.02, peak: a * 0.3, decay: 0.35, freq: 2100, q: 0.6 });
  const twigs = spread(t, t + 0.4, 14, 0.015, 1.8).map((x): [number, number] => [x, a * rand(0.12, 0.4)]);
  const g = p.gain(0);
  const end = hits(g.gain, twigs, 0.002, 0.02);
  wire(p.noise(t, end + 0.02), p.filter('bandpass', 3800, 1.1), g, out);
};

const houseWall: SfxFn = ({ p, out, t, i }) => {
  // off the back of the house: a big hollow "bonk" on the siding, and the kitchen window buzzes
  const a = 0.45 + 0.45 * i;
  p.tone(out, { t, type: 'triangle', freq: 150, to: 118, peak: a * 0.9, decay: 0.2 });
  p.tone(out, { t, freq: 330, to: 300, peak: a * 0.45, decay: 0.12 });
  p.burst(out, { t, peak: a * 0.6, decay: 0.025, freq: 1100, q: 0.9 });
  const g = p.gain(0);
  const end = env(g.gain, t + 0.01, a * 0.06, 0.004, 0.18);
  const hp = p.filter('bandpass', 3600, 4);
  wire(hp, g, out);
  for (const f of [3410, 3655, 4120]) wire(p.osc('square', f, t, end + 0.02), hp);
  p.lfo(g.gain, 46, a * 0.03, t, end + 0.02); // the pane buzzing in its frame
};

const splash: SfxFn = ({ p, out, t, i }) => {
  const a = 0.45 + 0.45 * i;
  p.tone(out, { t, freq: 280, to: 95, glide: 0.09, peak: a * 0.55, decay: 0.12 }); // plunk
  // the sheet of water
  p.burst(out, { t, attack: 0.006, hold: 0.04, peak: a * 0.6, decay: 0.45 + 0.35 * i, freq: 3400, to: 900, q: 0.7 });
  // spray falling back
  const drops = spread(t + 0.08, t + 0.75, 10, 0.03).map(
    (x, k): [number, number] => [x, a * 0.25 * rand(0.4, 1) * (1 - k / 12)],
  );
  const g = p.gain(0);
  const end = hits(g.gain, drops, 0.002, 0.02);
  wire(p.noise(t, end + 0.02), p.filter('highpass', 3800, 0.7), g, out);
  // bubbles: little rising blips
  for (let k = 0; k < 3; k++) {
    p.tone(out, {
      t: t + rand(0.12, 0.55),
      attack: 0.004,
      freq: rand(450, 750),
      to: rand(1300, 2100),
      glide: 0.05,
      peak: a * 0.12,
      decay: 0.06,
    });
  }
  // the waves slapping the side of the pool afterwards
  const laps = [0.9, 1.25, 1.55].map((x): [number, number] => [t + x + rand(0, 0.08), a * rand(0.08, 0.14)]);
  const lg = p.gain(0);
  const lend = hits(lg.gain, laps, 0.02, 0.12);
  wire(p.noise(t, lend + 0.02), p.filter('bandpass', 900, 1.4), lg, out);
};

const leaves: SfxFn = ({ p, out, t, i }) => {
  const a = 0.35 + 0.45 * i;
  const dur = 0.5 + 0.4 * i;
  p.burst(out, { t, peak: a * 0.45, decay: 0.012, filter: 'highpass', freq: 2500 }); // twig snap
  p.tone(out, { t, type: 'triangle', freq: 1800, to: 900, peak: a * 0.12, decay: 0.025 });
  p.burst(out, { t, attack: 0.04, peak: a * 0.18, decay: dur, freq: 2200, q: 0.5 }); // soft swish
  // rustle: two layers of crackly grains, densest right at impact
  for (const [freq, n] of [[3200, 18], [6000, 12]] as const) {
    const grains = spread(t, t + dur, n, 0.018, 1.8).map(
      (x): [number, number] => [x, a * rand(0.15, 0.5) * (1 - (x - t) / (dur * 1.15))],
    );
    const g = p.gain(0);
    const end = hits(g.gain, grains, 0.003, 0.03);
    wire(p.noise(t, end + 0.02), p.filter('bandpass', freq, 1.1), g, out);
  }
};

const dogBark: SfxFn = ({ p, out, t, i }) => {
  const a = 0.5 + 0.4 * i;
  // a shared "throat": soft-clipped growl into a mouth formant pair
  const drive = p.gain(1);
  const throat = p.shaper(softClip());
  const f1 = p.filter('bandpass', 900, 4);
  const f2 = p.filter('bandpass', 1750, 6);
  const f2g = p.gain(0.6);
  const post = p.gain(0.7);
  wire(drive, throat);
  wire(throat, f1, post);
  wire(throat, f2, f2g, post);
  post.connect(out);
  // "ruff-ruff": the second bark a little lower; sometimes a third, softer one
  const barks = Math.random() < 0.35 ? [0, 0.24, 0.5] : [0, 0.24];
  barks.forEach((dt, k) => {
    const tb = t + dt + (k ? rand(-0.02, 0.03) : 0);
    const s = k ? 0.9 - 0.04 * k : 1;
    const g = p.gain(0);
    const end = env(g.gain, tb, 1.6 * a * (k === 2 ? 0.6 : 1), 0.008, 0.12, 0.03);
    const o = p.osc('sawtooth', 300 * s, tb, end + 0.02);
    path(o.frequency, [[tb, 300 * s], [tb + 0.025, 540 * s], [tb + 0.16, 250 * s]]);
    wire(o, g, drive);
    glide(f1.frequency, tb, 1000 * s, 520 * s, 0.15); // "wuh" closing to "oof"
    p.burst(out, { t: tb, attack: 0.004, peak: a * 0.25, decay: 0.07, freq: 1300, q: 1.2 });
  });
};

const screenDoor: SfxFn = ({ p, out, t, i }) => {
  const a = 0.35 + 0.4 * i;
  // the hinge creaks open: stick-slip friction, a pulse train that speeds up and slows down
  const creak = p.osc('sawtooth', 60, t, t + 0.6);
  path(creak.frequency, [[t, 48], [t + 0.22, 125], [t + 0.42, 95], [t + 0.56, 70]]);
  const cg = p.gain(0);
  env(cg.gain, t, a * 0.5, 0.06, 0.4, 0.1);
  const body = p.gain(1);
  wire(creak, cg, body);
  wire(body, p.filter('bandpass', 950, 9), out);
  wire(body, p.filter('bandpass', 2100, 12), p.gain(0.6), out);
  // ...somebody goes through, and the spring yanks it shut
  const shut = t + rand(0.95, 1.2);
  const spring = p.tone(out, { t: shut - 0.12, freq: 210, to: 170, peak: a * 0.18, decay: 0.3 });
  p.lfo(spring.frequency, 13, 25, shut - 0.12, shut + 0.2);
  p.tone(out, { t: shut - 0.12, freq: 2350, peak: a * 0.05, decay: 0.25 });
  // WHAP, and the little second bounce of the frame
  for (const [dt, k] of [[0, 1], [0.085, 0.45]] as const) {
    const tt = shut + dt;
    p.tone(out, { t: tt, type: 'triangle', freq: 185, to: 140, peak: a * 0.9 * k, decay: 0.09 });
    p.tone(out, { t: tt, freq: 420, peak: a * 0.35 * k, decay: 0.05 });
    p.burst(out, { t: tt, peak: a * 0.7 * k, decay: 0.03, freq: 1400, q: 0.8 });
  }
  rattle(p, out, shut + 0.02, a * 0.15, 5, 3200, 0.02); // the screen mesh
};

// ── the crowd: kids on the bench and the neighbours in lawn chairs ───────────

interface Crowd {
  dur: number;
  /** kids' voices (high) */
  kids: number;
  /** grown-ups' voices (low) */
  adults: number;
  breaths: number;
  claps: number;
  /** how far voice entries are staggered (s) */
  stagger: number;
  level: number;
  kind: 'cheer' | 'aww' | 'ooh';
}

type Contour = 'yay' | 'woo' | 'aww' | 'ooh' | 'hey';

function contour(kind: Contour, t: number, len: number, f0: number): Array<[number, number]> {
  switch (kind) {
    case 'yay':
      return [[t, f0 * 0.8], [t + len * 0.18, f0 * 1.22], [t + len * 0.6, f0 * rand(1, 1.12)], [t + len, f0 * 0.82]];
    case 'woo':
      return [[t, f0 * 0.85], [t + len * 0.35, f0 * 1.5], [t + len * 0.7, f0 * 1.35], [t + len, f0 * 0.95]];
    case 'hey':
      return [[t, f0 * 1.1], [t + 0.08, f0 * 1.3], [t + len, f0 * 0.9]];
    case 'aww':
      return [[t, f0 * 1.12], [t + 0.1, f0 * 1.18], [t + len, f0 * 0.68]];
    case 'ooh':
      return [[t, f0 * 0.9], [t + len * 0.7, f0 * 1.25], [t + len, f0 * 1.15]];
  }
}

/** Excited shouting: loudness bobbing syllable-to-syllable, then trailing off. */
function syllables(g: AudioParam, t: number, len: number, peak: number): void {
  g.setValueAtTime(FLOOR, t);
  g.linearRampToValueAtTime(peak, t + 0.05);
  const stop = t + len * 0.75;
  let x = t + 0.05;
  let up = true;
  for (;;) {
    x += rand(0.08, 0.16);
    if (x >= stop) break;
    up = !up;
    g.linearRampToValueAtTime(peak * (up ? rand(0.85, 1) : rand(0.4, 0.65)), x);
  }
  g.exponentialRampToValueAtTime(FLOOR, t + len);
}

function crowd({ p, out, t }: Hit, c: Crowd): void {
  const end = t + c.dur;
  const aww = c.kind === 'aww';
  const ooh = c.kind === 'ooh';
  // one shared vowel filter for all the pitched voices: body + two formants
  const vox = p.gain(1);
  const f1 = p.filter('bandpass', aww ? 780 : ooh ? 480 : 1050, 3);
  const f2 = p.filter('bandpass', ooh ? 950 : 2000, 5);
  wire(vox, p.filter('lowpass', 1300, 0.7), p.gain(0.35), out);
  wire(vox, f1, out);
  wire(vox, f2, p.gain(0.7), out);
  if (aww) glide(f2.frequency, t, 1250, 950, c.dur); // "aww" darkens as it sinks
  else if (ooh) glide(f2.frequency, t, 800, 1100, c.dur);
  else glide(f2.frequency, t, 1800, 2700, c.dur * 0.5); // "yaay": eh -> ee

  const voices = c.kids + c.adults;
  const vPeak = c.level / Math.sqrt(Math.max(1, voices));
  for (let v = 0; v < voices; v++) {
    const adult = v >= c.kids;
    const ts = t + rand(0, c.stagger);
    const len = Math.max(0.3, Math.min(end - ts, c.dur * rand(0.55, 0.9)));
    const f0 = adult ? rand(120, 230) : rand(330, 620);
    const osc = p.osc(v % 3 === 2 ? 'triangle' : 'sawtooth', f0, ts, ts + len + 0.02);
    const kind: Contour = aww ? 'aww' : ooh ? 'ooh' : adult ? (Math.random() < 0.5 ? 'hey' : 'yay') : Math.random() < 0.6 ? 'yay' : 'woo';
    path(osc.frequency, contour(kind, ts, len, f0));
    p.lfo(osc.frequency, rand(5, 8), f0 * rand(0.02, 0.04), ts, ts + len + 0.02); // vibrato
    const g = p.gain(0);
    const pk = vPeak * (adult ? 1.25 : 1); // low voices need a little more to be heard
    if (aww || ooh) env(g.gain, ts, pk, ooh ? 0.25 : 0.14, Math.max(0.1, len * 0.65 - 0.14), len * 0.3);
    else syllables(g.gain, ts, len, pk);
    wire(osc, g, vox);
  }

  // breathy shouting: formant-filtered noise with a fast chatter tremolo
  for (let n = 0; n < c.breaths; n++) {
    const am = p.gain(0.6);
    p.lfo(am.gain, rand(6, 11), 0.4, t, end + 0.02);
    const g = p.gain(0);
    const attack = aww || ooh ? 0.2 : 0.06;
    env(g.gain, t, c.level * 0.35, attack, c.dur * 0.72 - attack, c.dur * 0.25);
    const bp = p.filter('bandpass', aww ? rand(600, 900) : ooh ? rand(450, 700) : rand(1000, 1700), 1.8);
    wire(p.noise(t, end + 0.02), bp, am, g, out);
  }

  // claps: two streams so neighbouring hands can overlap
  if (c.claps > 0) {
    const bp = p.filter('bandpass', 1500, 1.2);
    bp.connect(out);
    for (let s = 0; s < 2; s++) {
      const list = spread(t + 0.05, end - 0.25, Math.ceil(c.claps / 2), 0.07).map(
        (x): [number, number] => [x, c.level * rand(0.35, 0.7)],
      );
      const g = p.gain(0);
      const e = hits(g.gain, list, 0.002, 0.045);
      wire(p.noise(t, e + 0.02), g, bp);
    }
  }
}

/** A two-finger whistle from somebody's dad. */
function dadWhistle(p: Patch, out: AudioNode, t: number, level: number): void {
  const g = p.gain(0);
  const end = env(g.gain, t, level, 0.04, 0.2, 0.5);
  const o = p.osc('sine', 1800, t, end + 0.02);
  path(o.frequency, [[t, 1800], [t + 0.15, 3100], [t + 0.6, 2900], [t + 0.75, 2300]]);
  wire(o, g, out);
}

// intensity is how big the moment is: a routine single is ~0.3, a lead-changing hit late ~0.9
const cheer: SfxFn = (h) => {
  const i = h.i;
  crowd(h, {
    dur: 1 + 0.6 * i,
    kids: 3 + Math.round(4 * i),
    adults: i > 0.35 ? 1 + Math.round(2 * i) : 0,
    breaths: 1 + Math.round(i),
    claps: 4 + Math.round(10 * i),
    stagger: 0.1 + 0.15 * i,
    level: 0.35 + 0.2 * i,
    kind: 'cheer',
  });
  if (i > 0.75) dadWhistle(h.p, h.out, h.t + 0.25, 0.05);
};

const bigCheer: SfxFn = (h) => {
  const i = h.i;
  crowd(h, { dur: 2.3 + 0.5 * i, kids: 7 + Math.round(3 * i), adults: 3, breaths: 3, claps: 18 + Math.round(8 * i), stagger: 0.8, level: 0.6, kind: 'cheer' });
  dadWhistle(h.p, h.out, h.t + 0.3, 0.07);
};

const aww: SfxFn = (h) =>
  crowd(h, {
    dur: 1.1 + 0.5 * h.i,
    kids: 3 + Math.round(3 * h.i),
    adults: h.i > 0.4 ? 2 : 1,
    breaths: 2,
    claps: 0,
    stagger: 0.12,
    level: 0.3 + 0.2 * h.i,
    kind: 'aww',
  });

const ooh: SfxFn = (h) =>
  crowd(h, { dur: 1.2 + 0.4 * h.i, kids: 4 + Math.round(3 * h.i), adults: 2, breaths: 2, claps: 0, stagger: 0.15, level: 0.3 + 0.2 * h.i, kind: 'ooh' });

const giggle: SfxFn = ({ p, out, t, i }) => {
  // two kids on the bench trying not to laugh: "hee-hee-hee" falling in pitch
  for (let k = 0; k < 2; k++) {
    const t0 = t + k * rand(0.05, 0.18);
    const f0 = rand(420, 620);
    const n = 4 + Math.floor(Math.random() * 3);
    const step = rand(0.085, 0.11);
    const osc = p.osc('sawtooth', f0, t0, t0 + n * step + 0.1);
    path(osc.frequency, [[t0, f0 * 1.15], [t0 + n * step, f0 * 0.8]]);
    const g = p.gain(0);
    const list = Array.from({ length: n }, (_, j): [number, number] => [t0 + j * step, (0.12 + 0.1 * i) * (1 - j * 0.1)]);
    hits(g.gain, list, 0.012, 0.06);
    const f1 = p.filter('bandpass', 1300, 3);
    const f2 = p.filter('bandpass', 2700, 5);
    wire(osc, g);
    wire(g, f1, out);
    wire(g, f2, p.gain(0.6), out);
  }
};

// ── stingers & jingles ───────────────────────────────────────────────────────
// Played on the kids' own toy xylophone and glockenspiel.

const strike: SfxFn = ({ p, out, t }) => {
  xylo(p, out, t, [88], 0.06, 0.3); // E6
  xylo(p, out, t + 0.09, [93], 0.1, 0.34); // A6
};

const outSting: SfxFn = ({ p, out, t }) => {
  // down the xylophone, and the mallet bounces on the last bar
  xylo(p, out, t, [79], 0.07, 0.32); // G5
  xylo(p, out, t + 0.1, [75], 0.07, 0.32); // Eb5
  xylo(p, out, t + 0.2, [72], 0.1, 0.36); // C5
  xylo(p, out, t + 0.29, [72], 0.05, 0.12);
  xylo(p, out, t + 0.345, [72], 0.05, 0.05);
};

const safe: SfxFn = ({ p, out, t }) => {
  [84, 88, 91].forEach((m, k) => glock(p, out, t + k * 0.06, [m], 0.05, 0.22)); // C E G
  glock(p, out, t + 0.18, [96], 0.25, 0.2);
};

/** A slide whistle: up for a home run or a special, down for a big miss. */
function slideWhistle(p: Patch, out: AudioNode, t: number, from: number, to: number, dur: number, level: number): void {
  const g = p.gain(0);
  const end = env(g.gain, t, level, 0.03, 0.08, dur);
  const o = p.osc('sine', from, t, end + 0.02);
  path(o.frequency, [[t, from], [t + dur, to]]);
  p.lfo(o.frequency, 6, from * 0.012, t, end + 0.02);
  wire(o, g, out);
  const br = p.gain(0);
  env(br.gain, t, level * 0.6, 0.03, 0.08, dur);
  const bp = p.filter('bandpass', from, 5);
  glide(bp.frequency, t, from, to, dur);
  wire(p.noise(t, end + 0.02), bp, br, out);
}

const HR_STEP = 0.085;
const HR_START = 0.42; // after the slide whistle
const HOME_RUN: Array<[Inst, number, Seq]> = [
  ['kazoo', 0.3, notes('C5/1 E5/1 G5/1 C6/3 A5/1 C6/1 D6/2 E6/6')],
  ['kazoo', 0.16, notes('E4/1 G4/1 C5/1 E5/3 F5/1 A5/1 B5/2 G5/6')],
  ['ukeBass', 0.5, notes('C3/3 G3/3 F3/4 C3/6')],
  ['uke', 0.32, notes('./6 F4+A4+D5/4 C4+E4+G4+C5/6')],
  ['box', 0.55, beat('x.....x...x.....')],
  ['claps', 0.2, beat('......x.x.x.x...')],
  ['lid', 0.6, beat('..........X.....')],
];

const homeRun: SfxFn = ({ p, out, t }) => {
  slideWhistle(p, out, t, 520, 1900, 0.36, 0.2);
  const t0 = t + HR_START;
  for (const [inst, gain, seq] of HOME_RUN) {
    for (const ev of seq.evs) {
      INSTRUMENTS[inst](p, out, t0 + ev.step * HR_STEP, ev.midis, ev.len * HR_STEP * 0.9, ev.vel * gain);
    }
  }
  [96, 100, 103, 108].forEach((m, k) => glock(p, out, t0 + 10 * HR_STEP + 0.05 + k * 0.07, [m], 0.1, 0.1));
};

const special: SfxFn = ({ p, out, t, i }) => {
  const a = 0.35 + 0.3 * i;
  // somebody's slide whistle winding all the way up...
  slideWhistle(p, out, t, 380, 1600, 0.5, a * 0.4);
  // ...a little kazoo "ta-da"...
  kazoo(p, out, t + 0.52, [79], 0.08, a * 0.35);
  kazoo(p, out, t + 0.62, [84], 0.22, a * 0.4);
  // ...and glockenspiel sparkles on top
  const PINGS = [96, 98, 100, 103, 105, 108];
  for (let s = 0; s < 2; s++) {
    const times = spread(t + 0.3 + s * 0.04, t + 0.95, 5, 0.07);
    const g = p.gain(0);
    const end = hits(g.gain, times.map((x): [number, number] => [x, a * rand(0.12, 0.22)]), 0.003, 0.12);
    const o = p.osc('sine', mtof(96), t, end + 0.02);
    for (const x of times) o.frequency.setValueAtTime(mtof(pick(PINGS)), x);
    wire(o, g, out);
  }
};

const whistle: SfxFn = ({ p, out, t }) => {
  // Coach Toby's pea whistle: "fweet... FWEEEET"
  for (const [dt, hold, f] of [[0, 0.12, 2650], [0.3, 0.42, 2750]] as const) {
    const t0 = t + dt;
    const g = p.gain(0);
    const end = env(g.gain, t0, 0.26, 0.02, 0.06, hold);
    const o = p.osc('sine', f, t0, end + 0.02);
    path(o.frequency, [[t0, f * 0.9], [t0 + 0.03, f], [end, f * 0.97]]);
    p.lfo(o.frequency, rand(24, 30), 70, t0, end + 0.02); // the pea rattling
    wire(o, g, out);
    p.burst(out, { t: t0, attack: 0.02, hold, peak: 0.04, decay: 0.06, freq: 2600 }); // breath
  }
};

// ── cartoon accents: used sparingly, for the comic moments ───────────────────

const boing: SfxFn = ({ p, out, t, i }) => {
  // a door-stop spring: a twangy "boi-oi-oing" that settles
  const a = 0.25 + 0.2 * i;
  const g = p.gain(0);
  const end = env(g.gain, t, a, 0.004, 0.55);
  const o = p.osc('triangle', 190, t, end + 0.02);
  path(o.frequency, [[t, 150], [t + 0.05, 330], [t + 0.5, 260]]);
  const wob = p.gain(0);
  wob.gain.setValueAtTime(70, t);
  wob.gain.exponentialRampToValueAtTime(4, t + 0.5);
  wire(p.osc('sine', 17, t, end + 0.02), wob);
  wob.connect(o.frequency);
  const bp = p.filter('bandpass', 900, 3);
  path(bp.frequency, [[t, 600], [t + 0.12, 1500], [t + 0.5, 1100]]); // "oi" opening to "ng"
  wire(o, bp, g, out);
  wire(o, p.gain(0.3), g);
};

const bonk: SfxFn = ({ p, out, t, i }) => {
  // a wood-block knock and the cartoon pitch drop after it
  const a = 0.35 + 0.3 * i;
  modes(p, out, t, a, [[820, 0.8, 0.06], [1980, 0.35, 0.03]]);
  p.burst(out, { t, peak: a * 0.4, decay: 0.008, freq: 2500, q: 1.2 });
  p.tone(out, { t: t + 0.02, type: 'triangle', freq: 620, to: 190, glide: 0.18, peak: a * 0.45, decay: 0.2 });
};

const zip: SfxFn = (h) => {
  // a quick zip-up whoosh: off like a shot
  const { p, out, t, i } = h;
  const a = 0.2 + 0.25 * i;
  p.tone(out, { t, freq: 320, to: 2400, glide: 0.13, peak: a * 0.4, decay: 0.14, attack: 0.01 });
  const f = p.burst(out, { t, attack: 0.02, peak: a, decay: 0.12, q: 2 });
  path(f.frequency, [[t, 800], [t + 0.13, 5000]]);
  panSweep(h, -0.4, 0.4, 0.15);
};

const dizzy: SfxFn = (h) => {
  // little stars circling a kid's head: twinkles going round and round
  const { p, out, t } = h;
  const notesUp = [96, 100, 103, 100, 96, 100, 103, 100];
  notesUp.forEach((m, k) => glock(p, out, t + k * 0.09, [m], 0.05, 0.1 * (1 - k * 0.07)));
  if (h.panner) {
    for (let k = 0; k <= 8; k++) h.panner.pan.linearRampToValueAtTime(clamp(h.pan + 0.6 * Math.sin(k * 1.6), -1, 1), t + k * 0.09);
  }
};

const squeak: SfxFn = ({ p, out, t, i }) => {
  // sneakers skidding to a stop
  const a = 0.18 + 0.2 * i;
  const g = p.gain(0);
  const end = env(g.gain, t, a, 0.01, 0.06, 0.08);
  const o = p.osc('sawtooth', 2100, t, end + 0.02);
  path(o.frequency, [[t, 1900], [t + 0.05, 2400], [t + 0.1, 2050], [t + 0.15, 2250]]);
  p.lfo(o.frequency, 55, 120, t, end + 0.02);
  wire(o, p.filter('bandpass', 2300, 8), g, out);
};

const pop: SfxFn = ({ p, out, t, i }) => {
  // a cork popping: for a comic pop-up landing on screen
  const a = 0.3 + 0.25 * i;
  p.tone(out, { t, freq: 700, to: 1500, glide: 0.03, peak: a * 0.6, decay: 0.05 });
  p.burst(out, { t, peak: a * 0.6, decay: 0.006, freq: 1800, q: 1 });
};

// ── UI: cardboard, paper and bottle caps ─────────────────────────────────────

const uiTap: SfxFn = ({ p, out, t }) => {
  // a fingertip on a cardboard sign
  const v = rand(0.94, 1.06);
  p.tone(out, { t, type: 'triangle', freq: 360 * v, to: 230, glide: 0.03, peak: 0.32, decay: 0.045 });
  p.burst(out, { t, peak: 0.3, decay: 0.014, freq: 1700 * v, q: 0.9 });
};

const uiBack: SfxFn = ({ p, out, t }) => {
  // a sheet of poster board sliding away
  const f = p.burst(out, { t, attack: 0.03, peak: 0.28, decay: 0.12, q: 1.2 });
  path(f.frequency, [[t, 3200], [t + 0.15, 1100]]);
  p.tone(out, { t: t + 0.11, type: 'triangle', freq: 300, to: 200, glide: 0.03, peak: 0.2, decay: 0.04 });
};

const uiSelect: SfxFn = ({ p, out, t }) => {
  // a bottle cap set down on the table: a "tok" with a tiny metallic ring
  p.tone(out, { t, type: 'triangle', freq: 1250, to: 1150, peak: 0.22, decay: 0.03 });
  modes(p, out, t, 0.2, [[3900, 0.35, 0.12], [5650, 0.2, 0.08]]);
  p.tone(out, { t, freq: 260, to: 150, glide: 0.06, peak: 0.25, decay: 0.07 }); // the table underneath
  xylo(p, out, t + 0.06, [91], 0.05, 0.12);
};

// ── dispatch ─────────────────────────────────────────────────────────────────

export const SFX: Record<SfxName, SfxFn> = {
  batCrack,
  batTink,
  whiff,
  mittPop,
  catch: glovePop,
  bounce,
  bounceDirt,
  bouncePatio,
  fence,
  picket,
  hedge,
  houseWall,
  splash,
  leaves,
  cheer,
  bigCheer,
  aww,
  ooh,
  giggle,
  strike,
  out: outSting,
  safe,
  homeRun,
  special,
  uiTap,
  uiBack,
  uiSelect,
  whistle,
  dogBark,
  screenDoor,
  throw: throwWhoosh,
  boing,
  bonk,
  zip,
  dizzy,
  squeak,
  pop,
};

/**
 * Loudness trims in dB, balanced by rendering every sound offline and
 * measuring its loudest 50 ms through a phone-speaker-like 300 Hz high-pass
 * (see tests/audio-render.test.ts, which prints the table).
 */
export const TRIM_DB: Partial<Record<SfxName, number>> = {
  batCrack: 3,
  batTink: 6.5,
  whiff: 2.5,
  throw: 2,
  mittPop: 1,
  catch: 7,
  bounce: 4.5,
  bounceDirt: 5,
  bouncePatio: 4,
  fence: 3.5,
  picket: 3,
  hedge: 7.5,
  houseWall: 4,
  splash: 2,
  leaves: 8,
  cheer: 3,
  bigCheer: 3,
  aww: -0.5,
  ooh: -7,
  giggle: 11,
  strike: -2,
  out: 1,
  safe: 0,
  homeRun: 0,
  special: 0.5,
  uiTap: 8.5,
  uiBack: 5,
  uiSelect: 6.5,
  whistle: -1.5,
  dogBark: 2,
  screenDoor: 7,
  boing: 10,
  bonk: 0,
  zip: 3,
  dizzy: 3,
  squeak: -2,
  pop: 5,
};

/** Minimum seconds between repeats of one sound; big layered sounds get longer. */
const MIN_GAP: Partial<Record<SfxName, number>> = {
  cheer: 0.3,
  bigCheer: 0.6,
  aww: 0.3,
  ooh: 0.6,
  giggle: 0.8,
  homeRun: 0.6,
  special: 0.2,
  dogBark: 0.4,
  screenDoor: 2,
};
const DEFAULT_GAP = 0.03;
/** Cap on overlapping sound effects (keeps node counts sane on phones). */
const MAX_VOICES = 24;

let voices = 0;
const lastPlayed = new Map<SfxName, number>();

const num = (v: unknown, d: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : d);

/** Build a sound into a patch that ends at `dest` (the sfx bus, or an ambience send). */
export function buildSfx(p: Patch, dest: AudioNode, name: SfxName, opts?: PlayOpts, when = 0.005): void {
  const fn = SFX[name];
  const pan = clamp(num(opts?.pan, 0), -1, 1);
  const out = p.panner(pan);
  wire(out, p.gain(Math.pow(10, (TRIM_DB[name] ?? 0) / 20)), dest);
  const i = clamp(num(opts?.intensity, 0.7), 0, 1);
  fn({ p, out, t: p.ctx.currentTime + when, i, pan, panner: 'pan' in out ? out : null });
}

export function playSfx(name: SfxName, opts?: PlayOpts): void {
  const bus = getBus();
  if (!bus || !isReady() || isMuted()) return;
  if (!(SFX as Partial<Record<string, SfxFn>>)[name] || voices >= MAX_VOICES) return;
  const now = bus.ctx.currentTime;
  if (now - (lastPlayed.get(name) ?? -Infinity) < (MIN_GAP[name] ?? DEFAULT_GAP)) return;
  lastPlayed.set(name, now);

  voices++;
  const p = new Patch(bus, () => voices--);
  try {
    buildSfx(p, bus.sfx, name, opts);
  } finally {
    p.seal();
  }
}
