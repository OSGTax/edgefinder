/** Every sound effect, synthesized on demand. */
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

interface Hit {
  p: Patch;
  out: AudioNode;
  t: number;
  /** intensity 0..1 */
  i: number;
  pan: number;
  panner: StereoPannerNode | null;
}

type SfxFn = (h: Hit) => void;

/** Drift the sound across the stereo field around its base pan. */
function panSweep(h: Hit, from: number, to: number, dur: number): void {
  if (!h.panner) return;
  h.panner.pan.setValueAtTime(clamp(h.pan + from, -1, 1), h.t);
  h.panner.pan.linearRampToValueAtTime(clamp(h.pan + to, -1, 1), h.t + dur);
}

// ── contact ──────────────────────────────────────────────────────────────────

const batCrack: SfxFn = ({ p, out, t, i }) => {
  const a = 0.45 + 0.5 * i;
  // the crack: broadband burst, brighter and longer the harder the hit
  p.burst(out, { t, peak: a, decay: 0.03 + 0.1 * i, freq: 1500 + 1500 * i, q: 1.2 });
  p.burst(out, { t, peak: a * 0.45, decay: 0.01, filter: 'highpass', freq: 4000 });
  // wood: a pitched knock and a faint ring
  p.tone(out, { t, type: 'triangle', freq: 1000 + 300 * i, to: 640, glide: 0.04, peak: a * 0.4, decay: 0.06 + 0.08 * i });
  p.tone(out, { t, freq: 2400 + 500 * i, peak: a * 0.1, decay: 0.05 + 0.15 * i });
  if (i > 0.55) {
    // home-run crush: some weight underneath and an airy tail
    const k = (i - 0.55) / 0.45;
    p.tone(out, { t, freq: 170, to: 65, glide: 0.09, peak: 0.45 * k, decay: 0.14 });
    p.burst(out, { t: t + 0.008, attack: 0.015, peak: 0.14 * k, decay: 0.3 + 0.4 * k, freq: 3000, to: 1100, q: 0.8 });
  }
};

const batTink: SfxFn = ({ p, out, t, i }) => {
  const a = 0.3 + 0.35 * i;
  p.tone(out, { t, type: 'triangle', freq: 330, to: 170, glide: 0.05, peak: a, decay: 0.08 }); // dull thud
  p.burst(out, { t, peak: a * 0.45, decay: 0.016, freq: 2800, q: 2.5 }); // tick
  p.tone(out, { t: t + 0.003, type: 'square', freq: 1240, to: 1150, peak: a * 0.05, decay: 0.03 }); // sting
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
  const thump = p.tone(out, { t, freq: 200, peak: a * 0.65, decay: 0.13 });
  path(thump.frequency, [[t, 200], [t + 0.025, 118], [t + 0.13, 68]]);
  // the "pok" of the pocket: mid-range, so it still lands on phone speakers
  p.tone(out, { t, type: 'triangle', freq: 560, to: 260, glide: 0.04, peak: a * 0.85, decay: 0.06 });
  p.burst(out, { t, peak: a * 0.9, decay: 0.045, filter: 'lowpass', freq: 1500, q: 0.7 }); // leather slap
  p.burst(out, { t, peak: a * 0.7, decay: 0.02, freq: 850, q: 1.5 }); // pop
};

const glovePop: SfxFn = ({ p, out, t, i }) => {
  const a = 0.3 + 0.35 * i;
  const thump = p.tone(out, { t, freq: 240, peak: a * 0.5, decay: 0.09 });
  path(thump.frequency, [[t, 240], [t + 0.02, 150], [t + 0.09, 100]]);
  p.tone(out, { t, type: 'triangle', freq: 440, to: 240, glide: 0.035, peak: a * 0.9, decay: 0.05 });
  p.burst(out, { t, peak: a, decay: 0.035, filter: 'lowpass', freq: 1200, q: 0.7 });
};

// ── the yard ─────────────────────────────────────────────────────────────────

const bounce: SfxFn = ({ p, out, t, i }) => {
  const a = 0.15 + 0.6 * i;
  p.tone(out, { t, freq: 90 + 50 * i, to: 48, glide: 0.08, peak: a * 0.5, decay: 0.1 });
  p.tone(out, { t, type: 'triangle', freq: 330 + 80 * i, to: 170, glide: 0.05, peak: a * 0.9, decay: 0.07 }); // "thup"
  p.burst(out, { t, peak: a * 0.8, decay: 0.03, freq: 700, q: 0.8 });
  p.burst(out, { t: t + 0.004, peak: a * 0.12, decay: 0.05, filter: 'highpass', freq: 3500 }); // grass blades
};

const fence: SfxFn = ({ p, out, t, i }) => {
  const a = 0.4 + 0.45 * i;
  // hollow board knock: two damped modes plus the impact click
  p.tone(out, { t, type: 'triangle', freq: 235, to: 200, peak: a * 0.8, decay: 0.17 });
  p.tone(out, { t, freq: 560, to: 520, peak: a * 0.4, decay: 0.09 });
  p.burst(out, { t, peak: a * 0.5, decay: 0.02, freq: 1700 });
  // loose pickets rattling, slowing and fading
  const list: Array<[number, number]> = [];
  let x = t + 0.03;
  let gap = 0.032;
  let pk = a * 0.4;
  for (let k = 0; k < 7 + Math.round(3 * i); k++) {
    list.push([x, pk * rand(0.6, 1)]);
    x += gap * rand(0.8, 1.25);
    gap *= 1.08;
    pk *= 0.8;
  }
  const g = p.gain(0);
  const end = hits(g.gain, list, 0.002, 0.025);
  wire(p.noise(t, end + 0.02), p.filter('bandpass', 1250, 2.2), g, out);
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
  // "ruff-ruff": the second bark a little lower
  [0, 0.24].forEach((dt, k) => {
    const tb = t + dt;
    const s = k ? 0.9 : 1;
    const g = p.gain(0);
    const end = env(g.gain, tb, 1.6 * a, 0.008, 0.12, 0.03);
    const o = p.osc('sawtooth', 300 * s, tb, end + 0.02);
    path(o.frequency, [[tb, 300 * s], [tb + 0.025, 540 * s], [tb + 0.16, 250 * s]]);
    wire(o, g, drive);
    glide(f1.frequency, tb, 1000 * s, 520 * s, 0.15); // "wuh" closing to "oof"
    p.burst(out, { t: tb, attack: 0.004, peak: a * 0.25, decay: 0.07, freq: 1300, q: 1.2 });
  });
};

// ── kid crowd ────────────────────────────────────────────────────────────────

interface Crowd {
  dur: number;
  voices: number;
  breaths: number;
  claps: number;
  /** how far voice entries are staggered (s) */
  stagger: number;
  level: number;
  aww?: boolean;
}

type Contour = 'yay' | 'woo' | 'aww';

function contour(kind: Contour, t: number, len: number, f0: number): Array<[number, number]> {
  switch (kind) {
    case 'yay':
      return [[t, f0 * 0.8], [t + len * 0.18, f0 * 1.22], [t + len * 0.6, f0 * rand(1, 1.12)], [t + len, f0 * 0.82]];
    case 'woo':
      return [[t, f0 * 0.85], [t + len * 0.35, f0 * 1.5], [t + len * 0.7, f0 * 1.35], [t + len, f0 * 0.95]];
    case 'aww':
      return [[t, f0 * 1.12], [t + 0.1, f0 * 1.18], [t + len, f0 * 0.68]];
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

function kids({ p, out, t }: Hit, c: Crowd): void {
  const end = t + c.dur;
  // one shared vowel filter for all the pitched voices: body + two formants
  const vox = p.gain(1);
  const f1 = p.filter('bandpass', c.aww ? 780 : 1050, 3);
  const f2 = p.filter('bandpass', 2000, 5);
  wire(vox, p.filter('lowpass', 1300, 0.7), p.gain(0.35), out);
  wire(vox, f1, out);
  wire(vox, f2, p.gain(0.7), out);
  if (c.aww) glide(f2.frequency, t, 1250, 950, c.dur); // "aww" darkens as it sinks
  else glide(f2.frequency, t, 1800, 2700, c.dur * 0.5); // "yaay": eh -> ee

  const vPeak = c.level / Math.sqrt(c.voices);
  for (let v = 0; v < c.voices; v++) {
    const ts = t + rand(0, c.stagger);
    const len = Math.max(0.3, Math.min(end - ts, c.dur * rand(0.55, 0.9)));
    const f0 = rand(330, 620); // kid voices
    const osc = p.osc(v % 3 === 2 ? 'triangle' : 'sawtooth', f0, ts, ts + len + 0.02);
    const kind: Contour = c.aww ? 'aww' : Math.random() < 0.6 ? 'yay' : 'woo';
    path(osc.frequency, contour(kind, ts, len, f0));
    p.lfo(osc.frequency, rand(5, 8), f0 * rand(0.02, 0.04), ts, ts + len + 0.02); // vibrato
    const g = p.gain(0);
    if (c.aww) env(g.gain, ts, vPeak, 0.14, Math.max(0.1, len * 0.65 - 0.14), len * 0.3);
    else syllables(g.gain, ts, len, vPeak);
    wire(osc, g, vox);
  }

  // breathy shouting: formant-filtered noise with a fast chatter tremolo
  for (let n = 0; n < c.breaths; n++) {
    const am = p.gain(0.6);
    p.lfo(am.gain, rand(6, 11), 0.4, t, end + 0.02);
    const g = p.gain(0);
    const attack = c.aww ? 0.2 : 0.06;
    env(g.gain, t, c.level * 0.35, attack, c.dur * 0.72 - attack, c.dur * 0.25);
    const bp = p.filter('bandpass', c.aww ? rand(600, 900) : rand(1000, 1700), 1.8);
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

const cheer: SfxFn = (h) => kids(h, { dur: 1.2, voices: 6, breaths: 2, claps: 8, stagger: 0.15, level: 0.5 });

const bigCheer: SfxFn = (h) => {
  kids(h, { dur: 2.5, voices: 9, breaths: 3, claps: 22, stagger: 0.8, level: 0.6 });
  // somebody's two-finger whistle
  const { p, out, t } = h;
  const g = p.gain(0);
  const end = env(g.gain, t + 0.3, 0.07, 0.04, 0.2, 0.5);
  const o = p.osc('sine', 1800, t + 0.3, end + 0.02);
  path(o.frequency, [[t + 0.3, 1800], [t + 0.45, 3100], [t + 0.9, 2900], [t + 1.05, 2300]]);
  wire(o, g, out);
};

const aww: SfxFn = (h) =>
  kids(h, { dur: 1.4, voices: 6, breaths: 2, claps: 0, stagger: 0.12, level: 0.45, aww: true });

// ── stingers & jingles ───────────────────────────────────────────────────────

const { lead, bell } = INSTRUMENTS;

const strike: SfxFn = ({ p, out, t }) => {
  lead(p, out, t, [81], 0.06, 0.28); // A5
  lead(p, out, t + 0.1, [88], 0.12, 0.3); // E6
  bell(p, out, t + 0.1, [100], 0.1, 0.07);
};

const outSting: SfxFn = ({ p, out, t }) => {
  lead(p, out, t, [79], 0.07, 0.25); // G5
  lead(p, out, t + 0.09, [75], 0.07, 0.25); // Eb5
  // C5 sagging downward
  const t2 = t + 0.18;
  const g = p.gain(0);
  const end = env(g.gain, t2, 0.24, 0.005, 0.22, 0.06);
  const o = p.osc('square', mtof(72), t2, end + 0.02);
  glide(o.frequency, t2 + 0.05, mtof(72), mtof(65), 0.2);
  wire(o, p.filter('lowpass', 2400, 1), g, out);
};

const safe: SfxFn = ({ p, out, t }) => {
  [84, 88, 91].forEach((m, k) => lead(p, out, t + k * 0.055, [m], 0.05, 0.22)); // C E G
  bell(p, out, t + 0.165, [96], 0.25, 0.18);
};

const HR_STEP = 0.085;
const HOME_RUN: Array<[Inst, number, Seq]> = [
  ['lead', 0.3, notes('C5/1 E5/1 G5/1 C6/3 A5/1 C6/1 D6/2 E6/6')],
  ['soft', 0.13, notes('E4/1 G4/1 C5/1 E5/3 F5/1 A5/1 B5/2 G5/6')],
  ['bass', 0.4, notes('C3/3 G3/3 F3/4 C3/6')],
  ['keys', 0.08, notes('./6 F4+A4+D5/4 C4+E4+G4+C5/6')],
  ['kick', 0.5, beat('x.....x...x.....')],
  ['crash', 0.16, beat('..........X.....')],
];

const homeRun: SfxFn = ({ p, out, t }) => {
  for (const [inst, gain, seq] of HOME_RUN) {
    for (const ev of seq.evs) {
      INSTRUMENTS[inst](p, out, t + ev.step * HR_STEP, ev.midis, ev.len * HR_STEP * 0.9, ev.vel * gain);
    }
  }
  [96, 100, 103, 108].forEach((m, k) => bell(p, out, t + 10 * HR_STEP + 0.05 + k * 0.07, [m], 0.1, 0.06));
};

const special: SfxFn = ({ p, out, t, i }) => {
  const a = 0.35 + 0.3 * i;
  // rising whoosh
  p.burst(out, { t, attack: 0.3, hold: 0.1, peak: a * 0.6, decay: 0.35, freq: 300, to: 5000, glide: 0.55, q: 2.5 });
  // a detuned pair sweeping up, then shimmering
  for (const det of [-8, 8]) {
    const o = p.tone(out, {
      t,
      attack: 0.2,
      hold: 0.25,
      type: det < 0 ? 'triangle' : 'sine',
      freq: 330,
      to: 1650,
      glide: 0.5,
      peak: a * 0.22,
      decay: 0.4,
    });
    o.detune.value = det;
    p.lfo(o.frequency, 9, 30, t + 0.4, t + 0.9);
  }
  // sparkles: quick pentatonic pings in two overlapping streams
  const PINGS = [96, 98, 100, 103, 105, 108];
  for (let s = 0; s < 2; s++) {
    const times = spread(t + 0.25 + s * 0.04, t + 0.95, 5, 0.07);
    const g = p.gain(0);
    const end = hits(g.gain, times.map((x): [number, number] => [x, a * rand(0.12, 0.22)]), 0.003, 0.12);
    const o = p.osc('sine', mtof(96), t, end + 0.02);
    for (const x of times) o.frequency.setValueAtTime(mtof(pick(PINGS)), x);
    wire(o, g, out);
  }
};

// ── UI ───────────────────────────────────────────────────────────────────────

const uiTap: SfxFn = ({ p, out, t }) => {
  p.tone(out, { t, type: 'triangle', freq: 1150, to: 780, glide: 0.035, peak: 0.22, decay: 0.045 });
  p.burst(out, { t, peak: 0.06, decay: 0.008, filter: 'highpass', freq: 3000 });
};

const uiBack: SfxFn = ({ p, out, t }) => {
  p.tone(out, { t, type: 'triangle', freq: 880, to: 700, glide: 0.04, peak: 0.2, decay: 0.05 });
  p.tone(out, { t: t + 0.06, type: 'triangle', freq: 620, to: 470, glide: 0.06, peak: 0.2, decay: 0.07 });
};

const uiSelect: SfxFn = ({ p, out, t }) => {
  lead(p, out, t, [88], 0.04, 0.2); // E6
  lead(p, out, t + 0.055, [95], 0.08, 0.22); // B6
  p.tone(out, { t, freq: 260, to: 150, glide: 0.06, peak: 0.25, decay: 0.07 }); // soft thock underneath
};

const whistle: SfxFn = ({ p, out, t }) => {
  const g = p.gain(0);
  const end = env(g.gain, t, 0.28, 0.03, 0.1, 0.5);
  const o = p.osc('sine', 650, t, end + 0.02);
  path(o.frequency, [[t, 650], [t + 0.3, 1650], [t + 0.42, 1550], [t + 0.6, 950]]);
  p.lfo(o.frequency, 27, 40, t, end + 0.02); // pea-whistle trill
  wire(o, g, out);
  p.burst(out, { t, attack: 0.03, hold: 0.45, peak: 0.04, decay: 0.12, freq: 2600 }); // breath
};

// ── dispatch ─────────────────────────────────────────────────────────────────

export const SFX: Record<SfxName, SfxFn> = {
  batCrack,
  batTink,
  whiff,
  mittPop,
  catch: glovePop,
  bounce,
  fence,
  splash,
  leaves,
  cheer,
  bigCheer,
  aww,
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
  throw: throwWhoosh,
};

/**
 * Loudness trims in dB, balanced by rendering every sound offline and
 * measuring its loudest 50 ms through a phone-speaker-like 300 Hz high-pass.
 */
const TRIM_DB: Partial<Record<SfxName, number>> = {
  batCrack: 4.5,
  mittPop: 2,
  catch: 5,
  bounce: 2,
  batTink: 6,
  whiff: 3.5,
  throw: 4,
  fence: 4.5,
  splash: 1,
  leaves: 6,
  cheer: 3,
  bigCheer: 1,
  aww: -3.5,
  out: -3,
  safe: 3.5,
  homeRun: -2.5,
  special: 1.5,
  uiTap: 5,
  uiBack: 5,
  uiSelect: -1.5,
  whistle: -2,
  dogBark: 1.5,
};

/** Minimum seconds between repeats of one sound; big layered sounds get longer. */
const MIN_GAP: Partial<Record<SfxName, number>> = {
  cheer: 0.3,
  bigCheer: 0.6,
  aww: 0.3,
  homeRun: 0.6,
  special: 0.2,
  dogBark: 0.4,
};
const DEFAULT_GAP = 0.03;
/** Cap on overlapping sound effects (keeps node counts sane on phones). */
const MAX_VOICES = 24;

let voices = 0;
const lastPlayed = new Map<SfxName, number>();

const num = (v: unknown, d: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : d);

export function playSfx(name: SfxName, opts?: PlayOpts): void {
  const bus = getBus();
  if (!bus || !isReady() || isMuted()) return;
  const fn = (SFX as Partial<Record<string, SfxFn>>)[name];
  if (!fn || voices >= MAX_VOICES) return;
  const now = bus.ctx.currentTime;
  if (now - (lastPlayed.get(name) ?? -Infinity) < (MIN_GAP[name] ?? DEFAULT_GAP)) return;
  lastPlayed.set(name, now);

  voices++;
  const p = new Patch(bus, () => voices--);
  try {
    const pan = clamp(num(opts?.pan, 0), -1, 1);
    const out = p.panner(pan);
    wire(out, p.gain(Math.pow(10, (TRIM_DB[name] ?? 0) / 20)), bus.sfx);
    const i = clamp(num(opts?.intensity, 0.7), 0, 1);
    fn({ p, out, t: now + 0.005, i, pan, panner: 'pan' in out ? out : null });
  } finally {
    p.seal();
  }
}
