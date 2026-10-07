/**
 * The music instruments. Each call builds one note (or chord / drum hit) into a Patch.
 *
 * The band is the kids themselves, playing whatever was in the garage: a
 * kazoo, a ukulele, a toy keyboard with a built-in drum machine, a toy
 * glockenspiel, a xylophone, a music box, somebody whistling, a cardboard
 * box for a kick drum, hand claps and a pot lid for a cymbal. Everything is
 * voiced in the mids so it reads on a phone speaker; nothing relies on sub-bass.
 */
import { pluck as bakePluck } from './bake';
import { Patch, env, glide, harmonics, hits, mtof, pluck, rand, softClip, wire } from './dsp';

export type Inst =
  | 'lead' // plucky filtered square
  | 'soft' // rounder triangle lead
  | 'bell' // glockenspiel-ish triangle + octave sine
  | 'bass' // bouncy triangle
  | 'keys' // triangle chord stab
  | 'kazoo' // a kid humming into a kazoo
  | 'uke' // ukulele: chords are strummed down, single notes picked
  | 'ukeBass' // the low strings of a cheap guitar, plucked
  | 'toy' // a toy keyboard's "piano" preset
  | 'glock' // toy glockenspiel (metal bars)
  | 'xylo' // wooden xylophone
  | 'musicbox'
  | 'whistle' // somebody whistling
  | 'kick'
  | 'box' // a cardboard box, kicked
  | 'clap'
  | 'claps' // two kids clapping, not quite together
  | 'hat'
  | 'shaker'
  | 'crash'
  | 'lid' // a pot lid hit with a spoon
  | 'toyKick' // the toy keyboard's drum machine
  | 'toySnare'
  | 'toyHat';

export type InstFn = (
  p: Patch,
  out: AudioNode,
  t: number,
  midis: readonly number[],
  gate: number,
  vel: number,
) => void;

/** Filtered noise keeps only a sliver of its energy; this puts drum gains on the same scale as tones. */
const NOISE_GAIN = 6;

/** Struck bars: a few inharmonic partials, each with its own decay. */
function bar(
  p: Patch,
  out: AudioNode,
  t: number,
  f: number,
  vel: number,
  partials: ReadonlyArray<readonly [ratio: number, amp: number, decay: number]>,
): void {
  for (const [ratio, amp, decay] of partials) {
    const fr = f * ratio;
    if (fr > 9500) continue;
    const g = p.gain(0);
    const end = env(g.gain, t, vel * amp, 0.0015, decay);
    wire(p.osc('sine', fr, t, end + 0.02), g, out);
  }
}

const TOY_WAVE = harmonics(24, 1, (h) => Math.sin(Math.PI * h * 0.25)); // a 25% pulse, like the cheap ones

export const INSTRUMENTS: Record<Inst, InstFn> = {
  lead(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      const g = p.gain(0);
      const end = pluck(g.gain, t, gate, vel, 0.004, 0.12, 0.45, 0.09);
      const lp = p.filter('lowpass', f * 2.5, 2);
      glide(lp.frequency, t, Math.min(f * 7, 9000), f * 2.5, 0.16); // bright pluck, mellow body
      wire(p.osc('square', f, t, end + 0.02), lp, g, out);
    }
  },

  soft(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      const g = p.gain(0);
      const end = pluck(g.gain, t, gate, vel, 0.008, 0.18, 0.6, 0.12);
      const shine = p.gain(0.12);
      wire(p.osc('triangle', f, t, end + 0.02), g, out);
      wire(p.osc('sine', f * 2, t, end + 0.02), shine, g);
    }
  },

  bell(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      const g = p.gain(0);
      const end = env(g.gain, t, vel, 0.002, 0.45 + gate * 0.4);
      const shine = p.gain(0.3);
      wire(p.osc('triangle', f, t, end + 0.02), g, out);
      wire(p.osc('sine', f * 2, t, end + 0.02), shine, g);
    }
  },

  bass(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      const g = p.gain(0);
      const end = pluck(g.gain, t, gate, vel, 0.004, 0.14, 0.5, 0.06);
      const body = p.osc('triangle', f, t, end + 0.02);
      glide(body.frequency, t, f * 1.3, f, 0.03); // the "bounce": a quick pitch drop into the note
      wire(body, g, out);
      // a little filtered square on top so the line still reads on phone speakers
      const edge = p.gain(0.16);
      wire(p.osc('square', f, t, end + 0.02), p.filter('lowpass', f * 4, 0.7), edge, g);
    }
  },

  keys(p, out, t, midis, gate, vel) {
    const g = p.gain(0);
    const end = pluck(g.gain, t, gate, vel, 0.005, 0.1, 0.3, 0.07);
    const lp = p.filter('lowpass', 2400, 0.7);
    wire(lp, g, out);
    for (const m of midis) wire(p.osc('triangle', mtof(m), t, end + 0.02), lp);
  },

  kazoo(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      const g = p.gain(0);
      const end = pluck(g.gain, t, gate, vel, 0.025, 0.08, 0.8, 0.06);
      const o = p.osc('sawtooth', f, t, end + 0.02);
      // hummed notes scoop up into pitch, and wobble once they're held
      o.frequency.setValueAtTime(f * 0.955, t);
      o.frequency.exponentialRampToValueAtTime(f, t + 0.06);
      if (gate > 0.22) {
        const vib = p.gain(0);
        vib.gain.setValueAtTime(0, t);
        vib.gain.linearRampToValueAtTime(0, t + 0.14);
        vib.gain.linearRampToValueAtTime(f * 0.014, t + 0.32);
        wire(p.osc('sine', rand(5.2, 6.2), t, end + 0.02), vib);
        vib.connect(o.frequency);
      }
      // the paper membrane buzzing: drive into a soft clip, then the tube's formants
      const drive = p.gain(2.2);
      const buzz = p.shaper(softClip());
      const mix = p.gain(0.5);
      wire(o, drive, buzz);
      for (const [ff, q, a] of [[650, 2.5, 1], [1350, 4, 0.7], [2700, 5, 0.35]] as const) {
        wire(buzz, p.filter('bandpass', ff, q), p.gain(a), mix);
      }
      wire(mix, p.filter('lowpass', 3800, 0.5), g, out);
    }
  },

  uke(p, out, t, midis, gate, vel) {
    // a chord is strummed downward, one string after another
    const sorted = [...midis].sort((a, b) => a - b);
    sorted.forEach((m, k) => {
      const ts = t + k * 0.014 + rand(0, 0.004);
      const { buf, rate } = bakePluck(p.ctx, m, 0.55, 1.1);
      const g = p.gain(0);
      const v = vel * (1 - k * 0.08) * (sorted.length > 1 ? 0.62 : 1);
      g.gain.setValueAtTime(v, ts);
      const stop = ts + Math.max(gate, 0.08) + 0.12;
      g.gain.setValueAtTime(v, stop - 0.1);
      g.gain.exponentialRampToValueAtTime(0.0001, stop);
      wire(p.play(buf, ts, rate, false, stop + 0.01), g, out);
    });
  },

  ukeBass(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const { buf, rate } = bakePluck(p.ctx, m, 0.3, 1.3);
      const g = p.gain(0);
      g.gain.setValueAtTime(vel, t);
      const stop = t + Math.max(gate, 0.1) + 0.08;
      g.gain.setValueAtTime(vel, stop - 0.07);
      g.gain.exponentialRampToValueAtTime(0.0001, stop);
      // a gentle peak at the second harmonic keeps the line audible on small speakers
      const body = p.filter('peaking', mtof(m) * 2, 1);
      body.gain.value = 5;
      wire(p.play(buf, t, rate, false, stop + 0.01), body, p.filter('lowpass', 1800, 0), g, out);
    }
  },

  toy(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      const g = p.gain(0);
      const end = pluck(g.gain, t, gate, vel, 0.004, 0.06, 0.75, 0.05);
      const o = p.custom(TOY_WAVE[0], TOY_WAVE[1], f, t, end + 0.02);
      // the cheap keyboard's built-in vibrato, which kicks in late
      if (gate > 0.25) {
        const vib = p.gain(0);
        vib.gain.setValueAtTime(0, t + 0.2);
        vib.gain.linearRampToValueAtTime(f * 0.008, t + 0.35);
        wire(p.osc('triangle', 6.5, t, end + 0.02), vib);
        vib.connect(o.frequency);
      }
      wire(o, p.filter('lowpass', 3600, 0), g, out);
    }
  },

  glock(p, out, t, midis, gate, vel) {
    const ring = 0.5 + gate * 0.3;
    for (const m of midis) {
      bar(p, out, t, mtof(m), vel, [[1, 1, ring], [2.76, 0.32, ring * 0.35], [5.4, 0.14, 0.09], [8.93, 0.06, 0.04]]);
    }
  },

  xylo(p, out, t, midis, _gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      bar(p, out, t, f, vel, [[1, 1, 0.22], [3.93, 0.3, 0.05], [9.2, 0.1, 0.015]]);
      // the mallet's knock on the wood
      p.burst(out, { t, peak: vel * 0.35, decay: 0.012, freq: Math.min(f * 2, 4000), q: 1.5 });
    }
  },

  musicbox(p, out, t, midis, _gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      // two tines a hair apart beat gently against each other
      for (const det of [0.9985, 1.0015]) {
        bar(p, out, t, f * det, vel * 0.55, [[1, 1, 1.1], [2, 0.18, 0.45], [5.95, 0.12, 0.05]]);
      }
    }
  },

  whistle(p, out, t, midis, gate, vel) {
    for (const m of midis) {
      const f = mtof(m);
      const g = p.gain(0);
      const end = pluck(g.gain, t, gate, vel, 0.03, 0.1, 0.85, 0.07);
      const o = p.osc('sine', f, t, end + 0.02);
      o.frequency.setValueAtTime(f * 0.97, t);
      o.frequency.exponentialRampToValueAtTime(f, t + 0.045);
      if (gate > 0.2) p.lfo(o.frequency, rand(4.6, 5.4), f * 0.009, t + 0.12, end + 0.02);
      wire(o, g, out);
      // breath around the tone
      const br = p.gain(0);
      env(br.gain, t, vel * 0.5, 0.03, Math.max(0.05, gate * 0.8));
      wire(p.noise(t, end + 0.02), p.filter('bandpass', f, 6), br, out);
    }
  },

  kick(p, out, t, _m, _g, vel) {
    const g = p.gain(0);
    const end = env(g.gain, t, vel, 0.002, 0.24);
    const o = p.osc('sine', 150, t, end + 0.02);
    glide(o.frequency, t, 150, 46, 0.11);
    wire(o, g, out);
    // a soft beater click: the only part of a kick a phone speaker can reproduce
    const c = p.gain(0);
    env(c.gain, t, vel * 0.3, 0.001, 0.03);
    const tick = p.osc('triangle', 320, t, t + 0.05);
    glide(tick.frequency, t, 320, 120, 0.03);
    wire(tick, c, out);
  },

  box(p, out, t, _m, _g, vel) {
    // a sneaker into a cardboard box: a hollow "thup" with a papery slap
    p.tone(out, { t, type: 'triangle', freq: 190, to: 92, glide: 0.05, peak: vel * 0.9, decay: 0.11 });
    p.tone(out, { t, freq: 420, to: 300, glide: 0.03, peak: vel * 0.25, decay: 0.05 });
    p.burst(out, { t, peak: vel * NOISE_GAIN * 0.12, decay: 0.035, freq: 900, q: 0.9 });
  },

  clap(p, out, t, _m, _g, vel) {
    const g = p.gain(0);
    const v = vel * NOISE_GAIN;
    const end = hits(g.gain, [[t, v * 0.6], [t + 0.011, v * 0.7], [t + 0.022, v]], 0.001, 0.09);
    wire(p.noise(t, end + 0.02), p.filter('bandpass', 1500, 1.1), g, out);
  },

  claps(p, out, t, _m, _g, vel) {
    // two pairs of hands, never quite in time
    const v = vel * NOISE_GAIN;
    const t2 = t + rand(0.008, 0.026);
    const g = p.gain(0);
    const end = hits(g.gain, [[t, v * 0.8], [t + 0.009, v * 0.5], [t2, v * 0.6]], 0.001, 0.07);
    wire(p.noise(t, end + 0.02), p.filter('bandpass', rand(1300, 1800), 1.2), g, out);
  },

  hat(p, out, t, _m, _g, vel) {
    const g = p.gain(0);
    const end = env(g.gain, t, vel * NOISE_GAIN, 0.001, 0.035);
    wire(p.noise(t, end + 0.02), p.filter('highpass', 7500, 0.7), g, out);
  },

  shaker(p, out, t, _m, _g, vel) {
    const g = p.gain(0);
    const end = env(g.gain, t, vel * NOISE_GAIN, 0.012, 0.05);
    wire(p.noise(t, end + 0.02), p.filter('bandpass', 5500, 0.9), g, out);
  },

  crash(p, out, t, _m, _g, vel) {
    const g = p.gain(0);
    const end = env(g.gain, t, vel, 0.002, 1.3); // wideband: needs no make-up gain
    wire(p.noise(t, end + 0.02), p.filter('highpass', 4000, 0.5), g, out);
  },

  lid(p, out, t, _m, _g, vel) {
    // a saucepan lid: clangy inharmonic partials, a lot shorter than a cymbal
    const g = p.gain(0);
    const end = env(g.gain, t, vel * 0.18, 0.001, 0.75);
    const hp = p.filter('highpass', 1800, 0.7);
    wire(hp, g, out);
    for (const f of [587, 845, 1163, 1531, 2087, 2810]) wire(p.osc('square', f * rand(0.99, 1.01), t, end + 0.02), hp);
    p.burst(out, { t, peak: vel * 0.5, decay: 0.3, filter: 'highpass', freq: 5000 });
    p.tone(out, { t, freq: 290, peak: vel * 0.2, decay: 0.05 }); // the spoon's clunk
  },

  toyKick(p, out, t, _m, _g, vel) {
    p.tone(out, { t, freq: 210, to: 70, glide: 0.04, peak: vel, decay: 0.09 });
    p.tone(out, { t, type: 'square', freq: 420, to: 200, glide: 0.01, peak: vel * 0.12, decay: 0.015 });
  },

  toySnare(p, out, t, _m, _g, vel) {
    // "pff": the cheap drum machine's snare is mostly a burst of hiss
    p.burst(out, { t, peak: vel * NOISE_GAIN * 0.35, decay: 0.07, freq: 2600, q: 0.6 });
    p.tone(out, { t, type: 'triangle', freq: 330, to: 240, glide: 0.03, peak: vel * 0.3, decay: 0.04 });
  },

  toyHat(p, out, t, _m, _g, vel) {
    const g = p.gain(0);
    const end = env(g.gain, t, vel * NOISE_GAIN * 0.8, 0.001, 0.02);
    wire(p.noise(t, end + 0.02), p.filter('highpass', 6500, 0), g, out);
  },
};

/** Drums and percussion: tighter timing when humanized. */
export const PERCUSSION: ReadonlySet<Inst> = new Set<Inst>([
  'kick', 'box', 'clap', 'claps', 'hat', 'shaker', 'crash', 'lid', 'toyKick', 'toySnare', 'toyHat',
]);
