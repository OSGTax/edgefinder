/** The music instruments. Each call builds one note (or chord / drum hit) into a Patch. */
import { Patch, env, glide, hits, mtof, pluck, wire } from './dsp';

export type Inst =
  | 'lead' // plucky filtered square
  | 'soft' // rounder triangle lead
  | 'bell' // glockenspiel-ish triangle + octave sine
  | 'bass' // bouncy triangle
  | 'keys' // triangle chord stab
  | 'kick'
  | 'clap'
  | 'hat'
  | 'shaker'
  | 'crash';

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

  clap(p, out, t, _m, _g, vel) {
    const g = p.gain(0);
    const v = vel * NOISE_GAIN;
    const end = hits(g.gain, [[t, v * 0.6], [t + 0.011, v * 0.7], [t + 0.022, v]], 0.001, 0.09);
    wire(p.noise(t, end + 0.02), p.filter('bandpass', 1500, 1.1), g, out);
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
};
