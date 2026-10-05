/** Small synthesis building blocks shared by sfx, music and ambience. */
import type { Bus } from './context';

/** Exponential ramps can't reach 0, so envelopes bottom out here (-80 dB). */
export const FLOOR = 0.0001;

export const rand = (a: number, b: number): number => a + Math.random() * (b - a);
export const pick = <T>(xs: readonly T[]): T => xs[Math.floor(Math.random() * xs.length)];
export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
export const mtof = (m: number): number => 440 * Math.pow(2, (m - 69) / 12);

export function wire(...nodes: AudioNode[]): void {
  for (let i = 0; i < nodes.length - 1; i++) nodes[i].connect(nodes[i + 1]);
}

/** Attack → optional hold → exponential decay. Returns the time it reaches silence. */
export function env(
  p: AudioParam,
  t: number,
  peak: number,
  attack: number,
  decay: number,
  hold = 0,
): number {
  const pk = Math.max(peak, FLOOR * 2);
  p.setValueAtTime(FLOOR, t);
  p.linearRampToValueAtTime(pk, t + attack);
  if (hold > 0) p.setValueAtTime(pk, t + attack + hold);
  const end = t + attack + hold + Math.max(decay, 0.005);
  p.exponentialRampToValueAtTime(FLOOR, end);
  return end;
}

/** Note envelope: attack, decay to sustain (fraction of peak), hold for the gate, release. */
export function pluck(
  p: AudioParam,
  t: number,
  gate: number,
  peak: number,
  a: number,
  d: number,
  s: number,
  r: number,
): number {
  const pk = Math.max(peak, FLOOR * 2);
  p.setValueAtTime(FLOOR, t);
  p.linearRampToValueAtTime(pk, t + a);
  if (gate > a + d) {
    const sus = Math.max(pk * s, FLOOR * 2);
    p.exponentialRampToValueAtTime(sus, t + a + d);
    p.setValueAtTime(sus, t + gate);
  }
  const end = t + Math.max(gate, a + d * 0.5) + r;
  p.exponentialRampToValueAtTime(FLOOR, end);
  return end;
}

/** Exponential sweep of a positive param (frequency, cutoff). */
export function glide(p: AudioParam, t: number, from: number, to: number, dur: number): void {
  p.setValueAtTime(from, t);
  p.exponentialRampToValueAtTime(to, t + Math.max(dur, 0.001));
}

/** Exponential path through [time, value] points (values > 0, times ascending). */
export function path(p: AudioParam, pts: ReadonlyArray<readonly [number, number]>): void {
  p.setValueAtTime(pts[0][1], pts[0][0]);
  for (let i = 1; i < pts.length; i++) p.exponentialRampToValueAtTime(pts[i][1], pts[i][0]);
}

/**
 * A train of short hits on one gain param: claps, rattles and rustles from a
 * single noise source instead of one voice per hit. Hits must be ascending;
 * each decay is shortened so it ends before the next hit starts.
 */
export function hits(
  p: AudioParam,
  list: ReadonlyArray<readonly [number, number]>,
  attack: number,
  decay: number,
): number {
  let end = 0;
  for (let k = 0; k < list.length; k++) {
    const [t, peak] = list[k];
    const next = k + 1 < list.length ? list[k + 1][0] : Infinity;
    const d = Math.max(0.004, Math.min(decay, next - t - attack - 0.001));
    p.setValueAtTime(FLOOR, t);
    p.linearRampToValueAtTime(Math.max(peak, FLOOR * 2), t + attack);
    end = t + attack + d;
    p.exponentialRampToValueAtTime(FLOOR, end);
  }
  return end;
}

/** n ascending random times in [t0, t1], at least minGap apart; skew > 1 crowds them toward t0. */
export function spread(t0: number, t1: number, n: number, minGap: number, skew = 1): number[] {
  const xs = Array.from({ length: n }, () => t0 + (t1 - t0) * Math.pow(Math.random(), skew));
  xs.sort((a, b) => a - b);
  const out: number[] = [];
  for (const x of xs) {
    const y = out.length ? Math.max(x, out[out.length - 1] + minGap) : x;
    if (y <= t1) out.push(y);
  }
  return out;
}

let clipCurve: Float32Array<ArrayBuffer> | null = null;
/** tanh soft-clip transfer curve, built once. */
export function softClip(): Float32Array<ArrayBuffer> {
  if (!clipCurve) {
    const n = 1024;
    clipCurve = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * 2 - 1;
      clipCurve[i] = Math.tanh(2.5 * x) / Math.tanh(2.5);
    }
  }
  return clipCurve;
}

export interface ToneOpts {
  t: number;
  freq: number;
  peak: number;
  decay: number;
  type?: OscillatorType;
  /** Glide target frequency. */
  to?: number;
  /** Glide time (defaults to the whole note). */
  glide?: number;
  attack?: number;
  hold?: number;
}

export interface BurstOpts {
  t: number;
  peak: number;
  decay: number;
  attack?: number;
  hold?: number;
  filter?: BiquadFilterType;
  freq?: number;
  /** Cutoff sweep target. */
  to?: number;
  glide?: number;
  q?: number;
}

/**
 * The nodes behind one sound (or one note). Sources register themselves;
 * when the last one ends, every node is disconnected and onDone fires, so
 * nothing lingers in the graph on a phone.
 */
export class Patch {
  private readonly nodes: AudioNode[] = [];
  private readonly sources: AudioScheduledSourceNode[] = [];
  private live = 0;
  private sealed = false;
  private closed = false;

  constructor(
    readonly bus: Bus,
    private readonly onDone?: () => void,
  ) {}

  get ctx(): AudioContext {
    return this.bus.ctx;
  }

  own<T extends AudioNode>(n: T): T {
    this.nodes.push(n);
    return n;
  }

  gain(v = 1): GainNode {
    const g = this.own(this.ctx.createGain());
    g.gain.value = v;
    return g;
  }

  filter(type: BiquadFilterType, freq: number, q = 1): BiquadFilterNode {
    const f = this.own(this.ctx.createBiquadFilter());
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    return f;
  }

  /** StereoPanner where supported (Safari < 14.1 lacks it: falls back to a plain gain). */
  panner(pan: number): StereoPannerNode | GainNode {
    if (typeof this.ctx.createStereoPanner !== 'function') return this.gain(1);
    const n = this.own(this.ctx.createStereoPanner());
    n.pan.value = clamp(pan, -1, 1);
    return n;
  }

  shaper(curve: Float32Array<ArrayBuffer>): WaveShaperNode {
    const s = this.own(this.ctx.createWaveShaper());
    s.curve = curve;
    return s;
  }

  osc(type: OscillatorType, freq: number, t0: number, t1?: number): OscillatorNode {
    const o = this.own(this.ctx.createOscillator());
    o.type = type;
    o.frequency.value = freq;
    o.start(t0);
    this.track(o, t1);
    return o;
  }

  /** Looping white noise from a random offset, so repeated bursts never sound identical. */
  noise(t0: number, t1?: number): AudioBufferSourceNode {
    const s = this.own(this.ctx.createBufferSource());
    s.buffer = this.bus.noise;
    s.loop = true;
    s.start(t0, Math.random() * this.bus.noise.duration * 0.9);
    this.track(s, t1);
    return s;
  }

  /** Modulate `target` by ±depth at `rate` Hz. Returns the depth gain (connect it to more params if needed). */
  lfo(
    target: AudioParam,
    rate: number,
    depth: number,
    t0: number,
    t1?: number,
    type: OscillatorType = 'sine',
  ): GainNode {
    const g = this.gain(depth);
    this.osc(type, rate, t0, t1).connect(g);
    g.connect(target);
    return g;
  }

  /** Pitched voice: oscillator → envelope → out. */
  tone(out: AudioNode, o: ToneOpts): OscillatorNode {
    const g = this.gain(0);
    const end = env(g.gain, o.t, o.peak, o.attack ?? 0.002, o.decay, o.hold ?? 0);
    const osc = this.osc(o.type ?? 'sine', o.freq, o.t, end + 0.02);
    if (o.to !== undefined) glide(osc.frequency, o.t, o.freq, o.to, o.glide ?? end - o.t);
    wire(osc, g, out);
    return osc;
  }

  /** Filtered noise burst: noise → filter → envelope → out. Returns the filter for custom sweeps. */
  burst(out: AudioNode, o: BurstOpts): BiquadFilterNode {
    const g = this.gain(0);
    const end = env(g.gain, o.t, o.peak, o.attack ?? 0.001, o.decay, o.hold ?? 0);
    const freq = o.freq ?? 1000;
    const f = this.filter(o.filter ?? 'bandpass', freq, o.q ?? 1);
    if (o.to !== undefined) glide(f.frequency, o.t, freq, o.to, o.glide ?? end - o.t);
    wire(this.noise(o.t, end + 0.02), f, g, out);
    return f;
  }

  /** Stop every source at time t (for long-running patches such as wind). */
  stop(t: number): void {
    for (const s of this.sources) {
      try {
        s.stop(t);
      } catch {
        /* already stopped */
      }
    }
  }

  /** Call once the sound is fully built; a patch with no sources is released at once. */
  seal(): void {
    this.sealed = true;
    if (this.live <= 0) this.dispose();
  }

  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    for (const n of this.nodes) {
      try {
        n.disconnect();
      } catch {
        /* already disconnected */
      }
    }
    this.nodes.length = 0;
    this.sources.length = 0;
    this.onDone?.();
  }

  private track(src: AudioScheduledSourceNode, t1: number | undefined): void {
    this.live++;
    this.sources.push(src);
    src.onended = () => {
      this.live--;
      if (this.sealed && this.live <= 0) this.dispose();
    };
    if (t1 !== undefined) src.stop(t1);
  }
}
