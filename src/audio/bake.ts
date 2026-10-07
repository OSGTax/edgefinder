/**
 * Sounds that are cheaper to compute once in plain JS than to build out of
 * dozens of nodes every time: plucked strings (Karplus-Strong), the grill's
 * sizzle and the neighbour's sprinkler. Each is rendered into an AudioBuffer
 * the first time it's needed and cached per AudioContext, so at play time it
 * costs a single buffer source.
 *
 * Buffers are baked at 22.05 kHz: plenty for these sounds, half the work and
 * memory of 44.1/48 kHz, and every browser resamples them on playback.
 */

const RATE = 22050;
const caches = new WeakMap<BaseAudioContext, Map<string, AudioBuffer>>();

function cached(ctx: BaseAudioContext, key: string, make: () => Float32Array): AudioBuffer {
  let m = caches.get(ctx);
  if (!m) caches.set(ctx, (m = new Map()));
  let b = m.get(key);
  if (!b) {
    const data = make();
    b = ctx.createBuffer(1, data.length, RATE);
    b.getChannelData(0).set(data);
    m.set(key, b);
  }
  return b;
}

/** Small deterministic PRNG so baked textures are identical every time. */
function prng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) / 4294967296) * 2 - 1;
  };
}

/** RBJ biquad run over an array in place (lowpass / highpass / bandpass). */
function biquad(x: Float32Array, type: 'lowpass' | 'highpass' | 'bandpass', f: number, q: number): Float32Array {
  const w0 = (2 * Math.PI * Math.min(f, RATE * 0.49)) / RATE;
  const cw = Math.cos(w0);
  const al = Math.sin(w0) / (2 * q);
  let b0: number, b1: number, b2: number;
  if (type === 'lowpass') [b0, b1, b2] = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2];
  else if (type === 'highpass') [b0, b1, b2] = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2];
  else [b0, b1, b2] = [al, 0, -al];
  const a0 = 1 + al, a1 = -2 * cw, a2 = 1 - al;
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const y = (b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = y;
    x[i] = y;
  }
  return x;
}

function normalize(x: Float32Array, peak: number): Float32Array {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
  if (m > 0) for (let i = 0; i < x.length; i++) x[i] *= peak / m;
  return x;
}

/** Fold the last `fade` samples over the start so the buffer loops without a seam. */
function loopable(x: Float32Array, fade: number): Float32Array {
  const n = x.length - fade;
  const out = x.slice(0, n);
  for (let i = 0; i < fade; i++) {
    const w = i / fade;
    out[i] = x[i] * w + x[n + i] * (1 - w);
  }
  return out;
}

// ── plucked strings ──────────────────────────────────────────────────────────

export interface Pluck {
  buf: AudioBuffer;
  /** playbackRate that lands exactly on the requested pitch */
  rate: number;
}

/**
 * A Karplus-Strong string: noise in a tuned delay loop with an averaging
 * filter, which is what makes the top end die away first, like nylon.
 * `tone` 0..1 is how bright the pluck is (fingertip .. pick), `ring` the
 * sustain in seconds.
 */
export function pluck(ctx: BaseAudioContext, midi: number, tone = 0.5, ring = 1.4): Pluck {
  const f = 440 * Math.pow(2, (midi - 69) / 12);
  const N = Math.max(2, Math.round(RATE / f - 0.5));
  const actual = RATE / (N + 0.5);
  const key = `pluck:${N}:${tone.toFixed(2)}:${ring.toFixed(2)}`;
  const buf = cached(ctx, key, () => {
    const len = Math.ceil(RATE * (ring + 0.1));
    const y = new Float32Array(len);
    const rnd = prng(N * 7919 + Math.round(tone * 100));
    // excitation: noise, darker for a softer pluck, and a pick-position comb
    const ex = new Float32Array(N);
    let lp = 0;
    const k = 0.15 + 0.8 * tone;
    for (let i = 0; i < N; i++) {
      lp += k * (rnd() - lp);
      ex[i] = lp;
    }
    const pickAt = Math.max(1, Math.round(N * 0.18));
    for (let i = 0; i < N; i++) y[i] = ex[i] - 0.6 * (i >= pickAt ? ex[i - pickAt] : 0);
    // loop gain per period so the string falls ~60 dB over `ring` seconds
    const g = Math.pow(10, -3 / (ring * actual));
    for (let i = N; i < len; i++) {
      y[i] = g * 0.5 * (y[i - N] + y[i - N - 1 >= 0 ? i - N - 1 : i - N]);
    }
    // short fade at the very end
    for (let i = 0; i < 200 && i < len; i++) y[len - 1 - i] *= i / 200;
    return normalize(y, 0.9);
  });
  return { buf, rate: f / actual };
}

// ── the grill ────────────────────────────────────────────────────────────────

/** Mr. Mendoza's grill: a bed of hiss with fat spitting and crackling, 4 s loop. */
export function sizzle(ctx: BaseAudioContext): AudioBuffer {
  return cached(ctx, 'sizzle', () => {
    const fade = Math.round(RATE * 0.05);
    const len = RATE * 4 + fade;
    const rnd = prng(42);
    const hiss = new Float32Array(len);
    for (let i = 0; i < len; i++) hiss[i] = rnd();
    biquad(hiss, 'highpass', 2600, 0.7);
    biquad(hiss, 'lowpass', 9000, 0.7);
    // the hiss breathes a little as fat drips
    for (let i = 0; i < len; i++) {
      const t = i / RATE;
      hiss[i] *= 0.22 * (0.75 + 0.25 * Math.sin(2 * Math.PI * 0.7 * t) * Math.sin(2 * Math.PI * 1.9 * t + 1));
    }
    // crackles: short decaying ticks, a few bigger pops
    const crack = new Float32Array(len);
    let t = 0;
    while (t < len) {
      t += Math.round(RATE * (0.01 + 0.07 * Math.abs(rnd())));
      const big = Math.abs(rnd()) > 0.93;
      const amp = (big ? 1 : 0.25 + 0.35 * Math.abs(rnd())) * (rnd() > 0 ? 1 : -1);
      const dec = RATE * (big ? 0.012 : 0.003);
      for (let j = 0; j < dec * 5 && t + j < len; j++) crack[t + j] += amp * rnd() * Math.exp(-j / dec);
    }
    biquad(crack, 'bandpass', 3200, 0.8);
    for (let i = 0; i < len; i++) hiss[i] += crack[i] * 1.6;
    return normalize(loopable(hiss, fade), 0.8);
  });
}

// ── the neighbour's sprinkler ────────────────────────────────────────────────

/**
 * An impact sprinkler, one full sweep: "tch ... tch ... tch" as the arm
 * knocks it around, then the fast "trrrrrr" ratchet back to the start, over
 * a bed of spray pattering on leaves. About 11 s, loops.
 */
export function sprinkler(ctx: BaseAudioContext): AudioBuffer {
  return cached(ctx, 'sprinkler', () => {
    const fade = Math.round(RATE * 0.08);
    const len = Math.round(RATE * 11) + fade;
    const rnd = prng(7);
    const spray = new Float32Array(len);
    for (let i = 0; i < len; i++) spray[i] = rnd();
    biquad(spray, 'highpass', 1800, 0.7);
    biquad(spray, 'lowpass', 7000, 0.7);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = spray[i] * 0.08;

    const tick = (at: number, amp: number, ring: number) => {
      const i0 = Math.round(at * RATE);
      // the arm smacking the nozzle: a knock plus a little metallic ring
      for (let j = 0; j < RATE * 0.12 && i0 + j < len; j++) {
        const tt = j / RATE;
        const knock = rnd() * Math.exp(-tt / 0.004);
        const ping = Math.sin(2 * Math.PI * ring * tt) * Math.exp(-tt / 0.03) * 0.5;
        const gush = spray[(i0 + j) % len] * Math.exp(-tt / 0.06) * 1.6; // a spurt of water
        out[i0 + j] += amp * (knock * 0.8 + ping * 0.6 + gush);
      }
    };
    let t = 0.2;
    for (let k = 0; k < 17; k++) {
      tick(t, 0.75 + 0.2 * Math.abs(rnd()), 3100 + 120 * rnd());
      t += 0.42 + 0.03 * rnd();
    }
    t += 0.15;
    // ratchet back
    for (let k = 0; k < 34; k++) {
      tick(t, 0.32 + 0.08 * Math.abs(rnd()), 3400);
      t += 0.068;
    }
    return normalize(loopable(out, fade), 0.8);
  });
}

// ── cicadas ──────────────────────────────────────────────────────────────────

/** A late-summer cicada swell: a buzzing pulse train that rises and falls, ~7 s. */
export function cicada(ctx: BaseAudioContext): AudioBuffer {
  return cached(ctx, 'cicada', () => {
    const len = RATE * 7;
    const rnd = prng(99);
    const x = new Float32Array(len);
    for (let i = 0; i < len; i++) x[i] = rnd();
    biquad(x, 'bandpass', 4600, 6);
    biquad(x, 'bandpass', 4600, 3);
    for (let i = 0; i < len; i++) {
      const t = i / RATE;
      const swell = Math.pow(Math.sin(Math.PI * Math.min(1, t / 7)), 1.6);
      // ~140 Hz tymbal clicks, beating with a second bug at a slightly different rate
      const pulse = 0.5 + 0.5 * Math.sin(2 * Math.PI * 140 * t) * (0.7 + 0.3 * Math.sin(2 * Math.PI * 3.1 * t));
      x[i] *= swell * pulse;
    }
    return normalize(x, 0.7);
  });
}

// ── the bat ──────────────────────────────────────────────────────────────────

/**
 * The core of a solid wooden-bat crack, baked so every swing peaks the same:
 * a very short bright click, a burst of splintery noise, the ash barrel
 * ringing at its bending modes, the ball's blunt "thock", and the Mendozas'
 * house throwing a little of it back 48 ms later. Four variants (slightly
 * different bats and contact points), each normalized.
 */
export function crack(ctx: BaseAudioContext, variant: number): AudioBuffer {
  const vtn = ((variant % 4) + 4) % 4;
  return cached(ctx, `crack:${vtn}`, () => {
    const len = Math.round(RATE * 0.22);
    const rnd = prng(1000 + vtn * 77);
    const v = [1, 0.965, 1.03, 0.985][vtn];
    const x = new Float32Array(len);
    // click: ~6 ms of high-passed noise
    const click = new Float32Array(len);
    for (let i = 0; i < len; i++) click[i] = rnd() * Math.exp(-i / (RATE * 0.0035));
    biquad(click, 'highpass', 1700, 0.7);
    // splinters: band-passed noise, ~45 ms
    const crackle = new Float32Array(len);
    for (let i = 0; i < len; i++) crackle[i] = rnd() * Math.exp(-i / (RATE * 0.02));
    biquad(crackle, 'bandpass', 2200 * v, 1.1);
    for (let i = 0; i < len; i++) {
      const t = i / RATE;
      let y = click[i] * 1.0 + crackle[i] * 1.6;
      // barrel modes, each starting a hair apart (the wave runs down the bat)
      for (const [f, a, d, dt] of [[610, 0.42, 0.09, 0.0006], [1190, 0.5, 0.07, 0], [2060, 0.3, 0.045, 0.0004], [3270, 0.14, 0.02, 0.0009]] as const) {
        const tt = t - dt;
        if (tt > 0) y += a * Math.sin(2 * Math.PI * f * v * tt * (1 - 0.01 * tt / 0.1)) * Math.exp(-tt / (d / 6.9)) * Math.min(1, tt / 0.0008);
      }
      // the ball squashing: a blunt triangle "thock" falling 320 -> 150 Hz
      const ph = 320 * v * t - (170 * v * t * t) / 0.08;
      const tri = 2 * Math.abs(2 * (ph - Math.floor(ph + 0.5))) - 1;
      y += 0.45 * tri * Math.exp(-t / (0.06 / 6.9)) * Math.min(1, t / 0.001);
      x[i] = y;
    }
    // the house: a darker, quieter copy 48 ms later
    const slap = Math.round(RATE * 0.048);
    const echo = x.slice(0, len - slap);
    biquad(echo, 'lowpass', 1800, 0.7);
    for (let i = 0; i < echo.length; i++) x[i + slap] += echo[i] * 0.16;
    for (let i = 0; i < 300; i++) x[len - 1 - i] *= i / 300;
    return normalize(x, 0.9);
  });
}
