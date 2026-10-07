/**
 * A small, strict Web Audio stand-in for tests.
 *
 * It throws wherever real browsers throw (exponential ramp to 0, non-finite
 * values, double start, stop before start), tracks connections so tests can
 * check that every finished sound is fully disconnected, and it can actually
 * render: `ctx.render(seconds)` pulls samples through the graph (oscillators,
 * noise buffers, biquads with the spec's formulas, shapers, panners, delays,
 * param automation and audio-rate modulation), so tests can measure peaks,
 * loudness, NaNs and clicks of every sound without a browser.
 */

export interface World {
  now: number;
  live: Set<FakeNode>;
  sources: Set<FakeSource>;
  created: number;
  violations: string[];
}

export let world: World = fresh();

function fresh(): World {
  return { now: 0, live: new Set(), sources: new Set(), created: 0, violations: [] };
}

export function resetWorld(): World {
  world = fresh();
  return world;
}

function check(ok: boolean, msg: string, E: ErrorConstructor = Error): void {
  if (!ok) {
    world.violations.push(msg);
    throw new E(msg);
  }
}

// ── params ───────────────────────────────────────────────────────────────────

type Ev =
  | { k: 'set'; t: number; v: number }
  | { k: 'lin'; t: number; v: number }
  | { k: 'exp'; t: number; v: number }
  | { k: 'target'; t: number; v: number; c: number };

export class FakeParam {
  private evs: Ev[] = [];
  readonly inputs = new Set<FakeNode>();
  /** cached value for render pass */
  private ci = -1;
  private cv = 0;
  constructor(
    public value: number,
    private readonly label: string,
    readonly ctx: FakeAudioContext,
  ) {}
  private ok(v: number, t: number): void {
    check(Number.isFinite(v), `${this.label}: non-finite value ${v}`, TypeError);
    check(Number.isFinite(t) && t >= 0, `${this.label}: bad time ${t}`, RangeError);
  }
  private add(e: Ev): this {
    // like browsers: an event at the same time and kind replaces; otherwise keep time order
    let i = this.evs.length;
    while (i > 0 && this.evs[i - 1].t > e.t) i--;
    this.evs.splice(i, 0, e);
    return this;
  }
  setValueAtTime(v: number, t: number): this {
    this.ok(v, t);
    return this.add({ k: 'set', t, v });
  }
  linearRampToValueAtTime(v: number, t: number): this {
    this.ok(v, t);
    return this.add({ k: 'lin', t, v });
  }
  exponentialRampToValueAtTime(v: number, t: number): this {
    this.ok(v, t);
    check(v > 0, `${this.label}: exponential ramp to ${v}`, RangeError);
    return this.add({ k: 'exp', t, v });
  }
  setTargetAtTime(v: number, t: number, c: number): this {
    this.ok(v, t);
    check(c >= 0, `${this.label}: negative time constant`, RangeError);
    return this.add({ k: 'target', t, v, c });
  }
  cancelScheduledValues(t: number): this {
    this.ok(0, t);
    this.evs = this.evs.filter((e) => e.t < t);
    return this;
  }
  cancelAndHoldAtTime(t: number): this {
    return this.cancelScheduledValues(t);
  }

  /** The automation value at time t (no audio-rate inputs). */
  intrinsic(t: number): number {
    const evs = this.evs;
    if (!evs.length) return this.value;
    let v = this.value;
    let vt = 0;
    for (let i = 0; i < evs.length; i++) {
      const e = evs[i];
      if (e.t > t) {
        if (e.k === 'lin') {
          const span = e.t - vt;
          return span <= 0 ? e.v : v + (e.v - v) * ((t - vt) / span);
        }
        if (e.k === 'exp') {
          const span = e.t - vt;
          if (span <= 0 || v <= 0) return v;
          return v * Math.pow(e.v / v, (t - vt) / span);
        }
        return v;
      }
      if (e.k === 'target') {
        // value approaches the target from the value at e.t until the next event
        const start = v;
        const next = evs[i + 1];
        const until = next && next.t <= t ? next.t : t;
        v = e.c === 0 ? e.v : e.v + (start - e.v) * Math.exp(-(until - e.t) / e.c);
        vt = until;
        if (!next || next.t > t) return v;
        continue;
      }
      v = e.v;
      vt = e.t;
    }
    return v;
  }

  /** Render-time value: automation plus any connected audio-rate inputs. */
  at(i: number): number {
    if (i === this.ci) return this.cv;
    let v = this.intrinsic(this.ctx.t0 + i / this.ctx.sampleRate);
    for (const n of this.inputs) {
      n.pull(i);
      v += (n.l + n.r) * 0.5;
    }
    this.ci = i;
    this.cv = v;
    return v;
  }
}

// ── nodes ────────────────────────────────────────────────────────────────────

export class FakeNode {
  readonly outputs = new Set<FakeNode | FakeParam>();
  readonly inputs = new Set<FakeNode>();
  l = 0;
  r = 0;
  private ci = -1;
  constructor(readonly ctx: FakeAudioContext) {
    world.created++;
  }
  connect<T>(dest: T): T {
    check(dest != null, 'connect() to nothing');
    const d = dest as unknown as FakeNode | FakeParam;
    this.outputs.add(d);
    d.inputs.add(this);
    world.live.add(this);
    return dest;
  }
  disconnect(): void {
    for (const d of this.outputs) d.inputs.delete(this);
    this.outputs.clear();
    world.live.delete(this);
  }
  /** Sum of inputs into this.l / this.r at sample i (cached per sample). */
  pull(i: number): void {
    if (i === this.ci) return;
    this.ci = i;
    this.l = 0;
    this.r = 0;
    this.process(i);
  }
  protected sumInputs(i: number): [number, number] {
    let l = 0;
    let r = 0;
    for (const n of this.inputs) {
      n.pull(i);
      l += n.l;
      r += n.r;
    }
    return [l, r];
  }
  protected process(i: number): void {
    const [l, r] = this.sumInputs(i);
    this.l = l;
    this.r = r;
  }
}

export class FakeSource extends FakeNode {
  onended: (() => void) | null = null;
  started = false;
  startAt = 0;
  stopAt = Infinity;
  start(when = 0): void {
    check(!this.started, 'start() called twice');
    check(Number.isFinite(when) && when >= 0, `bad start time ${when}`, RangeError);
    this.started = true;
    this.startAt = when;
    world.sources.add(this);
  }
  stop(when = 0): void {
    check(this.started, 'stop() before start()');
    check(Number.isFinite(when) && when >= 0, `bad stop time ${when}`, RangeError);
    this.stopAt = when;
  }
  protected playing(i: number): boolean {
    const t = this.ctx.t0 + i / this.ctx.sampleRate;
    return this.started && t >= this.startAt && t < this.stopAt;
  }
}

export class FakePeriodicWave {
  constructor(
    readonly real: Float32Array,
    readonly imag: Float32Array,
    readonly norm: number,
  ) {}
}

export class FakeOscillator extends FakeSource {
  type = 'sine';
  readonly frequency: FakeParam;
  readonly detune: FakeParam;
  private phase = 0;
  private wave: FakePeriodicWave | null = null;
  constructor(ctx: FakeAudioContext) {
    super(ctx);
    this.frequency = new FakeParam(440, 'osc.frequency', ctx);
    this.detune = new FakeParam(0, 'osc.detune', ctx);
  }
  setPeriodicWave(w: FakePeriodicWave): void {
    this.wave = w;
    this.type = 'custom';
  }
  protected override process(i: number): void {
    if (!this.playing(i)) return;
    const f = this.frequency.at(i) * Math.pow(2, this.detune.at(i) / 1200);
    const ph = this.phase;
    this.phase = (ph + f / this.ctx.sampleRate) % 1;
    if (this.phase < 0) this.phase += 1;
    let v: number;
    switch (this.type) {
      case 'square':
        v = ph < 0.5 ? 1 : -1;
        break;
      case 'sawtooth':
        v = 2 * ph - 1;
        break;
      case 'triangle':
        v = ph < 0.25 ? 4 * ph : ph < 0.75 ? 2 - 4 * ph : 4 * ph - 4;
        break;
      case 'custom': {
        const w = this.wave!;
        v = 0;
        for (let h = 1; h < w.real.length; h++) {
          if (h * f > this.ctx.sampleRate / 2) break;
          const a = 2 * Math.PI * h * ph;
          v += w.real[h] * Math.cos(a) + w.imag[h] * Math.sin(a);
        }
        v *= w.norm;
        break;
      }
      default:
        v = Math.sin(2 * Math.PI * ph);
    }
    this.l = v;
    this.r = v;
  }
}

export class FakeBuffer {
  readonly duration: number;
  private readonly data: Float32Array[];
  constructor(
    readonly numberOfChannels: number,
    readonly length: number,
    readonly sampleRate: number,
  ) {
    this.data = Array.from({ length: numberOfChannels }, () => new Float32Array(length));
    this.duration = length / sampleRate;
  }
  getChannelData(c = 0): Float32Array {
    return this.data[c];
  }
}

export class FakeBufferSource extends FakeSource {
  buffer: FakeBuffer | null = null;
  loop = false;
  readonly playbackRate: FakeParam;
  private pos = -1;
  private offset = 0;
  constructor(ctx: FakeAudioContext) {
    super(ctx);
    this.playbackRate = new FakeParam(1, 'playbackRate', ctx);
  }
  override start(when = 0, offset = 0): void {
    check(this.buffer !== null, 'buffer source started without a buffer');
    check(Number.isFinite(offset) && offset >= 0, `bad offset ${offset}`, RangeError);
    super.start(when);
    this.offset = offset;
    if (!this.loop && this.buffer) this.stopAt = Math.min(this.stopAt, when + this.buffer.duration - offset);
  }
  protected override process(i: number): void {
    if (!this.playing(i) || !this.buffer) return;
    const b = this.buffer;
    if (this.pos < 0) this.pos = this.offset * b.sampleRate;
    const len = b.length;
    let p = this.pos;
    if (this.loop) p %= len;
    else if (p >= len) return;
    const k = Math.floor(p);
    const fr = p - k;
    const d0 = b.getChannelData(0);
    const d1 = b.numberOfChannels > 1 ? b.getChannelData(1) : d0;
    const k1 = this.loop ? (k + 1) % len : Math.min(k + 1, len - 1);
    this.l = d0[k] + (d0[k1] - d0[k]) * fr;
    this.r = d1[k] + (d1[k1] - d1[k]) * fr;
    this.pos = p + this.playbackRate.at(i) * (b.sampleRate / this.ctx.sampleRate);
  }
}

export class FakeGain extends FakeNode {
  readonly gain: FakeParam;
  constructor(ctx: FakeAudioContext) {
    super(ctx);
    this.gain = new FakeParam(1, 'gain', ctx);
  }
  protected override process(i: number): void {
    if (!this.inputs.size) return;
    const [l, r] = this.sumInputs(i);
    const g = this.gain.at(i);
    this.l = l * g;
    this.r = r * g;
  }
}

export class FakeBiquad extends FakeNode {
  type = 'lowpass';
  readonly frequency: FakeParam;
  readonly Q: FakeParam;
  readonly gain: FakeParam;
  readonly detune: FakeParam;
  private c = [1, 0, 0, 0, 0];
  private s = [0, 0, 0, 0, 0, 0, 0, 0]; // x1 x2 y1 y2 per channel
  private key = '';
  constructor(ctx: FakeAudioContext) {
    super(ctx);
    this.frequency = new FakeParam(350, 'filter.frequency', ctx);
    this.Q = new FakeParam(1, 'filter.Q', ctx);
    this.gain = new FakeParam(0, 'filter.gain', ctx);
    this.detune = new FakeParam(0, 'filter.detune', ctx);
  }
  private coeffs(i: number): void {
    const sr = this.ctx.sampleRate;
    const f = Math.min(sr / 2, Math.max(0, this.frequency.at(i) * Math.pow(2, this.detune.at(i) / 1200)));
    const Q = this.Q.at(i);
    const G = this.gain.at(i);
    const key = `${f.toFixed(2)}|${Q.toFixed(3)}|${G.toFixed(2)}`;
    if (key === this.key) return;
    this.key = key;
    const w0 = (2 * Math.PI * f) / sr;
    const cw = Math.cos(w0);
    const sw = Math.sin(w0);
    const A = Math.pow(10, G / 40);
    let b0 = 1, b1 = 0, b2 = 0, a0 = 1, a1 = 0, a2 = 0;
    switch (this.type) {
      case 'lowpass': {
        const al = sw / (2 * Math.pow(10, Q / 20)); // Q in dB, per the spec
        b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al;
        break;
      }
      case 'highpass': {
        const al = sw / (2 * Math.pow(10, Q / 20));
        b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al;
        break;
      }
      case 'bandpass': {
        const al = sw / (2 * Math.max(Q, 1e-4));
        b0 = al; b1 = 0; b2 = -al; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al;
        break;
      }
      case 'notch': {
        const al = sw / (2 * Math.max(Q, 1e-4));
        b0 = 1; b1 = -2 * cw; b2 = 1; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al;
        break;
      }
      case 'allpass': {
        const al = sw / (2 * Math.max(Q, 1e-4));
        b0 = 1 - al; b1 = -2 * cw; b2 = 1 + al; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al;
        break;
      }
      case 'peaking': {
        const al = sw / (2 * Math.max(Q, 1e-4));
        b0 = 1 + al * A; b1 = -2 * cw; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cw; a2 = 1 - al / A;
        break;
      }
      case 'lowshelf':
      case 'highshelf': {
        const S = 1;
        const al = (sw / 2) * Math.sqrt((A + 1 / A) * (1 / S - 1) + 2);
        const sa = 2 * Math.sqrt(A) * al;
        if (this.type === 'lowshelf') {
          b0 = A * (A + 1 - (A - 1) * cw + sa); b1 = 2 * A * (A - 1 - (A + 1) * cw); b2 = A * (A + 1 - (A - 1) * cw - sa);
          a0 = A + 1 + (A - 1) * cw + sa; a1 = -2 * (A - 1 + (A + 1) * cw); a2 = A + 1 + (A - 1) * cw - sa;
        } else {
          b0 = A * (A + 1 + (A - 1) * cw + sa); b1 = -2 * A * (A - 1 + (A + 1) * cw); b2 = A * (A + 1 + (A - 1) * cw - sa);
          a0 = A + 1 - (A - 1) * cw + sa; a1 = 2 * (A - 1 - (A + 1) * cw); a2 = A + 1 - (A - 1) * cw - sa;
        }
        break;
      }
    }
    this.c = [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
  }
  protected override process(i: number): void {
    if (!this.inputs.size && this.s.every((x) => x === 0)) return;
    if ((i & 7) === 0 || !this.key) this.coeffs(i);
    const [xl, xr] = this.sumInputs(i);
    const [b0, b1, b2, a1, a2] = this.c;
    const s = this.s;
    const yl = b0 * xl + b1 * s[0] + b2 * s[1] - a1 * s[2] - a2 * s[3];
    s[1] = s[0]; s[0] = xl; s[3] = s[2]; s[2] = Math.abs(yl) < 1e-25 ? 0 : yl;
    const yr = b0 * xr + b1 * s[4] + b2 * s[5] - a1 * s[6] - a2 * s[7];
    s[5] = s[4]; s[4] = xr; s[7] = s[6]; s[6] = Math.abs(yr) < 1e-25 ? 0 : yr;
    this.l = yl;
    this.r = yr;
  }
}

export class FakePanner extends FakeNode {
  readonly pan: FakeParam;
  constructor(ctx: FakeAudioContext) {
    super(ctx);
    this.pan = new FakeParam(0, 'pan', ctx);
  }
  protected override process(i: number): void {
    const [l, r] = this.sumInputs(i);
    const p = Math.min(1, Math.max(-1, this.pan.at(i)));
    const x = (p + 1) / 2;
    const m = (l + r) / 2; // our sounds are mono into the panner
    this.l = m * Math.cos((x * Math.PI) / 2);
    this.r = m * Math.sin((x * Math.PI) / 2);
  }
}

export class FakeShaper extends FakeNode {
  curve: Float32Array | null = null;
  oversample = 'none';
  private shape(x: number): number {
    const c = this.curve;
    if (!c || c.length < 2) return x;
    const pos = ((Math.min(1, Math.max(-1, x)) + 1) / 2) * (c.length - 1);
    const k = Math.floor(pos);
    const k1 = Math.min(k + 1, c.length - 1);
    return c[k] + (c[k1] - c[k]) * (pos - k);
  }
  protected override process(i: number): void {
    const [l, r] = this.sumInputs(i);
    this.l = this.shape(l);
    this.r = this.shape(r);
  }
}

export class FakeDelay extends FakeNode {
  readonly delayTime: FakeParam;
  private readonly bl: Float32Array;
  private readonly br: Float32Array;
  constructor(ctx: FakeAudioContext, max = 1) {
    super(ctx);
    this.delayTime = new FakeParam(0, 'delayTime', ctx);
    const n = Math.ceil(max * ctx.sampleRate) + 2;
    this.bl = new Float32Array(n);
    this.br = new Float32Array(n);
  }
  private di = -1;
  override pull(i: number): void {
    // output first (from the past), then feed the input in: cycles through a delay are fine
    if (i === this.di) return;
    this.di = i;
    const n = this.bl.length;
    const idx = (j: number) => ((j % n) + n) % n;
    const d = Math.max(1, Math.min(n - 2, this.delayTime.at(i) * this.ctx.sampleRate));
    const k = Math.floor(i - d);
    const fr = i - d - k;
    const outL = k >= 0 ? this.bl[idx(k)] + (this.bl[idx(k + 1)] - this.bl[idx(k)]) * fr : 0;
    const outR = k >= 0 ? this.br[idx(k)] + (this.br[idx(k + 1)] - this.br[idx(k)]) * fr : 0;
    this.l = outL;
    this.r = outR;
    let l = 0, r = 0;
    for (const nd of this.inputs) {
      nd.pull(i);
      l += nd.l;
      r += nd.r;
    }
    this.bl[idx(i)] = l;
    this.br[idx(i)] = r;
    this.l = outL;
    this.r = outR;
  }
}

export class FakeCompressor extends FakeNode {
  readonly threshold: FakeParam;
  readonly knee: FakeParam;
  readonly ratio: FakeParam;
  readonly attack: FakeParam;
  readonly release: FakeParam;
  reduction = 0;
  private envDb = -120;
  /** Chrome and Safari delay the signal ~6 ms so the gain can come down before a transient arrives */
  private readonly look: Float32Array[] = [new Float32Array(1), new Float32Array(1)];
  private li = 0;
  constructor(ctx: FakeAudioContext) {
    super(ctx);
    this.threshold = new FakeParam(-24, 'threshold', ctx);
    this.knee = new FakeParam(30, 'knee', ctx);
    this.ratio = new FakeParam(12, 'ratio', ctx);
    this.attack = new FakeParam(0.003, 'attack', ctx);
    this.release = new FakeParam(0.25, 'release', ctx);
  }
  /** Static curve (dB in -> dB out) with a soft knee. */
  private curve(x: number, T: number, K: number, R: number): number {
    if (x < T - K / 2) return x;
    if (x > T + K / 2) return T + (x - T) / R;
    const d = x - T + K / 2;
    return x + ((1 / R - 1) * d * d) / (2 * K);
  }
  protected override process(i: number): void {
    const [l, r] = this.sumInputs(i);
    const T = this.threshold.at(i), K = this.knee.at(i), R = this.ratio.at(i);
    const lvl = Math.max(Math.abs(l), Math.abs(r));
    const xDb = lvl > 1e-6 ? 20 * Math.log10(lvl) : -120;
    const sr = this.ctx.sampleRate;
    const a = Math.exp(-1 / (Math.max(1e-4, this.attack.at(i)) * sr));
    const rl = Math.exp(-1 / (Math.max(1e-3, this.release.at(i)) * sr));
    this.envDb = xDb > this.envDb ? a * this.envDb + (1 - a) * xDb : rl * this.envDb + (1 - rl) * xDb;
    const gr = this.curve(this.envDb, T, K, R) - this.envDb;
    // Chrome adds automatic make-up gain: (1 / gain at 0 dBFS) ^ 0.6
    const makeup = Math.pow(Math.pow(10, -(this.curve(0, T, K, R)) / 20), 0.6);
    const g = Math.pow(10, gr / 20) * makeup;
    this.reduction = gr;
    const n = Math.round(0.006 * sr);
    if (this.look[0].length !== n) this.look.splice(0, 2, new Float32Array(n), new Float32Array(n));
    const k = this.li++ % n;
    const dl = this.look[0][k], dr = this.look[1][k];
    this.look[0][k] = l;
    this.look[1][k] = r;
    this.l = dl * g;
    this.r = dr * g;
  }
}

/** Records the summed output as a destination does. */
class FakeDestination extends FakeNode {}

export class FakeAudioContext {
  state: 'suspended' | 'running' | 'closed' | 'interrupted' = 'suspended';
  sampleRate: number;
  readonly destination: FakeNode;
  onstatechange: (() => void) | null = null;
  /** render bookkeeping: time of sample 0 and the next sample index to render */
  t0 = 0;
  private next = 0;
  static instances: FakeAudioContext[] = [];
  /** Tests can make resume() fail (an iOS interruption that won't let go). */
  static resumeFails = false;
  constructor(opts?: { sampleRate?: number }) {
    this.sampleRate = opts?.sampleRate ?? FakeAudioContext.defaultRate;
    this.destination = new FakeDestination(this);
    FakeAudioContext.instances.push(this);
  }
  static defaultRate = 22050;
  /** Nodes whose peak level is tracked while rendering (see watch()). */
  readonly peaks = new Map<FakeNode, number>();
  watch(n: unknown): void {
    this.peaks.set(n as FakeNode, 0);
  }
  get currentTime(): number {
    return world.now;
  }
  setState(s: FakeAudioContext['state']): void {
    this.state = s;
    this.onstatechange?.();
  }
  resume(): Promise<void> {
    if (this.state === 'closed') return Promise.reject(new Error('closed'));
    if (FakeAudioContext.resumeFails) return Promise.resolve();
    this.setState('running');
    return Promise.resolve();
  }
  suspend(): Promise<void> {
    if (this.state !== 'closed') this.setState('suspended');
    return Promise.resolve();
  }
  close(): Promise<void> {
    this.setState('closed');
    return Promise.resolve();
  }
  createGain(): FakeGain {
    return new FakeGain(this);
  }
  createOscillator(): FakeOscillator {
    return new FakeOscillator(this);
  }
  createBufferSource(): FakeBufferSource {
    return new FakeBufferSource(this);
  }
  createBuffer(channels: number, length: number, rate: number): FakeBuffer {
    check(length > 0 && Number.isInteger(length), `bad buffer length ${length}`, RangeError);
    return new FakeBuffer(channels, length, rate);
  }
  createBiquadFilter(): FakeBiquad {
    return new FakeBiquad(this);
  }
  createStereoPanner(): FakePanner {
    return new FakePanner(this);
  }
  createWaveShaper(): FakeShaper {
    return new FakeShaper(this);
  }
  createDelay(max = 1): FakeDelay {
    return new FakeDelay(this, max);
  }
  createDynamicsCompressor(): FakeCompressor {
    return new FakeCompressor(this);
  }
  createPeriodicWave(real: Float32Array, imag: Float32Array, opts?: { disableNormalization?: boolean }): FakePeriodicWave {
    check(real.length === imag.length && real.length >= 2, 'periodic wave: bad arrays');
    let norm = 1;
    if (!opts?.disableNormalization) {
      // browsers scale the wave so its peak is 1
      let peak = 0;
      for (let s = 0; s < 256; s++) {
        const ph = s / 256;
        let v = 0;
        for (let h = 1; h < real.length; h++) v += real[h] * Math.cos(2 * Math.PI * h * ph) + imag[h] * Math.sin(2 * Math.PI * h * ph);
        peak = Math.max(peak, Math.abs(v));
      }
      norm = peak > 0 ? 1 / peak : 1;
    }
    return new FakePeriodicWave(Float32Array.from(real), Float32Array.from(imag), norm);
  }

  /**
   * Render from the current render position up to `seconds` past the moment
   * rendering started, pulling from `tap` (default: the destination). Returns
   * [left, right]. The world clock is moved along so currentTime follows.
   */
  render(seconds: number, tap: FakeNode = this.destination): [Float32Array, Float32Array] {
    if (this.next === 0) this.t0 = world.now;
    const n = Math.round(seconds * this.sampleRate);
    const L = new Float32Array(n);
    const R = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const i = this.next++;
      tap.pull(i);
      L[k] = tap.l;
      R[k] = tap.r;
      for (const [nd, pk] of this.peaks) {
        const v = Math.max(Math.abs(nd.l), Math.abs(nd.r));
        if (v > pk) this.peaks.set(nd, v);
      }
      if ((k & 127) === 0) {
        world.now = this.t0 + i / this.sampleRate;
        endSources(world.now);
      }
    }
    world.now = this.t0 + this.next / this.sampleRate;
    endSources(world.now);
    return [L, R];
  }
}

export function endSources(now: number): void {
  for (const s of [...world.sources]) {
    if (s.stopAt <= now) {
      world.sources.delete(s);
      s.onended?.();
    }
  }
}

// ── measurements ─────────────────────────────────────────────────────────────

export interface Stats {
  peak: number;
  rms: number;
  /** loudest 50 ms RMS window */
  loud: number;
  nan: boolean;
  first: number;
  /** absolute level of the last 5 ms */
  tail: number;
  /** biggest jump between neighbouring samples, relative to the local level */
  jump: number;
}

export function stats(...chans: Float32Array[]): Stats {
  let peak = 0, sum = 0, nan = false, jump = 0;
  const n = chans[0].length;
  for (const c of chans) {
    for (let k = 0; k < n; k++) {
      const x = c[k];
      if (!Number.isFinite(x)) nan = true;
      const a = Math.abs(x);
      if (a > peak) peak = a;
      sum += x * x;
      if (k > 0) jump = Math.max(jump, Math.abs(x - c[k - 1]));
    }
  }
  const win = 1100; // ~50 ms at 22050
  let loud = 0;
  for (const c of chans) {
    let acc = 0;
    for (let k = 0; k < n; k++) {
      acc += c[k] * c[k];
      if (k >= win) acc -= c[k - win] * c[k - win];
      if (k >= win - 1) loud = Math.max(loud, Math.sqrt(Math.max(0, acc) / win));
    }
  }
  const tailN = Math.min(n, 110);
  let tail = 0;
  for (const c of chans) for (let k = n - tailN; k < n; k++) tail = Math.max(tail, Math.abs(c[k]));
  return {
    peak, nan, jump, tail, loud,
    rms: Math.sqrt(sum / (n * chans.length)),
    first: Math.max(...chans.map((c) => Math.abs(c[0]))),
  };
}

/** One-pole high-pass: roughly what a phone speaker can reproduce. */
export function phoneSpeaker(x: Float32Array, sr: number, cutoff = 300): Float32Array {
  const out = new Float32Array(x.length);
  const rc = 1 / (2 * Math.PI * cutoff);
  const a = rc / (rc + 1 / sr);
  let py = 0, px = 0;
  for (let k = 0; k < x.length; k++) {
    py = a * (py + x[k] - px);
    px = x[k];
    out[k] = py;
  }
  // twice, for a steeper slope
  const out2 = new Float32Array(x.length);
  py = 0; px = 0;
  for (let k = 0; k < x.length; k++) {
    py = a * (py + out[k] - px);
    px = out[k];
    out2[k] = py;
  }
  return out2;
}

export const db = (x: number): number => 20 * Math.log10(Math.max(x, 1e-9));
