/**
 * Backyard ambience: a soft gusting wind bed, FM bird chirps every few
 * seconds, and now and then a lawnmower droning a few yards over.
 */
import { getBus, isMuted, isReady, onBus, pageHidden, reportError, type Bus } from './context';
import { FLOOR, Patch, rand, wire } from './dsp';

interface Chirp {
  t: number;
  d: number;
  f0: number;
  f1: number;
}

/** One bird's phrase: a "tweet-tweet", a quick trill, or a rising-falling "whee-oo". */
function birdPhrase(t: number): Chirp[] {
  const out: Chirp[] = [];
  const kind = Math.floor(Math.random() * 3);
  if (kind === 0) {
    const f = rand(3600, 4600);
    let x = t;
    for (let k = 0, n = 2 + Math.floor(Math.random() * 2); k < n; k++) {
      out.push({ t: x, d: rand(0.06, 0.09), f0: f, f1: f * 0.68 });
      x += rand(0.15, 0.2);
    }
  } else if (kind === 1) {
    const f = rand(3000, 3800);
    for (let k = 0, n = 5 + Math.floor(Math.random() * 4); k < n; k++) {
      out.push({ t: t + k * 0.065, d: 0.04, f0: f * 0.94, f1: f * 1.06 });
    }
  } else {
    const f = rand(2200, 2800);
    out.push({ t, d: 0.16, f0: f, f1: f * 1.55 }, { t: t + 0.24, d: 0.2, f0: f * 1.45, f1: f * 0.95 });
  }
  return out;
}

class Scene {
  private readonly out: GainNode;
  private readonly wind: Patch;
  private readonly timers = new Set<ReturnType<typeof setTimeout>>();
  private alive = true;

  constructor(private readonly bus: Bus) {
    const t = bus.ctx.currentTime;
    this.out = bus.ctx.createGain();
    this.out.gain.setValueAtTime(0, t);
    this.out.gain.linearRampToValueAtTime(1, t + 2);
    this.out.connect(bus.sfx);
    this.wind = this.makeWind(t);
    this.later(rand(1.5, 4), () => this.bird());
    this.later(rand(12, 30), () => this.mower());
  }

  stop(): void {
    this.alive = false;
    for (const id of this.timers) clearTimeout(id);
    this.timers.clear();
    const t = this.bus.ctx.currentTime;
    const g = this.out.gain;
    const v = g.value;
    g.cancelScheduledValues(t);
    g.setValueAtTime(v, t);
    g.linearRampToValueAtTime(0, t + 1);
    this.wind.stop(t + 1.05);
    setTimeout(() => this.out.disconnect(), 1200);
  }

  private later(sec: number, fn: () => void): void {
    const id = setTimeout(() => {
      this.timers.delete(id);
      if (!this.alive) return;
      try {
        fn();
      } catch (e) {
        reportError(e);
      }
    }, sec * 1000);
    this.timers.add(id);
  }

  /** Should we spend nodes on a one-off sound right now? */
  private audible(): boolean {
    return isReady() && !isMuted() && !pageHidden();
  }

  private makeWind(t: number): Patch {
    const p = new Patch(this.bus);
    const lp = p.filter('lowpass', 520, 0.6);
    const g = p.gain(0.02);
    p.lfo(lp.frequency, 0.05, 260, t); // slow brightness drift
    p.lfo(g.gain, 0.083, 0.011, t); // gusts, made irregular by a second unrelated rate
    p.lfo(g.gain, 0.131, 0.006, t);
    wire(p.noise(t), lp, g, this.out);
    p.seal();
    return p;
  }

  private bird(): void {
    this.later(rand(2.5, 8), () => this.bird());
    if (!this.audible()) return;
    const t = this.bus.ctx.currentTime + 0.05;
    const chirps = birdPhrase(t);
    const last = chirps[chirps.length - 1];
    const end = last.t + last.d + 0.05;
    const peak = rand(0.04, 0.07);

    const p = new Patch(this.bus);
    const pan = p.panner(rand(-0.85, 0.85));
    pan.connect(this.out);
    const amp = p.gain(0);
    const car = p.osc('sine', chirps[0].f0, t, end);
    // FM from a low modulator makes the chirp warble; some birds whistle pure
    if (Math.random() < 0.7) p.lfo(car.frequency, rand(40, 130), rand(150, 450), t, end);
    for (const c of chirps) {
      car.frequency.setValueAtTime(c.f0, c.t);
      car.frequency.exponentialRampToValueAtTime(c.f1, c.t + c.d);
      amp.gain.setValueAtTime(FLOOR, c.t);
      amp.gain.linearRampToValueAtTime(peak, c.t + Math.min(0.012, c.d / 3));
      amp.gain.exponentialRampToValueAtTime(FLOOR, c.t + c.d);
    }
    wire(car, amp, pan);
    p.seal();
  }

  private mower(): void {
    this.later(rand(35, 80), () => this.mower());
    if (!this.audible() || Math.random() < 0.35) return;
    const t = this.bus.ctx.currentTime + 0.05;
    const dur = rand(7, 14);
    const end = t + dur;

    const p = new Patch(this.bus);
    const pan = p.panner(0);
    if ('pan' in pan) {
      // trundling across the neighbour's yard
      const side = Math.random() < 0.5 ? -1 : 1;
      pan.pan.setValueAtTime(side * rand(0.3, 0.8), t);
      pan.pan.linearRampToValueAtTime(-side * rand(0.1, 0.6), end);
    }
    pan.connect(this.out);
    const g = p.gain(0);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.018, t + 2.5);
    g.gain.setValueAtTime(0.018, end - 3);
    g.gain.linearRampToValueAtTime(0, end);
    const lp = p.filter('lowpass', 300, 0.7); // distance eats the top end
    const f = rand(38, 50);
    const a = p.osc('sawtooth', f, t, end + 0.05);
    const b = p.osc('sawtooth', f * 1.013, t, end + 0.05); // slight detune: the engine's chug
    p.lfo(a.frequency, 0.35, f * 0.03, t, end + 0.05).connect(b.frequency); // load wobble
    wire(a, lp);
    wire(b, lp);
    wire(lp, g, pan);
    p.seal();
  }
}

let wanted = false;
let scene: Scene | null = null;

export function setAmbience(on: boolean): void {
  wanted = !!on;
  const bus = getBus();
  if (!bus) return;
  if (wanted && !scene) scene = new Scene(bus);
  else if (!wanted && scene) {
    scene.stop();
    scene = null;
  }
}

onBus(() => {
  if (wanted) setAmbience(true);
});
