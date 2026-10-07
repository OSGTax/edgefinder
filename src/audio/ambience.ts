/**
 * The Mendozas' backyard on a late-summer afternoon, as heard from home plate.
 *
 * Always there, very quietly: a gusting breeze, Mr. Mendoza's grill sizzling
 * off to the left, and a neighbour's sprinkler ticking away behind the hedge.
 * Now and then: birds, a cicada swell, a lawnmower a few yards over, a dog
 * down the street, a bike bell, the back screen door, Mr. Mendoza flipping
 * something, and once in a long while the ice-cream truck two streets over
 * playing its little tune as it drives past.
 */
import { cicada, sizzle, sprinkler } from './bake';
import { getBus, isMuted, isReady, onBus, onBusLost, pageHidden, reportError, type Bus } from './context';
import { FLOOR, Patch, hits, mtof, rand, softClip, wire } from './dsp';
import { buildSfx } from './sfx';
import type { SfxName } from './types';

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

/** The ice-cream truck's tune (original): [midi | 0 rest, eighths]. Plays in D, a bit out of tune. */
const TRUCK: ReadonlyArray<readonly [number, number]> = [
  [81, 1], [78, 1], [81, 1], [83, 1], [81, 1], [78, 1], [74, 2],
  [76, 1], [78, 1], [79, 1], [76, 1], [78, 2], [74, 2],
  [81, 1], [78, 1], [81, 1], [83, 1], [81, 1], [78, 1], [74, 1], [78, 1],
  [76, 2], [69, 2], [74, 3], [0, 1],
];

class Scene {
  private readonly out: GainNode;
  private readonly beds: Patch[] = [];
  private readonly timers = new Set<ReturnType<typeof setTimeout>>();
  private alive = true;

  constructor(readonly bus: Bus) {
    const t = bus.ctx.currentTime;
    this.out = bus.ctx.createGain();
    this.out.gain.setValueAtTime(0, t);
    this.out.gain.linearRampToValueAtTime(1, t + 2);
    this.out.connect(bus.sfx);
    this.beds.push(this.makeWind(t), this.makeGrill(t), this.makeSprinkler(t));
    this.later(rand(1.5, 4), () => this.bird());
    this.later(rand(12, 30), () => this.mower());
    this.later(rand(8, 20), () => this.cicadas());
    this.later(rand(20, 45), () => this.grillFlip());
    this.later(rand(40, 90), () => this.neighbourhood());
    this.later(rand(70, 140), () => this.iceCreamTruck());
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
    for (const b of this.beds) b.stop(t + 1.05);
    setTimeout(() => this.out.disconnect(), 1200);
  }

  /** The bus died under us (rebuilt after an interruption). */
  kill(): void {
    this.alive = false;
    for (const id of this.timers) clearTimeout(id);
    this.timers.clear();
    try {
      this.out.disconnect();
    } catch {
      /* gone */
    }
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

  /** A send for something `far` away (0 = here .. 1 = down the street): quieter and duller. */
  private place(p: Patch, pan: number, far: number, level = 1): AudioNode {
    const panner = p.panner(pan);
    const lp = p.filter('lowpass', 9000 * Math.pow(0.12, far), 0);
    wire(lp, panner, p.gain(level * (1 - 0.75 * far)), this.out);
    return lp;
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

  private makeGrill(t: number): Patch {
    // the grill is behind third base, a few steps off: always on, never loud
    const p = new Patch(this.bus);
    const g = p.gain(0);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.022, t + 3);
    p.lfo(g.gain, 0.04, 0.006, t);
    wire(p.play(sizzle(p.ctx), t, 1, true, undefined, rand(0, 3)), g, this.place(p, -0.55, 0.15));
    p.seal();
    return p;
  }

  private makeSprinkler(t: number): Patch {
    // somebody's impact sprinkler, over the hedge in left
    const p = new Patch(this.bus);
    const pan = p.panner(-0.7);
    if ('pan' in pan) {
      pan.pan.setValueAtTime(-0.7, t);
      p.lfo(pan.pan, 1 / 11, 0.2, t); // it turns with the sweep
    }
    const g = p.gain(0);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.03, t + 4);
    wire(p.play(sprinkler(p.ctx), t, 1, true, undefined, rand(0, 10)), p.filter('lowpass', 3800, 0), g, pan, this.out);
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

  private cicadas(): void {
    this.later(rand(25, 60), () => this.cicadas());
    if (!this.audible()) return;
    const p = new Patch(this.bus);
    const t = this.bus.ctx.currentTime + 0.05;
    wire(p.play(cicada(p.ctx), t, rand(0.96, 1.04)), p.gain(rand(0.012, 0.02)), this.place(p, rand(-0.9, 0.9), 0.3));
    p.seal();
  }

  private mower(): void {
    this.later(rand(35, 80), () => this.mower());
    if (!this.audible() || Math.random() < 0.35) return;
    const t = this.bus.ctx.currentTime + 0.05;
    const dur = rand(8, 15);
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
    g.gain.linearRampToValueAtTime(0.02, t + 2.5);
    g.gain.setValueAtTime(0.02, end - 3);
    g.gain.linearRampToValueAtTime(0, end);
    const lp = p.filter('lowpass', 420, 0.7); // distance eats the top end
    const f = rand(36, 46);
    // a two-stroke engine: a buzzy pulse at the firing rate, a second cylinder of sound a hair off
    const a = p.osc('sawtooth', f, t, end + 0.05);
    const b = p.osc('square', f * 2.02, t, end + 0.05);
    const bg = p.gain(0.35);
    // it bogs down in thick grass and picks back up
    for (let x = t + rand(1.5, 3); x < end - 1.5; x += rand(2, 4)) {
      a.frequency.setTargetAtTime(f * rand(0.82, 0.9), x, 0.15);
      a.frequency.setTargetAtTime(f, x + rand(0.5, 0.9), 0.4);
      b.frequency.setTargetAtTime(f * 2.02 * 0.86, x, 0.15);
      b.frequency.setTargetAtTime(f * 2.02, x + 0.7, 0.4);
    }
    wire(a, lp);
    wire(b, bg, lp);
    // the blade whine, far off
    const whine = p.gain(0.05);
    wire(p.osc('triangle', f * 9, t, end + 0.05), whine, lp);
    wire(lp, g, pan);
    p.seal();
  }

  private grillFlip(): void {
    this.later(rand(25, 55), () => this.grillFlip());
    if (!this.audible()) return;
    // Mr. Mendoza flips something: a spatula scrape and a fresh burst of sizzle
    const p = new Patch(this.bus);
    const t = this.bus.ctx.currentTime + 0.05;
    const dest = this.place(p, -0.55, 0.15, 0.5);
    const scrape = p.burst(dest, { t, attack: 0.02, hold: 0.06, peak: 0.25, decay: 0.08, freq: 2400, to: 3400, q: 3 });
    void scrape;
    p.tone(dest, { t: t + 0.12, freq: 1900, peak: 0.04, decay: 0.08 }); // the spatula's ting on the grate
    const g = p.gain(0);
    g.gain.setValueAtTime(FLOOR, t + 0.15);
    g.gain.linearRampToValueAtTime(0.12, t + 0.25);
    g.gain.exponentialRampToValueAtTime(FLOOR, t + 2.2);
    wire(p.play(sizzle(p.ctx), t + 0.15, 1.08, false, t + 2.3, rand(0, 1.5)), g, dest);
    p.seal();
  }

  private neighbourhood(): void {
    this.later(rand(45, 110), () => this.neighbourhood());
    if (!this.audible()) return;
    const r = Math.random();
    const p = new Patch(this.bus);
    try {
      if (r < 0.35) this.oneShot(p, 'dogBark', rand(-0.9, 0.9), 0.75, rand(0.1, 0.4));
      else if (r < 0.65) this.oneShot(p, 'screenDoor', rand(-0.3, 0.3), 0.25, 0.3);
      else this.bikeBell(p);
    } finally {
      p.seal();
    }
  }

  private oneShot(p: Patch, name: SfxName, pan: number, far: number, intensity: number): void {
    buildSfx(p, this.place(p, pan, far, 0.6), name, { intensity }, 0.05);
  }

  private bikeBell(p: Patch): void {
    // a kid on a bike out on the street: "brring-brring"
    const t = this.bus.ctx.currentTime + 0.05;
    const dest = this.place(p, rand(-0.6, 0.6), 0.55, 0.35);
    for (const dt of [0, 0.32]) {
      const t0 = t + dt;
      const g = p.gain(0);
      const end = hits(g.gain, [[t0, 0.3], [t0 + 0.045, 0.25], [t0 + 0.09, 0.22]], 0.002, 0.04);
      g.gain.exponentialRampToValueAtTime(FLOOR, end + 0.5);
      for (const [f, a] of [[2450, 1], [2493, 0.8], [5580, 0.3]] as const) wire(p.osc('sine', f, t0, end + 0.55), p.gain(a), g);
      g.connect(dest);
    }
  }

  private iceCreamTruck(): void {
    this.later(rand(160, 300), () => this.iceCreamTruck());
    if (!this.audible()) return;
    const p = new Patch(this.bus);
    const t = this.bus.ctx.currentTime + 0.1;
    const eighth = 0.2;
    const loops = 2;
    const tuneLen = TRUCK.reduce((s, [, n]) => s + n, 0) * eighth;
    const end = t + tuneLen * loops + 0.5;
    const side = Math.random() < 0.5 ? -1 : 1;

    // a tinny loudspeaker on the roof, two streets over
    const pan = p.panner(side * 0.8);
    if ('pan' in pan) {
      pan.pan.setValueAtTime(side * 0.8, t);
      pan.pan.linearRampToValueAtTime(-side * 0.7, end);
    }
    const vol = p.gain(0);
    vol.gain.setValueAtTime(0, t);
    vol.gain.linearRampToValueAtTime(0.05, t + tuneLen * 0.8);
    vol.gain.linearRampToValueAtTime(0.05, t + tuneLen * 1.2);
    vol.gain.linearRampToValueAtTime(0, end);
    const drive = p.gain(1.6);
    const horn = p.shaper(softClip());
    wire(drive, horn, p.filter('bandpass', 1300, 0.8), p.filter('lowpass', 2600, 0), vol, pan, this.out);

    const amp = p.gain(0);
    const chime = p.osc('triangle', mtof(81), t, end);
    const shine = p.osc('sine', mtof(93), t, end);
    wire(chime, amp);
    wire(shine, p.gain(0.3), amp);
    amp.connect(drive);
    // the truck goes by: a little Doppler bend in the middle, and the tape wobbles
    const det = [chime.detune, shine.detune];
    for (const d of det) {
      d.setValueAtTime(25, t);
      d.linearRampToValueAtTime(25, t + tuneLen * 0.9);
      d.linearRampToValueAtTime(-30, t + tuneLen * 1.1);
    }
    const wob = p.gain(9);
    wire(p.osc('sine', 0.7, t, end), wob);
    for (const d of det) wob.connect(d);

    const notesList: Array<[number, number]> = [];
    let x = t;
    for (let l = 0; l < loops; l++) {
      for (const [m, n] of TRUCK) {
        if (m) {
          chime.frequency.setValueAtTime(mtof(m), x);
          shine.frequency.setValueAtTime(mtof(m + 12), x);
          notesList.push([x, 0.8]);
        }
        x += n * eighth;
      }
    }
    hits(amp.gain, notesList, 0.004, 0.3);
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

onBusLost((old) => {
  if (scene?.bus === old) {
    scene.kill();
    scene = null;
  }
});
