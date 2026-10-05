import { afterEach, describe, expect, it, vi } from 'vitest';

type AudioModule = typeof import('../src/audio/index');

const EXPECTED_SFX = [
  'batCrack', 'batTink', 'whiff', 'mittPop', 'catch', 'bounce', 'fence', 'splash', 'leaves',
  'cheer', 'bigCheer', 'aww', 'strike', 'out', 'safe', 'homeRun', 'special', 'uiTap', 'uiBack',
  'uiSelect', 'whistle', 'dogBark', 'throw',
];
const EXPECTED_TRACKS = ['title', 'game', 'victory', 'season'];

async function freshModule(): Promise<AudioModule> {
  vi.resetModules();
  return import('../src/audio/index');
}

function exerciseEverything(mod: AudioModule): void {
  const { audio, SFX_NAMES, MUSIC_TRACKS } = mod;
  audio.unlock();
  audio.unlock();
  for (const name of SFX_NAMES) {
    audio.play(name);
    audio.play(name, { intensity: 0, pan: -1 });
    audio.play(name, { intensity: 1, pan: 1 });
    audio.play(name, { intensity: Number.NaN, pan: 7 });
  }
  for (const track of MUSIC_TRACKS) audio.playMusic(track);
  audio.stopMusic();
  audio.setAmbience(true);
  audio.setAmbience(false);
  audio.setSfxVolume(0.5);
  audio.setSfxVolume(-3);
  audio.setMusicVolume(2);
  audio.setMusicVolume(Number.NaN);
  audio.setMuted(true);
  audio.setMuted(false);
  // garbage from untyped callers must not throw either
  (audio.play as (n: unknown) => void)('notASound');
  (audio.playMusic as (n: unknown) => void)(undefined);
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('audio without Web Audio', () => {
  it('exports the full sound and track lists', async () => {
    const { SFX_NAMES, MUSIC_TRACKS } = await freshModule();
    expect([...SFX_NAMES].sort()).toEqual([...EXPECTED_SFX].sort());
    expect([...MUSIC_TRACKS].sort()).toEqual([...EXPECTED_TRACKS].sort());
  });

  it('is a silent no-op in Node (no window)', async () => {
    const mod = await freshModule();
    expect(mod.audio.ready).toBe(false);
    expect(() => exerciseEverything(mod)).not.toThrow();
    expect(mod.audio.ready).toBe(false);
  });

  it('is a silent no-op in a browser without AudioContext', async () => {
    vi.stubGlobal('window', {});
    const mod = await freshModule();
    expect(() => exerciseEverything(mod)).not.toThrow();
    expect(mod.audio.ready).toBe(false);
  });

  it('survives an AudioContext constructor that throws', async () => {
    vi.stubGlobal('window', {
      AudioContext: class {
        constructor() {
          throw new Error('NotAllowedError');
        }
      },
    });
    const mod = await freshModule();
    const errors: unknown[] = [];
    mod.setAudioErrorHandler((e) => errors.push(e));
    expect(() => exerciseEverything(mod)).not.toThrow();
    expect(mod.audio.ready).toBe(false);
    expect(errors.length).toBeGreaterThan(0); // reported, not thrown
  });
});

describe('songs', () => {
  it('parse, are whole bars, and the victory fanfare is ~4 s', async () => {
    vi.resetModules();
    const { SONGS } = await import('../src/audio/tracks');
    for (const song of Object.values(SONGS)) {
      for (const part of song.parts) {
        expect(part.len % 16).toBe(0);
        expect(song.length % part.len).toBe(0);
      }
    }
    expect(SONGS.title.length).toBe(256); // 16 bars, A + B
    const v = SONGS.victory;
    const seconds = v.length * v.stepDur + 0.4;
    expect(seconds).toBeGreaterThan(3.5);
    expect(seconds).toBeLessThan(4.5);
    expect(v.loop).toBe(false);
  });
});

// ── a strict fake Web Audio graph ────────────────────────────────────────────
// It throws wherever real browsers throw (exponential ramp to 0, non-finite
// values, double start, stop before start) and tracks connections so we can
// check that every finished sound is fully disconnected.

interface World {
  now: number;
  live: Set<FakeNode>;
  sources: Set<FakeSource>;
  created: number;
  violations: string[];
}
let world: World;

function check(ok: boolean, msg: string, E: ErrorConstructor = Error): void {
  if (!ok) {
    world.violations.push(msg);
    throw new E(msg);
  }
}

class FakeParam {
  constructor(
    public value: number,
    private readonly label: string,
  ) {}
  private ok(v: number, t: number): void {
    check(Number.isFinite(v), `${this.label}: non-finite value ${v}`, TypeError);
    check(Number.isFinite(t) && t >= 0, `${this.label}: bad time ${t}`, RangeError);
  }
  setValueAtTime(v: number, t: number): this {
    this.ok(v, t);
    return this;
  }
  linearRampToValueAtTime(v: number, t: number): this {
    this.ok(v, t);
    return this;
  }
  exponentialRampToValueAtTime(v: number, t: number): this {
    this.ok(v, t);
    check(v > 0, `${this.label}: exponential ramp to ${v}`, RangeError);
    return this;
  }
  setTargetAtTime(v: number, t: number, c: number): this {
    this.ok(v, t);
    check(c >= 0, `${this.label}: negative time constant`, RangeError);
    return this;
  }
  cancelScheduledValues(t: number): this {
    this.ok(0, t);
    return this;
  }
}

class FakeNode {
  readonly outputs = new Set<unknown>();
  constructor() {
    world.created++;
  }
  connect<T>(dest: T): T {
    check(dest != null, 'connect() to nothing');
    this.outputs.add(dest);
    world.live.add(this);
    return dest;
  }
  disconnect(): void {
    this.outputs.clear();
    world.live.delete(this);
  }
}

class FakeSource extends FakeNode {
  onended: (() => void) | null = null;
  started = false;
  stopAt = Infinity;
  start(when = 0): void {
    check(!this.started, 'start() called twice');
    check(Number.isFinite(when) && when >= 0, `bad start time ${when}`, RangeError);
    this.started = true;
    world.sources.add(this);
  }
  stop(when = 0): void {
    check(this.started, 'stop() before start()');
    check(Number.isFinite(when) && when >= 0, `bad stop time ${when}`, RangeError);
    this.stopAt = when;
  }
}

class FakeOscillator extends FakeSource {
  type = 'sine';
  readonly frequency = new FakeParam(440, 'osc.frequency');
  readonly detune = new FakeParam(0, 'osc.detune');
}

class FakeBuffer {
  readonly duration: number;
  private readonly data: Float32Array;
  constructor(
    readonly numberOfChannels: number,
    readonly length: number,
    readonly sampleRate: number,
  ) {
    this.data = new Float32Array(length);
    this.duration = length / sampleRate;
  }
  getChannelData(): Float32Array {
    return this.data;
  }
}

class FakeBufferSource extends FakeSource {
  buffer: FakeBuffer | null = null;
  loop = false;
  readonly playbackRate = new FakeParam(1, 'playbackRate');
  override start(when = 0, offset = 0): void {
    check(this.buffer !== null, 'buffer source started without a buffer');
    check(Number.isFinite(offset) && offset >= 0, `bad offset ${offset}`, RangeError);
    super.start(when);
    if (!this.loop && this.buffer) this.stopAt = Math.min(this.stopAt, when + this.buffer.duration);
  }
}

class FakeGain extends FakeNode {
  readonly gain = new FakeParam(1, 'gain');
}

class FakeBiquad extends FakeNode {
  type = 'lowpass';
  readonly frequency = new FakeParam(350, 'filter.frequency');
  readonly Q = new FakeParam(1, 'filter.Q');
  readonly gain = new FakeParam(0, 'filter.gain');
  readonly detune = new FakeParam(0, 'filter.detune');
}

class FakePanner extends FakeNode {
  readonly pan = new FakeParam(0, 'pan');
}

class FakeShaper extends FakeNode {
  curve: Float32Array | null = null;
  oversample = 'none';
}

class FakeCompressor extends FakeNode {
  readonly threshold = new FakeParam(-24, 'threshold');
  readonly knee = new FakeParam(30, 'knee');
  readonly ratio = new FakeParam(12, 'ratio');
  readonly attack = new FakeParam(0.003, 'attack');
  readonly release = new FakeParam(0.25, 'release');
}

class FakeAudioContext {
  state: 'suspended' | 'running' | 'closed' = 'suspended';
  readonly sampleRate = 22050;
  readonly destination = new FakeNode();
  get currentTime(): number {
    return world.now;
  }
  resume(): Promise<void> {
    this.state = 'running';
    return Promise.resolve();
  }
  close(): Promise<void> {
    this.state = 'closed';
    return Promise.resolve();
  }
  createGain(): FakeGain {
    return new FakeGain();
  }
  createOscillator(): FakeOscillator {
    return new FakeOscillator();
  }
  createBufferSource(): FakeBufferSource {
    return new FakeBufferSource();
  }
  createBuffer(channels: number, length: number, rate: number): FakeBuffer {
    return new FakeBuffer(channels, length, rate);
  }
  createBiquadFilter(): FakeBiquad {
    return new FakeBiquad();
  }
  createStereoPanner(): FakePanner {
    return new FakePanner();
  }
  createWaveShaper(): FakeShaper {
    return new FakeShaper();
  }
  createDynamicsCompressor(): FakeCompressor {
    return new FakeCompressor();
  }
}

/** Move the audio clock and the JS timers forward together, firing onended like a browser. */
function advance(seconds: number, tickMs = 25): void {
  const ticks = Math.round((seconds * 1000) / tickMs);
  for (let k = 0; k < ticks; k++) {
    world.now += tickMs / 1000;
    for (const s of [...world.sources]) {
      if (s.stopAt <= world.now) {
        world.sources.delete(s);
        s.onended?.();
      }
    }
    vi.advanceTimersByTime(tickMs);
  }
}

/** The four permanent bus nodes: sfx, music, compressor, master. */
const BUS_NODES = 4;

async function withFakeAudio(): Promise<{ mod: AudioModule; errors: unknown[] }> {
  world = { now: 0, live: new Set(), sources: new Set(), created: 0, violations: [] };
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  vi.stubGlobal('window', { AudioContext: FakeAudioContext });
  const mod = await freshModule();
  const errors: unknown[] = [];
  mod.setAudioErrorHandler((e) => errors.push(e));
  return { mod, errors };
}

describe('audio with a (fake) AudioContext', () => {
  it('stays silent until unlock, then reports ready', async () => {
    const { mod } = await withFakeAudio();
    mod.audio.play('batCrack');
    expect(world.created).toBe(0);
    expect(mod.audio.ready).toBe(false);
    mod.audio.unlock();
    expect(mod.audio.ready).toBe(true);
  });

  it('builds every sound effect without errors and cleans up after itself', async () => {
    const { mod, errors } = await withFakeAudio();
    const { audio, SFX_NAMES } = mod;
    audio.unlock();
    for (const name of SFX_NAMES) {
      for (const intensity of [0, 0.5, 1]) {
        const before = world.created;
        audio.play(name, { intensity, pan: intensity * 2 - 1 });
        expect(world.created, `${name} made no sound`).toBeGreaterThan(before + 2);
        advance(0.7); // past every sound's repeat guard
      }
    }
    advance(4);
    expect(world.violations).toEqual([]);
    expect(errors).toEqual([]);
    expect(world.sources.size).toBe(0);
    expect(world.live.size).toBe(BUS_NODES);
  });

  it('drops sound effects while muted and caps overlapping voices', async () => {
    const { mod } = await withFakeAudio();
    const { audio } = mod;
    audio.unlock();
    const base = world.created;
    audio.setMuted(true);
    audio.play('cheer');
    expect(world.created).toBe(base);
    audio.setMuted(false);
    for (let k = 0; k < 100; k++) {
      audio.play('bounce');
      world.now += 0.031; // just past the per-sound repeat guard, without letting anything finish
    }
    const panners = [...world.live].filter((n) => n instanceof FakePanner).length;
    expect(panners).toBeLessThanOrEqual(24);
  });

  it('plays every track through (and loops) without errors', async () => {
    const { mod, errors } = await withFakeAudio();
    const { audio } = mod;
    audio.playMusic('title'); // requested before unlock: starts on unlock
    expect(world.created).toBe(0);
    audio.unlock();
    advance(36); // a full 16-bar loop of the title plus a little
    const created = world.created;
    advance(1);
    expect(world.created).toBeGreaterThan(created); // still going: it looped
    audio.playMusic('game');
    advance(25);
    audio.playMusic('season');
    advance(45);
    audio.playMusic('victory');
    advance(8);
    audio.stopMusic();
    advance(5);
    expect(world.violations).toEqual([]);
    expect(errors).toEqual([]);
    expect(world.sources.size).toBe(0);
    expect(world.live.size).toBe(BUS_NODES);
  });

  it('skips ahead after a stall instead of bursting stale notes', async () => {
    const { mod } = await withFakeAudio();
    mod.audio.unlock();
    mod.audio.playMusic('title');
    advance(1);
    world.now += 30; // 30 s pass with no timer ticks (frozen background tab)
    const before = world.created;
    vi.advanceTimersByTime(25);
    // one tick may schedule ~0.1 s of music, never the 30 s that was missed
    expect(world.created - before).toBeLessThan(80);
  });

  it('runs the backyard ambience (birds, mower) and tears it down', async () => {
    const { mod, errors } = await withFakeAudio();
    const { audio } = mod;
    audio.setAmbience(true); // before unlock: remembered
    audio.unlock();
    const bed = world.created;
    advance(240, 50);
    expect(world.created - bed).toBeGreaterThan(100); // dozens of bird calls happened
    audio.setAmbience(false);
    advance(16, 50); // longest mower pass is 14 s
    expect(world.violations).toEqual([]);
    expect(errors).toEqual([]);
    expect(world.sources.size).toBe(0);
    expect(world.live.size).toBe(BUS_NODES);
  });
});
