import { afterEach, describe, expect, it, vi } from 'vitest';
import { FakeAudioContext, FakePanner, resetWorld, world, endSources } from './fake-audio';

type AudioModule = typeof import('../src/audio/index');

const EXPECTED_SFX = [
  'batCrack', 'batTink', 'whiff', 'mittPop', 'catch', 'bounce', 'bounceDirt', 'bouncePatio', 'fence',
  'picket', 'hedge', 'houseWall', 'splash', 'leaves', 'cheer', 'bigCheer', 'aww', 'ooh', 'giggle',
  'strike', 'out', 'safe', 'homeRun', 'special', 'uiTap', 'uiBack', 'uiSelect', 'whistle', 'dogBark',
  'screenDoor', 'throw',
];
const EXPECTED_TRACKS = ['title', 'game', 'inning', 'victory', 'defeat', 'season'];

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
  audio.speak('kai', 'SUNDAY! SUNDAY! SUNDAY!');
  audio.speak('Chet', 'Here comes Kaboom to the plate.');
  audio.speak('nobody-we-know', '');
  audio.setAmbience(true);
  audio.setAmbience(false);
  audio.setSfxVolume(0.5);
  audio.setSfxVolume(-3);
  audio.setMusicVolume(2);
  audio.setMusicVolume(Number.NaN);
  audio.setVoiceVolume(0.4);
  audio.setMuted(true);
  audio.setMuted(false);
  // garbage from untyped callers must not throw either
  (audio.play as (n: unknown) => void)('notASound');
  (audio.playMusic as (n: unknown) => void)(undefined);
  (audio.speak as (w: unknown, t: unknown) => void)(undefined, null);
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  FakeAudioContext.resumeFails = false;
  FakeAudioContext.instances.length = 0;
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
  it('parse, are whole bars, and the one-shots are the right length', async () => {
    vi.resetModules();
    const { SONGS } = await import('../src/audio/tracks');
    for (const song of Object.values(SONGS)) {
      for (const part of song.parts) {
        expect(part.len % 16).toBe(0);
        expect(song.length % part.len).toBe(0);
      }
    }
    expect(SONGS.title.length).toBe(256); // 16 bars, A + B
    const secs = (name: keyof typeof SONGS) => SONGS[name].length * SONGS[name].stepDur + 0.4;
    expect(secs('victory')).toBeGreaterThan(3.5);
    expect(secs('victory')).toBeLessThan(4.5);
    expect(secs('defeat')).toBeLessThan(6);
    expect(secs('inning')).toBeLessThan(7.5);
    expect(SONGS.victory.loop).toBe(false);
    expect(SONGS.inning.then).toBe(true); // hands back to the game music
    expect(SONGS.victory.then).toBe(false);
  });
});

// ── with a fake (strict) AudioContext ────────────────────────────────────────

/** Move the audio clock and the JS timers forward together, firing onended like a browser. */
function advance(seconds: number, tickMs = 25): void {
  const ticks = Math.round((seconds * 1000) / tickMs);
  for (let k = 0; k < ticks; k++) {
    world.now += tickMs / 1000;
    endSources(world.now);
    vi.advanceTimersByTime(tickMs);
  }
}

/** The permanent bus nodes: sfx, voice, music, duck, compressor, master, safety clip. */
const BUS_NODES = 7;

interface FakeDoc {
  visibilityState: 'visible' | 'hidden';
  listeners: Map<string, Array<() => void>>;
  addEventListener(ev: string, fn: () => void): void;
  fire(ev: string): void;
}

function fakeDocument(): FakeDoc {
  const doc: FakeDoc = {
    visibilityState: 'visible',
    listeners: new Map(),
    addEventListener(ev, fn) {
      const l = doc.listeners.get(ev) ?? [];
      l.push(fn);
      doc.listeners.set(ev, l);
    },
    fire(ev) {
      for (const fn of doc.listeners.get(ev) ?? []) fn();
    },
  };
  return doc;
}

async function withFakeAudio(doc?: FakeDoc): Promise<{ mod: AudioModule; errors: unknown[] }> {
  resetWorld();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  vi.stubGlobal('window', { AudioContext: FakeAudioContext, addEventListener: () => undefined });
  if (doc) vi.stubGlobal('document', doc);
  const mod = await freshModule();
  const errors: unknown[] = [];
  mod.setAudioErrorHandler((e) => errors.push(e));
  return { mod, errors };
}

/** Let resolved promises (resume()) run. */
const flush = () => Promise.resolve().then(() => Promise.resolve());

describe('audio with a (fake) AudioContext', () => {
  it('stays silent until unlock, then reports ready', async () => {
    const { mod } = await withFakeAudio();
    mod.audio.play('batCrack');
    expect(world.created).toBe(0);
    expect(mod.audio.ready).toBe(false);
    mod.audio.unlock();
    await flush();
    expect(mod.audio.ready).toBe(true);
  });

  it('builds every sound effect without errors and cleans up after itself', async () => {
    const { mod, errors } = await withFakeAudio();
    const { audio, SFX_NAMES } = mod;
    audio.unlock();
    await flush();
    for (const name of SFX_NAMES) {
      for (const intensity of [0, 0.5, 1]) {
        const before = world.created;
        audio.play(name, { intensity, pan: intensity * 2 - 1 });
        expect(world.created, `${name} made no sound`).toBeGreaterThan(before + 2);
        advance(2.1); // past every sound's repeat guard
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
    await flush();
    const base = world.created;
    audio.setMuted(true);
    audio.play('cheer');
    audio.speak('kai', 'KA-BOOM!');
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
    await flush();
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
    audio.playMusic('defeat');
    advance(8);
    audio.stopMusic();
    advance(5);
    expect(world.violations).toEqual([]);
    expect(errors).toEqual([]);
    expect(world.sources.size).toBe(0);
    expect(world.live.size).toBe(BUS_NODES);
  });

  it('plays the between-innings jingle once, then goes back to the game music', async () => {
    const { mod } = await withFakeAudio();
    const music = await import('../src/audio/music');
    mod.audio.unlock();
    await flush();
    mod.audio.playMusic('game');
    advance(2);
    mod.audio.playMusic('inning');
    expect(music.nowPlaying()).toBe('inning');
    advance(9);
    expect(music.nowPlaying()).toBe('game');
    mod.audio.playMusic('victory');
    advance(8);
    expect(music.nowPlaying()).toBe(null); // the fanfare doesn't hand back
  });

  it('skips ahead after a stall instead of bursting stale notes', async () => {
    const { mod } = await withFakeAudio();
    mod.audio.unlock();
    await flush();
    mod.audio.playMusic('title');
    advance(1);
    world.now += 30; // 30 s pass with no timer ticks (frozen background tab)
    const before = world.created;
    vi.advanceTimersByTime(25);
    // one tick may schedule ~0.1 s of music, never the 30 s that was missed
    expect(world.created - before).toBeLessThan(150);
  });

  it('runs the backyard ambience and tears it down', async () => {
    const { mod, errors } = await withFakeAudio();
    const { audio } = mod;
    audio.setAmbience(true); // before unlock: remembered
    audio.unlock();
    await flush();
    const bed = world.created;
    advance(400, 50); // long enough for the truck, the mower, the dog...
    expect(world.created - bed).toBeGreaterThan(100); // dozens of birds, cicadas, flips happened
    audio.setAmbience(false);
    advance(30, 50); // the longest one-off (the ice-cream truck) is ~22 s
    expect(world.violations).toEqual([]);
    expect(errors).toEqual([]);
    expect(world.sources.size).toBe(0);
    expect(world.live.size).toBe(BUS_NODES);
  });

  it('says lines for every kid, both announcers and the grown-ups', async () => {
    const { mod, errors } = await withFakeAudio();
    const { KIDS } = await import('../src/data/kids');
    mod.audio.unlock();
    await flush();
    const who = [...KIDS.map((k) => k.id), 'Chet', 'Dottie', 'mrsMendoza', 'mrMendoza', 'a-future-kid'];
    for (const w of who) {
      const line = KIDS.find((k) => k.id === w)?.quips[0] ?? 'And that is the ballgame, folks!';
      const d = mod.audio.speak(w, line);
      expect(d, w).toBeGreaterThan(0.2);
      expect(d, w).toBeLessThan(4);
      advance(d + 0.5);
    }
    mod.audio.setVoiceVolume(0);
    expect(mod.audio.speak('kai', 'BE THERE!')).toBe(0); // voices off: nothing built
    advance(1);
    expect(world.violations).toEqual([]);
    expect(errors).toEqual([]);
    expect(world.live.size).toBe(BUS_NODES);
  });
});

describe('phones: unlock, interruptions, background', () => {
  it('unlocks on the first touch anywhere, with no game code involved', async () => {
    const doc = fakeDocument();
    const { mod } = await withFakeAudio(doc);
    mod.audio.playMusic('title');
    expect(mod.audio.ready).toBe(false);
    doc.fire('touchend');
    await flush();
    expect(mod.audio.ready).toBe(true);
  });

  it('fades out and suspends while hidden, and comes back when shown', async () => {
    const doc = fakeDocument();
    const { mod } = await withFakeAudio(doc);
    mod.audio.unlock();
    await flush();
    const ctx = FakeAudioContext.instances[0];
    doc.visibilityState = 'hidden';
    doc.fire('visibilitychange');
    advance(0.2);
    await flush();
    expect(ctx.state).toBe('suspended');
    // sounds asked for while hidden are simply dropped
    const before = world.created;
    mod.audio.play('batCrack');
    expect(world.created).toBe(before);
    doc.visibilityState = 'visible';
    doc.fire('visibilitychange');
    await flush();
    expect(ctx.state).toBe('running');
  });

  it('recovers after an interruption (a phone call, Siri)', async () => {
    const doc = fakeDocument();
    const { mod } = await withFakeAudio(doc);
    mod.audio.unlock();
    await flush();
    const ctx = FakeAudioContext.instances[0];
    ctx.setState('interrupted'); // iOS: the call comes in
    await flush();
    expect(ctx.state).toBe('running'); // resumed as soon as it's allowed
  });

  it('rebuilds a context iOS leaves stuck, and the music carries on', async () => {
    const doc = fakeDocument();
    const { mod, errors } = await withFakeAudio(doc);
    const music = await import('../src/audio/music');
    mod.audio.unlock();
    await flush();
    mod.audio.playMusic('game');
    mod.audio.setAmbience(true);
    advance(1);
    const first = FakeAudioContext.instances[0];
    FakeAudioContext.resumeFails = true; // resume() resolves but nothing runs
    first.setState('interrupted');
    await flush();
    doc.fire('touchend');
    await flush();
    expect(first.state).toBe('interrupted');
    FakeAudioContext.resumeFails = false;
    doc.fire('touchend'); // the next touch gives up on it and starts fresh
    await flush();
    expect(FakeAudioContext.instances.length).toBe(2);
    expect(first.state).toBe('closed');
    expect(FakeAudioContext.instances[1].state).toBe('running');
    expect(mod.audio.ready).toBe(true);
    expect(music.nowPlaying()).toBe('game');
    advance(2);
    expect(errors).toEqual([]);
  });
});
