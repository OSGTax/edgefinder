/**
 * Renders every sound, voice and tune sample by sample through the fake Web
 * Audio graph (see fake-audio.ts) and checks what comes out: no NaNs, no
 * clipping at the speaker, no click at the start, silence at the end, and a
 * sane loudness on a phone-sized speaker. `AUDIO_TABLE=1 npx vitest run
 * tests/audio-render.test.ts` prints the loudness table used for TRIM_DB.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeAudioContext, db, phoneSpeaker, resetWorld, stats, world, type FakeNode } from './fake-audio';

type AudioModule = typeof import('../src/audio/index');
type Ctx = typeof import('../src/audio/context');

/** Print the measurement tables (AUDIO_TABLE=1). */
const SHOW = !!(globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env.AUDIO_TABLE;

let mod: AudioModule;
let ctxm: Ctx;

beforeEach(async () => {
  resetWorld();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  vi.stubGlobal('window', { AudioContext: FakeAudioContext });
  vi.resetModules();
  mod = await import('../src/audio/index');
  ctxm = await import('../src/audio/context');
  mod.audio.unlock();
  await Promise.resolve();
  mod.audio.setSfxVolume(1); // worst case: everything turned all the way up
  mod.audio.setMusicVolume(1);
  mod.audio.setVoiceVolume(1);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

/** Render until every source has finished (max `cap` s); returns [L, R] at the tap and at the speaker. */
function renderAll(cap: number, tap?: FakeNode): { at: Float32Array[]; out: Float32Array[]; secs: number } {
  const bus = ctxm.getBus()!;
  const ctx = bus.ctx as unknown as FakeAudioContext;
  const at: Float32Array[][] = [];
  const out: Float32Array[][] = [];
  let secs = 0;
  // render the tap and the destination together: the destination pulls the tap
  while (secs < cap && (secs === 0 || world.sources.size > 0)) {
    const o = ctx.render(0.25);
    out.push(o);
    if (tap) at.push([Float32Array.from([tap.l]), Float32Array.from([tap.r])]);
    secs += 0.25;
    vi.advanceTimersByTime(250);
  }
  const join = (xs: Float32Array[][], c: number) => {
    const n = xs.reduce((s, x) => s + x[c].length, 0);
    const r = new Float32Array(n);
    let k = 0;
    for (const x of xs) { r.set(x[c], k); k += x[c].length; }
    return r;
  };
  return { at: tap ? [join(at, 0), join(at, 1)] : [], out: [join(out, 0), join(out, 1)], secs };
}

describe('every sound effect, rendered', () => {
  it('has no NaNs, no clipping, no click in, and dies away to silence', () => {
    const rows: string[] = [];
    const sr = (ctxm.getBus()!.ctx as unknown as FakeAudioContext).sampleRate;
    for (const name of mod.SFX_NAMES) {
      for (const intensity of [0.7, 1]) {
        const fctx = ctxm.getBus()!.ctx as unknown as FakeAudioContext;
        fctx.render(1); // past every repeat guard
        const master = ctxm.getBus()!.master;
        fctx.watch(master);
        mod.audio.play(name, { intensity });
        const { out, secs } = renderAll(5);
        const st = stats(...out);
        // the safety limiter may round off a transient, but nothing should lean on it (this fake
        // compressor lets more of a transient through than Chrome's does, so this is pessimistic)
        const drive = fctx.peaks.get(master as never) ?? 0;
        expect(db(drive), `${name} @${intensity}: drives the limiter`).toBeLessThan(2.5);
        expect(st.nan, `${name}: NaN`).toBe(false);
        expect(st.peak, `${name} @${intensity}: clips (${db(st.peak).toFixed(1)} dB)`).toBeLessThan(1);
        expect(st.first, `${name}: starts with a click`).toBeLessThan(0.01);
        expect(st.tail, `${name}: doesn't end in silence`).toBeLessThan(0.002);
        expect(st.loud, `${name}: inaudible`).toBeGreaterThan(0.003);
        const ph = stats(...out.map((c) => phoneSpeaker(c, sr)));
        // on a phone speaker everything must still be there: nothing relies on bass
        expect(ph.loud / st.loud, `${name}: mostly bass`).toBeGreaterThan(0.25);
        rows.push(`${name.padEnd(12)} i=${intensity}  drive ${db(drive).toFixed(1).padStart(5)}  peak ${db(st.peak).toFixed(1).padStart(6)}  loud ${db(st.loud).toFixed(1).padStart(6)}  phone ${db(ph.loud).toFixed(1).padStart(6)}  ${secs}s`);
      }
    }
    if (SHOW) console.log(rows.join('\n'));
  }, 120000);
});

describe('voices, rendered', () => {
  it('every speaker is clean, short and never shrill', async () => {
    const { KIDS } = await import('../src/data/kids');
    const sr = (ctxm.getBus()!.ctx as unknown as FakeAudioContext).sampleRate;
    const rows: string[] = [];
    const lines: Array<[string, string]> = [
      ...KIDS.map((k): [string, string] => [k.id, k.quips[0]]),
      ['Chet', 'Bases loaded, and Kaboom steps in. I can\'t look. I\'m looking.'],
      ['Dottie', 'Back in my big-league days, I batted ninth.'],
      ['mrsMendoza', 'KAI! Aim LEFT, sweetie!'],
    ];
    for (const [who, text] of lines) {
      const d = mod.audio.speak(who, text);
      expect(d, who).toBeGreaterThan(0);
      const { out } = renderAll(6);
      const st = stats(...out);
      expect(st.nan, `${who}: NaN`).toBe(false);
      expect(st.peak, `${who}: clips`).toBeLessThan(1);
      expect(st.first, `${who}: click`).toBeLessThan(0.01);
      expect(st.tail, `${who}: no silence at the end`).toBeLessThan(0.002);
      expect(st.loud, `${who}: inaudible`).toBeGreaterThan(0.01);
      const ph = stats(...out.map((c) => phoneSpeaker(c, sr)));
      rows.push(`${who.padEnd(11)} ${d.toFixed(2)}s  peak ${db(st.peak).toFixed(1)}  loud ${db(st.loud).toFixed(1)}  phone ${db(ph.loud).toFixed(1)}`);
    }
    if (SHOW) console.log(rows.join('\n'));
  }, 120000);
});

describe('music, rendered', () => {
  it('each track is clean and sits under the sound effects', () => {
    const rows: string[] = [];
    for (const track of mod.MUSIC_TRACKS) {
      mod.audio.playMusic(track);
      const { out } = renderAll(track === 'title' || track === 'season' || track === 'game' ? 8 : 9);
      mod.audio.stopMusic();
      renderAll(4);
      const st = stats(...out);
      expect(st.nan, `${track}: NaN`).toBe(false);
      expect(st.peak, `${track}: clips`).toBeLessThan(1);
      expect(st.loud, `${track}: silent`).toBeGreaterThan(0.01);
      rows.push(`${track.padEnd(8)} peak ${db(st.peak).toFixed(1)}  loud ${db(st.loud).toFixed(1)}  rms ${db(st.rms).toFixed(1)}`);
    }
    if (SHOW) console.log(rows.join('\n'));
  }, 180000);
});
