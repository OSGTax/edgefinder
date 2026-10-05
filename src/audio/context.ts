/**
 * AudioContext lifecycle and the master bus.
 *
 *   sfx gain ───┐
 *               ├─> compressor ─> master (mute) ─> destination
 *   music gain ─┘
 *
 * Nothing is created until unlock() runs inside a user gesture. Without Web
 * Audio (Node, locked-down browsers) getBus() stays null and every caller
 * quietly does nothing.
 */

export interface Bus {
  readonly ctx: AudioContext;
  readonly sfx: GainNode;
  readonly music: GainNode;
  readonly master: GainNode;
  /** Two seconds of white noise shared by every noise-based sound. */
  readonly noise: AudioBuffer;
}

type AudioContextCtor = new (options?: AudioContextOptions) => AudioContext;

const settings = { sfx: 0.8, music: 0.45, muted: false };
/** Output level after the compressor, offsetting its automatic make-up gain (~+5.6 dB here). */
const OUTPUT = 0.6;
let bus: Bus | null = null;
const busListeners: Array<(b: Bus) => void> = [];
let errorHandler: ((e: unknown) => void) | null = null;

export function setErrorHandler(fn: ((e: unknown) => void) | null): void {
  errorHandler = fn;
}

export function reportError(e: unknown): void {
  if (errorHandler) errorHandler(e);
  else if (import.meta.env?.DEV) console.warn('[audio]', e);
}

/** Audio must never break the game: run fn and swallow (but report) anything it throws. */
export function safely(fn: () => void): void {
  try {
    fn();
  } catch (e) {
    reportError(e);
  }
}

export const getBus = (): Bus | null => bus;
export const isReady = (): boolean => bus !== null && bus.ctx.state === 'running';
export const isMuted = (): boolean => settings.muted;
export const pageHidden = (): boolean =>
  typeof document !== 'undefined' && document.visibilityState === 'hidden';

/** Run fn once the bus exists (used to start music/ambience requested before unlock). */
export function onBus(fn: (b: Bus) => void): void {
  busListeners.push(fn);
}

function findCtor(): AudioContextCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    AudioContext?: AudioContextCtor;
    webkitAudioContext?: AudioContextCtor;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

function createBus(Ctor: AudioContextCtor): Bus {
  let ctx: AudioContext;
  try {
    ctx = new Ctor({ latencyHint: 'interactive' });
  } catch {
    ctx = new Ctor(); // old webkitAudioContext takes no options
  }
  try {
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.knee.value = 10;
    comp.ratio.value = 3;
    comp.attack.value = 0.004;
    comp.release.value = 0.2;

    const master = ctx.createGain();
    master.gain.value = settings.muted ? 0 : OUTPUT;
    const sfx = ctx.createGain();
    sfx.gain.value = settings.sfx;
    const music = ctx.createGain();
    music.gain.value = settings.music;
    sfx.connect(comp);
    music.connect(comp);
    comp.connect(master);
    master.connect(ctx.destination);

    const len = Math.floor(ctx.sampleRate * 2);
    const noise = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;

    return { ctx, sfx, music, master, noise };
  } catch (e) {
    void ctx.close?.().catch(() => undefined); // don't leak contexts (browsers cap them)
    throw e;
  }
}

/** iOS only opens the output once something actually plays inside the gesture. */
function primeOutput(ctx: AudioContext): void {
  const src = ctx.createBufferSource();
  src.buffer = ctx.createBuffer(1, 1, ctx.sampleRate);
  src.connect(ctx.destination);
  src.onended = () => src.disconnect();
  src.start(0);
}

function watchVisibility(ctx: AudioContext): void {
  if (typeof document === 'undefined' || typeof document.addEventListener !== 'function') return;
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && ctx.state === 'suspended') {
      void ctx.resume().catch(() => undefined);
    }
  });
}

export function unlock(): void {
  if (!bus) {
    const Ctor = findCtor();
    if (!Ctor) return;
    const created = createBus(Ctor);
    bus = created;
    watchVisibility(created.ctx);
    for (const fn of busListeners) safely(() => fn(created));
  }
  const { ctx } = bus;
  if (ctx.state !== 'running' && ctx.state !== 'closed') {
    primeOutput(ctx);
    void ctx.resume().catch(() => undefined);
  }
}

const level = (v: unknown, fallback: number): number =>
  typeof v === 'number' && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : fallback;

function glideTo(p: AudioParam, v: number): void {
  if (bus) p.setTargetAtTime(v, bus.ctx.currentTime, 0.02);
}

export function setSfxVolume(v: number): void {
  settings.sfx = level(v, settings.sfx);
  if (bus) glideTo(bus.sfx.gain, settings.sfx);
}

export function setMusicVolume(v: number): void {
  settings.music = level(v, settings.music);
  if (bus) glideTo(bus.music.gain, settings.music);
}

export function setMuted(m: boolean): void {
  settings.muted = !!m;
  if (bus) glideTo(bus.master.gain, settings.muted ? 0 : OUTPUT);
}
