/**
 * AudioContext lifecycle and the master bus.
 *
 *   sfx gain ─────────────┐
 *   voice gain ───────────┼─> compressor ─> master (mute) ─> safety clip ─> destination
 *   music gain ─> duck ───┘
 *
 * Nothing is created until unlock() runs inside a user gesture. Without Web
 * Audio (Node, locked-down browsers) getBus() stays null and every caller
 * quietly does nothing.
 *
 * Phones: unlock() is wired to the first touch / click / key by itself (iOS
 * only opens audio inside touchend or click), the output fades out and the
 * context suspends while the page is hidden (no battery drain, no pop), and
 * after an interruption (a call, Siri, an alarm) the context is resumed on
 * the next chance, or rebuilt from scratch on the next touch if iOS leaves it
 * stuck. Music and ambience re-attach themselves to a rebuilt bus.
 */

export interface Bus {
  readonly ctx: AudioContext;
  readonly sfx: GainNode;
  readonly music: GainNode;
  readonly voice: GainNode;
  /** Music passes through here so voices can duck it. */
  readonly duck: GainNode;
  readonly master: GainNode;
  /** Two seconds of white noise shared by every noise-based sound. */
  readonly noise: AudioBuffer;
}

type AudioContextCtor = new (options?: AudioContextOptions) => AudioContext;

const settings = { sfx: 0.8, music: 0.45, voice: 0.8, muted: false };
/**
 * Output level after the compressor, offsetting its automatic make-up gain (~+5.6 dB here)
 * and leaving the loudest transients (a crushed bat crack at full volume) just under the
 * safety limiter.
 */
const OUTPUT = 0.52;
let bus: Bus | null = null;
const busListeners: Array<(b: Bus) => void> = [];
const lostListeners: Array<(b: Bus) => void> = [];
let errorHandler: ((e: unknown) => void) | null = null;
/** resume() calls that came back without the context running (iOS after a call). */
let stuck = 0;
/** We suspended the context ourselves (page hidden): don't fight it. */
let parked = false;
let parkTimer: ReturnType<typeof setTimeout> | undefined;

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

/** Run fn whenever a bus is created (used to start music/ambience requested before unlock). */
export function onBus(fn: (b: Bus) => void): void {
  busListeners.push(fn);
}

/** Run fn when a bus is torn down to be rebuilt (drop anything that holds its nodes). */
export function onBusLost(fn: (b: Bus) => void): void {
  lostListeners.push(fn);
}

function findCtor(): AudioContextCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    AudioContext?: AudioContextCtor;
    webkitAudioContext?: AudioContextCtor;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

let clipCurve: Float32Array<ArrayBuffer> | null = null;
/** Transparent below 0.8, then a smooth shoulder to 1: a last line of defence against clipping. */
function safetyCurve(): Float32Array<ArrayBuffer> {
  if (!clipCurve) {
    const n = 2048;
    clipCurve = new Float32Array(n);
    const k = 0.8;
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * 2 - 1;
      const a = Math.abs(x);
      const y = a <= k ? a : k + (1 - k) * Math.tanh((a - k) / (1 - k));
      clipCurve[i] = Math.sign(x) * y;
    }
  }
  return clipCurve;
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
    comp.attack.value = 0.003;
    comp.release.value = 0.2;

    const master = ctx.createGain();
    master.gain.value = settings.muted ? 0 : OUTPUT;
    const clip = ctx.createWaveShaper();
    clip.curve = safetyCurve();
    const sfx = ctx.createGain();
    sfx.gain.value = settings.sfx;
    const voice = ctx.createGain();
    voice.gain.value = settings.voice;
    const music = ctx.createGain();
    music.gain.value = settings.music;
    const duck = ctx.createGain();
    sfx.connect(comp);
    voice.connect(comp);
    music.connect(duck);
    duck.connect(comp);
    comp.connect(master);
    master.connect(clip);
    clip.connect(ctx.destination);

    const len = Math.floor(ctx.sampleRate * 2);
    const noise = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;

    return { ctx, sfx, music, voice, duck, master, noise };
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

/** Bring the output level back smoothly (after a resume), so nothing pops. */
function fadeIn(b: Bus): void {
  const g = b.master.gain;
  const t = b.ctx.currentTime;
  g.cancelScheduledValues(t);
  g.setValueAtTime(0, t);
  g.linearRampToValueAtTime(settings.muted ? 0 : OUTPUT, t + 0.12);
}

function tryResume(b: Bus): void {
  if (b.ctx.state === 'closed') return;
  b.ctx.resume().then(
    () => {
      if (b.ctx.state === 'running') {
        stuck = 0;
        fadeIn(b);
      } else stuck++;
    },
    () => stuck++,
  );
}

function park(b: Bus): void {
  if (b.ctx.state !== 'running') return;
  parked = true;
  const t = b.ctx.currentTime;
  b.master.gain.cancelScheduledValues(t);
  b.master.gain.setTargetAtTime(0, t, 0.012);
  clearTimeout(parkTimer);
  parkTimer = setTimeout(() => {
    if (bus === b && pageHidden()) void b.ctx.suspend().catch(() => undefined);
  }, 80);
}

function unpark(b: Bus): void {
  parked = false;
  clearTimeout(parkTimer);
  if (b.ctx.state === 'running') fadeIn(b);
  else tryResume(b);
}

function watchLifecycle(b: Bus): void {
  b.ctx.onstatechange = () => {
    if (bus !== b) return;
    const s = b.ctx.state as string;
    // an interruption (call, Siri, another app grabbing audio) while we're on screen
    if ((s === 'interrupted' || s === 'suspended') && !parked && !pageHidden()) tryResume(b);
  };
}

let docWired = false;
function wireDocument(): void {
  if (docWired || typeof document === 'undefined' || typeof document.addEventListener !== 'function') return;
  docWired = true;
  const gesture = () => safely(unlock);
  // touchend/click are what iOS counts as a user gesture; the rest cover everything else
  for (const ev of ['touchend', 'click', 'pointerdown', 'keydown', 'touchstart']) {
    document.addEventListener(ev, gesture, { capture: true, passive: true });
  }
  document.addEventListener('visibilitychange', () =>
    safely(() => {
      if (!bus) return;
      if (pageHidden()) park(bus);
      else unpark(bus);
    }),
  );
  if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
    // back/forward cache: the page comes back without a visibilitychange on some browsers
    window.addEventListener('pageshow', () => safely(() => bus && !pageHidden() && unpark(bus)));
    window.addEventListener('focus', () => safely(() => bus && !pageHidden() && unpark(bus)));
  }
}

/** Throw the bus away and build a new one (iOS sometimes never lets a context run again). */
function rebuild(): void {
  const old = bus;
  if (!old) return;
  bus = null;
  stuck = 0;
  for (const fn of lostListeners) safely(() => fn(old));
  old.ctx.onstatechange = null;
  void old.ctx.close?.().catch(() => undefined);
}

export function unlock(): void {
  wireDocument();
  if (bus && (bus.ctx.state === 'closed' || stuck >= 2)) rebuild();
  if (!bus) {
    const Ctor = findCtor();
    if (!Ctor) return;
    const created = createBus(Ctor);
    bus = created;
    watchLifecycle(created);
    for (const fn of busListeners) safely(() => fn(created));
  }
  const b = bus;
  if (b.ctx.state !== 'running' && b.ctx.state !== 'closed' && !pageHidden()) {
    parked = false;
    primeOutput(b.ctx);
    tryResume(b);
  }
}

// Hook the gestures up as soon as the module loads in a browser, so the very
// first touch anywhere opens the audio even before any menu code runs.
safely(wireDocument);

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

export function setVoiceVolume(v: number): void {
  settings.voice = level(v, settings.voice);
  if (bus) glideTo(bus.voice.gain, settings.voice);
}

export const voiceVolume = (): number => settings.voice;

export function setMuted(m: boolean): void {
  settings.muted = !!m;
  if (bus) glideTo(bus.master.gain, settings.muted ? 0 : OUTPUT);
}

/** Dip the music under a voice for `dur` seconds (by `depth`, 0..1), then bring it back. */
export function duckMusic(dur: number, depth = 0.4): void {
  if (!bus) return;
  const g = bus.duck.gain;
  const t = bus.ctx.currentTime;
  g.cancelScheduledValues(t);
  g.setTargetAtTime(1 - depth, t, 0.06);
  g.setTargetAtTime(1, t + Math.max(0.1, dur), 0.25);
}
