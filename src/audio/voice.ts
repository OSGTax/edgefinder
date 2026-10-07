/**
 * Gibberish voices: a tiny formant synthesizer.
 *
 * A line of text is "said" in a made-up language that follows the real line:
 * one sung syllable per syllable of text, the vowel colours of its vowels,
 * a little stop or hiss for its consonants, pauses at commas, a rise for a
 * question, a lift for an exclamation and a shout for a WORD IN CAPS. The
 * same speaker saying the same line always sounds the same (the melody is
 * seeded from the text), so it reads like a language rather than noise.
 *
 * Signal path (one set per utterance, whatever its length):
 *
 *   glottal pulse ─┐
 *   breath noise ──┴> amp envelope ─> F1/F2/F3 band-passes + a warm body ─┐
 *   noise ─> consonant filter ─> consonant envelope ──────────────────────┴> tone ─> pan ─> voice bus
 *
 * Announcers go through a toy-microphone "broadcast" filter on the way out.
 */
import { duckMusic, getBus, isMuted, isReady, voiceVolume } from './context';
import { FLOOR, Patch, clamp, harmonics, softClip, wire } from './dsp';
import { recipe, type VoiceRecipe } from './voices';

/** Formants (Hz) for an adult voice; kids scale them up by recipe.formant. */
const VOWELS: Record<string, readonly [number, number, number]> = {
  a: [760, 1250, 2600],
  e: [520, 1850, 2550],
  i: [300, 2250, 3000],
  o: [560, 900, 2500],
  u: [330, 860, 2300],
  y: [440, 1500, 2450], // the neutral "uh" of unstressed syllables
};

type Onset = 'none' | 'stop' | 'hiss' | 'hum' | 'glide' | 'breath';

interface Syl {
  vowel: string;
  onset: Onset;
  /** consonant colour for the noise burst (Hz) */
  burst: number;
  stress: boolean;
  shout: boolean;
  /** pause after this syllable (s) */
  gap: number;
  /** last syllable of a clause: '?' '!' '.' or '' */
  end: string;
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function prng(seed: number): () => number {
  let s = seed || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

function onsetOf(cons: string): [Onset, number] {
  const c = cons.toLowerCase();
  if (!c) return ['none', 0];
  const first = c[0];
  if ('ptkbdgcq'.includes(first)) return ['stop', 'pb'.includes(first) ? 900 : 'dt'.includes(first) ? 3000 : 2000];
  if ('sz'.includes(first) || c.startsWith('sh') || c.startsWith('ch') || first === 'x') return ['hiss', c.startsWith('sh') || c.startsWith('ch') ? 2800 : 5200];
  if ('fv'.includes(first)) return ['hiss', 3800];
  if (first === 'h') return ['breath', 1500];
  if ('mn'.includes(first)) return ['hum', 0];
  return ['glide', 0]; // l r w y j
}

function vowelOf(group: string): string {
  const g = group.toLowerCase();
  if (g.startsWith('oo') || g === 'ou' || g === 'ew') return 'u';
  if (g.startsWith('ee') || g === 'ie' || g === 'ey') return 'i';
  if (g.startsWith('ai') || g === 'ay') return 'e';
  const c = g[0];
  return c === 'y' ? 'i' : VOWELS[c] ? c : 'y';
}

/** Split a line into syllables with just enough information to sing it. */
export function syllabify(text: string, max: number): Syl[] {
  const out: Syl[] = [];
  const words = text.replace(/["“”()]/g, '').split(/\s+/).filter(Boolean);
  for (let w = 0; w < words.length && out.length < max; w++) {
    const raw = words[w];
    const letters = raw.replace(/[^A-Za-z'-]/g, '');
    if (!letters && !/\d/.test(raw)) continue;
    const shout = letters.length > 1 && letters === letters.toUpperCase() && /[A-Z]/.test(letters);
    const punct = /[?]$/.test(raw) ? '?' : /[!]$/.test(raw) ? '!' : /[.]$/.test(raw) ? '.' : '';
    const comma = /[,;:—-]$/.test(raw);
    // numbers: say one syllable per digit, near enough
    const body = /\d/.test(raw) && !letters ? 'na'.repeat(Math.min(3, raw.replace(/\D/g, '').length)) : letters;
    const re = /([^aeiouy]*)([aeiouy]+)/gi;
    const syls: Syl[] = [];
    let m: RegExpExecArray | null;
    while ((m = re.exec(body)) && syls.length < 5) {
      const [onset, burst] = onsetOf(m[1]);
      syls.push({ vowel: vowelOf(m[2]), onset, burst, stress: false, shout, gap: 0, end: '' });
    }
    // a silent final 'e' ("make", "base") isn't a syllable
    if (syls.length > 1 && /[^aeiouy]e'?s?$/i.test(body)) syls.pop();
    if (!syls.length) syls.push({ vowel: 'y', onset: 'none', burst: 0, stress: false, shout, gap: 0, end: '' });
    // stress: the first syllable of a word with more than one, or any short content word
    if (syls.length > 1 || letters.length > 3) syls[0].stress = true;
    const last = syls[syls.length - 1];
    last.gap = punct ? 0.2 : comma ? 0.12 : 0.03;
    last.end = punct;
    out.push(...syls);
  }
  if (out.length > max) out.length = max;
  if (out.length) {
    const last = out[out.length - 1];
    if (!last.end) last.end = '.';
  }
  return out;
}

export interface SpeakOpts {
  /** -1..1 */
  pan?: number;
  /** extra loudness, 0..1.5 (default 1) */
  level?: number;
  /** most syllables to say (long captions trail off) */
  max?: number;
  /** seconds from now */
  delay?: number;
  /** dip the music while talking */
  duck?: boolean;
}

/** Utterances in flight (keeps a pile-up of voices from forming). */
let speaking = 0;
const MAX_SPEAKING = 3;

/** Seconds the line will take, or 0 if nothing was said. */
export function speak(who: string, text: string, opts: SpeakOpts = {}): number {
  const bus = getBus();
  if (!bus || !isReady() || isMuted() || voiceVolume() <= 0 || speaking >= MAX_SPEAKING) return 0;
  const r = recipe(who);
  const syls = syllabify(text, opts.max ?? r.max);
  if (!syls.length) return 0;
  speaking++;
  const p = new Patch(bus, () => speaking--);
  let dur = 0;
  try {
    dur = build(p, r, syls, hash(`${who}|${text}`), opts);
  } finally {
    p.seal();
  }
  if (opts.duck && dur > 0) duckMusic(dur, 0.35);
  return dur;
}

function build(p: Patch, r: VoiceRecipe, syls: Syl[], seed: number, opts: SpeakOpts): number {
  const rnd = prng(seed);
  const jitter = (amt: number) => (rnd() * 2 - 1) * amt;
  const t0 = p.ctx.currentTime + 0.02 + (opts.delay ?? 0);
  const anyShout = syls.some((s) => s.shout);
  const exclaim = syls.some((s) => s.end === '!');

  // plan the timing first so sources know when to stop
  const plan: Array<{ t: number; d: number; s: Syl; k: number }> = [];
  let t = t0;
  syls.forEach((s, k) => {
    const final = k === syls.length - 1 || s.end !== '';
    let d = (1 / r.rate) * (s.stress ? 1.2 : 0.85) * (0.9 + 0.2 * rnd());
    if (final) d *= r.stretch;
    if (s.onset === 'stop') t += 0.025; // closure before a stop
    plan.push({ t, d, s, k });
    t += d + s.gap;
  });
  const end = t + 0.12;

  const level = clamp(opts.level ?? 1, 0, 1.5) * r.level;
  const out = p.panner(clamp(opts.pan ?? 0, -1, 1));
  const tone = p.gain(1);
  let tail: AudioNode = tone;
  if (r.radio) {
    // a toy microphone into a public-access transmitter: thin, a little crunchy
    const drive = p.gain(1.7);
    const crunch = p.shaper(softClip());
    const mid = p.filter('peaking', 1800, 1.2);
    mid.gain.value = 5;
    wire(tone, p.filter('highpass', 360, 0.5), mid, p.filter('lowpass', 3400, 0.5), drive, crunch);
    tail = p.gain(0.55);
    crunch.connect(tail);
  } else {
    // keep the very top soft so a run of voices never gets shrill
    const lp = p.filter('lowpass', 4200, 0);
    tone.connect(lp);
    tail = lp;
  }
  wire(tail, out, p.gain(1), p.bus.voice);

  // ── sources ──
  const [re, im] = harmonics(28, r.tilt);
  const src = p.custom(re, im, r.f0, t0, end);
  const amp = p.gain(0);
  amp.gain.setValueAtTime(FLOOR, t0);
  wire(src, amp);
  if (r.breath > 0) wire(p.noise(t0, end), p.filter('highpass', 900, 0), p.gain(r.breath * 0.6), amp);
  if (r.gruff > 0) p.lfo(src.frequency, 27 + 6 * rnd(), r.f0 * 0.035 * r.gruff, t0, end, 'triangle'); // rasp
  // a slow natural wobble in every voice
  p.lfo(src.frequency, 4.5 + rnd(), r.f0 * 0.006, t0, end);

  // ── formant bank ──
  const fs = r.formant;
  const F = [0, 1, 2].map((n) => p.filter('bandpass', VOWELS.y[n] * fs, [6, 9, 12][n] * r.focus));
  const FG = [1, 0.55, 0.22 * r.bright];
  F.forEach((f, n) => wire(amp, f, p.gain(FG[n] * 2.2), tone));
  wire(amp, p.filter('lowpass', 700 * fs, 0), p.gain(0.25), tone); // chest/body

  // ── consonants ──
  const cg = p.gain(0);
  cg.gain.setValueAtTime(FLOOR, t0);
  const cf = p.filter('bandpass', 3000, 1.6);
  wire(p.noise(t0, end), cf, cg, tone);

  // ── the performance ──
  const n = plan.length;
  for (const { t: ts, d, s, k } of plan) {
    const pos = n > 1 ? k / (n - 1) : 0;
    let semis = r.range * (0.35 - 0.6 * pos); // declination across the line
    if (s.stress) semis += r.range * 0.45;
    semis += r.lilt * r.range * (k % 2 ? -0.35 : 0.35);
    semis += jitter(r.range * 0.18);
    if (exclaim) semis += 2.5;
    if (s.shout) semis += 3 + 3 * r.shout;
    const final = s.end !== '' || k === n - 1;
    let semisEnd = semis - 0.6;
    if (final && s.end === '?') semisEnd = semis + r.range * 0.9;
    else if (final) semisEnd = semis - r.fall;
    const f = r.f0 * Math.pow(2, semis / 12);
    const fe = r.f0 * Math.pow(2, semisEnd / 12);
    const fq = src.frequency;
    fq.setTargetAtTime(f, ts, r.glide);
    fq.setTargetAtTime(fe, ts + d * 0.55, d * 0.35);

    // vowel colour, with a little coarticulation glide in
    const v = VOWELS[s.vowel] ?? VOWELS.y;
    const vv = s.stress ? v : mix(v, VOWELS.y, 0.35); // unstressed vowels drift toward "uh"
    F.forEach((fl, j) => {
      const target = vv[j] * fs * (s.shout ? 1.06 : 1);
      if (s.onset === 'glide' && j === 1) fl.frequency.setTargetAtTime(VOWELS.u[1] * fs, ts - 0.02, 0.01);
      fl.frequency.setTargetAtTime(target, ts, 0.025);
    });

    // loudness: stressed and shouted syllables are bigger; hums start closed
    let a = level * (s.stress ? 1 : 0.72) * (s.shout ? 1.25 + 0.3 * r.shout : 1) * (anyShout && !s.shout ? 0.85 : 1);
    a *= 0.85 + 0.3 * rnd();
    const g = amp.gain;
    const atk = s.onset === 'glide' || s.onset === 'none' ? 0.03 : 0.012;
    if (s.onset === 'hum') {
      g.setValueAtTime(FLOOR, ts);
      g.linearRampToValueAtTime(a * 0.3, ts + 0.012);
      g.setValueAtTime(a * 0.3, ts + 0.045);
      g.linearRampToValueAtTime(a, ts + 0.065);
    } else {
      g.setValueAtTime(FLOOR, ts);
      g.linearRampToValueAtTime(a, ts + atk);
    }
    const rel = Math.min(0.05, d * 0.3);
    g.setValueAtTime(a * 0.9, ts + d - rel);
    g.exponentialRampToValueAtTime(FLOOR, ts + d + (s.gap > 0.05 ? 0.02 : 0.004));

    // the consonant in front of it
    if (s.onset === 'stop' || s.onset === 'hiss' || s.onset === 'breath') {
      const cs = s.onset === 'hiss' ? ts - 0.05 : ts - 0.012;
      const cl = s.onset === 'hiss' ? 0.055 : s.onset === 'breath' ? 0.04 : 0.014;
      const ca = level * (s.onset === 'hiss' ? 0.5 : s.onset === 'breath' ? 0.25 : 0.7) * r.crisp;
      cf.frequency.setValueAtTime(s.burst * (s.onset === 'breath' ? fs : 1), Math.max(t0, cs));
      cg.gain.setValueAtTime(FLOOR, Math.max(t0, cs));
      cg.gain.linearRampToValueAtTime(ca, Math.max(t0, cs) + 0.004);
      cg.gain.exponentialRampToValueAtTime(FLOOR, Math.max(t0, cs) + cl);
    }
  }
  amp.gain.setValueAtTime(FLOOR, end - 0.05);
  return end - t0;
}

function mix(a: readonly number[], b: readonly number[], k: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}
