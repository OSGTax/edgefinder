/**
 * Who sounds like what. One hand-picked voice per kid (they're 9 to 12, so
 * every one is a kid's voice, just doing their grown-up impression), the two
 * announcers on Channel 4½, and the Mendozas. Keyed by kid id, like the face
 * recipes; anyone missing (a future team) gets a voice derived from their id.
 */

export interface VoiceRecipe {
  /** speaking pitch, Hz */
  f0: number;
  /** how far the pitch moves, semitones */
  range: number;
  /** syllables per second */
  rate: number;
  /** formant scale: 1 = grown-up, kids sit around 1.2–1.35 */
  formant: number;
  /** 0..1 air in the voice */
  breath: number;
  /** source darkness: higher = mellower (1.2 bright .. 2 dark) */
  tilt: number;
  /** 0..1 rasp */
  gruff: number;
  /** 0..1 sing-song up-and-down */
  lilt: number;
  /** 0..1 how hard they shout a word in caps */
  shout: number;
  /** semitones the pitch drops at the end of a statement */
  fall: number;
  /** final-syllable stretch (1 = none) */
  stretch: number;
  /** pitch glide time constant (s): small = choppy, big = smooth */
  glide: number;
  /** formant sharpness (1 = normal) */
  focus: number;
  /** brightness of the third formant */
  bright: number;
  /** consonant crispness */
  crisp: number;
  /** loudness */
  level: number;
  /** most syllables per line */
  max: number;
  /** broadcast filter */
  radio: boolean;
}

const BASE: VoiceRecipe = {
  f0: 290, range: 5, rate: 5.6, formant: 1.28, breath: 0.12, tilt: 1.5, gruff: 0, lilt: 0.2, shout: 0.4,
  fall: 3, stretch: 1.3, glide: 0.03, focus: 1, bright: 1, crisp: 1, level: 0.5, max: 10, radio: false,
};

const v = (o: Partial<VoiceRecipe>): VoiceRecipe => ({ ...BASE, ...o });

export const VOICES: Record<string, VoiceRecipe> = {
  // ── the booth: Channel 4½, Maple Hollow Public Access ──
  // Chet, 10, doing his deepest newsman voice: quick, punchy, every line lands down
  Chet: v({ f0: 245, range: 5, rate: 6.6, formant: 1.2, tilt: 1.6, lilt: 0.35, fall: 4.5, stretch: 1.2, level: 0.3, max: 12, radio: true }),
  // Dottie, 12, unimpressed: slower, wider, a drawl at the end of every line
  Dottie: v({ f0: 270, range: 6.5, rate: 5.2, formant: 1.24, breath: 0.2, tilt: 1.7, lilt: 0.15, fall: 3.5, stretch: 1.7, glide: 0.05, level: 0.3, max: 11, radio: true }),

  // ── Maple Street Mudcats ──
  bo: v({ f0: 200, range: 3, rate: 4.6, formant: 1.16, tilt: 1.9, gruff: 0.6, fall: 4, stretch: 1.4, crisp: 0.7 }), // grumbles like a retired plumber
  ines: v({ f0: 300, range: 9, rate: 4.2, formant: 1.3, breath: 0.35, lilt: 0.45, stretch: 2, glide: 0.06, shout: 0.7 }), // every word a season finale
  toby: v({ f0: 275, range: 4, rate: 6.4, formant: 1.27, tilt: 1.3, shout: 0.9, stretch: 1.05, glide: 0.015, crisp: 1.3, level: 0.6 }), // barks orders
  wren: v({ f0: 365, range: 5, rate: 5.4, formant: 1.38, breath: 0.6, tilt: 1.8, lilt: 0.35, shout: 0.1, level: 0.36, crisp: 0.6 }), // whispers so she won't scare the birds
  dez: v({ f0: 150, range: 3, rate: 3.8, formant: 1.12, breath: 0.25, tilt: 2, lilt: 0.1, fall: 2, stretch: 1.8, glide: 0.08, crisp: 0.6 }), // two octaves under his real voice, smooth
  priya: v({ f0: 290, range: 2.5, rate: 7.2, formant: 1.3, tilt: 1.4, lilt: 0, stretch: 1, glide: 0.012, crisp: 1.4, focus: 1.2 }), // precise, itemized
  gus: v({ f0: 215, range: 4, rate: 4.5, formant: 1.15, tilt: 1.7, gruff: 0.3, shout: 0.8, level: 0.5 }), // TIMBERRR
  molly: v({ f0: 320, range: 6, rate: 6, formant: 1.32, tilt: 1.3, lilt: 0.55, bright: 1.6, focus: 1.3 }), // deli-counter sing-song, a bit nasal
  jun: v({ f0: 290, range: 8, rate: 7.2, formant: 1.28, tilt: 1.3, shout: 0.6, fall: 1.5, lilt: 0.3, level: 0.55 }), // the infomercial pitch, always going up

  // ── Cedar Lane Comets ──
  kai: v({ f0: 255, range: 7, rate: 6, formant: 1.24, tilt: 1.2, gruff: 0.25, shout: 1, bright: 1.4, level: 0.55 }), // full volume, all the time
  ruby: v({ f0: 335, range: 8, rate: 6.4, formant: 1.34, tilt: 1.35, lilt: 0.5, fall: 1.5, bright: 1.3 }), // up since 5 a.m. and THRILLED
  ezra: v({ f0: 245, range: 4, rate: 6.6, formant: 1.22, tilt: 1.6, lilt: 0.2, glide: 0.04, crisp: 1.1 }), // the smooth sales patter
  maya: v({ f0: 320, range: 6, rate: 7, formant: 1.33, tilt: 1.4, stretch: 1.1, glide: 0.02 }), // quick, on her route
  leo: v({ f0: 255, range: 7, rate: 5, formant: 1.24, tilt: 1.5, lilt: 0.8, fall: -2, stretch: 1.6, glide: 0.06 }), // "French", sing-song, rises at the end
  anya: v({ f0: 300, range: 6, rate: 5.6, formant: 1.3, tilt: 1.5, lilt: 0.4, fall: 3, stretch: 1.4 }), // the weather-lady lilt
  darius: v({ f0: 230, range: 6, rate: 6, formant: 1.2, tilt: 1.5, shout: 0.8, crisp: 1.3, level: 0.58 }), // OBJECTION
  pepper: v({ f0: 330, range: 3.5, rate: 11, formant: 1.35, tilt: 1.4, lilt: 0.3, stretch: 1, glide: 0.01, crisp: 1.2, max: 16 }), // the auctioneer's patter
  hank: v({ f0: 175, range: 3, rate: 4, formant: 1.1, breath: 0.2, tilt: 1.9, shout: 0, fall: 2, level: 0.45 }), // biggest, gentlest

  // ── grown-ups ──
  mrsMendoza: v({ f0: 215, range: 7, rate: 6, formant: 1.06, tilt: 1.6, lilt: 0.3, shout: 0.6, level: 0.5 }),
  mrMendoza: v({ f0: 120, range: 4, rate: 4.6, formant: 1, tilt: 1.8, gruff: 0.15, level: 0.5 }),
};

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** The recipe for a speaker; unknown kids get a stable voice made from their id. */
export function recipe(who: string): VoiceRecipe {
  const r = VOICES[who];
  if (r) return r;
  const h = hash(who);
  const u = (k: number) => ((h >>> (k * 4)) & 15) / 15;
  return v({ f0: 240 + 120 * u(0), range: 4 + 4 * u(1), rate: 5 + 2 * u(2), formant: 1.2 + 0.15 * u(3), lilt: 0.5 * u(4), breath: 0.3 * u(5) });
}
