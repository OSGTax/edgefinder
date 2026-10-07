import type { Kid } from '../data/types';

// A hand-picked face for every kid (and Mr. Mendoza), chosen to fit their
// persona: eye shape, size and spacing, brows, nose, resting mouth, ears,
// freckles, teeth and cheeks. The model, the face painter and the rig all read
// from here, so a kid's face is designed in one place.

/** Eye outline. All of them have big dark irises; the shape changes the opening. */
export type EyeShape =
  | 'round'   // big and open: curious, sweet
  | 'wide'    // a little wider than tall: eager, sparkly
  | 'tall'    // narrow ovals: a bit theatrical
  | 'almond'  // slightly narrower with a lifted outer corner: sharp, confident
  | 'sleepy'  // lids rest a touch lower: relaxed, mellow
  | 'button'; // small round eyes: gentle, old-soul

export type BrowStyle =
  | 'straight' | 'arched' | 'bushy' | 'thin' | 'angled' | 'worried' | 'tufty' | 'soft' | 'flat';

export type NoseStyle = 'button' | 'round' | 'small' | 'long' | 'snub' | 'broad';

/** How the mouth sits when the kid isn't showing any particular feeling. */
export type RestMouth =
  | 'smile'    // soft smile with little corner tucks
  | 'grin'     // wide closed smile, cheeks up
  | 'toothy'   // open smile showing the top teeth
  | 'smirk'    // one-sided
  | 'tongue'   // smile with the tongue poking out
  | 'grumble'  // flat little old-man mouth (still friendly)
  | 'chatter'  // small open "oh!" mid-sentence
  | 'pursed'   // small, pleased, fussy
  | 'tight'    // small neat closed smile
  | 'cool'     // relaxed half smile
  | 'dimples'; // smile with dimples

export interface FaceRecipe {
  eye: EyeShape;
  /** eye size multiplier (about 0.85..1.15) */
  eyeSize: number;
  /** eye spacing multiplier (about 0.9..1.1) */
  eyeGap: number;
  /** iris colour (kept dark enough to read as a friendly dot from far away) */
  iris: string;
  /** 0 none, 1 a little outer flick, 2 three lashes */
  lashes: 0 | 1 | 2;
  brow: BrowStyle;
  /** brow thickness multiplier */
  browThick: number;
  nose: NoseStyle;
  mouth: RestMouth;
  /** mouth width multiplier */
  mouthWidth: number;
  /** ear size multiplier */
  ears: number;
  freckles: 0 | 1 | 2;
  /** cheek blush strength 0..1 */
  cheeks: number;
  gapTooth?: boolean;
  braces?: boolean;
  /** a little beauty mark (Ines) */
  mole?: boolean;
  /** extra brow height in face degrees (to clear big glasses) */
  browLift?: number;
}

const BROWN = '#4a2c17', DARK = '#2a1a10', HAZEL = '#5b4a22', BLUE = '#2f5a86', GREEN = '#36643a', AMBER = '#6a4416', GREY = '#4b5a66';

export const FACE_RECIPES: Record<string, FaceRecipe> = {
  // ── Maple Street Mudcats
  // Retired plumber: little kind button eyes under bushy brows, a big round nose, grumbly mouth.
  bo: { eye: 'button', eyeSize: 0.9, eyeGap: 1.04, iris: BROWN, lashes: 0, brow: 'bushy', browThick: 1.35, nose: 'round', mouth: 'grumble', mouthWidth: 0.9, ears: 1.2, freckles: 0, cheeks: 0.45 },
  // Soap-opera star: dramatic thin arched brows, lashes, a smirk and a beauty mark.
  ines: { eye: 'almond', eyeSize: 1.0, eyeGap: 0.98, iris: DARK, lashes: 2, brow: 'arched', browThick: 0.8, nose: 'small', mouth: 'smirk', mouthWidth: 1.0, ears: 0.9, freckles: 0, cheeks: 0.35, mole: true },
  // Gym teacher: wide-open eager eyes, straight no-nonsense brows, a big coach grin.
  toby: { eye: 'wide', eyeSize: 1.02, eyeGap: 1.0, iris: BLUE, lashes: 0, brow: 'straight', browThick: 1.2, nose: 'button', mouth: 'grin', mouthWidth: 1.12, ears: 1.05, freckles: 0, cheeks: 0.4 },
  // Bird watcher: the biggest, roundest eyes in the league, curious high brows, freckles, a gap tooth.
  wren: { eye: 'round', eyeSize: 1.14, eyeGap: 0.96, iris: GREEN, lashes: 1, brow: 'thin', browThick: 0.85, nose: 'snub', mouth: 'chatter', mouthWidth: 0.8, ears: 1.0, freckles: 2, cheeks: 0.5, gapTooth: true },
  // Late-night DJ: mellow sleepy eyes, flat heavy brows, a cool half smile.
  dez: { eye: 'sleepy', eyeSize: 0.98, eyeGap: 1.04, iris: DARK, lashes: 0, brow: 'flat', browThick: 1.3, nose: 'broad', mouth: 'cool', mouthWidth: 1.0, ears: 1.0, freckles: 0, cheeks: 0.2 },
  // CPA: neat almond eyes, tidy straight-thin brows, a small precise smile with braces.
  priya: { eye: 'almond', eyeSize: 0.98, eyeGap: 0.95, iris: DARK, lashes: 1, brow: 'thin', browThick: 1.0, nose: 'small', mouth: 'tight', mouthWidth: 0.82, ears: 0.95, freckles: 0, cheeks: 0.35, braces: true },
  // Lumberjack: small eyes, bushy brows, a big open toothy laugh with a gap.
  gus: { eye: 'button', eyeSize: 0.92, eyeGap: 1.0, iris: GREY, lashes: 0, brow: 'tufty', browThick: 1.4, nose: 'round', mouth: 'toothy', mouthWidth: 1.15, ears: 1.15, freckles: 1, cheeks: 0.55, gapTooth: true },
  // Deli-counter lady: round eyes, short tufty brows, tongue out in concentration, rosy freckles.
  molly: { eye: 'round', eyeSize: 1.06, eyeGap: 1.02, iris: HAZEL, lashes: 1, brow: 'soft', browThick: 1.05, nose: 'snub', mouth: 'tongue', mouthWidth: 0.95, ears: 1.0, freckles: 2, cheeks: 0.6 },
  // Infomercial host: wide sparkly eyes, high arched brows, a showroom smile.
  jun: { eye: 'wide', eyeSize: 1.06, eyeGap: 0.98, iris: DARK, lashes: 0, brow: 'arched', browThick: 1.15, nose: 'small', mouth: 'toothy', mouthWidth: 1.1, ears: 1.0, freckles: 0, cheeks: 0.35 },

  // ── Cedar Lane Comets
  // Monster-truck announcer: wide eyes, angled intense brows, a huge grin.
  kai: { eye: 'wide', eyeSize: 1.0, eyeGap: 1.04, iris: DARK, lashes: 0, brow: 'angled', browThick: 1.35, nose: 'broad', mouth: 'grin', mouthWidth: 1.2, ears: 1.0, freckles: 0, cheeks: 0.3 },
  // Morning-show host: big bright eyes with lashes, arched brows, the cheeriest grin, freckles.
  ruby: { eye: 'round', eyeSize: 1.08, eyeGap: 1.0, iris: AMBER, lashes: 2, brow: 'soft', browThick: 0.95, nose: 'button', mouth: 'toothy', mouthWidth: 1.05, ears: 0.95, freckles: 1, cheeks: 0.65 },
  // Insurance salesman: round eager eyes behind glasses, hopeful worried brows, a sales smile.
  ezra: { eye: 'round', eyeSize: 0.96, eyeGap: 0.98, iris: BROWN, lashes: 0, brow: 'worried', browThick: 1.1, nose: 'long', mouth: 'smile', mouthWidth: 0.95, ears: 1.15, freckles: 0, cheeks: 0.4 },
  // Mail carrier: bright almond eyes, determined angled brows, dimples.
  maya: { eye: 'almond', eyeSize: 1.04, eyeGap: 1.0, iris: DARK, lashes: 1, brow: 'angled', browThick: 0.95, nose: 'small', mouth: 'dimples', mouthWidth: 1.0, ears: 0.95, freckles: 0, cheeks: 0.3 },
  // "French" chef: tall theatrical eyes, snooty thin arches, a pursed pleased mouth.
  leo: { eye: 'tall', eyeSize: 1.0, eyeGap: 0.96, iris: BLUE, lashes: 0, brow: 'arched', browThick: 0.9, nose: 'long', mouth: 'pursed', mouthWidth: 0.85, ears: 1.0, freckles: 0, cheeks: 0.45 },
  // Weather lady: big round eyes with lashes, thin high brows, a bright grin.
  anya: { eye: 'round', eyeSize: 1.1, eyeGap: 1.02, iris: BLUE, lashes: 2, brow: 'thin', browThick: 0.8, nose: 'button', mouth: 'grin', mouthWidth: 1.05, ears: 0.95, freckles: 0, cheeks: 0.55 },
  // Lawyer: confident almond eyes, strong straight brows, a closing-argument smirk.
  darius: { eye: 'almond', eyeSize: 0.96, eyeGap: 1.02, iris: DARK, lashes: 0, brow: 'straight', browThick: 1.3, nose: 'broad', mouth: 'smirk', mouthWidth: 1.05, ears: 1.0, freckles: 0, cheeks: 0.2 },
  // Auctioneer: big eyes, short tufty brows, always mid-sentence, a gap tooth.
  pepper: { eye: 'wide', eyeSize: 1.1, eyeGap: 0.97, iris: DARK, lashes: 1, brow: 'tufty', browThick: 1.0, nose: 'snub', mouth: 'chatter', mouthWidth: 0.95, ears: 1.0, freckles: 1, cheeks: 0.5, gapTooth: true },
  // Mall security guard: gentle small eyes, soft worried brows, a shy kind smile.
  hank: { eye: 'button', eyeSize: 0.92, eyeGap: 1.06, iris: HAZEL, lashes: 0, brow: 'worried', browThick: 1.3, nose: 'button', mouth: 'smile', mouthWidth: 0.9, ears: 1.15, freckles: 0, cheeks: 0.5 },

  // ── grown-ups
  // Mr. Mendoza at the grill: crinkly sleepy-happy eyes, bushy brows, a big grin under the mustache.
  mrMendoza: { eye: 'sleepy', eyeSize: 0.88, eyeGap: 1.05, iris: DARK, lashes: 0, brow: 'bushy', browThick: 1.5, nose: 'broad', mouth: 'toothy', mouthWidth: 1.15, ears: 1.1, freckles: 0, cheeks: 0.35, browLift: 4 },
};

const FALLBACK: FaceRecipe = {
  eye: 'round', eyeSize: 1, eyeGap: 1, iris: BROWN, lashes: 0, brow: 'soft', browThick: 1, nose: 'button', mouth: 'smile', mouthWidth: 1, ears: 1, freckles: 0, cheeks: 0.4,
};

/** The recipe for a kid (a friendly default for anyone without one). Freckles in the kid's look always show. */
export function faceRecipe(kid: Pick<Kid, 'id' | 'look'>): FaceRecipe {
  const r = FACE_RECIPES[kid.id] ?? FALLBACK;
  return kid.look.freckles && !r.freckles ? { ...r, freckles: 1 } : r;
}

/** Eye opening by shape: width and height multipliers of the eyeball, and how far the lid rests open (radians). */
export const EYE_SHAPES: Record<EyeShape, { w: number; h: number; lidOpen: number }> = {
  round: { w: 0.86, h: 1.0, lidOpen: -1.3 },
  wide: { w: 0.94, h: 0.96, lidOpen: -1.25 },
  tall: { w: 0.76, h: 1.04, lidOpen: -1.3 },
  almond: { w: 0.88, h: 0.92, lidOpen: -1.1 },
  sleepy: { w: 0.88, h: 0.94, lidOpen: -0.95 },
  button: { w: 0.86, h: 0.9, lidOpen: -1.3 },
};
