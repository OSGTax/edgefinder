export type Hand = 'R' | 'L';
export type Bats = 'R' | 'L' | 'S';
export type Position = 'P' | 'C' | '1B' | '2B' | 'SS' | '3B' | 'LF' | 'CF' | 'RF';
export const POSITIONS: Position[] = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF'];

export type PitchType = 'fastball' | 'curve' | 'changeup' | 'slider' | 'sinker' | 'knuckler';

/**
 * Every kid has one signature special. Batting/pitching specials are fired
 * from a full hype meter; the others are passive perks that are always on.
 */
export type Special =
  | 'moonshot'   // bat: towering power swing
  | 'laser'      // bat: guaranteed hard line drive if you make contact
  | 'eagleEye'   // bat: huge sweet spot
  | 'heater'     // pitch: blazing fastball
  | 'wobbler'    // pitch: wild knuckleball dance
  | 'loopy'      // pitch: rainbow moonball that wrecks timing
  | 'freeze'     // pitch: ball stalls mid-air, then zips
  | 'rocketArm'  // perk: big throwing arm
  | 'flypaper'   // perk: sure hands, bigger catch radius
  | 'springs'    // perk: jumps high — can rob home runs
  | 'zoomies';   // perk: extra speed on the bases

export const SPECIAL_INFO: Record<Special, { label: string; kind: 'bat' | 'pitch' | 'perk'; blurb: string }> = {
  moonshot: { label: 'Moonshot', kind: 'bat', blurb: 'A towering power swing.' },
  laser: { label: 'Laser Beam', kind: 'bat', blurb: 'Any contact becomes a screaming liner.' },
  eagleEye: { label: 'Eagle Eye', kind: 'bat', blurb: 'The sweet spot gets huge.' },
  heater: { label: 'Heater', kind: 'pitch', blurb: 'A blazing fastball.' },
  wobbler: { label: 'Wobbler', kind: 'pitch', blurb: 'A knuckleball that dances all over.' },
  loopy: { label: 'Loop-de-Loop', kind: 'pitch', blurb: 'A rainbow moonball that wrecks timing.' },
  freeze: { label: 'Brain Freeze', kind: 'pitch', blurb: 'The ball stalls in mid-air, then zips.' },
  rocketArm: { label: 'Rocket Arm', kind: 'perk', blurb: 'Throws twice as hard as anyone.' },
  flypaper: { label: 'Flypaper Glove', kind: 'perk', blurb: 'Nothing gets past this glove.' },
  springs: { label: 'Spring Sneakers', kind: 'perk', blurb: 'Leaps high enough to rob home runs.' },
  zoomies: { label: 'The Zoomies', kind: 'perk', blurb: 'Extra speed on the bases.' },
};

export type HairStyle =
  | 'buzz' | 'spiky' | 'bowl' | 'curly' | 'afro' | 'ponytail' | 'pigtails'
  | 'bob' | 'long' | 'mohawk' | 'braids' | 'bun' | 'messy' | 'sidepart';

export type Mouth = 'grin' | 'gap' | 'smirk' | 'braces' | 'open' | 'tongue' | 'frown' | 'whistle';
export type HeadShape = 'round' | 'oval' | 'square' | 'wide';

// The kids are mini adults: every one dresses and acts like a grown-up type.
export type Hat = 'cap' | 'capBack' | 'visor' | 'bucket' | 'trucker' | 'flatcap' | 'cowboy' | 'hardhat';
export type Eyewear = 'none' | 'glasses' | 'reading' | 'shades' | 'aviators' | 'goggles' | 'monocle';
/** Facial hair is marker-drawn or stick-on — they're kids. */
export type FaceFlair = 'none' | 'mustache' | 'handlebar' | 'walrus' | 'goatee' | 'beard' | 'unibrow' | 'zinc';
export type Neck = 'none' | 'tie' | 'bowtie' | 'pearls' | 'whistle' | 'scarf' | 'lanyard' | 'bandana' | 'medal';
export type Extra = 'none' | 'headset' | 'curlers' | 'pencil' | 'earpiece' | 'headband' | 'sweatband' | 'bandaid' | 'bow' | 'earrings' | 'flower';
export type BodyFlair = 'none' | 'suspenders' | 'apron' | 'pocketProtector' | 'toolbelt' | 'badge' | 'cape' | 'vest' | 'overalls' | 'fannyPack';
/** What they hold in portraits and between plays. */
export type Holding =
  | 'none' | 'coffee' | 'clipboard' | 'briefcase' | 'newspaper' | 'calculator' | 'binoculars'
  | 'gavel' | 'microphone' | 'magnifier' | 'phone' | 'juicebox' | 'lunchpail' | 'wand' | 'trophy'
  | 'rollingPin' | 'wrench' | 'spatula' | 'flag' | 'horseshoe' | 'bowlingBall' | 'fishingRod' | 'crystalBall' | 'dumbbell';

export interface KidLook {
  skin: number;       // index into SKIN palette
  hair: HairStyle;
  hairColor: number;  // index into HAIR palette
  head: HeadShape;
  mouth: Mouth;
  hat: Hat;
  eyewear: Eyewear;
  face: FaceFlair;
  neck: Neck;
  extra: Extra;
  body: BodyFlair;
  holding: Holding;
  freckles: boolean;
  /** 0 = shortest kid in the league, 1 = tallest */
  height: number;
  /** 0 = skinny, 1 = stocky */
  build: number;
}

/**
 * Every kid has exactly four traits, each 1–10, and no two kids share the
 * same four numbers. Hitting covers contact and power, Fielding covers glove
 * and arm. Everyone can pitch.
 */
export interface Traits {
  hitting: number;
  speed: number;
  fielding: number;
  pitching: number;
}

export const TRAIT_LABELS: Record<keyof Traits, string> = {
  hitting: 'Hitting',
  speed: 'Speed',
  fielding: 'Fielding',
  pitching: 'Pitching',
};

export interface Kid {
  id: string;
  first: string;
  last: string;
  nick: string;
  age: number;
  bats: Bats;
  throws: Hand;
  traits: Traits;
  pitches: PitchType[];
  special: Special;
  look: KidLook;
  /** The grown-up they're impersonating, e.g. "The Retired Plumber". */
  persona: string;
  bio: string;
  /** Catchphrases shouted in speech bubbles on big moments. */
  quips: string[];
}

export interface TeamColors {
  primary: string;
  secondary: string;
  accent: string;
}

export type LogoIcon = 'mudcat' | 'owl' | 'comet' | 'pinecone' | 'frog' | 'bee' | 'rocket' | 'lightning';

export interface Team {
  id: string;
  street: string;
  name: string;
  abbr: string;
  colors: TeamColors;
  icon: LogoIcon;
  yardId: string;
  division: 'Front Porch' | 'Back Fence';
  roster: string[]; // kid ids — the default nine
}

export type FenceKind = 'picket' | 'wood' | 'chain' | 'hedge' | 'garage' | 'house' | 'sunflower' | 'reeds' | 'barn';

export interface FenceSeg {
  a: [number, number];
  b: [number, number];
  height: number;
  kind: FenceKind;
  color?: string;
  /** a ball that hits (doesn't clear) this segment is dead: ground-rule double */
  splash?: boolean;
}

export type Surface = 'grass' | 'dirt' | 'sand' | 'mud' | 'water' | 'patio';

export interface Patch {
  surface: Surface;
  poly: [number, number][];
}

export type PropKind =
  | 'tree' | 'pool' | 'shed' | 'doghouse' | 'swingset' | 'sandbox' | 'car' | 'grill'
  | 'garden' | 'treehouse' | 'trampoline' | 'birdbath' | 'lawnchair' | 'flamingo'
  | 'sprinkler' | 'gnome' | 'wagon' | 'barn' | 'hay' | 'tire' | 'scarecrow' | 'shrub'
  | 'flowers' | 'lemonade' | 'bench' | 'grownup' | 'tractor' | 'clothesline' | 'cattails' | 'lilypads';

/** Obstacles the ball physically interacts with. */
export interface Obstacle {
  kind: 'cylinder' | 'box' | 'canopy';
  x: number; y: number;
  /** cylinder/canopy radius, box half-width */
  r: number;
  /** box half-depth */
  d?: number;
  /** cylinder/box top; canopy center height */
  h: number;
  /** canopy vertical radius */
  rz?: number;
  rot?: number;
  bounce?: number;
  /** what happens when the ball lands/hits here */
  effect?: 'dog' | 'splash';
}

export interface Prop {
  kind: PropKind;
  x: number; y: number;
  rot?: number;
  scale?: number;
  color?: string;
  variant?: number;
  /** for grown-ups: who they are, shown in the yard intro */
  label?: string;
}

export interface YardTheme {
  grass: string;
  grassAlt: string;
  dirt: string;
  sky: [string, string];
  houseWall: string;
  houseRoof: string;
  houseTrim: string;
  /** time of day for lighting */
  time: 'morning' | 'noon' | 'afternoon' | 'sunset';
  mowStripes: boolean;
}

export interface Yard {
  id: string;
  name: string;
  owner: string;
  blurb: string;
  basePath: number; // feet between bases
  moundDist: number;
  fence: FenceSeg[]; // closed polygon, house segment behind home
  /** how the infield is worn in */
  infield: 'paths' | 'dirt' | 'sand' | 'grass';
  patches: Patch[];
  /** decorations; trees, sheds, cars, hay etc. also become physical obstacles */
  props: Prop[];
  theme: YardTheme;
  /** plain-English ground rules shown before the game */
  rules: string[];
}
