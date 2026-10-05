import type { FenceKind, FenceSeg, Patch, Prop, Yard } from './types';

// Field layout: home plate at the origin, second base straight up +y.
// Fences are closed rings; segment i runs from point i to point i+1.

type RingPt = [x: number, y: number, height: number, kind: FenceKind, opts?: { splash?: boolean; color?: string }];

function ring(pts: RingPt[]): FenceSeg[] {
  return pts.map(([x, y, height, kind, opts], i) => {
    const [nx, ny] = pts[(i + 1) % pts.length];
    return { a: [x, y], b: [nx, ny], height, kind, ...opts };
  });
}

const rect = (cx: number, cy: number, hw: number, hd: number, rot = 0): [number, number][] => {
  const c = Math.cos(rot), s = Math.sin(rot);
  return [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].map(([x, y]) => [cx + x * c - y * s, cy + x * s + y * c]);
};

const blob = (cx: number, cy: number, rx: number, ry: number, n = 14, wobble = 0.12, seed = 1): [number, number][] =>
  Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2;
    const w = 1 + wobble * Math.sin(a * 3 + seed) * Math.cos(a * 2 - seed);
    return [cx + Math.cos(a) * rx * w, cy + Math.sin(a) * ry * w];
  });

export const YARDS: Yard[] = [
  {
    id: 'mudpuddle',
    name: 'Mudpuddle Meadow',
    owner: 'The Rutherfords\' back forty',
    blurb: 'Mr. Rutherford said "just don\'t wreck the lawn." They wrecked the lawn.',
    basePath: 60, moundDist: 44,
    infield: 'paths',
    fence: ring([
      [-58, -30, 16, 'house'],
      [58, -30, 6, 'wood'],
      [96, 36, 6, 'wood'],
      [112, 112, 6, 'wood'],
      [62, 168, 6, 'wood'],
      [-48, 172, 6, 'wood'],
      [-104, 104, 6, 'wood'],
      [-92, 32, 6, 'wood'],
    ]),
    patches: [
      { surface: 'mud', poly: blob(-38, 112, 15, 10, 14, 0.2, 2) },
      { surface: 'mud', poly: blob(56, 132, 7, 5, 10, 0.25, 5) },
    ],
    props: [
      { kind: 'tree', x: -118, y: 150, scale: 1.3 },
      { kind: 'tree', x: 100, y: 190, scale: 1.1 },
      { kind: 'sprinkler', x: 78, y: 98 },
      { kind: 'gnome', x: -84, y: 40 },
      { kind: 'wagon', x: 84, y: 30, rot: 0.4 },
      { kind: 'tire', x: -70, y: 6 },
      { kind: 'grownup', x: 40, y: -26, variant: 2, label: 'Mr. Rutherford, worried about his lawn' },
      { kind: 'lemonade', x: -42, y: -22 },
      { kind: 'flowers', x: -20, y: -27 },
      { kind: 'flowers', x: 22, y: -27 },
    ],
    theme: {
      grass: '#5fae3c', grassAlt: '#57a336', dirt: '#a8794d',
      sky: ['#8fd3ff', '#e8f7ff'],
      houseWall: '#d7c6a5', houseRoof: '#6a4a3a', houseTrim: '#ffffff',
      time: 'afternoon', mowStripes: false,
    },
    rules: [
      'Ball in the mud puddle stops dead. Good luck getting it out.',
      'Over the wooden fence is a home run. Mr. Rutherford will be told.',
    ],
  },
  {
    id: 'poolparty',
    name: 'Pool Party Paradise',
    owner: 'The Mendozas\' place',
    blurb: 'An in-ground pool in shallow right field. Mrs. Mendoza has asked everyone to please aim left.',
    basePath: 60, moundDist: 44,
    infield: 'paths',
    fence: ring([
      [-60, -28, 18, 'house'],
      [60, -28, 5, 'picket'],
      [90, 40, 5, 'picket'],
      [100, 100, 5, 'picket'],
      [56, 150, 5, 'picket'],
      [-40, 172, 7, 'hedge'],
      [-112, 112, 7, 'hedge'],
      [-96, 34, 7, 'hedge'],
    ]),
    patches: [
      { surface: 'water', poly: rect(58, 104, 22, 11, -0.45) },
      { surface: 'patio', poly: rect(-34, -18, 20, 8) },
    ],
    props: [
      { kind: 'pool', x: 58, y: 104, rot: -0.45 },
      { kind: 'lawnchair', x: 86, y: 78, rot: 2.6 },
      { kind: 'lawnchair', x: 30, y: 88, rot: 0.4 },
      { kind: 'flamingo', x: 82, y: 132 },
      { kind: 'flamingo', x: 87, y: 128 },
      { kind: 'grill', x: -46, y: -16 },
      { kind: 'grownup', x: -40, y: -14, variant: 0, label: 'Mr. Mendoza, at the grill since 9 a.m.' },
      { kind: 'tree', x: -120, y: 168, scale: 1.2 },
      { kind: 'shrub', x: -60, y: 178 },
      { kind: 'shrub', x: -88, y: 150 },
    ],
    theme: {
      grass: '#66b845', grassAlt: '#5aa83b', dirt: '#b98a5a',
      sky: ['#6ec6ff', '#e6f6ff'],
      houseWall: '#f2d3b3', houseRoof: '#b5523b', houseTrim: '#ffffff',
      time: 'noon', mowStripes: true,
    },
    rules: [
      'Ball in the pool is a Splash Double. Mr. Mendoza fishes it out with the skimmer.',
      'The right-field picket fence is short. Very short. Suspiciously short.',
    ],
  },
  {
    id: 'lilypad',
    name: 'Lily Pad Pond',
    owner: 'The Fujimotos\' yard',
    blurb: 'Right field slopes down into a frog pond. Over the cattails is a home run; rolling in is a Splash Double.',
    basePath: 60, moundDist: 44,
    infield: 'paths',
    fence: ring([
      [-58, -30, 16, 'house'],
      [58, -30, 5, 'picket'],
      [92, 30, 2.5, 'reeds', { splash: true }],
      [118, 118, 2.5, 'reeds', { splash: true }],
      [40, 172, 6, 'wood'],
      [-60, 168, 6, 'wood'],
      [-108, 108, 6, 'wood'],
      [-94, 30, 6, 'wood'],
    ]),
    patches: [],
    props: [
      { kind: 'cattails', x: 105, y: 72 },
      { kind: 'cattails', x: 98, y: 120 },
      { kind: 'cattails', x: 80, y: 148 },
      { kind: 'lilypads', x: 128, y: 100 },
      { kind: 'doghouse', x: -78, y: 22, rot: 0.6 },
      { kind: 'birdbath', x: -64, y: 128 },
      { kind: 'tree', x: -40, y: 196, scale: 1.4 },
      { kind: 'tree', x: -120, y: 130, scale: 1 },
      { kind: 'grownup', x: -36, y: -24, variant: 3, label: 'Grandpa Fujimoto, who keeps score in his head' },
      { kind: 'bench', x: -30, y: -22 },
    ],
    theme: {
      grass: '#5aa94a', grassAlt: '#519c41', dirt: '#9e7a52',
      sky: ['#9ad8ff', '#f0faff'],
      houseWall: '#c9d6c3', houseRoof: '#3e4a5c', houseTrim: '#f7f2e8',
      time: 'morning', mowStripes: false,
    },
    rules: [
      'Over the cattails on the fly: home run into the pond.',
      'Rolls into the cattails: Splash Double. Watch for frogs.',
    ],
  },
  {
    id: 'junklot',
    name: 'The Junk Lot',
    owner: 'Behind the Elm Court garages',
    blurb: 'Chain-link, cracked dirt, and the old station wagon nobody can remember owning.',
    basePath: 60, moundDist: 44,
    infield: 'dirt',
    fence: ring([
      [-56, -26, 12, 'garage', { color: '#8b8f96' }],
      [56, -26, 7, 'chain'],
      [94, 40, 7, 'chain'],
      [106, 106, 7, 'chain'],
      [60, 160, 11, 'garage', { color: '#b0675a' }],
      [-30, 168, 11, 'garage', { color: '#d9c48a' }],
      [-108, 108, 7, 'chain'],
      [-92, 32, 7, 'chain'],
    ]),
    patches: [
      { surface: 'dirt', poly: blob(-50, 120, 22, 14, 12, 0.25, 7) },
      { surface: 'dirt', poly: blob(40, 96, 14, 10, 12, 0.3, 3) },
    ],
    props: [
      { kind: 'car', x: 66, y: 128, rot: 0.9 },
      { kind: 'tire', x: -70, y: 140 },
      { kind: 'tire', x: -64, y: 146 },
      { kind: 'tire', x: 80, y: 20 },
      { kind: 'shed', x: -80, y: 12, rot: 0.2 },
      { kind: 'grownup', x: 38, y: -22, variant: 2, label: 'Mr. Price, who just wants to park his car' },
    ],
    theme: {
      grass: '#7aa04a', grassAlt: '#6f9443', dirt: '#b08a62',
      sky: ['#f7b267', '#fde2c0'],
      houseWall: '#8b8f96', houseRoof: '#4a4e55', houseTrim: '#c9cdd3',
      time: 'sunset', mowStripes: false,
    },
    rules: [
      'Off the station wagon is in play. Off its roof is still in play. Through its window costs your allowance.',
      'Over the garages: home run.',
    ],
  },
  {
    id: 'treehouse',
    name: 'Treehouse Woods',
    owner: 'The Quills\' yard',
    blurb: 'A giant oak with a treehouse looms over left-center. The woods beyond the hedge go back forever.',
    basePath: 60, moundDist: 44,
    infield: 'paths',
    fence: ring([
      [-60, -30, 18, 'house'],
      [60, -30, 7, 'hedge'],
      [96, 40, 7, 'hedge'],
      [116, 116, 7, 'hedge'],
      [56, 182, 8, 'hedge'],
      [-56, 184, 8, 'hedge'],
      [-112, 112, 7, 'hedge'],
      [-96, 36, 7, 'hedge'],
    ]),
    patches: [],
    props: [
      { kind: 'treehouse', x: -54, y: 128, scale: 1.25 },
      { kind: 'tree', x: 130, y: 150, scale: 1.3 },
      { kind: 'tree', x: -130, y: 160, scale: 1.4 },
      { kind: 'tree', x: 20, y: 214, scale: 1.5 },
      { kind: 'tree', x: -70, y: 220, scale: 1.3 },
      { kind: 'tire', x: 90, y: 60 },
      { kind: 'grownup', x: 42, y: -24, variant: 1, label: 'Mrs. Quill, reading on the deck' },
    ],
    theme: {
      grass: '#4f9e3a', grassAlt: '#469233', dirt: '#9a7148',
      sky: ['#a6dcff', '#effaff'],
      houseWall: '#b9cbd9', houseRoof: '#3d4e6a', houseTrim: '#ffffff',
      time: 'afternoon', mowStripes: true,
    },
    rules: [
      'Ball stuck in the treehouse oak? It\'s in play until it comes down.',
      'Over the hedge is a home run. Nobody goes into the woods.',
    ],
  },
  {
    id: 'grandmabea',
    name: 'Grandma Bea\'s Garden',
    owner: 'Grandma Bea\'s house',
    blurb: 'Tomato beds in right-center, gnomes everywhere, and Grandma Bea watching from her rocker.',
    basePath: 60, moundDist: 44,
    infield: 'grass',
    fence: ring([
      [-56, -28, 16, 'house'],
      [56, -28, 4.5, 'picket'],
      [94, 34, 4.5, 'picket'],
      [108, 108, 4.5, 'picket'],
      [50, 164, 4.5, 'picket'],
      [-50, 166, 4.5, 'picket'],
      [-104, 104, 4.5, 'picket'],
      [-90, 30, 4.5, 'picket'],
    ]),
    patches: [
      { surface: 'dirt', poly: rect(46, 128, 16, 7, -0.5) },
      { surface: 'dirt', poly: rect(66, 112, 12, 6, -0.5) },
    ],
    props: [
      { kind: 'garden', x: 46, y: 128, rot: -0.5 },
      { kind: 'garden', x: 66, y: 112, rot: -0.5, variant: 1 },
      { kind: 'birdbath', x: 0, y: 150 },
      { kind: 'gnome', x: -40, y: 160 },
      { kind: 'gnome', x: 30, y: 158 },
      { kind: 'gnome', x: -96, y: 60 },
      { kind: 'tree', x: -96, y: 8, scale: 1.2 },
      { kind: 'clothesline', x: -60, y: 190 },
      { kind: 'grownup', x: -30, y: -24, variant: 1, label: 'Grandma Bea, in her rocker since 1987' },
      { kind: 'flowers', x: 30, y: -25 },
    ],
    theme: {
      grass: '#68b448', grassAlt: '#5ea640', dirt: '#ab7f55',
      sky: ['#8fd3ff', '#fff7e6'],
      houseWall: '#f6e7c8', houseRoof: '#7c5b8a', houseTrim: '#ffffff',
      time: 'afternoon', mowStripes: false,
    },
    rules: [
      'Ball in the tomato beds: play it where it lies. Do NOT step on the tomatoes.',
      'Hit the birdbath and Grandma Bea gives you a cookie. It is still in play.',
    ],
  },
  {
    id: 'sandbox',
    name: 'Sandbox Stadium',
    owner: 'The Duffys\' yard',
    blurb: 'The world\'s biggest sandbox swallows the middle of the infield. Grounders go there to die.',
    basePath: 60, moundDist: 44,
    infield: 'sand',
    fence: ring([
      [-54, -28, 15, 'house'],
      [54, -28, 5, 'picket'],
      [86, 30, 5, 'picket'],
      [100, 100, 5, 'wood'],
      [44, 154, 5, 'wood'],
      [-56, 150, 5, 'wood'],
      [-98, 98, 5, 'wood'],
      [-86, 28, 5, 'picket'],
    ]),
    patches: [],
    props: [
      { kind: 'sandbox', x: 0, y: 64 },
      { kind: 'swingset', x: -74, y: 16, rot: 0.7 },
      { kind: 'trampoline', x: -60, y: 182 },
      { kind: 'wagon', x: 72, y: 26, rot: -0.6 },
      { kind: 'tree', x: 112, y: 170, scale: 1.1 },
      { kind: 'grownup', x: 36, y: -24, variant: 3, label: 'Mrs. Duffy, cheering for both teams' },
      { kind: 'lemonade', x: -36, y: -22 },
    ],
    theme: {
      grass: '#6cbb4a', grassAlt: '#62ad42', dirt: '#e3c98f',
      sky: ['#7fd0ff', '#eefaff'],
      houseWall: '#ffe9a8', houseRoof: '#c45d3a', houseTrim: '#ffffff',
      time: 'noon', mowStripes: true,
    },
    rules: [
      'The whole infield is sandbox. Bunts are basically home runs.',
      'Over the fence onto the trampoline: home run, and it might bounce back.',
    ],
  },
  {
    id: 'sunflower',
    name: 'Sunflower Farm',
    owner: 'Grandpa Lou\'s farm',
    blurb: 'Hay bales, a wall of sunflowers in right, and the big red barn in dead center.',
    basePath: 60, moundDist: 44,
    infield: 'paths',
    fence: ring([
      [-64, -32, 18, 'house'],
      [64, -32, 6, 'wood'],
      [100, 40, 9, 'sunflower'],
      [118, 118, 9, 'sunflower'],
      [40, 186, 22, 'barn'],
      [-40, 190, 6, 'wood'],
      [-122, 122, 6, 'wood'],
      [-100, 36, 6, 'wood'],
    ]),
    patches: [
      { surface: 'dirt', poly: blob(-70, 150, 14, 9, 12, 0.2, 4) },
    ],
    props: [
      { kind: 'hay', x: -84, y: 120, rot: 0.3 },
      { kind: 'hay', x: -78, y: 126, rot: 0.2 },
      { kind: 'hay', x: -82, y: 124, rot: 0.25, variant: 1 },
      { kind: 'scarecrow', x: 62, y: 150 },
      { kind: 'barn', x: 0, y: 210 },
      { kind: 'tractor', x: -112, y: 60, rot: 1.2 },
      { kind: 'grownup', x: -104, y: 66, variant: 2, label: 'Grandpa Lou, on the tractor, refusing to move it' },
      { kind: 'tree', x: -150, y: 170, scale: 1.4 },
    ],
    theme: {
      grass: '#79b642', grassAlt: '#6fa83c', dirt: '#b58653',
      sky: ['#ffd89a', '#fff6e0'],
      houseWall: '#fbf3e4', houseRoof: '#8a3b2e', houseTrim: '#ffffff',
      time: 'sunset', mowStripes: false,
    },
    rules: [
      'Off the barn wall is in play. Over the barn roof is a Barn Burner home run.',
      'Into the sunflowers is in play if anyone can find it.',
    ],
  },
];

export const YARD_BY_ID: Record<string, Yard> = Object.fromEntries(YARDS.map((y) => [y.id, y]));

export function yard(id: string): Yard {
  const y = YARD_BY_ID[id];
  if (!y) throw new Error(`unknown yard ${id}`);
  return y;
}

export type { Patch, Prop };
