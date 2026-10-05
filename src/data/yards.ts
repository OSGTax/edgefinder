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

export const YARDS: Yard[] = [
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
      // the pool deck (hard and bouncy), then the water itself
      { surface: 'patio', poly: rect(58 - 2 * Math.cos(-0.45), 104 - 2 * Math.sin(-0.45), 30.4, 17.4, -0.45) },
      { surface: 'water', poly: rect(58, 104, 22, 11, -0.45) },
      { surface: 'patio', poly: rect(-34, -18, 20, 8) },
    ],
    props: [
      { kind: 'pool', x: 58, y: 104, rot: -0.45 },
      { kind: 'lawnchair', x: 86, y: 78, rot: 2.6 },
      { kind: 'lawnchair', x: 30, y: 88, rot: 0.4 },
      { kind: 'flamingo', x: 87, y: 57, rot: 2.2 },
      { kind: 'flamingo', x: 89.5, y: 64, rot: 2.9 },
      { kind: 'gnome', x: 9, y: -24.5, rot: 0.3 },
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
];

export const YARD_BY_ID: Record<string, Yard> = Object.fromEntries(YARDS.map((y) => [y.id, y]));

export function yard(id: string): Yard {
  const y = YARD_BY_ID[id];
  if (!y) throw new Error(`unknown yard ${id}`);
  return y;
}

export type { Patch, Prop };
