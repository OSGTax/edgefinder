import {
  Color, DoubleSide, MeshLambertMaterial, MeshStandardMaterial, Vector2, type Material, type Texture,
} from 'three';
import {
  asphaltTex, barkTex, brickTex, concreteTex, mulchTex, poolTileTex, shingleTex, sidingTex, stripeTex, woodTex,
  type TexSet,
} from './textures';

// One shared material per look. Textures are tiled in feet: `tile` is how
// many feet one copy of the texture covers.

let texSize = 512;
/** Surface textures are painted once (neutral) and tinted per material, at most 512 px. */
export function setMaterialTexSize(n: number) { texSize = Math.min(512, n); }

const TOON_NORMAL = 0.45;

const NEUTRAL: [number, number, number] = [240, 240, 240];

const cache = new Map<string, Material>();
function memo<T extends Material>(key: string, make: () => T): T {
  let m = cache.get(key) as T | undefined;
  if (!m) { m = make(); m.name = key; cache.set(key, m); }
  return m;
}

function tiled(t: Texture | undefined, tile: number | [number, number]): Texture | null {
  if (!t) return null;
  const c = t.clone();
  const [u, v] = typeof tile === 'number' ? [tile, tile] : tile;
  c.repeat.set(1 / u, 1 / v);
  c.needsUpdate = true;
  return c;
}

function textured(key: string, set: TexSet, tile: number | [number, number], o: { roughness?: number; normal?: number; color?: string; metalness?: number } = {}) {
  return memo(key, () => new MeshStandardMaterial({
    map: tiled(set.map, tile),
    normalMap: tiled(set.normal, tile),
    // cartoon world: textures suggest the material, they don't shout (flatter bumps)
    normalScale: new Vector2((o.normal ?? 1) * TOON_NORMAL, (o.normal ?? 1) * TOON_NORMAL),
    roughness: o.roughness ?? 0.85,
    metalness: o.metalness ?? 0,
    color: o.color ?? '#ffffff',
  }));
}

const rgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

/** Brighten a hex colour (for multiplying onto darker neutral textures). */
function lift(hex: string, k: number): string {
  const [r, g, b] = rgb(hex).map((v) => Math.min(255, Math.round(v * k)));
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}

export const M = {
  /** plain painted / plastic surface */
  paint: (hex: string, roughness = 0.6, metalness = 0) =>
    memo(`paint${hex}${roughness}${metalness}`, () => new MeshStandardMaterial({ color: hex, roughness, metalness })),
  /** cheap matte surface for far-away things */
  matte: (hex: string) => memo(`matte${hex}`, () => new MeshLambertMaterial({ color: hex })),
  doubleSided: (hex: string, roughness = 0.8) =>
    memo(`ds${hex}${roughness}`, () => new MeshStandardMaterial({ color: hex, roughness, side: DoubleSide })),
  trim: () => M.paint('#f4f1ea', 0.55),
  siding: (hex: string) => textured(`siding${hex}`, sidingTex(texSize, NEUTRAL), 4, { roughness: 0.7, normal: 0.9, color: hex }),
  shingle: (hex: string) => textured(`shingle${hex}`, shingleTex(texSize, NEUTRAL), 5, { roughness: 0.95, normal: 1.2, color: lift(hex, 1.15) }),
  brick: (hex = '#96463a') => textured(`brick${hex}`, brickTex(texSize, rgb(hex)), 2.7, { roughness: 0.9, normal: 1 }),
  concrete: (hex = '#cdc8c0') => textured(`concrete${hex}`, concreteTex(texSize, NEUTRAL), 7, { roughness: 0.92, normal: 0.6, color: lift(hex, 255 / 240) }),
  asphalt: () => textured('asphalt', asphaltTex(texSize), 9, { roughness: 0.96, normal: 0.8 }),
  wood: (hex: string, planks = 4, tile: number | [number, number] = 2.5) =>
    textured(`wood${hex}${planks}${tile}`, woodTex(texSize, NEUTRAL, planks, true, planks + 2), tile, { roughness: 0.8, normal: 0.9, color: hex }),
  woodSolid: (hex: string, tile: number | [number, number] = 2) =>
    textured(`woodsolid${hex}${tile}`, woodTex(texSize, NEUTRAL, 1, false, 9), tile, { roughness: 0.75, normal: 0.6, color: hex }),
  bark: () => textured('bark', barkTex(texSize), [3, 6], { roughness: 0.95, normal: 1.4 }),
  mulch: () => textured('mulch', mulchTex(texSize), 3, { roughness: 1, normal: 1.2 }),
  poolTile: () => textured('pooltile', poolTileTex(texSize), 4, { roughness: 0.25 }),
  chrome: () => memo('chrome', () => new MeshStandardMaterial({ color: '#e8ecef', roughness: 0.12, metalness: 1 })),
  metal: (hex = '#8a9096', roughness = 0.45) => memo(`metal${hex}${roughness}`, () => new MeshStandardMaterial({ color: hex, roughness, metalness: 0.85 })),
  glass: () => memo('glass', () => new MeshStandardMaterial({ color: '#28323a', roughness: 0.04, metalness: 0.25, envMapIntensity: 1.6 })),
  stripes: (colors: string[], vertical = true) =>
    memo(`stripes${colors.join()}${vertical}`, () => new MeshStandardMaterial({ map: stripeTex(colors, 256, vertical), roughness: 0.85, side: DoubleSide })),
  tex: (key: string, map: Texture, o: { roughness?: number; metalness?: number; transparent?: boolean; double?: boolean; emissive?: string } = {}) =>
    memo(key, () => {
      const m = new MeshStandardMaterial({ map, roughness: o.roughness ?? 0.6, metalness: o.metalness ?? 0, transparent: !!o.transparent });
      if (o.double) m.side = DoubleSide;
      if (o.emissive) { m.emissive = new Color(o.emissive); m.emissiveMap = map; }
      return m;
    }),
  bulb: (hex = '#fff1c4') => memo(`bulb${hex}`, () => new MeshStandardMaterial({ color: hex, emissive: hex, emissiveIntensity: 1.6, roughness: 0.3 })),
};
