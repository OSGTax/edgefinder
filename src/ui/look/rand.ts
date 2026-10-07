// Cosmetic randomness for the hand-made look. Seeded from a string so the same
// word always comes out with the same wobble (no jitter when the HUD redraws).

export function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** mulberry32: returns a function giving floats in [0, 1). */
export function rng(seed: number | string): () => number {
  let a = typeof seed === 'string' ? hashStr(seed) : seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A float in [-1, 1). */
export const sym = (r: () => number) => r() * 2 - 1;

/** Number formatted for SVG (two decimals, no trailing zeros). */
export const n2 = (v: number) => String(Math.round(v * 100) / 100);
