// Seeded, tileable value noise for procedural textures and geometry.

export class Noise2 {
  private perm: Uint8Array;
  private grad: Float32Array;
  constructor(seed = 1) {
    let s = seed >>> 0 || 1;
    const rnd = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), s | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    this.perm = new Uint8Array(512);
    this.grad = new Float32Array(256);
    const p = Array.from({ length: 256 }, (_, i) => i);
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [p[i], p[j]] = [p[j], p[i]];
    }
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
    for (let i = 0; i < 256; i++) this.grad[i] = rnd() * 2 - 1;
  }

  /** Value noise in [-1, 1]; `period` (integer cells) makes it tile. */
  value(x: number, y: number, period = 256): number {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const P = this.perm, G = this.grad;
    let x0: number, x1: number, y0: number, y1: number;
    if (period === (period | 0)) {
      // integer period (the common case): wrap without closures or double modulo
      x0 = xi % period; if (x0 < 0) x0 += period;
      y0 = yi % period; if (y0 < 0) y0 += period;
      x1 = x0 + 1; if (x1 >= period) x1 -= period;
      y1 = y0 + 1; if (y1 >= period) y1 -= period;
      x0 &= 255; x1 &= 255; y0 &= 255; y1 &= 255;
    } else {
      const w = (a: number) => ((a % period) + period) % period & 255;
      x0 = w(xi); x1 = w(xi + 1); y0 = w(yi); y1 = w(yi + 1);
    }
    const p0 = P[x0], p1 = P[x1];
    const a = G[P[p0 + y0]], b = G[P[p1 + y0]], c = G[P[p0 + y1]], d = G[P[p1 + y1]];
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  }

  /** Fractal sum of octaves; tiles when x,y span `period` cells at octave 0. */
  fbm(x: number, y: number, octaves = 4, period = 256, lacunarity = 2, gain = 0.5): number {
    let sum = 0, amp = 0.5, f = 1, norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * this.value(x * f, y * f, period * f);
      norm += amp;
      amp *= gain;
      f *= lacunarity;
    }
    return sum / norm;
  }
}

export function mulberry(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
