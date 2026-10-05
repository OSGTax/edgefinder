import {
  CanvasTexture, ClampToEdgeWrapping, LinearMipmapLinearFilter, NoColorSpace, RepeatWrapping, SRGBColorSpace, type Texture,
} from 'three';
import { Noise2, mulberry } from './noise';

// Every surface texture in the game is painted here, in code, onto canvases:
// color maps plus normal maps derived from a painted height field.

export interface TexSet { map: Texture; normal?: Texture; rough?: Texture }

const cache = new Map<string, unknown>();
function memo<T>(key: string, make: () => T): T {
  if (!cache.has(key)) {
    const t0 = performance.now();
    cache.set(key, make());
    const ms = performance.now() - t0;
    if (import.meta.env?.DEV && ms > 15) console.debug(`[tex] ${key} ${Math.round(ms)} ms`);
  }
  return cache.get(key) as T;
}

function canvas(w: number, h = w) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function tex(c: HTMLCanvasElement, srgb = true, repeat = true): CanvasTexture {
  const t = new CanvasTexture(c);
  t.colorSpace = srgb ? SRGBColorSpace : NoColorSpace;
  t.wrapS = t.wrapT = repeat ? RepeatWrapping : ClampToEdgeWrapping;
  t.anisotropy = 8;
  t.minFilter = LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

/** Normal map from a grayscale height field (Float32Array, size×size, tileable). */
function normalFromHeight(h: Float32Array, size: number, strength: number): HTMLCanvasElement {
  const c = canvas(size);
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  for (let y = 0; y < size; y++) {
    const row = y * size;
    const up = ((y + size - 1) % size) * size, dn = ((y + 1) % size) * size;
    for (let x = 0; x < size; x++) {
      const xl = x === 0 ? size - 1 : x - 1, xr = x === size - 1 ? 0 : x + 1;
      const dx = (h[row + xr] - h[row + xl]) * strength;
      const dy = (h[dn + x] - h[up + x]) * strength;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const i = (row + x) * 4;
      d[i] = (-dx * inv * 0.5 + 0.5) * 255;
      d[i + 1] = (dy * inv * 0.5 + 0.5) * 255;
      d[i + 2] = (inv * 0.5 + 0.5) * 255;
      d[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Read a canvas back as a height field (luminance). */
function heightOf(c: HTMLCanvasElement): Float32Array {
  const ctx = c.getContext('2d')!;
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const out = new Float32Array(c.width * c.height);
  for (let i = 0; i < out.length; i++) out[i] = (d[i * 4] * 0.3 + d[i * 4 + 1] * 0.59 + d[i * 4 + 2] * 0.11) / 255;
  return out;
}

/** Draw something so it wraps seamlessly around the tile edges. */
function wrapped(size: number, x: number, y: number, r: number, draw: (x: number, y: number) => void) {
  for (const ox of [-size, 0, size]) for (const oy of [-size, 0, size]) {
    const px = x + ox, py = y + oy;
    if (px + r < 0 || px - r > size || py + r < 0 || py - r > size) continue;
    draw(px, py);
  }
}

function noiseFill(ctx: CanvasRenderingContext2D, size: number, noise: Noise2, cells: number, color: (n: number, x: number, y: number) => [number, number, number]) {
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const n = noise.fbm((x / size) * cells, (y / size) * cells, 4, cells);
      const [r, g, b] = color(n, x, y);
      const i = (y * size + x) * 4;
      img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

const hsl = (h: number, s: number, l: number, a = 1) => `hsla(${h},${s}%,${l}%,${a})`;

// ───────────────────────────────────────────────────────────────── grass

export function grassTex(size: number): TexSet {
  return memo(`grass${size}`, () => {
    const noise = new Noise2(11);
    const rnd = mulberry(5);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    noiseFill(ctx, size, noise, 6, (n) => [58 + n * 26, 104 + n * 30, 38 + n * 14]);
    const hc = canvas(size);
    const hctx = hc.getContext('2d')!;
    hctx.fillStyle = '#404040';
    hctx.fillRect(0, 0, size, size);
    const blades = Math.round(size * size * 0.045);
    const L = size / 64;
    for (let i = 0; i < blades; i++) {
      const x = rnd() * size, y = rnd() * size;
      const len = L * (0.6 + rnd() * 1.2);
      const ang = -Math.PI / 2 + (rnd() - 0.5) * 1.4;
      const ex = Math.cos(ang) * len, ey = Math.sin(ang) * len;
      const light = 24 + rnd() * 24;
      const hue = 80 + rnd() * 24;
      const col = hsl(hue, 38 + rnd() * 22, light, 0.8);
      const w = Math.max(1, size / 512) * (0.6 + rnd() * 0.8);
      wrapped(size, x, y, len + 2, (px, py) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = w;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + ex, py + ey); ctx.stroke();
        hctx.strokeStyle = `rgba(255,255,255,${0.25 + rnd() * 0.5})`;
        hctx.lineWidth = w;
        hctx.beginPath(); hctx.moveTo(px, py); hctx.lineTo(px + ex, py + ey); hctx.stroke();
      });
    }
    // clover patches and the odd dandelion
    for (let i = 0; i < size / 18; i++) {
      const x = rnd() * size, y = rnd() * size;
      wrapped(size, x, y, 8, (px, py) => {
        for (let k = 0; k < 3; k++) {
          const a = (k / 3) * Math.PI * 2;
          ctx.fillStyle = hsl(100, 40, 34, 0.8);
          ctx.beginPath(); ctx.arc(px + Math.cos(a) * 2, py + Math.sin(a) * 2, 2.2 * (size / 512), 0, Math.PI * 2); ctx.fill();
        }
      });
    }
    for (let i = 0; i < size / 90; i++) {
      const x = rnd() * size, y = rnd() * size;
      ctx.fillStyle = hsl(50, 95, 60);
      ctx.beginPath(); ctx.arc(x, y, 1.6 * (size / 512), 0, Math.PI * 2); ctx.fill();
    }
    const normal = normalFromHeight(heightOf(hc), size, 2.2);
    return { map: tex(c), normal: tex(normal, false) };
  });
}

// ───────────────────────────────────────────────────────────────── dirt

export function dirtTex(size: number, sandy = false): TexSet {
  return memo(`dirt${size}${sandy}`, () => {
    const noise = new Noise2(23);
    const rnd = mulberry(17);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const base = sandy ? [222, 196, 140] : [150, 112, 80];
    noiseFill(ctx, size, noise, 8, (n) => [base[0] + n * 38, base[1] + n * 30, base[2] + n * 24]);
    const hc = canvas(size);
    const hctx = hc.getContext('2d')!;
    noiseFill(hctx, size, noise, 16, (n) => { const v = 100 + n * 60; return [v, v, v]; });
    const pebbles = Math.round(size * size * 0.0025);
    for (let i = 0; i < pebbles; i++) {
      const x = rnd() * size, y = rnd() * size;
      const r = (0.6 + rnd() * 2.2) * (size / 512);
      const l = 40 + rnd() * 30;
      wrapped(size, x, y, r + 1, (px, py) => {
        ctx.fillStyle = hsl(28, 20 + rnd() * 20, l);
        ctx.beginPath(); ctx.ellipse(px, py, r, r * 0.75, rnd() * 3, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,0.25)';
        ctx.beginPath(); ctx.arc(px - r * 0.3, py - r * 0.3, r * 0.35, 0, Math.PI * 2); ctx.fill();
        hctx.fillStyle = '#e0e0e0';
        hctx.beginPath(); hctx.arc(px, py, r, 0, Math.PI * 2); hctx.fill();
      });
    }
    const normal = normalFromHeight(heightOf(hc), size, 3);
    return { map: tex(c), normal: tex(normal, false) };
  });
}

// ───────────────────────────────────────────────────────────────── wood

export function woodTex(size: number, color: [number, number, number], planks = 4, gap = true, seed = 3): TexSet {
  return memo(`wood${size}${color.join()}${planks}${gap}${seed}`, () => {
    const noise = new Noise2(seed);
    const rnd = mulberry(seed * 7);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(size, size);
    const hgt = new Float32Array(size * size);
    const pw = size / planks;
    const tints = Array.from({ length: planks }, () => 0.88 + rnd() * 0.24);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const p = Math.floor(x / pw);
        const lx = (x % pw) / pw;
        const grain = noise.fbm((x / size) * 40 + p * 13.7, (y / size) * 3, 3, 40);
        const rings = Math.sin((grain * 9 + (x / size) * 30)) * 0.5 + 0.5;
        let v = (0.82 + rings * 0.18) * tints[p];
        let hv = 0.6 + rings * 0.1;
        if (gap && (lx < 0.035 || lx > 0.965)) { v *= 0.35; hv = 0; }
        const i = (y * size + x) * 4;
        img.data[i] = color[0] * v; img.data[i + 1] = color[1] * v; img.data[i + 2] = color[2] * v; img.data[i + 3] = 255;
        hgt[y * size + x] = hv;
      }
    }
    ctx.putImageData(img, 0, 0);
    // a few knots
    for (let k = 0; k < planks; k++) {
      const x = (k + 0.3 + rnd() * 0.4) * pw, y = rnd() * size;
      ctx.fillStyle = `rgba(60,35,15,0.35)`;
      ctx.beginPath(); ctx.ellipse(x, y, pw * 0.06, pw * 0.12, 0, 0, Math.PI * 2); ctx.fill();
    }
    return { map: tex(c), normal: tex(normalFromHeight(hgt, size, 4), false) };
  });
}

// ───────────────────────────────────────────────────────────────── house

export function sidingTex(size: number, color: [number, number, number]): TexSet {
  return memo(`siding${size}${color.join()}`, () => {
    const noise = new Noise2(41);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(size, size);
    const hgt = new Float32Array(size * size);
    const boards = 8;
    const bh = size / boards;
    for (let y = 0; y < size; y++) {
      const ly = (y % bh) / bh;
      for (let x = 0; x < size; x++) {
        const n = noise.fbm((x / size) * 30, (y / size) * 4, 3, 30) * 0.05;
        let v = 0.9 + ly * 0.12 + n;
        if (ly > 0.94) v *= 0.55;
        const i = (y * size + x) * 4;
        img.data[i] = Math.min(255, color[0] * v); img.data[i + 1] = Math.min(255, color[1] * v); img.data[i + 2] = Math.min(255, color[2] * v); img.data[i + 3] = 255;
        hgt[y * size + x] = ly > 0.94 ? 0 : ly;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { map: tex(c), normal: tex(normalFromHeight(hgt, size, 3), false) };
  });
}

export function shingleTex(size: number, color: [number, number, number]): TexSet {
  return memo(`shingle${size}${color.join()}`, () => {
    const rnd = mulberry(77);
    const noise = new Noise2(78);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const hc = canvas(size);
    const hctx = hc.getContext('2d')!;
    const rows = 10, cols = 8;
    const rh = size / rows, cw = size / cols;
    for (let r = 0; r < rows; r++) {
      for (let k = -1; k <= cols; k++) {
        const x = k * cw + (r % 2 ? cw / 2 : 0);
        const v = 0.75 + rnd() * 0.35;
        ctx.fillStyle = `rgb(${color[0] * v},${color[1] * v},${color[2] * v})`;
        ctx.fillRect(x + 1, r * rh, cw - 2, rh);
        const g = ctx.createLinearGradient(0, r * rh, 0, (r + 1) * rh);
        g.addColorStop(0, 'rgba(0,0,0,0.35)');
        g.addColorStop(0.25, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x + 1, r * rh, cw - 2, rh);
        const hg = hctx.createLinearGradient(0, r * rh, 0, (r + 1) * rh);
        hg.addColorStop(0, '#000');
        hg.addColorStop(1, '#fff');
        hctx.fillStyle = hg;
        hctx.fillRect(x + 1, r * rh, cw - 2, rh);
      }
    }
    // granule speckle
    const img = ctx.getImageData(0, 0, size, size);
    for (let i = 0; i < img.data.length; i += 4) {
      const n = (noise.value((i / 4) % size * 0.9, Math.floor(i / 4 / size) * 0.9) * 18);
      img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n;
    }
    ctx.putImageData(img, 0, 0);
    return { map: tex(c), normal: tex(normalFromHeight(heightOf(hc), size, 2), false) };
  });
}

/** Concrete, painted near-white; materials tint it to their colour. */
export function concreteTex(size: number, tint: [number, number, number] = [240, 240, 240]): TexSet {
  return memo(`concrete${size}${tint.join()}`, () => {
    const noise = new Noise2(91);
    const rnd = mulberry(92);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    noiseFill(ctx, size, noise, 10, (n) => [tint[0] + n * 22, tint[1] + n * 22, tint[2] + n * 22]);
    for (let i = 0; i < size * size * 0.004; i++) {
      ctx.fillStyle = rnd() > 0.5 ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.18)';
      ctx.fillRect(rnd() * size, rnd() * size, 1.4, 1.4);
    }
    const hc = canvas(size);
    noiseFill(hc.getContext('2d')!, size, noise, 32, (n) => { const v = 128 + n * 40; return [v, v, v]; });
    return { map: tex(c), normal: tex(normalFromHeight(heightOf(hc), size, 1.5), false) };
  });
}

export function poolTileTex(size: number): TexSet {
  return memo(`pooltile${size}`, () => {
    const rnd = mulberry(55);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#d9eef5';
    ctx.fillRect(0, 0, size, size);
    const n = 16, s = size / n;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      ctx.fillStyle = hsl(195 + rnd() * 14, 60 + rnd() * 20, 52 + rnd() * 16);
      ctx.fillRect(x * s + 1, y * s + 1, s - 2, s - 2);
    }
    return { map: tex(c) };
  });
}

export function barkTex(size: number): TexSet {
  return memo(`bark${size}`, () => {
    const noise = new Noise2(66);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(size, size);
    const hgt = new Float32Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const n = noise.fbm((x / size) * 10, (y / size) * 2.5, 4, 10);
      const ridge = 1 - Math.abs(Math.sin((x / size) * Math.PI * 9 + n * 4));
      const v = 0.45 + ridge * 0.4 + n * 0.15;
      const i = (y * size + x) * 4;
      img.data[i] = 92 * v + 20; img.data[i + 1] = 68 * v + 12; img.data[i + 2] = 48 * v + 8; img.data[i + 3] = 255;
      hgt[y * size + x] = ridge;
    }
    ctx.putImageData(img, 0, 0);
    return { map: tex(c), normal: tex(normalFromHeight(hgt, size, 5), false) };
  });
}

/** A 2×2 atlas of leaf sprites with alpha, for tree foliage cards. */
export function leafAtlas(size: number, hueBase = 100): Texture {
  return memo(`leaves${size}${hueBase}`, () => {
    const rnd = mulberry(31 + hueBase);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const half = size / 2;
    for (let q = 0; q < 4; q++) {
      const ox = (q % 2) * half, oy = Math.floor(q / 2) * half;
      // each cell is a small cluster of leaves
      for (let k = 0; k < 9; k++) {
        const cx = ox + half * (0.25 + rnd() * 0.5), cy = oy + half * (0.25 + rnd() * 0.5);
        const len = half * (0.18 + rnd() * 0.12), wid = len * 0.5;
        const ang = rnd() * Math.PI * 2;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(ang);
        const g = ctx.createLinearGradient(-wid, 0, wid, 0);
        const h = hueBase + rnd() * 25 - 8, l = 30 + rnd() * 20;
        g.addColorStop(0, hsl(h, 50, l - 6));
        g.addColorStop(0.5, hsl(h, 55, l + 8));
        g.addColorStop(1, hsl(h, 50, l - 4));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.moveTo(0, -len);
        ctx.quadraticCurveTo(wid, -len * 0.2, 0, len);
        ctx.quadraticCurveTo(-wid, -len * 0.2, 0, -len);
        ctx.fill();
        ctx.strokeStyle = hsl(h, 40, l + 18, 0.6);
        ctx.lineWidth = Math.max(1, size / 512);
        ctx.beginPath(); ctx.moveTo(0, -len * 0.9); ctx.lineTo(0, len * 0.9); ctx.stroke();
        ctx.restore();
      }
    }
    const t = tex(c, true, false);
    return t;
  });
}

/** Tileable dense-hedge leaves. */
export function hedgeTex(size: number): TexSet {
  return memo(`hedge${size}`, () => {
    const rnd = mulberry(303);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const hc = canvas(size);
    const hctx = hc.getContext('2d')!;
    ctx.fillStyle = '#2c5222';
    ctx.fillRect(0, 0, size, size);
    hctx.fillStyle = '#000';
    hctx.fillRect(0, 0, size, size);
    const n = Math.round(size * size * 0.016);
    for (let i = 0; i < n; i++) {
      const x = rnd() * size, y = rnd() * size;
      const len = (3 + rnd() * 5) * (size / 512);
      const ang = rnd() * Math.PI * 2;
      const l = 26 + rnd() * 30;
      wrapped(size, x, y, len * 2, (px, py) => {
        ctx.save(); ctx.translate(px, py); ctx.rotate(ang);
        ctx.fillStyle = hsl(92 + rnd() * 22, 42 + rnd() * 15, l);
        ctx.beginPath(); ctx.ellipse(0, 0, len, len * 0.55, 0, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
        hctx.fillStyle = `rgba(255,255,255,${0.2 + rnd() * 0.5})`;
        hctx.beginPath(); hctx.ellipse(px, py, len, len * 0.55, ang, 0, Math.PI * 2); hctx.fill();
      });
    }
    return { map: tex(c), normal: tex(normalFromHeight(heightOf(hc), size, 3), false) };
  });
}

export function waterNormal(size: number): Texture {
  return memo(`water${size}`, () => {
    const noise = new Noise2(808);
    const hgt = new Float32Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      hgt[y * size + x] = noise.fbm((x / size) * 6, (y / size) * 6, 4, 6) * 0.5 + 0.5;
    }
    return tex(normalFromHeight(hgt, size, 6), false);
  });
}

/** Subtle knit for uniforms. */
export function fabricNormal(size = 256): Texture {
  return memo(`fabric${size}`, () => {
    const hgt = new Float32Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const a = Math.sin((x + y) * 0.9) * Math.sin((x - y) * 0.9);
      hgt[y * size + x] = a * 0.5 + 0.5;
    }
    return tex(normalFromHeight(hgt, size, 0.6), false);
  });
}

export function leatherTex(size = 256): TexSet {
  return memo(`leather${size}`, () => {
    const noise = new Noise2(444);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    noiseFill(ctx, size, noise, 12, (n) => [128 + n * 30, 78 + n * 20, 40 + n * 12]);
    const hc = canvas(size);
    noiseFill(hc.getContext('2d')!, size, noise, 40, (n) => { const v = 128 + n * 90; return [v, v, v]; });
    return { map: tex(c), normal: tex(normalFromHeight(heightOf(hc), size, 1.2), false) };
  });
}

/** Soft cloud sprites with alpha: `count` clouds stacked vertically, each size × size/2. */
export function cloudAtlas(size = 256, count = 5): Texture {
  return memo(`clouds${size}x${count}`, () => {
    const c = canvas(size, (size / 2) * count);
    const ctx = c.getContext('2d')!;
    for (let k = 0; k < count; k++) {
      ctx.save();
      ctx.translate(0, (size / 2) * k);
      ctx.beginPath(); ctx.rect(0, 0, size, size / 2); ctx.clip();
      paintCloud(ctx, size, k + 1);
      ctx.restore();
    }
    return tex(c, true, false);
  });
}

function paintCloud(ctx: CanvasRenderingContext2D, size: number, seed: number) {
  const rnd = mulberry(seed * 97);
  for (let i = 0; i < 26; i++) {
    const x = size * (0.2 + rnd() * 0.6), y = size * 0.25 + (rnd() - 0.6) * size * 0.16;
    const r = size * (0.06 + rnd() * 0.1);
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, 'rgba(255,255,255,0.75)');
    g.addColorStop(0.6, 'rgba(250,252,255,0.35)');
    g.addColorStop(1, 'rgba(240,245,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  }
}

/** Paint arbitrary content onto a texture (labels, signs, jerseys). */
export function paintTex(w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void, repeat = false): CanvasTexture {
  const c = canvas(w, h);
  draw(c.getContext('2d')!);
  return tex(c, true, repeat);
}

// ───────────────────────────────────────────────────────────────── more surfaces

export function brickTex(size: number, color: [number, number, number] = [150, 70, 52]): TexSet {
  return memo(`brick${size}${color.join()}`, () => {
    const rnd = mulberry(612);
    const noise = new Noise2(613);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const hc = canvas(size);
    const hctx = hc.getContext('2d')!;
    ctx.fillStyle = '#b9b2a6';
    ctx.fillRect(0, 0, size, size);
    hctx.fillStyle = '#000';
    hctx.fillRect(0, 0, size, size);
    const rows = 16, cols = 4; // 8" × 2.67" bricks over a 32" tile
    const rh = size / rows, cw = size / cols, m = Math.max(1, size / 180);
    for (let r = 0; r < rows; r++) for (let k = -1; k <= cols; k++) {
      const x = k * cw + (r % 2 ? cw / 2 : 0);
      const v = 0.72 + rnd() * 0.4;
      ctx.fillStyle = `rgb(${color[0] * v},${color[1] * v},${color[2] * v})`;
      ctx.fillRect(x + m, r * rh + m, cw - 2 * m, rh - 2 * m);
      hctx.fillStyle = '#ddd';
      hctx.fillRect(x + m, r * rh + m, cw - 2 * m, rh - 2 * m);
    }
    const img = ctx.getImageData(0, 0, size, size);
    for (let i = 0; i < img.data.length; i += 4) {
      const p = i / 4, n = noise.fbm(((p % size) / size) * 24, (Math.floor(p / size) / size) * 24, 3, 24) * 22;
      img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n;
    }
    ctx.putImageData(img, 0, 0);
    return { map: tex(c), normal: tex(normalFromHeight(heightOf(hc), size, 2.5), false) };
  });
}

export function asphaltTex(size: number): TexSet {
  return memo(`asphalt${size}`, () => {
    const noise = new Noise2(707);
    const rnd = mulberry(708);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    noiseFill(ctx, size, noise, 8, (n) => { const v = 70 + n * 14; return [v, v, v + 3]; });
    for (let i = 0; i < size * size * 0.02; i++) {
      const v = 40 + rnd() * 90;
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(rnd() * size, rnd() * size, 1, 1);
    }
    const hc = canvas(size);
    noiseFill(hc.getContext('2d')!, size, noise, 48, (n) => { const v = 128 + n * 80; return [v, v, v]; });
    return { map: tex(c), normal: tex(normalFromHeight(heightOf(hc), size, 1.2), false) };
  });
}

/** Awning / umbrella / towel stripes. */
export function stripeTex(colors: string[], size = 256, vertical = true): Texture {
  return memo(`stripe${colors.join()}${size}${vertical}`, () => {
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const w = size / colors.length;
    colors.forEach((col, i) => {
      ctx.fillStyle = col;
      if (vertical) ctx.fillRect(i * w, 0, w + 1, size);
      else ctx.fillRect(0, i * w, size, w + 1);
    });
    // a little fabric weave
    const img = ctx.getImageData(0, 0, size, size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4, n = ((x + y) % 3 === 0 ? -6 : 0) + ((x * 7 + y * 13) % 11) - 5;
      img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n;
    }
    ctx.putImageData(img, 0, 0);
    return tex(c);
  });
}

/** How many window variants (different rooms/curtain widths) sit side by side in a window atlas. */
export const WINDOW_VARIANTS = 3;

/**
 * Windows as seen from outside: frame, mullions, glass with a dim room and
 * curtains. One atlas holds WINDOW_VARIANTS variants side by side, so a whole
 * street of windows shares one texture and one draw call.
 */
export function windowTex(kind: 'double' | 'slider' | 'picture' = 'double', curtain = '#e9d9b5'): Texture {
  return memo(`window${kind}${curtain}`, () => {
    const W = 256, H = kind === 'slider' ? 224 : 320;
    const c = canvas(W * WINDOW_VARIANTS, H);
    const ctx = c.getContext('2d')!;
    for (let v = 0; v < WINDOW_VARIANTS; v++) {
      ctx.save();
      ctx.translate(v * W, 0);
      ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
      paintWindow(ctx, kind, curtain, v, W, H);
      ctx.restore();
    }
    return tex(c, true, false);
  });
}

function paintWindow(ctx: CanvasRenderingContext2D, kind: 'double' | 'slider' | 'picture', curtain: string, seed: number, W: number, H: number) {
  const rnd = mulberry(seed * 31 + 7);
  // the room behind the glass
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#3a3f45');
  g.addColorStop(1, '#202428');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  // hints of furniture / a lamp
  ctx.fillStyle = 'rgba(255,220,160,0.18)';
  ctx.beginPath(); ctx.arc(W * (0.3 + rnd() * 0.4), H * 0.45, W * 0.14, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.fillRect(W * 0.1, H * 0.7, W * 0.8, H * 0.3);
  // curtains, gathered at the sides
  const cw = W * (0.18 + rnd() * 0.1);
  for (const side of [0, 1]) {
    const x0 = side ? W - cw : 0;
    for (let k = 0; k < 6; k++) {
      const f = ctx.createLinearGradient(x0 + (k * cw) / 6, 0, x0 + ((k + 1) * cw) / 6, 0);
      f.addColorStop(0, curtain);
      f.addColorStop(0.5, shade(curtain, 0.78));
      f.addColorStop(1, curtain);
      ctx.fillStyle = f;
      ctx.fillRect(x0 + (k * cw) / 6, 0, cw / 6 + 1, H);
    }
  }
  // sky reflection streak across the glass
  const r = ctx.createLinearGradient(0, 0, W, H);
  r.addColorStop(0, 'rgba(200,225,255,0.35)');
  r.addColorStop(0.45, 'rgba(200,225,255,0.08)');
  r.addColorStop(0.55, 'rgba(255,255,255,0.22)');
  r.addColorStop(0.7, 'rgba(200,225,255,0.05)');
  ctx.fillStyle = r;
  ctx.fillRect(0, 0, W, H);
  // sash frames and muntins
  ctx.strokeStyle = '#f3f1ea';
  ctx.lineWidth = 14;
  ctx.strokeRect(7, 7, W - 14, H - 14);
  ctx.lineWidth = 6;
  if (kind === 'double') {
    ctx.lineWidth = 12;
    ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    ctx.lineWidth = 5;
    for (const hy of [0, H / 2]) {
      ctx.beginPath(); ctx.moveTo(W / 2, hy); ctx.lineTo(W / 2, hy + H / 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, hy + H / 4); ctx.lineTo(W, hy + H / 4); ctx.stroke();
    }
  } else if (kind === 'slider') {
    ctx.lineWidth = 12;
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
    // door handle
    ctx.fillStyle = '#9aa0a6';
    ctx.fillRect(W / 2 - 22, H * 0.48, 6, 26);
  }
}

function shade(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = ((n >> 16) & 255) * k, g = ((n >> 8) & 255) * k, b = (n & 255) * k;
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

/** Mulch / flower-bed soil. */
export function mulchTex(size: number): TexSet {
  return memo(`mulch${size}`, () => {
    const rnd = mulberry(919);
    const c = canvas(size);
    const ctx = c.getContext('2d')!;
    const hc = canvas(size);
    const hctx = hc.getContext('2d')!;
    ctx.fillStyle = '#4a2e1c';
    ctx.fillRect(0, 0, size, size);
    hctx.fillStyle = '#000';
    hctx.fillRect(0, 0, size, size);
    for (let i = 0; i < size * size * 0.01; i++) {
      const x = rnd() * size, y = rnd() * size, len = (4 + rnd() * 9) * (size / 512), a = rnd() * Math.PI;
      const l = 18 + rnd() * 22;
      wrapped(size, x, y, len, (px, py) => {
        ctx.strokeStyle = hsl(22 + rnd() * 10, 45, l);
        ctx.lineWidth = (1.5 + rnd() * 2) * (size / 512);
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + Math.cos(a) * len, py + Math.sin(a) * len); ctx.stroke();
        hctx.strokeStyle = `rgba(255,255,255,${0.3 + rnd() * 0.5})`;
        hctx.lineWidth = ctx.lineWidth;
        hctx.beginPath(); hctx.moveTo(px, py); hctx.lineTo(px + Math.cos(a) * len, py + Math.sin(a) * len); hctx.stroke();
      });
    }
    return { map: tex(c), normal: tex(normalFromHeight(heightOf(hc), size, 3), false) };
  });
}
