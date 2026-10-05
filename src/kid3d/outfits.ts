import { CanvasTexture, LinearMipmapLinearFilter, SRGBColorSpace } from 'three';
import { JERSEY_V0 } from './uniform';

// Non-uniform outfits for the grown-ups who wander through the yard.

/** A loud Hawaiian shirt (in the jersey texture layout: torso wrap on top, a spare strip below). */
export function hawaiianShirt(base = '#1f8a8a', size = 512): CanvasTexture {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  g.fillStyle = base;
  g.fillRect(0, 0, size, size);
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const flower = (x: number, y: number, r: number, petal: string, mid: string) => {
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 + r;
      g.fillStyle = petal;
      g.beginPath();
      g.ellipse(x + Math.cos(a) * r * 0.55, y + Math.sin(a) * r * 0.55, r * 0.55, r * 0.32, a, 0, Math.PI * 2);
      g.fill();
    }
    g.fillStyle = mid;
    g.beginPath(); g.arc(x, y, r * 0.22, 0, Math.PI * 2); g.fill();
  };
  const leaf = (x: number, y: number, r: number, a: number) => {
    g.save(); g.translate(x, y); g.rotate(a);
    g.fillStyle = '#2f6b2a';
    g.beginPath(); g.moveTo(0, -r); g.quadraticCurveTo(r * 0.6, 0, 0, r); g.quadraticCurveTo(-r * 0.6, 0, 0, -r); g.fill();
    g.strokeStyle = '#4f9a3a'; g.lineWidth = 2; g.beginPath(); g.moveTo(0, -r * 0.9); g.lineTo(0, r * 0.9); g.stroke();
    g.restore();
  };
  for (let i = 0; i < 40; i++) leaf(rnd() * size, rnd() * size * (1 - JERSEY_V0), 18 + rnd() * 16, rnd() * 6);
  const petals = ['#ff6b8a', '#ffd23f', '#ffffff', '#ff922b'];
  for (let i = 0; i < 34; i++) flower(rnd() * size, rnd() * size * (1 - JERSEY_V0), 14 + rnd() * 12, petals[i % 4], '#ffe680');
  // buttons down the front
  g.fillStyle = '#f4ead2';
  for (let k = 0; k < 5; k++) { g.beginPath(); g.arc(size * 0.5 + 4, size * (1 - JERSEY_V0) * (0.15 + k * 0.17), 4, 0, 7); g.fill(); }
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.minFilter = LinearMipmapLinearFilter;
  t.anisotropy = 4;
  return t;
}

/** "KISS THE COOK" apron bib, painted for a quad. */
export function apronSign(size = 256): CanvasTexture {
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const g = c.getContext('2d')!;
  g.fillStyle = '#f6f1e6';
  g.fillRect(0, 0, size, size);
  g.fillStyle = '#c0392b';
  g.font = `900 ${Math.round(size * 0.16)}px "Trebuchet MS", sans-serif`;
  g.textAlign = 'center';
  g.fillText('GRILL', size / 2, size * 0.42);
  g.fillText('SERGEANT', size / 2, size * 0.62);
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  return t;
}
