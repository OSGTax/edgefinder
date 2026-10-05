import { drawPortrait } from '../art/kid';
import type { Kid, Team } from '../data/types';

/** A kid's portrait on its own canvas (for cards, scoreboards, bubbles). */
export function portraitCanvas(k: Kid, team: Team | null, w: number, h = w, bg: string | null = null): HTMLCanvasElement {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const c = document.createElement('canvas');
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  c.style.width = `${w}px`;
  c.style.height = `${h}px`;
  c.className = 'portrait';
  const ctx = c.getContext('2d')!;
  ctx.scale(dpr, dpr);
  if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h); }
  drawPortrait(ctx, k, team, w, h, 1.3);
  return c;
}
