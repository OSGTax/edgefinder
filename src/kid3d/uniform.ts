import { CanvasTexture, LinearMipmapLinearFilter, SRGBColorSpace } from 'three';
import type { Kid, Team } from '../data/types';

// Jerseys are painted per kid: the torso wrap (team script on the front,
// name and number on the back) fills the top 3/4 of the texture; the bottom
// strip holds the cap logo.

export const JERSEY_V0 = 0.25;
export const CAP_LOGO_UV = { u0: 0, u1: 0.25, v0: 0, v1: 0.25 };

const NUMBERS: Record<string, number> = {
  ines: 7, bo: 44, dez: 23, priya: 11, gus: 31, toby: 2, molly: 9, wren: 1, jun: 17,
  kai: 99, ezra: 12, leo: 8, darius: 32, anya: 21, ruby: 5, hank: 55, maya: 3, pepper: 14,
};
export const jerseyNumber = (k: Kid) => NUMBERS[k.id] ?? (k.id.charCodeAt(0) % 60) + 1;

export interface UniformColors { jersey: string; trim: string; script: string; pants: string; socks: string; sockStripe: string; cap: string; brim: string }

export function uniformColors(t: Team): UniformColors {
  const c = t.colors;
  return { jersey: c.primary, trim: c.secondary, script: c.secondary, pants: t.id === 'comets' ? '#dfe5ec' : c.accent, socks: c.secondary, sockStripe: c.primary, cap: c.primary, brim: c.primary };
}

export function paintJersey(kid: Kid, team: Team, size = 512): CanvasTexture {
  const col = uniformColors(team);
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const g = c.getContext('2d')!;
  const W = size, H = size;
  const torsoTop = 0, torsoBot = H * (1 - JERSEY_V0); // canvas y of v=1 and v=0.25
  const ty = (v: number) => torsoBot - (torsoBot - torsoTop) * v; // v in torso 0..1 → canvas y
  g.fillStyle = col.jersey;
  g.fillRect(0, 0, W, torsoBot);
  // subtle knit shading
  for (let i = 0; i < 2600; i++) {
    g.fillStyle = `rgba(${i % 2 ? 255 : 0},${i % 2 ? 255 : 0},${i % 2 ? 255 : 0},0.03)`;
    g.fillRect((i * 97) % W, (i * 57) % torsoBot, 2, 2);
  }
  // side piping
  g.fillStyle = col.trim;
  for (const u of [0.25, 0.75]) g.fillRect(u * W - 3, ty(0.5), 6, ty(0) - ty(0.5));
  // front placket + buttons
  g.fillStyle = 'rgba(0,0,0,0.18)';
  g.fillRect(W * 0.5 - 1.5, ty(0.92), 3, ty(0) - ty(0.92));
  g.fillStyle = '#f4f1e8';
  for (let k = 0; k < 5; k++) { g.beginPath(); g.arc(W * 0.5 + 5, ty(0.12 + k * 0.17), 3.2, 0, 7); g.fill(); }
  // team script across the chest, with an outline and a swash underline
  const script = team.name;
  g.save();
  g.translate(W * 0.5, ty(0.66));
  g.rotate(-0.08);
  g.font = `italic bold ${Math.round(size * 0.085)}px "Brush Script MT", "Trebuchet MS", cursive`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.lineWidth = size * 0.012;
  g.strokeStyle = shadeHex(col.jersey, 0.5);
  g.strokeText(script, 0, 0);
  g.fillStyle = col.script;
  g.fillText(script, 0, 0);
  g.strokeStyle = col.script;
  g.lineWidth = size * 0.008;
  g.beginPath(); g.moveTo(-W * 0.12, size * 0.045); g.quadraticCurveTo(0, size * 0.065, W * 0.14, size * 0.03); g.stroke();
  g.restore();
  // little number on the front (left chest, below the script)
  const num = String(jerseyNumber(kid));
  g.font = `bold ${Math.round(size * 0.05)}px "Arial Black", "Trebuchet MS", sans-serif`;
  g.textAlign = 'center';
  g.fillStyle = col.script;
  g.fillText(num, W * 0.6, ty(0.45));
  // back: name arched over a big block number (centred on u = 0 / 1, so draw twice)
  for (const cx of [0, W]) {
    g.save();
    g.translate(cx, 0);
    g.font = `bold ${Math.round(size * 0.05)}px "Arial Black", "Trebuchet MS", sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillStyle = col.script;
    const name = kid.last.toUpperCase();
    // arch the name
    const R = size * 0.9, arc = Math.min(0.5, name.length * 0.035);
    for (let i = 0; i < name.length; i++) {
      const a = -arc / 2 + (arc * (i + 0.5)) / name.length;
      g.save();
      g.translate(Math.sin(a) * R, ty(0.86) + R - Math.cos(a) * R);
      g.rotate(a);
      g.fillText(name[i], 0, 0);
      g.restore();
    }
    g.font = `bold ${Math.round(size * 0.2)}px "Arial Black", "Trebuchet MS", sans-serif`;
    g.lineWidth = size * 0.014;
    g.strokeStyle = shadeHex(col.jersey, 0.45);
    g.strokeText(num, 0, ty(0.5));
    g.fillStyle = '#f7f3ea';
    g.fillText(num, 0, ty(0.5));
    g.lineWidth = size * 0.006;
    g.strokeStyle = col.script;
    g.strokeText(num, 0, ty(0.5));
    g.restore();
  }
  // hem stripe
  g.fillStyle = col.trim;
  g.fillRect(0, ty(0.035), W, size * 0.012);
  // cap logo (bottom-left strip): the team initial on the cap's front panel colour
  const lx = W * 0.125, ly = H - (H * JERSEY_V0) / 2;
  g.fillStyle = col.cap;
  g.fillRect(0, H * (1 - JERSEY_V0), W * 0.25, H * JERSEY_V0);
  g.font = `bold ${Math.round(size * 0.17)}px "Georgia", serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.lineWidth = size * 0.012;
  g.strokeStyle = '#f7f3ea';
  g.strokeText(team.name[0], lx, ly);
  g.fillStyle = col.script;
  g.fillText(team.name[0], lx, ly);
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.minFilter = LinearMipmapLinearFilter;
  t.anisotropy = 4;
  return t;
}

function shadeHex(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.round(((n >> 16) & 255) * k), gg = Math.round(((n >> 8) & 255) * k), b = Math.round((n & 255) * k);
  return `rgb(${r},${gg},${b})`;
}
