import { CanvasTexture, SRGBColorSpace, LinearMipmapLinearFilter } from 'three';
import type { KidLook } from '../data/types';
import { HAIR, SKIN } from '../data/palette';

// Faces: the eyes and nose are real 3D, everything else (brows, mouth,
// cheeks, freckles, marker-drawn flair) is painted into a per-kid atlas with
// one cell per expression. The face patch spans ±70° around the head and
// from 60° below to 40° above the eye line, so features map like this:

export const FACE_PATCH = { phi: (70 * Math.PI) / 180, thetaLo: (-60 * Math.PI) / 180, thetaHi: (40 * Math.PI) / 180 };
export const EXPRESSIONS = ['neutral', 'happy', 'focus', 'surprised', 'sad', 'yell', 'smug', 'oops'] as const;
export type Expression = (typeof EXPRESSIONS)[number];
export const ATLAS_COLS = 4, ATLAS_ROWS = 2;

/** Texture coords (0..1, y up) of a point on the head at yaw/pitch angles (radians). */
export function faceUV(phi: number, theta: number): [number, number] {
  return [(phi + FACE_PATCH.phi) / (2 * FACE_PATCH.phi), (theta - FACE_PATCH.thetaLo) / (FACE_PATCH.thetaHi - FACE_PATCH.thetaLo)];
}

const deg = (d: number) => (d * Math.PI) / 180;

interface FaceSpec {
  look: KidLook;
  /** eyes' angular half-spacing and size (deg) so brows sit above them */
  eyePhi: number; eyeTheta: number; eyeSize: number;
}

export function paintFaceAtlas(spec: FaceSpec, cell = 256): CanvasTexture {
  const c = document.createElement('canvas');
  c.width = cell * ATLAS_COLS;
  c.height = cell * ATLAS_ROWS;
  const ctx = c.getContext('2d')!;
  EXPRESSIONS.forEach((e, i) => {
    const ox = (i % ATLAS_COLS) * cell, oy = Math.floor(i / ATLAS_COLS) * cell;
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, cell, cell);
    ctx.clip();
    ctx.translate(ox, oy);
    paintFace(ctx, cell, spec, e);
    ctx.restore();
  });
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.minFilter = LinearMipmapLinearFilter;
  t.anisotropy = 4;
  return t;
}

function paintFace(ctx: CanvasRenderingContext2D, S: number, spec: FaceSpec, e: Expression) {
  const { look } = spec;
  // helper: angles → canvas px (canvas y grows downward)
  const P = (phiDeg: number, thetaDeg: number): [number, number] => {
    const [u, v] = faceUV(deg(phiDeg), deg(thetaDeg));
    return [u * S, (1 - v) * S];
  };
  const skin = SKIN[look.skin] ?? SKIN[1];
  const ink = '#2a1a12';
  const lip = shade(skin, 0.72, 1.05);
  const hairCol = HAIR[look.hairColor] ?? HAIR[0];
  const brow = shade(hairCol, look.hairColor >= 4 ? 0.75 : 1, 1);

  // cheeks
  const blush = e === 'happy' || e === 'yell' || e === 'oops' ? 0.32 : 0.18;
  for (const sx of [-1, 1]) {
    const [x, y] = P(sx * 33, -14);
    const g = ctx.createRadialGradient(x, y, 0, x, y, S * 0.09);
    g.addColorStop(0, `rgba(240,110,110,${blush})`);
    g.addColorStop(1, 'rgba(240,110,110,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, S * 0.09, 0, Math.PI * 2); ctx.fill();
  }
  if (look.freckles) {
    ctx.fillStyle = shade(skin, 0.62, 1);
    const pts = [[-30, -6], [-24, -10], [-34, -11], [-27, -15], [-20, -6], [30, -6], [24, -10], [34, -11], [27, -15], [20, -6], [-6, -4], [6, -4]];
    for (const [a, b] of pts) { const [x, y] = P(a, b); ctx.beginPath(); ctx.arc(x, y, S * 0.008, 0, 7); ctx.fill(); }
  }
  if (look.face === 'zinc') {
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    const [x, y] = P(0, -4);
    ctx.beginPath(); ctx.ellipse(x, y, S * 0.05, S * 0.06, 0, 0, 7); ctx.fill();
    for (const sx of [-1, 1]) { const [cx, cy] = P(sx * 30, -10); ctx.beginPath(); ctx.ellipse(cx, cy, S * 0.04, S * 0.018, 0, 0, 7); ctx.fill(); }
  }

  // brows: angle and height by expression
  const browShape: Record<Expression, { lift: number; tilt: number; arch: number }> = {
    neutral: { lift: 0, tilt: 0, arch: 0.4 },
    happy: { lift: 2, tilt: -4, arch: 0.7 },
    focus: { lift: -3, tilt: 14, arch: 0.1 },
    surprised: { lift: 7, tilt: -6, arch: 1 },
    sad: { lift: 2, tilt: -16, arch: 0.2 },
    yell: { lift: -2, tilt: 18, arch: 0 },
    smug: { lift: 1, tilt: 6, arch: 0.5 },
    oops: { lift: 5, tilt: -12, arch: 0.6 },
  };
  const bs = browShape[e];
  const by = spec.eyeTheta + spec.eyeSize + 4 + bs.lift;
  ctx.strokeStyle = brow;
  ctx.lineCap = 'round';
  ctx.lineWidth = S * (look.face === 'unibrow' ? 0.04 : 0.032);
  for (const sx of [-1, 1]) {
    // inner end tilts down for angry/focused (positive tilt), up for sad
    const inner = P(sx * (spec.eyePhi - spec.eyeSize * 0.9), by - bs.tilt * 0.35);
    const outer = P(sx * (spec.eyePhi + spec.eyeSize * 1.1), by + bs.tilt * 0.2);
    const mid = P(sx * spec.eyePhi, by + 2 + bs.arch * 3);
    ctx.beginPath(); ctx.moveTo(inner[0], inner[1]); ctx.quadraticCurveTo(mid[0], mid[1], outer[0], outer[1]); ctx.stroke();
    // smug: one brow raised
    if (e === 'smug' && sx === 1) {
      const o2 = P(sx * (spec.eyePhi + spec.eyeSize * 1.1), by + 5);
      ctx.beginPath(); ctx.moveTo(inner[0], inner[1] - S * 0.02); ctx.quadraticCurveTo(mid[0], mid[1] - S * 0.04, o2[0], o2[1]); ctx.stroke();
    }
  }
  if (look.face === 'unibrow') {
    const a = P(-(spec.eyePhi - spec.eyeSize), by - bs.tilt * 0.35), b = P(spec.eyePhi - spec.eyeSize, by - bs.tilt * 0.35);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }

  // marker-drawn flair (3D mustaches are added separately for walrus/handlebar)
  if (look.face === 'mustache' || look.face === 'goatee') {
    ctx.fillStyle = brow;
    if (look.face === 'mustache') {
      for (const sx of [-1, 1]) {
        const [x, y] = P(sx * 9, -19);
        ctx.beginPath(); ctx.ellipse(x, y, S * 0.06, S * 0.022, sx * 0.25, 0, 7); ctx.fill();
      }
    } else {
      const [x, y] = P(0, -44);
      ctx.beginPath(); ctx.moveTo(x - S * 0.05, y - S * 0.02); ctx.quadraticCurveTo(x, y + S * 0.08, x + S * 0.05, y - S * 0.02); ctx.closePath(); ctx.fill();
      for (const sx of [-1, 1]) { const [mx, my] = P(sx * 12, -20); ctx.beginPath(); ctx.ellipse(mx, my, S * 0.05, S * 0.014, sx * 0.3, 0, 7); ctx.fill(); }
    }
  }

  // mouth
  const [mx, my] = P(0, -29);
  const w = S * 0.11;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const style = look.mouth;
  /** an open mouth: corners at my + corner, top lip curve through my + top, bottom through my + bottom */
  const openMouth = (wid: number, top: number, bottom: number, corner: number) => {
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(mx - wid, my + corner);
      ctx.quadraticCurveTo(mx, my + top * 2 - corner, mx + wid, my + corner);
      ctx.quadraticCurveTo(mx, my + bottom * 2 - corner, mx - wid, my + corner);
      ctx.closePath();
    };
    ctx.fillStyle = '#5a1f1c';
    path();
    ctx.fill();
    ctx.save();
    ctx.clip();
    // upper teeth along the top lip, tongue at the bottom
    const teethY = my + top;
    ctx.fillStyle = '#fbf7ef';
    const th = S * 0.032;
    if (style !== 'gap') ctx.fillRect(mx - wid, teethY - S * 0.05, wid * 2, S * 0.05 + th);
    else { ctx.fillRect(mx - wid, teethY - S * 0.05, wid * 0.85, S * 0.05 + th); ctx.fillRect(mx + wid * 0.15, teethY - S * 0.05, wid * 0.85, S * 0.05 + th); }
    if (style === 'braces') {
      ctx.fillStyle = '#9aa3ad';
      ctx.fillRect(mx - wid, teethY + th * 0.4, wid * 2, th * 0.25);
      for (let i = -3; i <= 3; i++) ctx.fillRect(mx + i * wid * 0.28 - S * 0.006, teethY + th * 0.2, S * 0.012, th * 0.6);
    }
    ctx.fillStyle = '#e06a74';
    ctx.beginPath(); ctx.ellipse(mx, my + bottom + S * 0.01, wid * 0.6, Math.max(S * 0.025, (bottom - top) * 0.45), 0, 0, 7); ctx.fill();
    ctx.restore();
    ctx.strokeStyle = ink;
    ctx.lineWidth = S * 0.013;
    path();
    ctx.stroke();
  };
  const line = (pts: [number, number][], width = 0.018) => {
    ctx.strokeStyle = ink;
    ctx.lineWidth = S * width;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    if (pts.length === 3) ctx.quadraticCurveTo(pts[1][0], pts[1][1], pts[2][0], pts[2][1]);
    else for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
    ctx.stroke();
  };
  const smile = (curve: number, wid = w) => line([[mx - wid, my - curve * 0.3], [mx, my + curve], [mx + wid, my - curve * 0.3]]);
  switch (e) {
    case 'neutral':
      if (style === 'open' || style === 'gap' || style === 'braces') openMouth(w * 0.8, -S * 0.005, S * 0.06, -S * 0.025);
      else if (style === 'tongue') {
        smile(S * 0.04);
        ctx.fillStyle = '#e06a74';
        ctx.beginPath(); ctx.ellipse(mx + w * 0.35, my + S * 0.03, S * 0.028, S * 0.035, 0.3, 0, 7); ctx.fill();
        ctx.strokeStyle = ink; ctx.lineWidth = S * 0.01; ctx.stroke();
      } else if (style === 'smirk') line([[mx - w * 0.8, my + S * 0.01], [mx + w * 0.2, my + S * 0.025], [mx + w * 0.9, my - S * 0.03]]);
      else if (style === 'frown') smile(-S * 0.03);
      else if (style === 'whistle') { ctx.fillStyle = '#5a1f1c'; ctx.beginPath(); ctx.ellipse(mx, my, S * 0.022, S * 0.026, 0, 0, 7); ctx.fill(); line([[mx - S * 0.03, my - S * 0.035], [mx, my - S * 0.045], [mx + S * 0.03, my - S * 0.035]], 0.012); }
      else smile(S * 0.045);
      break;
    case 'happy': openMouth(w * 1.05, -S * 0.012, S * 0.1, -S * 0.04); break;
    case 'focus':
      line([[mx - w * 0.7, my], [mx, my + S * 0.006], [mx + w * 0.7, my]]);
      ctx.fillStyle = '#fbf7ef';
      if (style === 'braces' || style === 'gap') ctx.fillRect(mx - w * 0.4, my - S * 0.004, w * 0.8, S * 0.016);
      break;
    case 'surprised':
      ctx.fillStyle = '#5a1f1c';
      ctx.beginPath(); ctx.ellipse(mx, my + S * 0.015, S * 0.045, S * 0.062, 0, 0, 7); ctx.fill();
      ctx.strokeStyle = ink; ctx.lineWidth = S * 0.012; ctx.stroke();
      break;
    case 'sad': smile(-S * 0.05, w * 0.8); break;
    case 'yell': openMouth(w * 0.78, -S * 0.06, S * 0.11, -S * 0.005); break;
    case 'smug': line([[mx - w * 0.9, my + S * 0.005], [mx + w * 0.1, my + S * 0.03], [mx + w * 1.0, my - S * 0.045]], 0.02); break;
    case 'oops':
      line([[mx - w * 0.8, my], [mx - w * 0.4, my - S * 0.02], [mx, my], [mx + w * 0.4, my - S * 0.02], [mx + w * 0.8, my]], 0.016);
      // sweat drop
      ctx.fillStyle = 'rgba(140,200,255,0.9)';
      { const [sx, sy] = P(48, 20); ctx.beginPath(); ctx.moveTo(sx, sy - S * 0.04); ctx.quadraticCurveTo(sx + S * 0.03, sy + S * 0.01, sx, sy + S * 0.02); ctx.quadraticCurveTo(sx - S * 0.03, sy + S * 0.01, sx, sy - S * 0.04); ctx.fill(); }
      break;
  }
  void lip;
}

/** Multiply a hex colour's lightness (k) and saturation-ish (sat). */
export function shade(hex: string, k: number, sat = 1): string {
  const n = parseInt(hex.slice(1), 16);
  let r = ((n >> 16) & 255) * k, g = ((n >> 8) & 255) * k, b = (n & 255) * k;
  const avg = (r + g + b) / 3;
  r = avg + (r - avg) * sat; g = avg + (g - avg) * sat; b = avg + (b - avg) * sat;
  const cl = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  return `#${((cl(r) << 16) | (cl(g) << 8) | cl(b)).toString(16).padStart(6, '0')}`;
}
