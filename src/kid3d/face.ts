import { CanvasTexture, SRGBColorSpace, LinearMipmapLinearFilter } from 'three';
import type { KidLook } from '../data/types';
import { HAIR, SKIN } from '../data/palette';
import type { BrowStyle, FaceRecipe } from './face-recipes';

// Faces: the eyes and nose are real 3D, everything else (brows, mouth,
// cheeks, freckles, lid creases, marker-drawn flair) is painted into a per-kid
// atlas with one cell per expression. The face patch spans ±70° around the head
// and from 60° below to 40° above the head's equator.
//
// Painting happens in "face degrees": x = yaw (phi, + toward the kid's left =
// the viewer's right), y = pitch (theta, + up). One degree is the same length
// both ways on the head, so circles stay circles once the patch is wrapped on.

export const FACE_PATCH = { phi: (70 * Math.PI) / 180, thetaLo: (-60 * Math.PI) / 180, thetaHi: (40 * Math.PI) / 180 };
export const EXPRESSIONS = ['neutral', 'happy', 'focus', 'surprised', 'sad', 'yell', 'smug', 'oops'] as const;
export type Expression = (typeof EXPRESSIONS)[number];
export const ATLAS_COLS = 4, ATLAS_ROWS = 2;

/** Texture coords (0..1, y up) of a point on the head at yaw/pitch angles (radians). */
export function faceUV(phi: number, theta: number): [number, number] {
  return [(phi + FACE_PATCH.phi) / (2 * FACE_PATCH.phi), (theta - FACE_PATCH.thetaLo) / (FACE_PATCH.thetaHi - FACE_PATCH.thetaLo)];
}

/** Where the 3D features sit, in face degrees, so the painted ones line up with them. */
export interface FaceSpec {
  look: KidLook;
  recipe: FaceRecipe;
  /** eye centre (yaw of the kid's left eye, pitch) and the visible half-width/height */
  eyePhi: number; eyeTheta: number; eyeW: number; eyeH: number;
  noseTheta: number; mouthTheta: number;
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
    // face degrees → pixels: phi −70..70 across, theta 40..−60 down
    const deg2px = (FACE_PATCH.phi * 2 * 180) / Math.PI;
    const span = ((FACE_PATCH.thetaHi - FACE_PATCH.thetaLo) * 180) / Math.PI;
    const top = (FACE_PATCH.thetaHi * 180) / Math.PI;
    ctx.setTransform(cell / deg2px, 0, 0, -cell / span, ox + cell / 2, oy + (cell * top) / span);
    paintFace(ctx, spec, e);
    ctx.restore();
  });
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.minFilter = LinearMipmapLinearFilter;
  t.anisotropy = 4;
  return t;
}

type Ctx = CanvasRenderingContext2D;
type Pt = [number, number];

const INK = '#3a2117';
const MOUTH_IN = '#6a2622';
const TONGUE = '#e46f78';
const TEETH = '#fffaf1';

/** Per-expression brow pose: lift (deg), tilt (+ = inner ends down, cross), extra arch. */
const BROW_POSE: Record<Expression, { lift: number; tilt: number; arch: number }> = {
  neutral: { lift: 0, tilt: 0, arch: 0 },
  happy: { lift: 1.8, tilt: -0.6, arch: 1.0 },
  focus: { lift: -1.3, tilt: 3.0, arch: -0.4 },
  surprised: { lift: 4.6, tilt: -0.8, arch: 1.5 },
  sad: { lift: 1.2, tilt: -4.6, arch: -0.3 },
  yell: { lift: -1.2, tilt: 4.2, arch: -0.5 },
  smug: { lift: 0.3, tilt: 0.9, arch: 0.2 },
  oops: { lift: 2.6, tilt: -3.4, arch: 0.4 },
};

/** Brow designs: arch height, built-in tilt, width at the inner/outer end (deg), length, extra touches. */
const BROWS: Record<BrowStyle, { arch: number; tilt: number; w0: number; w1: number; len: number; lift: number; rough?: boolean; tufts?: boolean }> = {
  straight: { arch: 0.5, tilt: 0, w0: 1.9, w1: 1.4, len: 1, lift: 0 },
  arched: { arch: 2.4, tilt: -0.4, w0: 1.35, w1: 0.75, len: 1.05, lift: 0.6 },
  bushy: { arch: 1.0, tilt: 0.2, w0: 2.4, w1: 2.0, len: 1.05, lift: 0, rough: true },
  thin: { arch: 1.5, tilt: -0.2, w0: 1.1, w1: 0.7, len: 0.95, lift: 1.0 },
  angled: { arch: 0.7, tilt: 1.6, w0: 2.0, w1: 1.2, len: 1, lift: 0 },
  worried: { arch: 0.6, tilt: -1.8, w0: 1.8, w1: 1.4, len: 0.95, lift: 0.4 },
  tufty: { arch: 1.0, tilt: 0, w0: 2.0, w1: 1.5, len: 0.8, lift: 0.4, tufts: true },
  soft: { arch: 1.3, tilt: -0.2, w0: 1.9, w1: 1.5, len: 0.82, lift: 0.5 },
  flat: { arch: 0.1, tilt: 0.3, w0: 2.2, w1: 1.9, len: 1, lift: -0.4 },
};

function paintFace(ctx: Ctx, spec: FaceSpec, e: Expression) {
  const { look, recipe: r } = spec;
  const skin = SKIN[look.skin] ?? SKIN[1];
  const hairCol = HAIR[look.hairColor] ?? HAIR[0];
  // brows a little darker than the hair (blond and white hair still need to read)
  const browCol = shade(hairCol, look.hairColor >= 4 ? 0.62 : 0.88, 1.05);
  const crease = shade(skin, look.skin >= 4 ? 0.7 : 0.66, 1.15);
  const ex = spec.eyePhi, ey = spec.eyeTheta, ew = spec.eyeW, eh = spec.eyeH;
  const my = spec.mouthTheta;
  const W = 12 * r.mouthWidth;
  const rnd = mulberry(hashStr(look.hair + look.skin + r.brow + r.mouth));

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // ── cheeks: soft rosy patches under the outer half of each eye
  const blushK = (e === 'happy' || e === 'yell' || e === 'oops' ? 1.5 : 1) * (r.mouth === 'grin' && e === 'neutral' ? 1.25 : 1);
  const blushA = Math.min(0.6, (0.12 + 0.3 * r.cheeks) * blushK);
  for (const sx of [-1, 1]) {
    const cx = sx * (ex + ew * 0.55), cy = ey - eh - 6.5 + (e === 'happy' || r.mouth === 'grin' ? 1 : 0);
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, 7.5);
    g.addColorStop(0, `rgba(238,108,104,${blushA})`);
    g.addColorStop(0.55, `rgba(238,108,104,${blushA * 0.55})`);
    g.addColorStop(1, 'rgba(238,108,104,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx, cy, 7.5, 0, Math.PI * 2); ctx.fill();
  }

  // ── freckles across the nose and cheeks (fixed per kid so every expression matches)
  if (r.freckles) {
    ctx.fillStyle = shade(skin, 0.7, 1.25);
    const n = r.freckles === 2 ? 9 : 5;
    const f = mulberry(hashStr(look.hair + 'freckles' + look.skin));
    for (const sx of [-1, 1]) for (let i = 0; i < n; i++) {
      const a = f() * Math.PI * 2, d = Math.sqrt(f()) * 5;
      const fx = sx * (ex * 0.72 + Math.cos(a) * d * 1.2), fy = ey - eh - 4 + Math.sin(a) * d * 0.6;
      ctx.beginPath(); ctx.arc(fx, fy, 0.45 + f() * 0.3, 0, Math.PI * 2); ctx.fill();
    }
    for (let i = 0; i < (r.freckles === 2 ? 3 : 1); i++) { ctx.beginPath(); ctx.arc((i - (r.freckles === 2 ? 1 : 0)) * 1.6, spec.noseTheta + 2.5 + (i % 2) * 0.8, 0.45, 0, Math.PI * 2); ctx.fill(); }
  }
  if (r.mole) {
    ctx.fillStyle = shade(skin, 0.45, 1.1);
    ctx.beginPath(); ctx.arc(W + 2.5, my + 3.5, 0.65, 0, Math.PI * 2); ctx.fill();
  }

  // ── zinc (sunscreen) stripes
  if (look.face === 'zinc') {
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.beginPath(); ctx.ellipse(0, spec.noseTheta, 4, 4.5, 0, 0, Math.PI * 2); ctx.fill();
    for (const sx of [-1, 1]) { ctx.beginPath(); ctx.ellipse(sx * (ex + 2), ey - eh - 5, 4.5, 1.6, 0, 0, Math.PI * 2); ctx.fill(); }
  }

  // ── lid creases: a soft line just above each eye, and lashes for some kids
  for (const sx of [-1, 1]) {
    ctx.strokeStyle = crease;
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 0.75;
    ctx.beginPath();
    ctx.ellipse(sx * ex, ey + 0.5, ew * 1.12, eh * 1.12, 0, Math.PI * 0.25, Math.PI * 0.75);
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (r.lashes) {
      // little lashes at the outer corner (the side away from the nose)
      ctx.strokeStyle = INK;
      const count = r.lashes === 2 ? 3 : 1;
      for (let k = 0; k < count; k++) {
        const a = (0.1 + k * 0.12) * Math.PI;           // angle on the eye outline from the outer corner up
        const ax = sx * (ex + Math.cos(a) * ew * 0.98), ay = ey + Math.sin(a) * eh * 0.98;
        const len = r.lashes === 2 ? 1.6 - k * 0.3 : 1.6;
        ctx.lineWidth = 0.65;
        ctx.beginPath(); ctx.moveTo(ax, ay);
        ctx.quadraticCurveTo(ax + sx * len * 0.8, ay + len * 0.1, ax + sx * len * 1.0, ay + len * 0.75);
        ctx.stroke();
      }
    }
  }

  // ── brows
  const bp = BROW_POSE[e], bs = BROWS[r.brow];
  ctx.fillStyle = browCol;
  for (const sx of [-1, 1]) {
    let lift = bp.lift + bs.lift, tilt = bp.tilt + bs.tilt, arch = Math.max(-0.5, bs.arch + bp.arch);
    if (e === 'smug') { if (sx === 1) { lift += 2.4; arch += 0.8; tilt -= 1.5; } else { lift -= 0.6; } }
    const base = ey + eh + 3.6 + lift;
    const half = ew * 1.05 * bs.len;
    const ix = sx * (ex - half * 0.92), ox = sx * (ex + half * 1.08);
    const iy = base - tilt * 0.65, oy = base + tilt * 0.35 - (r.brow === 'arched' ? 0.6 : 0);
    const mid: Pt = [sx * (ex + half * 0.05), base + arch * 2];
    const th = r.browThick * 1.35;
    taper(ctx, [ix, iy], mid, [ox, oy], bs.w0 * th, bs.w1 * th);
    if (bs.rough) {
      // a few stray hairs so bushy brows look bushy
      ctx.strokeStyle = browCol; ctx.lineWidth = 0.55;
      for (let k = 0; k < 5; k++) {
        const t = 0.15 + k * 0.17, x = ix + (ox - ix) * t, y = iy + (oy - iy) * t + arch * 4 * t * (1 - t);
        const up = k % 2 ? 1 : -1;
        ctx.lineWidth = 0.7 + rnd() * 0.3;
        ctx.beginPath(); ctx.moveTo(x - sx * 0.6, y + up * bs.w0 * th * 0.3); ctx.lineTo(x + sx * 1.4, y + up * (bs.w0 * th * 0.5 + 0.4)); ctx.stroke();
      }
    }
    if (bs.tufts) {
      ctx.strokeStyle = browCol; ctx.lineWidth = 0.7;
      const x = ix + sx * 0.4, y = iy + bs.w0 * th * 0.4;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - sx * 0.6, y + 1.3); ctx.stroke();
    }
  }
  if (look.face === 'unibrow') {
    const base = ey + eh + 3.6 + bp.lift;
    taper(ctx, [-(ex - ew), base - bp.tilt * 0.65], [0, base - bp.tilt * 0.65 - 0.4], [ex - ew, base - bp.tilt * 0.65], 1.6, 1.6);
  }

  // ── marker-drawn facial hair (3D mustaches are added separately for walrus/handlebar)
  if (look.face === 'mustache' || look.face === 'goatee') {
    ctx.fillStyle = browCol;
    const ny = spec.noseTheta - 5.2;
    for (const sx of [-1, 1]) {
      ctx.beginPath(); ctx.ellipse(sx * 3.4, ny, 4.2, 1.7, sx * -0.22, 0, Math.PI * 2); ctx.fill();
    }
    if (look.face === 'goatee') {
      ctx.beginPath(); ctx.moveTo(-3, my - 5); ctx.quadraticCurveTo(0, my - 10.5, 3, my - 5); ctx.quadraticCurveTo(0, my - 6, -3, my - 5); ctx.fill();
    }
  }

  paintMouth(ctx, spec, e, W, skin);
}

function paintMouth(ctx: Ctx, spec: FaceSpec, e: Expression, W: number, skin: string) {
  const r = spec.recipe;
  const my = spec.mouthTheta;
  const lipCol = mixHex(skin, '#d9696f', 0.38);
  ctx.save();
  ctx.translate(0, my);
  // the head curves away below the nose, so mouths are painted a size up to read at portrait size
  ctx.scale(1.15, 1.25);

  /** closed smile: corners at ±w, the middle `curve` below them; asymmetric corner lifts. */
  const smile = (w: number, curve: number, o: { tucks?: boolean | 'L' | 'R'; lip?: number; liftL?: number; liftR?: number; width?: number } = {}) => {
    const yl = curve * 0.25 + (o.liftL ?? 0), yr = curve * 0.25 + (o.liftR ?? 0);
    if (o.lip) {
      const g = ctx.createRadialGradient(0, -curve - 1.9, 0, 0, -curve - 1.9, w * 0.5);
      g.addColorStop(0, rgba(lipCol, o.lip));
      g.addColorStop(1, rgba(lipCol, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.ellipse(0, -curve - 1.9, w * 0.5, 1.5, 0, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = INK;
    ctx.lineWidth = (o.width ?? 1.25) * 1.2;
    ctx.beginPath();
    ctx.moveTo(-w, yl);
    ctx.bezierCurveTo(-w * 0.45, -curve * 1.05 + yl * 0.3, w * 0.45, -curve * 1.05 + yr * 0.3, w, yr);
    ctx.stroke();
    if (o.tucks) {
      ctx.lineWidth = 0.85;
      for (const sx of [-1, 1]) {
        if (o.tucks === 'L' && sx === 1) continue;
        if (o.tucks === 'R' && sx === -1) continue;
        const y = sx < 0 ? yl : yr;
        ctx.beginPath();
        ctx.moveTo(sx * (w - 0.3), y + 1.25);
        ctx.quadraticCurveTo(sx * (w + 0.9), y + 0.4, sx * (w + 0.35), y - 0.9);
        ctx.stroke();
      }
    }
  };

  /** open mouth: corners at ±w lifted by `lift`; the top edge bows up by `top`, the bottom drops `depth`. */
  const open = (w: number, top: number, depth: number, lift: number, o: { teeth?: number; tongue?: number; tucks?: boolean } = {}) => {
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(-w, lift);
      ctx.bezierCurveTo(-w * 0.5, lift + top * 1.33, w * 0.5, lift + top * 1.33, w, lift);
      ctx.bezierCurveTo(w * 0.62, lift - depth * 1.33, -w * 0.62, lift - depth * 1.33, -w, lift);
      ctx.closePath();
    };
    ctx.fillStyle = MOUTH_IN;
    path(); ctx.fill();
    ctx.save();
    path(); ctx.clip();
    const tH = (o.teeth ?? 2.1) * Math.max(1, depth / 6);
    if (tH > 0) {
      const ty = lift + top;
      ctx.fillStyle = TEETH;
      ctx.fillRect(-w, ty - tH, w * 2, tH + 4);
      if (r.gapTooth) { ctx.fillStyle = MOUTH_IN; ctx.fillRect(-0.45, ty - tH, 0.9, tH + 4); }
      if (r.braces) {
        ctx.fillStyle = '#9aa6b2';
        ctx.fillRect(-w, ty - tH * 0.55, w * 2, 0.45);
        for (let i = -3; i <= 3; i++) ctx.fillRect(i * w * 0.27 - 0.4, ty - tH * 0.7, 0.8, 0.8);
      }
    }
    if ((o.tongue ?? 1) > 0) {
      ctx.fillStyle = TONGUE;
      ctx.beginPath(); ctx.ellipse(0, lift - depth - 0.4, w * 0.58, Math.max(1.4, depth * 0.5) * (o.tongue ?? 1), 0, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.25;
    path(); ctx.stroke();
    if (o.tucks) {
      ctx.lineWidth = 0.8;
      for (const sx of [-1, 1]) { ctx.beginPath(); ctx.moveTo(sx * (w - 0.2), lift + 1.2); ctx.quadraticCurveTo(sx * (w + 0.9), lift + 0.3, sx * (w + 0.3), lift - 0.9); ctx.stroke(); }
    }
  };

  const dimples = (w: number) => {
    ctx.strokeStyle = shade(skin, 0.68, 1.2); ctx.lineWidth = 0.7;
    for (const sx of [-1, 1]) { ctx.beginPath(); ctx.arc(sx * (w + 2.0), 0.0, 1.1, sx > 0 ? Math.PI * 0.6 : -Math.PI * 0.4, sx > 0 ? Math.PI * 1.4 : Math.PI * 0.4); ctx.stroke(); }
  };

  const rest = () => {
    switch (r.mouth) {
      case 'smile': smile(W * 0.7, 1.9, { tucks: true, lip: 0.45 }); break;
      case 'grin': smile(W * 0.82, 2.9, { tucks: true, lip: 0.35, width: 1.35 }); break;
      case 'toothy': open(W * 0.72, -0.6, 3.6, 1.0, { teeth: 2.0, tongue: 0.8, tucks: true }); break;
      case 'smirk': smile(W * 0.64, 1.2, { tucks: 'R', lip: 0.4, liftR: 1.5, liftL: -0.1 }); break;
      case 'tongue': {
        smile(W * 0.68, 1.9, { tucks: true });
        ctx.fillStyle = TONGUE;
        ctx.beginPath(); ctx.ellipse(W * 0.26, -2.4, 1.7, 1.9, 0.25, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = INK; ctx.lineWidth = 0.75; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(W * 0.26 + 0.1, -1.4); ctx.lineTo(W * 0.26 - 0.2, -3.0); ctx.lineWidth = 0.5; ctx.stroke();
        break;
      }
      case 'grumble': {
        // a little flat mouth with a pushed-out lower lip: grumpy-cute, not mean
        const g = ctx.createRadialGradient(0, -1.8, 0, 0, -1.8, W * 0.45);
        g.addColorStop(0, rgba(lipCol, 0.6)); g.addColorStop(1, rgba(lipCol, 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(0, -1.8, W * 0.45, 1.5, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(-W * 0.58, -0.3); ctx.bezierCurveTo(-W * 0.2, 0.5, W * 0.2, 0.5, W * 0.58, -0.3); ctx.stroke();
        break;
      }
      case 'chatter': open(W * 0.42, -0.5, 2.1, 0.4, { teeth: 1.3, tongue: 1, tucks: true }); break;
      case 'pursed': {
        ctx.fillStyle = rgba(lipCol, 0.75);
        ctx.beginPath(); ctx.ellipse(0, -0.5, 2.6, 1.5, 0, 0, Math.PI * 2); ctx.fill();
        smile(W * 0.32, 0.7, { width: 1.1 });
        break;
      }
      case 'tight': smile(W * 0.52, 1.2, { tucks: true, lip: 0.35, width: 1.15 }); break;
      case 'cool': smile(W * 0.68, 1.0, { tucks: 'L', lip: 0.5, liftL: 1.1 }); break;
      case 'dimples': smile(W * 0.68, 2.1, { tucks: false, lip: 0.4 }); dimples(W * 0.68); break;
    }
  };

  switch (e) {
    case 'neutral': rest(); break;
    case 'happy': open(W * 1.18, -2.2, 8.6, 2.6, { teeth: 2.2, tongue: 1, tucks: true }); break;
    case 'focus':
      // determined: lips pressed, pulled a little to one side
      smile(W * 0.45, -0.3, { tucks: true, lip: 0.45, liftR: 0.6, width: 1.45 });
      break;
    case 'surprised':
      ctx.fillStyle = MOUTH_IN;
      ctx.beginPath(); ctx.ellipse(0, -2.4, 4.4, 5.8, 0, 0, Math.PI * 2); ctx.fill();
      ctx.save(); ctx.clip();
      ctx.fillStyle = TONGUE; ctx.beginPath(); ctx.ellipse(0, -7.0, 3.6, 2.6, 0, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.ellipse(0, -2.4, 4.4, 5.8, 0, 0, Math.PI * 2); ctx.stroke();
      break;
    case 'sad':
      // a pout: corners down, lower lip pushed out
      smile(W * 0.45, -2.4, { lip: 0.7, width: 1.3 });
      break;
    case 'yell': open(W * 0.9, 3.4, 12.5, 0.8, { teeth: 1.9, tongue: 1 }); break;
    case 'smug': smile(W * 0.62, 1.2, { tucks: 'R', lip: 0.4, liftR: 2.8, liftL: -0.4, width: 1.3 }); break;
    case 'oops': {
      ctx.strokeStyle = INK; ctx.lineWidth = 1.15;
      ctx.beginPath(); ctx.moveTo(-W * 0.62, 0);
      const n = 4;
      for (let i = 0; i < n; i++) {
        const x0 = -W * 0.62 + (i * 1.24 * W) / n, x1 = x0 + (1.24 * W) / n;
        ctx.quadraticCurveTo((x0 + x1) / 2, i % 2 ? -1.1 : 1.1, x1, 0);
      }
      ctx.stroke();
      break;
    }
  }
  ctx.restore();

  if (e === 'oops') {
    // a sweat drop by the temple
    const sx = spec.eyePhi + spec.eyeW + 6, sy = spec.eyeTheta + spec.eyeH + 1;
    ctx.fillStyle = 'rgba(150,205,255,0.95)';
    ctx.beginPath(); ctx.moveTo(sx, sy + 4.5); ctx.quadraticCurveTo(sx + 3.2, sy - 0.8, sx, sy - 2.2); ctx.quadraticCurveTo(sx - 3.2, sy - 0.8, sx, sy + 4.5); ctx.fill();
    ctx.strokeStyle = 'rgba(60,110,170,0.8)'; ctx.lineWidth = 0.5; ctx.stroke();
  }
}

/** A filled brush stroke along a quadratic curve, w0 wide at the start tapering to w1 (rounded ends). */
function taper(ctx: Ctx, a: Pt, c: Pt, b: Pt, w0: number, w1: number) {
  const n = 14, L: Pt[] = [], R: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    const x = u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], y = u * u * a[1] + 2 * u * t * c[1] + t * t * b[1];
    const dx = 2 * u * (c[0] - a[0]) + 2 * t * (b[0] - c[0]), dy = 2 * u * (c[1] - a[1]) + 2 * t * (b[1] - c[1]);
    const len = Math.hypot(dx, dy) || 1;
    // fuller in the first third, then tapering off
    const w = (w0 + (w1 - w0) * t) * (0.85 + 0.25 * Math.sin(Math.min(1, t * 1.6) * Math.PI * 0.5)) * 0.5;
    L.push([x - (dy / len) * w, y + (dx / len) * w]);
    R.push([x + (dy / len) * w, y - (dx / len) * w]);
  }
  ctx.beginPath(); ctx.arc(a[0], a[1], w0 * 0.47, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(b[0], b[1], w1 * 0.5, 0, Math.PI * 2); ctx.fill();
  // fill the body as one polygon (the arcs above only round the ends)
  ctx.beginPath();
  ctx.moveTo(L[0][0], L[0][1]);
  for (const p of L) ctx.lineTo(p[0], p[1]);
  for (let i = n; i >= 0; i--) ctx.lineTo(R[i][0], R[i][1]);
  ctx.closePath();
  ctx.fill();
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

function mixHex(a: string, b: string, t: number): string {
  const na = parseInt(a.slice(1), 16), nb = parseInt(b.slice(1), 16);
  const ch = (s: number) => Math.round(((na >> s) & 255) * (1 - t) + ((nb >> s) & 255) * t);
  return `#${((ch(16) << 16) | (ch(8) << 8) | ch(0)).toString(16).padStart(6, '0')}`;
}

function rgba(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function mulberry(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
