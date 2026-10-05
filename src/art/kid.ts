import { hashString } from '../engine/rng';
import { HAIR, INK, SKIN, SKIN_SHADE } from '../data/palette';
import type { Kid, KidLook, Team } from '../data/types';
import { kidHeightFt } from '../sim/field';

// Every kid is drawn from code: a chibi cartoon with a big head, team
// uniform, and whatever grown-up costume pieces their persona calls for.

export type KidAnim =
  | 'idle' | 'ready' | 'run' | 'bat' | 'pitch' | 'throw' | 'catch' | 'dive' | 'jump'
  | 'cheer' | 'out' | 'stumble' | 'crouch' | 'trot' | 'wave' | 'bunt';

export interface KidPose {
  anim: KidAnim;
  t: number;
  view: 'front' | 'back';
  /** mirror horizontally (lefties, facing left) */
  flip?: boolean;
  /** 0..1 progress of a swing (bat anim) */
  swing?: number;
  /** windup progress 0..1, then follow-through 1..2 (pitch anim) */
  windup?: number;
  lift?: number;
  ball?: boolean;
  /** run direction on screen: -1 left, 1 right, 0 toward/away */
  lean?: number;
  /** show the held prop (portraits, dugout) */
  prop?: boolean;
  /** a lefty at the plate: the bat goes over the other shoulder */
  batLeft?: boolean;
}

interface Pal {
  skin: string; skinShade: string; hair: string; jersey: string; trim: string; pants: string; cap: string; brim: string; letter: string;
}

function pal(k: Kid, team: Team | null): Pal {
  const c = team?.colors ?? { primary: '#3a6ea5', secondary: '#f2b134', accent: '#f4f1e8' };
  return {
    skin: SKIN[k.look.skin], skinShade: SKIN_SHADE[k.look.skin], hair: HAIR[k.look.hairColor],
    jersey: c.primary, trim: c.secondary, pants: c.accent, cap: c.primary, brim: c.secondary,
    letter: c.secondary,
  };
}

export const jerseyNumber = (k: Kid) => (hashString(k.id) % 98) + 1;

export interface Dims {
  H: number; headR: number; headCY: number; shoulderY: number; hipY: number; torsoW: number; hipW: number; legW: number;
}

export function dims(look: KidLook): Dims {
  const H = kidHeightFt(look.height);
  const headR = 0.15 * H + 0.06;
  const headCY = -(H - headR);
  const shoulderY = -(H - headR * 2 + 0.12);
  const hipY = -(0.35 * H);
  const torsoW = 0.95 + look.build * 0.55;
  const hipW = 0.85 + look.build * 0.5;
  const legW = 0.4 + look.build * 0.16;
  return { H, headR, headCY, shoulderY, hipY, torsoW, hipW, legW };
}

function rr(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rad = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function blob(ctx: CanvasRenderingContext2D, fill: string, stroke = true) {
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) ctx.stroke();
}

/** A limb from (x1,y1) to (x2,y2) with round caps: outline, then fill. */
function limb(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, w: number, color: string, ink: number) {
  ctx.lineCap = 'round';
  ctx.strokeStyle = INK;
  ctx.lineWidth = w + ink * 2;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = w;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

function bentLimb(ctx: CanvasRenderingContext2D, x1: number, y1: number, kx: number, ky: number, x2: number, y2: number, w: number, color: string, ink: number) {
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const [c, ww] of [[INK, w + ink * 2], [color, w]] as const) {
    ctx.strokeStyle = c;
    ctx.lineWidth = ww;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(kx, ky); ctx.lineTo(x2, y2); ctx.stroke();
  }
}

// ──────────────────────────────────────────────────────────── the body

/**
 * Draw a kid standing at screen point (x, y) (their feet), `px` pixels per
 * foot. All internal geometry is in feet.
 */
export function drawKid(ctx: CanvasRenderingContext2D, k: Kid, team: Team | null, x: number, y: number, px: number, pose: KidPose) {
  const d = dims(k.look);
  const tall = d.H * px;
  if (tall < 3) return;
  ctx.save();
  ctx.translate(x, y);
  const lift = (pose.lift ?? 0) * px;
  // shadow
  ctx.fillStyle = 'rgba(0,0,0,0.22)';
  ctx.beginPath();
  ctx.ellipse(0, 0, d.torsoW * 0.75 * px, d.torsoW * 0.22 * px, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.translate(0, -lift);
  ctx.scale(px * (pose.flip ? -1 : 1), px);
  if (tall < 22) drawTiny(ctx, k, pal(k, team), d, pose);
  else drawFull(ctx, k, pal(k, team), d, pose, tall);
  ctx.restore();
}

function drawTiny(ctx: CanvasRenderingContext2D, k: Kid, p: Pal, d: Dims, pose: KidPose) {
  const run = pose.anim === 'run' || pose.anim === 'trot';
  const ph = run ? Math.sin(pose.t * 14) * 0.35 : 0;
  ctx.lineWidth = 0.12;
  ctx.strokeStyle = INK;
  ctx.fillStyle = p.pants;
  ctx.fillRect(-d.hipW / 2, d.hipY, d.hipW * 0.45, -d.hipY + ph);
  ctx.fillRect(d.hipW * 0.05, d.hipY, d.hipW * 0.45, -d.hipY - ph);
  rr(ctx, -d.torsoW / 2, d.shoulderY, d.torsoW, d.hipY - d.shoulderY + 0.2, 0.3);
  blob(ctx, p.jersey);
  ctx.beginPath();
  ctx.arc(0, d.headCY, d.headR, 0, Math.PI * 2);
  blob(ctx, p.skin);
  ctx.beginPath();
  ctx.arc(0, d.headCY - d.headR * 0.15, d.headR * 1.02, Math.PI, 0);
  blob(ctx, k.look.hat === 'hardhat' ? '#f4c430' : p.cap);
}

function drawFull(ctx: CanvasRenderingContext2D, k: Kid, p: Pal, d: Dims, pose: KidPose, tall: number) {
  const ink = Math.max(0.045, 1.1 / (tall / d.H));
  ctx.lineWidth = ink;
  ctx.strokeStyle = INK;
  ctx.lineJoin = 'round';
  const front = pose.view === 'front';
  const t = pose.t;
  const anim = pose.anim;

  // posture offsets
  let crouch = 0;      // how far the hips drop
  let tilt = 0;        // whole-body lean (radians)
  let bob = 0;
  let legA = 0, legB = 0; // leg swing angles
  let armL = { x: -0.15, y: 0.9 }, armR = { x: 0.15, y: 0.9 }; // hand offsets from shoulders (ft, +y down)
  let squash = 1;
  switch (anim) {
    case 'run': case 'trot': {
      const sp = anim === 'run' ? 13 : 8;
      const ph = Math.sin(t * sp);
      legA = ph * 0.7; legB = -ph * 0.7;
      bob = Math.abs(Math.cos(t * sp)) * 0.18;
      armL = { x: -0.35 - ph * 0.35, y: 0.55 }; armR = { x: 0.35 - ph * 0.35, y: 0.55 };
      tilt = (pose.lean ?? 0) * 0.15;
      break;
    }
    case 'ready': case 'crouch':
      crouch = anim === 'crouch' ? 0.65 : 0.32;
      legA = -0.25; legB = 0.25;
      armL = { x: -0.25, y: 0.75 }; armR = { x: 0.25, y: 0.75 };
      break;
    case 'cheer': {
      const jump = Math.abs(Math.sin(t * 9)) * 0.45;
      bob = -jump;
      armL = { x: -0.55, y: -1.25 }; armR = { x: 0.55, y: -1.25 };
      legA = -0.15; legB = 0.15;
      break;
    }
    case 'wave':
      armR = { x: 0.65 + Math.sin(t * 10) * 0.25, y: -1.1 };
      break;
    case 'out':
      squash = 0.93;
      tilt = 0.08;
      armL = { x: -0.2, y: 1.05 }; armR = { x: 0.2, y: 1.05 };
      break;
    case 'stumble':
      tilt = Math.sin(t * 18) * 0.25;
      armL = { x: -0.9, y: -0.4 }; armR = { x: 0.9, y: -0.2 };
      legA = 0.4; legB = -0.2;
      break;
    case 'dive':
      tilt = -1.2;
      crouch = 0.5;
      armL = { x: -0.3, y: -1.5 }; armR = { x: 0.3, y: -1.5 };
      legA = 0.3; legB = 0.6;
      break;
    case 'jump':
      bob = -0.6;
      armL = { x: -0.15, y: -1.6 }; armR = { x: 0.4, y: -1.2 };
      legA = 0.3; legB = -0.2;
      break;
    case 'catch':
      armL = { x: -0.15, y: -0.6 }; armR = { x: 0.2, y: -0.3 };
      break;
    case 'throw': {
      const q = Math.min(1, t / 0.3);
      armR = q < 0.5 ? { x: 0.8, y: -0.9 } : { x: -0.5, y: 0.4 };
      armL = { x: -0.6, y: 0.2 };
      tilt = q < 0.5 ? 0.12 : -0.15;
      legA = -0.3; legB = 0.35;
      break;
    }
    case 'pitch': {
      const w = pose.windup ?? 0;
      if (w < 0.55) { // leg kick
        const q = w / 0.55;
        legA = -q * 1.2; armL = { x: -0.1, y: 0.1 }; armR = { x: 0.1, y: 0.1 };
        crouch = -q * 0.1;
      } else if (w < 1) { // stride, arm back
        const q = (w - 0.55) / 0.45;
        legA = -1.2 + q * 1.6; legB = 0.3;
        armR = { x: 0.9, y: -0.8 * q }; armL = { x: -0.7, y: 0.1 };
        tilt = 0.1 * q;
      } else { // follow through
        legA = 0.45; legB = -0.4;
        armR = { x: -0.55, y: 0.85 }; armL = { x: -0.7, y: 0.4 };
        tilt = -0.18; crouch = 0.15;
      }
      break;
    }
    case 'bunt':
      crouch = 0.25;
      armL = { x: 0.35, y: 0.25 }; armR = { x: 0.55, y: 0.2 };
      break;
    default:
      break;
  }
  if (anim === 'idle') {
    bob = Math.sin(t * 2.2) * 0.04;
    armL = { x: -0.25, y: 0.95 }; armR = { x: 0.25, y: 0.95 };
  }

  ctx.save();
  ctx.translate(0, -bob);
  ctx.rotate(tilt);
  ctx.scale(1, squash);

  const hipY = d.hipY + crouch;
  const shY = d.shoulderY + crouch * 0.9;
  const headCY = d.headCY + crouch * 0.9;
  const legLen = -d.hipY;

  // cape goes behind everything
  if (k.look.body === 'cape') {
    ctx.beginPath();
    ctx.moveTo(-d.torsoW * 0.55, shY);
    ctx.quadraticCurveTo(-d.torsoW * 0.95, hipY + 0.6, -d.torsoW * 0.7 + Math.sin(t * 6) * 0.08, -0.25);
    ctx.lineTo(d.torsoW * 0.7 + Math.sin(t * 6 + 1) * 0.08, -0.25);
    ctx.quadraticCurveTo(d.torsoW * 0.95, hipY + 0.6, d.torsoW * 0.55, shY);
    ctx.closePath();
    blob(ctx, front ? '#6b1f7a' : '#8a2a9c');
  }
  // hair that hangs behind the head
  if (front) drawBackHair(ctx, k.look, p, headCY, d.headR, t);

  // legs
  const hipL = -d.hipW * 0.24, hipR = d.hipW * 0.24;
  const legPts = (hx: number, a: number) => {
    const knee = { x: hx + Math.sin(a) * legLen * 0.5, y: hipY + Math.cos(a) * legLen * 0.5 };
    const back = anim === 'run' || anim === 'trot' ? Math.max(0, -a) * 0.5 : 0;
    const foot = { x: knee.x + Math.sin(a * 0.4) * legLen * 0.45 - back * 0.2, y: knee.y + legLen * 0.48 - back * 0.25 - crouch * 0.4 };
    if (crouch > 0) { knee.x += (hx < 0 ? -1 : 1) * crouch * 0.5; }
    return { knee, foot };
  };
  const LA = legPts(hipL, legA), LB = legPts(hipR, legB);
  for (const [hx, L] of [[hipL, LA], [hipR, LB]] as const) {
    bentLimb(ctx, hx, hipY, L.knee.x, L.knee.y, L.foot.x, L.foot.y - 0.15, d.legW, p.pants, ink);
    // socks
    limb(ctx, L.knee.x + (L.foot.x - L.knee.x) * 0.55, L.knee.y + (L.foot.y - L.knee.y) * 0.55, L.foot.x, L.foot.y - 0.18, d.legW * 0.92, p.trim, ink * 0.6);
    // shoe
    ctx.beginPath();
    ctx.ellipse(L.foot.x + (front ? 0 : 0.02), L.foot.y - 0.06, d.legW * 0.95, 0.16, 0, 0, Math.PI * 2);
    blob(ctx, '#2d2a2e');
    ctx.fillStyle = '#f2f2f2';
    ctx.fillRect(L.foot.x - d.legW * 0.6, L.foot.y - 0.05, d.legW * 1.2, 0.06);
  }

  // overalls/pants top
  rr(ctx, -d.hipW / 2, hipY - 0.35, d.hipW, 0.55, 0.2);
  blob(ctx, k.look.body === 'overalls' ? '#4a6fa5' : p.pants);
  // belt
  ctx.fillStyle = k.look.body === 'toolbelt' ? '#8a5a2b' : '#3a2a20';
  ctx.fillRect(-d.hipW / 2 + 0.05, hipY - 0.32, d.hipW - 0.1, k.look.body === 'toolbelt' ? 0.16 : 0.09);

  // arms behind torso when seen from the back
  const shL = { x: -d.torsoW * 0.47, y: shY + 0.16 }, shR = { x: d.torsoW * 0.47, y: shY + 0.16 };
  const handL = { x: shL.x + armL.x, y: shL.y + armL.y }, handR = { x: shR.x + armR.x, y: shR.y + armR.y };
  const gloveHand = k.throws === 'R' ? 'L' : 'R';
  const drawArm = (sh: { x: number; y: number }, hand: { x: number; y: number }, isGlove: boolean) => {
    const elbow = { x: (sh.x + hand.x) / 2 + (sh.x < 0 ? -0.12 : 0.12), y: (sh.y + hand.y) / 2 };
    bentLimb(ctx, sh.x, sh.y, elbow.x, elbow.y, hand.x, hand.y, 0.25, p.skin, ink);
    // short sleeve
    limb(ctx, sh.x, sh.y, sh.x + (elbow.x - sh.x) * 0.55, sh.y + (elbow.y - sh.y) * 0.55, 0.32, p.jersey, ink);
    const showGlove = isGlove && anim !== 'bat' && anim !== 'bunt' && anim !== 'cheer' && anim !== 'wave' && anim !== 'idle' && anim !== 'out' && anim !== 'trot';
    if (showGlove) {
      ctx.beginPath();
      ctx.ellipse(hand.x, hand.y, 0.3, 0.34, 0, 0, Math.PI * 2);
      blob(ctx, '#8b5a2b');
      ctx.strokeStyle = '#5a3818';
      ctx.beginPath(); ctx.moveTo(hand.x - 0.15, hand.y - 0.1); ctx.lineTo(hand.x + 0.15, hand.y - 0.1); ctx.stroke();
      ctx.strokeStyle = INK;
      if (pose.ball) { ctx.beginPath(); ctx.arc(hand.x, hand.y - 0.05, 0.12, 0, Math.PI * 2); blob(ctx, '#ffffff'); }
    } else {
      ctx.beginPath();
      ctx.arc(hand.x, hand.y, 0.13, 0, Math.PI * 2);
      blob(ctx, p.skin);
    }
  };
  if (!front) {
    drawArm(shL, handL, gloveHand === 'L');
    drawArm(shR, handR, gloveHand === 'R');
  }

  // torso / jersey
  ctx.beginPath();
  ctx.moveTo(-d.torsoW / 2, shY + 0.15);
  ctx.quadraticCurveTo(-d.torsoW / 2 - 0.05, shY - 0.05, -d.torsoW * 0.3, shY - 0.04);
  ctx.lineTo(d.torsoW * 0.3, shY - 0.04);
  ctx.quadraticCurveTo(d.torsoW / 2 + 0.05, shY - 0.05, d.torsoW / 2, shY + 0.15);
  ctx.lineTo(d.hipW / 2 + 0.02, hipY - 0.15);
  ctx.quadraticCurveTo(0, hipY + 0.05, -d.hipW / 2 - 0.02, hipY - 0.15);
  ctx.closePath();
  blob(ctx, p.jersey);
  // piping
  ctx.strokeStyle = p.trim;
  ctx.lineWidth = 0.07;
  ctx.beginPath();
  ctx.moveTo(-d.torsoW * 0.42, shY + 0.1); ctx.lineTo(-d.hipW * 0.42, hipY - 0.18);
  ctx.moveTo(d.torsoW * 0.42, shY + 0.1); ctx.lineTo(d.hipW * 0.42, hipY - 0.18);
  ctx.stroke();
  ctx.strokeStyle = INK;
  ctx.lineWidth = ink;

  const torsoH = hipY - shY;
  if (front) {
    drawBodyFlair(ctx, k.look, d, shY, hipY);
    if (!['apron', 'overalls', 'vest'].includes(k.look.body)) jerseyText(ctx, k, p, 0, shY + torsoH * 0.45, torsoH * 0.36, 'front');
    drawNeck(ctx, k.look, shY, torsoH, t);
  } else {
    jerseyText(ctx, k, p, 0, shY + torsoH * 0.5, torsoH * 0.5, 'back');
    if (k.look.body === 'overalls' || k.look.body === 'suspenders' || k.look.body === 'apron') {
      ctx.strokeStyle = k.look.body === 'overalls' ? '#4a6fa5' : k.look.body === 'apron' ? '#f6f6f6' : '#c0392b';
      ctx.lineWidth = 0.13;
      ctx.beginPath();
      ctx.moveTo(-d.torsoW * 0.25, shY); ctx.lineTo(d.torsoW * 0.18, hipY - 0.2);
      ctx.moveTo(d.torsoW * 0.25, shY); ctx.lineTo(-d.torsoW * 0.18, hipY - 0.2);
      ctx.stroke();
      ctx.strokeStyle = INK; ctx.lineWidth = ink;
    }
  }

  // head
  drawHead(ctx, k, p, 0, headCY, d.headR, pose, ink);

  if (front) {
    drawArm(shL, handL, gloveHand === 'L');
    drawArm(shR, handR, gloveHand === 'R');
    if (pose.prop && k.look.holding !== 'none') drawHolding(ctx, k.look.holding, handR.x + 0.1, handR.y, t);
  }

  // the bat
  if (anim === 'bat' || anim === 'bunt') drawBat(ctx, pose, shY, front);

  ctx.restore();
}

function drawBat(ctx: CanvasRenderingContext2D, pose: KidPose, shY: number, front: boolean) {
  const s = pose.swing ?? 0;
  const dir = (front ? 1 : -1) * (pose.batLeft ? -1 : 1);
  const hands = { x: 0.32 * dir, y: shY + 0.35 };
  let ang: number;
  if (pose.anim === 'bunt') ang = -0.1;
  else if (s <= 0) ang = -2.35 + Math.sin(pose.t * 3) * 0.06; // waggle over the shoulder
  else if (s < 1) ang = -2.35 + s * 4.2;
  else ang = 1.85 + Math.min(1, (s - 1)) * 0.9; // follow-through
  const len = 2.4;
  const ex = hands.x + Math.cos(ang) * len * dir;
  const ey = hands.y + Math.sin(ang) * len * 0.75;
  ctx.lineCap = 'round';
  ctx.strokeStyle = INK;
  ctx.lineWidth = 0.32;
  ctx.beginPath(); ctx.moveTo(hands.x, hands.y); ctx.lineTo(ex, ey); ctx.stroke();
  ctx.strokeStyle = '#d9a35f';
  ctx.lineWidth = 0.2;
  ctx.beginPath(); ctx.moveTo(hands.x, hands.y); ctx.lineTo(ex, ey); ctx.stroke();
  ctx.strokeStyle = '#b8803f';
  ctx.lineWidth = 0.26;
  ctx.beginPath();
  ctx.moveTo(hands.x + (ex - hands.x) * 0.55, hands.y + (ey - hands.y) * 0.55);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.strokeStyle = INK;
  ctx.beginPath(); ctx.arc(hands.x, hands.y, 0.17, 0, Math.PI * 2); ctx.fillStyle = '#e8c39e'; ctx.lineWidth = 0.05; ctx.fill(); ctx.stroke();
}

function jerseyText(ctx: CanvasRenderingContext2D, k: Kid, p: Pal, x: number, y: number, size: number, side: 'front' | 'back') {
  ctx.save();
  ctx.fillStyle = p.letter;
  ctx.strokeStyle = INK;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const m = ctx.getTransform();
  const scale = Math.hypot(m.a, m.b);
  if (scale * size < 4) { ctx.restore(); return; }
  ctx.font = `900 ${size}px "Trebuchet MS", "Arial Black", sans-serif`;
  if (side === 'back') {
    const n = String(jerseyNumber(k));
    ctx.lineWidth = size * 0.12;
    ctx.strokeText(n, x, y + size * 0.15);
    ctx.fillText(n, x, y + size * 0.15);
    ctx.font = `800 ${size * 0.32}px "Trebuchet MS", sans-serif`;
    ctx.fillText(k.nick.toUpperCase().slice(0, 10), x, y - size * 0.62);
  } else {
    const n = String(jerseyNumber(k));
    ctx.lineWidth = size * 0.1;
    ctx.strokeText(n, x + size * 0.45, y);
    ctx.fillText(n, x + size * 0.45, y);
  }
  ctx.restore();
}

// ──────────────────────────────────────────────────────── costume bits

function drawBodyFlair(ctx: CanvasRenderingContext2D, look: KidLook, d: Dims, shY: number, hipY: number) {
  const w = d.torsoW;
  switch (look.body) {
    case 'suspenders':
      ctx.strokeStyle = '#c0392b';
      ctx.lineWidth = 0.12;
      ctx.beginPath();
      ctx.moveTo(-w * 0.27, shY); ctx.lineTo(-w * 0.22, hipY - 0.2);
      ctx.moveTo(w * 0.27, shY); ctx.lineTo(w * 0.22, hipY - 0.2);
      ctx.stroke();
      break;
    case 'apron':
      rr(ctx, -w * 0.36, shY + 0.15, w * 0.72, hipY - shY + 0.25, 0.08);
      blob(ctx, '#fbfbf7');
      ctx.beginPath(); ctx.arc(w * 0.12, hipY - 0.35, 0.1, 0, Math.PI * 2); ctx.fillStyle = '#e74c3c'; ctx.fill();
      break;
    case 'overalls':
      rr(ctx, -w * 0.32, shY + 0.3, w * 0.64, hipY - shY, 0.06);
      blob(ctx, '#4a6fa5');
      ctx.fillStyle = '#f1c40f';
      ctx.beginPath(); ctx.arc(-w * 0.22, shY + 0.38, 0.07, 0, Math.PI * 2); ctx.arc(w * 0.22, shY + 0.38, 0.07, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#4a6fa5'; ctx.lineWidth = 0.12;
      ctx.beginPath(); ctx.moveTo(-w * 0.22, shY + 0.35); ctx.lineTo(-w * 0.35, shY); ctx.moveTo(w * 0.22, shY + 0.35); ctx.lineTo(w * 0.35, shY); ctx.stroke();
      break;
    case 'vest':
      ctx.beginPath();
      ctx.moveTo(-w * 0.45, shY + 0.1); ctx.lineTo(0, hipY - 0.6); ctx.lineTo(w * 0.45, shY + 0.1);
      ctx.lineTo(w * 0.45, hipY - 0.15); ctx.lineTo(-w * 0.45, hipY - 0.15); ctx.closePath();
      blob(ctx, '#8e6c3a');
      ctx.strokeStyle = '#6b4f28'; ctx.lineWidth = 0.05;
      for (let i = 0; i < 4; i++) { ctx.beginPath(); ctx.moveTo(-w * 0.4, hipY - 0.3 - i * 0.18); ctx.lineTo(w * 0.4, hipY - 0.3 - i * 0.18); ctx.stroke(); }
      break;
    case 'pocketProtector':
      rr(ctx, -w * 0.32, shY + 0.3, 0.32, 0.3, 0.03);
      blob(ctx, '#ffffff');
      for (const [i, c] of ['#e74c3c', '#2980b9', '#27ae60'].entries()) {
        ctx.fillStyle = c; ctx.fillRect(-w * 0.3 + i * 0.09, shY + 0.18, 0.05, 0.18);
      }
      break;
    case 'badge':
      star(ctx, -w * 0.22, shY + 0.42, 0.2, '#f1c40f');
      break;
    case 'toolbelt':
      ctx.fillStyle = '#7f8c8d'; ctx.fillRect(w * 0.15, hipY - 0.25, 0.08, 0.35);
      ctx.fillStyle = '#c0392b'; ctx.fillRect(-w * 0.3, hipY - 0.25, 0.12, 0.25);
      break;
    case 'fannyPack':
      rr(ctx, -0.32, hipY - 0.4, 0.64, 0.26, 0.12);
      blob(ctx, '#e056a0');
      break;
    default:
      break;
  }
}

function star(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr2 = i % 2 ? r * 0.45 : r;
    ctx.lineTo(x + Math.cos(a) * rr2, y + Math.sin(a) * rr2);
  }
  ctx.closePath();
  blob(ctx, color);
}

function drawNeck(ctx: CanvasRenderingContext2D, look: KidLook, shY: number, torsoH: number, t: number) {
  const y = shY + 0.02;
  switch (look.neck) {
    case 'tie':
      ctx.beginPath();
      ctx.moveTo(-0.1, y); ctx.lineTo(0.1, y); ctx.lineTo(0.06, y + 0.12);
      ctx.lineTo(0.14 + Math.sin(t * 3) * 0.02, y + torsoH * 0.75); ctx.lineTo(0, y + torsoH * 0.85); ctx.lineTo(-0.14, y + torsoH * 0.75);
      ctx.lineTo(-0.06, y + 0.12); ctx.closePath();
      blob(ctx, '#c0392b');
      ctx.strokeStyle = '#f1c40f'; ctx.lineWidth = 0.04;
      ctx.beginPath(); ctx.moveTo(-0.08, y + 0.35); ctx.lineTo(0.1, y + 0.45); ctx.moveTo(-0.1, y + 0.55); ctx.lineTo(0.11, y + 0.65); ctx.stroke();
      ctx.strokeStyle = INK;
      break;
    case 'bowtie':
      ctx.beginPath();
      ctx.moveTo(0, y + 0.08); ctx.lineTo(-0.28, y - 0.06); ctx.lineTo(-0.28, y + 0.22); ctx.closePath();
      ctx.moveTo(0, y + 0.08); ctx.lineTo(0.28, y - 0.06); ctx.lineTo(0.28, y + 0.22); ctx.closePath();
      blob(ctx, '#d63384');
      ctx.beginPath(); ctx.arc(0, y + 0.08, 0.07, 0, Math.PI * 2); blob(ctx, '#a61e63');
      break;
    case 'pearls':
      for (let i = -4; i <= 4; i++) {
        ctx.beginPath();
        ctx.arc(i * 0.09, y + 0.1 + (16 - i * i) * 0.008, 0.055, 0, Math.PI * 2);
        ctx.fillStyle = '#fdfaf0'; ctx.fill(); ctx.lineWidth = 0.02; ctx.stroke();
      }
      break;
    case 'whistle':
    case 'lanyard':
    case 'medal': {
      const color = look.neck === 'medal' ? '#2980b9' : look.neck === 'lanyard' ? '#e74c3c' : '#f39c12';
      ctx.strokeStyle = color; ctx.lineWidth = 0.06;
      ctx.beginPath(); ctx.moveTo(-0.22, y); ctx.lineTo(0, y + torsoH * 0.42); ctx.lineTo(0.22, y); ctx.stroke();
      ctx.strokeStyle = INK; ctx.lineWidth = 0.04;
      if (look.neck === 'whistle') { rr(ctx, -0.12, y + torsoH * 0.4, 0.26, 0.13, 0.05); blob(ctx, '#bdc3c7'); }
      else if (look.neck === 'medal') { ctx.beginPath(); ctx.arc(0, y + torsoH * 0.48, 0.13, 0, Math.PI * 2); blob(ctx, '#f1c40f'); }
      else { rr(ctx, -0.15, y + torsoH * 0.42, 0.3, 0.2, 0.03); blob(ctx, '#ffffff'); }
      break;
    }
    case 'scarf':
      rr(ctx, -0.42, y - 0.08, 0.84, 0.22, 0.1);
      blob(ctx, '#e67e22');
      rr(ctx, 0.12, y + 0.05, 0.2, 0.55, 0.06);
      blob(ctx, '#e67e22');
      break;
    case 'bandana':
      ctx.beginPath(); ctx.moveTo(-0.38, y - 0.02); ctx.lineTo(0.38, y - 0.02); ctx.lineTo(0, y + 0.4); ctx.closePath();
      blob(ctx, '#c0392b');
      ctx.fillStyle = '#fff';
      for (const [dx, dy] of [[-0.15, 0.06], [0.12, 0.08], [0, 0.2]]) { ctx.beginPath(); ctx.arc(dx, y + dy, 0.03, 0, Math.PI * 2); ctx.fill(); }
      break;
    default:
      break;
  }
}

// ───────────────────────────────────────────────────────────── head

function drawBackHair(ctx: CanvasRenderingContext2D, look: KidLook, p: Pal, cy: number, R: number, t: number) {
  ctx.fillStyle = p.hair;
  switch (look.hair) {
    case 'long':
      rr(ctx, -R * 1.02, cy - R * 0.3, R * 2.04, R * 1.9, R * 0.5);
      blob(ctx, p.hair);
      break;
    case 'afro':
      ctx.beginPath(); ctx.arc(0, cy - R * 0.15, R * 1.38, 0, Math.PI * 2);
      blob(ctx, p.hair);
      break;
    case 'ponytail': {
      const sw = Math.sin(t * 5) * 0.1;
      ctx.beginPath();
      ctx.ellipse(R * 0.95 + sw, cy + R * 0.25, R * 0.32, R * 0.75, 0.4 + sw, 0, Math.PI * 2);
      blob(ctx, p.hair);
      break;
    }
    case 'pigtails':
      for (const s of [-1, 1]) {
        ctx.beginPath();
        ctx.ellipse(s * R * 1.12, cy + R * 0.35, R * 0.3, R * 0.6, s * 0.4, 0, Math.PI * 2);
        blob(ctx, p.hair);
      }
      break;
    case 'braids':
      for (const s of [-1, 1]) {
        for (let i = 0; i < 4; i++) {
          ctx.beginPath(); ctx.arc(s * R * 0.9, cy + R * (0.2 + i * 0.32), R * 0.18, 0, Math.PI * 2);
          blob(ctx, p.hair);
        }
      }
      break;
    case 'bun':
      ctx.beginPath(); ctx.arc(0, cy - R * 1.12, R * 0.4, 0, Math.PI * 2);
      blob(ctx, p.hair);
      break;
    case 'curly':
      for (let i = 0; i < 9; i++) {
        const a = Math.PI * (0.95 + i * 0.14);
        ctx.beginPath(); ctx.arc(Math.cos(a) * R * 1.02, cy + Math.sin(a) * R * 1.02, R * 0.3, 0, Math.PI * 2);
        blob(ctx, p.hair);
      }
      break;
    default:
      break;
  }
}

function headPath(ctx: CanvasRenderingContext2D, look: KidLook, cx: number, cy: number, R: number) {
  ctx.beginPath();
  switch (look.head) {
    case 'oval': ctx.ellipse(cx, cy, R * 0.88, R * 1.08, 0, 0, Math.PI * 2); break;
    case 'square': rr(ctx, cx - R * 0.98, cy - R * 0.98, R * 1.96, R * 1.96, R * 0.55); break;
    case 'wide': ctx.ellipse(cx, cy, R * 1.12, R * 0.94, 0, 0, Math.PI * 2); break;
    default: ctx.arc(cx, cy, R, 0, Math.PI * 2);
  }
}

function drawHead(ctx: CanvasRenderingContext2D, k: Kid, p: Pal, cx: number, cy: number, R: number, pose: KidPose, ink: number) {
  const look = k.look;
  const front = pose.view === 'front';
  ctx.lineWidth = ink;
  // ears
  for (const s of [-1, 1]) {
    ctx.beginPath(); ctx.ellipse(cx + s * R * 0.98, cy + R * 0.1, R * 0.2, R * 0.27, 0, 0, Math.PI * 2);
    blob(ctx, p.skin);
  }
  headPath(ctx, look, cx, cy, R);
  blob(ctx, p.skin);

  if (!front) {
    // back of the head: all hair
    ctx.save();
    headPath(ctx, look, cx, cy, R);
    ctx.clip();
    ctx.fillStyle = p.hair;
    const hairLow = look.hair === 'buzz' ? 0.15 : look.hair === 'mohawk' ? -0.2 : 0.55;
    ctx.fillRect(cx - R * 1.3, cy - R * 1.3, R * 2.6, R * (1.3 + hairLow));
    ctx.restore();
    headPath(ctx, look, cx, cy, R);
    ctx.stroke();
    if (look.hair === 'ponytail' || look.hair === 'pigtails' || look.hair === 'braids' || look.hair === 'long') {
      ctx.save();
      ctx.scale(-1, 1);
      drawBackHair(ctx, look, p, cy, R, pose.t);
      ctx.restore();
    }
    if (look.hair === 'afro') { ctx.beginPath(); ctx.arc(cx, cy - R * 0.15, R * 1.3, 0, Math.PI * 2); blob(ctx, p.hair); }
    if (look.hair === 'bun') { ctx.beginPath(); ctx.arc(cx, cy - R * 1.12, R * 0.4, 0, Math.PI * 2); blob(ctx, p.hair); }
    drawHat(ctx, k, p, cx, cy, R, false, pose.t);
    if (look.extra === 'curlers') drawCurlers(ctx, cx, cy, R, false);
    if (look.extra === 'headset' || look.extra === 'headband' || look.extra === 'sweatband') {
      ctx.strokeStyle = look.extra === 'headset' ? '#2c3e50' : '#e74c3c';
      ctx.lineWidth = R * 0.14;
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.98, Math.PI * 1.05, Math.PI * 1.95); ctx.stroke();
      ctx.strokeStyle = INK;
    }
    return;
  }

  // ── face (front)
  const blink = (Math.floor(pose.t * 1.3 + hashString(k.id) % 7) % 9 === 0) && (pose.t * 1.3 % 1) < 0.15;
  const eyeY = cy + R * 0.02;
  const eyeX = R * 0.37;
  // blush
  ctx.fillStyle = 'rgba(255,105,120,0.25)';
  for (const s of [-1, 1]) { ctx.beginPath(); ctx.ellipse(cx + s * R * 0.58, cy + R * 0.38, R * 0.18, R * 0.11, 0, 0, Math.PI * 2); ctx.fill(); }
  if (look.freckles) {
    ctx.fillStyle = '#b5651d';
    for (const s of [-1, 1]) for (const [fx, fy] of [[0.5, 0.28], [0.62, 0.36], [0.45, 0.42], [0.66, 0.24]]) {
      ctx.beginPath(); ctx.arc(cx + s * R * fx, cy + R * fy, R * 0.035, 0, Math.PI * 2); ctx.fill();
    }
  }
  const sad = pose.anim === 'out';
  const happy = pose.anim === 'cheer';
  // eyes
  for (const s of [-1, 1]) {
    const ex = cx + s * eyeX;
    if (blink || happy) {
      ctx.lineWidth = R * 0.08;
      ctx.beginPath();
      if (happy) ctx.arc(ex, eyeY + R * 0.05, R * 0.14, Math.PI * 1.1, Math.PI * 1.9);
      else { ctx.moveTo(ex - R * 0.15, eyeY); ctx.lineTo(ex + R * 0.15, eyeY); }
      ctx.stroke();
      continue;
    }
    ctx.lineWidth = ink;
    ctx.beginPath(); ctx.ellipse(ex, eyeY, R * 0.19, R * 0.24, 0, 0, Math.PI * 2);
    blob(ctx, '#ffffff');
    ctx.fillStyle = INK;
    ctx.beginPath(); ctx.arc(ex + R * 0.03, eyeY + R * 0.04, R * 0.11, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(ex + R * 0.07, eyeY - R * 0.01, R * 0.04, 0, Math.PI * 2); ctx.fill();
  }
  // brows
  ctx.lineWidth = R * 0.08;
  ctx.lineCap = 'round';
  ctx.strokeStyle = look.face === 'unibrow' ? '#1f1a17' : p.hair;
  const grumpy = look.mouth === 'frown' || look.face === 'unibrow';
  if (look.face === 'unibrow') {
    ctx.lineWidth = R * 0.14;
    ctx.beginPath(); ctx.moveTo(cx - eyeX - R * 0.2, eyeY - R * 0.36); ctx.quadraticCurveTo(cx, eyeY - R * 0.2, cx + eyeX + R * 0.2, eyeY - R * 0.36); ctx.stroke();
  } else {
    for (const s of [-1, 1]) {
      ctx.beginPath();
      const inner = grumpy ? 0.24 : sad ? 0.42 : 0.36;
      ctx.moveTo(cx + s * (eyeX - R * 0.18), eyeY - R * inner);
      ctx.lineTo(cx + s * (eyeX + R * 0.16), eyeY - R * (grumpy ? 0.38 : 0.36));
      ctx.stroke();
    }
  }
  ctx.strokeStyle = INK;
  // nose
  if (look.face === 'zinc') {
    ctx.beginPath(); ctx.ellipse(cx, cy + R * 0.2, R * 0.16, R * 0.13, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = ink * 0.6; ctx.stroke();
  } else {
    ctx.fillStyle = p.skinShade;
    ctx.beginPath(); ctx.ellipse(cx, cy + R * 0.22, R * 0.09, R * 0.07, 0, 0, Math.PI * 2); ctx.fill();
  }
  // mouth
  drawMouth(ctx, look, cx, cy + R * 0.5, R, sad, happy, pose.t, ink);
  drawFaceFlair(ctx, look, p, cx, cy, R);
  drawEyewear(ctx, look, cx, eyeY, R, eyeX, ink);

  // hair in front, hat, head extras
  drawFrontHair(ctx, look, p, cx, cy, R);
  if (look.extra === 'curlers') drawCurlers(ctx, cx, cy, R, true);
  drawHat(ctx, k, p, cx, cy, R, true, pose.t);
  drawExtra(ctx, look, cx, cy, R, ink);
}

function drawMouth(ctx: CanvasRenderingContext2D, look: KidLook, cx: number, my: number, R: number, sad: boolean, happy: boolean, t: number, ink: number) {
  ctx.lineWidth = ink;
  const m = sad ? 'frown' : happy ? 'open' : look.mouth;
  switch (m) {
    case 'frown':
      ctx.lineWidth = R * 0.08;
      ctx.beginPath(); ctx.arc(cx, my + R * 0.22, R * 0.24, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke();
      break;
    case 'smirk':
      ctx.lineWidth = R * 0.08;
      ctx.beginPath(); ctx.moveTo(cx - R * 0.22, my); ctx.quadraticCurveTo(cx + R * 0.05, my + R * 0.1, cx + R * 0.28, my - R * 0.1); ctx.stroke();
      break;
    case 'open': {
      const o = 0.16 + Math.abs(Math.sin(t * 8)) * 0.06;
      ctx.beginPath(); ctx.ellipse(cx, my + R * 0.04, R * 0.22, R * o, 0, 0, Math.PI * 2);
      blob(ctx, '#7a1f2b');
      ctx.fillStyle = '#ff7f8f';
      ctx.beginPath(); ctx.ellipse(cx, my + R * 0.12, R * 0.12, R * 0.06, 0, 0, Math.PI * 2); ctx.fill();
      break;
    }
    case 'whistle':
      ctx.beginPath(); ctx.arc(cx, my, R * 0.08, 0, Math.PI * 2); blob(ctx, '#7a1f2b');
      break;
    default: {
      // grin family
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.32, my - R * 0.04);
      ctx.quadraticCurveTo(cx, my + R * 0.36, cx + R * 0.32, my - R * 0.04);
      ctx.closePath();
      blob(ctx, '#7a1f2b');
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.28, my - R * 0.02);
      ctx.lineTo(cx + R * 0.28, my - R * 0.02);
      ctx.quadraticCurveTo(cx, my + R * 0.1, cx - R * 0.28, my - R * 0.02);
      ctx.fill();
      if (m === 'gap') { ctx.fillStyle = '#7a1f2b'; ctx.fillRect(cx - R * 0.03, my - R * 0.03, R * 0.07, R * 0.09); }
      if (m === 'braces') {
        ctx.strokeStyle = '#95a5a6'; ctx.lineWidth = R * 0.05;
        ctx.beginPath(); ctx.moveTo(cx - R * 0.24, my + R * 0.02); ctx.lineTo(cx + R * 0.24, my + R * 0.02); ctx.stroke();
        ctx.strokeStyle = INK;
      }
      if (m === 'tongue') {
        ctx.beginPath(); ctx.ellipse(cx + R * 0.08, my + R * 0.22, R * 0.11, R * 0.13, 0.2, 0, Math.PI * 2);
        ctx.fillStyle = '#ff6f86'; ctx.fill(); ctx.lineWidth = ink * 0.7; ctx.stroke();
      }
    }
  }
}

function drawFaceFlair(ctx: CanvasRenderingContext2D, look: KidLook, p: Pal, cx: number, cy: number, R: number) {
  const my = cy + R * 0.36;
  ctx.fillStyle = '#2b1d14';
  switch (look.face) {
    case 'mustache':
      ctx.beginPath();
      ctx.moveTo(cx, my - R * 0.02);
      ctx.bezierCurveTo(cx - R * 0.15, my - R * 0.12, cx - R * 0.42, my - R * 0.05, cx - R * 0.38, my + R * 0.08);
      ctx.bezierCurveTo(cx - R * 0.22, my + R * 0.02, cx - R * 0.1, my + R * 0.06, cx, my + R * 0.04);
      ctx.bezierCurveTo(cx + R * 0.1, my + R * 0.06, cx + R * 0.22, my + R * 0.02, cx + R * 0.38, my + R * 0.08);
      ctx.bezierCurveTo(cx + R * 0.42, my - R * 0.05, cx + R * 0.15, my - R * 0.12, cx, my - R * 0.02);
      ctx.fill();
      break;
    case 'handlebar':
      ctx.lineWidth = R * 0.1; ctx.strokeStyle = '#2b1d14'; ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx, my); ctx.quadraticCurveTo(cx - R * 0.35, my - R * 0.08, cx - R * 0.42, my - R * 0.28);
      ctx.arc(cx - R * 0.34, my - R * 0.28, R * 0.08, Math.PI, Math.PI * 2.3);
      ctx.moveTo(cx, my); ctx.quadraticCurveTo(cx + R * 0.35, my - R * 0.08, cx + R * 0.42, my - R * 0.28);
      ctx.arc(cx + R * 0.34, my - R * 0.28, R * 0.08, 0, -Math.PI * 1.3, true);
      ctx.stroke();
      ctx.strokeStyle = INK;
      break;
    case 'walrus':
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.5, my + R * 0.22);
      ctx.quadraticCurveTo(cx - R * 0.42, my - R * 0.15, cx, my - R * 0.08);
      ctx.quadraticCurveTo(cx + R * 0.42, my - R * 0.15, cx + R * 0.5, my + R * 0.22);
      ctx.quadraticCurveTo(cx, my + R * 0.05, cx - R * 0.5, my + R * 0.22);
      ctx.fill();
      break;
    case 'goatee':
      ctx.beginPath(); ctx.ellipse(cx, cy + R * 0.86, R * 0.13, R * 0.16, 0, 0, Math.PI * 2); ctx.fill();
      break;
    case 'beard':
      // a cut-up kitchen sponge, attached with an elastic
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.78, cy + R * 0.2);
      ctx.quadraticCurveTo(cx - R * 0.7, cy + R * 1.25, cx, cy + R * 1.3);
      ctx.quadraticCurveTo(cx + R * 0.7, cy + R * 1.25, cx + R * 0.78, cy + R * 0.2);
      ctx.quadraticCurveTo(cx + R * 0.4, cy + R * 0.65, cx, cy + R * 0.62);
      ctx.quadraticCurveTo(cx - R * 0.4, cy + R * 0.65, cx - R * 0.78, cy + R * 0.2);
      blob(ctx, look.hairColor === 4 ? '#e9c85a' : '#8a5a2b');
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      for (let i = 0; i < 6; i++) { ctx.beginPath(); ctx.arc(cx - R * 0.4 + i * R * 0.16, cy + R * (0.9 + (i % 2) * 0.15), R * 0.035, 0, Math.PI * 2); ctx.fill(); }
      break;
    default:
      break;
  }
  void p;
}

function drawEyewear(ctx: CanvasRenderingContext2D, look: KidLook, cx: number, ey: number, R: number, ex: number, ink: number) {
  ctx.lineWidth = ink * 1.1;
  switch (look.eyewear) {
    case 'glasses':
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.arc(cx + s * ex, ey, R * 0.27, 0, Math.PI * 2); ctx.fillStyle = 'rgba(200,230,255,0.25)'; ctx.fill(); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(cx - ex + R * 0.27, ey); ctx.lineTo(cx + ex - R * 0.27, ey); ctx.stroke();
      break;
    case 'reading': {
      const y = ey + R * 0.2;
      ctx.fillStyle = 'rgba(200,230,255,0.3)';
      for (const s of [-1, 1]) {
        ctx.beginPath(); ctx.ellipse(cx + s * ex * 0.85, y, R * 0.22, R * 0.13, 0, 0, Math.PI); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      ctx.beginPath(); ctx.moveTo(cx - ex * 0.85 + R * 0.22, y); ctx.lineTo(cx + ex * 0.85 - R * 0.22, y); ctx.stroke();
      ctx.strokeStyle = '#d4ac0d'; ctx.lineWidth = ink * 0.6;
      ctx.beginPath(); ctx.moveTo(cx - ex * 0.85 - R * 0.22, y); ctx.quadraticCurveTo(cx - R * 1.1, y + R * 1.1, cx - R * 0.9, y + R * 1.6);
      ctx.moveTo(cx + ex * 0.85 + R * 0.22, y); ctx.quadraticCurveTo(cx + R * 1.1, y + R * 1.1, cx + R * 0.9, y + R * 1.6); ctx.stroke();
      ctx.strokeStyle = INK;
      break;
    }
    case 'shades':
    case 'aviators': {
      const av = look.eyewear === 'aviators';
      ctx.fillStyle = av ? '#5d4a2e' : '#111';
      for (const s of [-1, 1]) {
        ctx.beginPath();
        if (av) ctx.ellipse(cx + s * ex, ey + R * 0.05, R * 0.26, R * 0.22, s * 0.25, 0, Math.PI * 2);
        else rr(ctx, cx + s * ex - R * 0.27, ey - R * 0.17, R * 0.54, R * 0.34, R * 0.1);
        ctx.fill();
        ctx.strokeStyle = av ? '#d4ac0d' : INK; ctx.stroke();
      }
      ctx.beginPath(); ctx.moveTo(cx - ex + R * 0.24, ey - R * 0.05); ctx.lineTo(cx + ex - R * 0.24, ey - R * 0.05); ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.ellipse(cx + s * ex - R * 0.08, ey - R * 0.05, R * 0.06, R * 0.03, -0.5, 0, Math.PI * 2); ctx.fill(); }
      ctx.strokeStyle = INK;
      break;
    }
    case 'goggles':
      ctx.strokeStyle = '#2c3e50'; ctx.lineWidth = R * 0.12;
      ctx.beginPath(); ctx.moveTo(cx - R * 1.0, ey); ctx.lineTo(cx + R * 1.0, ey); ctx.stroke();
      ctx.strokeStyle = INK; ctx.lineWidth = ink;
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.arc(cx + s * ex, ey, R * 0.3, 0, Math.PI * 2); ctx.fillStyle = 'rgba(120,200,255,0.45)'; ctx.fill(); ctx.lineWidth = R * 0.09; ctx.strokeStyle = '#f39c12'; ctx.stroke(); }
      ctx.strokeStyle = INK;
      break;
    case 'monocle':
      ctx.strokeStyle = '#d4ac0d'; ctx.lineWidth = ink * 1.3;
      ctx.beginPath(); ctx.arc(cx + ex, ey, R * 0.27, 0, Math.PI * 2); ctx.fillStyle = 'rgba(220,240,255,0.3)'; ctx.fill(); ctx.stroke();
      ctx.lineWidth = ink * 0.5;
      ctx.beginPath(); ctx.moveTo(cx + ex + R * 0.2, ey + R * 0.2); ctx.quadraticCurveTo(cx + R * 0.9, cy0(ey, R), cx + R * 0.6, ey + R * 1.6); ctx.stroke();
      ctx.strokeStyle = INK;
      break;
    default:
      break;
  }
}
const cy0 = (ey: number, R: number) => ey + R * 1.2;

function drawFrontHair(ctx: CanvasRenderingContext2D, look: KidLook, p: Pal, cx: number, cy: number, R: number) {
  ctx.fillStyle = p.hair;
  const top = cy - R;
  switch (look.hair) {
    case 'bowl':
      ctx.beginPath();
      ctx.moveTo(cx - R * 1.02, cy - R * 0.05);
      ctx.quadraticCurveTo(cx - R * 1.05, top - R * 0.05, cx, top - R * 0.08);
      ctx.quadraticCurveTo(cx + R * 1.05, top - R * 0.05, cx + R * 1.02, cy - R * 0.05);
      ctx.lineTo(cx + R * 0.8, cy - R * 0.32); ctx.lineTo(cx - R * 0.8, cy - R * 0.32); ctx.closePath();
      blob(ctx, p.hair);
      break;
    case 'sidepart':
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.95, cy - R * 0.1);
      ctx.quadraticCurveTo(cx - R * 0.9, top - R * 0.1, cx + R * 0.2, top - R * 0.05);
      ctx.quadraticCurveTo(cx + R * 0.98, top + R * 0.1, cx + R * 0.97, cy - R * 0.15);
      ctx.quadraticCurveTo(cx + R * 0.3, cy - R * 0.62, cx - R * 0.6, cy - R * 0.4);
      ctx.closePath();
      blob(ctx, p.hair);
      break;
    case 'spiky':
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.95, cy - R * 0.2);
      for (let i = 0; i <= 6; i++) {
        const x = cx - R * 0.95 + (i * R * 1.9) / 6;
        ctx.lineTo(x + R * 0.16, cy - R * (0.42 + (i % 2) * 0.22));
      }
      ctx.lineTo(cx + R * 0.95, cy - R * 0.2);
      ctx.lineTo(cx + R * 0.9, top); ctx.lineTo(cx - R * 0.9, top); ctx.closePath();
      blob(ctx, p.hair);
      break;
    case 'messy':
    case 'curly':
      for (let i = 0; i < 6; i++) {
        ctx.beginPath(); ctx.arc(cx - R * 0.7 + i * R * 0.28, cy - R * 0.55 + (i % 2) * R * 0.08, R * 0.22, 0, Math.PI * 2);
        blob(ctx, p.hair);
      }
      break;
    case 'mohawk':
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.15, cy - R * 0.5);
      for (let i = 0; i < 5; i++) ctx.lineTo(cx + (i % 2 ? 0.12 : -0.12) * R, top - R * (0.25 + i * 0.12));
      ctx.lineTo(cx + R * 0.15, cy - R * 0.5);
      ctx.closePath();
      blob(ctx, p.hair);
      break;
    case 'long':
    case 'ponytail':
    case 'pigtails':
    case 'braids':
    case 'bob':
    case 'bun':
      ctx.beginPath();
      ctx.moveTo(cx - R * 1.0, cy + (look.hair === 'bob' ? R * 0.4 : R * 0.1));
      ctx.quadraticCurveTo(cx - R * 1.05, top - R * 0.1, cx, top - R * 0.08);
      ctx.quadraticCurveTo(cx + R * 1.05, top - R * 0.1, cx + R * 1.0, cy + (look.hair === 'bob' ? R * 0.4 : R * 0.1));
      ctx.lineTo(cx + R * 0.82, cy - R * 0.25);
      ctx.quadraticCurveTo(cx, cy - R * 0.58, cx - R * 0.82, cy - R * 0.25);
      ctx.closePath();
      blob(ctx, p.hair);
      break;
    case 'afro':
      ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.62, R * 0.95, R * 0.42, 0, Math.PI, 0); ctx.closePath();
      blob(ctx, p.hair);
      break;
    case 'buzz':
      ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.55, R * 0.88, R * 0.42, 0, Math.PI, 0); ctx.closePath();
      ctx.fillStyle = p.hair; ctx.globalAlpha = 0.85; ctx.fill(); ctx.globalAlpha = 1;
      break;
    default:
      break;
  }
}

function drawCurlers(ctx: CanvasRenderingContext2D, cx: number, cy: number, R: number, front: boolean) {
  const colors = ['#ff6fa8', '#6fc3ff', '#ffd86f', '#9cff6f'];
  for (let i = 0; i < 5; i++) {
    const x = cx - R * 0.9 + i * R * 0.45;
    const y = cy - R * (front ? 0.45 : 0.6) - Math.abs(i - 2) * R * -0.06;
    rr(ctx, x - R * 0.13, y - R * 0.12, R * 0.26, R * 0.24, R * 0.08);
    blob(ctx, colors[i % colors.length]);
  }
}

function drawHat(ctx: CanvasRenderingContext2D, k: Kid, p: Pal, cx: number, cy: number, R: number, front: boolean, t: number) {
  const hat = k.look.hat;
  const top = cy - R;
  const crown = (color: string, h = 0.62, w = 1.06) => {
    ctx.beginPath();
    ctx.moveTo(cx - R * w, cy - R * 0.28);
    ctx.bezierCurveTo(cx - R * w, top - R * h, cx + R * w, top - R * h, cx + R * w, cy - R * 0.28);
    ctx.closePath();
    blob(ctx, color);
  };
  const logo = (y: number) => {
    if (!front) return;
    ctx.save();
    const m = ctx.getTransform();
    if (Math.hypot(m.a, m.b) * R < 8) { ctx.restore(); return; }
    ctx.fillStyle = p.letter;
    ctx.font = `900 ${R * 0.5}px "Trebuchet MS", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(k.id === 'q' ? 'Q' : (k.first[0] ?? 'X'), cx, y);
    ctx.restore();
  };
  switch (hat) {
    case 'cap':
      crown(p.cap);
      if (front) {
        ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.3, R * 1.0, R * 0.2, 0, 0, Math.PI); blob(ctx, p.brim);
        logo(cy - R * 0.72);
      } else {
        ctx.fillStyle = p.brim; ctx.fillRect(cx - R * 0.25, cy - R * 0.4, R * 0.5, R * 0.1);
      }
      ctx.beginPath(); ctx.arc(cx, top - R * 0.42, R * 0.08, 0, Math.PI * 2); blob(ctx, p.cap);
      break;
    case 'capBack':
      crown(p.cap);
      if (front) {
        ctx.fillStyle = p.brim; ctx.fillRect(cx - R * 0.3, cy - R * 0.42, R * 0.6, R * 0.12);
      } else {
        ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.3, R * 1.0, R * 0.2, 0, 0, Math.PI); blob(ctx, p.brim);
        logo(cy - R * 0.72);
      }
      break;
    case 'visor':
      ctx.fillStyle = p.cap;
      rr(ctx, cx - R * 1.0, cy - R * 0.55, R * 2.0, R * 0.26, R * 0.1); blob(ctx, p.cap);
      if (front) { ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.3, R * 1.05, R * 0.24, 0, 0, Math.PI); blob(ctx, p.brim); }
      break;
    case 'bucket':
      crown('#c8b88a', 0.75, 0.9);
      ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.3, R * 1.35, R * 0.3, 0, 0, Math.PI * 2); blob(ctx, '#b5a36f');
      crown('#c8b88a', 0.75, 0.9);
      if (k.id === 'sammy') { // fishing lures
        ctx.fillStyle = '#e74c3c'; ctx.fillRect(cx + R * 0.4, cy - R * 0.7, R * 0.12, R * 0.2);
        ctx.fillStyle = '#f1c40f'; ctx.fillRect(cx - R * 0.5, cy - R * 0.75, R * 0.12, R * 0.2);
      }
      break;
    case 'trucker':
      crown('#f4f1e8', 0.75);
      ctx.save(); ctx.beginPath();
      ctx.moveTo(cx - R * 1.06, cy - R * 0.28);
      ctx.bezierCurveTo(cx - R * 1.06, top - R * 0.75, cx + R * 1.06, top - R * 0.75, cx + R * 1.06, cy - R * 0.28);
      ctx.closePath(); ctx.clip();
      ctx.fillStyle = p.cap;
      ctx.fillRect(cx - R * 1.2, cy - R * 2, front ? R * 2.4 : 0, R * 3);
      ctx.restore();
      if (front) {
        ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.3, R * 1.05, R * 0.22, 0, 0, Math.PI); blob(ctx, p.brim);
        logo(cy - R * 0.78);
      }
      break;
    case 'flatcap':
      ctx.beginPath();
      ctx.moveTo(cx - R * 1.05, cy - R * 0.3);
      ctx.quadraticCurveTo(cx - R * 0.9, top - R * 0.3, cx + R * 0.2, top - R * 0.22);
      ctx.quadraticCurveTo(cx + R * 1.2, top - R * 0.05, cx + R * 1.08, cy - R * 0.3);
      ctx.closePath();
      blob(ctx, '#7f6a52');
      if (front) { ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.3, R * 0.9, R * 0.14, 0, 0, Math.PI); blob(ctx, '#6a5843'); }
      break;
    case 'cowboy':
      ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.38, R * 1.65, R * 0.32, 0, 0, Math.PI * 2); blob(ctx, '#a0703c');
      ctx.beginPath();
      ctx.moveTo(cx - R * 0.8, cy - R * 0.4);
      ctx.lineTo(cx - R * 0.75, top - R * 0.6);
      ctx.quadraticCurveTo(cx, top - R * 0.35, cx + R * 0.75, top - R * 0.6);
      ctx.lineTo(cx + R * 0.8, cy - R * 0.4);
      ctx.closePath();
      blob(ctx, '#a0703c');
      ctx.fillStyle = p.cap; ctx.fillRect(cx - R * 0.78, cy - R * 0.62, R * 1.56, R * 0.16);
      break;
    case 'hardhat':
      crown('#f4c430', 0.85, 1.08);
      ctx.beginPath(); ctx.ellipse(cx, cy - R * 0.3, R * 1.22, R * 0.18, 0, 0, Math.PI * 2); blob(ctx, '#e0b020');
      ctx.strokeStyle = '#c99a10'; ctx.lineWidth = R * 0.08;
      ctx.beginPath(); ctx.moveTo(cx, top - R * 0.55); ctx.lineTo(cx, cy - R * 0.35); ctx.stroke();
      ctx.strokeStyle = INK;
      break;
  }
  void t;
}

function drawExtra(ctx: CanvasRenderingContext2D, look: KidLook, cx: number, cy: number, R: number, ink: number) {
  switch (look.extra) {
    case 'headset':
      ctx.strokeStyle = '#2c3e50'; ctx.lineWidth = R * 0.1;
      ctx.beginPath(); ctx.arc(cx, cy - R * 0.1, R * 1.05, Math.PI * 1.05, Math.PI * 1.95); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - R * 1.0, cy + R * 0.1); ctx.quadraticCurveTo(cx - R * 0.9, cy + R * 0.7, cx - R * 0.3, cy + R * 0.6); ctx.stroke();
      ctx.strokeStyle = INK; ctx.lineWidth = ink;
      ctx.beginPath(); ctx.ellipse(cx - R * 1.02, cy + R * 0.05, R * 0.17, R * 0.24, 0, 0, Math.PI * 2); blob(ctx, '#34495e');
      ctx.beginPath(); ctx.arc(cx - R * 0.28, cy + R * 0.6, R * 0.08, 0, Math.PI * 2); blob(ctx, '#111');
      break;
    case 'pencil':
      ctx.save(); ctx.translate(cx + R * 1.0, cy - R * 0.1); ctx.rotate(-0.6);
      ctx.fillStyle = '#f1c40f'; ctx.fillRect(-R * 0.06, -R * 0.5, R * 0.12, R * 0.8); ctx.strokeRect(-R * 0.06, -R * 0.5, R * 0.12, R * 0.8);
      ctx.fillStyle = '#ff9ab0'; ctx.fillRect(-R * 0.06, R * 0.3, R * 0.12, R * 0.1);
      ctx.restore();
      break;
    case 'earpiece':
      ctx.strokeStyle = '#7f8c8d'; ctx.lineWidth = R * 0.05;
      ctx.beginPath(); ctx.moveTo(cx + R * 1.02, cy + R * 0.15);
      for (let i = 0; i < 5; i++) ctx.quadraticCurveTo(cx + R * (1.15 - (i % 2) * 0.15), cy + R * (0.3 + i * 0.15), cx + R * 1.05, cy + R * (0.38 + i * 0.15));
      ctx.stroke(); ctx.strokeStyle = INK;
      break;
    case 'headband':
    case 'sweatband':
      rr(ctx, cx - R * 1.0, cy - R * 0.5, R * 2.0, R * 0.2, R * 0.08);
      blob(ctx, look.extra === 'sweatband' ? '#ffffff' : '#e74c3c');
      if (look.extra === 'sweatband') { ctx.fillStyle = '#e74c3c'; ctx.fillRect(cx - R * 1.0, cy - R * 0.43, R * 2, R * 0.05); }
      break;
    case 'bandaid':
      ctx.save(); ctx.translate(cx + R * 0.55, cy - R * 0.25); ctx.rotate(0.6);
      rr(ctx, -R * 0.22, -R * 0.08, R * 0.44, R * 0.16, R * 0.06); blob(ctx, '#f5cba7');
      ctx.restore();
      break;
    case 'bow':
      ctx.save(); ctx.translate(cx + R * 0.65, cy - R * 0.75);
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-R * 0.4, -R * 0.25); ctx.lineTo(-R * 0.4, R * 0.25); ctx.closePath();
      ctx.moveTo(0, 0); ctx.lineTo(R * 0.4, -R * 0.25); ctx.lineTo(R * 0.4, R * 0.25); ctx.closePath();
      blob(ctx, '#ff4fa3');
      ctx.restore();
      break;
    case 'earrings':
      ctx.strokeStyle = '#f1c40f'; ctx.lineWidth = R * 0.06;
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.arc(cx + s * R * 0.98, cy + R * 0.45, R * 0.13, 0, Math.PI * 2); ctx.stroke(); }
      ctx.strokeStyle = INK;
      break;
    case 'flower':
      for (let i = 0; i < 5; i++) {
        const a = (i / 5) * Math.PI * 2;
        ctx.beginPath(); ctx.arc(cx - R * 0.95 + Math.cos(a) * R * 0.15, cy - R * 0.35 + Math.sin(a) * R * 0.15, R * 0.12, 0, Math.PI * 2);
        blob(ctx, '#ff7eb6');
      }
      ctx.beginPath(); ctx.arc(cx - R * 0.95, cy - R * 0.35, R * 0.08, 0, Math.PI * 2); blob(ctx, '#f9e04b');
      break;
    default:
      break;
  }
}

// ───────────────────────────────────────────────────────────── props

export function drawHolding(ctx: CanvasRenderingContext2D, h: KidLook['holding'], x: number, y: number, t: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.lineWidth = 0.05;
  ctx.strokeStyle = INK;
  switch (h) {
    case 'coffee':
      rr(ctx, -0.15, -0.35, 0.3, 0.38, 0.05); blob(ctx, '#ffffff');
      ctx.fillStyle = '#8b5a2b'; ctx.fillRect(-0.15, -0.22, 0.3, 0.1);
      ctx.strokeStyle = 'rgba(200,200,200,0.8)';
      ctx.beginPath(); ctx.moveTo(0, -0.45); ctx.quadraticCurveTo(0.1 + Math.sin(t * 3) * 0.05, -0.6, 0, -0.75); ctx.stroke();
      break;
    case 'clipboard':
      rr(ctx, -0.25, -0.6, 0.5, 0.65, 0.04); blob(ctx, '#a0703c');
      ctx.fillStyle = '#ffffff'; ctx.fillRect(-0.2, -0.52, 0.4, 0.52);
      ctx.fillStyle = '#7f8c8d'; ctx.fillRect(-0.1, -0.64, 0.2, 0.08);
      ctx.strokeStyle = '#95a5a6';
      for (let i = 0; i < 4; i++) { ctx.beginPath(); ctx.moveTo(-0.15, -0.42 + i * 0.1); ctx.lineTo(0.15, -0.42 + i * 0.1); ctx.stroke(); }
      break;
    case 'briefcase':
      rr(ctx, -0.35, -0.1, 0.7, 0.45, 0.05); blob(ctx, '#6e2c00');
      ctx.strokeRect(-0.1, -0.2, 0.2, 0.1);
      ctx.fillStyle = '#f1c40f'; ctx.fillRect(-0.04, 0, 0.08, 0.06);
      break;
    case 'newspaper':
      rr(ctx, -0.3, -0.5, 0.6, 0.5, 0.02); blob(ctx, '#f2f0e6');
      ctx.fillStyle = '#555'; ctx.fillRect(-0.25, -0.45, 0.5, 0.08);
      ctx.fillStyle = '#999'; for (let i = 0; i < 4; i++) ctx.fillRect(-0.25, -0.32 + i * 0.07, 0.5, 0.03);
      break;
    case 'calculator':
      rr(ctx, -0.18, -0.4, 0.36, 0.45, 0.04); blob(ctx, '#34495e');
      ctx.fillStyle = '#a8e6cf'; ctx.fillRect(-0.14, -0.36, 0.28, 0.1);
      ctx.fillStyle = '#ecf0f1'; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) ctx.fillRect(-0.13 + i * 0.1, -0.22 + j * 0.08, 0.06, 0.05);
      break;
    case 'binoculars':
      for (const s of [-1, 1]) { rr(ctx, s * 0.12 - 0.1, -0.35, 0.2, 0.35, 0.06); blob(ctx, '#2c3e50'); }
      break;
    case 'gavel':
      ctx.save(); ctx.rotate(-0.5 + Math.abs(Math.sin(t * 4)) * 0.3);
      ctx.fillStyle = '#8b5a2b'; ctx.fillRect(-0.03, -0.6, 0.07, 0.6); ctx.strokeRect(-0.03, -0.6, 0.07, 0.6);
      rr(ctx, -0.2, -0.75, 0.4, 0.18, 0.04); blob(ctx, '#6e3b12');
      ctx.restore();
      break;
    case 'microphone':
      ctx.fillStyle = '#2c3e50'; ctx.fillRect(-0.04, -0.45, 0.08, 0.45);
      ctx.beginPath(); ctx.arc(0, -0.5, 0.12, 0, Math.PI * 2); blob(ctx, '#95a5a6');
      break;
    case 'magnifier':
      ctx.strokeStyle = '#8b5a2b'; ctx.lineWidth = 0.08; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0.15, -0.3); ctx.stroke();
      ctx.lineWidth = 0.06; ctx.strokeStyle = INK;
      ctx.beginPath(); ctx.arc(0.25, -0.5, 0.2, 0, Math.PI * 2); ctx.fillStyle = 'rgba(200,230,255,0.5)'; ctx.fill(); ctx.stroke();
      break;
    case 'phone':
      rr(ctx, -0.1, -0.4, 0.2, 0.36, 0.04); blob(ctx, '#2c3e50');
      ctx.fillStyle = '#6ec6ff'; ctx.fillRect(-0.07, -0.36, 0.14, 0.26);
      break;
    case 'juicebox':
      rr(ctx, -0.13, -0.35, 0.26, 0.35, 0.03); blob(ctx, '#ff9f43');
      ctx.strokeStyle = '#ecf0f1'; ctx.lineWidth = 0.04; ctx.beginPath(); ctx.moveTo(0.05, -0.35); ctx.lineTo(0.12, -0.55); ctx.stroke();
      break;
    case 'lunchpail':
      rr(ctx, -0.3, -0.3, 0.6, 0.35, 0.05); blob(ctx, '#7f8c8d');
      ctx.beginPath(); ctx.arc(0, -0.3, 0.16, Math.PI, 0); ctx.stroke();
      break;
    case 'wand':
      ctx.save(); ctx.rotate(-0.6);
      ctx.fillStyle = '#111'; ctx.fillRect(-0.03, -0.7, 0.06, 0.7);
      ctx.fillStyle = '#fff'; ctx.fillRect(-0.03, -0.75, 0.06, 0.1);
      ctx.restore();
      star(ctx, -0.45 + Math.sin(t * 6) * 0.05, -0.75, 0.1, '#f9e04b');
      break;
    case 'trophy':
      ctx.beginPath(); ctx.moveTo(-0.2, -0.6); ctx.lineTo(0.2, -0.6); ctx.quadraticCurveTo(0.2, -0.25, 0, -0.22); ctx.quadraticCurveTo(-0.2, -0.25, -0.2, -0.6); blob(ctx, '#f1c40f');
      ctx.fillStyle = '#d4ac0d'; ctx.fillRect(-0.05, -0.22, 0.1, 0.12); ctx.fillRect(-0.15, -0.1, 0.3, 0.08);
      break;
    case 'rollingPin':
      ctx.save(); ctx.rotate(0.3);
      rr(ctx, -0.35, -0.08, 0.7, 0.16, 0.07); blob(ctx, '#e0b77a');
      ctx.restore();
      break;
    case 'wrench':
      ctx.save(); ctx.rotate(-0.4);
      ctx.fillStyle = '#95a5a6'; ctx.fillRect(-0.04, -0.6, 0.08, 0.6); ctx.strokeRect(-0.04, -0.6, 0.08, 0.6);
      ctx.beginPath(); ctx.arc(0, -0.66, 0.11, 0.6, Math.PI * 2.4); blob(ctx, '#95a5a6');
      ctx.restore();
      break;
    case 'flag':
      ctx.fillStyle = '#7f8c8d'; ctx.fillRect(-0.02, -0.8, 0.04, 0.8);
      ctx.beginPath(); ctx.moveTo(0.02, -0.8); ctx.lineTo(0.4 + Math.sin(t * 8) * 0.05, -0.68); ctx.lineTo(0.02, -0.56); ctx.closePath(); blob(ctx, '#ff7f11');
      break;
    case 'horseshoe':
      ctx.strokeStyle = '#7f8c8d'; ctx.lineWidth = 0.09;
      ctx.beginPath(); ctx.arc(0, -0.25, 0.17, Math.PI * 0.85, Math.PI * 2.15); ctx.stroke();
      ctx.strokeStyle = INK;
      break;
    case 'bowlingBall':
      ctx.beginPath(); ctx.arc(0, -0.2, 0.28, 0, Math.PI * 2); blob(ctx, '#1f3a93');
      ctx.fillStyle = '#111'; for (const [dx, dy] of [[-0.06, -0.3], [0.06, -0.3], [0, -0.18]]) { ctx.beginPath(); ctx.arc(dx, dy, 0.04, 0, Math.PI * 2); ctx.fill(); }
      break;
    case 'fishingRod':
      ctx.strokeStyle = '#8b5a2b'; ctx.lineWidth = 0.05;
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0.5, -1.4); ctx.stroke();
      ctx.strokeStyle = '#bdc3c7'; ctx.lineWidth = 0.015;
      ctx.beginPath(); ctx.moveTo(0.5, -1.4); ctx.lineTo(0.7, -0.4 + Math.sin(t * 3) * 0.05); ctx.stroke();
      ctx.beginPath(); ctx.arc(0.7, -0.38, 0.05, 0, Math.PI * 2); ctx.fillStyle = '#e74c3c'; ctx.fill();
      break;
    case 'crystalBall': {
      ctx.beginPath(); ctx.arc(0, -0.3, 0.25, 0, Math.PI * 2);
      const g = ctx.createRadialGradient(-0.08, -0.38, 0.02, 0, -0.3, 0.25);
      g.addColorStop(0, '#f5e6ff'); g.addColorStop(1, '#8e44ad');
      ctx.fillStyle = g; ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#6e2c00'; ctx.fillRect(-0.18, -0.07, 0.36, 0.08);
      break;
    }
    case 'dumbbell':
      ctx.fillStyle = '#7f8c8d'; ctx.fillRect(-0.25, -0.05, 0.5, 0.08);
      for (const s of [-1, 1]) { rr(ctx, s * 0.25 - 0.08, -0.15, 0.16, 0.28, 0.04); blob(ctx, '#2c3e50'); }
      break;
    default:
      break;
  }
  ctx.restore();
}

// ─────────────────────────────────────────────────────────── portrait

/** Waist-up card portrait, holding their signature prop. */
export function drawPortrait(ctx: CanvasRenderingContext2D, k: Kid, team: Team | null, w: number, h: number, t = 0) {
  const d = dims(k.look);
  const waist = -d.hipY * 0.85;
  const top = d.H + 0.55;
  const px = h / (top - waist);
  drawKid(ctx, k, team, w / 2, h + waist * px, px, { anim: 'idle', t, view: 'front', prop: true });
}
