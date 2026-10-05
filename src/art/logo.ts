import { INK } from '../data/palette';
import type { Team } from '../data/types';

/** Draw a team's round badge logo centered at (cx, cy) with radius r. */
export function drawLogo(ctx: CanvasRenderingContext2D, team: Team, cx: number, cy: number, r: number) {
  const c = team.colors;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = INK;
  ctx.lineWidth = r * 0.07;
  // badge
  ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fillStyle = c.secondary; ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.arc(0, 0, r * 0.84, 0, Math.PI * 2);
  ctx.fillStyle = c.primary; ctx.fill(); ctx.stroke();
  // baseball stitches around the rim
  ctx.strokeStyle = '#e74c3c';
  ctx.lineWidth = r * 0.035;
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(Math.cos(a) * r * 0.9, Math.sin(a) * r * 0.9);
    ctx.lineTo(Math.cos(a + 0.12) * r * 0.95, Math.sin(a + 0.12) * r * 0.95);
    ctx.stroke();
  }
  ctx.strokeStyle = INK;
  ctx.lineWidth = r * 0.06;
  ctx.scale(r / 50, r / 50);
  icon(ctx, team);
  ctx.restore();
}

function fillStroke(ctx: CanvasRenderingContext2D, fill: string) {
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.stroke();
}

function eye(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); fillStroke(ctx, '#fff');
  ctx.beginPath(); ctx.arc(x + r * 0.2, y + r * 0.1, r * 0.5, 0, Math.PI * 2); ctx.fillStyle = INK; ctx.fill();
}

function icon(ctx: CanvasRenderingContext2D, team: Team) {
  const c = team.colors;
  ctx.lineWidth = 3;
  switch (team.icon) {
    case 'mudcat': {
      ctx.beginPath(); ctx.ellipse(0, 4, 30, 22, 0, 0, Math.PI * 2); fillStroke(ctx, '#8d6e63');
      ctx.beginPath(); ctx.moveTo(-28, -2); ctx.lineTo(-40, -14); ctx.moveTo(-28, 6); ctx.lineTo(-42, 8); ctx.moveTo(28, -2); ctx.lineTo(40, -14); ctx.moveTo(28, 6); ctx.lineTo(42, 8); ctx.stroke();
      eye(ctx, -11, -4, 7); eye(ctx, 11, -4, 7);
      ctx.beginPath(); ctx.arc(0, 10, 12, 0.15, Math.PI - 0.15); ctx.stroke();
      ctx.fillStyle = c.secondary; ctx.beginPath(); ctx.arc(-18, 18, 3, 0, 7); ctx.arc(20, 16, 2.5, 0, 7); ctx.fill();
      break;
    }
    case 'owl': {
      ctx.beginPath(); ctx.ellipse(0, 6, 26, 30, 0, 0, Math.PI * 2); fillStroke(ctx, '#a1887f');
      ctx.beginPath(); ctx.moveTo(-22, -16); ctx.lineTo(-18, -34); ctx.lineTo(-6, -22); ctx.closePath(); fillStroke(ctx, '#a1887f');
      ctx.beginPath(); ctx.moveTo(22, -16); ctx.lineTo(18, -34); ctx.lineTo(6, -22); ctx.closePath(); fillStroke(ctx, '#a1887f');
      ctx.beginPath(); ctx.arc(-11, -4, 11, 0, 7); fillStroke(ctx, c.secondary);
      ctx.beginPath(); ctx.arc(11, -4, 11, 0, 7); fillStroke(ctx, c.secondary);
      eye(ctx, -11, -4, 6); eye(ctx, 11, -4, 6);
      ctx.beginPath(); ctx.moveTo(-4, 6); ctx.lineTo(4, 6); ctx.lineTo(0, 14); ctx.closePath(); fillStroke(ctx, '#f39c12');
      break;
    }
    case 'comet': {
      ctx.fillStyle = c.secondary;
      for (let i = 0; i < 3; i++) {
        ctx.beginPath(); ctx.moveTo(8, -8 + i * 8); ctx.quadraticCurveTo(-20, -24 + i * 14, -40, -30 + i * 18); ctx.lineTo(-30, -16 + i * 14); ctx.closePath(); fillStroke(ctx, i === 1 ? '#fff' : c.secondary);
      }
      ctx.beginPath(); ctx.arc(14, 6, 18, 0, 7); fillStroke(ctx, '#fff');
      ctx.strokeStyle = '#e74c3c'; ctx.beginPath(); ctx.arc(2, 6, 12, -0.9, 0.9); ctx.arc(26, 6, 12, Math.PI - 0.9, Math.PI + 0.9); ctx.stroke();
      ctx.strokeStyle = INK;
      break;
    }
    case 'pinecone': {
      ctx.beginPath(); ctx.ellipse(0, 4, 20, 30, 0, 0, Math.PI * 2); fillStroke(ctx, '#a0522d');
      ctx.strokeStyle = '#5d2e12';
      for (let i = -3; i <= 3; i++) { ctx.beginPath(); ctx.moveTo(-18, i * 8 + 4); ctx.quadraticCurveTo(0, i * 8 + 12, 18, i * 8 + 4); ctx.stroke(); }
      ctx.strokeStyle = INK;
      ctx.beginPath(); ctx.moveTo(-6, -26); ctx.lineTo(-14, -38); ctx.lineTo(0, -30); ctx.lineTo(14, -38); ctx.lineTo(6, -26); ctx.closePath(); fillStroke(ctx, '#27ae60');
      eye(ctx, -8, -4, 5); eye(ctx, 8, -4, 5);
      break;
    }
    case 'frog': {
      ctx.beginPath(); ctx.ellipse(0, 8, 32, 22, 0, 0, Math.PI * 2); fillStroke(ctx, '#7ccf3f');
      ctx.beginPath(); ctx.arc(-14, -12, 11, 0, 7); fillStroke(ctx, '#7ccf3f');
      ctx.beginPath(); ctx.arc(14, -12, 11, 0, 7); fillStroke(ctx, '#7ccf3f');
      eye(ctx, -14, -12, 7); eye(ctx, 14, -12, 7);
      ctx.beginPath(); ctx.arc(0, 8, 18, 0.2, Math.PI - 0.2); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(0, 22, 8, 4, 0, 0, Math.PI * 2); fillStroke(ctx, '#ff7f8f');
      break;
    }
    case 'bee': {
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.ellipse(s * 16, -18, 14, 10, s * 0.5, 0, Math.PI * 2); fillStroke(ctx, 'rgba(255,255,255,0.85)'); }
      ctx.beginPath(); ctx.ellipse(0, 6, 22, 26, 0, 0, Math.PI * 2); fillStroke(ctx, '#f5c518');
      ctx.fillStyle = INK;
      for (const y of [0, 12]) ctx.fillRect(-21, y, 42, 5);
      eye(ctx, -8, -8, 5); eye(ctx, 8, -8, 5);
      ctx.beginPath(); ctx.moveTo(-6, -28); ctx.lineTo(-12, -38); ctx.moveTo(6, -28); ctx.lineTo(12, -38); ctx.stroke();
      break;
    }
    case 'rocket': {
      ctx.save(); ctx.rotate(-0.7);
      ctx.beginPath(); ctx.moveTo(0, -38); ctx.quadraticCurveTo(16, -18, 12, 18); ctx.lineTo(-12, 18); ctx.quadraticCurveTo(-16, -18, 0, -38); fillStroke(ctx, '#ecf0f1');
      ctx.beginPath(); ctx.arc(0, -8, 7, 0, 7); fillStroke(ctx, '#6ec6ff');
      ctx.beginPath(); ctx.moveTo(-12, 8); ctx.lineTo(-22, 24); ctx.lineTo(-10, 18); ctx.closePath(); fillStroke(ctx, c.primary === '#c8312f' ? '#2c3e50' : c.secondary);
      ctx.beginPath(); ctx.moveTo(12, 8); ctx.lineTo(22, 24); ctx.lineTo(10, 18); ctx.closePath(); fillStroke(ctx, c.primary === '#c8312f' ? '#2c3e50' : c.secondary);
      ctx.beginPath(); ctx.moveTo(-8, 18); ctx.quadraticCurveTo(0, 42, 8, 18); fillStroke(ctx, '#f39c12');
      ctx.restore();
      break;
    }
    case 'lightning': {
      ctx.beginPath(); ctx.moveTo(8, -38); ctx.lineTo(-18, 4); ctx.lineTo(-2, 4); ctx.lineTo(-10, 38); ctx.lineTo(20, -8); ctx.lineTo(4, -8); ctx.closePath(); fillStroke(ctx, c.secondary);
      break;
    }
  }
}

/** A logo on its own little canvas, handy for DOM menus. */
export function logoCanvas(team: Team, size: number): HTMLCanvasElement {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const c = document.createElement('canvas');
  c.width = c.height = Math.round(size * dpr);
  c.style.width = c.style.height = `${size}px`;
  c.className = 'logo';
  const ctx = c.getContext('2d')!;
  ctx.scale(dpr, dpr);
  drawLogo(ctx, team, size / 2, size / 2, size * 0.46);
  return c;
}
