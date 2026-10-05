import { INK } from '../data/palette';

// The grown-ups in the background: normal-proportioned adults, which makes
// the big-headed kids look even more like tiny professionals.

const LOOKS = [
  { skin: '#f0c7a0', hair: '#5a3825', shirt: '#e74c3c', pants: '#34495e' },  // the griller
  { skin: '#f5d6bd', hair: '#e8e8e8', shirt: '#9b59b6', pants: '#7f8c8d' },  // grandma in her rocker
  { skin: '#c48a58', hair: '#1f1a17', shirt: '#2980b9', pants: '#2c3e50' },  // worried neighbor
  { skin: '#9a6238', hair: '#2b1d14', shirt: '#27ae60', pants: '#34495e' },  // the cheerer
];

export function drawGrownup(ctx: CanvasRenderingContext2D, x: number, y: number, px: number, variant: number, t: number) {
  if (px * 6 < 6) return;
  const L = LOOKS[variant % LOOKS.length];
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(px, px);
  ctx.lineWidth = Math.max(0.05, 1 / px);
  ctx.strokeStyle = INK;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  const sitting = variant === 1;
  const rock = sitting ? Math.sin(t * 1.6) * 0.08 : 0;
  ctx.fillStyle = 'rgba(0,0,0,0.2)';
  ctx.beginPath(); ctx.ellipse(0, 0, 1.2, 0.3, 0, 0, Math.PI * 2); ctx.fill();
  ctx.rotate(rock);
  if (sitting) {
    // rocking chair
    ctx.strokeStyle = '#7a4f2a'; ctx.lineWidth = 0.18;
    ctx.beginPath(); ctx.moveTo(-1.4, -0.1); ctx.quadraticCurveTo(0, 0.25, 1.4, -0.1); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-0.9, -0.1); ctx.lineTo(-0.9, -4); ctx.moveTo(0.9, -0.1); ctx.lineTo(0.9, -1.8); ctx.stroke();
    ctx.fillStyle = '#a0703c'; ctx.fillRect(-1, -2, 2, 0.25);
    ctx.lineWidth = 0.06; ctx.strokeStyle = INK;
  }
  const hip = sitting ? -2 : -3.1;
  // legs
  ctx.strokeStyle = INK; ctx.lineWidth = 0.5;
  if (sitting) { ctx.beginPath(); ctx.moveTo(-0.3, hip); ctx.lineTo(0.9, hip); ctx.lineTo(0.9, -0.2); ctx.moveTo(0.3, hip); ctx.lineTo(1.3, hip); ctx.lineTo(1.3, -0.2); ctx.stroke(); }
  else { ctx.beginPath(); ctx.moveTo(-0.3, hip); ctx.lineTo(-0.35, -0.1); ctx.moveTo(0.3, hip); ctx.lineTo(0.35, -0.1); ctx.stroke(); }
  ctx.strokeStyle = L.pants; ctx.lineWidth = 0.38;
  if (sitting) { ctx.beginPath(); ctx.moveTo(-0.3, hip); ctx.lineTo(0.9, hip); ctx.lineTo(0.9, -0.2); ctx.moveTo(0.3, hip); ctx.lineTo(1.3, hip); ctx.lineTo(1.3, -0.2); ctx.stroke(); }
  else { ctx.beginPath(); ctx.moveTo(-0.3, hip); ctx.lineTo(-0.35, -0.1); ctx.moveTo(0.3, hip); ctx.lineTo(0.35, -0.1); ctx.stroke(); }
  // torso
  const sh = hip - 2.3;
  ctx.lineWidth = 0.06; ctx.strokeStyle = INK;
  ctx.beginPath(); ctx.moveTo(-0.65, sh); ctx.lineTo(0.65, sh); ctx.lineTo(0.55, hip); ctx.lineTo(-0.55, hip); ctx.closePath();
  ctx.fillStyle = L.shirt; ctx.fill(); ctx.stroke();
  if (variant === 0) { ctx.fillStyle = '#ffffff'; ctx.fillRect(-0.45, sh + 0.4, 0.9, 2); ctx.strokeRect(-0.45, sh + 0.4, 0.9, 2); }
  // arms
  const arm = (x0: number, x1: number, y1: number) => {
    ctx.strokeStyle = INK; ctx.lineWidth = 0.34; ctx.beginPath(); ctx.moveTo(x0, sh + 0.2); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.strokeStyle = L.skin; ctx.lineWidth = 0.24; ctx.beginPath(); ctx.moveTo(x0, sh + 0.2); ctx.lineTo(x1, y1); ctx.stroke();
  };
  if (variant === 0) { // flipping burgers with the spatula
    const f = Math.sin(t * 3) * 0.3;
    arm(-0.6, -0.9, sh + 1.8); arm(0.6, 1.2, sh + 0.6 + f);
    ctx.strokeStyle = '#7f8c8d'; ctx.lineWidth = 0.1; ctx.beginPath(); ctx.moveTo(1.2, sh + 0.6 + f); ctx.lineTo(1.8, sh - 0.2 + f); ctx.stroke();
  } else if (variant === 2) { // arms crossed, worried about the lawn
    ctx.strokeStyle = L.shirt; ctx.lineWidth = 0.35; ctx.beginPath(); ctx.moveTo(-0.6, sh + 0.9); ctx.lineTo(0.6, sh + 1.1); ctx.stroke();
  } else if (variant === 3) { // cheering
    const w = Math.sin(t * 8) * 0.3;
    arm(-0.6, -1.1 + w, sh - 1.4); arm(0.6, 1.1 - w, sh - 1.4);
  } else {
    arm(-0.6, -0.6, hip + 0.1); arm(0.6, 0.6, hip + 0.1);
  }
  // head (normal sized — the joke is the kids' giant heads)
  const hy = sh - 0.75;
  ctx.lineWidth = 0.06; ctx.strokeStyle = INK;
  ctx.beginPath(); ctx.arc(0, hy, 0.62, 0, Math.PI * 2); ctx.fillStyle = L.skin; ctx.fill(); ctx.stroke();
  ctx.fillStyle = L.hair;
  if (variant === 1) { ctx.beginPath(); ctx.arc(0, hy - 0.35, 0.55, Math.PI, 0); ctx.fill(); ctx.beginPath(); ctx.arc(0, hy - 0.8, 0.28, 0, Math.PI * 2); ctx.fill(); }
  else { ctx.beginPath(); ctx.arc(0, hy - 0.2, 0.64, Math.PI * 1.05, Math.PI * 1.95); ctx.fill(); }
  ctx.fillStyle = INK;
  ctx.beginPath(); ctx.arc(-0.2, hy - 0.02, 0.06, 0, Math.PI * 2); ctx.arc(0.2, hy - 0.02, 0.06, 0, Math.PI * 2); ctx.fill();
  if (variant === 1) { ctx.strokeStyle = '#555'; ctx.lineWidth = 0.05; ctx.beginPath(); ctx.arc(-0.2, hy, 0.15, 0, Math.PI * 2); ctx.arc(0.2, hy, 0.15, 0, Math.PI * 2); ctx.stroke(); }
  ctx.strokeStyle = INK; ctx.lineWidth = 0.06;
  ctx.beginPath();
  if (variant === 2) ctx.arc(0, hy + 0.4, 0.18, Math.PI * 1.2, Math.PI * 1.8);
  else ctx.arc(0, hy + 0.12, 0.2, Math.PI * 0.2, Math.PI * 0.8);
  ctx.stroke();
  ctx.restore();
}
