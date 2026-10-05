import { KIDS } from '../data/kids';
import { TEAMS } from '../data/teams';
import { drawKid, drawPortrait, type KidAnim } from '../art/kid';

/** Dev-only sheet of every kid, for eyeballing the procedural art. */
export function showGallery(root: HTMLElement) {
  const c = document.createElement('canvas');
  const cols = 12;
  const cw = 130, ch = 170;
  c.width = cols * cw;
  c.height = Math.ceil(KIDS.length / cols) * ch + 420;
  root.appendChild(c);
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#8fd18a';
  ctx.fillRect(0, 0, c.width, c.height);
  const teamOf = (id: string) => TEAMS.find((t) => t.roster.includes(id)) ?? null;
  KIDS.forEach((k, i) => {
    const x = (i % cols) * cw, y = Math.floor(i / cols) * ch;
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, cw, ch); ctx.clip();
    drawKid(ctx, k, teamOf(k.id), x + cw / 2, y + ch - 22, 26, { anim: 'idle', t: 1, view: 'front', prop: true });
    ctx.fillStyle = '#222'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(k.nick, x + cw / 2, y + ch - 6);
    ctx.restore();
  });
  const base = Math.ceil(KIDS.length / cols) * ch;
  const anims: KidAnim[] = ['ready', 'run', 'bat', 'pitch', 'throw', 'catch', 'dive', 'jump', 'cheer', 'out', 'stumble', 'crouch'];
  anims.forEach((a, i) => {
    const k = KIDS[i * 5];
    drawKid(ctx, k, teamOf(k.id), 70 + i * 120, base + 180, 24, { anim: a, t: 0.6, view: i % 3 === 2 ? 'back' : 'front', swing: 0.5, windup: 0.8 });
    ctx.fillStyle = '#222'; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(a, 70 + i * 120, base + 200);
  });
  for (let i = 0; i < 6; i++) {
    const k = KIDS[i * 11];
    ctx.save();
    ctx.translate(40 + i * 230, base + 220);
    ctx.fillStyle = '#fff3d6'; ctx.fillRect(0, 0, 200, 190);
    ctx.beginPath(); ctx.rect(0, 0, 200, 190); ctx.clip();
    drawPortrait(ctx, k, teamOf(k.id), 200, 190, 1);
    ctx.restore();
  }
}
