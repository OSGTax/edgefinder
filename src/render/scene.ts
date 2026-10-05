import type { Vec3 } from '../engine/math';
import { INK } from '../data/palette';
import type { Kid, Team } from '../data/types';
import { drawKid, type KidPose } from '../art/kid';
import { drawGround, drawSky, yardDrawables, type Drawable } from '../art/yard';
import type { Field } from '../sim/field';
import { Camera } from './camera';

export interface KidSprite {
  kid: Kid;
  team: Team | null;
  x: number;
  y: number;
  /** world facing angle (radians, 0 = toward +y / center field) */
  facing: number;
  pose: Omit<KidPose, 'view'> & { view?: KidPose['view'] };
  /** extra highlight ring (the kid you control) */
  ring?: string;
}

export interface Popup {
  text: string;
  color: string;
  t: number;
  dur: number;
  size: number;
  /** screen position as a fraction of the viewport */
  fx: number;
  fy: number;
}

export interface Puff { x: number; y: number; z: number; t: number; dur: number; r: number; color: string }

export interface SceneFrame {
  kids: KidSprite[];
  ball: Vec3 | null;
  /** draw the ball bigger than life so it reads on a phone */
  ballScale?: number;
  t: number;
  /** extra screen-space drawing after the 3D scene (zone, reticle, markers) */
  overlay?: (ctx: CanvasRenderingContext2D, cam: Camera) => void;
}

export class SceneRenderer {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  readonly cam = new Camera();
  field: Field | null = null;
  popups: Popup[] = [];
  puffs: Puff[] = [];
  dpr = 1;
  cssW = 800;
  cssH = 450;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
  }

  resize(w: number, h: number) {
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.cssW = w;
    this.cssH = h;
    this.canvas.width = Math.round(w * this.dpr);
    this.canvas.height = Math.round(h * this.dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.cam.setViewport(w, h);
  }

  popup(text: string, color = '#ffffff', size = 1, dur = 1.2, fx = 0.5, fy = 0.36) {
    // stack on top of anything still showing near the same spot
    const busy = this.popups.filter((p) => p.t < p.dur * 0.7 && Math.abs(p.fx - fx) < 0.2).length;
    this.popups.push({ text, color, t: 0, dur, size, fx, fy: fy + busy * 0.12 });
  }

  puff(x: number, y: number, z = 0, r = 1.5, color = 'rgba(190,150,100,0.6)') {
    this.puffs.push({ x, y, z, t: 0, dur: 0.6, r, color });
  }

  draw(frame: SceneFrame, dt: number) {
    const ctx = this.ctx;
    const f = this.field;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.cam.update();
    const cam = this.cam;
    if (!f) {
      ctx.fillStyle = '#5fae3c';
      ctx.fillRect(0, 0, this.cssW, this.cssH);
      return;
    }
    drawSky(ctx, cam, f.yard, frame.t);
    drawGround(ctx, cam, f, frame.t);

    // ball shadow lies on the ground under everything else
    if (frame.ball) {
      const sh = cam.project(frame.ball.x, frame.ball.y, 0);
      if (sh) {
        const r = Math.max(2, 0.35 * sh.s) * (1 / (1 + frame.ball.z * 0.04));
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.beginPath();
        ctx.ellipse(sh.x, sh.y, r * 1.2, r * 0.5, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const items: Drawable[] = yardDrawables(cam, f, frame.t);
    const fwd = { x: cam.pose.target.x - cam.pose.pos.x, y: cam.pose.target.y - cam.pose.pos.y };
    for (const s of frame.kids) {
      const p = cam.project(s.x, s.y, 0);
      if (!p) continue;
      const fx = Math.sin(s.facing), fy = Math.cos(s.facing);
      const facingCam = fx * fwd.x + fy * fwd.y < 0;
      const pose: KidPose = { ...s.pose, view: s.pose.view ?? (facingCam ? 'front' : 'back') } as KidPose;
      items.push({
        z: p.z,
        draw: (c) => {
          if (s.ring) {
            c.strokeStyle = s.ring;
            c.lineWidth = Math.max(2, 0.25 * p.s);
            c.beginPath();
            c.ellipse(p.x, p.y, 1.6 * p.s, 0.55 * p.s, 0, 0, Math.PI * 2);
            c.stroke();
          }
          drawKid(c, s.kid, s.team, p.x, p.y, p.s, pose);
        },
      });
    }
    if (frame.ball) {
      const b = frame.ball;
      const p = cam.project(b.x, b.y, b.z);
      if (p) {
        items.push({
          z: p.z - 0.5,
          draw: (c) => {
            const r = Math.max(2.5, 0.12 * (frame.ballScale ?? 2.4) * p.s);
            c.fillStyle = '#ffffff';
            c.strokeStyle = INK;
            c.lineWidth = Math.max(1, r * 0.22);
            c.beginPath();
            c.arc(p.x, p.y, r, 0, Math.PI * 2);
            c.fill();
            c.stroke();
            if (r > 5) {
              c.strokeStyle = '#e74c3c';
              c.lineWidth = Math.max(1, r * 0.14);
              c.beginPath();
              c.arc(p.x - r * 0.9, p.y, r * 0.7, -0.9, 0.9);
              c.arc(p.x + r * 0.9, p.y, r * 0.7, Math.PI - 0.9, Math.PI + 0.9);
              c.stroke();
            }
          },
        });
      }
    }
    items.sort((a, b) => b.z - a.z);
    for (const it of items) it.draw(ctx);

    // dust puffs
    for (const pf of this.puffs) {
      pf.t += dt;
      const p = cam.project(pf.x, pf.y, pf.z + pf.t * 1.5);
      if (!p) continue;
      const u = pf.t / pf.dur;
      ctx.fillStyle = pf.color.replace(/[\d.]+\)$/, `${(1 - u) * 0.6})`);
      ctx.beginPath();
      ctx.arc(p.x, p.y, pf.r * (0.6 + u) * p.s, 0, Math.PI * 2);
      ctx.fill();
    }
    this.puffs = this.puffs.filter((p) => p.t < p.dur);

    frame.overlay?.(ctx, cam);
    this.drawPopups(dt);
  }

  private drawPopups(dt: number) {
    const ctx = this.ctx;
    for (const p of this.popups) {
      p.t += dt;
      const u = p.t / p.dur;
      const pop = u < 0.15 ? 0.4 + (u / 0.15) * 0.8 : u < 0.25 ? 1.2 - ((u - 0.15) / 0.1) * 0.2 : 1;
      const alpha = u > 0.75 ? 1 - (u - 0.75) / 0.25 : 1;
      const base = Math.min(this.cssW, this.cssH * 1.6) * 0.085 * p.size;
      ctx.save();
      ctx.globalAlpha = Math.max(0, alpha);
      ctx.translate(this.cssW * p.fx, this.cssH * p.fy - u * 12);
      ctx.scale(pop, pop);
      ctx.rotate(-0.06);
      ctx.font = `900 ${base}px "Arial Rounded MT Bold", "Trebuchet MS", "Arial Black", sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.lineJoin = 'round';
      ctx.lineWidth = base * 0.22;
      ctx.strokeStyle = INK;
      ctx.strokeText(p.text, 0, 0);
      ctx.fillStyle = p.color;
      ctx.fillText(p.text, 0, 0);
      ctx.restore();
    }
    this.popups = this.popups.filter((p) => p.t < p.dur);
  }
}
