import type { WebGLRenderer } from 'three';
import { shortGpuName } from './quality';

// A small speed readout in the corner (Settings → Speed readout): frames per
// second, the graphics tier and the graphics chip the browser reports, so a
// player can send real numbers from any device.

export class PerfReadout {
  readonly el: HTMLDivElement;
  private frames = 0;
  private t = 0;
  private worst = 0;

  constructor(private gpu: string) {
    this.el = document.createElement('div');
    this.el.className = 'perf-readout';
    this.el.style.cssText = [
      'position:fixed', 'left:50%', 'transform:translateX(-50%)', 'top:calc(2px + env(safe-area-inset-top, 0px))', 'text-align:center',
      'z-index:50', 'pointer-events:none', 'font:600 10px/1.3 ui-monospace,Menlo,Consolas,monospace', 'color:#fffbe8',
      'background:rgba(30,26,20,.62)', 'padding:3px 7px', 'border-radius:4px', 'white-space:pre', 'display:none',
    ].join(';');
    document.body.appendChild(this.el);
  }

  set visible(v: boolean) { this.el.style.display = v ? 'block' : 'none'; }
  get visible() { return this.el.style.display !== 'none'; }

  /** Count one rendered frame; refreshes the text twice a second. */
  frame(dtMs: number, r: WebGLRenderer, tier: string, scale: number) {
    if (!this.visible) return;
    this.frames++;
    this.t += dtMs;
    this.worst = Math.max(this.worst, dtMs);
    if (this.t < 500) return;
    const fps = (this.frames * 1000) / this.t;
    const px = r.getPixelRatio();
    this.el.textContent = `${fps.toFixed(0)} fps · worst ${this.worst.toFixed(0)} ms · ${tier}${scale < 1 ? ` @${Math.round(scale * 100)}%` : ''} · ${px.toFixed(2)}x\n`
      + `${r.info.render.calls} draws · ${Math.round(r.info.render.triangles / 1000)}k tris · ${shortGpuName(this.gpu) || 'unknown GPU'}`;
    this.frames = 0;
    this.t = 0;
    this.worst = 0;
  }

  dispose() { this.el.remove(); }
}
