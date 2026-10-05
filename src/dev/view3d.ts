import { PerspectiveCamera, Scene } from 'three';
import { yard } from '../data/yards';
import { buildField } from '../sim/field';
import { createRenderer } from '../gfx/renderer';
import { Stadium } from '../world/stadium';
import { getQuality } from '../gfx/quality';
import { W } from '../gfx/units';

/** Dev-only: render the stadium from a named camera for screenshots. */
export function devView(root: HTMLElement, camName: string) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%';
  root.appendChild(canvas);
  const q = getQuality();
  const r = createRenderer(canvas, q);
  r.setSize(window.innerWidth, window.innerHeight, false);
  const scene = new Scene();
  const field = buildField(yard('poolparty'));
  const tb = performance.now();
  const stadium = new Stadium(scene, r, field, q);
  console.warn(`stadium built in ${Math.round(performance.now() - tb)} ms`);
  const cam = new PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.3, 12000);
  const views: Record<string, [number[], number[], number]> = {
    bat: [[3, -17.5, 12.5], [-0.4, 17, 0], 42],
    behind: [[0, -9, 5.5], [0, 30, 3], 50],
    field: [[0, -70, 60], [0, 80, 0], 52],
    high: [[60, -60, 140], [0, 70, 0], 55],
    low: [[30, 20, 4], [0, 90, 2], 55],
    over: [[90, 240, 70], [0, 40, 0], 50],
    house: [[16, 48, 10], [-12, -28, 11], 55],
    patio: [[-20, 8, 6], [-38, -22, 5], 60],
    pool: [[20, 70, 14], [62, 108, 0], 55],
    lf: [[-20, 60, 8], [-100, 120, 6], 60],
    dugout: [[-20, 30, 7], [-47, 13, 3], 55],
    plate: [[0, -6, 3], [0, 2, 0], 60],
    street: [[60, -40, 50], [-10, -120, 0], 60],
    cf: [[-10, 60, 14], [0, 260, 10], 55],
  };
  const setCam = (name: string) => {
    const [p, t, fov] = views[name] ?? views.field;
    cam.position.copy(W(p[0], p[1], p[2]));
    cam.fov = fov;
    cam.updateProjectionMatrix();
    cam.lookAt(W(t[0], t[1], t[2]));
  };
  setCam(camName);
  (window as unknown as { __setCam: (n: string) => void }).__setCam = setCam;
  (window as unknown as { __scene: Scene }).__scene = scene;
  const t0 = performance.now();
  let last = 0;
  const loop = () => {
    const tt = (performance.now() - t0) / 1000;
    stadium.update(tt, tt - last);
    last = tt;
    (window as unknown as { __ready: boolean }).__ready = true;
    r.render(scene, cam);
    requestAnimationFrame(loop);
  };
  loop();
}
