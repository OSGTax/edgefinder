import {
  Color, DirectionalLight, HemisphereLight, PerspectiveCamera, Scene, Vector2, Vector3, Vector4, type WebGLRenderer,
} from 'three';
import type { KidModel } from '../kid3d/model';
import type { Expression } from '../kid3d/face';
import { headCentre } from '../kid3d/model';

// Head-and-shoulders portraits of the real 3D kids, rendered with the game's
// renderer into a corner of the canvas and copied out to small images for
// the HUD and menus. Shaders compile asynchronously, so a request waits for
// its kid's materials to be ready before the picture is taken.

interface Job { model: KidModel; team: { colors: { primary: string; secondary: string } }; expr: Expression; size: number; key: string; imgs: HTMLImageElement[]; state: 'new' | 'compiling' | 'ready' }

export class PortraitStudio {
  private scene = new Scene();
  private cam = new PerspectiveCamera(26, 1, 0.1, 50);
  private cache = new Map<string, string>();
  private jobs: Job[] = [];
  private compiled = new Set<string>();

  constructor(private renderer: WebGLRenderer, env: Scene['environment']) {
    this.scene.environment = env;
    this.scene.environmentIntensity = 0.7;
    const key = new DirectionalLight('#fff4e6', 2.6);
    key.position.set(2.5, 4, 5);
    const rim = new DirectionalLight('#cfe6ff', 1.6);
    rim.position.set(-4, 3, -3);
    this.scene.add(key, rim, new HemisphereLight('#dff0ff', '#6d8a4a', 0.8));
  }

  /** Fill an <img> with a portrait (now if cached, otherwise in a frame or two). */
  into(img: HTMLImageElement, model: KidModel, team: Job['team'], expr: Expression = 'happy', size = 160) {
    const key = `${model.kid.id}|${expr}|${size}`;
    const hit = this.cache.get(key);
    if (hit) { img.src = hit; return; }
    img.style.background = team.colors.secondary;
    const job = this.jobs.find((j) => j.key === key);
    if (job) job.imgs.push(img);
    else this.jobs.push({ model, team, expr, size, key, imgs: [img], state: this.compiled.has(model.kid.id) ? 'ready' : 'new' });
  }

  /** Call once per frame (before the main render): compiles new kids, takes a few pictures. */
  update() {
    for (const job of this.jobs) {
      if (job.state !== 'new') continue;
      if (this.compiled.has(job.model.kid.id)) { job.state = 'ready'; continue; }
      job.state = 'compiling';
      const restore = this.borrow(job.model, job.expr);
      this.renderer.compileAsync(this.scene, this.cam).then(() => {
        this.compiled.add(job.model.kid.id);
        for (const j of this.jobs) if (j.model === job.model && j.state === 'compiling') j.state = 'ready';
      }).catch(() => { job.state = 'ready'; });
      restore();
    }
    for (let n = 0; n < 3; n++) {
      const job = this.jobs.find((j) => j.state === 'ready');
      if (!job) return;
      const url = this.capture(job);
      this.cache.set(job.key, url);
      for (const img of job.imgs) img.src = url;
      this.jobs.splice(this.jobs.indexOf(job), 1);
    }
  }

  /** Move the model into the studio in a neutral standing pose; returns a function that puts it back. */
  private borrow(model: KidModel, expr: Expression): () => void {
    const g = model.group;
    const parent = g.parent;
    const saved = { pos: g.position.clone(), rot: g.rotation.y, quats: Object.values(model.bones).map((b) => b.quaternion.clone()), hip: model.bones.hips.position.clone(), expr: model.expression };
    for (const b of Object.values(model.bones)) b.quaternion.identity();
    model.bones.hips.position.copy(model.p.joints.hips);
    model.bones.head.rotation.set(0.05, -0.22, 0.04);
    model.bones.lidL.rotation.x = model.bones.lidR.rotation.x = -0.6;
    model.bones.armL.rotation.z = 0.15;
    model.bones.armR.rotation.z = -0.15;
    model.setExpression(expr);
    g.position.set(0, 0, 0);
    g.rotation.y = 0.35;
    this.scene.add(g);
    g.updateMatrixWorld(true);
    const hc = headCentre(model.p);
    const target = new Vector3(hc.x, hc.y - model.p.headR * 0.45, hc.z);
    this.cam.position.set(target.x + 0.9, target.y + 0.35, target.z + 4.6);
    this.cam.lookAt(target);
    this.cam.aspect = 1;
    this.cam.updateProjectionMatrix();
    return () => {
      this.scene.remove(g);
      if (parent) parent.add(g);
      g.position.copy(saved.pos);
      g.rotation.y = saved.rot;
      Object.values(model.bones).forEach((b, i) => b.quaternion.copy(saved.quats[i]));
      model.bones.hips.position.copy(saved.hip);
      model.setExpression(saved.expr);
      g.updateMatrixWorld(true);
    };
  }

  private capture(job: Job): string {
    const r = this.renderer;
    const restore = this.borrow(job.model, job.expr);
    this.scene.background = new Color(job.team.colors.secondary);
    const size = job.size;
    const pr = r.getPixelRatio();
    const css = size / pr;
    const buf = r.getDrawingBufferSize(new Vector2());
    const oldVp = r.getViewport(new Vector4()), oldSc = r.getScissor(new Vector4()), oldTest = r.getScissorTest();
    r.setViewport(0, 0, css, css);
    r.setScissor(0, 0, css, css);
    r.setScissorTest(true);
    // the first draw after the main scene can inherit stale depth state: draw twice
    r.state.buffers.depth.setMask(true);
    r.render(this.scene, this.cam);
    r.render(this.scene, this.cam);
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(r.domElement, 0, buf.y - size, size, size, 0, 0, size, size);
    const url = c.toDataURL('image/png');
    r.setViewport(oldVp);
    r.setScissor(oldSc);
    r.setScissorTest(oldTest);
    restore();
    return url;
  }
}
