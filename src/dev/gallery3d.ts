import { Color, Mesh, MeshStandardMaterial, PerspectiveCamera, PlaneGeometry, Scene } from 'three';
import { KIDS } from '../data/kids';
import { TEAMS } from '../data/teams';
import { createRenderer } from '../gfx/renderer';
import { Environment } from '../gfx/environment';
import { getQuality } from '../gfx/quality';
import { KidModel } from '../kid3d/model';
import { EXPRESSIONS } from '../kid3d/face';
import { Animator, type Mode } from '../kid3d/anim';
import { makeBat, makeGlove, makeBall } from '../kid3d/items';
import { Clock, Vector3, type Mesh as MeshT, type Group as GroupT } from 'three';

/** Dev-only: every kid in a lineup, for checking the character art. */
export function devGallery(root: HTMLElement, opts: URLSearchParams) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%';
  root.appendChild(canvas);
  const q = getQuality();
  const r = createRenderer(canvas, q);
  r.setSize(window.innerWidth, window.innerHeight, false);
  const scene = new Scene();
  scene.background = new Color('#9cc7e8');
  const env = new Environment(scene, r, q, { elevation: 40, azimuth: -30 });
  env.sun.shadow.camera.left = -40; env.sun.shadow.camera.right = 40; env.sun.shadow.camera.top = 40; env.sun.shadow.camera.bottom = -40;
  env.sun.target.position.set(0, 0, 0);
  env.sun.position.copy(env.sunDir).multiplyScalar(300);
  env.sun.shadow.camera.updateProjectionMatrix();
  const floor = new Mesh(new PlaneGeometry(200, 200), new MeshStandardMaterial({ color: '#6aa84f', roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const only = opts.get('kid');
  const list = only ? KIDS.filter((k) => k.id === only) : KIDS;
  const expr = opts.get('expr');
  const models: KidModel[] = [];
  list.forEach((k, i) => {
    const team = TEAMS.find((t) => t.roster.includes(k.id)) ?? TEAMS[0];
    const m = new KidModel(k, team);
    const row = Math.floor(i / 9), col = i % 9;
    if (opts.has('faces') && row > 0) { m.group.visible = false; }
    m.group.position.set(only ? 0 : (col - 4) * 2.4, 0, -row * 4);
    if (opts.has('back')) m.group.rotation.y = Math.PI;
    if (expr) m.setExpression(expr as (typeof EXPRESSIONS)[number]);
    scene.add(m.group);
    models.push(m);
  });
  const cam = new PerspectiveCamera(only ? 22 : 30, window.innerWidth / window.innerHeight, 0.1, 500);
  if (only) { cam.position.set(0, 4.0, 6.5); cam.lookAt(0, 3.6, 0); }
  else if (opts.has('faces')) { cam.position.set(0, 4.3, 14); cam.lookAt(0, 3.9, 0); cam.fov = 30; cam.updateProjectionMatrix(); }
  else { cam.position.set(0, 7, 28); cam.lookAt(0, 2.4, -2); }
  (window as unknown as { __models: KidModel[] }).__models = models;
  // pose test: each kid gets a mode (cycling through the list), a glove and a bat
  const poseModes = opts.get('poses')?.split(',') as Mode[] | undefined;
  const anims: { a: Animator; mode: Mode; bat: MeshT; glove: GroupT; t0: number }[] = [];
  if (poseModes) {
    models.forEach((m, i) => {
      const a = new Animator(m);
      const mode = poseModes[i % poseModes.length];
      const bat = makeBat();
      bat.visible = false;
      scene.add(bat);
      const glove = makeGlove(m.p.s);
      const lefty = m.kid.throws === 'L';
      (lefty ? m.bones.handR : m.bones.handL).add(glove);
      glove.rotation.y = lefty ? Math.PI / 2 : -Math.PI / 2;
      glove.position.y = -0.05;
      if (mode === 'bat' || mode === 'swing' || mode === 'bunt') glove.visible = false;
      if (mode === 'windup') { const ball = makeBall(); (lefty ? m.bones.handL : m.bones.handR).add(ball); ball.position.set(0, -0.25, 0.05); }
      anims.push({ a, mode, bat, glove, t0: Number(opts.get('t') ?? 0) });
    });
  }
  const clock = new Clock();
  let T = 0;
  const loop = () => {
    const dt = Math.min(0.05, clock.getDelta());
    T += dt;
    const freeze = opts.has('t');
    for (const an of anims) {
      const lefty = an.a.kid.kid.throws === 'L';
      const t = freeze ? an.t0 : (T % 1.6);
      // advance in small steps so the eased pose settles for a frozen frame
      const steps = freeze ? 30 : 1;
      for (let i = 0; i < steps; i++) an.a.update(freeze ? 0.05 : dt, { mode: an.mode, t, speed: 20, lefty, lookAt: new Vector3(0, 4, 30), windup: 0.8 });
      an.bat.visible = an.a.batActive;
      if (an.a.batActive) {
        an.bat.position.copy(an.a.batHandle);
        an.bat.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), an.a.batDir);
      }
    }
    r.render(scene, cam);
    (window as unknown as { __ready: boolean }).__ready = true;
    requestAnimationFrame(loop);
  };
  loop();
}
