import { Color, Mesh, MeshStandardMaterial, PerspectiveCamera, PlaneGeometry, Scene } from 'three';
import { KIDS } from '../data/kids';
import { TEAMS } from '../data/teams';
import { createRenderer } from '../gfx/renderer';
import { Environment } from '../gfx/environment';
import { getQuality } from '../gfx/quality';
import { KidModel } from '../kid3d/model';
import { hawaiianShirt } from '../kid3d/outfits';
import { MR_MENDOZA } from '../world/grownups';
import { EXPRESSIONS } from '../kid3d/face';
import { Animator, type Mode } from '../kid3d/anim';
import { makeBat, makeGlove, makeBall } from '../kid3d/items';
import { Clock, Vector3, type Mesh as MeshT, type Group as GroupT } from 'three';
import { PortraitStudio } from '../game/portraits';
import type { Expression } from '../kid3d/face';

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
  let list = only === MR_MENDOZA.id ? [MR_MENDOZA] : only ? KIDS.filter((k) => k.id === only) : KIDS;
  // &exprs (with &kid=): the same kid eight times, one per expression, as a contact sheet
  const exprSheet = opts.has('exprs') && list.length === 1;
  if (exprSheet) list = EXPRESSIONS.map(() => list[0]);
  const expr = opts.get('expr');
  const models: KidModel[] = [];
  list.forEach((k, i) => {
    const team = TEAMS.find((t) => t.roster.includes(k.id)) ?? TEAMS[0];
    const m = k === MR_MENDOZA
      ? new KidModel(k, TEAMS[1], { outfit: { shirt: hawaiianShirt('#1f8a8a'), colors: { pants: '#c8b48a', trim: '#1f8a8a', jersey: '#1f8a8a', socks: '#f4f4f0', sockStripe: '#f4f4f0' } } })
      : new KidModel(k, team);
    const row = Math.floor(i / 9), col = i % 9;
    if (opts.has('faces') && row > 0) { m.group.visible = false; }
    m.group.position.set(only ? 0 : (col - 4) * 2.4, 0, -row * 4);
    if (opts.has('grid')) {
      // heads-and-shoulders contact sheet: 6 × 3, every head centre on a grid point
      const gc = i % 6, gr = Math.floor(i / 6);
      m.group.position.set((gc - 2.5) * 2.4, 20 - gr * 3.3 - (m.p.joints.head.y + m.p.headR * 0.92), gr * 3);
    }
    // &lite: the far-away detail level
    if (opts.has('lite')) m.setDetail('lite');
    if (opts.has('back')) m.group.rotation.y = Math.PI;
    if (opts.has('yaw')) m.group.rotation.y = Number(opts.get('yaw'));
    if (expr) m.setExpression(expr as (typeof EXPRESSIONS)[number]);
    if (exprSheet) {
      m.setExpression(EXPRESSIONS[i]);
      m.group.position.set(((i % 5) - 2) * 1.85, 20 - Math.floor(i / 5) * 2.1 - (m.p.joints.head.y + m.p.headR * 0.92), 0);
    }
    // &hide=face,hair,... hides those part meshes (by material name prefix) for debugging
    for (const h of opts.get('hide')?.split(',') ?? []) for (const me of m.meshes) if ((me.material as { name: string }).name.toLowerCase().includes(h)) me.visible = false;
    scene.add(m.group);
    models.push(m);
  });
  const cam = new PerspectiveCamera(only ? 22 : 30, window.innerWidth / window.innerHeight, 0.1, 500);
  if (exprSheet) { floor.visible = false; cam.fov = 4.4; cam.position.set(0, 18.95, 70); cam.lookAt(0, 18.95, 0); cam.aspect = window.innerWidth / window.innerHeight; cam.far = 1000; cam.updateProjectionMatrix(); }
  else if (opts.has('grid')) { floor.visible = false; cam.fov = 6.4; cam.position.set(0, 15.4, 150); cam.lookAt(0, 15.4, 0); cam.far = 1000; cam.updateProjectionMatrix(); }
  else if (only && models[0] && !exprSheet) {
    // aim at the head (or the whole kid with &body); &zoom=2 moves in, &yaw= turns the kid
    const m = models[0];
    const z = Number(opts.get('zoom') ?? 1);
    const ty = opts.has('body') ? m.p.H * 0.5 : m.p.joints.head.y + m.p.headR * 0.8;
    const dist = (opts.has('body') ? 13 : 6.5) / z;
    cam.position.set(0, ty + dist * 0.06, dist);
    cam.lookAt(0, ty, 0);
  }
  else if (opts.has('faces')) { cam.position.set(0, 4.3, 14); cam.lookAt(0, 3.9, 0); cam.fov = 30; cam.updateProjectionMatrix(); }
  else { cam.position.set(0, 7, 28); cam.lookAt(0, 2.4, -2); }
  // &cam=x,y,z,tx,ty,tz overrides the camera
  const cv = opts.get('cam')?.split(',').map(Number);
  if (cv && cv.length === 6) { cam.position.set(cv[0], cv[1], cv[2]); cam.lookAt(cv[3], cv[4], cv[5]); }
  (window as unknown as { __models: KidModel[] }).__models = models;
  // &atlas: show the first kid's painted face atlas (all expressions) over the scene
  if (opts.has('atlas') && models[0]) {
    const src = models[0].faceMat.map!.image as HTMLCanvasElement;
    src.style.cssText = 'position:fixed;left:0;top:0;width:100%;background:#d9b08c;z-index:5';
    root.appendChild(src);
  }
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
  // &portraits: every kid's menu portrait (152 px) and HUD portrait (112 px shown at 56), as the game makes them
  let studio: PortraitStudio | null = null;
  if (opts.has('portraits')) {
    studio = new PortraitStudio(r, scene.environment);
    const wrap = document.createElement('div');
    wrap.style.cssText = 'position:fixed;inset:0;display:flex;flex-wrap:wrap;gap:8px;padding:8px;background:#efe6d2;z-index:5;align-content:flex-start';
    root.appendChild(wrap);
    const pe = (opts.get('pexpr') ?? 'happy') as Expression;
    for (const m of models) {
      const team = TEAMS.find((t) => t.roster.includes(m.kid.id)) ?? TEAMS[1];
      const big = document.createElement('img'), small = document.createElement('img');
      big.width = big.height = 76; small.width = small.height = 28;
      big.style.borderRadius = small.style.borderRadius = '8px';
      studio.into(big, m, team, pe, 152);
      studio.into(small, m, team, pe, 56);
      wrap.append(big, small);
    }
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
    studio?.update();
    r.render(scene, cam);
    (window as unknown as { __ready: boolean }).__ready = !studio || !!(window as unknown as { __shots?: boolean }).__shots || [...document.images].every((im) => im.src);
    requestAnimationFrame(loop);
  };
  loop();
}
