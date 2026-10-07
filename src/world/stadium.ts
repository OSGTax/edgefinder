import { Group, Vector3, type Mesh, type Object3D, type Scene, type WebGLRenderer } from 'three';
import type { Field } from '../sim/field';
import { Environment } from '../gfx/environment';
import { Ground } from '../gfx/ground';
import { Batch, T, boxFt, cyl } from '../gfx/build';
import { M, setMaterialTexSize } from '../gfx/materials';
import type { Quality } from '../gfx/quality';
import type { Step, Steps } from '../engine/steps';
import { W } from '../gfx/units';
import { TEAMS } from '../data/teams';
import { buildHouse, FACADE, MENDOZA_STYLE } from './house';
import { buildPool, poolDeckPoly, type PoolSpec } from './pool';
import { buildHedge, buildPicketFence } from './fences';
import { buildTrees, type TreeSpec } from './trees';
import { buildNeighborhood, NEIGHBOR_TREES } from './neighborhood';
import {
  bases, bike, cooler, dugout, flowers, gardenHose, gasGrill, gnome, homePlate, lawnChair, lawnFlamingo, patioSet,
  rubber, shrubs, stringLights,
} from './props';

const rect = (cx: number, cy: number, hw: number, hd: number, rot = 0): [number, number][] => {
  const c = Math.cos(rot), s = Math.sin(rot);
  return [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].map(([x, y]) => [cx + x * c - y * s, cy + x * s + y * c]);
};

/** Where things are in the Mendozas' yard (sim coordinates). */
export const LAYOUT = {
  patio: { x0: -54, x1: -14, y0: -28, y1: -10 },
  patioSet: { x: -27, y: -19 },
  grill: { x: -46, y: -16 },
  dugouts: [
    { x: -47, y: 15, rot: 1.94 },  // third-base side
    { x: 47, y: 15, rot: -1.94 },  // first-base side
  ],
};

/** Everything static (and gently animated) about the ballpark. */
export class Stadium {
  readonly group = new Group();
  env!: Environment;
  ground!: Ground;
  /** where Mr. Mendoza's grill smoke comes out (three space) */
  grillTop = new Vector3();
  private updaters: ((t: number, dt: number) => void)[] = [];
  /** shadow casters that can go without on the Fast tier (neighbours' houses, the hedge) */
  readonly farCasters: Object3D[] = [];
  private casts = new Map<Mesh, boolean>();

  /** Fast tier: skip the shadow pass for scenery away from the playfield. */
  setFarShadows(on: boolean) {
    for (const g of this.farCasters) g.traverse((o) => {
      if (!(o as Mesh).isMesh) return;
      const m = o as Mesh;
      if (!this.casts.has(m)) this.casts.set(m, m.castShadow);
      m.castShadow = on && this.casts.get(m)!;
    });
  }

  /** Nothing is built until `build()` runs (see engine/steps). */
  constructor(private scene: Scene, private renderer: WebGLRenderer, readonly field: Field, readonly q: Quality) {}

  /** Build the ballpark a piece at a time; `from`..`to` is its share of the loading bar. */
  *build(from = 0, to = 1): Steps {
    const { scene, renderer, field, q } = this;
    const at = (f: number, msg: string): Step => ({ done: from + (to - from) * f, msg });
    setMaterialTexSize(q.texSize);
    const timer = (label: string, t0: number) => { if (import.meta.env?.DEV) console.debug(`[stadium] ${label} ${Math.round(performance.now() - t0)} ms`); };
    yield at(0, 'Painting the sky');
    let t0 = performance.now();
    this.env = new Environment(scene, renderer, q, { elevation: 36, azimuth: -122 });
    timer('environment', t0);

    const yard = field.yard;
    const poolProp = yard.props.find((p) => p.kind === 'pool');
    const pool: PoolSpec | null = poolProp ? { x: poolProp.x, y: poolProp.y, hw: 22, hd: 11, rot: poolProp.rot ?? 0 } : null;
    const L = LAYOUT;
    const patioPoly = rect((L.patio.x0 + L.patio.x1) / 2, (L.patio.y0 + L.patio.y1) / 2, (L.patio.x1 - L.patio.x0) / 2, (L.patio.y1 - L.patio.y0) / 2);
    const blankets = L.dugouts.map((d) => {
      // blanket sits 2.4 ft in front of the dugout origin (local +z)
      const fx = Math.sin(d.rot) * 2.4, fy = -Math.cos(d.rot) * 2.4;
      return rect(d.x + fx, d.y + fy, 5.2, 3.2, -d.rot);
    });
    const bed = rect(4, -26.5, 9.2, 1.6);
    yield at(0.1, 'Mowing the lawn');
    t0 = performance.now();
    this.ground = new Ground(field, q, { holes: [...(pool ? [poolDeckPoly(pool)] : []), patioPoly, bed, ...blankets] });
    this.group.add(this.ground.mesh);
    if (this.ground.grass) this.group.add(this.ground.grass);
    timer('ground', t0);

    yield at(0.3, 'Building the Mendozas\' house');
    t0 = performance.now();
    this.group.add(buildHouse(MENDOZA_STYLE));
    timer('house', t0);

    yield at(0.49, 'Painting the picket fence');
    t0 = performance.now();
    const fence = yard.fence;
    this.group.add(buildPicketFence(fence.filter((f) => f.kind === 'picket')));
    const hedge = buildHedge(fence.filter((f) => f.kind === 'hedge'), q);
    this.group.add(hedge);
    this.farCasters.push(hedge);
    timer('fences', t0);

    yield at(0.56, 'Planting trees');
    t0 = performance.now();
    const treeSpecs: TreeSpec[] = [];
    // yard trees match their physics canopies (see sim/field obstaclesFromProps)
    for (const p of yard.props) {
      if (p.kind !== 'tree') continue;
      const sc = p.scale ?? 1;
      treeSpecs.push({ kind: 'oak', x: p.x, z: -p.y, crownY: 18 * sc, crownR: 11.5 * sc, crownRv: 8.5 * sc });
    }
    treeSpecs.push(
      { kind: 'maple', x: 12, z: -212, crownY: 30, crownR: 17, crownRv: 15 },
      { kind: 'maple', x: -74, z: -206 },
      { kind: 'oak', x: 74, z: -200, crownY: 24, crownR: 16 },
      { kind: 'pine', x: 140, z: -122, height: 46 },
      { kind: 'pine', x: 152, z: -66, height: 40 },
      { kind: 'pine', x: 128, z: -176, height: 52 },
      { kind: 'birch', x: -86, z: 14 },
      { kind: 'birch', x: -94, z: 2, crownY: 20, crownR: 7, crownRv: 9 },
      { kind: 'maple', x: -150, z: -62, crownY: 30, crownR: 18, crownRv: 15 },
      { kind: 'oak', x: -62, z: 66, crownY: 28, crownR: 20, crownRv: 13 },
      ...NEIGHBOR_TREES,
    );
    const trees = buildTrees(treeSpecs, q);
    this.group.add(trees.group);
    timer('trees', t0);

    yield at(0.71, 'Filling the pool');
    t0 = performance.now();
    if (pool) {
      const p = buildPool(pool);
      this.group.add(p.group);
      this.onUpdate((t) => p.update(t));
    }
    timer('pool', t0);

    yield at(0.75, 'Lighting the grill');
    t0 = performance.now();
    const b = new Batch();
    // ── the field
    homePlate(b);
    bases(b, field.bases.slice(1));
    rubber(b, field.mound.x, field.mound.y);

    // ── patio: slab, furniture, grill, cooler, string lights on posts
    const pc = W((L.patio.x0 + L.patio.x1) / 2, (L.patio.y0 + L.patio.y1) / 2, 0);
    b.add(M.concrete('#cbc4b8'), boxFt(L.patio.x1 - L.patio.x0, 0.5, L.patio.y1 - L.patio.y0), T(pc.x, 0.08, pc.z));
    for (let x = L.patio.x0 + 6; x < L.patio.x1; x += 6) b.add(M.paint('#8f897e', 0.95), boxFt(0.06, 0.02, L.patio.y1 - L.patio.y0), T(x, 0.34, pc.z), { castShadow: false });
    patioSet(b, L.patioSet.x, L.patioSet.y, 0.3);
    const g = W(L.grill.x, L.grill.y, 0);
    const toDad = Math.atan2(6, -2);
    this.grillTop.copy(gasGrill(b, L.grill.x, L.grill.y, toDad));
    void g;
    cooler(b, -17, -13, 0.4);
    const posts = [W(-54, -10, 0), W(-34, -10, 0), W(-14, -10, 0)];
    for (const p of posts) {
      b.add(M.woodSolid('#8a6a4a', 3), boxFt(0.4, 9.6, 0.4), T(p.x, 4.8, p.z));
      b.add(M.paint('#5d6b74', 0.6), cyl(0.9, 0.7, 1.6, 14), T(p.x, 0.8 + 0.33, p.z));
    }
    const eaveY = 10.2, hz = FACADE - 0.4;
    const H = (x: number) => new Vector3(x, eaveY, hz);
    const top = (p: Vector3) => new Vector3(p.x, 9.3, p.z);
    stringLights(b, [
      [H(-52), top(posts[0])], [top(posts[0]), H(-42)], [H(-42), top(posts[1])], [top(posts[1]), H(-24)],
      [H(-24), top(posts[2])], [top(posts[0]), top(posts[1])], [top(posts[1]), top(posts[2])],
    ]);

    // ── dugouts for the two teams
    L.dugouts.forEach((d, i) => {
      const team = TEAMS[i] ?? TEAMS[0];
      dugout(b, d.x, d.y, d.rot, { name: team.name, ...team.colors });
    });

    // ── yard clutter
    for (const p of yard.props) {
      if (p.kind === 'flamingo') lawnFlamingo(b, p.x, p.y, p.rot ?? 0);
      if (p.kind === 'lawnchair') lawnChair(b, p.x, p.y, p.rot ?? 0, p.x > 60 ? ['#2f9e66', '#ffffff'] : ['#e8590c', '#ffe8a3']);
      if (p.kind === 'gnome') gnome(b, p.x, p.y, p.rot ?? 0);
    }
    bike(b, 24, -18, 0.6);
    gardenHose(b, [[19, -26], [23, -21], [30, -19], [27, -13], [34, -9], [40, -11]], new Vector3(18, 1.4, FACADE - 0.1));
    b.add(M.mulch(), boxFt(18, 0.25, 3), T(4, 0.06, FACADE - 1.5), { castShadow: false });
    this.group.add(b.build('props'));
    flowers(this.group, [
      { x: -2, y: -26.6, r: 1.4, n: 18 }, { x: 4, y: -26.6, r: 1.4, n: 16 }, { x: 11, y: -26.6, r: 1.3, n: 14, palette: ['#ff5e7e', '#ffffff'] },
      { x: 88, y: 51, r: 2, n: 22 }, { x: 90.5, y: 70, r: 2.2, n: 24, palette: ['#ffd43b', '#ff922b', '#ffffff'] },
      { x: -50, y: -25, r: 1.6, n: 14, palette: ['#c084fc', '#ffffff'] },
    ]);
    const sh = shrubs(this.group, [
      { x: -49, y: -26, r: 1.6 }, { x: 14.5, y: -26.4, r: 1.4 }, { x: -4.5, y: -26.3, r: 1.1 },
      { x: -60, y: 178, r: 3.2 }, { x: -88, y: 150, r: 3.5 }, { x: 94, y: 46, r: 2.4 }, { x: 96, y: 86, r: 2.6 },
      { x: -14, y: -27, r: 1.0 },
    ]);
    trees.leaves.push(sh);
    timer('props', t0);

    yield at(0.79, 'Waking up the neighbors');
    t0 = performance.now();
    const hood = buildNeighborhood(q.name === 'low').group;
    this.group.add(hood);
    this.farCasters.push(hood);
    timer('neighborhood', t0);
    this.onUpdate((t) => { for (const l of trees.leaves) l.userData.uTime.value = t; });
    scene.add(this.group);
  }

  onUpdate(f: (t: number, dt: number) => void) { this.updaters.push(f); }

  update(t: number, dt: number) {
    this.env.update(t);
    this.ground.update(t);
    for (const f of this.updaters) f(t, dt);
  }
}
