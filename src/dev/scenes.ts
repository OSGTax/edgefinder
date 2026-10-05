import { kid } from '../data/kids';
import { TEAMS } from '../data/teams';
import { YARD_BY_ID } from '../data/yards';
import { buildField } from '../sim/field';
import { FIELD_ORDER } from '../sim/play';
import { autoDefense } from '../sim/lineup';
import { SceneRenderer, type KidSprite } from '../render/scene';
import { BAT_CAM, FIELD_CAM, OVERVIEW_CAM } from '../render/cameras';

/** Dev-only: a static scene of a yard with everyone at their spots. */
export function showScene(root: HTMLElement, yardId: string, camName: string) {
  const canvas = document.createElement('canvas');
  root.appendChild(canvas);
  const r = new SceneRenderer(canvas);
  r.resize(window.innerWidth, window.innerHeight);
  const yard = YARD_BY_ID[yardId] ?? Object.values(YARD_BY_ID)[0];
  r.field = buildField(yard);
  const home = TEAMS.find((t) => t.yardId === yard.id) ?? TEAMS[0];
  const away = TEAMS.find((t) => t !== home)!;
  r.cam.pose = camName === 'field' ? FIELD_CAM : camName === 'overview' ? OVERVIEW_CAM : BAT_CAM;
  const def = autoDefense(home.roster.map(kid));
  const kids: KidSprite[] = def.map((id, i) => {
    const pos = FIELD_ORDER[i];
    const spot = pos === 'P' ? r.field!.defaultSpots.P : r.field!.defaultSpots[pos];
    return {
      kid: kid(id), team: home, x: spot.x, y: pos === 'P' ? spot.y - 1 : pos === 'C' ? -3 : spot.y,
      facing: pos === 'C' ? 0 : Math.PI, pose: { anim: pos === 'C' ? 'crouch' : pos === 'P' ? 'pitch' : 'ready', t: 0.3, windup: 0.3 },
    };
  });
  const batter = kid(away.roster[2]);
  kids.push({ kid: batter, team: away, x: -2.3, y: 0.2, facing: Math.PI / 2, pose: { anim: 'bat', t: 0.5, swing: 0, view: 'back' } });
  kids.push({ kid: kid(away.roster[5]), team: away, x: r.field.bases[1].x + 3, y: r.field.bases[1].y + 2, facing: -Math.PI / 4, pose: { anim: 'ready', t: 0 } });
  const t0 = performance.now();
  const loop = () => {
    const t = (performance.now() - t0) / 1000;
    r.draw({ kids, ball: { x: -1, y: 18, z: 4.2 }, t }, 1 / 60);
    requestAnimationFrame(loop);
  };
  loop();
}
