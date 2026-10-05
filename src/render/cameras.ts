import type { CamPose } from './camera';

/**
 * Behind the plate, shifted to the side away from the batter so neither the
 * catcher's head nor the batter hides the strike zone.
 */
export function batCam(side: 'R' | 'L'): CamPose {
  const s = side === 'R' ? 1 : -1;
  return { pos: { x: 3 * s, y: -17.5, z: 13.2 }, target: { x: -0.4 * s, y: 17, z: 0 }, fov: 42 };
}
export const BAT_CAM: CamPose = batCam('R');

/** High above the back deck: the whole yard while the ball is in play. */
export const FIELD_CAM: CamPose = { pos: { x: 0, y: -62, z: 66 }, target: { x: 0, y: 80, z: 0 }, fov: 54 };

/** Out past the outfield looking back at the house, for intros and inning breaks. */
export const OVERVIEW_CAM: CamPose = { pos: { x: 60, y: 240, z: 62 }, target: { x: 0, y: 40, z: 0 }, fov: 50 };

/** Field camera that follows the ball, framing it with the infield. */
export function followCam(bx: number, by: number, bz: number): CamPose {
  const tx = bx * 0.55;
  const ty = Math.max(38, Math.min(135, (by + 40) * 0.55 + 10));
  const lift = Math.min(22, bz * 0.35);
  return {
    pos: { x: tx * 0.5, y: ty - 92, z: 50 + lift },
    target: { x: tx, y: ty, z: 0 },
    fov: 50,
  };
}
