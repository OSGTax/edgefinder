import { Vector3 } from 'three';

// The simulation works in feet with +z up and +y toward center field.
// Three.js is y-up, so: three.x = sim.x, three.y = sim.z, three.z = -sim.y.
// (Looking from home plate toward center field is looking down -z.)

export const W = (x: number, y: number, z = 0) => new Vector3(x, z, -y);

export function setW(v: Vector3, x: number, y: number, z = 0) {
  return v.set(x, z, -y);
}

/** World yaw for something facing sim angle `a` (0 = toward +y, clockwise toward +x). */
export const yawOf = (simFacing: number) => Math.PI - simFacing;
