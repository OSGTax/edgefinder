import { Quaternion, Vector3, type Bone } from 'three';

// Analytic two-bone IK (shoulder → elbow → wrist) that keeps each bone's
// current twist: we rotate bones by the shortest arc onto the solved
// directions, in world space, then convert back to local rotations.

const _a = new Vector3(), _b = new Vector3(), _c = new Vector3();
const _dir = new Vector3(), _pole = new Vector3(), _elbow = new Vector3();
const _q = new Quaternion(), _pq = new Quaternion(), _wq = new Quaternion();
const _cur = new Vector3(), _want = new Vector3();

function aimBone(bone: Bone, from: Vector3, curTip: Vector3, wantTip: Vector3) {
  _cur.copy(curTip).sub(from).normalize();
  _want.copy(wantTip).sub(from).normalize();
  _q.setFromUnitVectors(_cur, _want);
  bone.getWorldQuaternion(_wq);
  _wq.premultiply(_q);
  bone.parent!.getWorldQuaternion(_pq);
  bone.quaternion.copy(_pq.invert().multiply(_wq));
  bone.updateMatrixWorld(true);
}

/**
 * Bend `upper`/`lower` so the end of `end` reaches `target` (world). `pole`
 * is a world-space point the elbow/knee should bend toward. `blend` (0..1)
 * mixes from the current pose.
 */
export function twoBoneIK(upper: Bone, lower: Bone, end: Bone, target: Vector3, pole: Vector3, blend = 1) {
  if (blend <= 0) return;
  upper.updateMatrixWorld(true);
  upper.getWorldPosition(_a);
  lower.getWorldPosition(_b);
  end.getWorldPosition(_c);
  const l1 = _a.distanceTo(_b), l2 = _b.distanceTo(_c);
  const tgt = _dir.copy(target);
  if (blend < 1) tgt.lerpVectors(_c, target, blend);
  const toT = tgt.clone().sub(_a);
  const d = Math.min(Math.max(toT.length(), Math.abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3);
  toT.normalize();
  const cosA = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d);
  const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  _pole.copy(pole).sub(_a);
  _pole.addScaledVector(toT, -_pole.dot(toT));
  if (_pole.lengthSq() < 1e-8) _pole.set(0, -1, 0).addScaledVector(toT, toT.y);
  _pole.normalize();
  _elbow.copy(_a).addScaledVector(toT, l1 * cosA).addScaledVector(_pole, l1 * sinA);
  aimBone(upper, _a, _b, _elbow);
  lower.getWorldPosition(_b);
  end.getWorldPosition(_c);
  const wrist = _a.clone().addScaledVector(toT, d);
  aimBone(lower, _b, _c, wrist);
}

/** Turn a bone (world space) so its local axis `axis` points along `dir`, keeping the rest of the pose. */
export function pointBone(bone: Bone, axis: Vector3, dir: Vector3, blend = 1) {
  bone.updateMatrixWorld(true);
  bone.getWorldQuaternion(_wq);
  _cur.copy(axis).applyQuaternion(_wq).normalize();
  _want.copy(dir).normalize();
  _q.setFromUnitVectors(_cur, _want);
  if (blend < 1) _q.slerp(new Quaternion(), 1 - blend);
  _wq.premultiply(_q);
  bone.parent!.getWorldQuaternion(_pq);
  bone.quaternion.copy(_pq.invert().multiply(_wq));
  bone.updateMatrixWorld(true);
}
