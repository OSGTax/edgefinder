import { MathUtils, Vector3, type PerspectiveCamera } from 'three';
import { W } from '../gfx/units';
import type { Match } from '../sim/match';

// The camera director: a broadcast-style view behind the plate for each
// pitch, a high follow-cam for balls in play, a chase for home runs, and slow
// orbits for intros and breaks. Every shot is a target pose the real camera
// glides toward.

export type Shot = 'intro' | 'bat' | 'live' | 'homer' | 'break' | 'final' | 'title';

interface Pose { pos: Vector3; look: Vector3; fov: number }

export class Director {
  shot: Shot = 'intro';
  shotT = 0;
  private cur: Pose = { pos: W(160, 210, 80), look: W(0, 60, 0), fov: 50 };
  private ballGround = new Vector3();
  private follow = new Vector3();
  private homerAt = new Vector3();
  /** set by the screen: e.g. the intro is showing */
  forced: Shot | null = null;
  private time = 0;

  constructor(readonly cam: PerspectiveCamera) {}

  update(m: Match | null, dt: number, ball: Vector3 | null) {
    this.time += dt;
    let shot: Shot = this.forced ?? 'bat';
    if (!this.forced && m) {
      const play = m.play ?? (m.phase === 'result' ? m.lastPlay : null);
      if (m.phase === 'over') shot = 'final';
      else if (m.phase === 'halfOver') shot = 'break';
      else if (play && play.deadKind === 'hr') shot = 'homer';
      else if (m.phase === 'live' || (m.phase === 'result' && m.lastPlay)) shot = 'live';
    }
    if (shot !== this.shot) {
      if (shot === 'homer' && ball) this.homerAt.copy(ball);
      this.shot = shot;
      this.shotT = 0;
    }
    this.shotT += dt;
    const want = this.pose(m, ball, dt);
    const snap = this.shot === 'bat' && this.shotT > 1.2;
    const rate = this.shot === 'bat' ? (this.shotT < 0.15 ? 2.5 : 6) : this.shot === 'live' ? 3.2 : this.shot === 'homer' ? 2.2 : 1.2;
    const k = snap ? 1 : 1 - Math.exp(-dt * rate);
    this.cur.pos.lerp(want.pos, k);
    this.cur.look.lerp(want.look, k);
    // keep the same sideways view on tall (portrait) screens
    const aspect = this.cam.aspect || 16 / 9;
    if (aspect < 1.5) {
      const h = 2 * Math.atan(Math.tan((want.fov * Math.PI) / 360) * (16 / 9));
      want.fov = Math.min(95, Math.max(want.fov, (2 * Math.atan(Math.tan(h / 2) / aspect) * 180) / Math.PI * 0.82));
    }
    this.cur.fov = MathUtils.lerp(this.cur.fov, want.fov, k);
    this.cam.position.copy(this.cur.pos);
    // a kick: a quick punch-in that springs back
    this.kickAmt *= Math.exp(-dt * 9);
    const fov = this.cur.fov * (1 - this.kickAmt * 0.06);
    if (this.shakeAmt > 0.001) {
      const a = this.shakeAmt;
      this.cam.position.add(new Vector3(Math.sin(this.time * 61) * a * 0.3, Math.sin(this.time * 47 + 1) * a * 0.25, 0));
      this.shakeAmt *= Math.exp(-dt * 6);
    }
    this.cam.lookAt(this.cur.look);
    if (Math.abs(this.cam.fov - fov) > 0.01) { this.cam.fov = fov; this.cam.updateProjectionMatrix(); }
  }

  private shakeAmt = 0;
  /** a little camera kick (big hits) */
  shake(a: number) { this.shakeAmt = Math.max(this.shakeAmt, a); }

  private kickAmt = 0;
  /** a punch-in on contact, catches and outs (0..1) */
  kick(a: number) { this.kickAmt = Math.max(this.kickAmt, Math.min(1, a)); this.shake(a * 0.12); }

  /** start from wherever the camera is now */
  adopt(cam: PerspectiveCamera) {
    this.cur.pos.copy(cam.position);
    this.cur.look.copy(cam.getWorldDirection(new Vector3()).multiplyScalar(60).add(cam.position));
    this.cur.fov = cam.fov;
  }

  /** jump straight to the current shot (no glide) */
  cut() { this.shotT = 99; }

  private pose(m: Match | null, ball: Vector3 | null, dt: number): Pose {
    const mound = m ? m.field.mound.y : 44;
    switch (this.shot) {
      case 'bat': {
        const side = m?.batterSide ?? 'R';
        const off = side === 'R' ? 1.25 : -1.25;
        // wide phone screens (up to 19.5:9) are short: come in lower and tighter
        // so the batter, the zone and the pitcher fill the height
        const w = MathUtils.clamp(((this.cam.aspect || 16 / 9) - 1.6) / 0.5, 0, 1);
        if (w <= 0) return { pos: W(off, -18.5, 9.8), look: W(off * 0.1, mound * 0.52, 2.0), fov: 40 };
        return {
          pos: W(off, MathUtils.lerp(-18.5, -15.5, w), MathUtils.lerp(9.8, 9.1, w)),
          look: W(off * 0.1, MathUtils.lerp(mound * 0.52, 14, w), MathUtils.lerp(2.0, 0.5, w)),
          fov: MathUtils.lerp(40, 33, w),
        };
      }
      case 'live': {
        if (ball) {
          this.ballGround.set(ball.x, 0, ball.z);
          this.follow.lerp(this.ballGround, 1 - Math.exp(-dt * 4));
        }
        const bx = this.follow.x, by = -this.follow.z;
        const dist = Math.hypot(bx, by);
        const h = ball ? Math.max(0, ball.y) : 0;
        // on a short, wide phone screen ride a little closer so the kids aren't specks
        const w = MathUtils.clamp(((this.cam.aspect || 16 / 9) - 1.6) / 0.5, 0, 1);
        const near = 1 - w * 0.18;
        return {
          pos: W(bx * 0.28, Math.min(-12, (by * 0.2 - 20 - dist * 0.06) * near + by * 0.2 * (1 - near)), (22 + dist * 0.16) * near + h * 0.25),
          look: W(bx * 0.9, by * 0.9 + 4, 2 + h * 0.45),
          fov: (46 + Math.min(14, dist * 0.04)) * (1 - w * 0.1),
        };
      }
      case 'homer': {
        // ride behind the ball as it sails out, then swing round to the trot
        const t = this.shotT;
        if (ball && t < 2.6) {
          const b = ball.clone();
          const dir = new Vector3(b.x, 0, b.z).normalize();
          return { pos: b.clone().addScaledVector(dir, -38).add(new Vector3(0, 14, 0)), look: b, fov: 50 };
        }
        const a = 0.6 + (t - 2.6) * 0.12;
        return { pos: W(Math.sin(a) * 70, Math.cos(a) * 70 + 30, 26), look: W(0, 30, 2), fov: 52 };
      }
      case 'break':
      case 'intro':
      case 'title': {
        const a = this.time * 0.07 + 0.4;
        const r = this.shot === 'title' ? 150 : 170;
        return { pos: W(Math.sin(a) * r, 70 + Math.cos(a) * r * 0.85, this.shot === 'title' ? 46 : 60), look: W(0, 62, 4), fov: 50 };
      }
      case 'final': {
        const a = this.time * 0.18;
        return { pos: W(Math.sin(a) * 34, mound - 8 + Math.cos(a) * 34, 12), look: W(0, mound - 8, 3.5), fov: 48 };
      }
    }
  }
}
