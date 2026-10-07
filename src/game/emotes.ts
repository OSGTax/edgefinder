import { CanvasTexture, type Camera, Object3D, Scene, Sprite, SpriteMaterial, SRGBColorSpace, Vector3 } from 'three';

// Cartoon symbols over the kids' heads: stars circling a kid who just
// stumbled, a sweat drop on a pitcher in a jam, a "!" over the fielder about
// to make the play, a few musical notes when someone's happy. Drawn in code
// (bold shapes, warm ink outline) and used sparingly so the screen stays clean:
// the game asks for one at a real moment, and a kid only ever wears one at a time.

export type EmoteKind = 'stars' | 'sweat' | 'bang' | 'notes';

const INK = '#2b1d14';

function canvas(draw: (g: CanvasRenderingContext2D, s: number) => void): CanvasTexture {
  const s = 64;
  const c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d')!;
  g.lineJoin = 'round'; g.lineCap = 'round';
  draw(g, s);
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  return t;
}

function star(g: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5 + 0.08, rr = i % 2 ? r * 0.48 : r;
    g.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr);
  }
  g.closePath();
}

let TEX: Record<EmoteKind, CanvasTexture> | null = null;
function textures(): Record<EmoteKind, CanvasTexture> {
  if (TEX) return TEX;
  TEX = {
    stars: canvas((g, s) => {
      star(g, s / 2, s / 2 + 2, s * 0.4);
      g.fillStyle = '#ffd43b'; g.fill();
      g.lineWidth = 5; g.strokeStyle = INK; g.stroke();
      g.fillStyle = '#fff6c4'; g.beginPath(); g.ellipse(s * 0.42, s * 0.42, 5, 3, -0.6, 0, Math.PI * 2); g.fill();
    }),
    sweat: canvas((g, s) => {
      g.beginPath();
      g.moveTo(s * 0.5, s * 0.08);
      g.bezierCurveTo(s * 0.62, s * 0.34, s * 0.8, s * 0.5, s * 0.8, s * 0.66);
      g.arc(s * 0.5, s * 0.66, s * 0.3, 0, Math.PI);
      g.bezierCurveTo(s * 0.2, s * 0.5, s * 0.38, s * 0.34, s * 0.5, s * 0.08);
      g.fillStyle = '#8fd3ff'; g.fill();
      g.lineWidth = 4.5; g.strokeStyle = INK; g.stroke();
      g.fillStyle = '#ffffff'; g.beginPath(); g.ellipse(s * 0.4, s * 0.62, 4, 7, 0.3, 0, Math.PI * 2); g.fill();
    }),
    bang: canvas((g, s) => {
      // a slightly tilted, hand-cut "!"
      g.translate(s / 2, s / 2); g.rotate(0.12); g.translate(-s / 2, -s / 2);
      g.beginPath();
      g.moveTo(s * 0.36, s * 0.08); g.lineTo(s * 0.66, s * 0.08); g.lineTo(s * 0.58, s * 0.64); g.lineTo(s * 0.43, s * 0.64);
      g.closePath();
      g.fillStyle = '#ff5a3c'; g.fill(); g.lineWidth = 5; g.strokeStyle = INK; g.stroke();
      g.beginPath(); g.arc(s * 0.5, s * 0.81, s * 0.1, 0, Math.PI * 2);
      g.fill(); g.stroke();
    }),
    notes: canvas((g, s) => {
      g.lineWidth = 5; g.strokeStyle = INK; g.fillStyle = '#7c5cff';
      // a beamed pair of eighth notes
      g.beginPath(); g.ellipse(s * 0.28, s * 0.74, 10, 7, -0.4, 0, Math.PI * 2); g.fill(); g.stroke();
      g.beginPath(); g.ellipse(s * 0.7, s * 0.64, 10, 7, -0.4, 0, Math.PI * 2); g.fill(); g.stroke();
      g.beginPath(); g.moveTo(s * 0.4, s * 0.72); g.lineTo(s * 0.4, s * 0.2); g.lineTo(s * 0.82, s * 0.1); g.lineTo(s * 0.82, s * 0.62); g.stroke();
      g.lineWidth = 9; g.beginPath(); g.moveTo(s * 0.4, s * 0.22); g.lineTo(s * 0.82, s * 0.12); g.stroke();
    }),
  };
  return TEX;
}

interface Live {
  kind: EmoteKind;
  head: Object3D;
  /** kid height scale, so grown-ups' symbols sit higher and bigger */
  size: number;
  t: number;
  dur: number;
  sprites: Sprite[];
}

const _p = new Vector3();
const COUNT: Record<EmoteKind, number> = { stars: 3, sweat: 1, bang: 1, notes: 2 };

export class Emotes {
  private live: Live[] = [];
  private pool: Sprite[] = [];

  constructor(private scene: Scene) {}

  private sprite(kind: EmoteKind): Sprite {
    let s = this.pool.pop();
    if (!s) {
      s = new Sprite(new SpriteMaterial({ transparent: true, depthWrite: false }));
      s.renderOrder = 12;
      this.scene.add(s);
    }
    (s.material as SpriteMaterial).map = textures()[kind];
    (s.material as SpriteMaterial).needsUpdate = true;
    s.visible = true;
    return s;
  }

  /** Put a symbol over a head (the head bone) for `dur` seconds. One per kid. */
  show(kind: EmoteKind, head: Object3D, dur = 1.5, size = 1) {
    if (this.live.some((l) => l.head === head)) return;
    const sprites: Sprite[] = [];
    for (let i = 0; i < COUNT[kind]; i++) sprites.push(this.sprite(kind));
    this.live.push({ kind, head, size, t: 0, dur, sprites });
  }

  has(head: Object3D) { return this.live.some((l) => l.head === head); }

  /** `camera`: symbols keep a readable size on screen however far away the kid is */
  update(dt: number, camera?: Camera) {
    for (let i = this.live.length - 1; i >= 0; i--) {
      const l = this.live[i];
      l.t += dt;
      if (l.t >= l.dur) {
        for (const s of l.sprites) { s.visible = false; this.pool.push(s); }
        this.live.splice(i, 1);
        continue;
      }
      l.head.getWorldPosition(_p);
      const far = camera ? Math.min(6, Math.max(1, camera.position.distanceTo(_p) / 15)) : 1;
      // sprites grow with distance; their offsets from the head only a little
      const u = l.t / l.dur, z = l.size * (1 + (far - 1) * 0.3), zs = l.size * far;
      // pop in with overshoot, shrink away at the end
      const pop = l.t < 0.18 ? 1.25 * Math.sin((l.t / 0.18) * Math.PI * 0.5) : l.t < 0.3 ? 1.25 - (l.t - 0.18) * 2 : 1;
      const fade = u > 0.8 ? 1 - (u - 0.8) / 0.2 : 1;
      const k = pop * fade * zs;
      l.sprites.forEach((s, j) => {
        const m = s.material as SpriteMaterial;
        m.opacity = Math.min(1, fade * 1.5);
        switch (l.kind) {
          case 'stars': {
            const a = l.t * 7 + (j * Math.PI * 2) / 3;
            s.position.set(_p.x + Math.cos(a) * 0.75 * z, _p.y + 1.95 * z + Math.sin(a * 2) * 0.08, _p.z + Math.sin(a) * 0.75 * z);
            s.scale.setScalar(0.6 * k);
            break;
          }
          case 'sweat':
            // beads up by the temple, then rolls down
            s.position.set(_p.x + 0.8 * z, _p.y + (1.55 - Math.max(0, u - 0.35) * 0.9) * z, _p.z);
            s.scale.set(0.42 * k, 0.5 * k, 1);
            break;
          case 'bang':
            s.position.set(_p.x, _p.y + (2.35 + Math.sin(l.t * 9) * 0.05) * z, _p.z);
            s.scale.setScalar(0.8 * k);
            break;
          case 'notes': {
            const w = (l.t + j * 0.6) % 1.2;
            s.position.set(_p.x + (j ? -0.5 : 0.55) * z + Math.sin(w * 6) * 0.15, _p.y + (1.7 + w * 0.9) * z, _p.z);
            s.scale.setScalar(0.5 * zs * fade * Math.min(1, w * 5) * (1 - w / 1.5));
            break;
          }
        }
      });
    }
  }
}
