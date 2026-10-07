import {
  BufferAttribute, CanvasTexture, InstancedBufferAttribute, Color, DynamicDrawUsage, InstancedMesh, LinearFilter, Mesh, MeshLambertMaterial, MeshStandardMaterial,
  NoColorSpace, PlaneGeometry, BufferGeometry, Vector2, type IUniform,
} from 'three';
import { pointInPoly } from '../engine/math';
import type { Field } from '../sim/field';
import { dirtTex, grassTex } from './textures';
import { mulberry, Noise2 } from './noise';
import type { Quality } from './quality';

// The ground is one big mesh whose shader blends tiled grass and dirt using
// a "splat map" painted in code: R = dirt, G = chalk, B = mow stripes,
// A = 0 cuts holes (the pool).

export const SPLAT = { minX: -170, minY: -90, size: 340 };

const hills = new Noise2(5);
/** Height of the land (three space x/z): flat around the yard, rolling hills far away. */
export function terrainHeight(x: number, z: number): number {
  const r = Math.hypot(x, z + 60);
  const t = Math.min(1, Math.max(0, (r - 420) / 500));
  return t * t * (3 - 2 * t) * (hills.fbm(x / 420, z / 420, 3) * 70 + 30);
}

export interface GroundOptions {
  /** extra holes in the lawn (sim polygons), e.g. the pool */
  holes?: [number, number][][];
  /** extra dirt areas painted softly */
  dirt?: [number, number][][];
  stripes?: boolean;
}

export class Ground {
  readonly mesh: Mesh;
  readonly splat: HTMLCanvasElement;
  readonly uniforms: Record<string, IUniform> = {};
  private splatData: Uint8ClampedArray | null = null;
  grass: InstancedMesh | null = null;

  constructor(readonly field: Field, readonly q: Quality, readonly o: GroundOptions = {}) {
    let t = performance.now();
    const lap = (l: string) => { if (import.meta.env?.DEV) console.debug(`[stadium] ground.${l} ${Math.round(performance.now() - t)} ms`); t = performance.now(); };
    this.splat = this.paintSplat();
    lap('splat');
    this.mesh = this.buildMesh();
    lap('mesh');
    if (q.grassBlades > 0) this.grass = this.buildBlades(q.grassBlades);
    lap('blades');
    this.splatData = null; // only needed to place the blades (up to 16 MB)
  }

  // ─────────────────────────────────────────────────────── splat map

  private toPx(x: number, y: number, size: number): [number, number] {
    return [((x - SPLAT.minX) / SPLAT.size) * size, (1 - (y - SPLAT.minY) / SPLAT.size) * size];
  }

  private paintSplat(): HTMLCanvasElement {
    const size = this.q.splatSize;
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d')!;
    const f = this.field;
    const px = (x: number, y: number) => this.toPx(x, y, size);
    const ppf = size / SPLAT.size;
    // neutral lawn
    ctx.fillStyle = 'rgb(0,0,128)';
    ctx.fillRect(0, 0, size, size);

    // mow stripes inside the fence, in alternating 12 ft bands toward center field
    if (this.o.stripes !== false) {
      ctx.save();
      clipPoly(ctx, f.fence.map((s) => px(s.a[0], s.a[1])));
      for (let i = -20; i < 20; i++) {
        ctx.fillStyle = i % 2 ? 'rgb(0,0,90)' : 'rgb(0,0,168)';
        const a = px(i * 12, -80), b = px(i * 12 + 12, 260);
        ctx.fillRect(a[0], b[1], b[0] - a[0], a[1] - b[1]);
      }
      ctx.restore();
    }

    // worn dirt: soft-edged polygons (stacked strokes make a feathered edge)
    const softFill = (poly: [number, number][], feather: number, strength = 1) => {
      const pts = poly.map(([x, y]) => px(x, y));
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      const steps = 6;
      for (let i = steps; i >= 1; i--) {
        ctx.strokeStyle = `rgba(${Math.round((255 * strength) / steps / 1.4)},0,0,1)`;
        ctx.lineWidth = (feather * ppf * i) / steps;
        ctx.lineJoin = 'round';
        pathPoly(ctx, pts);
        ctx.stroke();
      }
      ctx.fillStyle = `rgba(${Math.round(255 * strength)},0,0,1)`;
      pathPoly(ctx, pts);
      ctx.fill();
      ctx.restore();
    };
    for (const a of f.worn) softFill(a, 2.4);
    for (const d of this.o.dirt ?? []) softFill(d, 3);
    // scuffed patches in the outfield where the fielders stand
    const rnd = mulberry(77);
    for (const pos of ['LF', 'CF', 'RF', 'SS', '2B'] as const) {
      const s = f.defaultSpots[pos];
      const poly: [number, number][] = Array.from({ length: 10 }, (_, i) => {
        const a = (i / 10) * Math.PI * 2;
        const r = 2.2 + rnd() * 1.5;
        return [s.x + Math.cos(a) * r, s.y + Math.sin(a) * r * 1.3];
      });
      softFill(poly, 3, 0.18);
    }

    // chalk (green channel): foul lines and the batter's boxes, a little wobbly
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    clipPoly(ctx, f.fence.map((s) => px(s.a[0], s.a[1])));
    ctx.strokeStyle = 'rgb(0,230,0)';
    ctx.lineCap = 'round';
    ctx.lineWidth = Math.max(1.5, 0.26 * ppf);
    const line = (x1: number, y1: number, x2: number, y2: number) => {
      const n = 24;
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const t = i / n;
        const wob = Math.sin(t * 17 + x2) * 0.05;
        const [qx, qy] = px(x1 + (x2 - x1) * t + wob, y1 + (y2 - y1) * t - wob);
        if (i === 0) ctx.moveTo(qx, qy);
        else ctx.lineTo(qx, qy);
      }
      ctx.stroke();
    };
    const far = 300;
    line(0.6, 0.6, far, far);
    line(-0.6, 0.6, -far, far);
    for (const sx of [-1, 1]) {
      const bx = sx * 2.4;
      line(bx - 1.45, -2.9, bx + 1.45, -2.9);
      line(bx + 1.45, -2.9, bx + 1.45, 2.9);
      line(bx + 1.45, 2.9, bx - 1.45, 2.9);
      line(bx - 1.45, 2.9, bx - 1.45, -2.9);
    }
    ctx.restore();

    // holes (alpha 0)
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    for (const h of this.o.holes ?? []) {
      pathPoly(ctx, h.map(([x, y]) => px(x, y)));
      ctx.fill();
    }
    ctx.restore();
    this.splatData = ctx.getImageData(0, 0, size, size).data;
    return c;
  }

  /** Sample the splat map at a sim position (0..1 per channel). */
  sample(x: number, y: number): [number, number, number, number] {
    const size = this.q.splatSize;
    const [qx, qy] = this.toPx(x, y, size);
    const ix = Math.max(0, Math.min(size - 1, Math.floor(qx)));
    const iy = Math.max(0, Math.min(size - 1, Math.floor(qy)));
    const d = this.splatData;
    if (!d) return [0, 0, 0.5, 1];
    const i = (iy * size + ix) * 4;
    return [d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, d[i + 3] / 255];
  }

  // ─────────────────────────────────────────────────────── ground mesh

  private buildMesh(): Mesh {
    const ext = 1800;
    const seg = this.q.groundSeg;
    const geo = new PlaneGeometry(ext * 2, ext * 2, seg, seg);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position as BufferAttribute;
    const uv = geo.attributes.uv as BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      pos.setY(i, terrainHeight(x, z) - 0.02);
      uv.setXY(i, x / 9, z / 9);
    }
    geo.computeVertexNormals();
    const gt = grassTex(this.q.texSize);
    const dt = dirtTex(this.q.texSize, this.field.yard.infield === 'sand');
    const splatTex = new CanvasTexture(this.splat);
    splatTex.colorSpace = NoColorSpace;
    splatTex.flipY = true;
    splatTex.minFilter = LinearFilter;
    splatTex.generateMipmaps = false;
    const mat = new MeshStandardMaterial({ map: gt.map, normalMap: gt.normal, roughness: 0.93, metalness: 0, normalScale: new Vector2(0.35, 0.35) });
    Object.assign(this.uniforms, {
      uSplat: { value: splatTex },
      uSplatRect: { value: new Vector2(SPLAT.minX, SPLAT.minY) },
      uSplatSize: { value: SPLAT.size },
      uDirt: { value: dt.map },
      uDirtN: { value: dt.normal },
      uDirtTint: { value: new Color(this.field.yard.theme.dirt) },
    });
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, this.uniforms);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>
varying vec3 vWPos;
uniform sampler2D uSplat; uniform vec2 uSplatRect; uniform float uSplatSize;
uniform sampler2D uDirt; uniform sampler2D uDirtN; uniform vec3 uDirtTint;
vec4 splatAt() {
  vec2 s = vec2((vWPos.x - uSplatRect.x) / uSplatSize, (-vWPos.z - uSplatRect.y) / uSplatSize);
  if (s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) return vec4(0.0, 0.0, 0.5, 1.0);
  return texture2D(uSplat, s);
}
vec4 gSplat;`)
        .replace('#include <map_fragment>', `
gSplat = splatAt();
if (gSplat.a < 0.5) discard;
vec4 grassC = texture2D(map, vMapUv);
// cartoon lawn: mostly one clean saturated green, the painted texture only as a soft hint
grassC.rgb = mix(vec3(0.20, 0.42, 0.11), grassC.rgb, 0.35);
// large-scale color variation so the lawn never looks tiled
float big = sin(vWPos.x * 0.031 + sin(vWPos.z * 0.023) * 2.0) * 0.5 + 0.5;
grassC.rgb *= mix(0.9, 1.08, big);
grassC.rgb *= mix(0.84, 1.16, gSplat.b);
// sunny and a touch warm, with sun-dried straw patches out on the hills
grassC.rgb *= vec3(1.03, 1.0, 0.88);
float farA = smoothstep(140.0, 700.0, length(vWPos.xz + vec2(0.0, 60.0)));
float dryN = sin(vWPos.x * 0.011 + sin(vWPos.z * 0.017) * 1.7) * sin(vWPos.z * 0.009 - vWPos.x * 0.004) * 0.5 + 0.5;
float dry = farA * smoothstep(0.35, 0.8, dryN) * 0.55 + farA * clamp(vWPos.y / 90.0, 0.0, 0.3);
grassC.rgb = mix(grassC.rgb, grassC.rgb * vec3(1.45, 1.25, 0.7) + vec3(0.03, 0.02, 0.0), dry);
vec4 dirtC = texture2D(uDirt, vMapUv * 1.6);
dirtC.rgb *= uDirtTint / vec3(0.485, 0.254, 0.102);
dirtC.rgb = mix(uDirtTint * 1.05, dirtC.rgb, 0.4);
float dirtAmt = smoothstep(0.08, 0.75, gSplat.r);
vec4 baseC = mix(grassC, dirtC, dirtAmt);
baseC.rgb = mix(baseC.rgb, vec3(0.93, 0.93, 0.9), smoothstep(0.25, 0.8, gSplat.g));
diffuseColor *= baseC;`)
        .replace('#include <normal_fragment_maps>', `
vec3 nG = texture2D(normalMap, vNormalMapUv).xyz * 2.0 - 1.0;
vec3 nD = texture2D(uDirtN, vNormalMapUv * 1.6).xyz * 2.0 - 1.0;
vec3 mapN = normalize(mix(nG, nD, smoothstep(0.08, 0.75, gSplat.r)));
mapN.xy *= normalScale;
normal = normalize(tbn * mapN);`)
        .replace('#include <roughnessmap_fragment>', `
float roughnessFactor = mix(roughness, 0.98, smoothstep(0.08, 0.75, gSplat.r));`);
    };
    const mesh = new Mesh(geo, mat);
    mesh.receiveShadow = true;
    mesh.name = 'ground';
    return mesh;
  }

  // ─────────────────────────────────────────────────────── grass blades

  private buildBlades(count: number): InstancedMesh {
    // a tapered blade: 3 segments, 7 vertices
    const g = new BufferGeometry();
    const h = 1, w = 0.032;
    const verts = [-w, 0, 0, w, 0, 0, -w * 0.75, h * 0.4, 0.02, w * 0.75, h * 0.4, 0.02, -w * 0.4, h * 0.75, 0.06, w * 0.4, h * 0.75, 0.06, 0, h, 0.13];
    const idx = [0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5, 4, 5, 6];
    g.setAttribute('position', new BufferAttribute(new Float32Array(verts), 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    const mat = new MeshLambertMaterial({ color: 0xffffff, side: 2 });
    const time = { value: 0 };
    this.uniforms.uTime = time;
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = time;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nuniform float uTime;')
        .replace('#include <begin_vertex>', `#include <begin_vertex>
vec4 iw = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
float sway = sin(uTime * 1.7 + iw.x * 0.25 + iw.z * 0.19) * 0.5 + sin(uTime * 3.1 + iw.x * 0.9) * 0.15;
transformed.x += sway * position.y * position.y * 0.18;
transformed.z += sway * position.y * position.y * 0.1;`);
      // normals point up so blades shade like the lawn under them (both faces)
      sh.vertexShader = sh.vertexShader.replace('#include <beginnormal_vertex>', 'vec3 objectNormal = vec3(0.0, 1.0, 0.0);');
      sh.fragmentShader = sh.fragmentShader.replace('#include <normal_fragment_begin>', 'vec3 normal = normalize(vNormal); vec3 nonPerturbedNormal = normal;');
    };
    const mesh = new InstancedMesh(g, mat, count);
    mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    const rnd = mulberry(99);
    const col = new Color();
    const fencePoly = this.field.fence.map((s) => s.a);
    // write instance matrices straight into the buffer (rotation Y·X·Z tilt, then scale): no Object3D per blade
    const arr = mesh.instanceMatrix.array as Float32Array;
    const colors = new Float32Array(count * 3);
    let n = 0, guard = 0;
    while (n < count && guard++ < count * 8) {
      // denser near home plate where the batting camera looks
      const r = Math.pow(rnd(), 0.55) * 150;
      const a = (rnd() * 2 - 1) * Math.PI * 0.62;
      const x = Math.sin(a) * r, y = Math.cos(a) * r - 6;
      const [dirt, chalk, , alpha] = this.sample(x, y);
      if (dirt > 0.12 || chalk > 0.2 || alpha < 0.5) continue;
      if (!pointInPoly(x, y, fencePoly)) continue;
      const rx = (rnd() - 0.5) * 0.5, ry = rnd() * Math.PI * 2, rz = (rnd() - 0.5) * 0.5;
      const sy = 0.16 + rnd() * 0.22, sx = 0.8 + rnd() * 0.6;
      // Euler XYZ rotation matrix (same as Object3D's default order)
      const cx = Math.cos(rx), snx = Math.sin(rx), cy = Math.cos(ry), sny = Math.sin(ry), cz = Math.cos(rz), snz = Math.sin(rz);
      const o = n * 16;
      arr[o] = cy * cz * sx; arr[o + 1] = (cx * snz + snx * sny * cz) * sx; arr[o + 2] = (snx * snz - cx * sny * cz) * sx; arr[o + 3] = 0;
      arr[o + 4] = -cy * snz * sy; arr[o + 5] = (cx * cz - snx * sny * snz) * sy; arr[o + 6] = (snx * cz + cx * sny * snz) * sy; arr[o + 7] = 0;
      arr[o + 8] = sny; arr[o + 9] = -snx * cy; arr[o + 10] = cx * cy; arr[o + 11] = 0;
      arr[o + 12] = x; arr[o + 13] = 0; arr[o + 14] = -y; arr[o + 15] = 1;
      col.setHSL(0.24 + rnd() * 0.05, 0.48 + rnd() * 0.15, 0.17 + rnd() * 0.1);
      colors[n * 3] = col.r; colors[n * 3 + 1] = col.g; colors[n * 3 + 2] = col.b;
      n++;
    }
    mesh.instanceColor = new InstancedBufferAttribute(colors, 3);
    mesh.count = n;
    mesh.receiveShadow = true;
    mesh.castShadow = false;
    mesh.frustumCulled = false;
    mesh.name = 'grassBlades';
    return mesh;
  }

  update(t: number) {
    if (this.uniforms.uTime) this.uniforms.uTime.value = t;
  }
}

function pathPoly(ctx: CanvasRenderingContext2D, pts: [number, number][]) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

function clipPoly(ctx: CanvasRenderingContext2D, pts: [number, number][]) {
  pathPoly(ctx, pts);
  ctx.clip();
}
