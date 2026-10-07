import { BackSide, Color, Mesh, ShaderMaterial, SphereGeometry, Vector3 } from 'three';

// A painterly summer sky: deep blue overhead fading to a pale hazy horizon,
// a warm glow around the sun and a soft disc. Cheaper and far easier to art
// direct than a physical scattering model.

export interface SkyColors {
  zenith: string;
  horizon: string;
  ground: string;
  sunGlow: string;
}

export const NOON_SKY: SkyColors = { zenith: '#2f6fc4', horizon: '#cfe4f3', ground: '#9fb7a2', sunGlow: '#fff4d6' };
/** Late-summer afternoon: a deep but softer blue overhead, warm dusty haze at the horizon. */
export const AFTERNOON_SKY: SkyColors = { zenith: '#3471bd', horizon: '#e6e0cc', ground: '#a7ad88', sunGlow: '#ffdca6' };

export function makeSky(sunDir: Vector3, c: SkyColors = NOON_SKY, radius = 8000): Mesh {
  const lin = (h: string) => new Color(h);
  const mat = new ShaderMaterial({
    side: BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uZenith: { value: lin(c.zenith) },
      uHorizon: { value: lin(c.horizon) },
      uGround: { value: lin(c.ground) },
      uGlow: { value: lin(c.sunGlow) },
      uSun: { value: sunDir.clone().normalize() },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_Position = p.xyww;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uZenith; uniform vec3 uHorizon; uniform vec3 uGround; uniform vec3 uGlow; uniform vec3 uSun;
      varying vec3 vDir;
      void main() {
        vec3 d = normalize(vDir);
        float h = d.y;
        float up = pow(clamp(h, 0.0, 1.0), 0.42);
        vec3 col = mix(uHorizon, uZenith, up);
        // a thin bright haze band right at the horizon
        col = mix(col, uHorizon * 1.06, exp(-abs(h) * 28.0) * 0.6);
        if (h < 0.0) col = mix(uHorizon, uGround, clamp(-h * 6.0, 0.0, 1.0));
        float sd = max(dot(d, uSun), 0.0);
        col += uGlow * (pow(sd, 6.0) * 0.18 + pow(sd, 64.0) * 0.35);
        col = mix(col, vec3(1.6, 1.5, 1.3), smoothstep(0.9993, 0.9997, sd));
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const mesh = new Mesh(new SphereGeometry(radius, 48, 24), mat);
  mesh.renderOrder = -10;
  mesh.frustumCulled = false;
  mesh.name = 'sky';
  return mesh;
}
