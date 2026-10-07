import {
  BackSide, Color, DataTexture, LinearFilter, MeshBasicMaterial, MeshToonMaterial, RedFormat, type MeshToonMaterialParameters,
} from 'three';

// Cartoon shading for the kids: a soft three-tone light ramp (shadow, mid, lit with
// blurred steps between them), a warm fill so the shadow side never goes muddy, and a
// soft rim of sky light. Plus the ink outline material (an inverted hull).

let ramp: DataTexture | null = null;
/** The light ramp: 0 = facing away from the sun, 1 = facing it. Linear filtering softens the steps. */
function toonRamp(): DataTexture {
  if (ramp) return ramp;
  const steps = [0.5, 0.52, 0.55, 0.58, 0.8, 0.83, 1, 1];
  const data = new Uint8Array(steps.map((v) => Math.round(v * 255)));
  ramp = new DataTexture(data, steps.length, 1, RedFormat);
  ramp.minFilter = ramp.magFilter = LinearFilter;
  ramp.generateMipmaps = false;
  ramp.needsUpdate = true;
  return ramp;
}

export interface ToonOptions {
  /** rim light strength (0 = none) */
  rim?: number;
  /** warm self-light as a fraction of the base colour (skin glows a little) */
  fill?: number;
}

/** A kid material in the cartoon style. */
export function toonMaterial(params: MeshToonMaterialParameters, o: ToonOptions = {}): MeshToonMaterial {
  const m = new MeshToonMaterial({ gradientMap: toonRamp(), ...params });
  const rim = o.rim ?? 0.16;
  if (o.fill) m.emissive = new Color(params.color ?? '#ffffff').multiply(new Color('#ffb08c')).multiplyScalar(o.fill);
  m.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
      float kidRim = pow(1.0 - clamp(dot(normalize(normal), normalize(vViewPosition)), 0.0, 1.0), 2.4);
      totalEmissiveRadiance += vec3(1.0, 0.9, 0.8) * kidRim * ${rim.toFixed(3)};`);
  };
  m.customProgramCacheKey = () => `kidToon${rim.toFixed(3)}`;
  return m;
}

/** Warm dark ink, not pure black. */
export const INK_HEX = '#2e1c14';

let ink: MeshBasicMaterial | null = null;
/**
 * The outline: the hull of the kid pushed out along its normals and drawn inside-out in ink.
 * The push grows with distance so the line stays about 1.5 px wide, within limits in feet.
 */
export function outlineMaterial(): MeshBasicMaterial {
  if (ink) return ink;
  ink = new MeshBasicMaterial({ color: INK_HEX, side: BackSide });
  ink.name = 'kidOutline';
  ink.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace('#include <skinning_vertex>', `#include <skinning_vertex>
      {
        float depth = max(0.1, -(modelViewMatrix * vec4(transformed, 1.0)).z);
        transformed += normalize(objectNormal) * clamp(depth * 0.003, 0.013, 0.07);
      }`);
  };
  ink.customProgramCacheKey = () => 'kidOutline';
  return ink;
}
