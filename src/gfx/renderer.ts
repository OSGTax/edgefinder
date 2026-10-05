import { ACESFilmicToneMapping, PCFSoftShadowMap, SRGBColorSpace, WebGLRenderer } from 'three';
import type { Quality } from './quality';

export function createRenderer(canvas: HTMLCanvasElement, q: Quality): WebGLRenderer {
  const r = new WebGLRenderer({ canvas, antialias: q.antialias, powerPreference: 'high-performance', preserveDrawingBuffer: false });
  r.setPixelRatio(q.pixelRatio);
  r.outputColorSpace = SRGBColorSpace;
  r.toneMapping = ACESFilmicToneMapping;
  r.toneMappingExposure = 1.0;
  r.shadowMap.enabled = true;
  r.shadowMap.type = PCFSoftShadowMap;
  return r;
}
