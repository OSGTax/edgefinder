import {
  Color, DirectionalLight, Fog, Group, HemisphereLight, MathUtils, Mesh, MeshBasicMaterial, Object3D, PMREMGenerator, PlaneGeometry,
  Scene, ShaderMaterial, Vector3, type WebGLRenderer,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { cloudAtlas } from './textures';
import { makeSky, AFTERNOON_SKY as SKY } from './sky';
import { mulberry } from './noise';
import type { Quality } from './quality';

export interface SunOptions {
  /** degrees above the horizon */
  elevation: number;
  /** degrees, 0 = sun behind home plate, positive toward the first-base side */
  azimuth: number;
  turbidity?: number;
}

/** Sky dome, sun with soft shadows, sky fill light, image-based lighting, clouds and haze. */
export class Environment {
  readonly sky: Mesh;
  readonly sun = new DirectionalLight(0xffdfb4, 2.9);
  readonly hemi = new HemisphereLight(0xc9dcec, 0x6a6638, 0.62);
  readonly clouds = new Group();
  readonly sunDir = new Vector3();

  constructor(scene: Scene, renderer: WebGLRenderer, q: Quality, o: SunOptions) {
    const phi = MathUtils.degToRad(90 - o.elevation);
    const theta = MathUtils.degToRad(o.azimuth);
    // three: +z is behind home plate, +x toward first base
    this.sunDir.setFromSphericalCoords(1, phi, theta);
    this.sky = makeSky(this.sunDir, SKY);
    scene.add(this.sky);

    this.buildEnvMap(scene, renderer);

    // the sun: soft shadows over the whole yard
    const s = this.sun;
    s.castShadow = true;
    s.shadow.mapSize.set(q.shadowMap, q.shadowMap);
    const cam = s.shadow.camera;
    cam.left = -230; cam.right = 230; cam.top = 230; cam.bottom = -230;
    cam.near = 50; cam.far = 1400;
    s.shadow.bias = -0.0004;
    s.shadow.normalBias = 0.45;
    s.shadow.radius = 3;
    s.target.position.set(0, 0, -70);
    s.position.copy(this.sunDir).multiplyScalar(600).add(s.target.position);
    scene.add(s, s.target, this.hemi);

    // warm aerial haze: the hills and far trees fade into the afternoon instead of staying flat green
    scene.fog = new Fog(new Color(SKY.horizon), 600, 4200);

    this.addClouds();
    scene.add(this.clouds);
  }

  /** Image-based lighting from the same sky over a green ground bounce (again after a lost WebGL context). */
  buildEnvMap(scene: Scene, renderer: WebGLRenderer) {
    const pmrem = new PMREMGenerator(renderer);
    const envScene = new Scene();
    const sky = makeSky(this.sunDir, SKY, 500);
    envScene.add(sky);
    const groundBounce = new Mesh(new PlaneGeometry(2000, 2000), new MeshBasicMaterial({ color: 0x3f6a2a }));
    groundBounce.rotation.x = -Math.PI / 2;
    groundBounce.position.y = -2;
    envScene.add(groundBounce);
    scene.environment?.dispose();
    scene.environment = pmrem.fromScene(envScene, 0.02).texture;
    scene.environmentIntensity = 0.6;
    pmrem.dispose();
    sky.geometry.dispose();
    (sky.material as ShaderMaterial).dispose();
    groundBounce.geometry.dispose();
    groundBounce.material.dispose();
  }

  /** Change the sun's shadow map resolution (the renderer reallocates it next frame). */
  setShadowMapSize(n: number) {
    const sh = this.sun.shadow;
    if (sh.mapSize.x === n) return;
    sh.mapSize.set(n, n);
    sh.map?.dispose();
    sh.map = null;
  }

  /** Sixteen cloud cards sharing one atlas, merged into a single mesh (one draw call). */
  private addClouds() {
    const rnd = mulberry(1234);
    const kinds = 5;
    const parts: PlaneGeometry[] = [];
    const o = new Object3D();
    for (let i = 0; i < 16; i++) {
      const w = 900 + rnd() * 1300;
      const g = new PlaneGeometry(w, w * 0.5);
      // pick cloud (i % kinds) from the vertically stacked atlas (canvas top = v 1)
      const uv = g.attributes.uv, row = kinds - 1 - (i % kinds);
      for (let k = 0; k < uv.count; k++) uv.setY(k, (uv.getY(k) + row) / kinds);
      const a = (rnd() * 1.6 - 0.8) * Math.PI;
      const r = 3200 + rnd() * 1600;
      o.position.set(Math.sin(a) * r, 600 + rnd() * 900, -Math.cos(a) * r);
      o.lookAt(0, o.position.y * 0.6, 0);
      o.updateMatrix();
      parts.push(g.applyMatrix4(o.matrix) as PlaneGeometry);
    }
    const m = new MeshBasicMaterial({ map: cloudAtlas(256, kinds), transparent: true, depthWrite: false, fog: false, opacity: 0.95 });
    const mesh = new Mesh(mergeGeometries(parts, false), m);
    mesh.renderOrder = -1;
    mesh.frustumCulled = false;
    mesh.name = 'clouds';
    this.clouds.add(mesh);
    for (const g of parts) g.dispose();
  }

  update(t: number) {
    // clouds drift slowly across the sky
    this.clouds.rotation.y = t * 0.0015;
  }
}
